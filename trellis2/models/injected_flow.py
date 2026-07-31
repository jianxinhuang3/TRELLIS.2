from typing import *
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from ..modules.utils import manual_cast
from ..modules import sparse as sp
from ..modules.injection import (
    GatedCrossAttnInjector,
    SparseGatedCrossAttnInjector,
    GeoTokenizer,
)
from .sparse_structure_flow import SparseStructureFlowModel
from .structured_latent_flow import SLatFlowModel


def _load_base_config(pretrained_base: str) -> dict:
    """
    Load the config json next to a local checkpoint.
    """
    config_file = f"{pretrained_base}.json"
    if not os.path.exists(config_file):
        from huggingface_hub import hf_hub_download
        path_parts = pretrained_base.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
    with open(config_file, 'r') as f:
        return json.load(f)


def _load_base_weights(pretrained_base: str) -> dict:
    model_file = f"{pretrained_base}.safetensors"
    if not os.path.exists(model_file):
        from huggingface_hub import hf_hub_download
        path_parts = pretrained_base.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")
    return load_file(model_file)


class InjectedSparseStructureFlowModel(nn.Module):
    """
    Frozen SparseStructureFlowModel with trainable decoupled gated cross-attention
    injection layers (one per block) and a GeoTokenizer for the proxy latent.

    The base model is frozen via requires_grad_(False) but still runs inside the
    autograd graph so that gradients can flow through it to earlier injectors.

    Args:
        pretrained_base (str): Path to the base checkpoint (local `{path}.json` +
            `{path}.safetensors`, or a Hugging Face model name).
        injector_args (dict): Extra args for GatedCrossAttnInjector.
        geo_tokenizer_args (dict): Extra args for GeoTokenizer.
        **base_args: Overrides for the base model constructor args.
    """
    def __init__(
        self,
        pretrained_base: str,
        injector_args: dict = {},
        geo_tokenizer_args: dict = {},
        **base_args,
    ):
        super().__init__()
        config = _load_base_config(pretrained_base)
        assert config['name'] == 'SparseStructureFlowModel', \
            f"Expected SparseStructureFlowModel base, got {config['name']}"
        cfg_args = dict(config['args'])
        cfg_args.update(base_args)

        self.base = SparseStructureFlowModel(**cfg_args)
        self.base.load_state_dict(_load_base_weights(pretrained_base), strict=False)
        self.base.requires_grad_(False)

        _injector_args = {
            'channels': self.base.model_channels,
            'ctx_channels': self.base.cond_channels,
            'num_heads': self.base.num_heads,
        }
        _injector_args.update(injector_args)
        self.injectors = nn.ModuleList([
            GatedCrossAttnInjector(**_injector_args)
            for _ in range(self.base.num_blocks)
        ])

        _geo_tokenizer_args = {'out_channels': _injector_args['ctx_channels']}
        _geo_tokenizer_args.update(geo_tokenizer_args)
        self.geo_tokenizer = GeoTokenizer(**_geo_tokenizer_args)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return self.base.dtype

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        proxy_latent: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        base = self.base
        assert [*x.shape] == [x.shape[0], base.in_channels, *[base.resolution] * 3], \
                f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], base.in_channels, *[base.resolution] * 3]}"

        h = x.view(*x.shape[:2], -1).permute(0, 2, 1).contiguous()

        h = base.input_layer(h)
        if base.pe_mode == "ape":
            h = h + base.pos_emb[None]
        t_emb = base.t_embedder(t)
        if base.share_mod:
            t_emb = base.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, base.dtype)
        h = manual_cast(h, base.dtype)
        cond = manual_cast(cond, base.dtype)
        g_tokens = self.geo_tokenizer(proxy_latent) if proxy_latent is not None else None
        for block, injector in zip(base.blocks, self.injectors):
            h = block(h, t_emb, cond, base.rope_phases)
            if g_tokens is not None:
                h = injector(h, g_tokens)
        h = manual_cast(h, x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = base.out_layer(h)

        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[base.resolution] * 3).contiguous()

        return h


class InjectedSLatFlowModel(nn.Module):
    """
    Frozen SLatFlowModel with trainable decoupled gated cross-attention injection
    layers (sparse version, one per block) and a GeoTokenizer for the proxy latent.

    NOTE: The non-elastic SLatFlowModel is wrapped because the ElasticMixin
    controls the whole forward via memory-ratio checkpointing, which conflicts
    with per-block injection. Per-block gradient checkpointing can still be
    enabled via base_args `use_checkpoint=True`.

    Args:
        pretrained_base (str): Path to the base checkpoint.
        injector_args (dict): Extra args for SparseGatedCrossAttnInjector.
        geo_tokenizer_args (dict): Extra args for GeoTokenizer.
        **base_args: Overrides for the base model constructor args.
    """
    def __init__(
        self,
        pretrained_base: str,
        injector_args: dict = {},
        geo_tokenizer_args: dict = {},
        **base_args,
    ):
        super().__init__()
        config = _load_base_config(pretrained_base)
        assert config['name'] in ['SLatFlowModel', 'ElasticSLatFlowModel'], \
            f"Expected SLatFlowModel base, got {config['name']}"
        cfg_args = dict(config['args'])
        cfg_args.update(base_args)

        self.base = SLatFlowModel(**cfg_args)
        self.base.load_state_dict(_load_base_weights(pretrained_base), strict=False)
        self.base.requires_grad_(False)

        _injector_args = {
            'channels': self.base.model_channels,
            'ctx_channels': self.base.cond_channels,
            'num_heads': self.base.num_heads,
        }
        _injector_args.update(injector_args)
        self.injectors = nn.ModuleList([
            SparseGatedCrossAttnInjector(**_injector_args)
            for _ in range(self.base.num_blocks)
        ])

        _geo_tokenizer_args = {'out_channels': _injector_args['ctx_channels']}
        _geo_tokenizer_args.update(geo_tokenizer_args)
        self.geo_tokenizer = GeoTokenizer(**_geo_tokenizer_args)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return self.base.dtype

    def forward(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        cond: Union[torch.Tensor, List[torch.Tensor]],
        concat_cond: Optional[sp.SparseTensor] = None,
        proxy_latent: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> sp.SparseTensor:
        base = self.base
        if concat_cond is not None:
            x = sp.sparse_cat([x, concat_cond], dim=-1)
        if isinstance(cond, list):
            cond = sp.VarLenTensor.from_tensor_list(cond)

        h = base.input_layer(x)
        h = manual_cast(h, base.dtype)
        t_emb = base.t_embedder(t)
        if base.share_mod:
            t_emb = base.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, base.dtype)
        cond = manual_cast(cond, base.dtype)

        if base.pe_mode == "ape":
            pe = base.pos_embedder(h.coords[:, 1:])
            h = h + manual_cast(pe, base.dtype)
        g_tokens = self.geo_tokenizer(proxy_latent) if proxy_latent is not None else None
        for block, injector in zip(base.blocks, self.injectors):
            h = block(h, t_emb, cond)
            if g_tokens is not None:
                h = injector(h, g_tokens)

        h = manual_cast(h, x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = base.out_layer(h)
        return h
