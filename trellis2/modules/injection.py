from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sparse import SparseTensor
from .sparse.attention import sparse_scaled_dot_product_attention


class GatedCrossAttnInjector(nn.Module):
    """
    Decoupled gated cross-attention injection layer (dense version).

    Applied after each frozen DiT block:
        x = x + tanh(gate) * CrossAttn(LN(x), ctx)
    The gate is zero-initialized so that the module is an identity mapping
    at initialization, keeping the wrapped model equivalent to the base model.

    A bottleneck inner dimension is used to keep the trainable parameter
    count small (<80M for 30 blocks).

    Args:
        channels (int): Channels of the DiT hidden states.
        ctx_channels (int): Channels of the injected context tokens.
        num_heads (int): Number of attention heads.
        inner_channels (int): Bottleneck channels of the attention.
    """
    def __init__(
        self,
        channels: int = 1536,
        ctx_channels: int = 1024,
        num_heads: int = 12,
        inner_channels: int = 384,
        gate_init: float = 0.0,
    ):
        super().__init__()
        assert inner_channels % num_heads == 0
        self.channels = channels
        self.ctx_channels = ctx_channels
        self.num_heads = num_heads
        self.inner_channels = inner_channels
        self.head_dim = inner_channels // num_heads

        self.norm = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.to_q = nn.Linear(channels, inner_channels)
        self.to_kv = nn.Linear(ctx_channels, inner_channels * 2)
        self.to_out = nn.Linear(inner_channels, channels)
        if gate_init != 0.0:
            nn.init.zeros_(self.to_out.weight)
            nn.init.zeros_(self.to_out.bias)
        self.gate = nn.Parameter(torch.full((1,), gate_init))

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, C] hidden states (possibly fp16/bf16 from the frozen torso).
            ctx: [B, Lc, ctx_channels] context tokens.
        Returns:
            [B, L, C] tensor with the same dtype as x.
        """
        B, L, _ = x.shape
        Lc = ctx.shape[1]
        # keep injector computation in fp32 (params are fp32, managed by AMP if enabled)
        h = self.norm(x.float())
        q = self.to_q(h).reshape(B, L, self.num_heads, self.head_dim)
        kv = self.to_kv(ctx.float()).reshape(B, Lc, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        # torch sdpa expects [B, H, L, D]
        o = F.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
        )
        o = o.permute(0, 2, 1, 3).reshape(B, L, self.inner_channels)
        out = self.to_out(o)
        return x + (torch.tanh(self.gate) * out).to(x.dtype)


class SparseGatedCrossAttnInjector(nn.Module):
    """
    Decoupled gated cross-attention injection layer (sparse version).

    Same as GatedCrossAttnInjector but the queries come from a SparseTensor
    with variable-length batch layout; the context is a dense [B, Lc, C] tensor.
    """
    def __init__(
        self,
        channels: int = 1536,
        ctx_channels: int = 1024,
        num_heads: int = 12,
        inner_channels: int = 384,
        gate_init: float = 0.0,
    ):
        super().__init__()
        assert inner_channels % num_heads == 0
        self.channels = channels
        self.ctx_channels = ctx_channels
        self.num_heads = num_heads
        self.inner_channels = inner_channels
        self.head_dim = inner_channels // num_heads

        self.norm = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.to_q = nn.Linear(channels, inner_channels)
        self.to_kv = nn.Linear(ctx_channels, inner_channels * 2)
        self.to_out = nn.Linear(inner_channels, channels)
        if gate_init != 0.0:
            nn.init.zeros_(self.to_out.weight)
            nn.init.zeros_(self.to_out.bias)
        self.gate = nn.Parameter(torch.full((1,), gate_init))

    def forward(self, x: SparseTensor, ctx: torch.Tensor) -> SparseTensor:
        """
        Args:
            x: SparseTensor with feats [T, C].
            ctx: [B, Lc, ctx_channels] dense context tokens, B == x.shape[0].
        Returns:
            SparseTensor with the same layout and dtype as x.
        """
        B = x.shape[0]
        Lc = ctx.shape[1]
        feats = x.feats
        h = self.norm(feats.float())
        q = self.to_q(h)                                    # [T, inner]
        kv = self.to_kv(ctx.float())                        # [B, Lc, 2 * inner]
        # sparse varlen attention (flash-attn) requires 16-bit dtypes
        attn_dtype = feats.dtype if feats.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        q = x.replace(q.to(attn_dtype)).reshape(self.num_heads, self.head_dim)
        kv = kv.reshape(B, Lc, 2, self.num_heads, self.head_dim).to(attn_dtype)
        o = sparse_scaled_dot_product_attention(q, kv)      # SparseTensor [N, *, H, D]
        o = o.reshape(-1)                                   # [T, inner]
        out = self.to_out(o.feats.float())                  # [T, C]
        return x.replace(feats + (torch.tanh(self.gate) * out).to(feats.dtype))


class GeoTokenizer(nn.Module):
    """
    Tokenizer for the proxy geometry latent (output of the frozen ss_enc).

    z_proxy [B, 8, 16, 16, 16] -> Conv3d stem -> stride-2 downsample to 8^3
    -> flatten to [B, 512, out_channels] + learnable positional embedding.

    Args:
        in_channels (int): Channels of the proxy latent.
        out_channels (int): Channels of the output tokens.
        hidden_channels (int): Channels of the conv stem.
        resolution (int): Spatial resolution of the proxy latent.
    """
    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 1024,
        hidden_channels: int = 256,
        resolution: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.num_tokens = (resolution // 2) ** 3

        self.proj_in = nn.Conv3d(in_channels, hidden_channels, 3, padding=1)
        self.down = nn.Conv3d(hidden_channels, out_channels, 2, stride=2)
        self.norm = nn.LayerNorm(out_channels)
        self.pos_emb = nn.Parameter(torch.zeros(self.num_tokens, out_channels))
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, in_channels, R, R, R] proxy latent.
        Returns:
            [B, (R/2)^3, out_channels] context tokens (fp32).
        """
        h = self.proj_in(z.float())
        h = self.down(F.silu(h))                            # [B, C, R/2, R/2, R/2]
        h = h.reshape(h.shape[0], h.shape[1], -1).permute(0, 2, 1).contiguous()
        h = self.norm(h) + self.pos_emb[None]
        return h
