from typing import *
import torch

from ....utils import dist_utils
from .... import models
from .image_conditioned import ImageConditionedMixin


class GeoConditionedMixin(ImageConditionedMixin):
    """
    Mixin for geometry (proxy) + image conditioned models.

    The dataset provides a dense occupancy grid 'proxy_voxel'
    ([B, 64, 64, 64] or [B, 1, 64, 64, 64], float 0/1). It is encoded by a
    frozen sparse structure encoder (ss_enc) into 'proxy_latent'
    ([B, 8, 16, 16, 16]), which is passed to the denoiser as a kwarg and
    consumed by the injection layers.

    NOTE: get_cond() cannot be used to modify the kwargs forwarded to the
    denoiser (training_losses passes its own kwargs copy), so the
    proxy_voxel -> proxy_latent transform is done in training_losses /
    get_inference_cond instead.

    Args:
        geo_cond_model: Config of the geometry conditioning model, e.g.
            {'pretrained': 'weights/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16'}.
        p_uncond_geo: Probability of dropping the geometry condition
            (per-sample zeroing of proxy_latent), independent of the image p_uncond.
    """
    def __init__(self, *args, geo_cond_model: dict, p_uncond_geo: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.geo_cond_model_config = geo_cond_model
        self.p_uncond_geo = p_uncond_geo
        self.geo_cond_model = None      # the model is init lazily

    def _init_geo_cond_model(self):
        """
        Initialize the frozen geometry conditioning model (ss_enc).
        """
        with dist_utils.local_master_first():
            encoder = models.from_pretrained(self.geo_cond_model_config['pretrained'])
        encoder.eval()
        encoder.requires_grad_(False)
        self.geo_cond_model = encoder.cuda()

    @torch.no_grad()
    def encode_geo(self, proxy_voxel: torch.Tensor) -> torch.Tensor:
        """
        Encode the proxy occupancy grid into the proxy latent.
        """
        if self.geo_cond_model is None:
            self._init_geo_cond_model()
        if proxy_voxel.ndim == 4:
            proxy_voxel = proxy_voxel.unsqueeze(1)
        proxy_voxel = proxy_voxel.float().cuda()
        return self.geo_cond_model(proxy_voxel, sample_posterior=False)

    def training_losses(self, x_0, cond=None, proxy_voxel=None, **kwargs):
        """
        Transform proxy_voxel -> proxy_latent (with independent CFG drop)
        before delegating to the underlying trainer.
        """
        if proxy_voxel is not None:
            proxy_latent = self.encode_geo(proxy_voxel)
            if self.p_uncond_geo > 0:
                # per-sample drop: zeroed proxy_latent acts as the geo-unconditional input
                mask = torch.rand(proxy_latent.shape[0], device=proxy_latent.device) < self.p_uncond_geo
                proxy_latent = torch.where(
                    mask.view(-1, *[1] * (proxy_latent.ndim - 1)),
                    torch.zeros_like(proxy_latent),
                    proxy_latent,
                )
            kwargs['proxy_latent'] = proxy_latent
        return super().training_losses(x_0, cond=cond, **kwargs)

    def get_inference_cond(self, cond, proxy_voxel=None, **kwargs):
        """
        Get the conditioning data for inference.
        """
        if proxy_voxel is not None:
            kwargs['proxy_latent'] = self.encode_geo(proxy_voxel)
        return super().get_inference_cond(cond, **kwargs)

    def vis_cond(self, cond, **kwargs):
        """
        Visualize the conditioning data (image only; extra kwargs tolerated).
        """
        return super().vis_cond(cond, **kwargs)
