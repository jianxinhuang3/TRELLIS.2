from .flow_matching import FlowMatchingCFGTrainer
from .sparse_flow_matching import SparseFlowMatchingCFGTrainer
from .mixins.geo_conditioned import GeoConditionedMixin


class GeoImageConditionedFlowMatchingCFGTrainer(GeoConditionedMixin, FlowMatchingCFGTrainer):
    """
    Trainer for geometry (proxy) + image conditioned dense flow matching models
    with classifier-free guidance (used for the SS flow injection training).

    Args:
        models (dict[str, nn.Module]): Models to train ('denoiser': injected flow model).
        dataset (torch.utils.data.Dataset): Dataset providing x_0/cond/proxy_voxel.
        ... (see FlowMatchingCFGTrainer)
        p_uncond (float): Probability of dropping the image condition.
        p_uncond_geo (float): Probability of dropping the geometry condition.
        image_cond_model (dict): Image conditioning model config.
        geo_cond_model (dict): Geometry conditioning model config
            ({'pretrained': path to frozen ss_enc}).
    """
    pass


class GeoImageConditionedSparseFlowMatchingCFGTrainer(GeoConditionedMixin, SparseFlowMatchingCFGTrainer):
    """
    Trainer for geometry (proxy) + image conditioned sparse flow matching models
    with classifier-free guidance (used for the shape flow injection training).

    Args:
        models (dict[str, nn.Module]): Models to train ('denoiser': injected flow model).
        dataset (torch.utils.data.Dataset): Dataset providing x_0/cond/proxy_voxel.
        ... (see SparseFlowMatchingCFGTrainer)
        p_uncond (float): Probability of dropping the image condition.
        p_uncond_geo (float): Probability of dropping the geometry condition.
        image_cond_model (dict): Image conditioning model config.
        geo_cond_model (dict): Geometry conditioning model config
            ({'pretrained': path to frozen ss_enc}).
    """
    pass
