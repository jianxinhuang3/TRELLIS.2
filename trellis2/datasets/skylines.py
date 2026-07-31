import os
from typing import *
import numpy as np
import torch
from .sparse_structure_latent import ImageConditionedSparseStructureLatent
from .structured_latent_shape import ImageConditionedSLatShape
from .structured_latent_svpbr import ImageConditionedSLatPbr


class _RepeatMixin:
    """
    Mixin that virtually repeats the dataset `repeat` times (for tiny datasets /
    single-sample overfitting). Compatible with BalancedResumableSampler by
    extending `loads` accordingly.
    """
    def __init__(self, roots, *, repeat: int = 1, **kwargs):
        self.repeat = repeat
        super().__init__(roots, **kwargs)
        if self.repeat > 1 and hasattr(self, 'loads'):
            self.loads = list(self.loads) * self.repeat

    def __len__(self):
        return len(self.instances) * self.repeat

    def __getitem__(self, index):
        return super().__getitem__(index % len(self.instances))


class _ProxyVoxelMixin:
    """
    Mixin that loads the proxy occupancy grid from root['proxy']/{sha256}.npz
    ('voxel' key, [64, 64, 64] 0/1) into pack['proxy_voxel'].
    """
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata['proxy_built'] == True]
        stats['Proxy built'] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
        proxy = np.load(os.path.join(root['proxy'], f'{instance}.npz'))['voxel']
        pack['proxy_voxel'] = torch.from_numpy(np.ascontiguousarray(proxy)).float()
        return pack


class SkylinesSSLatent(_RepeatMixin, _ProxyVoxelMixin, ImageConditionedSparseStructureLatent):
    """
    Skylines sparse structure latent dataset with proxy voxel condition.

    Args:
        roots (str): JSON dict of dataset roots, e.g.
            {"skylines": {"metadata": ..., "ss_latent": ..., "render_cond": ..., "proxy": ...}}
        repeat (int): Virtually repeat the dataset this many times.
        ... (see ImageConditionedSparseStructureLatent)
    """
    pass


class SkylinesSLatShape(_RepeatMixin, _ProxyVoxelMixin, ImageConditionedSLatShape):
    """
    Skylines structured latent (shape) dataset with proxy voxel condition.

    Args:
        roots (str): JSON dict of dataset roots (needs 'shape_latent', 'render_cond', 'proxy').
        repeat (int): Virtually repeat the dataset this many times.
        ... (see ImageConditionedSLatShape)
    """
    pass


class SkylinesSLatPbr(_RepeatMixin, _ProxyVoxelMixin, ImageConditionedSLatPbr):
    """
    Skylines structured latent (PBR texture) dataset with proxy voxel condition.

    NOTE: pipeline.md T3 conditions are {A, C}, but in the current test_on_test
    stage C is skipped, so the proxy (G tokens) is fed to the injection layers
    instead to give the trainable injectors a non-trivial signal (the frozen
    base alone has no trainable params in the forward).

    Args:
        roots (str): JSON dict of dataset roots (needs 'pbr_latent', 'shape_latent',
            'render_cond', 'proxy').
        repeat (int): Virtually repeat the dataset this many times.
        ... (see ImageConditionedSLatPbr)
    """
    pass
