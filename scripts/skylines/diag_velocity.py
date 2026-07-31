"""
Diagnostic: measure velocity prediction MSE across timesteps t.

For a single-sample overfit flow model, the TRUE velocity field on the
training trajectory is v_true = (1-sigma_min)*noise - x_0.

We evaluate the model at various t levels on x_t = (1-t)*x_0 + sigma_t*noise
and compare pred_v to v_true. This reveals whether the model learned the
field at high t (near pure noise) vs low t (near data).

Also does a full Euler trajectory diagnostic: at each sampling step, measure
how far the current x_t is from the ideal on-trajectory point.
"""
import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trellis2 import models, datasets
from trellis2.modules.sparse import SparseTensor


def load_ema_weights(model, ckpt_dir, step, ema_rate=0.9999):
    ckpt_path = os.path.join(ckpt_dir, 'ckpts', f'denoiser_ema{ema_rate}_step{step:07d}.pt')
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    print(f'Loaded EMA weights from {ckpt_path}')
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    sigma_min = 1e-5

    with open(args.config, 'r') as f:
        cfg = edict(json.load(f))

    model = getattr(models, cfg.models.denoiser.name)(**cfg.models.denoiser.args).cuda()
    model = load_ema_weights(model, args.ckpt_dir, args.step)
    model.eval()

    dataset = getattr(datasets, cfg.dataset.name)(args.data_dir, **cfg.dataset.args)

    # geo + image cond models
    geo_cond_model = models.from_pretrained(
        cfg.trainer.args.geo_cond_model.pretrained).cuda().eval()
    geo_cond_model.requires_grad_(False)
    from trellis2.trainers import DinoV3FeatureExtractor
    image_cond_model = DinoV3FeatureExtractor(**cfg.trainer.args.image_cond_model.args)
    image_cond_model.cuda()

    # single sample
    sample = dataset[0]
    batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
    x_0 = batch['x_0'].cuda()
    cond = batch['cond'].cuda()
    if cond.ndim == 4:
        cond = image_cond_model(cond)

    proxy_voxel = batch['proxy_voxel'].cuda()
    if proxy_voxel.ndim == 3:
        proxy_voxel = proxy_voxel.unsqueeze(0)
    if proxy_voxel.ndim == 4:
        proxy_voxel = proxy_voxel.unsqueeze(1)
    proxy_latent = geo_cond_model(proxy_voxel.float(), sample_posterior=False)

    torch.manual_seed(args.seed)
    noise = torch.randn_like(x_0)

    print('\n=== Velocity prediction MSE on training trajectory across t ===')
    print(f'{"t":<8} {"v_true_norm":<14} {"pred_v_norm":<14} {"v_MSE":<12} {"rel_err":<10}')
    print('-' * 60)
    for t in [0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 0.02]:
        # on-trajectory x_t
        x_t = (1 - t) * x_0 + (sigma_min + (1 - sigma_min) * t) * noise
        v_true = (1 - sigma_min) * noise - x_0
        t_tensor = torch.full((1,), t * 1000, device='cuda')
        with torch.no_grad():
            pred_v = model(x_t, t_tensor, cond, proxy_latent=proxy_latent)
        v_mse = F.mse_loss(pred_v, v_true).item()
        v_true_norm = v_true.norm().item()
        pred_v_norm = pred_v.norm().item()
        rel_err = (pred_v - v_true).norm().item() / max(v_true_norm, 1e-8)
        print(f'{t:<8.2f} {v_true_norm:<14.4f} {pred_v_norm:<14.4f} {v_mse:<12.6f} {rel_err:<10.4f}')

    # Full Euler trajectory drift diagnostic
    print('\n=== Euler trajectory drift (guidance=1.0, no CFG) ===')
    print(f'{"step":<6} {"t":<8} {"x_t_MSE_vs_ideal":<20} {"pred_v_MSE_vs_true":<20}')
    print('-' * 60)
    steps = 50
    t_seq = np.linspace(1, 0, steps + 1)
    x_t = noise.clone()
    for i in range(steps):
        t = t_seq[i]
        t_prev = t_seq[i + 1]
        # ideal on-trajectory x_t at this t
        ideal_x_t = (1 - t) * x_0 + (sigma_min + (1 - sigma_min) * t) * noise
        x_t_drift = F.mse_loss(x_t, ideal_x_t).item()
        # true velocity
        v_true = (1 - sigma_min) * noise - x_0
        t_tensor = torch.full((1,), t * 1000, device='cuda')
        with torch.no_grad():
            pred_v = model(x_t, t_tensor, cond, proxy_latent=proxy_latent)
        v_mse = F.mse_loss(pred_v, v_true).item()
        if i % 5 == 0 or i >= steps - 3:
            print(f'{i:<6} {t:<8.3f} {x_t_drift:<20.6f} {v_mse:<20.6f}')
        # Euler step
        x_t = x_t - (t - t_prev) * pred_v

    final_mse = F.mse_loss(x_t, x_0).item()
    print(f'\nFinal sample MSE vs x_0: {final_mse:.6f}')


if __name__ == '__main__':
    main()
