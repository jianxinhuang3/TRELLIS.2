"""
Evaluation script for single-sample overfitting experiments (T1/T2/T3).

Functionality:
  - Loads an EMA checkpoint at a specified training step.
  - Runs multi-step flow sampling on a single data sample.
  - Computes reconstruction metrics (IoU for SS, latent MSE for Shape/Tex).
  - Gate=0 consistency check: verifies that the injected model with
    proxy_latent=random vs proxy_latent=None produces max|delta| < 1e-3
    when gates are reset to zero.
  - Saves results as a JSON file + optional visualization.

Usage:
    source /data5/jianxin/anaconda3/bin/activate trellis2
    cd /data5/jianxin/TRELLIS.2
    export PYTHONPATH=/data5/jianxin/TRELLIS.2:$PYTHONPATH
    python scripts/skylines/eval_overfit.py \\
        --config configs/skylines/t1_ss_flow_inject.json \\
        --ckpt_dir outputs/skylines_t1_toy \\
        --step 2000 \\
        --data_dir '<roots JSON>' \\
        --output_dir outputs/skylines_t1_toy/eval
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
    """Load EMA weights for the denoiser model."""
    ckpt_pattern = os.path.join(ckpt_dir, 'ckpts', f'denoiser_ema{ema_rate}_step{step:07d}.pt')
    if os.path.exists(ckpt_pattern):
        state_dict = torch.load(ckpt_pattern, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f'Loaded EMA weights from {ckpt_pattern}')
    else:
        # Fallback: load regular checkpoint
        ckpt_pattern = os.path.join(ckpt_dir, 'ckpts', f'denoiser_step{step:07d}.pt')
        if os.path.exists(ckpt_pattern):
            state_dict = torch.load(ckpt_pattern, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            print(f'Loaded regular weights from {ckpt_pattern}')
        else:
            raise FileNotFoundError(f'No checkpoint found at step {step} in {ckpt_dir}')
    return model


def build_model_and_dataset(cfg, data_dir):
    """Build model and dataset from config."""
    model = getattr(models, cfg.models.denoiser.name)(**cfg.models.denoiser.args).cuda()
    dataset = getattr(datasets, cfg.dataset.name)(data_dir, **cfg.dataset.args)
    return model, dataset


def get_single_sample(dataset):
    """Get a single sample from the dataset."""
    sample = dataset[0]
    # Convert to batched
    if isinstance(sample.get('x_0'), SparseTensor):
        # For sparse: use collate_fn
        batch = dataset.collate_fn([sample])
    else:
        batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                 for k, v in sample.items()}
    return batch


@torch.no_grad()
def one_step_denoise(model, batch, t_noise=0.3, sigma_min=1e-5, geo_cond_model=None, image_cond_model=None):
    """
    One-step denoising evaluation: add noise at level t_noise, predict x_0 in one step.
    This directly evaluates whether the model has memorized x_0.
    """
    x_0 = batch['x_0']
    if isinstance(x_0, SparseTensor):
        x_0 = x_0.cuda()
        noise = x_0.replace(torch.randn_like(x_0.feats))
    else:
        x_0 = x_0.cuda()
        noise = torch.randn_like(x_0)

    # Create noisy input at t_noise
    if isinstance(x_0, SparseTensor):
        x_t_feats = (1 - t_noise) * x_0.feats + (sigma_min + (1-sigma_min)*t_noise) * noise.feats
        x_t = x_0.replace(x_t_feats)
    else:
        x_t = (1 - t_noise) * x_0 + (sigma_min + (1-sigma_min)*t_noise) * noise

    # Prepare conditioning
    cond = batch.get('cond')
    if cond is not None:
        if isinstance(cond, torch.Tensor):
            cond = cond.cuda()
        if cond.ndim == 4 and image_cond_model is not None:
            cond = image_cond_model(cond)
        elif cond.ndim == 4:
            B = cond.shape[0]
            cond = torch.randn(B, 1024, 1024, device='cuda')

    kwargs = {}
    proxy_voxel = batch.get('proxy_voxel')
    if proxy_voxel is not None and geo_cond_model is not None:
        proxy_voxel = proxy_voxel.cuda()
        if proxy_voxel.ndim == 3:
            proxy_voxel = proxy_voxel.unsqueeze(0)
        if proxy_voxel.ndim == 4:
            proxy_voxel = proxy_voxel.unsqueeze(1)
        proxy_latent = geo_cond_model(proxy_voxel.float(), sample_posterior=False)
        kwargs['proxy_latent'] = proxy_latent

    # Model forward: predict velocity
    t_tensor = torch.full((x_t.shape[0],), t_noise * 1000, device='cuda')
    pred_v = model(x_t, t_tensor, cond, **kwargs)

    # Recover x_0: x_0 = (1-sigma_min)*x_t - (sigma_min + (1-sigma_min)*t)*v
    if isinstance(pred_v, SparseTensor):
        pred_x0_feats = (1-sigma_min)*x_t.feats - (sigma_min + (1-sigma_min)*t_noise)*pred_v.feats
        pred_x0 = x_0.replace(pred_x0_feats)
    else:
        pred_x0 = (1-sigma_min)*x_t - (sigma_min + (1-sigma_min)*t_noise)*pred_v

    print(f'  One-step denoise from t={t_noise}: latent MSE={F.mse_loss(pred_x0 if not isinstance(pred_x0, SparseTensor) else pred_x0.feats, x_0 if not isinstance(x_0, SparseTensor) else x_0.feats).item():.6f}')
    return pred_x0, x_0


@torch.no_grad()
def sample_flow(model, batch, sigma_min=1e-5, steps=50, geo_cond_model=None, image_cond_model=None, use_cfg=True, guidance_strength=3.0):
    """Run flow-euler sampling from noise to generate a sample."""
    from trellis2.pipelines.samplers.flow_euler import FlowEulerCfgSampler, FlowEulerSampler

    x_0 = batch['x_0']
    if isinstance(x_0, SparseTensor):
        x_0 = x_0.cuda()
        noise = x_0.replace(torch.randn_like(x_0.feats))
    else:
        x_0 = x_0.cuda()
        noise = torch.randn_like(x_0)

    # Prepare conditioning
    cond = batch.get('cond')
    if cond is not None:
        if isinstance(cond, torch.Tensor):
            cond = cond.cuda()
        # Encode image through DINOv3 if raw image tensor
        if cond.ndim == 4 and image_cond_model is not None:  # image [B,3,H,W]
            cond = image_cond_model(cond)
        elif cond.ndim == 4:  # fallback: random features
            B = cond.shape[0]
            cond = torch.randn(B, 1024, 1024, device='cuda')

    proxy_voxel = batch.get('proxy_voxel')
    kwargs = {}
    if proxy_voxel is not None and geo_cond_model is not None:
        proxy_voxel = proxy_voxel.cuda()
        if proxy_voxel.ndim == 3:
            proxy_voxel = proxy_voxel.unsqueeze(0)
        if proxy_voxel.ndim == 4:
            proxy_voxel = proxy_voxel.unsqueeze(1)
        proxy_latent = geo_cond_model(proxy_voxel.float(), sample_posterior=False)
        kwargs['proxy_latent'] = proxy_latent

    # Concat cond for tex stage
    concat_cond = batch.get('concat_cond')
    if concat_cond is not None:
        if isinstance(concat_cond, SparseTensor):
            kwargs['concat_cond'] = concat_cond.cuda()
        else:
            kwargs['concat_cond'] = concat_cond.cuda()

    if use_cfg:
        sampler = FlowEulerCfgSampler(sigma_min)
        neg_cond = torch.zeros_like(cond)
        result = sampler.sample(
            model, noise=noise, cond=cond, neg_cond=neg_cond,
            steps=steps, guidance_strength=guidance_strength, verbose=True, **kwargs
        )
    else:
        sampler = FlowEulerSampler(sigma_min)
        result = sampler.sample(
            model, noise=noise, cond=cond,
            steps=steps, verbose=True, **kwargs
        )
    return result.samples, x_0


def compute_ss_iou(pred, gt, threshold=0.5):
    """Compute IoU for sparse structure (dense 64^3 occupancy)."""
    # Decode through ss_dec if needed, or just compare latent-decoded occupancy
    # For simplicity: if pred/gt are latent, skip decode and compute latent MSE
    if pred.shape[-1] == 16 and pred.ndim == 5:
        # These are latents [B, 8, 16, 16, 16] — compute MSE
        mse = F.mse_loss(pred, gt).item()
        return {'latent_mse': mse}
    # Otherwise assume decoded occupancy
    pred_occ = (pred > threshold).float()
    gt_occ = (gt > threshold).float()
    intersection = (pred_occ * gt_occ).sum().item()
    union = ((pred_occ + gt_occ) > 0).float().sum().item()
    iou = intersection / max(union, 1e-8)
    return {'iou': iou}


def compute_sparse_mse(pred, gt):
    """Compute MSE for sparse tensors."""
    if isinstance(pred, SparseTensor) and isinstance(gt, SparseTensor):
        mse = F.mse_loss(pred.feats, gt.feats).item()
    elif isinstance(pred, torch.Tensor) and isinstance(gt, torch.Tensor):
        mse = F.mse_loss(pred, gt).item()
    else:
        mse = float('nan')
    return {'latent_mse': mse}


@torch.no_grad()
def gate_zero_consistency_check(model, batch, geo_cond_model=None):
    """
    Check that with gates=0, the injected model with proxy_latent=random
    vs proxy_latent=None produces nearly identical outputs (max|delta| < 1e-3).
    """
    # Save original gate values
    original_gates = []
    for inj in model.injectors:
        original_gates.append(inj.gate.data.clone())
        inj.gate.data.zero_()

    x_0 = batch['x_0']
    if isinstance(x_0, SparseTensor):
        x = x_0.replace(torch.randn_like(x_0.feats)).cuda()
    else:
        x = torch.randn_like(x_0).cuda()

    B = x.shape[0] if isinstance(x, torch.Tensor) else x.shape[0]
    t = torch.full((B,), 500.0, device='cuda')
    cond = torch.randn(B, 1024, 1024, device='cuda')

    proxy_latent = torch.randn(B, 8, 16, 16, 16, device='cuda')
    kwargs_with = {'proxy_latent': proxy_latent}
    kwargs_without = {'proxy_latent': None}

    # Handle concat_cond
    concat_cond = batch.get('concat_cond')
    if concat_cond is not None:
        if isinstance(concat_cond, SparseTensor):
            kwargs_with['concat_cond'] = concat_cond.cuda()
            kwargs_without['concat_cond'] = concat_cond.cuda()
        else:
            kwargs_with['concat_cond'] = concat_cond.cuda()
            kwargs_without['concat_cond'] = concat_cond.cuda()

    y_with = model(x, t, cond, **kwargs_with)
    y_without = model(x, t, cond, **kwargs_without)

    if isinstance(y_with, SparseTensor):
        max_diff = (y_with.feats - y_without.feats).abs().max().item()
    else:
        max_diff = (y_with - y_without).abs().max().item()

    # Restore gates
    for inj, g in zip(model.injectors, original_gates):
        inj.gate.data.copy_(g)

    return {'max_abs_diff': max_diff, 'pass': max_diff < 1e-3}


def main():
    parser = argparse.ArgumentParser(description='Eval overfitting for skylines injection training')
    parser.add_argument('--config', type=str, required=True, help='Training config JSON')
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Checkpoint directory')
    parser.add_argument('--step', type=int, required=True, help='Checkpoint step to evaluate')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory (roots JSON)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--steps', type=int, default=50, help='Sampling steps')
    parser.add_argument('--no_cfg', action='store_true', help='Disable CFG sampling')
    parser.add_argument('--guidance', type=float, default=3.0, help='CFG guidance strength')
    parser.add_argument('--denoise_t', type=float, default=None, help='One-step denoising eval: add noise at this t level and predict x_0')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    with open(args.config, 'r') as f:
        cfg = edict(json.load(f))

    # Build model and dataset
    model, dataset = build_model_and_dataset(cfg, args.data_dir)
    model = load_ema_weights(model, args.ckpt_dir, args.step)
    model.eval()

    # Load geo_cond_model (ss_enc) for proxy encoding
    geo_cond_model = None
    geo_cfg = cfg.get('trainer', {}).get('args', {}).get('geo_cond_model', None)
    if geo_cfg is not None:
        geo_cond_model = models.from_pretrained(geo_cfg['pretrained']).cuda().eval()
        geo_cond_model.requires_grad_(False)

    # Load image_cond_model (DINOv3) for image encoding
    image_cond_model = None
    img_cfg = cfg.get('trainer', {}).get('args', {}).get('image_cond_model', None)
    if img_cfg is not None:
        from trellis2.trainers import DinoV3FeatureExtractor
        image_cond_model = DinoV3FeatureExtractor(**img_cfg.get('args', {}))
        image_cond_model.cuda()

    # Get single sample
    batch = get_single_sample(dataset)

    # Run sampling
    print('\n=== Running flow sampling ===')
    if args.denoise_t is not None:
        pred, gt = one_step_denoise(model, batch, t_noise=args.denoise_t, 
                                    geo_cond_model=geo_cond_model, image_cond_model=image_cond_model)
    else:
        pred, gt = sample_flow(model, batch, steps=args.steps, geo_cond_model=geo_cond_model, 
                               image_cond_model=image_cond_model, use_cfg=not args.no_cfg, 
                               guidance_strength=args.guidance)

    # Compute metrics
    print('\n=== Computing metrics ===')
    if isinstance(pred, SparseTensor):
        metrics = compute_sparse_mse(pred, gt)
    else:
        # First compute latent MSE
        metrics = compute_ss_iou(pred, gt)
        # If latent, decode through ss_dec for IoU
        if 'latent_mse' in metrics:
            ss_dec_path = cfg.get('dataset', {}).get('args', {}).get('pretrained_ss_dec', None)
            if ss_dec_path is not None:
                print(f'Decoding latents through ss_dec for IoU...')
                ss_dec = models.from_pretrained(ss_dec_path).cuda().eval()
                with torch.no_grad():
                    pred_occ = ss_dec(pred.cuda())
                    gt_occ = ss_dec(gt.cuda())
                iou_metrics = compute_ss_iou(pred_occ, gt_occ, threshold=0.0)
                metrics.update(iou_metrics)
                del ss_dec
                torch.cuda.empty_cache()
    print(f'Metrics: {metrics}')

    # Gate=0 consistency
    print('\n=== Gate=0 consistency check ===')
    gate_check = gate_zero_consistency_check(model, batch, geo_cond_model)
    print(f'Gate=0 check: {gate_check}')

    # Summary
    results = {
        'config': args.config,
        'step': args.step,
        'sampling_steps': args.steps,
        'metrics': metrics,
        'gate_zero_consistency': gate_check,
    }
    print(f'\n=== Results ===')
    print(json.dumps(results, indent=2))

    # Save
    out_path = os.path.join(args.output_dir, f'eval_step{args.step:07d}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
