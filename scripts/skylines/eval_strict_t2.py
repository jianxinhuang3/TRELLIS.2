"""
Strict multi-step evaluation for T2 Shape flow single-sample overfitting.

Same protocol as eval_strict.py (T1), but for the SPARSE shape flow
(InjectedSLatFlowModel): evaluates from PURE Gaussian noise on the GT
coords Sg (teacher-forcing) using manual multi-step Euler sampling.

Metrics:
  - latent MSE (pred_feats vs gt_feats, normalized space; random baseline ~= 2.0
    for two independent N(0,1) fields, gt-variance baseline ~= 1.0)
  - gate=0 consistency check (max|delta| < 1e-3)
  - optional shape_dec decode + normal-map rendering comparison
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trellis2 import models, datasets
from trellis2.modules.sparse import SparseTensor


def load_weights(model, ckpt_dir, step, weights='raw', ema_rate=0.9999):
    """Load RAW (denoiser_step*.pt) or EMA weights for the denoiser."""
    if weights == 'ema':
        ckpt_path = os.path.join(ckpt_dir, 'ckpts', f'denoiser_ema{ema_rate}_step{step:07d}.pt')
    else:
        ckpt_path = os.path.join(ckpt_dir, 'ckpts', f'denoiser_step{step:07d}.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'No checkpoint found at {ckpt_path}')
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert len(missing) == 0 and len(unexpected) == 0, \
        f'state_dict mismatch: missing={len(missing)} unexpected={len(unexpected)}'
    print(f'Loaded {weights.upper()} weights from {ckpt_path}')
    return model


def get_single_sample(dataset):
    """Get a single collated sample (SparseTensor x_0)."""
    sample = dataset[0]
    if isinstance(sample.get('x_0'), SparseTensor) or 'coords' in sample:
        return dataset.collate_fn([sample])
    return {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
            for k, v in sample.items()}


@torch.no_grad()
def sample_euler_manual_sparse(model, noise, cond, neg_cond, proxy_latent,
                               steps=50, guidance=1.0, cfg_mode='joint'):
    """
    Manual multi-step Euler sampling for sparse flow with proper CFG handling.

    Args:
        noise: SparseTensor with pure Gaussian feats on GT coords Sg.
        cond / neg_cond: [B, Lc, 1024] DINOv3 tokens / zeros.
        proxy_latent: [B, 8, 16, 16, 16] geometry condition.
        cfg_mode: 'joint' (neg drops proxy too) or 'standard' (neg keeps proxy).
    Returns:
        SparseTensor with the final sampled feats.
    """
    x_t = noise
    t_seq = np.linspace(1, 0, steps + 1)
    t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(steps)]

    for t, t_prev in tqdm(t_pairs, desc=f'Euler (g={guidance}, {cfg_mode})'):
        t_tensor = torch.full((x_t.shape[0],), t * 1000, device=x_t.device, dtype=torch.float32)

        if guidance == 1.0:
            pred_v = model(x_t, t_tensor, cond, proxy_latent=proxy_latent)
        else:
            pred_pos = model(x_t, t_tensor, cond, proxy_latent=proxy_latent)
            if cfg_mode == 'joint':
                pred_neg = model(x_t, t_tensor, neg_cond, proxy_latent=None)
            else:
                pred_neg = model(x_t, t_tensor, neg_cond, proxy_latent=proxy_latent)
            pred_v = pred_pos.replace(
                guidance * pred_pos.feats + (1 - guidance) * pred_neg.feats)

        dt = t - t_prev
        x_t = x_t.replace(x_t.feats - dt * pred_v.feats)

    return x_t


@torch.no_grad()
def gate_zero_consistency_check(model, x_0):
    """With gates=0, proxy_latent=random vs None must give max|delta| < 1e-3."""
    original_gates = []
    for inj in model.injectors:
        original_gates.append(inj.gate.data.clone())
        inj.gate.data.zero_()

    x = x_0.replace(torch.randn_like(x_0.feats))
    B = x.shape[0]
    t = torch.full((B,), 500.0, device='cuda')
    cond = torch.randn(B, 1024, 1024, device='cuda')
    proxy_latent = torch.randn(B, 8, 16, 16, 16, device='cuda')

    y_with = model(x, t, cond, proxy_latent=proxy_latent)
    y_without = model(x, t, cond, proxy_latent=None)
    max_diff = (y_with.feats - y_without.feats).abs().max().item()

    for inj, g in zip(model.injectors, original_gates):
        inj.gate.data.copy_(g)

    return {'max_abs_diff': max_diff, 'pass': max_diff < 1e-3}


@torch.no_grad()
def render_comparison(dataset, pred, gt, save_path, title=''):
    """Decode pred/gt latents through shape_dec and save normal renders."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    images = []
    for name, z in [('Predicted', pred), ('GT', gt)]:
        img = dataset.visualize_sample(z.cuda())  # [1, 3, 1024, 1024]
        images.append((name, img[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()))

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, (name, img) in zip(axes, images):
        ax.imshow(img)
        ax.set_title(name)
        ax.set_axis_off()
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Visualization saved: {save_path}')


def main():
    parser = argparse.ArgumentParser(description='Strict multi-step evaluation for T2 Shape flow')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--guidances', type=str, default='1.0,3.0')
    parser.add_argument('--cfg_modes', type=str, default='joint,standard')
    parser.add_argument('--weights', type=str, default='raw', choices=['raw', 'ema'])
    parser.add_argument('--no_render', action='store_true', help='Skip shape_dec rendering')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    guidances = [float(g) for g in args.guidances.split(',')]
    cfg_modes = [m.strip() for m in args.cfg_modes.split(',')]

    with open(args.config, 'r') as f:
        cfg = edict(json.load(f))

    print('=== Building model and dataset ===')
    model = getattr(models, cfg.models.denoiser.name)(**cfg.models.denoiser.args).cuda()
    dataset = getattr(datasets, cfg.dataset.name)(args.data_dir, **cfg.dataset.args)
    model = load_weights(model, args.ckpt_dir, args.step, weights=args.weights)
    model.eval()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {n_trainable / 1e6:.2f}M')

    # geo_cond_model (frozen ss_enc)
    geo_cfg = cfg.get('trainer', {}).get('args', {}).get('geo_cond_model', None)
    geo_cond_model = None
    if geo_cfg is not None:
        geo_cond_model = models.from_pretrained(geo_cfg['pretrained']).cuda().eval()
        geo_cond_model.requires_grad_(False)
        print(f'Loaded geo_cond_model: {geo_cfg["pretrained"]}')

    # image_cond_model (DINOv3)
    image_cond_model = None
    img_cfg = cfg.get('trainer', {}).get('args', {}).get('image_cond_model', None)
    if img_cfg is not None:
        from trellis2.trainers import DinoV3FeatureExtractor
        image_cond_model = DinoV3FeatureExtractor(**img_cfg.get('args', {}))
        image_cond_model.cuda()
        print('Loaded image_cond_model: DinoV3FeatureExtractor')

    batch = get_single_sample(dataset)
    x_0 = batch['x_0'].cuda()
    print(f'x_0: tokens={x_0.feats.shape[0]}, dim={x_0.feats.shape[1]}, '
          f'feats std={x_0.feats.std().item():.4f}')

    cond = batch.get('cond')
    if cond is not None:
        cond = cond.cuda()
        if cond.ndim == 4 and image_cond_model is not None:
            cond = image_cond_model(cond)
    neg_cond = torch.zeros_like(cond)

    proxy_latent = None
    proxy_voxel = batch.get('proxy_voxel')
    if proxy_voxel is not None and geo_cond_model is not None:
        proxy_voxel = proxy_voxel.cuda()
        if proxy_voxel.ndim == 3:
            proxy_voxel = proxy_voxel.unsqueeze(0)
        if proxy_voxel.ndim == 4:
            proxy_voxel = proxy_voxel.unsqueeze(1)
        proxy_latent = geo_cond_model(proxy_voxel.float(), sample_posterior=False)
        print(f'proxy_latent shape: {proxy_latent.shape}')

    print('\n=== Gate values ===')
    gates = [torch.tanh(inj.gate).item() for inj in model.injectors]
    print(f'  Mean: {np.mean(gates):.4f}, Min: {np.min(gates):.4f}, Max: {np.max(gates):.4f}')

    # Random baseline: pure noise feats vs gt feats
    torch.manual_seed(args.seed)
    rand_baseline = F.mse_loss(torch.randn_like(x_0.feats), x_0.feats).item()
    print(f'Random baseline MSE (noise vs gt): {rand_baseline:.4f}')

    print(f'\n=== Multi-step Euler sweep: steps={args.steps}, guidances={guidances}, modes={cfg_modes} ===')
    results = {}
    best_mse = float('inf')
    best_config = None
    best_pred = None

    for cfg_mode in cfg_modes:
        for guidance in guidances:
            if guidance == 1.0 and cfg_mode == 'standard':
                continue
            config_key = f'{cfg_mode}_g{guidance}'
            print(f'\n--- Config: {config_key} ---')

            torch.manual_seed(args.seed)
            noise = x_0.replace(torch.randn_like(x_0.feats))

            pred = sample_euler_manual_sparse(
                model, noise, cond, neg_cond, proxy_latent,
                steps=args.steps, guidance=guidance, cfg_mode=cfg_mode)

            latent_mse = F.mse_loss(pred.feats, x_0.feats).item()
            print(f'  Latent MSE: {latent_mse:.6f} (random baseline {rand_baseline:.4f})')

            results[config_key] = {
                'guidance': guidance,
                'cfg_mode': cfg_mode,
                'latent_mse': latent_mse,
            }
            if latent_mse < best_mse:
                best_mse = latent_mse
                best_config = config_key
                best_pred = pred

    print('\n' + '=' * 60)
    print('=== EVALUATION RESULTS ===')
    print('=' * 60)
    print(f'{"Config":<20} {"Guidance":<10} {"CFG Mode":<10} {"Latent MSE":<12}')
    print('-' * 52)
    for key, res in results.items():
        print(f'{key:<20} {res["guidance"]:<10.1f} {res["cfg_mode"]:<10} {res["latent_mse"]:<12.6f}')
    print(f'\nBest config: {best_config} with latent MSE = {best_mse:.6f}')
    print(f'Target: MSE < 0.1 | {"PASS" if best_mse < 0.1 else "FAIL"}')

    print('\n=== Gate=0 consistency check ===')
    gate_check = gate_zero_consistency_check(model, x_0)
    print(f'Gate=0 check: {gate_check}')

    # Optional shape_dec rendering
    vis_path = None
    if not args.no_render and best_pred is not None:
        try:
            vis_path = os.path.join(args.output_dir, f'vis_best_{best_config}_step{args.step}.png')
            render_comparison(dataset, best_pred, x_0, vis_path,
                              title=f'Best: {best_config} | MSE={best_mse:.6f} | Step {args.step}')
        except Exception as e:
            print(f'[warn] rendering failed, skipped: {e}')
            vis_path = None

    summary = {
        'step': args.step,
        'weights': args.weights,
        'sampling_steps': args.steps,
        'seed': args.seed,
        'trainable_params_M': n_trainable / 1e6,
        'random_baseline_mse': rand_baseline,
        'results': results,
        'best_config': best_config,
        'best_latent_mse': best_mse,
        'pass_threshold_mse': 0.1,
        'passed': best_mse < 0.1,
        'gate_zero_consistency': gate_check,
        'visualization': vis_path,
    }
    out_json = os.path.join(args.output_dir, f'eval_strict_step{args.step:07d}.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nResults saved: {out_json}')


if __name__ == '__main__':
    main()
