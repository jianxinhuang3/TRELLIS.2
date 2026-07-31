"""
Strict multi-step evaluation for T1 SS flow single-sample overfitting.

Evaluates from PURE Gaussian noise using multi-step Euler sampling (no one-step denoise).
Supports two CFG modes:
  - "standard": neg branch still receives proxy_latent (both branches conditioned on geometry)
  - "joint": neg branch has proxy_latent=None (truly unconditional on geometry)

Sweeps guidance strengths and reports SS IoU for each configuration.
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_weights(model, ckpt_dir, step, weights='raw', ema_rate=0.9999):
    """Load weights for the denoiser model.

    weights='raw'  -> denoiser_step{...}.pt (the trained model; recommended for
                      short overfit runs where the slow EMA (0.9999) has not
                      caught up).
    weights='ema'  -> denoiser_ema{rate}_step{...}.pt
    """
    if weights == 'ema':
        ckpt_path = os.path.join(ckpt_dir, 'ckpts', f'denoiser_ema{ema_rate}_step{step:07d}.pt')
    else:
        ckpt_path = os.path.join(ckpt_dir, 'ckpts', f'denoiser_step{step:07d}.pt')
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        assert len(missing) == 0 and len(unexpected) == 0, \
            f'state_dict mismatch: missing={len(missing)} unexpected={len(unexpected)}'
        print(f'Loaded {weights.upper()} weights from {ckpt_path}')
    else:
        raise FileNotFoundError(f'No checkpoint found at {ckpt_path}')
    return model


def build_model_and_dataset(cfg, data_dir):
    """Build model and dataset from config."""
    model = getattr(models, cfg.models.denoiser.name)(**cfg.models.denoiser.args).cuda()
    dataset = getattr(datasets, cfg.dataset.name)(data_dir, **cfg.dataset.args)
    return model, dataset


def get_single_sample(dataset):
    """Get a single sample from dataset."""
    sample = dataset[0]
    if isinstance(sample.get('x_0'), SparseTensor):
        batch = dataset.collate_fn([sample])
    else:
        batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                 for k, v in sample.items()}
    return batch


@torch.no_grad()
def sample_euler_manual(model, noise, cond, neg_cond, proxy_latent,
                        steps=50, guidance=1.0, cfg_mode='joint', sigma_min=1e-5):
    """
    Manual multi-step Euler sampling with proper CFG handling.
    
    Args:
        model: The denoiser model
        noise: Initial pure Gaussian noise [B, 8, 16, 16, 16]
        cond: Positive condition (DINOv3 features) [B, 1029, 1024]
        neg_cond: Negative condition (zeros) [B, 1029, 1024]
        proxy_latent: Geometry condition [B, 8, 16, 16, 16]
        steps: Number of Euler steps
        guidance: CFG guidance strength (1.0 = no CFG)
        cfg_mode: 'standard' (neg branch keeps proxy_latent) or 'joint' (neg branch proxy_latent=None)
        sigma_min: Flow matching sigma_min
    
    Returns:
        Final sample tensor [B, 8, 16, 16, 16]
    """
    x_t = noise
    t_seq = np.linspace(1, 0, steps + 1)
    t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(steps)]
    
    for t, t_prev in tqdm(t_pairs, desc=f'Euler (g={guidance}, {cfg_mode})'):
        t_tensor = torch.full((x_t.shape[0],), t * 1000, device=x_t.device, dtype=torch.float32)
        
        if guidance == 1.0:
            # No CFG, just conditional prediction
            pred_v = model(x_t, t_tensor, cond, proxy_latent=proxy_latent)
        else:
            # Positive branch: full conditioning
            pred_pos = model(x_t, t_tensor, cond, proxy_latent=proxy_latent)
            
            # Negative branch: depends on cfg_mode
            if cfg_mode == 'joint':
                # Joint CFG: neg branch drops BOTH image cond and geometry
                pred_neg = model(x_t, t_tensor, neg_cond, proxy_latent=None)
            else:
                # Standard CFG: neg branch drops image cond only, keeps geometry
                pred_neg = model(x_t, t_tensor, neg_cond, proxy_latent=proxy_latent)
            
            # CFG combination
            pred_v = guidance * pred_pos + (1 - guidance) * pred_neg
        
        # Euler step: x_{t_prev} = x_t - (t - t_prev) * v
        dt = t - t_prev
        x_t = x_t - dt * pred_v
    
    return x_t


def compute_ss_iou(pred_occ, gt_occ, threshold=0.0):
    """Compute IoU between predicted and GT occupancy grids."""
    pred_binary = (pred_occ > threshold).float()
    gt_binary = (gt_occ > threshold).float()
    intersection = (pred_binary * gt_binary).sum().item()
    union = ((pred_binary + gt_binary) > 0).float().sum().item()
    iou = intersection / max(union, 1e-8)
    return iou


def render_voxel_comparison(pred_occ_np, gt_occ_np, save_path, title=''):
    """Render pred vs GT occupancy as side-by-side 3D views."""
    from scripts.skylines.vis_proxy_3d import exposed_face_quads, render_matplotlib
    
    views = [(30, 45), (30, 135), (30, 225)]
    nrows = 2
    ncols = len(views)
    
    fig = plt.figure(figsize=(5 * ncols, 5 * nrows))
    
    volumes = [
        ('Predicted', pred_occ_np, '#4878cf'),
        ('GT', gt_occ_np, '#e8853a'),
    ]
    
    for r, (name, vox, color) in enumerate(volumes):
        for c, (elev, azim) in enumerate(views):
            ax = fig.add_subplot(nrows, ncols, r * ncols + c + 1, projection='3d')
            # Use the exposed_face_quads rendering
            quads, dirs = exposed_face_quads(vox)
            if len(quads) > 0:
                disp = quads[:, :, [2, 0, 1]]  # (x,y,z) -> (z,x,y)
                shade_lut = {0: 0.80, 1: 0.55, 2: 1.00, 3: 0.35, 4: 0.65, 5: 0.45}
                base = np.array(matplotlib.colors.to_rgb(color))
                colors = np.array([np.clip(base * shade_lut[d], 0, 1) for d in dirs])
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                pc = Poly3DCollection(disp, facecolors=colors, edgecolors='none')
                ax.add_collection3d(pc)
            ax.set_xlim(0, 64); ax.set_ylim(0, 64); ax.set_zlim(0, 64)
            ax.set_box_aspect((1, 1, 1))
            ax.set_axis_off()
            ax.set_title(f'{name} elev={elev} azim={azim}', fontsize=9)
            ax.view_init(elev=elev, azim=azim)
    
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Visualization saved: {save_path}')


def main():
    parser = argparse.ArgumentParser(description='Strict multi-step evaluation for T1 SS flow')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--steps', type=int, default=50, help='Euler sampling steps')
    parser.add_argument('--guidances', type=str, default='1.0,2.0,3.0,5.0',
                        help='Comma-separated guidance values to sweep')
    parser.add_argument('--cfg_modes', type=str, default='joint,standard',
                        help='Comma-separated CFG modes to test')
    parser.add_argument('--weights', type=str, default='raw', choices=['raw', 'ema'],
                        help='Which weights to load: raw (trained model) or ema')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse sweep params
    guidances = [float(g) for g in args.guidances.split(',')]
    cfg_modes = [m.strip() for m in args.cfg_modes.split(',')]

    # Load config
    with open(args.config, 'r') as f:
        cfg = edict(json.load(f))

    # Build model and dataset
    print('=== Building model and dataset ===')
    model, dataset = build_model_and_dataset(cfg, args.data_dir)
    model = load_weights(model, args.ckpt_dir, args.step, weights=args.weights)
    model.eval()

    # Load geo_cond_model (ss_enc) for proxy encoding
    geo_cond_model = None
    geo_cfg = cfg.get('trainer', {}).get('args', {}).get('geo_cond_model', None)
    if geo_cfg is not None:
        geo_cond_model = models.from_pretrained(geo_cfg['pretrained']).cuda().eval()
        geo_cond_model.requires_grad_(False)
        print(f'Loaded geo_cond_model: {geo_cfg["pretrained"]}')

    # Load image_cond_model (DINOv3)
    image_cond_model = None
    img_cfg = cfg.get('trainer', {}).get('args', {}).get('image_cond_model', None)
    if img_cfg is not None:
        from trellis2.trainers import DinoV3FeatureExtractor
        image_cond_model = DinoV3FeatureExtractor(**img_cfg.get('args', {}))
        image_cond_model.cuda()
        print(f'Loaded image_cond_model: DinoV3FeatureExtractor')

    # Load ss_dec for occupancy decoding
    ss_dec_path = cfg.get('dataset', {}).get('args', {}).get('pretrained_ss_dec', None)
    ss_dec = None
    if ss_dec_path:
        ss_dec = models.from_pretrained(ss_dec_path).cuda().eval()
        print(f'Loaded ss_dec: {ss_dec_path}')

    # Get single sample
    batch = get_single_sample(dataset)

    # Prepare conditioning
    x_0 = batch['x_0'].cuda()
    cond = batch.get('cond')
    if cond is not None:
        cond = cond.cuda()
        if cond.ndim == 4 and image_cond_model is not None:
            cond = image_cond_model(cond)
    neg_cond = torch.zeros_like(cond)

    # Encode proxy_latent
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

    # Decode GT occupancy
    gt_occ = None
    if ss_dec is not None:
        gt_occ = ss_dec(x_0)
        gt_occ_np = (gt_occ[0, 0].detach().cpu().numpy() > 0).astype(np.uint8)
        print(f'GT occupancy: shape={gt_occ.shape}, occupied voxels={gt_occ_np.sum()}')

    # Print gate values for reference
    print('\n=== Gate values (EMA) ===')
    gates = [torch.tanh(inj.gate).item() for inj in model.injectors]
    print(f'  Mean: {np.mean(gates):.4f}, Min: {np.min(gates):.4f}, Max: {np.max(gates):.4f}')

    # Sweep configurations
    print(f'\n=== Multi-step Euler sweep: steps={args.steps}, guidances={guidances}, modes={cfg_modes} ===')
    results = {}
    best_iou = -1
    best_config = None
    best_pred_occ_np = None

    for cfg_mode in cfg_modes:
        for guidance in guidances:
            # Skip standard mode when guidance=1.0 (same as joint with g=1)
            if guidance == 1.0 and cfg_mode == 'standard':
                continue

            config_key = f'{cfg_mode}_g{guidance}'
            print(f'\n--- Config: {config_key} ---')

            # Set seed for reproducibility
            torch.manual_seed(args.seed)
            noise = torch.randn_like(x_0)

            # Run sampling
            pred_latent = sample_euler_manual(
                model, noise, cond, neg_cond, proxy_latent,
                steps=args.steps, guidance=guidance, cfg_mode=cfg_mode
            )

            # Compute latent MSE
            latent_mse = F.mse_loss(pred_latent, x_0).item()
            print(f'  Latent MSE: {latent_mse:.6f}')

            # Decode and compute IoU
            iou = None
            pred_occ_np_local = None
            if ss_dec is not None:
                pred_occ = ss_dec(pred_latent)
                pred_occ_np_local = (pred_occ[0, 0].detach().cpu().numpy() > 0).astype(np.uint8)
                iou = compute_ss_iou(pred_occ, gt_occ, threshold=0.0)
                print(f'  SS IoU (threshold=0): {iou:.4f}')
                print(f'  Pred occupied voxels: {pred_occ_np_local.sum()}, GT: {gt_occ_np.sum()}')

            results[config_key] = {
                'guidance': guidance,
                'cfg_mode': cfg_mode,
                'latent_mse': latent_mse,
                'iou': iou,
                'pred_occupied': int(pred_occ_np_local.sum()) if pred_occ_np_local is not None else None,
                'gt_occupied': int(gt_occ_np.sum()) if gt_occ_np is not None else None,
            }

            if iou is not None and iou > best_iou:
                best_iou = iou
                best_config = config_key
                best_pred_occ_np = pred_occ_np_local

    # Summary
    print('\n' + '='*60)
    print('=== EVALUATION RESULTS ===')
    print('='*60)
    print(f'{"Config":<20} {"Guidance":<10} {"CFG Mode":<10} {"IoU":<10} {"Latent MSE":<12} {"Pred Occ":<10}')
    print('-'*72)
    for key, res in results.items():
        iou_str = f'{res["iou"]:.4f}' if res["iou"] is not None else 'N/A'
        print(f'{key:<20} {res["guidance"]:<10.1f} {res["cfg_mode"]:<10} {iou_str:<10} {res["latent_mse"]:<12.6f} {res["pred_occupied"]}')
    
    print(f'\nBest config: {best_config} with IoU = {best_iou:.4f}')
    print(f'Target: IoU >= 0.85 | {"PASS" if best_iou >= 0.85 else "FAIL"}')

    # Generate visualization for best config
    if best_pred_occ_np is not None and gt_occ_np is not None:
        vis_path = os.path.join(args.output_dir, f'vis_best_{best_config}_step{args.step}.png')
        render_voxel_comparison(
            best_pred_occ_np, gt_occ_np, vis_path,
            title=f'Best: {best_config} | IoU={best_iou:.4f} | Step {args.step}'
        )

    # Save results JSON
    summary = {
        'step': args.step,
        'sampling_steps': args.steps,
        'seed': args.seed,
        'results': results,
        'best_config': best_config,
        'best_iou': best_iou,
        'pass_threshold': 0.85,
        'passed': best_iou >= 0.85,
    }
    out_json = os.path.join(args.output_dir, f'eval_strict_step{args.step:07d}.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nResults saved: {out_json}')


if __name__ == '__main__':
    main()
