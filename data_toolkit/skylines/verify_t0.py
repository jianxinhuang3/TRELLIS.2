"""
T0 step 5: verify all artifacts of one instance.

Checks:
    - metadata.csv row complete
    - ss latent [8,16,16,16], shape/tex latents [Mg,32] with identical coords,
      coords < 32 (token grid of the f16 encoders at 512), no NaN/Inf anywhere
    - unique(ss64_occupancy // 2) == latent coords
    - proxy voxel [64,64,64] uint8, non-empty, footprint overlaps GT
    - render_cond: transforms.json + 8 RGBA 512x512 PNGs with non-trivial alpha
    - frozen ss_dec: decode z*_ss -> 64^3 occupancy (>0), IoU with GT > 0.95
    - frozen shape_dec: decode z*_shape at 512 -> export mesh to vis/ for eyeballing

Usage:
    python verify_t0.py --instances <sha256>[,<sha256>...]
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import json
import argparse
import numpy as np
import torch
import trimesh
from PIL import Image

import trellis2.models as models
import trellis2.modules.sparse as sp
from data_toolkit.skylines import common

torch.set_grad_enabled(False)


def verify_one(sha256, ss_dec, shape_dec):
    results = {}

    # ---- metadata ----
    metadata = common.load_metadata()
    assert sha256 in metadata.index, f'{sha256} missing in metadata.csv'
    row = metadata.loc[sha256]
    for col in ['glb_path', 'height', 'size_x', 'size_z', 'aesthetic_score',
                'ss_latent_encoded', 'shape_latent_encoded', 'shape_latent_tokens',
                'tex_latent_encoded', 'tex_latent_tokens', 'cond_rendered', 'proxy_built']:
        assert col in metadata.columns and not pd_isna(row[col]), f'metadata field missing: {col}'
    results['metadata'] = 'OK'

    # ---- latents ----
    ss = np.load(common.ss_latent_path(sha256))
    z_ss = ss['z']
    assert z_ss.shape == (8, 16, 16, 16), f'ss latent shape {z_ss.shape}'
    assert np.isfinite(z_ss).all(), 'NaN/Inf in ss latent'
    results['ss_latent'] = f'z {z_ss.shape} std={z_ss.std():.3f}'

    shape = np.load(common.shape_latent_path(sha256))
    tex = np.load(common.tex_latent_path(sha256))
    Mg = shape['coords'].shape[0]
    assert shape['coords'].shape == (Mg, 3) and shape['feats'].shape == (Mg, 32), \
        f"shape latent shapes {shape['coords'].shape} {shape['feats'].shape}"
    assert tex['coords'].shape == (Mg, 3) and tex['feats'].shape == (Mg, 32), \
        f"tex latent shapes {tex['coords'].shape} {tex['feats'].shape}"
    assert np.array_equal(shape['coords'], tex['coords']), 'tex coords != shape coords'
    assert shape['coords'].max() < common.COARSE_RES, 'coords out of 32^3'
    assert np.isfinite(shape['feats']).all() and np.isfinite(tex['feats']).all(), 'NaN/Inf in slat feats'
    assert int(row['shape_latent_tokens']) == Mg and int(row['tex_latent_tokens']) == Mg, \
        'metadata token count mismatch'
    results['shape_latent'] = f"Mg={Mg} feats std={shape['feats'].std():.3f}"
    results['tex_latent'] = f"Mg={Mg} feats std={tex['feats'].std():.3f}"

    # ---- aux occupancy vs coords ----
    aux = np.load(common.aux_path(sha256))
    occ = aux['ss_occupancy']
    assert occ.shape == (64, 64, 64) and occ.any(), 'bad ss occupancy'
    ss64 = np.argwhere(occ)
    ss32 = np.unique(ss64 // 2, axis=0)
    coords_sorted = shape['coords'][np.lexsort(shape['coords'].T[::-1])]
    assert np.array_equal(ss32, coords_sorted), 'unique(ss64 // 2) != latent coords'
    results['occupancy'] = f'ss64={len(ss64)} voxels, consistent with coords'

    # ---- proxy ----
    proxy = np.load(common.proxy_path(sha256))
    pv = proxy['voxel']
    assert pv.shape == (64, 64, 64) and pv.dtype == np.uint8 and pv.any(), 'bad proxy voxel'
    fp_gt = occ.any(axis=1)
    fp_px = pv.any(axis=1)
    fp_iou = np.logical_and(fp_gt, fp_px).sum() / np.logical_or(fp_gt, fp_px).sum()
    assert fp_iou > 0.5, f'proxy footprint IoU too low: {fp_iou:.3f}'
    results['proxy'] = f'voxels={int(pv.sum())} footprint IoU={fp_iou:.3f} ' \
                       f"h_gt={int(proxy['h_gt'])} h_proxy={int(proxy['h_proxy'])}"

    # ---- render_cond ----
    rdir = common.render_cond_dir(sha256)
    with open(os.path.join(rdir, 'transforms.json')) as f:
        transforms = json.load(f)
    assert len(transforms['frames']) == 8, f"expected 8 frames, got {len(transforms['frames'])}"
    for frame in transforms['frames']:
        img = Image.open(os.path.join(rdir, frame['file_path']))
        assert img.size == (512, 512) and img.mode == 'RGBA', \
            f"{frame['file_path']}: {img.size} {img.mode}"
        alpha = np.array(img)[..., 3]
        assert (alpha > 0).mean() > 0.01, f"{frame['file_path']}: alpha nearly empty"
    results['render_cond'] = f"8 frames RGBA 512x512, transforms.json OK"

    # ---- ss_dec roundtrip IoU ----
    z = torch.from_numpy(z_ss).float().cuda()[None]
    dec_occ = (ss_dec(z) > 0)[0, 0].cpu().numpy()
    gt_occ = occ.astype(bool)
    iou = np.logical_and(dec_occ, gt_occ).sum() / np.logical_or(dec_occ, gt_occ).sum()
    results['ss_dec_iou'] = f'{iou:.4f}'
    assert iou > 0.95, f'ss_dec roundtrip IoU {iou:.4f} <= 0.95'

    # ---- shape_dec roundtrip mesh ----
    coords = torch.from_numpy(shape['coords'].astype(np.int32))
    z_sp = sp.SparseTensor(
        torch.from_numpy(shape['feats']).float(),
        torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1),
    ).cuda()
    meshes = shape_dec(z_sp)
    mesh = meshes[0] if isinstance(meshes, (list, tuple)) else meshes
    v = mesh.vertices.cpu().numpy()
    f = mesh.faces.cpu().numpy()
    assert len(v) > 0 and len(f) > 0 and np.isfinite(v).all(), 'decoded mesh invalid'
    os.makedirs(common.vis_dir(), exist_ok=True)
    out_mesh = os.path.join(common.vis_dir(), f'{sha256}_shape_dec.ply')
    trimesh.Trimesh(vertices=v, faces=f, process=False).export(out_mesh)
    results['shape_dec'] = f'{len(v)} verts / {len(f)} faces -> {out_mesh}'

    print(f'\n===== verify_t0: {sha256} =====')
    for k, v_ in results.items():
        print(f'  [PASS] {k}: {v_}')
    print(f'===== ALL CHECKS PASSED ({sha256}) =====\n')


def pd_isna(v):
    import pandas as pd
    return pd.isna(v)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instances', type=str, required=True,
                        help='Comma separated sha256 list, or a file with one sha256 per line')
    opt = parser.parse_args()

    ss_dec = models.from_pretrained(common.SS_DEC_PRETRAINED).eval().cuda()
    shape_dec = models.from_pretrained(common.SHAPE_DEC_PRETRAINED).eval().cuda()
    shape_dec.set_resolution(common.DUAL_GRID_RES)

    for sha256 in common.parse_instances(opt.instances):
        verify_one(sha256, ss_dec, shape_dec)
    print('[verify_t0] all done')
