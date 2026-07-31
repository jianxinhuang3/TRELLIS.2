"""
T0 step 4 (condition side G): geometry proxy from the GT 64^3 occupancy.

Pipeline (pipeline.md L27-28):
    64^3 GT occupancy (from aux npz, built by voxelize_and_encode.py)
    -> top-down footprint mask (project along the gravity axis y)
    -> cv2.findContours + approxPolyDP simplification
       (epsilon ~ U(1, 3) px, vertex jitter +-1 px)
    -> filled simplified footprint
    -> height = GT occupied height * U(0.85, 1.15)
    -> extrude along y -> proxy voxel [64, 64, 64] uint8

Also writes visualization PNGs (footprint GT vs proxy, three-view projections)
to EXTRACT_ROOT/vis/ for manual inspection.

Usage:
    python build_proxy.py --instances <sha256>[,<sha256>...] [--seed 0]
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import argparse
import zlib
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_toolkit.skylines import common


def simplify_footprint(mask, rng):
    """mask: [64, 64] uint8 (indexed [x, z]). Returns (proxy_mask, params)."""
    epsilon = float(rng.uniform(1.0, 3.0))
    contours, _ = cv2.findContours(
        (mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) > 0, 'empty footprint mask'
    proxy = np.zeros_like(mask, dtype=np.uint8)
    total_vertices = 0
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, epsilon, True)          # [K, 1, 2]
        jitter = rng.integers(-1, 2, size=approx.shape)        # +-1 px vertex jitter
        approx = np.clip(approx + jitter, 0, mask.shape[0] - 1)
        if len(approx) < 3:
            continue
        cv2.fillPoly(proxy, [approx.astype(np.int32)], 1)
        total_vertices += len(approx)
    assert proxy.sum() > 0, 'proxy footprint empty after simplification'
    return proxy, {'epsilon': epsilon, 'num_vertices': total_vertices,
                   'num_contours': len(contours)}


def build_one(sha256, seed):
    aux = np.load(common.aux_path(sha256))
    occ = aux['ss_occupancy']                                  # [64,64,64], axes (x, y, z), y = up
    assert occ.shape == (common.SS_RES,) * 3 and occ.any(), f'bad ss_occupancy for {sha256}'

    rng = np.random.default_rng(seed + zlib.crc32(sha256.encode()))

    # top-down footprint: project along gravity axis y
    footprint_gt = occ.any(axis=1).astype(np.uint8)            # [x, z]
    proxy_fp, params = simplify_footprint(footprint_gt, rng)

    # perturbed height, extruded from the GT base layer
    ys = np.nonzero(occ.any(axis=(0, 2)))[0]
    y0, y1 = int(ys.min()), int(ys.max())
    h_gt = y1 - y0 + 1
    height_scale = float(rng.uniform(0.85, 1.15))
    h_proxy = int(np.clip(round(h_gt * height_scale), 1, common.SS_RES - y0))

    proxy = np.zeros_like(occ, dtype=np.uint8)
    proxy[:, y0:y0 + h_proxy, :] = proxy_fp[:, None, :]

    os.makedirs(os.path.dirname(common.proxy_path(sha256)), exist_ok=True)
    np.savez_compressed(
        common.proxy_path(sha256),
        voxel=proxy,
        epsilon=params['epsilon'],
        num_vertices=params['num_vertices'],
        num_contours=params['num_contours'],
        height_scale=height_scale,
        h_gt=h_gt,
        h_proxy=h_proxy,
        y_base=y0,
        seed=seed,
    )
    common.update_metadata(sha256, proxy_built=True)

    # ---- visualization ----
    os.makedirs(common.vis_dir(), exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].imshow(footprint_gt.T, origin='lower', cmap='gray')
    axes[0, 0].set_title('footprint GT (x-z)')
    axes[0, 1].imshow(proxy_fp.T, origin='lower', cmap='gray')
    axes[0, 1].set_title(f'footprint proxy (eps={params["epsilon"]:.2f})')
    overlay = np.stack([footprint_gt, proxy_fp, np.zeros_like(proxy_fp)], axis=-1) * 255
    axes[0, 2].imshow(overlay.transpose(1, 0, 2), origin='lower')
    axes[0, 2].set_title('overlay (R=GT, G=proxy)')
    axes[0, 3].axis('off')
    axes[0, 3].text(0.05, 0.5,
                    f'h_gt={h_gt}  h_proxy={h_proxy}\n'
                    f'height_scale={height_scale:.3f}\n'
                    f'vertices={params["num_vertices"]}  contours={params["num_contours"]}\n'
                    f'IoU(footprint)={np.logical_and(footprint_gt, proxy_fp).sum() / np.logical_or(footprint_gt, proxy_fp).sum():.3f}',
                    fontsize=11, va='center')
    for i, (vol, name) in enumerate([(occ, 'GT'), (proxy, 'proxy')]):
        axes[1, 2 * i].imshow(vol.any(axis=2).T, origin='lower', cmap='gray')     # x-y front
        axes[1, 2 * i].set_title(f'{name} front (x-y)')
        axes[1, 2 * i + 1].imshow(vol.any(axis=0).T, origin='lower', cmap='gray') # z-y side
        axes[1, 2 * i + 1].set_title(f'{name} side (z-y)')
    fig.suptitle(sha256)
    fig.tight_layout()
    vis_path = os.path.join(common.vis_dir(), f'{sha256}_proxy.png')
    fig.savefig(vis_path, dpi=100)
    plt.close(fig)

    fp_iou = np.logical_and(footprint_gt, proxy_fp).sum() / np.logical_or(footprint_gt, proxy_fp).sum()
    print(f'[build_proxy] {sha256}: h_gt={h_gt} h_proxy={h_proxy} '
          f'footprint IoU={fp_iou:.3f} vertices={params["num_vertices"]} vis={vis_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instances', type=str, required=True,
                        help='Comma separated sha256 list, or a file with one sha256 per line')
    parser.add_argument('--seed', type=int, default=0)
    opt = parser.parse_args()

    for sha256 in common.parse_instances(opt.instances):
        build_one(sha256, opt.seed)
    print('[build_proxy] all done')
