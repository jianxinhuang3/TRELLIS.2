"""
One-off 3D visualization of proxy voxels (and GT ss occupancy if available).

Voxel axis convention (see data_toolkit/skylines/build_proxy.py):
[64, 64, 64] indexed [x, y, z], y is the height (up) axis.

Rendering strategy:
1. Try trimesh: VoxelGrid.as_boxes() + scene.save_image (needs a GL context,
   fails on headless boxes without EGL/display).
2. Fallback: matplotlib Poly3DCollection over exposed voxel faces only
   (pure CPU, no GPU / GL required).

Usage:
python vis_proxy_3d.py --instance ddb33c1bdba1999ac314153ea8990cbd_obj_0
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXTRACT_ROOT = '/data5/jianxin/dataset/skylines_50k_data_extracted'
VIEWS = [(30, 45), (30, 135), (30, 225), (30, 315)]  # (elev, azim) deg


def exposed_face_quads(vox):
    """Return (quads [N,4,3] float, normal_axis_dir [N] int) of exterior faces.

    Each occupied voxel (i,j,k) spans [i,i+1]x[j,j+1]x[k,k+1]; a face is
    emitted only where the neighbor along that direction is empty.
    """
    occ = vox.astype(bool)
    pad = np.pad(occ, 1)
    quads, dirs = [], []
    # corner offsets in the two in-plane axes, CCW
    corner2d = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    for axis in range(3):
        for d in (1, -1):
            nb = np.roll(pad, -d, axis=axis)
            face = pad & ~nb
            idx = np.argwhere(face[1:-1, 1:-1, 1:-1])
            if len(idx) == 0:
                continue
            a1, a2 = [a for a in range(3) if a != axis]
            q = np.repeat(idx[:, None, :].astype(np.float64), 4, axis=1)
            q[:, :, axis] += 1.0 if d > 0 else 0.0
            q[:, :, a1] += corner2d[None, :, 0]
            q[:, :, a2] += corner2d[None, :, 1]
            quads.append(q)
            dirs.append(np.full(len(idx), axis * 2 + (0 if d > 0 else 1)))
    return np.concatenate(quads), np.concatenate(dirs)


def render_matplotlib(ax, vox, base_color, title):
    """Draw exposed voxel faces on a 3D axis. Data (x,y,z) y-up is mapped to
    display (X=z, Y=x, Z=y) -- a proper rotation, so chirality is kept."""
    quads, dirs = exposed_face_quads(vox)
    disp = quads[:, :, [2, 0, 1]]  # (x,y,z) -> (z,x,y), Z is up
    # simple per-face-direction lambert-ish shading: +y (top) brightest
    shade_lut = {0: 0.80, 1: 0.55, 2: 1.00, 3: 0.35, 4: 0.65, 5: 0.45}
    base = np.array(matplotlib.colors.to_rgb(base_color))
    colors = np.array([np.clip(base * shade_lut[d], 0, 1) for d in dirs])
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    pc = Poly3DCollection(disp, facecolors=colors, edgecolors='none')
    ax.add_collection3d(pc)
    ax.set_xlim(0, 64); ax.set_ylim(0, 64); ax.set_zlim(0, 64)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)


def try_trimesh_render(vox):
    """Return list of PNG bytes (one per view) or None on failure."""
    try:
        import trimesh
        from trimesh.voxel import VoxelGrid
        mesh = VoxelGrid(vox.astype(bool)).as_boxes()
        images = []
        for elev, azim in VIEWS:
            scene = mesh.scene()
            scene.set_camera(angles=(np.radians(-elev), np.radians(azim), 0))
            images.append(scene.save_image(resolution=(512, 512), visible=False))
        return images
    except BaseException as e:  # pyglet raises non-Exception on headless
        print(f'[vis_proxy_3d] trimesh offscreen render unavailable ({type(e).__name__}: {e}), '
              'falling back to matplotlib')
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', default='ddb33c1bdba1999ac314153ea8990cbd_obj_0')
    args = parser.parse_args()

    proxy_path = os.path.join(EXTRACT_ROOT, 'proxy', f'{args.instance}.npz')
    aux_path = os.path.join(EXTRACT_ROOT, 'aux', f'{args.instance}.npz')
    out_dir = os.path.join(EXTRACT_ROOT, 'vis')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{args.instance}_proxy_3d.png')

    proxy = np.load(proxy_path)['voxel']
    print(f'proxy voxel: {proxy.shape} {proxy.dtype}, occupied={int(proxy.sum())}')
    gt = None
    if os.path.isfile(aux_path):
        aux = np.load(aux_path)
        if 'ss_occupancy' in aux:
            gt = aux['ss_occupancy']
            print(f'GT ss_occupancy: {gt.shape}, occupied={int(gt.sum())}, '
                  f"up_axis={aux['up_axis']}")

    volumes = [('proxy', proxy, '#4878cf')]
    if gt is not None:
        volumes.append(('GT ss_occupancy', gt, '#e8853a'))

    trimesh_imgs = try_trimesh_render(proxy)
    nrows, ncols = len(volumes), len(VIEWS)
    if trimesh_imgs is not None and gt is not None:
        trimesh_imgs_gt = try_trimesh_render(gt)
    else:
        trimesh_imgs_gt = None

    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
    if trimesh_imgs is not None and (gt is None or trimesh_imgs_gt is not None):
        import io
        all_imgs = [trimesh_imgs] + ([trimesh_imgs_gt] if gt is not None else [])
        for r, ((name, _, _), imgs) in enumerate(zip(volumes, all_imgs)):
            for c, ((elev, azim), png) in enumerate(zip(VIEWS, imgs)):
                ax = fig.add_subplot(nrows, ncols, r * ncols + c + 1)
                ax.imshow(plt.imread(io.BytesIO(png)))
                ax.set_axis_off()
                ax.set_title(f'{name} elev={elev} azim={azim}', fontsize=9)
    else:
        for r, (name, vox, color) in enumerate(volumes):
            for c, (elev, azim) in enumerate(VIEWS):
                ax = fig.add_subplot(nrows, ncols, r * ncols + c + 1, projection='3d')
                render_matplotlib(ax, vox, color, f'{name} elev={elev} azim={azim}')
                ax.view_init(elev=elev, azim=azim)
    fig.suptitle(args.instance, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {out_path}')


if __name__ == '__main__':
    main()
