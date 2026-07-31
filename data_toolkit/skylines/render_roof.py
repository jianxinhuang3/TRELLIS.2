"""
T0 step 3 (condition side A): top-down "satellite style" renders.

For each instance renders 8 near-top-down views (pitch 75-88 deg from the
horizontal plane, random yaw), 512x512 RGBA with transparent background,
then applies satellite-style augmentation to the RGB channels (alpha kept).
Writes render_cond/{sha256}/000.png..007.png + transforms.json
(format compatible with trellis2/datasets/components.py: frames[].file_path).

Renderers:
    blender   : preferred; uses blender_render_roof.py (CYCLES, PBR materials)
    nvdiffrast: fallback; unlit base-color texture rasterization (CUDA, no GL)

Usage:
    python render_roof.py --instances <sha256> [--renderer auto|blender|nvdiffrast]
                          [--seed 0] [--blender_path /tmp/blender-4.5.1-linux-x64/blender]
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import json
import argparse
import zlib
import subprocess
import numpy as np
from PIL import Image

from data_toolkit.skylines import common

DEFAULT_BLENDER = '/tmp/blender-4.5.1-linux-x64/blender'
NUM_VIEWS = 8
RESOLUTION = 512


def make_views(rng):
    """8 near-top-down views: yaw uniform + jitter, pitch in [75, 88] deg."""
    views = []
    for i in range(NUM_VIEWS):
        yaw = 2 * np.pi * i / NUM_VIEWS + rng.uniform(0, 2 * np.pi / NUM_VIEWS)
        tilt = rng.uniform(2.0, 15.0)                      # deviation from straight down
        pitch = np.deg2rad(90.0 - tilt)                    # 75..88 deg
        fov = np.deg2rad(rng.uniform(30.0, 50.0))
        radius = np.sqrt(3) / 2 / np.sin(fov / 2)          # object fits the frame
        views.append({'yaw': float(yaw), 'pitch': float(pitch),
                      'radius': float(radius), 'fov': float(fov)})
    return views


def satellite_augment(img, rng):
    """
    Satellite-style augmentation on an RGBA uint8 array.
    RGB only: blur via random down-up sampling, color temperature shift,
    brightness perturbation, light gaussian noise. Alpha untouched.
    """
    rgb = img[..., :3].astype(np.float32)
    alpha = img[..., 3:]

    # random downsample-upsample blur (satellite GSD)
    factor = rng.uniform(1.5, 3.0)
    h, w = rgb.shape[:2]
    small = (max(1, int(w / factor)), max(1, int(h / factor)))
    pil = Image.fromarray(rgb.astype(np.uint8))
    rgb = np.asarray(pil.resize(small, Image.BILINEAR).resize((w, h), Image.BILINEAR)).astype(np.float32)

    # color temperature shift: warm/cool
    t = rng.uniform(-0.08, 0.08)
    rgb[..., 0] *= (1 + t)
    rgb[..., 2] *= (1 - t)

    # brightness perturbation
    rgb *= rng.uniform(0.85, 1.15)

    # light gaussian noise
    rgb += rng.normal(0, rng.uniform(1.0, 5.0), rgb.shape)

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.concatenate([rgb, alpha], axis=-1)


def augment_dir(out_dir, rng):
    for i in range(NUM_VIEWS):
        path = os.path.join(out_dir, f'{i:03d}.png')
        img = np.array(Image.open(path).convert('RGBA'))
        Image.fromarray(satellite_augment(img, rng)).save(path)


# --------------------------- blender path ---------------------------

def render_blender(glb_path, views, out_dir, blender_path):
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'blender_render_roof.py')
    args = [
        blender_path, '-b', '-P', script, '--',
        '--object', glb_path,
        '--views', json.dumps(views),
        '--output_folder', out_dir,
        '--resolution', str(RESOLUTION),
        '--engine', 'CYCLES',
    ]
    result = subprocess.run(args, capture_output=True, text=True, timeout=1800)
    if not os.path.exists(os.path.join(out_dir, 'transforms.json')):
        raise RuntimeError(
            f'blender render failed (returncode={result.returncode})\n'
            f'--- stdout tail ---\n{result.stdout[-2000:]}\n'
            f'--- stderr tail ---\n{result.stderr[-2000:]}'
        )


# --------------------------- nvdiffrast fallback ---------------------------

def _lookat(eye, center, up):
    f = center - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up); s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    view = np.eye(4, dtype=np.float32)
    view[0, :3], view[1, :3], view[2, :3] = s, u, -f
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def _perspective(fov, near=0.1, far=10.0):
    proj = np.zeros((4, 4), dtype=np.float32)
    t = 1 / np.tan(fov / 2)
    proj[0, 0] = t
    proj[1, 1] = t
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = 2 * far * near / (near - far)
    proj[3, 2] = -1
    return proj


def render_nvdiffrast(glb_path, views, out_dir):
    import torch
    import nvdiffrast.torch as dr

    geoms, _ = common.load_normalized_geoms(glb_path)
    verts, uvs, faces, face_mat, textures, factors = [], [], [], [], [], []
    vstart = 0
    for mid, g in enumerate(geoms):
        v = np.asarray(g.vertices, dtype=np.float32)
        f = np.asarray(g.faces, dtype=np.int64)
        uv = np.asarray(g.visual.uv, dtype=np.float32) if g.visual.uv is not None \
            else np.zeros((len(v), 2), dtype=np.float32)
        mat = g.visual.material
        tex = mat.baseColorTexture
        tex = np.asarray(tex.convert('RGB'), dtype=np.float32) / 255.0 if tex is not None else None
        factor = np.array(mat.baseColorFactor[:3], dtype=np.float32) / 255.0 \
            if mat.baseColorFactor is not None else np.ones(3, dtype=np.float32)
        verts.append(v); uvs.append(uv)
        faces.append(f + vstart)
        face_mat.append(np.full(len(f), mid, dtype=np.int64))
        textures.append(tex); factors.append(factor)
        vstart += len(v)

    device = 'cuda'
    verts = torch.from_numpy(np.concatenate(verts)).to(device)
    uvs = torch.from_numpy(np.concatenate(uvs)).to(device)
    faces = torch.from_numpy(np.concatenate(faces)).int().to(device)
    face_mat = torch.from_numpy(np.concatenate(face_mat)).to(device)
    ctx = dr.RasterizeCudaContext()

    os.makedirs(out_dir, exist_ok=True)
    to_export = {"aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], "frames": []}
    for i, view in enumerate(views):
        # y-up world: elevation = pitch above the horizontal (x-z) plane
        eye = view['radius'] * np.array([
            np.cos(view['yaw']) * np.cos(view['pitch']),
            np.sin(view['pitch']),
            np.sin(view['yaw']) * np.cos(view['pitch']),
        ], dtype=np.float32)
        up = np.array([0, 1, 0], dtype=np.float32)
        if abs(np.dot(eye / np.linalg.norm(eye), up)) > 0.999:
            up = np.array([1, 0, 0], dtype=np.float32)
        viewm = _lookat(eye, np.zeros(3, dtype=np.float32), up)
        mvp = torch.from_numpy(_perspective(view['fov']) @ viewm).to(device)

        pos = torch.cat([verts, torch.ones_like(verts[:, :1])], dim=1) @ mvp.T
        rast, _ = dr.rasterize(ctx, pos[None], faces, (RESOLUTION, RESOLUTION))
        uv_pix, _ = dr.interpolate(uvs[None], rast, faces)

        tri_id = rast[0, ..., 3].long() - 1                       # [-1 = background]
        mask = tri_id >= 0
        color = torch.zeros(RESOLUTION, RESOLUTION, 3, device=device)
        pix_mat = torch.full_like(tri_id, -1)
        pix_mat[mask] = face_mat[tri_id[mask]]
        for mid in range(len(geoms)):
            m = pix_mat == mid
            if not m.any():
                continue
            factor = torch.from_numpy(factors[mid]).to(device)
            if textures[mid] is None:
                color[m] = factor
                continue
            tex = torch.from_numpy(textures[mid]).to(device).permute(2, 0, 1)[None]
            uv_m = uv_pix[0][m]                                   # [K, 2]
            uv_m = uv_m - torch.floor(uv_m)                       # REPEAT wrap (glTF default)
            grid = torch.stack([uv_m[:, 0] * 2 - 1, (1 - uv_m[:, 1]) * 2 - 1], dim=-1)
            sampled = torch.nn.functional.grid_sample(
                tex, grid[None, None], mode='bilinear', align_corners=False)
            color[m] = sampled[0, :, 0].T * factor

        rgba = torch.cat([color, mask[..., None].float()], dim=-1)
        rgba = dr.antialias(rgba[None], rast, pos[None], faces)[0]
        img = (rgba.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        img = img[::-1]                                           # rasterizer y flip
        Image.fromarray(img, 'RGBA').save(os.path.join(out_dir, f'{i:03d}.png'))

        c2w = np.linalg.inv(viewm)
        to_export['frames'].append({
            'file_path': f'{i:03d}.png',
            'camera_angle_x': view['fov'],
            'transform_matrix': c2w.tolist(),
        })

    with open(os.path.join(out_dir, 'transforms.json'), 'w') as f:
        json.dump(to_export, f, indent=4)


# --------------------------- driver ---------------------------

def render_one(sha256, glb_path, renderer, blender_path, seed):
    rng = np.random.default_rng(seed + zlib.crc32(sha256.encode()))
    views = make_views(rng)
    out_dir = common.render_cond_dir(sha256)
    os.makedirs(out_dir, exist_ok=True)

    used = None
    if renderer in ('auto', 'blender'):
        try:
            render_blender(glb_path, views, out_dir, blender_path)
            used = 'blender'
        except Exception as e:
            print(f'[render_roof] blender failed for {sha256}: {e}')
            if renderer == 'blender':
                raise
    if used is None:
        render_nvdiffrast(glb_path, views, out_dir)
        used = 'nvdiffrast'

    augment_dir(out_dir, rng)
    common.update_metadata(sha256, cond_rendered=True)
    print(f'[render_roof] {sha256}: {NUM_VIEWS} views rendered with {used} -> {out_dir}')
    return used


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instances', type=str, required=True,
                        help='Comma separated sha256 list, or a file with one sha256 per line')
    parser.add_argument('--renderer', type=str, default='auto',
                        choices=['auto', 'blender', 'nvdiffrast'])
    parser.add_argument('--blender_path', type=str, default=DEFAULT_BLENDER)
    parser.add_argument('--seed', type=int, default=0)
    opt = parser.parse_args()

    metadata = common.load_metadata()
    for sha256 in common.parse_instances(opt.instances):
        render_one(sha256, metadata.loc[sha256, 'glb_path'], opt.renderer,
                   opt.blender_path, opt.seed)
    print('[render_roof] all done')
