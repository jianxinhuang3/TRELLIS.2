"""
Shared helpers for the SatSkylines T0 offline data construction pipeline.

All outputs are written under EXTRACT_ROOT following the protocol:
    metadata.csv
    ss_latents/ss_enc_conv3d_16l8_fp16_64/{sha256}.npz
    shape_latents/shape_enc_next_dc_f16c32_fp16_512/{sha256}.npz
    tex_latents/tex_enc_next_dc_f16c32_fp16_512/{sha256}.npz
    render_cond/{sha256}/transforms.json + 000.png..007.png
    proxy/{sha256}.npz
    vis/          (human inspection only)
    aux/          (intermediate: normalization params + 64^3 GT occupancy)
"""
import os
import numpy as np
import pandas as pd
import trimesh

TRELLIS2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
WEIGHTS_ROOT = os.path.join(TRELLIS2_ROOT, 'weights')
EXTRACT_ROOT = '/data5/jianxin/dataset/skylines_50k_data_extracted'
LABELS_TSV = '/data5/jianxin/dataset/SatSkylines/tools/data/skylines_50k/skylines_50k_labels.tsv'

SS_ENC_PRETRAINED = os.path.join(WEIGHTS_ROOT, 'TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16')
SS_DEC_PRETRAINED = os.path.join(WEIGHTS_ROOT, 'TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16')
SHAPE_ENC_PRETRAINED = os.path.join(WEIGHTS_ROOT, 'TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16')
SHAPE_DEC_PRETRAINED = os.path.join(WEIGHTS_ROOT, 'TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16')
TEX_ENC_PRETRAINED = os.path.join(WEIGHTS_ROOT, 'TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16')

SS_LATENT_NAME = 'ss_enc_conv3d_16l8_fp16_64'
SHAPE_LATENT_NAME = 'shape_enc_next_dc_f16c32_fp16_512'
TEX_LATENT_NAME = 'tex_enc_next_dc_f16c32_fp16_512'

DUAL_GRID_RES = 512     # dual-grid / PBR voxelization resolution
SS_RES = 64             # sparse structure occupancy resolution
COARSE_RES = 32         # token grid of the f16 encoders at 512 (512 / 16)


def ss_latent_path(sha256):
    return os.path.join(EXTRACT_ROOT, 'ss_latents', SS_LATENT_NAME, f'{sha256}.npz')


def shape_latent_path(sha256):
    return os.path.join(EXTRACT_ROOT, 'shape_latents', SHAPE_LATENT_NAME, f'{sha256}.npz')


def tex_latent_path(sha256):
    return os.path.join(EXTRACT_ROOT, 'tex_latents', TEX_LATENT_NAME, f'{sha256}.npz')


def render_cond_dir(sha256):
    return os.path.join(EXTRACT_ROOT, 'render_cond', sha256)


def proxy_path(sha256):
    return os.path.join(EXTRACT_ROOT, 'proxy', f'{sha256}.npz')


def aux_path(sha256):
    return os.path.join(EXTRACT_ROOT, 'aux', f'{sha256}.npz')


def vis_dir():
    return os.path.join(EXTRACT_ROOT, 'vis')


def metadata_path():
    return os.path.join(EXTRACT_ROOT, 'metadata.csv')


def update_metadata(sha256, **fields):
    """Update (or create) rows of metadata.csv keyed by sha256."""
    path = metadata_path()
    if os.path.exists(path):
        metadata = pd.read_csv(path).set_index('sha256')
    else:
        metadata = pd.DataFrame(columns=['sha256']).set_index('sha256')
    for k, v in fields.items():
        if k not in metadata.columns:
            metadata[k] = pd.Series(dtype=object)
        metadata[k] = metadata[k].astype(object)
        metadata.loc[sha256, k] = v
    metadata.reset_index().to_csv(path, index=False)


def load_metadata():
    path = metadata_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found; run build_metadata.py first')
    return pd.read_csv(path).set_index('sha256')


def parse_instances(instances):
    """--instances accepts a comma separated list or a file with one sha256 per line."""
    if os.path.exists(instances):
        with open(instances, 'r') as f:
            return [l.strip() for l in f.read().splitlines() if l.strip()]
    return [s.strip() for s in instances.split(',') if s.strip()]


def load_normalized_geoms(glb_path):
    """
    Load a GLB with trimesh and normalize all geometries into [-0.5, 0.5]^3.

    Follows the data_toolkit convention (dual_grid.py / voxelize_pbr.py / mesh2ovox.py):
        center = (bbox_min + bbox_max) / 2
        scale  = 0.99999 / (bbox_max - bbox_min).max()
        v_norm = (v - center) * scale

    NOTE: trimesh keeps the glTF y-up convention, so the gravity axis is +y.

    Returns:
        geoms (List[trimesh.Trimesh]): normalized geometries (world transforms applied)
        norm (dict): {'center': [3], 'scale': float, 'up_axis': 'y'}
    """
    scene = trimesh.load(glb_path, process=False)
    if isinstance(scene, trimesh.Trimesh):
        geoms = [scene]
    else:
        geoms = scene.dump()  # applies scene-graph transforms
    geoms = [g for g in geoms if len(g.vertices) > 0 and len(g.faces) > 0]
    assert len(geoms) > 0, f'no valid geometry in {glb_path}'

    vmin = np.min([g.vertices.min(axis=0) for g in geoms], axis=0)
    vmax = np.max([g.vertices.max(axis=0) for g in geoms], axis=0)
    center = (vmin + vmax) / 2
    scale = 0.99999 / (vmax - vmin).max()
    for g in geoms:
        g.apply_translation(-center)
        g.apply_scale(scale)

    norm = {'center': center.astype(np.float64), 'scale': float(scale), 'up_axis': 'y'}
    return geoms, norm


def sanitize_pbr_textures(geoms, max_size=2048):
    """
    Make geometries safe for o_voxel.convert.textured_mesh_to_volumetric_attr,
    whose C++ mipmap builder requires square power-of-two textures.

    - non-TextureVisuals (e.g. vertex-color-only GLBs) are replaced by a
      default PBRMaterial TextureVisuals
    - SimpleMaterial is converted to PBRMaterial
    - every texture map is resampled to the nearest square power-of-two size
      (capped at max_size). UVs are normalized, so resampling is transparent.

    Used by the batch scheduler (extract_all.py); the verified single-sample
    flow is unaffected for assets that already satisfy the constraint.
    """
    from PIL import Image
    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import PBRMaterial, SimpleMaterial

    def pow2_square(img):
        w, h = img.size
        if w == h and w > 0 and (w & (w - 1)) == 0 and w <= max_size:
            return img
        s = max(w, h)
        size = 1 << max(0, (s - 1).bit_length())     # next power of two >= s
        size = min(size, max_size)
        return img.resize((size, size), Image.BILINEAR)

    for g in geoms:
        if not isinstance(g.visual, TextureVisuals):
            g.visual = TextureVisuals(material=PBRMaterial())
        mat = g.visual.material
        if isinstance(mat, SimpleMaterial):
            mat = mat.to_pbr()
            g.visual.material = mat
        for key in ['baseColorTexture', 'metallicRoughnessTexture',
                    'emissiveTexture', 'normalTexture']:
            tex = getattr(mat, key, None)
            if tex is not None:
                setattr(mat, key, pow2_square(tex))
    return geoms
