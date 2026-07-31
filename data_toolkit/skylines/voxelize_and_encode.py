"""
T0 step 2 (core): GLB -> flow-matching supervision targets.

For each instance:
    1. load GLB with trimesh, normalize to [-0.5, 0.5]^3 (y-up kept)
    2. dual-grid voxelization at 512 (params copied from data_toolkit/dual_grid.py)
       -> frozen shape_enc -> z*_shape [Mg, 32] with coords at 32^3 (512 / f16)
    3. 64^3 occupancy = unique(voxel_indices_512 // 8)
       -> frozen ss_enc (sample_posterior=False) -> z*_ss [8, 16, 16, 16]
    4. PBR voxelization at 512 via o_voxel.convert.textured_mesh_to_volumetric_attr
       (attrs order base_color/metallic/roughness/alpha, cf.
        configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json)
       -> frozen tex_enc -> z*_tex [Mg, 32]
    5. hard checks: tex coords == shape coords element-wise,
       unique(voxel512 // 16) == shape coords, no NaN anywhere.

Consistency note: coords of the f16 encoders live on a 32^3 grid
(trellis2_image_to_3d.py L541: ss_res['512'] == 32; at inference the 64^3
ss decoding is max-pooled by 2). Saving ss occupancy as unique(voxel512 // 8)
guarantees unique(ss64 // 2) == unique(voxel512 // 16) == latent coords.

Usage:
    python voxelize_and_encode.py --instances <sha256>[,<sha256>...]
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import argparse
import numpy as np
import torch
import trimesh
import o_voxel

import trellis2.models as models
import trellis2.modules.sparse as sp
from data_toolkit.skylines import common

torch.set_grad_enabled(False)

PBR_ATTRS = ['base_color', 'metallic', 'roughness', 'alpha']  # order fixed by tex flow config


def merge_geoms(geoms):
    vertices, faces, start = [], [], 0
    for g in geoms:
        vertices.append(np.asarray(g.vertices, dtype=np.float32))
        faces.append(np.asarray(g.faces, dtype=np.int64) + start)
        start += len(g.vertices)
    vertices = torch.from_numpy(np.concatenate(vertices, axis=0)).float()
    faces = torch.from_numpy(np.concatenate(faces, axis=0)).long()
    return vertices, faces


def sort_by_seq(voxel_indices, *tensors):
    vid = o_voxel.serialize.encode_seq(voxel_indices)
    mapping = torch.argsort(vid)
    return (voxel_indices[mapping],) + tuple(t[mapping] for t in tensors)


def process_one(sha256, glb_path, shape_enc, ss_enc, tex_enc):
    geoms, norm = common.load_normalized_geoms(glb_path)
    vertices, faces = merge_geoms(geoms)
    assert torch.all(vertices >= -0.5) and torch.all(vertices <= 0.5), 'vertices out of range'
    res = common.DUAL_GRID_RES
    aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]

    # ---- geometry: flexible dual grid @ 512 (params from dual_grid.py) ----
    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices, faces,
        grid_size=res,
        aabb=aabb,
        face_weight=1.0,
        boundary_weight=0.2,
        regularization_weight=1e-2,
        timing=False,
    )
    voxel_indices, dual_vertices, intersected = sort_by_seq(voxel_indices, dual_vertices, intersected)
    print(f'[voxelize] {sha256}: dual grid voxels = {len(voxel_indices)}')

    # uint8 quantization roundtrip, identical to dual_grid.py storage format
    dual_vertices = dual_vertices * res - voxel_indices
    assert torch.all(dual_vertices >= -1e-3) and torch.all(dual_vertices <= 1 + 1e-3), 'dual_vertices out of range'
    dual_vertices = torch.clamp(dual_vertices, 0, 1)
    dual_vertices = (dual_vertices * 255).type(torch.uint8)
    intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)

    # rebuild sparse input exactly like encode_shape_latent.py loader
    coords512 = voxel_indices.int()
    dv = sp.SparseTensor(
        (dual_vertices / 255.0).float(),
        torch.cat([torch.zeros_like(coords512[:, 0:1]), coords512], dim=-1),
    )
    its = dv.replace(torch.cat([
        intersected % 2,
        intersected // 2 % 2,
        intersected // 4 % 2,
    ], dim=-1).bool())

    z_shape = shape_enc(dv.cuda(), its.cuda())
    torch.cuda.synchronize()
    assert torch.isfinite(z_shape.feats).all(), 'NaN/Inf in shape latent'
    # encoder output order is backend-dependent; sort by encode_seq so that
    # shape/tex latents share one deterministic coords ordering
    shape_coords, shape_feats = sort_by_seq(z_shape.coords[:, 1:].cpu().int(), z_shape.feats.cpu().float())
    shape_feats = shape_feats.numpy().astype(np.float32)
    Mg = shape_coords.shape[0]
    assert shape_coords.max() < common.COARSE_RES, \
        f'shape coords out of {common.COARSE_RES}^3: max={shape_coords.max()}'
    print(f'[voxelize] {sha256}: shape latent Mg = {Mg}, feats {shape_feats.shape}')

    # consistency: latent coords must equal unique(voxel512 // 16)
    expected = torch.unique(coords512 // (res // common.COARSE_RES), dim=0)
    expected, = sort_by_seq(expected)
    assert shape_coords.shape == expected.shape and torch.equal(shape_coords, expected.int()), \
        'shape latent coords != unique(voxel512 // 16)'

    # ---- sparse structure: 64^3 occupancy -> ss_enc ----
    ss64 = torch.unique(coords512 // (res // common.SS_RES), dim=0)
    ss = torch.zeros(1, common.SS_RES, common.SS_RES, common.SS_RES)
    ss[:, ss64[:, 0].long(), ss64[:, 1].long(), ss64[:, 2].long()] = 1
    # guarantee: maxpool2(ss64) == latent coords
    ss32 = torch.unique(ss64 // 2, dim=0)
    ss32, = sort_by_seq(ss32)
    assert torch.equal(ss32.int(), shape_coords), 'unique(ss64 // 2) != shape latent coords'

    z_ss = ss_enc(ss.cuda()[None].float(), sample_posterior=False)
    torch.cuda.synchronize()
    assert torch.isfinite(z_ss).all(), 'NaN/Inf in ss latent'
    z_ss = z_ss[0].cpu().numpy()
    assert z_ss.shape == (8, 16, 16, 16), f'unexpected ss latent shape {z_ss.shape}'
    print(f'[voxelize] {sha256}: ss64 occupancy = {len(ss64)} voxels, z_ss {z_ss.shape}')

    # ---- texture: PBR voxelization @ 512 -> tex_enc ----
    scene = trimesh.Scene(geoms)
    tex_coords, attr = o_voxel.convert.textured_mesh_to_volumetric_attr(
        scene,
        grid_size=res,
        aabb=aabb,
        timing=False,
    )
    order = torch.argsort(o_voxel.serialize.encode_seq(tex_coords))
    tex_coords = tex_coords[order]
    attr = {k: v[order] for k, v in attr.items()}
    assert tex_coords.shape == coords512.shape and torch.equal(tex_coords.int(), coords512), \
        f'PBR voxel coords != dual grid coords ({tex_coords.shape[0]} vs {coords512.shape[0]})'

    # feats layout copied from encode_pbr_latent.py: concat / 255 * 2 - 1
    pbr_feats = torch.cat([attr[k] for k in PBR_ATTRS], dim=-1).float() / 255.0 * 2 - 1
    voxels = sp.SparseTensor(
        pbr_feats,
        torch.cat([torch.zeros_like(coords512[:, 0:1]), coords512], dim=-1),
    )
    z_tex = tex_enc(voxels.cuda())
    torch.cuda.synchronize()
    assert torch.isfinite(z_tex.feats).all(), 'NaN/Inf in tex latent'
    tex_out_coords, tex_feats = sort_by_seq(z_tex.coords[:, 1:].cpu().int(), z_tex.feats.cpu().float())
    tex_feats = tex_feats.numpy().astype(np.float32)
    assert tex_out_coords.shape == shape_coords.shape and torch.equal(tex_out_coords, shape_coords), \
        'tex latent coords != shape latent coords'
    print(f'[voxelize] {sha256}: tex latent Mg = {tex_out_coords.shape[0]}, feats {tex_feats.shape}')

    # ---- save ----
    coords_u8 = shape_coords.numpy().astype(np.uint8)
    for path in [common.shape_latent_path(sha256), common.tex_latent_path(sha256),
                 common.ss_latent_path(sha256), common.aux_path(sha256)]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(common.shape_latent_path(sha256), feats=shape_feats, coords=coords_u8)
    np.savez_compressed(common.tex_latent_path(sha256), feats=tex_feats, coords=coords_u8)
    np.savez_compressed(common.ss_latent_path(sha256), z=z_ss)
    ss_occ = np.zeros((common.SS_RES,) * 3, dtype=np.uint8)
    ss_occ[ss64[:, 0], ss64[:, 1], ss64[:, 2]] = 1
    np.savez_compressed(
        common.aux_path(sha256),
        ss_occupancy=ss_occ,
        norm_center=norm['center'],
        norm_scale=norm['scale'],
        up_axis=norm['up_axis'],
        num_voxels_512=len(coords512),
    )
    common.update_metadata(
        sha256,
        ss_latent_encoded=True,
        shape_latent_encoded=True,
        shape_latent_tokens=Mg,
        tex_latent_encoded=True,
        tex_latent_tokens=Mg,
    )
    print(f'[voxelize] {sha256}: OK (Mg={Mg}, ss64={len(ss64)}, voxels512={len(coords512)})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instances', type=str, required=True,
                        help='Comma separated sha256 list, or a file with one sha256 per line')
    opt = parser.parse_args()

    instances = common.parse_instances(opt.instances)
    metadata = common.load_metadata()

    shape_enc = models.from_pretrained(common.SHAPE_ENC_PRETRAINED).eval().cuda()
    ss_enc = models.from_pretrained(common.SS_ENC_PRETRAINED).eval().cuda()
    tex_enc = models.from_pretrained(common.TEX_ENC_PRETRAINED).eval().cuda()

    for sha256 in instances:
        assert sha256 in metadata.index, f'{sha256} not in metadata.csv; run build_metadata.py first'
        process_one(sha256, metadata.loc[sha256, 'glb_path'], shape_enc, ss_enc, tex_enc)
    print('[voxelize] all done')
