import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory
import cv2
import imageio
import imageio.v3 as iio
import numpy as np
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

# 1. Setup Environment Map
# NOTE: cv2 5.0 cannot decode EXR (returns empty array even with OPENCV_IO_ENABLE_OPENEXR=1),
# so load the HDRI via imageio (RGB float32).
envmap = EnvMap(torch.tensor(
    np.ascontiguousarray(iio.imread('assets/hdri/forest.exr')[..., :3]),
    dtype=torch.float32, device='cuda'
))

# 2. Load Pipeline
# Offline mode: TRELLIS_CKPT_ROOT points to the local weights root (contains TRELLIS.2-4B/, BiRefNet/, ...)
_ckpt_root = os.environ.get('TRELLIS_CKPT_ROOT', '/data5/jianxin/ckpt')
os.environ.setdefault('TRELLIS_CKPT_ROOT', _ckpt_root)   # propagate to remap_ckpt_path in pipelines/base.py
_pipeline_path = f'{_ckpt_root}/TRELLIS.2-4B' if os.path.isdir(f'{_ckpt_root}/TRELLIS.2-4B') else "microsoft/TRELLIS.2-4B"
pipeline = Trellis2ImageTo3DPipeline.from_pretrained(_pipeline_path)
pipeline.cuda()

# 3. Load Image & Run
image = Image.open("assets/example_image/T.png")
mesh = pipeline.run(image)[0]
mesh.simplify(16777216) # nvdiffrast limit

# 4. Render Video
video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
imageio.mimsave("sample.mp4", video, fps=15)

# 5. Export to GLB
glb = o_voxel.postprocess.to_glb(
    vertices            =   mesh.vertices,
    faces               =   mesh.faces,
    attr_volume         =   mesh.attrs,
    coords              =   mesh.coords,
    attr_layout         =   mesh.layout,
    voxel_size          =   mesh.voxel_size,
    aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target   =   1000000,
    texture_size        =   4096,
    remesh              =   True,
    remesh_band         =   1,
    remesh_project      =   0,
    verbose             =   True
)
glb.export("sample.glb", extension_webp=True)