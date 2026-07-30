import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory
import cv2
import imageio
import imageio.v2 as iio
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

WEIGHTS = "/data5/jianxin/TRELLIS.2/weights/TRELLIS.2-4B"

# 1. Setup Environment Map (imageio reads EXR as RGB float32; opencv-headless lacks OpenEXR)
envmap = EnvMap(torch.tensor(
    iio.imread('assets/hdri/forest.exr'),
    dtype=torch.float32, device='cuda'
))

# 2. Load Pipeline (from local weights)
pipeline = Trellis2ImageTo3DPipeline.from_pretrained(WEIGHTS)
pipeline.cuda()

# 3. Load Image & Run
image = Image.open("assets/example_image/T.png")
mesh = pipeline.run(image)[0]
mesh.simplify(16777216)  # nvdiffrast limit

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
print("DONE: wrote sample.mp4 and sample.glb")
