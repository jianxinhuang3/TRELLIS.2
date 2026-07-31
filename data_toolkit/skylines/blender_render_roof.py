"""
Blender-side script: top-down (satellite style) RGBA renders of a GLB.

Adapted from data_toolkit/blender_script/render_cond.py, trimmed to the glTF
import path and made compatible with Blender 4.x. Views are passed as a JSON
list of {yaw, pitch, radius, fov} (radians); pitch is measured from the
horizontal plane, so pitch=pi/2 is straight top-down (Blender is z-up after
glTF import).

Run inside Blender:
    blender -b -P blender_render_roof.py -- --object x.glb --views '[...]' \
        --output_folder out/ --resolution 512
"""
import argparse
import sys
import os
import math
import json
from typing import *
import bpy
from mathutils import Vector
import numpy as np


def init_render(engine='CYCLES', resolution=512):
    bpy.context.scene.render.engine = engine
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.film_transparent = True

    if engine == 'CYCLES':
        bpy.context.scene.cycles.samples = 32
        bpy.context.scene.cycles.filter_type = 'BOX'
        bpy.context.scene.cycles.filter_width = 1
        bpy.context.scene.cycles.diffuse_bounces = 1
        bpy.context.scene.cycles.glossy_bounces = 1
        bpy.context.scene.cycles.transparent_max_bounces = 3
        bpy.context.scene.cycles.transmission_bounces = 3
        bpy.context.scene.cycles.use_denoising = True
        try:
            bpy.context.preferences.addons['cycles'].preferences.get_devices()
            bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
            bpy.context.scene.cycles.device = 'GPU'
        except Exception as e:
            print(f'[WARN] CUDA devices unavailable, falling back to CPU: {e}')
            bpy.context.scene.cycles.device = 'CPU'


def init_scene():
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def init_camera():
    cam = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.sensor_height = cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(cam_empty)
    cam_constraint.target = cam_empty
    return cam


def init_uniform_lighting():
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()
    if bpy.context.scene.world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    else:
        world = bpy.context.scene.world
    world.use_nodes = True
    node_tree = world.node_tree
    nodes = node_tree.nodes
    links = node_tree.links
    for node in nodes:
        nodes.remove(node)
    bg_node = nodes.new(type="ShaderNodeBackground")
    bg_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg_node.inputs["Strength"].default_value = 1.0
    output_node = nodes.new(type="ShaderNodeOutputWorld")
    links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])


def scene_bbox():
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, bpy.types.Mesh):
            found = True
            for coord in obj.bound_box:
                coord = obj.matrix_world @ Vector(coord)
                bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
                bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def normalize_scene():
    scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    if len(scene_root_objects) > 1:
        scene = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(scene)
        for obj in scene_root_objects:
            obj.parent = scene
    else:
        scene = scene_root_objects[0]
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    scene.scale = scene.scale * scale
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    scene.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    return scale, offset


def get_transform_matrix(obj):
    pos, rt, _ = obj.matrix_world.decompose()
    rt = rt.to_matrix()
    matrix = []
    for ii in range(3):
        a = []
        for jj in range(3):
            a.append(rt[ii][jj])
        a.append(pos[ii])
        matrix.append(a)
    matrix.append([0, 0, 0, 1])
    return matrix


def main(arg):
    init_scene()
    bpy.ops.import_scene.gltf(filepath=arg.object, merge_vertices=True, import_shading='NORMALS')
    print('[INFO] Scene initialized.')

    scale, offset = normalize_scene()
    print('[INFO] Scene normalized.')

    cam = init_camera()
    init_uniform_lighting()
    print('[INFO] Camera and lighting initialized.')

    init_render(engine=arg.engine, resolution=arg.resolution)
    to_export = {
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "scale": scale,
        "offset": [offset.x, offset.y, offset.z],
        "frames": []
    }
    views = json.loads(arg.views)
    for i, view in enumerate(views):
        cam_dir = np.array([
            np.cos(view['yaw']) * np.cos(view['pitch']),
            np.sin(view['yaw']) * np.cos(view['pitch']),
            np.sin(view['pitch'])
        ])
        cam.location = tuple(view['radius'] * cam_dir)
        cam.data.lens = 16 / np.tan(view['fov'] / 2)

        bpy.context.scene.render.filepath = os.path.join(arg.output_folder, f'{i:03d}.png')
        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()

        to_export["frames"].append({
            "file_path": f'{i:03d}.png',
            "camera_angle_x": view['fov'],
            "transform_matrix": get_transform_matrix(cam)
        })

    with open(os.path.join(arg.output_folder, 'transforms.json'), 'w') as f:
        json.dump(to_export, f, indent=4)
    print('[INFO] Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', type=str, required=True)
    parser.add_argument('--views', type=str, required=True,
                        help='JSON list of {yaw, pitch, radius, fov} in radians')
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--engine', type=str, default='CYCLES')
    argv = sys.argv[sys.argv.index("--") + 1:]
    main(parser.parse_args(argv))
