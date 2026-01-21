import bpy
import math
import os
from itertools import product

# =====================
# PATHS & CONSTANTS
# =====================
BASE_DIR = os.path.dirname(bpy.data.filepath)
if BASE_DIR == "":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(BASE_DIR, "preview")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OFFSET = 0.002

ROT_MIN = -15
ROT_MAX = -10
ROT_STEP = 5

CAM_DIST_MIN = 0.30
CAM_DIST_MAX = 0.35
CAM_DIST_STEP = 0.05

# =====================
# SCENE CLEANUP
# =====================
def setup_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

# =====================
# MATERIALS
# =====================
def create_material(color):
    m = bpy.data.materials.new(name=str(color))
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    return m


def create_transparent_material():
    m = bpy.data.materials.new(name="TRANSPARENT")
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Alpha"].default_value = 0.0
    m.blend_method = 'BLEND'
    return m


def create_materials():
    return {
        "GREEN": create_material((0, 1, 0)),
        "WHITE": create_material((1, 1, 1)),
        "BLACK": create_material((0, 0, 0)),
        "TRANSPARENT": create_transparent_material(),
    }

# =====================
# CAMERA & LIGHT
# =====================
def setup_camera_light(scene):
    bpy.ops.object.camera_add(location=(0, -1, 0))
    cam = bpy.context.object
    scene.camera = cam

    bpy.ops.object.light_add(type='AREA', location=(0, -1, 1))
    return cam

# =====================
# OBJECT CREATION
# =====================
def create_object(materials, bits):
    obj_list = []

    # bits = (b0, b1)
    bit_material = {
        0: materials["TRANSPARENT"],
        1: materials["WHITE"]
    }

    pos = []
    pos.append(OFFSET / 2 + 0.05 / 2)
    pos.append(pos[0] + 0.05 / 2 + 0.07 / 2 + OFFSET)

    def add_cube(location, scale, material):
        bpy.ops.mesh.primitive_cube_add(size=1)
        o = bpy.context.object
        o.location = location
        o.scale = scale
        o.data.materials.append(material)
        obj_list.append(o)

    # right side (bit 0 + end)
    add_cube(( pos[0], 0, 0), (0.05, 0.05, 0.05), bit_material[bits[0]])
    add_cube(( pos[1], 0, 0), (0.07, 0.05, 0.05), materials["GREEN"])

    # left side (bit 1 + start)
    add_cube((-pos[0], 0, 0), (0.05, 0.05, 0.05), bit_material[bits[1]])
    add_cube((-pos[1], 0, 0), (0.07, 0.05, 0.05), materials["GREEN"])

    bpy.ops.object.select_all(action='DESELECT')
    for o in obj_list:
        o.select_set(True)
    bpy.context.view_layer.objects.active = obj_list[0]
    bpy.ops.object.join()

    return bpy.context.object

# =====================
# RENDER SINGLE SAMPLE
# =====================
def render_sample(scene, cam, obj, rotation_z, cam_dist, filepath):
    obj.rotation_euler[2] = math.radians(rotation_z)

    cam.location = (0, -cam_dist, 0)
    cam.rotation_euler = (math.radians(90), 0, 0)

    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)

    bpy.data.objects.remove(obj, do_unlink=True)

# =====================
# DATASET GENERATION
# =====================
def generate_dataset():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512

    materials = create_materials()

    for bits in product([0, 1], repeat=2):
        seq_name = f"{bits[0]}{bits[1]}"
        img_dir = os.path.join(OUTPUT_DIR, "images", "train", seq_name)
        os.makedirs(img_dir, exist_ok=True)

        idx = 0
        setup_scene()
        cam = setup_camera_light(scene)

        rot = ROT_MIN
        while rot <= ROT_MAX:
            dist = CAM_DIST_MIN
            while dist <= CAM_DIST_MAX:
                obj = create_object(materials, bits)

                filename = f"img_{idx:05d}_rot_{rot}_dist_{dist:.2f}.png"
                filepath = os.path.join(img_dir, filename)

                render_sample(scene, cam, obj, rot, dist, filepath)

                idx += 1
                dist += CAM_DIST_STEP
            rot += ROT_STEP

# =====================
# RUN
# =====================
generate_dataset()
