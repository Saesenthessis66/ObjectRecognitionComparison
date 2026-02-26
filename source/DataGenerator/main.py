import bpy
import math
import os
from itertools import product
import random
import mathutils

LIGHT_ENERGY_MIN = 8
LIGHT_ENERGY_MAX = 40

LIGHT_COLOR_MIN = (0.9, 0.9, 0.9)
LIGHT_COLOR_MAX = (1.0, 1.0, 1.0)

LIGHT_SIZE_MIN = 0.05
LIGHT_SIZE_MAX = 0.3

TILT_X_MIN = -5
TILT_X_MAX = 5

TILT_Y_MIN = -5
TILT_Y_MAX = 5

BASE_DIR = os.path.dirname(bpy.data.filepath)
if BASE_DIR == "":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(BASE_DIR, "preview")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OFFSET = 0.002

ROT_MIN = -15
ROT_MAX = -10
ROT_STEP = 5

CAM_DIST_MIN = 0.40
CAM_DIST_MAX = 0.50
CAM_DIST_STEP = 0.05

CLASS_MAP = {
    "00": 0,
    "01": 1,
    "10": 2,
    "11": 3,
}

CURRENT_CLASS_ID = 0

BACKGROUND_TYPES = [
    # "LIGHT_SOLID",
    # "DARK_SOLID",
    # "NEUTRAL_COLOR",
    "HORIZONTAL_GRADIENT",
    "VERTICAL_GRADIENT",
    "NOISE_TEXTURE",
    "DIRTY_BACKGROUND"
]

def get_object_size_xy(obj):
    bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]

    xs = [v.x for v in bbox]
    ys = [v.y for v in bbox]

    size_x = max(xs) - min(xs)
    size_y = max(ys) - min(ys)

    return size_x, size_y

def setup_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def create_material(color):
    m = bpy.data.materials.new(name=str(color))
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    return m

def set_random_background(scene):
    bg_type = random.choice(BACKGROUND_TYPES)

    world = scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    nodes.clear()

    output = nodes.new(type='ShaderNodeOutputWorld')
    bg = nodes.new(type='ShaderNodeBackground')
    links.new(bg.outputs['Background'], output.inputs['Surface'])

    # --- 1. Jasne jednolite ---
    if bg_type == "LIGHT_SOLID":
        val = random.uniform(0.7, 0.85)
        bg.inputs['Color'].default_value = (val, val, val, 1)

    # --- 2. Ciemne jednolite ---
    elif bg_type == "DARK_SOLID":
        val = random.uniform(0.05, 0.3)
        bg.inputs['Color'].default_value = (val, val, val, 1)

    # --- 3. Neutralny kolor ---
    elif bg_type == "NEUTRAL_COLOR":
        r = random.uniform(0.6, 0.9)
        g = random.uniform(0.6, 0.9)
        b = random.uniform(0.5, 0.8)
        bg.inputs['Color'].default_value = (r, g, b, 1)

    # --- 4. Gradient ---
    elif bg_type in ["HORIZONTAL_GRADIENT", "VERTICAL_GRADIENT"]:
        tex_coord = nodes.new(type='ShaderNodeTexCoord')
        gradient = nodes.new(type='ShaderNodeTexGradient')
        mapping = nodes.new(type='ShaderNodeMapping')

        if bg_type == "VERTICAL_GRADIENT":
            mapping.inputs['Rotation'].default_value[2] = math.radians(90)

        links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
        links.new(mapping.outputs['Vector'], gradient.inputs['Vector'])
        links.new(gradient.outputs['Color'], bg.inputs['Color'])

    # --- 5. Lekki noise ---
    elif bg_type == "NOISE_TEXTURE":
        tex_coord = nodes.new(type='ShaderNodeTexCoord')
        noise = nodes.new(type='ShaderNodeTexNoise')

        noise.inputs['Scale'].default_value = random.uniform(2, 6)
        noise.inputs['Detail'].default_value = 2

        links.new(tex_coord.outputs['Generated'], noise.inputs['Vector'])
        links.new(noise.outputs['Color'], bg.inputs['Color'])

    # --- 6. Brudne tło ---
    elif bg_type == "DIRTY_BACKGROUND":
        tex_coord = nodes.new(type='ShaderNodeTexCoord')
        noise = nodes.new(type='ShaderNodeTexNoise')
        ramp = nodes.new(type='ShaderNodeValToRGB')

        noise.inputs['Scale'].default_value = random.uniform(5, 15)
        noise.inputs['Detail'].default_value = 5

        ramp.color_ramp.elements[0].color = (0.3, 0.3, 0.3, 1)
        ramp.color_ramp.elements[1].color = (0.7, 0.7, 0.7, 1)

        links.new(tex_coord.outputs['Generated'], noise.inputs['Vector'])
        links.new(noise.outputs['Fac'], ramp.inputs['Fac'])
        links.new(ramp.outputs['Color'], bg.inputs['Color'])

    # losowa siła tła (ważne!)
    bg.inputs['Strength'].default_value = random.uniform(0.7, 1.3)

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

def setup_camera_light(scene):
    bpy.ops.object.camera_add(location=(0, -1, 0))
    cam = bpy.context.object
    scene.camera = cam

    bpy.ops.object.light_add(type='AREA', location=(0, -1, 1))
    light = bpy.context.object

    return cam, light

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

def get_camera_view_size(scene, cam, distance):
    fov = cam.data.angle  # poziomy FOV w radianach

    aspect = scene.render.resolution_x / scene.render.resolution_y

    view_width = 2 * distance * math.tan(fov / 2)
    view_height = view_width / aspect

    return view_width, view_height

def render_sample(scene, cam, obj, rotation_z, cam_dist, filepath):

    # losowy mały tilt X/Y
    tilt_x = random.uniform(TILT_X_MIN, TILT_X_MAX)
    tilt_y = random.uniform(TILT_Y_MIN, TILT_Y_MAX)

    obj.rotation_euler[0] = math.radians(tilt_x)
    obj.rotation_euler[1] = math.radians(tilt_y)
    obj.rotation_euler[2] = math.radians(rotation_z)

    # wymuś update transformacji
    bpy.context.view_layer.update()

    view_w, view_h = get_camera_view_size(scene, cam, cam_dist)
    obj_w, obj_h = get_object_size_xy(obj)

    # ile może wyjść poza kadr (np 30%)
    overflow_ratio = 0.00

    max_offset_x = (view_w - obj_w) / 2 + obj_w * overflow_ratio
    max_offset_y = (view_h - obj_h) / 2 + obj_h * overflow_ratio

    offset_x = random.uniform(-max_offset_x, max_offset_x)
    offset_y = random.uniform(-max_offset_y, max_offset_y)

    obj.location.x += offset_x
    obj.location.z += offset_y

    cam.location = (0, -cam_dist, 0)
    cam.rotation_euler = (math.radians(90), 0, 0)
    scene.cycles.samples = random.randint(16, 64)
    scene.view_settings.exposure = random.uniform(-0.5, 0.5)
    scene.cycles.use_denoising = False

    scene.render.filepath = filepath
    bbox = get_yolo_bbox(scene, cam, obj)
    bpy.ops.render.render(write_still=True)

    if bbox is not None:
        save_yolo_label(scene.render.filepath, CURRENT_CLASS_ID, bbox)

    bpy.data.objects.remove(obj, do_unlink=True)

def generate_dataset():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512

    materials = create_materials()

    for bits in product([0, 1], repeat=2):
        seq_name = f"{bits[0]}{bits[1]}"
        global CURRENT_CLASS_ID
        CURRENT_CLASS_ID = CLASS_MAP[seq_name]
        img_dir = os.path.join(OUTPUT_DIR, "images", "train", seq_name)
        os.makedirs(img_dir, exist_ok=True)

        idx = 0
        setup_scene()
        cam, light = setup_camera_light(scene)

        rot = ROT_MIN
        while rot <= ROT_MAX:
            dist = CAM_DIST_MIN
            while dist <= CAM_DIST_MAX:
                obj = create_object(materials, bits)

                filename = f"img_{idx:05d}_rot_{rot}_dist_{dist:.2f}.png"
                filepath = os.path.join(img_dir, filename)

                light.data.energy = random.uniform(LIGHT_ENERGY_MIN, LIGHT_ENERGY_MAX)
                light.data.color = (
                    random.uniform(LIGHT_COLOR_MIN[0], LIGHT_COLOR_MAX[0]),
                    random.uniform(LIGHT_COLOR_MIN[1], LIGHT_COLOR_MAX[1]),
                    random.uniform(LIGHT_COLOR_MIN[2], LIGHT_COLOR_MAX[2]),
                )

                light.data.size = random.uniform(LIGHT_SIZE_MIN, LIGHT_SIZE_MAX)
                set_random_background(scene)

                render_sample(scene, cam, obj, rot, dist, filepath)

                idx += 1
                dist += CAM_DIST_STEP
            rot += ROT_STEP

from bpy_extras.object_utils import world_to_camera_view

def get_yolo_bbox(scene, cam, obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh()

    coords_2d = []

    for v in mesh.vertices:
        co_world = obj.matrix_world @ v.co
        co_ndc = world_to_camera_view(scene, cam, co_world)

        if 0.0 <= co_ndc.z <= 1.0:
            coords_2d.append((co_ndc.x, co_ndc.y))

    obj_eval.to_mesh_clear()

    if not coords_2d:
        return None

    xs = [c[0] for c in coords_2d]
    ys = [c[1] for c in coords_2d]

    min_x = max(min(xs), 0)
    max_x = min(max(xs), 1)
    min_y = max(min(ys), 0)
    max_y = min(max(ys), 1)

    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    w = max_x - min_x
    h = max_y - min_y

    return cx, cy, w, h

def save_yolo_label(filepath, class_id, bbox):
    label_path = filepath.replace("images", "labels").replace(".png", ".txt")
    os.makedirs(os.path.dirname(label_path), exist_ok=True)

    with open(label_path, "w") as f:
        f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}")


import shutil

def split_dataset(train_ratio=0.7, val_ratio=0.2):
    base_images = os.path.join(OUTPUT_DIR, "images", "train")
    base_labels = os.path.join(OUTPUT_DIR, "labels", "train")

    classes = os.listdir(base_images)

    for cls in classes:
        cls_img_dir = os.path.join(base_images, cls)
        cls_lbl_dir = os.path.join(base_labels, cls)

        images = [
            os.path.join(cls_img_dir, f)
            for f in os.listdir(cls_img_dir)
            if f.endswith(".png")
        ]

        random.shuffle(images)

        total = len(images)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split_name, split_files in splits.items():
            for img_path in split_files:
                filename = os.path.basename(img_path)
                label_path = os.path.join(cls_lbl_dir, filename.replace(".png", ".txt"))

                new_img = os.path.join(
                    OUTPUT_DIR, "images", split_name, cls, filename
                )
                new_lbl = os.path.join(
                    OUTPUT_DIR, "labels", split_name, cls, filename.replace(".png", ".txt")
                )

                os.makedirs(os.path.dirname(new_img), exist_ok=True)
                os.makedirs(os.path.dirname(new_lbl), exist_ok=True)

                if os.path.exists(img_path):
                    shutil.move(img_path, new_img)

                if os.path.exists(label_path):
                    shutil.move(label_path, new_lbl)

def create_data_yaml():
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"""
path: {OUTPUT_DIR}
train: images/train
val: images/val
test: images/test

names:
  0: "00"
  1: "01"
  2: "10"
  3: "11"
""")


generate_dataset()
split_dataset()
create_data_yaml()