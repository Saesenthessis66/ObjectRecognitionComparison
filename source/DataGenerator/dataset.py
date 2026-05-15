import bpy
import os
import math
import random
import shutil
from itertools import product
from collections import defaultdict

from bpy_extras.object_utils import world_to_camera_view

from constants import *
from setup import (
    setup_scene,
    configure_render_engine,
    setup_camera_light,
    create_materials,
    set_random_background,
    create_object,
    get_object_size_xy,
    get_camera_view_size,
)

def export_to_dae(obj, path):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Ustaw origin (ważne dla Gazebo)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

    # Apply transformacje
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    bpy.ops.wm.collada_export(
        filepath=path,
        apply_modifiers=True,
        selected=True
    )


def export_all_markers(materials):
    dae_dir = os.path.join(OUTPUT_DIR, "dae")
    os.makedirs(dae_dir, exist_ok=True)

    for bits in product([0, 1], repeat=2):
        setup_scene()
        obj = create_object(materials, bits)

        name = f"{bits[0]}{bits[1]}"
        path = os.path.join(dae_dir, f"marker_{name}.dae")

        export_to_dae(obj, path)

# Render single dataset sample with random rotation, tilt, lighting and camera distance
def render_sample(scene, cam, obj, rotation_z, cam_dist, filepath, class_id, light):

    cam.location = (0, -cam_dist, 0)
    cam.rotation_euler = (math.radians(90), 0, 0)

    scene.cycles.samples = random.randint(16, 64)
    scene.view_settings.exposure = random.uniform(-0.5, 0.5)
    scene.cycles.use_denoising = False

    scene.render.filepath = filepath

    max_tries = 20
    bbox = None

    for attempt in range(max_tries):

        obj.location = (0, 0, 0)

        # rotation
        tilt_x = random.uniform(TILT_X_MIN, TILT_X_MAX)
        tilt_y = random.uniform(TILT_Y_MIN, TILT_Y_MAX)

        obj.rotation_euler = (
            math.radians(tilt_x),
            math.radians(tilt_y),
            math.radians(rotation_z),
        )

        # progressively shrink translation range
        shrink = 1.0 - (attempt / max_tries)
        max_offset = 0.15 * cam_dist * shrink

        obj.location.x = random.uniform(-max_offset, max_offset)
        obj.location.z = random.uniform(-max_offset, max_offset)

        bpy.context.view_layer.update()

        bbox_candidate = get_yolo_bbox(scene, cam, obj)

        if bbox_candidate is None:
            continue

        cx, cy, w, h = bbox_candidate

        if (
            (cx - w / 2) > 0 and
            (cx + w / 2) < 1 and
            (cy - h / 2) > 0 and
            (cy + h / 2) < 1
        ):
            bbox = bbox_candidate
            break

    if bbox is None:
        return False

    bpy.ops.render.render(write_still=True)
    save_yolo_label(scene.render.filepath, class_id, bbox)

    return True



# Compute YOLO bounding box by projecting mesh vertices to camera view
def get_yolo_bbox(scene, cam, obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)

    # Evaluated mesh contains modifiers and final geometry
    mesh = obj_eval.to_mesh()

    coords_2d = []

    for v in mesh.vertices:

        # Convert vertex to world coordinates
        co_world = obj.matrix_world @ v.co

        # Convert world coordinate to normalized camera space (0-1)
        co_ndc = world_to_camera_view(scene, cam, co_world)

        # Only keep vertices inside camera depth range
        if 0.0 <= co_ndc.z <= 1.0:
            coords_2d.append((co_ndc.x, 1 - co_ndc.y))

    obj_eval.to_mesh_clear()

    if not coords_2d:
        return None

    xs = [c[0] for c in coords_2d]
    ys = [c[1] for c in coords_2d]

    # Clamp bbox to image boundaries
    min_x = max(min(xs), 0)
    max_x = min(max(xs), 1)
    min_y = max(min(ys), 0)
    max_y = min(max(ys), 1)

    # Convert to YOLO format (center_x, center_y, width, height)
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    w = max_x - min_x
    h = max_y - min_y

    return cx, cy, w, h


# Save YOLO label file corresponding to rendered image
def save_yolo_label(filepath, class_id, bbox):
    label_path = filepath.replace("images", "labels").replace(".png", ".txt")
    os.makedirs(os.path.dirname(label_path), exist_ok=True)

    with open(label_path, "w") as f:
        f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}")


# Main dataset generation loop
def generate_dataset():
    scene = bpy.context.scene

    configure_render_engine(scene)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    materials = create_materials()
    for m in materials.values():
        m.use_fake_user = True

    export_all_markers(materials)

    # Generate all bit combinations (00, 01, 10, 11)
    for bits in product([0, 1], repeat=2):
        seq_name = f"{bits[0]}{bits[1]}"
        class_id = CLASS_MAP[seq_name]

        img_dir = os.path.join(OUTPUT_DIR, "images", "train")
        os.makedirs(img_dir, exist_ok=True)

        idx = 0

        setup_scene()
        cam, light = setup_camera_light(scene)

        rot = ROT_MIN

        # Create object representing current class
        obj = create_object(materials, bits)

        # Iterate through rotation values
        while rot <= ROT_MAX:

            dist = CAM_DIST_MIN

            # Iterate through camera distances
            while dist <= CAM_DIST_MAX:

                obj.location = (0,0,0)
                obj.rotation_euler = (0,0,0)

                filename = f"img_{seq_name}_{idx:05d}_rot_{rot}_dist_{dist:.2f}.png"
                filepath = os.path.join(img_dir, filename)

                # Randomize light energy
                light.data.energy = random.uniform(LIGHT_ENERGY_MIN, LIGHT_ENERGY_MAX)

                # Randomize light color
                light.data.color = (
                    random.uniform(LIGHT_COLOR_MIN[0], LIGHT_COLOR_MAX[0]),
                    random.uniform(LIGHT_COLOR_MIN[1], LIGHT_COLOR_MAX[1]),
                    random.uniform(LIGHT_COLOR_MIN[2], LIGHT_COLOR_MAX[2]),
                )

                # Randomize light size (soft shadows)
                light.data.size = random.uniform(LIGHT_SIZE_MIN, LIGHT_SIZE_MAX)

                if idx % 20 == 0:
                    set_random_background(scene)

                render_sample(
                    scene,
                    cam,
                    obj,
                    rot,
                    dist,
                    filepath,
                    class_id,
                    light,
                )

                idx += 1

                if idx % 50 == 0:
                    bpy.ops.outliner.orphans_purge(do_recursive=True)

                dist += CAM_DIST_STEP

            rot += ROT_STEP

        # Remove object to free memory before next class
        bpy.data.objects.remove(obj, do_unlink=True)


# Split dataset into train/val/test keeping class balance
def split_dataset(train_ratio=0.7, val_ratio=0.2):
    images_root = os.path.join(OUTPUT_DIR, "images")
    labels_root = os.path.join(OUTPUT_DIR, "labels")

    train_dir = os.path.join(images_root, "train")
    val_dir = os.path.join(images_root, "val")
    test_dir = os.path.join(images_root, "test")

    label_train_dir = os.path.join(labels_root, "train")
    label_val_dir = os.path.join(labels_root, "val")
    label_test_dir = os.path.join(labels_root, "test")

    for d in [val_dir, test_dir, label_val_dir, label_test_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    class_files = defaultdict(list)

    # Group images by class
    for filename in os.listdir(train_dir):

        if not filename.endswith(".png"):
            continue

        parts = filename.split("_")

        if len(parts) < 3:
            continue

        class_name = parts[1]
        class_files[class_name].append(filename)

    for class_name, files in class_files.items():

        random.shuffle(files)

        total = len(files)

        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        test_count = total - train_count - val_count

        train_files = files[:train_count]
        val_files = files[train_count:train_count + val_count]
        test_files = files[train_count + val_count:]

        for split_name, split_files, img_dst, lbl_dst in [

            ("train", train_files, train_dir, label_train_dir),
            ("val", val_files, val_dir, label_val_dir),
            ("test", test_files, test_dir, label_test_dir),

        ]:

            os.makedirs(img_dst, exist_ok=True)
            os.makedirs(lbl_dst, exist_ok=True)

            for file in split_files:

                src_img = os.path.join(train_dir, file)
                dst_img = os.path.join(img_dst, file)

                label_file = file.replace(".png", ".txt")

                src_lbl = os.path.join(label_train_dir, label_file)
                dst_lbl = os.path.join(lbl_dst, label_file)

                if split_name != "train":

                    shutil.move(src_img, dst_img)

                    if os.path.exists(src_lbl):
                        shutil.move(src_lbl, dst_lbl)

    print("Dataset successfully split per class.")


# Create YOLO dataset configuration file
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

def generate_eval_datasets():
    scene = bpy.context.scene
    configure_render_engine(scene)

    base_dir = os.path.join(OUTPUT_DIR, "eval")
    dist_dir = os.path.join(base_dir, "distance")
    rot_dir = os.path.join(base_dir, "rotation")

    os.makedirs(dist_dir, exist_ok=True)
    os.makedirs(rot_dir, exist_ok=True)

    materials = create_materials()

    # =========================
    # DISTANCE DATASET
    # =========================
    for bits in product([0, 1], repeat=2):
        seq_name = f"{bits[0]}{bits[1]}"
        class_id = CLASS_MAP[seq_name]

        setup_scene()
        cam, light = setup_camera_light(scene)

        obj = create_object(materials, bits)

        # fixed conditions
        light.data.energy = EVAL_LIGHT_ENERGY
        light.data.color = EVAL_LIGHT_COLOR
        light.data.size = EVAL_LIGHT_SIZE
        scene.world.use_nodes = False

        dist = EVAL_DIST_MIN
        idx = 0

        while dist <= EVAL_DIST_MAX:

            obj.location = (0, 0, 0)
            obj.rotation_euler = (
                math.radians(EVAL_TILT_X),
                math.radians(EVAL_TILT_Y),
                math.radians(EVAL_DIST_ROT),
            )

            cam.location = (0, -dist, 0)
            cam.rotation_euler = (math.radians(90), 0, 0)

            filename = f"{seq_name}_dist_{dist:.2f}_{idx:04d}.png"
            filepath = os.path.join(dist_dir, filename)

            bbox = get_yolo_bbox(scene, cam, obj)

            scene.render.filepath = filepath
            bpy.ops.render.render(write_still=True)

            if bbox:
                save_yolo_label(filepath, class_id, bbox)

            dist += EVAL_DIST_STEP
            idx += 1

        bpy.data.objects.remove(obj, do_unlink=True)

    # =========================
    # ROTATION DATASET
    # =========================
    for bits in product([0, 1], repeat=2):
        seq_name = f"{bits[0]}{bits[1]}"
        class_id = CLASS_MAP[seq_name]

        setup_scene()
        cam, light = setup_camera_light(scene)

        obj = create_object(materials, bits)

        # fixed conditions
        light.data.energy = EVAL_LIGHT_ENERGY
        light.data.color = EVAL_LIGHT_COLOR
        light.data.size = EVAL_LIGHT_SIZE
        scene.world.use_nodes = False

        rot = EVAL_ROT_MIN
        idx = 0

        while rot <= EVAL_ROT_MAX:

            obj.location = (0, 0, 0)
            obj.rotation_euler = (
                math.radians(EVAL_TILT_X),
                math.radians(EVAL_TILT_Y),
                math.radians(rot),
            )

            cam.location = (0, -EVAL_ROT_DIST, 0)
            cam.rotation_euler = (math.radians(90), 0, 0)

            filename = f"{seq_name}_rot_{rot}_{idx:04d}.png"
            filepath = os.path.join(rot_dir, filename)

            bbox = get_yolo_bbox(scene, cam, obj)

            scene.render.filepath = filepath
            bpy.ops.render.render(write_still=True)

            if bbox:
                save_yolo_label(filepath, class_id, bbox)

            rot += EVAL_ROT_STEP
            idx += 1

        bpy.data.objects.remove(obj, do_unlink=True)

    print("Evaluation datasets generated.")

def balance_dataset():
    images_dir = os.path.join(OUTPUT_DIR, "images", "train")
    labels_dir = os.path.join(OUTPUT_DIR, "labels", "train")

    balanced_img_dir = os.path.join(OUTPUT_DIR, "images", "train_balanced")
    balanced_lbl_dir = os.path.join(OUTPUT_DIR, "labels", "train_balanced")

    if os.path.exists(balanced_img_dir):
        shutil.rmtree(balanced_img_dir)
    if os.path.exists(balanced_lbl_dir):
        shutil.rmtree(balanced_lbl_dir)

    os.makedirs(balanced_img_dir, exist_ok=True)
    os.makedirs(balanced_lbl_dir, exist_ok=True)

    class_files = defaultdict(list)

    # Group images by class (same logic as split_dataset)
    for filename in os.listdir(images_dir):

        if not filename.endswith(".png"):
            continue

        parts = filename.split("_")

        if len(parts) < 3:
            continue

        class_name = parts[1]
        class_files[class_name].append(filename)

    # Find smallest class
    min_count = min(len(files) for files in class_files.values())

    print(f"[INFO] Balancing dataset to {min_count} samples per class")

    for class_name, files in class_files.items():

        random.shuffle(files)

        selected_files = files[:min_count]

        for file in selected_files:

            src_img = os.path.join(images_dir, file)
            dst_img = os.path.join(balanced_img_dir, file)

            label_file = file.replace(".png", ".txt")

            src_lbl = os.path.join(labels_dir, label_file)
            dst_lbl = os.path.join(balanced_lbl_dir, label_file)

            shutil.copy2(src_img, dst_img)

            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, dst_lbl)

    # Replace original train dirs
    shutil.rmtree(images_dir)
    shutil.rmtree(labels_dir)

    os.rename(balanced_img_dir, images_dir)
    os.rename(balanced_lbl_dir, labels_dir)

    print("Dataset balanced to smallest class.")