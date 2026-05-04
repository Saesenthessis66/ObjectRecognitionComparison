from ultralytics import YOLO
import os
import shutil
from pathlib import Path

SRC_DATASET = "/workspace/preview"
POSE_DATASET = "/workspace/preview_pose"

DATA_YAML_POSE = os.path.join(POSE_DATASET, "data.yaml")

IMG_SIZE = 320
EPOCHS = 80
BATCH_SIZE = 64

PROJECT_NAME = "blocks_pose"
RUN_NAME = "yolov8n_pose"


def copy_dataset():
    if os.path.exists(POSE_DATASET):
        shutil.rmtree(POSE_DATASET)

    shutil.copytree(SRC_DATASET, POSE_DATASET)
    print("Dataset copied to pose version.")


def bbox_to_pose(labels_dir):
    labels_path = Path(labels_dir)

    for file in labels_path.glob("*.txt"):
        with open(file, "r") as f:
            parts = f.readline().strip().split()

        if len(parts) < 5:
            continue

        class_id = parts[0]
        cx, cy, w, h = map(float, parts[1:5])

        # corners from bbox
        x1 = cx - w / 2
        y1 = cy - h / 2

        x2 = cx + w / 2
        y2 = cy - h / 2

        x3 = cx + w / 2
        y3 = cy + h / 2

        x4 = cx - w / 2
        y4 = cy + h / 2

        pose_annotation = f"{class_id} {cx} {cy} {w} {h}"

        for x, y in [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]:
            pose_annotation += f" {x} {y} 2"

        with open(file, "w") as f:
            f.write(pose_annotation)

    print(f"Converted labels in: {labels_dir}")


def convert_all_labels():
    for split in ["train", "val", "test"]:
        label_dir = os.path.join(POSE_DATASET, "labels", split)

        if os.path.exists(label_dir):
            bbox_to_pose(label_dir)


def create_pose_yaml():
    yaml_path = DATA_YAML_POSE

    with open(yaml_path, "w") as f:
        f.write(f"""
path: {POSE_DATASET}
train: images/train
val: images/val
test: images/test

names:
  0: "00"
  1: "01"
  2: "10"
  3: "11"

kpt_shape: [4, 3]
""")

    print(f"Created: {yaml_path}")

def train_pose():
    model = YOLO("yolov8n-pose.pt")

    model.train(
        data=DATA_YAML_POSE,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        project=PROJECT_NAME,
        name=RUN_NAME,
        pretrained=True,

        optimizer="Adam",
        lr0=0.001,
        cos_lr=True,
        patience=20,
        workers=0,

        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        erasing=0.0,

        auto_augment=None,
        close_mosaic=0
    )

def main():
    copy_dataset()
    convert_all_labels()
    create_pose_yaml()
    train_pose()


if __name__ == "__main__":
    main()