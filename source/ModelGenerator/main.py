from ultralytics import YOLO
import torch
import os

DATA_YAML = "/workspace/preview/data.yaml"

IMG_SIZE = 320
EPOCHS = 80
BATCH_SIZE = 64

PROJECT_NAME = "blocks_detector"
RUN_NAME = "yolov8n_320_noaug"


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    data_yaml_abs = os.path.abspath(DATA_YAML)
    print(f"Using data.yaml at: {data_yaml_abs}")

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=data_yaml_abs,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        device=device,
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


if __name__ == "__main__":
    main()