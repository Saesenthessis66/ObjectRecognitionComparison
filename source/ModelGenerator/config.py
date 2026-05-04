import torch

DATA_YAML = "/workspace/preview/data.yaml"

IMG_SIZE = 320
EPOCHS = 80
BATCH_SIZE = 64

PROJECT_NAME = "blocks_detector"
RUN_NAME = "yolov8n_320_noaug"

DEVICE = 0 if torch.cuda.is_available() else "cpu"

TRAIN_ARGS = dict(
    pretrained=True,
    optimizer="Adam",
    lr0=0.001,
    cos_lr=True,
    patience=20,
    workers=0,
    save_period=1,

    # augmentation OFF
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