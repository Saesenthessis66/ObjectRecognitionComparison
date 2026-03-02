from ultralytics import YOLO
import torch
import os

# ====== KONFIGURACJA ======
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(PROJECT_ROOT, "DataGenerator", "preview", "data.yaml")
IMG_SIZE = 320
EPOCHS = 80
BATCH_SIZE = 64
PROJECT_NAME = "blocks_detector"
RUN_NAME = "yolov8n_320"

# ===========================

def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    
    # Force absolute path
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
        workers=4 #TODO REMOVE AUGMENTATION
    )

if __name__ == "__main__":
    main()