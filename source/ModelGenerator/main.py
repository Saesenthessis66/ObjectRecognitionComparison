from ultralytics import YOLO
import os

from config import DEVICE, DATA_YAML, IMG_SIZE, EPOCHS, BATCH_SIZE, PROJECT_NAME, RUN_NAME, TRAIN_ARGS
from export import export_data
from eval import evaluate_eval_dataset


def main():
    print("Using device:", DEVICE)
    print("Using data:", DATA_YAML)

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT_NAME,
        name=RUN_NAME,
        **TRAIN_ARGS
    )

    run_dir = str(results.save_dir)

    export_data(run_dir, DATA_YAML)

    best_model = os.path.join(run_dir, "weights", "best.pt")
    evaluate_eval_dataset(best_model)


if __name__ == "__main__":
    main()