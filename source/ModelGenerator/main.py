import os

from config import DATA_YAML, RUN_NAME
from export import export_data
from eval import evaluate_eval_dataset


def main():
    print("===== POST-TRAINING PIPELINE =====")

    # YOLOv5 working dir = /app/yolov5
    run_dir = os.path.join("../yolov5/runs/train", RUN_NAME)

    if not os.path.exists(run_dir):
        raise RuntimeError(f"Run directory not found: {run_dir}")

    print("Using run_dir:", run_dir)

    # -----------------------------
    # STEP 1: Evaluate ALL epochs
    # -----------------------------
    print("\n===== STEP 1: PER-EPOCH VALIDATION =====")
    export_data(run_dir, DATA_YAML)

    # -----------------------------
    # STEP 2: Evaluate BEST model on custom dataset
    # -----------------------------
    best_model = os.path.join(run_dir, "weights", "best.pt")

    if not os.path.exists(best_model):
        raise RuntimeError("best.pt not found")

    print("\n===== STEP 2: CUSTOM EVAL (DISTANCE / ROTATION) =====")
    evaluate_eval_dataset(best_model)

    print("\n===== DONE =====")


if __name__ == "__main__":
    main()