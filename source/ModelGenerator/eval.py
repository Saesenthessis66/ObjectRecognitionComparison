import os
import glob
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import time
import re
from PIL import Image

EVAL_DIR = "/workspace/preview/eval"
OUT_DIR = "eval_results"
IMG_SIZE = 640

DETECT_DIR = "/app/yolov5/runs/detect_eval/exp"
DATA_YAML = "/workspace/preview/data.yaml"

IOU_THRESH = 0.5


# ------------------ FILENAME PARSING ------------------

def parse_eval_filename(name):
    name = name.replace(".png", "")

    match = re.match(r"(\d+)_(dist|rot)_(-?\d+\.?\d*)_(\d+)", name)
    if not match:
        print("[ERROR] Failed to parse:", name)
        return None, None, None

    cls = int(match.group(1))
    mode = match.group(2)
    value = float(match.group(3))

    return cls, mode, value


# ------------------ YOLO UTILS ------------------

def yolo_to_xyxy(x, y, w, h, img_w, img_h):
    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h
    return x1, y1, x2, y2


def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    union = box1_area + box2_area - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


# ------------------ INFERENCE ------------------

def run_inference(weights, image_path):
    if os.path.exists(DETECT_DIR):
        shutil.rmtree(DETECT_DIR)

    cmd = [
        "python", "detect.py",
        "--weights", weights,
        "--source", image_path,
        "--img", str(IMG_SIZE),
        "--conf", "0.05",
        "--save-txt",
        "--save-conf",
        "--nosave",
        "--data", DATA_YAML,
        "--project", "runs/detect_eval",
        "--name", "exp",
        "--exist-ok"
    ]

    process = subprocess.Popen(
        cmd,
        cwd="/app/yolov5",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    try:
        for line in process.stdout:
            print("[YOLO]", line.strip())

        process.wait(timeout=60)

    except subprocess.TimeoutExpired:
        process.kill()
        return False

    return process.returncode == 0


# ------------------ READ DATA ------------------

def read_predictions(label_path):
    preds = []

    if not os.path.exists(label_path):
        return preds

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:5])
            conf = float(parts[5]) if len(parts) > 5 else 0

            preds.append({
                "cls": cls,
                "bbox": (x, y, w, h),
                "conf": conf
            })

    return preds


def read_ground_truth(label_path):
    gts = []

    if not os.path.exists(label_path):
        return gts

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:5])

            gts.append({
                "cls": cls,
                "bbox": (x, y, w, h)
            })

    return gts


# ------------------ MATCHING ------------------

def evaluate_image(preds, gts, img_w, img_h):
    if not preds:
        return False, 0.0, None, True

    if not gts:
        return False, 0.0, None, False

    best_iou = 0
    best_pred = None
    best_gt = None

    for gt in gts:
        gt_box = yolo_to_xyxy(*gt["bbox"], img_w, img_h)

        for pred in preds:
            pred_box = yolo_to_xyxy(*pred["bbox"], img_w, img_h)

            iou = compute_iou(pred_box, gt_box)

            if iou > best_iou:
                best_iou = iou
                best_pred = pred
                best_gt = gt

    if best_pred is None:
        return False, 0.0, None, False

    correct = (best_iou >= IOU_THRESH) and (best_pred["cls"] == best_gt["cls"])

    return correct, best_iou, best_pred["conf"], False


# ------------------ PLOTTING ------------------

def plot_results(df):
    os.makedirs(OUT_DIR, exist_ok=True)

    df["detected"] = df["iou"] > 0

    for mode in ["distance", "rotation"]:
        sub = df[df["mode"] == mode]

        if sub.empty:
            continue

        grouped = sub.groupby("value").agg({
            "correct": "mean",
            "iou": "mean",
            "conf": "mean",
            "detected": "mean"
        }).sort_index()

        # Accuracy
        plt.figure()
        plt.plot(grouped.index, grouped["correct"], marker="o")
        plt.xlabel(mode)
        plt.ylabel("accuracy (IoU≥0.5)")
        plt.title(f"Accuracy vs {mode}")
        plt.grid()
        plt.savefig(os.path.join(OUT_DIR, f"{mode}_accuracy.png"))
        plt.close()

        # IoU
        plt.figure()
        plt.plot(grouped.index, grouped["iou"], marker="o")
        plt.xlabel(mode)
        plt.ylabel("mean IoU")
        plt.title(f"IoU vs {mode}")
        plt.grid()
        plt.savefig(os.path.join(OUT_DIR, f"{mode}_iou.png"))
        plt.close()

        # Detection rate
        plt.figure()
        plt.plot(grouped.index, grouped["detected"], marker="o")
        plt.xlabel(mode)
        plt.ylabel("detection rate")
        plt.title(f"Detection rate vs {mode}")
        plt.grid()
        plt.savefig(os.path.join(OUT_DIR, f"{mode}_detected.png"))
        plt.close()


# ------------------ MAIN ------------------

def evaluate_eval_dataset(model_path):
    results = []
    no_pred_count = 0

    for folder_mode in ["distance", "rotation"]:
        img_dir = os.path.join(EVAL_DIR, folder_mode)

        images = glob.glob(os.path.join(img_dir, "*.png"))

        print(f"[INFO] {folder_mode}: {len(images)} images")

        for i, img_path in enumerate(images):
            filename = os.path.basename(img_path)

            cls, parsed_mode, value = parse_eval_filename(filename)
            if cls is None:
                continue

            mode = "distance" if parsed_mode == "dist" else "rotation"

            print(f"[{mode}] {i+1}/{len(images)}: {filename} → value={value}")

            success = run_inference(model_path, img_path)
            if not success:
                continue

            pred_label = os.path.join(
                DETECT_DIR,
                "labels",
                filename.replace(".png", ".txt")
            )

            gt_label = os.path.join(
                img_dir,
                filename.replace(".png", ".txt")
            )

            preds = read_predictions(pred_label)
            gts = read_ground_truth(gt_label)

            img = Image.open(img_path)
            img_w, img_h = img.size

            correct, iou, conf, no_pred = evaluate_image(preds, gts, img_w, img_h)

            if no_pred:
                no_pred_count += 1

            print(f"[DEBUG] IoU={iou:.3f}, CONF={conf}, CORRECT={correct}")

            results.append({
                "mode": mode,
                "value": value,
                "iou": iou,
                "conf": conf,
                "correct": correct
            })

    df = pd.DataFrame(results)

    if df.empty:
        print("[ERROR] No results.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "raw_results.csv"), index=False)

    print(f"[INFO] No detections: {no_pred_count}")

    plot_results(df)

    print("===== DONE =====")