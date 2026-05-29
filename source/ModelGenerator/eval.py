import os
import glob
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import time

EVAL_DIR = "/workspace/preview/eval"
OUT_DIR = "eval_results"
IMG_SIZE = 320

DETECT_DIR = "/app/yolov5/runs/detect_eval/exp"
DATA_YAML = "/workspace/preview/data.yaml"


# run inference
def run_inference(weights, image_path):
    print(f"\n[INFO] Running inference on: {image_path}")

    # clean previous results
    if os.path.exists(DETECT_DIR):
        shutil.rmtree(DETECT_DIR)

    cmd = [
        "python", "detect.py",
        "--weights", weights,
        "--source", image_path,
        "--img", str(IMG_SIZE),
        "--conf", "0.25",
        "--save-txt",
        "--save-conf",  
        "--nosave",
        "--data", DATA_YAML,  
        "--project", "runs/detect_eval",
        "--name", "exp",
        "--exist-ok"
    ]

    start_time = time.time()

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
        print("[ERROR] Inference timeout! Killing process...")
        process.kill()
        return False

    elapsed = time.time() - start_time
    print(f"[INFO] Inference finished in {elapsed:.2f}s")

    if process.returncode != 0:
        print("[ERROR] detect.py failed")
        return False

    return True


# read prediction
def read_prediction(label_path):
    if not os.path.exists(label_path):
        print("[DEBUG] No label file:", label_path)
        return None, None

    with open(label_path, "r") as f:
        lines = f.readlines()

    if not lines:
        print("[DEBUG] Empty label file:", label_path)
        return None, None

    parts = lines[0].strip().split()

    cls = int(parts[0])
    conf = float(parts[5]) if len(parts) > 5 else None  

    return cls, conf


# parse filename
def parse_eval_filename(name):
    parts = name.replace(".png", "").split("_")

    cls = parts[0]

    if "dist" in parts:
        return cls, "distance", float(parts[2])

    if "rot" in parts:
        return cls, "rotation", float(parts[2])

    return None, None, None


# plot results
def plot_results(df):
    os.makedirs(OUT_DIR, exist_ok=True)

    for mode in ["distance", "rotation"]:
        sub = df[df["mode"] == mode]

        if sub.empty:
            print(f"[WARNING] No data for {mode}")
            continue

        grouped = sub.groupby("value")["correct"].mean()

        plt.figure()
        plt.plot(grouped.index, grouped.values, marker="o")
        plt.xlabel(mode)
        plt.ylabel("accuracy")
        plt.title(f"Accuracy vs {mode}")
        plt.grid()

        out_path = os.path.join(OUT_DIR, f"{mode}_curve.png")
        plt.savefig(out_path)
        plt.close()

        print(f"[INFO] Saved plot: {out_path}")


# main evaluation
def evaluate_eval_dataset(model_path):
    results = []

    for mode in ["distance", "rotation"]:
        img_dir = os.path.join(EVAL_DIR, mode)
        images = glob.glob(os.path.join(img_dir, "*.png"))

        print(f"[INFO] Found {len(images)} images in {mode}")

        for i, img_path in enumerate(images):
            print(f"\n[PROGRESS] {mode} {i+1}/{len(images)}")

            filename = os.path.basename(img_path)

            gt, _, value = parse_eval_filename(filename)
            if gt is None:
                continue

            gt_id = int(gt) 

            success = run_inference(model_path, img_path)
            if not success:
                continue

            label_file = os.path.join(
                DETECT_DIR,
                "labels",
                filename.replace(".png", ".txt")
            )

            pred_cls_id, conf = read_prediction(label_file)

            print(f"[DEBUG] GT={gt_id}, PRED={pred_cls_id}, CONF={conf}")

            results.append({
                "mode": mode,
                "value": value,
                "gt": gt_id,
                "pred": pred_cls_id,
                "conf": conf,
                "correct": pred_cls_id == gt_id
            })

    df = pd.DataFrame(results)

    if df.empty:
        print("[ERROR] No evaluation results generated.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUT_DIR, "raw_results.csv")
    df.to_csv(csv_path, index=False)

    print(f"[INFO] Saved CSV: {csv_path}")

    plot_results(df)

    print("\n===== EVALUATION DONE =====")