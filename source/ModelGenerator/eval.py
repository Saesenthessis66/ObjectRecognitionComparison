from ultralytics import YOLO
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


EVAL_DIR = "/workspace/preview/eval"
OUT_DIR = "eval_results"


def parse_eval_filename(name):
    parts = name.replace(".png", "").split("_")

    cls = parts[0]

    if "dist" in parts:
        return cls, "distance", float(parts[2])

    if "rot" in parts:
        return cls, "rotation", float(parts[2])

    return None, None, None


def plot_results(df):
    os.makedirs(OUT_DIR, exist_ok=True)

    for mode in ["distance", "rotation"]:
        sub = df[df["mode"] == mode]
        grouped = sub.groupby("value")["correct"].mean()

        plt.figure()
        plt.plot(grouped.index, grouped.values, marker="o")
        plt.xlabel(mode)
        plt.ylabel("accuracy")
        plt.title(f"Accuracy vs {mode}")
        plt.grid()

        plt.savefig(os.path.join(OUT_DIR, f"{mode}_curve.png"))
        plt.close()


def evaluate_eval_dataset(model_path):
    model = YOLO(model_path)

    results = []

    for mode in ["distance", "rotation"]:
        img_dir = os.path.join(EVAL_DIR, mode)

        for img_path in glob.glob(os.path.join(img_dir, "*.png")):
            filename = os.path.basename(img_path)

            gt, _, value = parse_eval_filename(filename)
            if gt is None:
                continue

            pred = model(img_path, verbose=False)[0]

            if len(pred.boxes) == 0:
                pred_cls = None
                conf = 0.0
            else:
                box = pred.boxes[0]
                pred_cls = model.names[int(box.cls)]
                conf = float(box.conf)

            results.append({
                "mode": mode,
                "value": value,
                "gt": gt,
                "pred": pred_cls,
                "conf": conf,
                "correct": pred_cls == gt
            })

    df = pd.DataFrame(results)

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "raw_results.csv"), index=False)

    plot_results(df)

    print("Evaluation done.")