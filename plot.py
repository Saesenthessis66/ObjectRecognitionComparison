import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
BENCHMARKS = {
    "CPU": "benchmark_cpu_detailed.csv",
    "GPU": "benchmark_gpu_detailed.csv",
    "DPU": "benchmark_dpu_detailed.csv"
}

CLASS_METRIC_CSVS = {
    "CPU": "benchmark_cpu_class_metrics.csv",
    "GPU": "benchmark_gpu_class_metrics.csv",
    "DPU": "benchmark_dpu_class_metrics.csv"
}

PR_CURVE_CSVS = {
    "CPU": "benchmark_cpu_pr_curve.csv",
    "GPU": "benchmark_gpu_pr_curve.csv",
    "DPU": "benchmark_dpu_pr_curve.csv"
}

NUM_CLASSES = 4
CLASS_NAMES = [f"class_{i}" for i in range(NUM_CLASSES)]
OUT_DIR = "plots_validation"

COLORS = {
    "CPU": "#1f77b4",
    "GPU": "#ff7f0e",
    "DPU": "#2ca02c"
}

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- LOAD ----------------
def normalize_detailed_df(df):
    df = df.copy()

    for col in ["pred_class", "gt_class", "tp", "fp", "inference_time"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if "fn" not in df.columns:
        df["fn"] = ((df["pred_class"] == -1) & (df["gt_class"] != -1)).astype(int)
        df.loc[df["fn"] == 1, "fp"] = 0

    if "confidence" not in df.columns:
        df["confidence"] = 0.0

    if "iou" not in df.columns:
        df["iou"] = 0.0

    if "outcome" not in df.columns:
        df["outcome"] = ""
        df.loc[df["tp"] == 1, "outcome"] = "TP"
        df.loc[(df["fp"] == 1) & (df["fn"] == 0), "outcome"] = "FP"
        df.loc[(df["fp"] == 0) & (df["fn"] == 1), "outcome"] = "FN"
        df.loc[(df["fp"] == 1) & (df["fn"] == 1), "outcome"] = "CLS_ERR"

    numeric_cols = ["pred_class", "gt_class", "confidence", "iou", "tp", "fp", "fn", "inference_time"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["pred_class"] = df["pred_class"].astype(int)
    df["gt_class"] = df["gt_class"].astype(int)
    df["tp"] = df["tp"].astype(int)
    df["fp"] = df["fp"].astype(int)
    df["fn"] = df["fn"].astype(int)

    return df


def load_data():
    data = {}
    for name, path in BENCHMARKS.items():
        if os.path.exists(path):
            data[name] = normalize_detailed_df(pd.read_csv(path))
        else:
            print(f"Skipped {name}: file not found: {path}")
    if not data:
        raise FileNotFoundError("No benchmark detailed CSV files found.")
    return data


data = load_data()

# ---------------- METRICS ----------------
def image_times(df):
    if "image" in df.columns:
        return df.groupby("image", sort=False)["inference_time"].first().values
    return df["inference_time"].values


def compute_stats(df):
    times = image_times(df)
    tp = int(df["tp"].sum())
    fp = int(df["fp"].sum())
    fn = int(df["fn"].sum())

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        "images": int(len(times)),
        "avg_time": float(np.mean(times)),
        "fps": float(1.0 / np.mean(times)),
        "min_time": float(np.min(times)),
        "max_time": float(np.max(times)),
        "std_time": float(np.std(times)),
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


stats = {k: compute_stats(v) for k, v in data.items()}
pd.DataFrame.from_dict(stats, orient="index").to_csv(os.path.join(OUT_DIR, "benchmark_stats_summary.csv"))

# ---------------- PLOT HELPERS ----------------
def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=300)
    plt.close()


def plot_bar(metric_name, ylabel, filename=None):
    labels = list(stats.keys())
    values = [stats[k][metric_name] for k in labels]
    colors = [COLORS.get(k, None) for k in labels]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values, color=colors)
    plt.ylabel(ylabel)
    plt.title(metric_name)

    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.4f}", ha="center", va="bottom")

    save_plot(filename or f"{metric_name}.png")


def plot_grouped_metrics(metrics, ylabel, filename):
    labels = list(stats.keys())
    x = np.arange(len(labels))
    width = 0.8 / len(metrics)

    plt.figure(figsize=(8, 5))
    for i, metric in enumerate(metrics):
        values = [stats[k][metric] for k in labels]
        offset = (i - (len(metrics) - 1) / 2) * width
        plt.bar(x + offset, values, width, label=metric)

    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title("Validation metrics")
    plt.legend()
    save_plot(filename)


# ---------------- LATENCY PLOTS ----------------
def plot_histogram():
    plt.figure(figsize=(8, 5))

    for name, df in data.items():
        plt.hist(
            image_times(df),
            bins=50,
            alpha=0.5,
            label=name,
            color=COLORS.get(name, None)
        )

    plt.xlabel("Inference time [s]")
    plt.ylabel("Count")
    plt.title("Latency distribution")
    plt.legend()
    save_plot("latency_distribution.png")


def plot_boxplot():
    plt.figure(figsize=(7, 5))

    labels = list(data.keys())
    times = [image_times(df) for df in data.values()]
    box = plt.boxplot(times, patch_artist=True, labels=labels)

    for patch, label in zip(box["boxes"], labels):
        if label in COLORS:
            patch.set_facecolor(COLORS[label])

    plt.ylabel("Inference time [s]")
    plt.title("Latency distribution (boxplot)")
    save_plot("latency_boxplot.png")


# ---------------- CONFUSION MATRIX ----------------
def confusion_matrix_from_df(df):
    bg = NUM_CLASSES
    cm = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1), dtype=np.int64)

    for _, row in df.iterrows():
        pred = int(row["pred_class"])
        gt = int(row["gt_class"])

        if gt == -1 and pred == -1:
            continue
        if gt == -1 and pred >= 0:
            cm[bg, pred] += 1
        elif gt >= 0 and pred == -1:
            cm[gt, bg] += 1
        elif gt >= 0 and pred >= 0:
            cm[gt, pred] += 1

    return cm


def plot_confusion_matrix(name, df, normalize=False):
    labels = CLASS_NAMES + ["background"]
    cm = confusion_matrix_from_df(df)
    values = cm.astype(float)

    if normalize:
        row_sum = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_sum, out=np.zeros_like(values), where=row_sum != 0)

    plt.figure(figsize=(8, 7))
    im = plt.imshow(values, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title(f"{name} confusion matrix" + (" normalized" if normalize else ""))

    threshold = values.max() * 0.5 if values.size and values.max() > 0 else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text = f"{values[i, j]:.2f}" if normalize else str(int(values[i, j]))
            color = "white" if values[i, j] > threshold else "black"
            plt.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    suffix = "normalized" if normalize else "raw"
    save_plot(f"{name.lower()}_confusion_matrix_{suffix}.png")

    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        os.path.join(OUT_DIR, f"{name.lower()}_confusion_matrix.csv")
    )


# ---------------- PER-CLASS METRICS ----------------
def per_class_metrics_from_detailed(df):
    rows = []
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        tp = int(((df["tp"] == 1) & (df["pred_class"] == cls_id) & (df["gt_class"] == cls_id)).sum())
        fp = int(((df["fp"] == 1) & (df["pred_class"] == cls_id)).sum())
        fn = int(((df["fn"] == 1) & (df["gt_class"] == cls_id)).sum())
        support = int((df["gt_class"] == cls_id).sum())

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        rows.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })
    return pd.DataFrame(rows)


def plot_per_class_metric(name, df, metric):
    pc = per_class_metrics_from_detailed(df)
    plt.figure(figsize=(8, 5))
    plt.bar(pc["class_name"], pc[metric])
    plt.ylabel(metric)
    plt.title(f"{name} per-class {metric}")
    plt.xticks(rotation=45, ha="right")
    save_plot(f"{name.lower()}_per_class_{metric}.png")
    pc.to_csv(os.path.join(OUT_DIR, f"{name.lower()}_per_class_metrics.csv"), index=False)


def plot_ap50_from_class_metrics():
    frames = []
    for name, path in CLASS_METRIC_CSVS.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "ap50" in df.columns:
                df = df.copy()
                df["backend"] = name
                frames.append(df)

    if not frames:
        return

    metrics = pd.concat(frames, ignore_index=True)
    labels = list(metrics["class_name"].drop_duplicates())
    backends = list(metrics["backend"].drop_duplicates())
    x = np.arange(len(labels))
    width = 0.8 / len(backends)

    plt.figure(figsize=(10, 5))
    for i, backend in enumerate(backends):
        subset = metrics[metrics["backend"] == backend].set_index("class_name")
        values = [float(subset.loc[label, "ap50"]) if label in subset.index else 0.0 for label in labels]
        offset = (i - (len(backends) - 1) / 2) * width
        plt.bar(x + offset, values, width, label=backend)

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("AP@0.50")
    plt.title("Per-class AP@0.50")
    plt.legend()
    save_plot("per_class_ap50.png")


# ---------------- PR AND CONFIDENCE PLOTS ----------------
def plot_pr_curve_from_detailed(name, df):
    det = df[df["pred_class"] != -1].copy()
    gt_total = int(((df["gt_class"] != -1) & ((df["tp"] == 1) | (df["fn"] == 1))).sum())

    if det.empty or gt_total == 0:
        return

    det = det.sort_values("confidence", ascending=False)
    tp_cum = det["tp"].cumsum().values
    fp_cum = det["fp"].cumsum().values
    precision = tp_cum / (tp_cum + fp_cum + 1e-6)
    recall = tp_cum / (gt_total + 1e-6)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=name, color=COLORS.get(name, None))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{name} precision-recall curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    save_plot(f"{name.lower()}_precision_recall_curve.png")


def plot_pr_curves_from_files():
    found = False
    plt.figure(figsize=(7, 6))

    for name, path in PR_CURVE_CSVS.items():
        if not os.path.exists(path):
            continue
        pr = pd.read_csv(path)
        if pr.empty or not {"precision", "recall"}.issubset(pr.columns):
            continue
        pr = pr.sort_values("recall")
        plt.plot(pr["recall"], pr["precision"], label=name, color=COLORS.get(name, None))
        found = True

    if not found:
        plt.close()
        return

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curves")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    save_plot("precision_recall_curves_from_pr_csv.png")


def plot_confidence_histogram(name, df):
    det = df[df["pred_class"] != -1].copy()
    if det.empty:
        return

    tp_conf = det[det["tp"] == 1]["confidence"].values
    fp_conf = det[det["tp"] == 0]["confidence"].values

    plt.figure(figsize=(7, 5))
    if len(tp_conf):
        plt.hist(tp_conf, bins=30, alpha=0.6, label="TP")
    if len(fp_conf):
        plt.hist(fp_conf, bins=30, alpha=0.6, label="FP")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title(f"{name} confidence distribution")
    plt.legend()
    save_plot(f"{name.lower()}_confidence_distribution.png")


# ---------------- GENERATE ALL ----------------
for metric, ylabel in [
    ("avg_time", "Average inference time [s]"),
    ("fps", "FPS"),
    ("min_time", "Min inference time [s]"),
    ("max_time", "Max inference time [s]"),
    ("std_time", "Std deviation [s]"),
    ("p50", "Median latency [s]"),
    ("p95", "p95 latency [s]"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1")
]:
    plot_bar(metric, ylabel)

plot_grouped_metrics(["precision", "recall", "f1"], "Score", "precision_recall_f1.png")
plot_histogram()
plot_boxplot()

for name, df in data.items():
    plot_confusion_matrix(name, df, normalize=False)
    plot_confusion_matrix(name, df, normalize=True)
    plot_per_class_metric(name, df, "precision")
    plot_per_class_metric(name, df, "recall")
    plot_per_class_metric(name, df, "f1")
    plot_pr_curve_from_detailed(name, df)
    plot_confidence_histogram(name, df)

plot_ap50_from_class_metrics()
plot_pr_curves_from_files()

print(f"All plots generated in: {OUT_DIR}")
