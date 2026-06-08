import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
CPU_CSV = "benchmark_cpu_detailed.csv"
GPU_CSV = "benchmark_gpu_detailed.csv"
DPU_CSV = "benchmark_dpu_detailed.csv"

COLORS = {
    "CPU": "#1f77b4",
    "GPU": "#ff7f0e",
    "DPU": "#2ca02c"
}

# ---------------- LOAD ----------------
cpu = pd.read_csv(CPU_CSV)
gpu = pd.read_csv(GPU_CSV)
dpu = pd.read_csv(DPU_CSV)

data = {
    "CPU": cpu,
    "GPU": gpu,
    "DPU": dpu
}

# ---------------- METRICS ----------------
def compute_stats(df):
    times = df["inference_time"].values

    tp = df["tp"].sum()
    fp = df["fp"].sum()

    precision = tp / (tp + fp + 1e-6)

    return {
        "avg_time": np.mean(times),
        "fps": 1.0 / np.mean(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "std_time": np.std(times),
        "p50": np.percentile(times, 50),
        "p95": np.percentile(times, 95),
        "precision": precision
    }

stats = {k: compute_stats(v) for k, v in data.items()}

# ---------------- BAR PLOTS ----------------
def plot_bar(metric_name, ylabel):
    labels = list(stats.keys())
    values = [stats[k][metric_name] for k in labels]
    colors = [COLORS[k] for k in labels]

    plt.figure()
    plt.bar(labels, values, color=colors)

    plt.ylabel(ylabel)
    plt.title(metric_name)

    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.4f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f"{metric_name}.png", dpi=300)
    plt.close()

# ---------------- HISTOGRAM ----------------
def plot_histogram():
    plt.figure()

    for name, df in data.items():
        plt.hist(
            df["inference_time"],
            bins=50,
            alpha=0.5,
            label=name
        )

    plt.xlabel("Inference time [s]")
    plt.ylabel("Count")
    plt.title("Latency distribution")
    plt.legend()

    plt.tight_layout()
    plt.savefig("latency_distribution.png", dpi=300)
    plt.close()

# ---------------- BOXPLOT ----------------
def plot_boxplot():
    plt.figure()

    times = [df["inference_time"].values for df in data.values()]
    labels = list(data.keys())

    box = plt.boxplot(times, patch_artist=True, labels=labels)

    for patch, label in zip(box['boxes'], labels):
        patch.set_facecolor(COLORS[label])

    plt.ylabel("Inference time [s]")
    plt.title("Latency distribution (boxplot)")

    plt.tight_layout()
    plt.savefig("latency_boxplot.png", dpi=300)
    plt.close()

# ---------------- GENERATE ALL ----------------
plot_bar("avg_time", "Average inference time [s]")
plot_bar("fps", "FPS")
plot_bar("min_time", "Min inference time [s]")
plot_bar("max_time", "Max inference time [s]")
plot_bar("std_time", "Std deviation [s]")
plot_bar("p50", "Median (p50) [s]")
plot_bar("p95", "p95 latency [s]")
plot_bar("precision", "Precision")

plot_histogram()
plot_boxplot()

print("All plots generated.")