from ultralytics import YOLO
import os
import pandas as pd
import glob


def export_data(run_dir, data_yaml):

    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv_path):
        print("results.csv not found!")
        return

    df = pd.read_csv(csv_path, encoding="latin1")
    df.to_csv(os.path.join(run_dir, "results_utf8.csv"), encoding="utf-8", index=False)

    weights = sorted(glob.glob(os.path.join(run_dir, "weights", "epoch*.pt")))

    if not weights:
        print("No epoch weights found!")
        return

    rows = []

    for w in weights:
        model = YOLO(w)
        metrics = model.val(data=data_yaml, verbose=False)

        for i, class_name in model.names.items():
            rows.append({
                "epoch": int(os.path.basename(w).split("epoch")[1].split(".")[0]),
                "class": class_name,
                "precision": metrics.box.p[i] if i < len(metrics.box.p) else None,
                "recall": metrics.box.r[i] if i < len(metrics.box.r) else None,
                "mAP50-95": metrics.box.maps[i] if i < len(metrics.box.maps) else None,
            })

    df = pd.DataFrame(rows)

    out_path = os.path.join(run_dir, "per_class_per_epoch.xlsx")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for cls in df["class"].unique():
            df[df["class"] == cls].to_excel(writer, sheet_name=str(cls)[:31], index=False)

    print(f"Saved: {out_path}")