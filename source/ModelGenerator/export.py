import os
import glob
import subprocess
import pandas as pd
import re

# run yolov5 validation
def run_val(weights, data_yaml):
    cmd = [
        "python", "val.py",
        "--weights", weights,
        "--data", data_yaml,
        "--img", "320"
    ]

    result = subprocess.run(
        cmd,
        cwd="/app/yolov5",
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("ERROR running val.py")
        print(result.stderr)
        raise RuntimeError("val.py failed")
    # important: yolo prints to stderr
    return result.stdout + "\n" + result.stderr


# parse yolo v5 output
def parse_output(output):
    rows = []

    for line in output.split("\n"):
        line = line.strip()
        # match table rows like: class images labels P R mAP50 mAP50-95
        match = re.match(
            r"^(\S+)\s+\d+\s+\d+\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)",
            line
        )

        if match:
            rows.append({
                "class": match.group(1),
                "precision": float(match.group(2)),
                "recall": float(match.group(3)),
                "mAP50": float(match.group(4)),
                "mAP50-95": float(match.group(5)),
            })

    return rows


# main export function
def export_data(run_dir, data_yaml):
    print("Starting export...")

    # fix encoding of training csv
    csv_path = os.path.join(run_dir, "results.csv")
    if os.path.exists(csv_path):
        df_train = pd.read_csv(csv_path, encoding="latin1")
        df_train.to_csv(
            os.path.join(run_dir, "results_utf8.csv"),
            encoding="utf-8",
            index=False
        )

    # load all epoch weights
    weights = sorted(glob.glob(os.path.join(run_dir, "weights", "epoch*.pt")))

    if not weights:
        print("No epoch weights found!")
        return

    rows = []

    # evaluate each epoch
    for w in weights:
        print(f"Evaluating: {w}")

        output = run_val(w, data_yaml)
        # debug optional
        print("---- RAW OUTPUT (first 200 chars) ----")
        print(output[:200])

        metrics = parse_output(output)

        print("Parsed metrics:", metrics)

        if not metrics:
            print("WARNING: No metrics parsed, skipping epoch.")
            continue

        epoch = int(os.path.basename(w).split("epoch")[1].split(".")[0])

        for m in metrics:
            rows.append({
                "epoch": epoch,
                "class": m["class"],
                "precision": m["precision"],
                "recall": m["recall"],
                "mAP50": m["mAP50"],
                "mAP50-95": m["mAP50-95"],
            })

    # build dataframe
    df = pd.DataFrame(rows)

    if df.empty or "class" not in df.columns:
        print("WARNING: No valid metrics collected. Skipping Excel export.")
        return

    out_path = os.path.join(run_dir, "per_class_per_epoch.xlsx")

    # write excel
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:

        # global all
        df_all = df[df["class"] == "all"].copy()
        if not df_all.empty:
            df_all = df_all.sort_values("epoch")
            df_all.to_excel(writer, sheet_name="ALL", index=False)

        # per-class
        classes = sorted(df["class"].unique())

        for cls in classes:
            if cls == "all":
                continue

            sub = df[df["class"] == cls].copy()

            if sub.empty:
                continue

            sub = sub.sort_values("epoch")

            sub.to_excel(
                writer,
                sheet_name=str(cls)[:31],
                index=False
            )

        # summary pivot tables
        df_no_all = df[df["class"] != "all"]

        if not df_no_all.empty:

            pivot_map50 = df_no_all.pivot_table(
                index="epoch",
                columns="class",
                values="mAP50"
            )

            pivot_map5095 = df_no_all.pivot_table(
                index="epoch",
                columns="class",
                values="mAP50-95"
            )

            pivot_precision = df_no_all.pivot_table(
                index="epoch",
                columns="class",
                values="precision"
            )

            pivot_recall = df_no_all.pivot_table(
                index="epoch",
                columns="class",
                values="recall"
            )

            start_row = 0

            for name, table in [
                ("mAP50", pivot_map50),
                ("mAP50-95", pivot_map5095),
                ("Precision", pivot_precision),
                ("Recall", pivot_recall),
            ]:
                table.to_excel(
                    writer,
                    sheet_name="SUMMARY",
                    startrow=start_row
                )

                start_row += len(table) + 4

    print(f"Saved Excel: {out_path}")