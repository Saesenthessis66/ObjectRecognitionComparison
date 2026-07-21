import argparse
import re
import subprocess
from pathlib import Path

import pandas as pd
import yaml


YOLOV5_DIR = Path("/app/yolov5")
IMG_SIZE = 640
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
ROW_PATTERN = re.compile(
    rf"^(?P<class>.+?)\s+"
    rf"(?P<images>\d+)\s+"
    rf"(?P<instances>\d+)\s+"
    rf"(?P<precision>{NUMBER})\s+"
    rf"(?P<recall>{NUMBER})\s+"
    rf"(?P<map50>{NUMBER})\s+"
    rf"(?P<map5095>{NUMBER})$"
)
ANSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
EPOCH_PATTERN = re.compile(r"epoch(\d+)\.pt$")


def get_epoch(path):
    match = EPOCH_PATTERN.search(Path(path).name)
    if not match:
        raise ValueError(f"Invalid checkpoint name: {path}")
    return int(match.group(1))


def get_class_names(data_yaml):
    with open(data_yaml, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    names = data.get("names")
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
    raise ValueError(f"Missing or invalid names in {data_yaml}")


def run_val(weights, data_yaml, yolov5_dir=YOLOV5_DIR, img_size=IMG_SIZE):
    weights = Path(weights).resolve()
    data_yaml = Path(data_yaml).resolve()
    yolov5_dir = Path(yolov5_dir).resolve()

    cmd = [
        "python",
        "val.py",
        "--weights",
        str(weights),
        "--data",
        str(data_yaml),
        "--img",
        str(img_size),
        "--conf-thres",
        "0.001",
        "--iou-thres",
        "0.6",
        "--task",
        "val",
        "--verbose",
    ]

    result = subprocess.run(
        cmd,
        cwd=yolov5_dir,
        capture_output=True,
        text=True,
    )

    output = result.stdout + "\n" + result.stderr

    if result.returncode != 0:
        raise RuntimeError(f"val.py failed for {weights}:\n{output}")

    return output


def parse_output(output):
    metrics = {}

    for raw_line in output.splitlines():
        line = ANSI_PATTERN.sub("", raw_line).strip()
        match = ROW_PATTERN.match(line)

        if not match:
            continue

        class_name = match.group("class").strip()
        metrics[class_name] = {
            "class": class_name,
            "images": int(match.group("images")),
            "instances": int(match.group("instances")),
            "precision": float(match.group("precision")),
            "recall": float(match.group("recall")),
            "mAP50": float(match.group("map50")),
            "mAP50-95": float(match.group("map5095")),
        }

    return list(metrics.values())


def complete_missing_classes(metrics, class_names):
    by_class = {row["class"]: row for row in metrics}
    all_row = by_class.get("all")

    if all_row is None:
        raise RuntimeError("The YOLOv5 'all' metrics row was not found")

    if all_row["instances"] == 0:
        raise RuntimeError("YOLOv5 found zero validation labels. Check data.yaml and label paths")

    class_rows = [row for row in metrics if row["class"] != "all"]

    if not class_rows:
        all_values = [
            all_row["precision"],
            all_row["recall"],
            all_row["mAP50"],
            all_row["mAP50-95"],
        ]
        if any(value != 0.0 for value in all_values):
            raise RuntimeError("Global metrics are non-zero, but per-class rows were not parsed")

        for class_name in class_names:
            metrics.append(
                {
                    "class": class_name,
                    "images": all_row["images"],
                    "instances": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "mAP50": 0.0,
                    "mAP50-95": 0.0,
                }
            )

    return metrics


def safe_sheet_name(name, used_names):
    name = re.sub(r"[\\/*?:\[\]]", "_", str(name))[:31] or "class"
    candidate = name
    counter = 1

    while candidate in used_names:
        suffix = f"_{counter}"
        candidate = f"{name[:31 - len(suffix)]}{suffix}"
        counter += 1

    used_names.add(candidate)
    return candidate


def save_excel(df, output_path):
    used_names = {"ALL", "SUMMARY", "DATA"}

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="DATA", index=False)

        df_all = df[df["class"] == "all"].sort_values("epoch")
        if not df_all.empty:
            df_all.to_excel(writer, sheet_name="ALL", index=False)

        class_names = sorted(name for name in df["class"].unique() if name != "all")

        for class_name in class_names:
            class_df = df[df["class"] == class_name].sort_values("epoch")
            sheet_name = safe_sheet_name(class_name, used_names)
            class_df.to_excel(writer, sheet_name=sheet_name, index=False)

        class_df = df[df["class"] != "all"]
        start_row = 0

        for metric in ["mAP50", "mAP50-95", "precision", "recall"]:
            table = class_df.pivot(index="epoch", columns="class", values=metric)
            table.to_excel(writer, sheet_name="SUMMARY", startrow=start_row)
            start_row += len(table) + 4


def export_data(run_dir, data_yaml, yolov5_dir=YOLOV5_DIR, img_size=IMG_SIZE):
    run_dir = Path(run_dir).resolve()
    data_yaml = Path(data_yaml).resolve()
    yolov5_dir = Path(yolov5_dir).resolve()

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Data file not found: {data_yaml}")
    if not (yolov5_dir / "val.py").is_file():
        raise FileNotFoundError(f"val.py not found in: {yolov5_dir}")

    weights = sorted(
        (run_dir / "weights").glob("epoch*.pt"),
        key=get_epoch,
    )

    if not weights:
        raise FileNotFoundError(f"No epoch checkpoints found in {run_dir / 'weights'}")

    class_names = get_class_names(data_yaml)
    rows = []

    for index, weights_path in enumerate(weights, start=1):
        epoch = get_epoch(weights_path)
        print(f"[{index}/{len(weights)}] Epoch {epoch}: {weights_path}")

        output = run_val(
            weights=weights_path,
            data_yaml=data_yaml,
            yolov5_dir=yolov5_dir,
            img_size=img_size,
        )
        metrics = complete_missing_classes(parse_output(output), class_names)

        for row in metrics:
            rows.append({"epoch": epoch, **row})

        print(
            f"{'Class':<20} {'Images':>8} {'Instances':>10} "
            f"{'P':>12} {'R':>12} {'mAP50':>12} {'mAP50-95':>12}"
        )
        for row in metrics:
            print(
                f"{row['class']:<20} "
                f"{row['images']:>8} "
                f"{row['instances']:>10} "
                f"{row['precision']:>12.6g} "
                f"{row['recall']:>12.6g} "
                f"{row['mAP50']:>12.6g} "
                f"{row['mAP50-95']:>12.6g}"
            )
        print()

    df = pd.DataFrame(rows)
    df = df.sort_values(["epoch", "class"]).reset_index(drop=True)

    duplicates = df.duplicated(["epoch", "class"], keep=False)
    if duplicates.any():
        raise RuntimeError(
            "Duplicate epoch/class rows found:\n"
            + df.loc[duplicates, ["epoch", "class"]].to_string(index=False)
        )

    output_path = run_dir / "per_class_per_epoch.xlsx"
    save_excel(df, output_path)
    print(f"Saved: {output_path}")
    return output_path