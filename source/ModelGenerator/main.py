from ultralytics import YOLO
import torch
import os
import pandas as pd
import glob

DATA_YAML = "/workspace/preview/data.yaml"

IMG_SIZE = 320
EPOCHS = 80
BATCH_SIZE = 64

PROJECT_NAME = "blocks_detector"
RUN_NAME = "yolov8n_320_noaug"


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

    all_rows = []

    for w in weights:
        model = YOLO(w)
        metrics = model.val(data=data_yaml, verbose=False)

        maps = metrics.box.maps
        prec = metrics.box.p
        rec = metrics.box.r

        for i, class_name in model.names.items():
            all_rows.append({
                "epoch": int(os.path.basename(w).split("epoch")[1].split(".")[0]),
                "class": class_name,
                "precision": prec[i] if i < len(prec) else None,
                "recall": rec[i] if i < len(rec) else None,
                "mAP50-95": maps[i] if i < len(maps) else None,
            })

    df = pd.DataFrame(all_rows)

    output_path = os.path.join(run_dir, "per_class_per_epoch.xlsx")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for cls in df["class"].unique():
            df[df["class"] == cls].to_excel(writer, sheet_name=str(cls)[:31], index=False)

    print(f"Saved: {output_path}")

def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    data_yaml_abs = os.path.abspath(DATA_YAML)
    print(f"Using data.yaml at: {data_yaml_abs}")

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=data_yaml_abs,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        device=device,
        project=PROJECT_NAME,
        name=RUN_NAME,
        pretrained=True,
        optimizer="Adam",
        lr0=0.001,
        cos_lr=True,
        patience=20,
        workers=0,

        save_period=1,

        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        erasing=0.0,

        auto_augment=None,
        close_mosaic=0
    )
    run_dir = str(results.save_dir)

    export_data(run_dir, data_yaml_abs)


if __name__ == "__main__":
    main()