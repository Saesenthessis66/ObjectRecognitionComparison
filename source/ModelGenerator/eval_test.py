from ultralytics import YOLO
import pandas as pd
import os


def evaluate_model(model_path, data_yaml, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    csv_file = os.path.join(out_dir, "results_test.csv")
    xlsx_file = os.path.join(out_dir, "results_test.xlsx")

    model = YOLO(model_path)

    metrics = model.val(
        data=data_yaml,
        split="test"
    )

    df_global = pd.DataFrame([metrics.results_dict])

    df_global = df_global.rename(columns={
        "metrics/precision(B)": "precision",
        "metrics/recall(B)": "recall",
        "metrics/mAP50(B)": "mAP50",
        "metrics/mAP50-95(B)": "mAP50-95"
    })

    df_global["class"] = "all"

    rows = []

    for i in range(len(model.names)):
        res = metrics.class_result(i)

        rows.append({
            "class": model.names[i],
            "precision": res[0],
            "recall": res[1],
            "mAP50": res[2],
            "mAP50-95": res[3],
        })

    df_class = pd.DataFrame(rows)

    df_all = pd.concat([df_global, df_class], ignore_index=True)

    df_all["F1"] = (
        2 * (df_all["precision"] * df_all["recall"]) /
        (df_all["precision"] + df_all["recall"])
    )

    def classify_status(row):
        if row["class"] == "all":
            return "GLOBAL"
        if row["recall"] < 0.7:
            return "BAD"
        elif row["recall"] < 0.8:
            return "OK"
        else:
            return "GOOD"

    df_all["status"] = df_all.apply(classify_status, axis=1)

    df_global = df_all[df_all["class"] == "all"]
    df_classes = df_all[df_all["class"] != "all"].sort_values(by="class")

    df_final = pd.concat([df_global, df_classes], ignore_index=True)

    column_order = [
        "class",
        "status",
        "precision",
        "recall",
        "mAP50",
        "mAP50-95",
        "F1"
    ]

    df_final = df_final[column_order].round(4)

    df_final.to_csv(csv_file, index=False)
    df_final.to_excel(xlsx_file, index=False)

    print("Evaluation saved:")
    print(csv_file)
    print(xlsx_file)


if __name__ == "__main__":
    evaluate_model(
        model_path="/home/app/dataBlender/workspace/preview/bestBlender416.pt",
        data_yaml="/home/app/dataBlender/workspace/preview/data.yaml",
        out_dir="/home/app/dataBlender/workspace/preview/eval_test"
    )