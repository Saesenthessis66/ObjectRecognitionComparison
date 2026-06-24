import os
import time
import csv
import cv2
import numpy as np
import onnxruntime as ort

# ---------------- CONFIG ----------------
ONNX_PATH = "DetectionModel_int.onnx"
IMAGE_DIR = "test/images"
LABEL_DIR = "test/labels"

INPUT_SIZE = 640
NUM_CLASSES = 4
CLASS_NAMES = [f"class_{i}" for i in range(NUM_CLASSES)]

OBJ_THRESH = 0.30
CONF_THRESH = 0.25
NMS_IOU_THRESH = 0.40
VAL_IOU_THRESH = 0.50

OUTPUT_PREFIX = "benchmark_cpu"

anchors = [
    [(10, 13), (16, 30), (33, 23)],
    [(30, 61), (62, 45), (59, 119)],
    [(116, 90), (156, 198), (373, 326)]
]

strides = [8, 16, 32]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------- ONNX ----------------
session = ort.InferenceSession(
    ONNX_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

# ---------------- UTILS ----------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def iou(a, b):
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))

    return inter / (area_a + area_b - inter + 1e-6)


def nms(boxes, iou_thresh=0.4, class_aware=True):
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []

    while boxes:
        best = boxes.pop(0)
        keep.append(best)

        remaining = []
        for b in boxes:
            different_class = int(best[5]) != int(b[5])
            if class_aware and different_class:
                remaining.append(b)
            elif iou(best, b) < iou_thresh:
                remaining.append(b)
        boxes = remaining

    return keep


def letterbox(img, new_size=640):
    h, w = img.shape[:2]
    scale = min(new_size / h, new_size / w)

    nh, nw = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (nw, nh))

    canvas = np.full((new_size, new_size, 3), 114, dtype=np.uint8)

    top = (new_size - nh) // 2
    left = (new_size - nw) // 2

    canvas[top:top + nh, left:left + nw] = img_resized
    return canvas, scale, left, top


def load_labels(path, w, h):
    boxes = []
    if not os.path.exists(path):
        return boxes

    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(f"Invalid YOLO label format in {path}:{line_no}: {line.strip()}")

            cls, x, y, bw, bh = map(float, parts)

            x1 = (x - bw / 2) * w
            y1 = (y - bh / 2) * h
            x2 = (x + bw / 2) * w
            y2 = (y + bh / 2) * h

            boxes.append([x1, y1, x2, y2, int(cls)])

    return boxes


def clip_boxes(boxes, w, h):
    clipped = []
    for b in boxes:
        x1, y1, x2, y2 = b[:4]
        rest = b[4:]
        x1 = min(max(float(x1), 0.0), float(w - 1))
        y1 = min(max(float(y1), 0.0), float(h - 1))
        x2 = min(max(float(x2), 0.0), float(w - 1))
        y2 = min(max(float(y2), 0.0), float(h - 1))
        if x2 > x1 and y2 > y1:
            clipped.append([x1, y1, x2, y2] + rest)
    return clipped


# ---------------- DECODE ----------------
def decode(output, anchors, stride):
    output = np.transpose(output, (0, 2, 3, 1))

    bs, h, w, c = output.shape
    output = output.reshape(bs, h, w, 3, 5 + NUM_CLASSES)

    grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    boxes = []

    for a in range(3):
        pred = output[0, :, :, a, :]

        x = (sigmoid(pred[..., 0]) * 2 - 0.5 + grid_x) * stride
        y = (sigmoid(pred[..., 1]) * 2 - 0.5 + grid_y) * stride

        w_box = (sigmoid(pred[..., 2]) * 2) ** 2 * anchors[a][0]
        h_box = (sigmoid(pred[..., 3]) * 2) ** 2 * anchors[a][1]

        obj = sigmoid(pred[..., 4])
        cls = sigmoid(pred[..., 5:])

        cls_score = np.max(cls, axis=-1)
        cls_id = np.argmax(cls, axis=-1)
        conf = obj * cls_score

        mask = (obj > OBJ_THRESH) & (conf > CONF_THRESH)

        ys, xs = np.where(mask)
        for i, j in zip(ys, xs):
            boxes.append([
                x[i, j] - w_box[i, j] / 2,
                y[i, j] - h_box[i, j] / 2,
                x[i, j] + w_box[i, j] / 2,
                y[i, j] + h_box[i, j] / 2,
                float(conf[i, j]),
                int(cls_id[i, j]),
                float(obj[i, j]),
                float(cls_score[i, j])
            ])

    return boxes


# ---------------- VALIDATION ----------------
def match_image_for_csv_and_confusion(preds, gts, image_name, inference_time, val_iou_thresh=0.5):
    rows = []
    cm = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1), dtype=np.int64)
    bg = NUM_CLASSES

    matched_gt = set()
    preds_sorted = sorted(preds, key=lambda x: x[4], reverse=True)

    for det in preds_sorted:
        pred_cls = int(det[5])
        best_iou = 0.0
        best_idx = -1
        best_gt_cls = -1

        for i, gt in enumerate(gts):
            if i in matched_gt:
                continue
            iou_val = iou(det, gt)
            if iou_val > best_iou:
                best_iou = iou_val
                best_idx = i
                best_gt_cls = int(gt[4])

        if best_idx != -1 and best_iou >= val_iou_thresh:
            matched_gt.add(best_idx)
            cm[best_gt_cls, pred_cls] += 1

            if pred_cls == best_gt_cls:
                rows.append([
                    image_name, pred_cls, best_gt_cls, float(det[4]), float(best_iou),
                    1, 0, 0, "TP", inference_time
                ])
            else:
                rows.append([
                    image_name, pred_cls, best_gt_cls, float(det[4]), float(best_iou),
                    0, 1, 1, "CLS_ERR", inference_time
                ])
        else:
            cm[bg, pred_cls] += 1
            rows.append([
                image_name, pred_cls, -1, float(det[4]), float(best_iou),
                0, 1, 0, "FP", inference_time
            ])

    for i, gt in enumerate(gts):
        if i not in matched_gt:
            gt_cls = int(gt[4])
            cm[gt_cls, bg] += 1
            rows.append([
                image_name, -1, gt_cls, 0.0, 0.0,
                0, 0, 1, "FN", inference_time
            ])

    if len(preds_sorted) == 0 and len(gts) == 0:
        rows.append([
            image_name, -1, -1, 0.0, 0.0,
            0, 0, 0, "TN_EMPTY", inference_time
        ])

    return rows, cm


def compute_ap(recall, precision):
    if len(recall) == 0 or len(precision) == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_class_metrics(all_dets, all_gts, image_names, val_iou_thresh=0.5):
    metrics = []
    pr_rows = []

    for cls in range(NUM_CLASSES):
        detections = []
        total_gt = 0

        for img_idx, (dlist, glist) in enumerate(zip(all_dets, all_gts)):
            gt_cls = [g for g in glist if int(g[4]) == cls]
            det_cls = [d for d in dlist if int(d[5]) == cls]
            total_gt += len(gt_cls)

            matched = set()
            det_cls = sorted(det_cls, key=lambda x: x[4], reverse=True)

            for det in det_cls:
                best_iou = 0.0
                best_idx = -1

                for i, gt in enumerate(gt_cls):
                    iou_val = iou(det, gt)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_idx = i

                is_tp = best_iou >= val_iou_thresh and best_idx not in matched
                if is_tp:
                    matched.add(best_idx)

                detections.append({
                    "image": image_names[img_idx],
                    "confidence": float(det[4]),
                    "tp": 1 if is_tp else 0,
                    "iou": float(best_iou)
                })

        detections.sort(key=lambda x: x["confidence"], reverse=True)

        if total_gt == 0:
            metrics.append({
                "class_id": cls,
                "class_name": CLASS_NAMES[cls],
                "support": 0,
                "tp": 0,
                "fp": len(detections),
                "fn": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "ap50": 0.0
            })
            continue

        tp = np.array([d["tp"] for d in detections], dtype=np.float32)
        fp = 1.0 - tp

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        precision_curve = tp_cum / (tp_cum + fp_cum + 1e-6)
        recall_curve = tp_cum / (total_gt + 1e-6)
        ap50 = compute_ap(recall_curve, precision_curve)

        for d, p, r in zip(detections, precision_curve, recall_curve):
            pr_rows.append({
                "class_id": cls,
                "class_name": CLASS_NAMES[cls],
                "image": d["image"],
                "confidence": d["confidence"],
                "iou": d["iou"],
                "tp": d["tp"],
                "precision": float(p),
                "recall": float(r)
            })

        tp_fixed = int(np.sum(tp))
        fp_fixed = int(np.sum(fp))
        fn_fixed = int(total_gt - tp_fixed)
        precision = tp_fixed / (tp_fixed + fp_fixed + 1e-6)
        recall = tp_fixed / (tp_fixed + fn_fixed + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        metrics.append({
            "class_id": cls,
            "class_name": CLASS_NAMES[cls],
            "support": int(total_gt),
            "tp": tp_fixed,
            "fp": fp_fixed,
            "fn": fn_fixed,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "ap50": float(ap50)
        })

    valid_aps = [m["ap50"] for m in metrics if m["support"] > 0]
    map50 = float(np.mean(valid_aps)) if valid_aps else 0.0
    return metrics, pr_rows, map50


def save_confusion_matrix(path, cm):
    labels = CLASS_NAMES + ["background"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\predicted"] + labels)
        for i, label in enumerate(labels):
            writer.writerow([label] + cm[i].tolist())


def save_class_metrics(path, metrics):
    fields = [
        "class_id", "class_name", "support", "tp", "fp", "fn",
        "precision", "recall", "f1", "ap50"
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)


def save_pr_curve(path, pr_rows):
    fields = [
        "class_id", "class_name", "image", "confidence", "iou", "tp",
        "precision", "recall"
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pr_rows)


# ---------------- BENCHMARK ----------------
times = []
all_dets = []
all_gts = []
image_names = []
confusion_matrix = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1), dtype=np.int64)

detailed_csv_path = f"{OUTPUT_PREFIX}_detailed.csv"
detailed_csv_file = open(detailed_csv_path, "w", newline="")
writer = csv.writer(detailed_csv_file)

writer.writerow([
    "image", "pred_class", "gt_class", "confidence", "iou",
    "tp", "fp", "fn", "outcome", "inference_time"
])

# warmup
dummy = np.random.rand(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
for _ in range(10):
    _ = session.run(None, {input_name: dummy})

image_files = [
    name for name in sorted(os.listdir(IMAGE_DIR))
    if os.path.splitext(name.lower())[1] in IMAGE_EXTS
]

for img_name in image_files:
    img_path = os.path.join(IMAGE_DIR, img_name)
    base = os.path.splitext(img_name)[0]
    label_path = os.path.join(LABEL_DIR, base + ".txt")

    img0 = cv2.imread(img_path)
    if img0 is None:
        print(f"Warning: skipped unreadable image: {img_path}")
        continue

    h0, w0 = img0.shape[:2]

    # ---------- PREPROCESS ----------
    img, scale, pad_x, pad_y = letterbox(img0, INPUT_SIZE)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)

    # ---------- INFERENCE ----------
    start = time.perf_counter()
    outputs = session.run(None, {input_name: img})
    end = time.perf_counter()

    inf_time = end - start
    times.append(inf_time)
    image_names.append(img_name)

    # ---------- POSTPROCESS ----------
    preds = []
    for i in range(3):
        preds.extend(decode(outputs[i], anchors[i], strides[i]))

    scaled = []
    for p in preds:
        x1, y1, x2, y2, conf, cls, obj, cls_score = p

        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        scaled.append([x1, y1, x2, y2, conf, cls])

    scaled = clip_boxes(scaled, w0, h0)
    final_preds = nms(scaled, NMS_IOU_THRESH, class_aware=True)
    gts = load_labels(label_path, w0, h0)

    all_dets.append(final_preds)
    all_gts.append(gts)

    rows, cm = match_image_for_csv_and_confusion(
        final_preds, gts, img_name, inf_time, VAL_IOU_THRESH
    )
    confusion_matrix += cm
    writer.writerows(rows)

detailed_csv_file.close()

# ---------------- RESULTS ----------------
class_metrics, pr_rows, map50 = compute_class_metrics(
    all_dets, all_gts, image_names, VAL_IOU_THRESH
)

avg_time = float(np.mean(times)) if times else 0.0
fps = float(1.0 / avg_time) if avg_time > 0 else 0.0

total_tp = int(sum(m["tp"] for m in class_metrics))
total_fp = int(sum(m["fp"] for m in class_metrics))
total_fn = int(sum(m["fn"] for m in class_metrics))
precision = total_tp / (total_tp + total_fp + 1e-6)
recall = total_tp / (total_tp + total_fn + 1e-6)
f1 = 2 * precision * recall / (precision + recall + 1e-6)

confusion_path = f"{OUTPUT_PREFIX}_confusion_matrix.csv"
class_metrics_path = f"{OUTPUT_PREFIX}_class_metrics.csv"
pr_curve_path = f"{OUTPUT_PREFIX}_pr_curve.csv"
summary_path = f"{OUTPUT_PREFIX}_summary.txt"

save_confusion_matrix(confusion_path, confusion_matrix)
save_class_metrics(class_metrics_path, class_metrics)
save_pr_curve(pr_curve_path, pr_rows)

print("\n===== CPU RESULTS =====")
print("Images:", len(times))
print("Avg inference time:", avg_time)
print("FPS (CPU inference only):", fps)
print(f"Precision@{VAL_IOU_THRESH:.2f}:", precision)
print(f"Recall@{VAL_IOU_THRESH:.2f}:", recall)
print(f"F1@{VAL_IOU_THRESH:.2f}:", f1)
print(f"mAP@{VAL_IOU_THRESH:.2f}:", map50)
print("Saved", detailed_csv_path)
print("Saved", confusion_path)
print("Saved", class_metrics_path)
print("Saved", pr_curve_path)

with open(summary_path, "w") as f:
    f.write("===== CPU RESULTS =====\n")
    f.write(f"Images: {len(times)}\n")
    f.write(f"Avg inference time: {avg_time}\n")
    f.write(f"FPS (CPU inference only): {fps}\n")
    f.write(f"Precision@{VAL_IOU_THRESH:.2f}: {precision}\n")
    f.write(f"Recall@{VAL_IOU_THRESH:.2f}: {recall}\n")
    f.write(f"F1@{VAL_IOU_THRESH:.2f}: {f1}\n")
    f.write(f"mAP@{VAL_IOU_THRESH:.2f}: {map50}\n")
    f.write(f"TP: {total_tp}\n")
    f.write(f"FP: {total_fp}\n")
    f.write(f"FN: {total_fn}\n")
    f.write(f"Validation IoU threshold: {VAL_IOU_THRESH}\n")
    f.write(f"Confidence threshold: {CONF_THRESH}\n")
    f.write(f"NMS IoU threshold: {NMS_IOU_THRESH}\n")

print("Saved", summary_path)
