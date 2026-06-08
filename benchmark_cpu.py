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
IOU_THRESH = 0.4

NUM_CLASSES = 4

anchors = [
    [(10,13), (16,30), (33,23)],
    [(30,61), (62,45), (59,119)],
    [(116,90), (156,198), (373,326)]
]

strides = [8, 16, 32]

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
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])

    return inter / (area_a + area_b - inter + 1e-6)

def nms(boxes, iou_thresh=0.4):
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []

    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes if iou(best, b) < iou_thresh]

    return keep

def letterbox(img, new_size=640):
    h, w = img.shape[:2]
    scale = min(new_size / h, new_size / w)

    nh, nw = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (nw, nh))

    canvas = np.full((new_size, new_size, 3), 114, dtype=np.uint8)

    top = (new_size - nh) // 2
    left = (new_size - nw) // 2

    canvas[top:top+nh, left:left+nw] = img_resized
    return canvas, scale, left, top

def load_labels(path, w, h):
    boxes = []
    if not os.path.exists(path):
        return boxes

    with open(path) as f:
        for line in f:
            cls, x, y, bw, bh = map(float, line.split())

            x1 = (x - bw/2) * w
            y1 = (y - bh/2) * h
            x2 = (x + bw/2) * w
            y2 = (y + bh/2) * h

            boxes.append([x1, y1, x2, y2, int(cls)])

    return boxes

# ---------------- DECODE ----------------
def decode(output, anchors, stride):
    output = np.transpose(output, (0, 2, 3, 1))

    bs, h, w, c = output.shape
    output = output.reshape(bs, h, w, 3, 5 + NUM_CLASSES)

    grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

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

        mask = (obj > 0.3) & (cls_score > 0.25)

        for i in range(h):
            for j in range(w):
                if mask[i, j]:
                    boxes.append([
                        x[i,j] - w_box[i,j]/2,
                        y[i,j] - h_box[i,j]/2,
                        x[i,j] + w_box[i,j]/2,
                        y[i,j] + h_box[i,j]/2,
                        obj[i,j],
                        cls_id[i,j],
                        cls_score[i,j]
                    ])

    return boxes

# ---------------- BENCHMARK ----------------
times = []
all_dets = []
all_gts = []

csv_file = open("benchmark_cpu_detailed.csv", "w", newline="")
writer = csv.writer(csv_file)

writer.writerow([
    "image",
    "pred_class",
    "gt_class",
    "confidence",
    "iou",
    "tp",
    "fp",
    "inference_time"
])

# warmup
dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)
for _ in range(10):
    _ = session.run(None, {input_name: dummy})

for img_name in os.listdir(IMAGE_DIR):
    img_path = os.path.join(IMAGE_DIR, img_name)
    base = os.path.splitext(img_name)[0]
    label_path = os.path.join(LABEL_DIR, base + ".txt")

    img0 = cv2.imread(img_path)
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

    # ---------- POSTPROCESS ----------
    preds = []
    for i in range(3):
        preds.extend(decode(outputs[i], anchors[i], strides[i]))

    scaled = []
    for p in preds:
        x1,y1,x2,y2,obj,cls,cls_score = p

        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        scaled.append([x1,y1,x2,y2,obj,cls,cls_score])

    nms_out = nms(scaled, 0.4)

    final_preds = []
    for p in nms_out:
        x1,y1,x2,y2,obj,cls,cls_score = p
        conf = obj * cls_score

        if conf > 0.25:
            final_preds.append([x1,y1,x2,y2,conf,cls])

    gts = load_labels(label_path, w0, h0)

    all_dets.append(final_preds)
    all_gts.append(gts)

    # ---------- CSV LOGGING ----------
    matched = set()

    for det in final_preds:
        x1,y1,x2,y2,conf,cls = det

        best_iou = 0
        best_gt_cls = -1
        best_idx = -1

        for i, g in enumerate(gts):
            if int(cls) != int(g[4]):
                continue

            iou_val = iou(det, g)
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt_cls = g[4]
                best_idx = i

        if best_iou > IOU_THRESH:
            tp = 1
            fp = 0
            matched.add(best_idx)
        else:
            tp = 0
            fp = 1

        writer.writerow([
            img_name,
            int(cls),
            int(best_gt_cls) if best_gt_cls != -1 else -1,
            float(conf),
            float(best_iou),
            tp,
            fp,
            inf_time
        ])

    # log missed GTs (FN)
    for i, g in enumerate(gts):
        if i not in matched:
            writer.writerow([
                img_name,
                -1,
                int(g[4]),
                0.0,
                0.0,
                0,
                1,
                inf_time
            ])

csv_file.close()

# ---------------- mAP ----------------
def compute_map(dets, gts):
    tp, fp = [], []
    total_gt = sum(len(x) for x in gts)

    for dlist, glist in zip(dets, gts):
        matched = set()

        for d in sorted(dlist, key=lambda x: x[4], reverse=True):
            best_iou = 0
            best_idx = -1

            for i, g in enumerate(glist):
                if int(d[5]) != int(g[4]):
                    continue

                iou_val = iou(d, g)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = i

            if best_iou > IOU_THRESH and best_idx not in matched:
                tp.append(1)
                fp.append(0)
                matched.add(best_idx)
            else:
                tp.append(0)
                fp.append(1)

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    recall = tp / (total_gt + 1e-6)
    precision = tp / (tp + fp + 1e-6)

    return np.trapz(precision, recall)

map50 = compute_map(all_dets, all_gts)

# ---------------- RESULTS ----------------
avg_time = np.mean(times)
fps = 1 / avg_time

print("\n===== CPU RESULTS =====")
print("Images:", len(times))
print("Avg inference time:", avg_time)
print("FPS (CPU only):", fps)
print("mAP@0.5:", map50)

print("Saved benchmark_cpu_detailed.csv")