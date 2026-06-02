import os
import time
import csv
import cv2
import numpy as np
import xir
import vart

# ---------------- CONFIG ----------------
MODEL_PATH = "yolov5_kv260.xmodel"
IMAGE_DIR = "test/images"
LABEL_DIR = "test/labels"

INPUT_SIZE = 640
CONF_THRESH = 0.25
IOU_THRESH = 0.5

INPUT_FIX = 7
OUTPUT_FIX = 3

NUM_CLASSES = 4

anchors = [
    [(10,13), (16,30), (33,23)],
    [(30,61), (62,45), (59,119)],
    [(116,90), (156,198), (373,326)]
]

strides = [8, 16, 32]

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

def nms(boxes, iou_thresh=0.45):
    if len(boxes) == 0:
        return []

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

        conf = obj * np.max(cls, axis=-1)

        mask = conf > CONF_THRESH

        for i in range(h):
            for j in range(w):
                if mask[i, j]:
                    x1 = x[i,j] - w_box[i,j]/2
                    y1 = y[i,j] - h_box[i,j]/2
                    x2 = x[i,j] + w_box[i,j]/2
                    y2 = y[i,j] + h_box[i,j]/2

                    boxes.append([x1, y1, x2, y2, conf[i,j]])

    return boxes

# ---------------- LOAD DPU ----------------
graph = xir.Graph.deserialize(MODEL_PATH)
subgraphs = graph.get_root_subgraph().toposort_child_subgraph()
dpu_subgraph = [sg for sg in subgraphs if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU"][0]

runner = vart.Runner.create_runner(dpu_subgraph, "run")
output_tensors = runner.get_output_tensors()

# ---------------- BENCHMARK ----------------
times = []
all_dets = []
all_gts = []
rows = []

# warmup
dummy = np.zeros((1, 640, 640, 3), dtype=np.int8)
for _ in range(10):
    out = [np.empty(tuple(t.dims), dtype=np.int8) for t in output_tensors]
    jid = runner.execute_async([dummy], out)
    runner.wait(jid)

for img_name in os.listdir(IMAGE_DIR):
    img_path = os.path.join(IMAGE_DIR, img_name)
    label_path = os.path.join(LABEL_DIR, img_name.replace(".png", ".txt"))

    img0 = cv2.imread(img_path)
    h0, w0 = img0.shape[:2]

    # ---------- PREPROCESS (NOT TIMED) ----------
    img, scale, pad_x, pad_y = letterbox(img0, INPUT_SIZE)
    img = img.astype(np.float32) / 255.0
    img = (img * (2 ** INPUT_FIX)).astype(np.int8)
    img = np.expand_dims(img, 0)

    input_data = [np.ascontiguousarray(img)]
    output_data = [np.empty(tuple(t.dims), dtype=np.int8) for t in output_tensors]

    # ---------- DPU INFERENCE ONLY ----------
    start = time.perf_counter()
    jid = runner.execute_async(input_data, output_data)
    runner.wait(jid)
    end = time.perf_counter()

    times.append(end - start)

    # ---------- POSTPROCESS ----------
    outputs = [o.astype(np.float32) / (2 ** OUTPUT_FIX) for o in output_data]

    preds = []
    for i in range(3):
        preds.extend(decode(outputs[i], anchors[i], strides[i]))

    # scale back
    scaled = []
    for p in preds:
        x1, y1, x2, y2, conf = p

        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        scaled.append([x1, y1, x2, y2, conf])

    # ✅ CRITICAL FIX
    final_preds = nms(scaled, 0.45)

    gts = load_labels(label_path, w0, h0)

    all_dets.append(final_preds)
    all_gts.append(gts)

    rows.append([img_name, end-start])

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

print("\n===== DPU RESULTS (FIXED) =====")
print("Images:", len(times))
print("Avg inference time:", avg_time)
print("FPS:", fps)
print("mAP@0.5:", map50)

# ---------------- CSV ----------------
with open("benchmark_dpu.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "inference_time"])
    writer.writerows(rows)

print("Saved benchmark_dpu.csv")
