import os
import time
import glob
import csv
import cv2
import numpy as np
import onnxruntime as ort
import xir
import vart

# configuration settings for dataset, model, and benchmark
IMG_SIZE = 640
DATASET_PATH = "dataset/images/*.png"
LABELS_PATH = "dataset/labels"  # yolo txt format

names = ["00", "01", "10", "11"]

anchors = [
    [(10,13), (16,30), (33,23)],
    [(30,61), (62,45), (59,119)],
    [(116,90), (156,198), (373,326)]
]

strides = [8, 16, 32]

XMODEL_PATH = "model.xmodel"
ONNX_INT8_PATH = "model_int8.onnx"
ONNX_FP32_PATH = "model_fp32.onnx"

N_EPOCHS = 10
IOU_THRESH = 0.5
CONF_THRESH = 0.25

CSV_SUMMARY = "summary.csv"
CSV_RUNS = "runs.csv"

def letterbox(img, new_size=320):
    h, w = img.shape[:2]
    scale = min(new_size / h, new_size / w)

    nh, nw = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (nw, nh))

    canvas = np.full((new_size, new_size, 3), 114, dtype=np.uint8)

    top = (new_size - nh) // 2
    left = (new_size - nw) // 2

    canvas[top:top+nh, left:left+nw] = img_resized

    return canvas, scale, left, top

# load dataset images and labels
def load_dataset():
    images = []
    labels = []

    paths = sorted(glob.glob(DATASET_PATH))

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue

        h, w = img.shape[:2]

        # Apply letterbox (same as training/inference)
        img_lb, scale, pad_x, pad_y = letterbox(img, IMG_SIZE)

        img_r = img_lb.astype(np.float32) / 255.0
        img_r = np.expand_dims(img_r, axis=0)

        images.append(np.ascontiguousarray(img_r))

        # Load labels
        label_file = os.path.join(
            LABELS_PATH,
            os.path.basename(p).replace(".png", ".txt")
        )

        gt = []

        if os.path.exists(label_file):
            with open(label_file) as f:
                for line in f:
                    cls, x, y, w_, h_ = map(float, line.split())

                    x1 = (x - w_ / 2) * w
                    y1 = (y - h_ / 2) * h
                    x2 = (x + w_ / 2) * w
                    y2 = (y + h_ / 2) * h

                    x1 = x1 * scale + pad_x
                    y1 = y1 * scale + pad_y
                    x2 = x2 * scale + pad_x
                    y2 = y2 * scale + pad_y

                    gt.append([x1, y1, x2, y2, int(cls)])

        labels.append(gt)

    return images, labels

# compute intersection over union for two boxes
def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])

    return inter / (area_a + area_b + 1e-6)

# evaluate predictions against ground truth labels
def evaluate(preds, gts):
    TP, FP, FN = 0, 0, 0

    for pred, gt in zip(preds, gts):
        matched = set()

        for p in pred:
            found = False
            for i, g in enumerate(gt):
                if i in matched:
                    continue
                if p[4] == g[4] and iou(p, g) > IOU_THRESH:
                    TP += 1
                    matched.add(i)
                    found = True
                    break
            if not found:
                FP += 1

        FN += len(gt) - len(matched)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)

    return precision, recall


# wrapper for DPU runtime model
class DPUModel:
    def __init__(self, path):
        graph = xir.Graph.deserialize(path)
        subgraphs = graph.get_root_subgraph().toposort_child_subgraph()

        dpu = None
        for sg in subgraphs:
            if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU":
                dpu = sg

        self.runner = vart.Runner.create_runner(dpu, "run")
        self.input_tensor = self.runner.get_input_tensors()[0]
        self.output_tensors = self.runner.get_output_tensors()

    def run(self, img):
        outputs = [np.empty(tuple(t.dims), dtype=np.float32)
                   for t in self.output_tensors]

        job_id = self.runner.execute_async([img], outputs)
        self.runner.wait(job_id)

        return outputs


# wrapper for onnx runtime model
class ONNXModel:
    def __init__(self, path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            path,
            sess_options=so,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def run(self, img):
        return self.session.run(None, {self.input_name: img})

# compute sigmoid activation for a tensor
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# decode one yolo output tensor to bounding boxes
def decode_yolo(output, anchors, stride):
    bs, h, w, c = output.shape
    num_anchors = len(anchors)

    output = output.reshape(bs, h, w, num_anchors, 5 + NUM_CLASSES)

    grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    boxes = []

    for a in range(num_anchors):
        pred = output[0, :, :, a, :]

        # Correct for YOLOv5 v7
        x = (sigmoid(pred[..., 0]) + grid_x) * stride
        y = (sigmoid(pred[..., 1]) + grid_y) * stride

        w_box = np.exp(pred[..., 2]) * anchors[a][0]
        h_box = np.exp(pred[..., 3]) * anchors[a][1]

        obj = sigmoid(pred[..., 4])
        cls = sigmoid(pred[..., 5:])

        conf = obj * np.max(cls, axis=-1)
        cls_id = np.argmax(cls, axis=-1)

        x1 = x - w_box / 2
        y1 = y - h_box / 2
        x2 = x + w_box / 2
        y2 = y + h_box / 2

        mask = conf > CONF_THRESH

        for i in range(h):
            for j in range(w):
                if mask[i, j]:
                    boxes.append([
                        x1[i, j], y1[i, j],
                        x2[i, j], y2[i, j],
                        conf[i, j],
                        cls_id[i, j]
                    ])

    return boxes

# apply non maximum suppression to filter overlapping boxes
def nms(boxes):
    if len(boxes) == 0:
        return []

    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []

    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes if iou(best, b) < IOU_THRESH]

    return keep

# decode raw model outputs into final detections
def decode(outputs):
    decoded = []

    # YOLOv5 has 3 outputs (small, medium, large)
    for i, out in enumerate(outputs):
        decoded.extend(decode_yolo(out, anchors[i], strides[i]))

    # Apply NMS
    final = nms(decoded)

    # Convert to [x1, y1, x2, y2, cls]
    result = []
    for b in final:
        x1, y1, x2, y2, conf, cls = b

        # filter low confidence boxes again for safety
        if conf < CONF_THRESH:
            continue

        result.append([x1, y1, x2, y2, int(cls)])

    return result


# run model benchmark and gather metrics
def benchmark(model, images, labels, name):
    runs = []
    preds_all = []

    # warmup
    for _ in range(5):
        for img in images:
            model.run(img)

    total_runs = len(images) * N_EPOCHS

    start = time.time()

    for epoch in range(N_EPOCHS):
        for i, img in enumerate(images):
            t0 = time.time()

            outputs = model.run(img)

            t1 = time.time()
            latency = t1 - t0

            runs.append([name, epoch, i, latency])

            preds = decode(outputs)
            preds_all.append(preds)

    end = time.time()

    total_time = end - start
    fps = total_runs / total_time
    avg_latency = total_time / total_runs

    precision, recall = evaluate(preds_all, labels * N_EPOCHS)

    return runs, {
        "model": name,
        "total_time": total_time,
        "avg_latency": avg_latency,
        "fps": fps,
        "precision": precision,
        "recall": recall
    }


# save benchmark results to csv files
def save_results(all_runs, summaries):
    with open(CSV_RUNS, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "epoch", "image_id", "latency"])
        writer.writerows(all_runs)

    with open(CSV_SUMMARY, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)


# main execution entrypoint
if __name__ == "__main__":
    images, labels = load_dataset()

    models = [
        ("DPU", DPUModel(XMODEL_PATH)),
        ("CPU_INT8", ONNXModel(ONNX_INT8_PATH)),
        ("CPU_FP32", ONNXModel(ONNX_FP32_PATH)),
    ]

    all_runs = []
    summaries = []

    for name, model in models:
        print("Running:", name)
        runs, summary = benchmark(model, images, labels, name)

        all_runs.extend(runs)
        summaries.append(summary)

    save_results(all_runs, summaries)

    print("Done. Results saved to CSV.")