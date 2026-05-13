import cv2
import numpy as np
import xir
import vart
import time

MODEL_PATH = "yolov5_kv260.xmodel"
IMAGE_PATH = "test.png"
OUTPUT_PATH = "result.png"

# -----------------------------
# CONFIG
# -----------------------------
NUM_CLASSES = 4
CONF_THRESH = 0.25
IOU_THRESH = 0.45

names = ["00", "01", "10", "11"]

anchors = [
    [(10,13), (16,30), (33,23)],
    [(30,61), (62,45), (59,119)],
    [(116,90), (156,198), (373,326)]
]

strides = [8, 16, 32]

# -----------------------------
# Utils
# -----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    return inter / (area_a + area_b - inter + 1e-6)


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


# -----------------------------
# CORRECT YOLOv5 (v7) DECODE
# -----------------------------
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


# -----------------------------
# LOAD MODEL
# -----------------------------
print("[DEBUG] Loading graph...")
graph = xir.Graph.deserialize(MODEL_PATH)

subgraphs = graph.get_root_subgraph().toposort_child_subgraph()

dpu_subgraph = None
for i, sg in enumerate(subgraphs):
    device = sg.get_attr("device") if sg.has_attr("device") else "None"
    print(f"[DEBUG] Subgraph {i}, device={device}")
    if device.upper() == "DPU":
        dpu_subgraph = sg

if dpu_subgraph is None:
    raise RuntimeError("No DPU found")

print("[DEBUG] Creating runner...")
runner = vart.Runner.create_runner(dpu_subgraph, "run")

input_tensor = runner.get_input_tensors()[0]
output_tensors = runner.get_output_tensors()

print("[DEBUG] Input shape:", tuple(input_tensor.dims))

# -----------------------------
# LOAD IMAGE (PNG)
# -----------------------------
img0 = cv2.imread(IMAGE_PATH)
if img0 is None:
    raise RuntimeError("Image not found")

h0, w0 = img0.shape[:2]

img = cv2.resize(img0, (320, 320))
img = img.astype(np.float32) / 255.0
img = np.expand_dims(img, axis=0)  # NHWC

print("[DEBUG] Input prepared:", img.shape)

# -----------------------------
# RUN INFERENCE
# -----------------------------
input_data = [np.ascontiguousarray(img)]

output_data = []
for t in output_tensors:
    output_data.append(np.empty(tuple(t.dims), dtype=np.float32))

print("[DEBUG] Running inference...")
start = time.time()

job_id = runner.execute_async(input_data, output_data)
runner.wait(job_id)

print(f"[DEBUG] Inference time: {time.time() - start:.4f} sec")

# -----------------------------
# DECODE
# -----------------------------
decoded = []

for i, out in enumerate(output_data):
    print(f"[DEBUG] Decoding output {i}")
    decoded.extend(decode_yolo(out, anchors[i], strides[i]))

print("[DEBUG] Total boxes:", len(decoded))

final_boxes = nms(decoded)

print("[DEBUG] After NMS:", len(final_boxes))

# -----------------------------
# DRAW RESULTS
# -----------------------------
for b in final_boxes:
    x1, y1, x2, y2, conf, cls = b

    # scale to original image
    x1 = int(x1 / 320 * w0)
    y1 = int(y1 / 320 * h0)
    x2 = int(x2 / 320 * w0)
    y2 = int(y2 / 320 * h0)

    # clamp
    x1 = max(0, min(w0, x1))
    y1 = max(0, min(h0, y1))
    x2 = max(0, min(w0, x2))
    y2 = max(0, min(h0, y2))

    label = f"{names[int(cls)]}:{conf:.2f}"

    cv2.rectangle(img0, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(img0, label, (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

# -----------------------------
# SAVE RESULT
# -----------------------------
cv2.imwrite(OUTPUT_PATH, img0)
print(f"[DEBUG] Result saved to {OUTPUT_PATH}")