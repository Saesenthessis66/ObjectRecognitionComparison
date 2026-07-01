import os
import csv
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import xir
import vart


ANCHORS = [
    [(10, 13), (16, 30), (33, 23)],
    [(30, 61), (62, 45), (59, 119)],
    [(116, 90), (156, 198), (373, 326)],
]

STRIDES = [8, 16, 32]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


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
    resized = cv2.resize(img, (nw, nh))

    canvas = np.full((new_size, new_size, 3), 114, dtype=np.uint8)

    top = (new_size - nh) // 2
    left = (new_size - nw) // 2

    canvas[top:top + nh, left:left + nw] = resized
    return canvas, scale, left, top


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


def decode(output, anchors, stride, num_classes, obj_thresh, conf_thresh):
    bs, h, w, c = output.shape
    expected_c = 3 * (5 + num_classes)

    if c != expected_c:
        raise RuntimeError(
            f"Output shape mismatch. Tensor has {c} channels, expected {expected_c}. "
            f"Check --num-classes. Current value: {num_classes}"
        )

    output = output.reshape(bs, h, w, 3, 5 + num_classes)

    grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    boxes = []

    for a in range(3):
        pred = output[0, :, :, a, :]

        x = (sigmoid(pred[..., 0]) * 2.0 - 0.5 + grid_x) * stride
        y = (sigmoid(pred[..., 1]) * 2.0 - 0.5 + grid_y) * stride

        w_box = (sigmoid(pred[..., 2]) * 2.0) ** 2 * anchors[a][0]
        h_box = (sigmoid(pred[..., 3]) * 2.0) ** 2 * anchors[a][1]

        obj = sigmoid(pred[..., 4])
        cls = sigmoid(pred[..., 5:])

        cls_score = np.max(cls, axis=-1)
        cls_id = np.argmax(cls, axis=-1)
        conf = obj * cls_score

        mask = (obj > obj_thresh) & (conf > conf_thresh)

        ys, xs = np.where(mask)
        for i, j in zip(ys, xs):
            boxes.append([
                x[i, j] - w_box[i, j] / 2.0,
                y[i, j] - h_box[i, j] / 2.0,
                x[i, j] + w_box[i, j] / 2.0,
                y[i, j] + h_box[i, j] / 2.0,
                float(conf[i, j]),
                int(cls_id[i, j]),
                float(obj[i, j]),
                float(cls_score[i, j]),
            ])

    return boxes


def get_dpu_subgraph(graph):
    root = graph.get_root_subgraph()
    subgraphs = root.toposort_child_subgraph()

    dpu_subgraphs = [
        sg for sg in subgraphs
        if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU"
    ]

    if not dpu_subgraphs:
        raise RuntimeError("No DPU subgraph found in xmodel.")

    return dpu_subgraphs[0]


def parse_class_names(class_names_arg, num_classes):
    if class_names_arg.strip():
        names = [x.strip() for x in class_names_arg.split(",")]
        if len(names) != num_classes:
            raise RuntimeError(
                f"--class-names has {len(names)} names, but --num-classes is {num_classes}"
            )
        return names

    return [f"class_{i}" for i in range(num_classes)]


def run_dpu_frame(
    frame,
    runner,
    input_tensor,
    output_tensors,
    input_size,
    num_classes,
    obj_thresh,
    conf_thresh,
    nms_iou_thresh,
):
    h0, w0 = frame.shape[:2]

    img, scale, pad_x, pad_y = letterbox(frame, input_size)

    input_fix = input_tensor.get_attr("fix_point")
    img = img.astype(np.float32) / 255.0
    img = (img * (2 ** input_fix)).astype(np.int8)
    img = np.expand_dims(img, axis=0)

    input_data = [np.ascontiguousarray(img)]
    output_data = [np.empty(tuple(t.dims), dtype=np.int8) for t in output_tensors]

    t0 = time.perf_counter()
    jid = runner.execute_async(input_data, output_data)
    runner.wait(jid)
    t1 = time.perf_counter()

    inference_time = t1 - t0

    outputs = []
    for i, tensor in enumerate(output_tensors):
        fix = tensor.get_attr("fix_point")
        out = output_data[i].astype(np.float32) / (2 ** fix)
        outputs.append(out)

    outputs = sorted(outputs, key=lambda x: x.shape[1], reverse=True)

    preds = []
    for out, anchors, stride in zip(outputs, ANCHORS, STRIDES):
        preds.extend(
            decode(
                out,
                anchors,
                stride,
                num_classes,
                obj_thresh,
                conf_thresh,
            )
        )

    scaled = []
    for p in preds:
        x1, y1, x2, y2, conf, cls, obj, cls_score = p

        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        scaled.append([x1, y1, x2, y2, conf, cls, obj, cls_score])

    scaled = clip_boxes(scaled, w0, h0)
    final_preds = nms(scaled, nms_iou_thresh, class_aware=True)

    return final_preds, inference_time


def draw_detections(frame, detections, class_names):
    out = frame.copy()

    for det in detections:
        x1, y1, x2, y2, conf, cls, obj, cls_score = det
        cls = int(cls)

        class_name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
        label = f"{class_name} {conf:.3f}"

        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))

        cv2.rectangle(out, p1, p2, (0, 255, 0), 2)

        text_y = max(20, p1[1] - 8)
        cv2.putText(
            out,
            label,
            (p1[0], text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="yolov5_kv260.xmodel")
    parser.add_argument("--frames-dir", default="video_frames")
    parser.add_argument("--out-dir", default="frames_dpu_results")

    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--class-names", default="")

    parser.add_argument("--conf-thresh", type=float, default=0.15)
    parser.add_argument("--obj-thresh", type=float, default=0.15)
    parser.add_argument("--nms-iou-thresh", type=float, default=0.40)

    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--source-fps", type=float, default=25.0)

    parser.add_argument("--save-hit-images", action="store_true")
    parser.add_argument("--max-hit-images", type=int, default=100)
    parser.add_argument("--stop-on-first-hit", action="store_true")

    args = parser.parse_args()

    if args.frame_step < 1:
        raise RuntimeError("--frame-step must be >= 1")

    class_names = parse_class_names(args.class_names, args.num_classes)

    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    hit_dir = out_dir / "hit_images"

    out_dir.mkdir(parents=True, exist_ok=True)
    hit_dir.mkdir(parents=True, exist_ok=True)

    detections_csv_path = out_dir / "detections.csv"
    frames_csv_path = out_dir / "frames.csv"
    summary_path = out_dir / "summary.txt"
    first_hit_path = out_dir / "first_detection.jpg"
    best_hit_path = out_dir / "best_detection.jpg"

    all_files = [
        f for f in sorted(os.listdir(frames_dir))
        if Path(f).suffix.lower() in IMAGE_EXTS
    ]

    if not all_files:
        raise RuntimeError(f"No image files found in {frames_dir}")

    selected = [
        (original_idx, name)
        for original_idx, name in enumerate(all_files)
        if original_idx % args.frame_step == 0
    ]

    print("Loading xmodel:", args.model)

    graph = xir.Graph.deserialize(args.model)
    dpu_subgraph = get_dpu_subgraph(graph)

    runner = vart.Runner.create_runner(dpu_subgraph, "run")
    input_tensor = runner.get_input_tensors()[0]
    output_tensors = runner.get_output_tensors()

    print("Input tensor:", input_tensor.dims)
    print("Output tensors:")
    for t in output_tensors:
        print(" ", t.name, t.dims)

    print("Frames directory:", frames_dir)
    print("Total frames found:", len(all_files))
    print("Frames selected:", len(selected))
    print("Frame step:", args.frame_step)
    print("Confidence threshold:", args.conf_thresh)
    print("Objectness threshold:", args.obj_thresh)

    detections_file = open(detections_csv_path, "w", newline="")
    frames_file = open(frames_csv_path, "w", newline="")

    det_writer = csv.writer(detections_file)
    frame_writer = csv.writer(frames_file)

    det_writer.writerow([
        "frame_file",
        "original_frame_index",
        "processed_frame_index",
        "video_time_sec",
        "inference_time_sec",
        "dpu_fps",
        "class_id",
        "class_name",
        "confidence",
        "objectness",
        "class_score",
        "x1",
        "y1",
        "x2",
        "y2",
        "box_width",
        "box_height",
        "saved_image",
    ])

    frame_writer.writerow([
        "frame_file",
        "original_frame_index",
        "processed_frame_index",
        "video_time_sec",
        "inference_time_sec",
        "dpu_fps",
        "detections",
        "best_class_id",
        "best_class_name",
        "best_confidence",
        "saved_image",
    ])

    processed_count = 0
    hit_frame_count = 0
    total_detection_count = 0
    saved_hit_count = 0

    best_det = None
    best_frame = None
    best_frame_name = ""
    best_original_idx = -1
    best_video_time = 0.0

    first_hit_saved = False
    inference_times = []

    wall_start = time.perf_counter()

    for processed_idx, (original_idx, name) in enumerate(selected):
        img_path = frames_dir / name
        frame = cv2.imread(str(img_path))

        if frame is None:
            print("Skipped unreadable image:", img_path)
            continue

        video_time_sec = original_idx / args.source_fps if args.source_fps > 0 else 0.0

        detections, inf_time = run_dpu_frame(
            frame=frame,
            runner=runner,
            input_tensor=input_tensor,
            output_tensors=output_tensors,
            input_size=args.input_size,
            num_classes=args.num_classes,
            obj_thresh=args.obj_thresh,
            conf_thresh=args.conf_thresh,
            nms_iou_thresh=args.nms_iou_thresh,
        )

        processed_count += 1
        inference_times.append(inf_time)

        dpu_fps = 1.0 / inf_time if inf_time > 0 else 0.0
        saved_image = ""

        best_class_id = -1
        best_class_name = ""
        best_conf = 0.0

        if detections:
            hit_frame_count += 1
            total_detection_count += len(detections)

            best_in_frame = max(detections, key=lambda d: d[4])
            best_class_id = int(best_in_frame[5])
            best_class_name = class_names[best_class_id] if best_class_id < len(class_names) else f"class_{best_class_id}"
            best_conf = float(best_in_frame[4])

            annotated = draw_detections(frame, detections, class_names)

            if not first_hit_saved:
                cv2.imwrite(str(first_hit_path), annotated)
                first_hit_saved = True

            if args.save_hit_images and saved_hit_count < args.max_hit_images:
                saved_image = str(hit_dir / f"{Path(name).stem}_det.jpg")
                cv2.imwrite(saved_image, annotated)
                saved_hit_count += 1

            if best_det is None or best_in_frame[4] > best_det[4]:
                best_det = best_in_frame
                best_frame = frame.copy()
                best_frame_name = name
                best_original_idx = original_idx
                best_video_time = video_time_sec

            print(
                f"HIT frame={original_idx} file={name} "
                f"time={video_time_sec:.3f}s "
                f"detections={len(detections)} "
                f"best={best_class_name} conf={best_conf:.3f}"
            )

        frame_writer.writerow([
            name,
            original_idx,
            processed_idx,
            f"{video_time_sec:.6f}",
            f"{inf_time:.9f}",
            f"{dpu_fps:.3f}",
            len(detections),
            best_class_id,
            best_class_name,
            f"{best_conf:.6f}",
            saved_image,
        ])

        for det in detections:
            x1, y1, x2, y2, conf, cls, obj, cls_score = det
            cls = int(cls)
            class_name = class_names[cls] if cls < len(class_names) else f"class_{cls}"

            det_writer.writerow([
                name,
                original_idx,
                processed_idx,
                f"{video_time_sec:.6f}",
                f"{inf_time:.9f}",
                f"{dpu_fps:.3f}",
                cls,
                class_name,
                f"{conf:.6f}",
                f"{obj:.6f}",
                f"{cls_score:.6f}",
                f"{x1:.2f}",
                f"{y1:.2f}",
                f"{x2:.2f}",
                f"{y2:.2f}",
                f"{x2 - x1:.2f}",
                f"{y2 - y1:.2f}",
                saved_image,
            ])

        if args.stop_on_first_hit and detections:
            break

    wall_end = time.perf_counter()

    detections_file.close()
    frames_file.close()

    if best_det is not None and best_frame is not None:
        best_annotated = draw_detections(best_frame, [best_det], class_names)
        cv2.imwrite(str(best_hit_path), best_annotated)

    avg_inf_time = float(np.mean(inference_times)) if inference_times else 0.0
    avg_dpu_fps = float(1.0 / avg_inf_time) if avg_inf_time > 0 else 0.0
    wall_time = wall_end - wall_start

    print()
    print("===== FRAME DPU SCAN RESULTS =====")
    print("Frames processed:", processed_count)
    print("Frames with detections:", hit_frame_count)
    print("Total detections:", total_detection_count)
    print("Avg DPU inference time:", avg_inf_time)
    print("Avg DPU FPS:", avg_dpu_fps)
    print("Wall time:", wall_time)
    print("Frames CSV:", frames_csv_path)
    print("Detections CSV:", detections_csv_path)
    print("Summary:", summary_path)

    if first_hit_saved:
        print("First detection image:", first_hit_path)

    if best_det is not None:
        print("Best detection image:", best_hit_path)
        print("Best detection frame:", best_frame_name)
        print("Best original frame index:", best_original_idx)
        print("Best video time sec:", best_video_time)
        print("Best confidence:", float(best_det[4]))
    else:
        print("No detections found.")

    with open(summary_path, "w") as f:
        f.write("===== FRAME DPU SCAN RESULTS =====\n")
        f.write(f"Frames directory: {frames_dir}\n")
        f.write(f"Total frames found: {len(all_files)}\n")
        f.write(f"Frames processed: {processed_count}\n")
        f.write(f"Frame step: {args.frame_step}\n")
        f.write(f"Source FPS used for timestamps: {args.source_fps}\n")
        f.write(f"Confidence threshold: {args.conf_thresh}\n")
        f.write(f"Objectness threshold: {args.obj_thresh}\n")
        f.write(f"NMS IoU threshold: {args.nms_iou_thresh}\n")
        f.write(f"Frames with detections: {hit_frame_count}\n")
        f.write(f"Total detections: {total_detection_count}\n")
        f.write(f"Avg DPU inference time: {avg_inf_time}\n")
        f.write(f"Avg DPU FPS: {avg_dpu_fps}\n")
        f.write(f"Wall time: {wall_time}\n")
        f.write(f"Frames CSV: {frames_csv_path}\n")
        f.write(f"Detections CSV: {detections_csv_path}\n")

        if first_hit_saved:
            f.write(f"First detection image: {first_hit_path}\n")

        if best_det is not None:
            f.write(f"Best detection image: {best_hit_path}\n")
            f.write(f"Best detection frame: {best_frame_name}\n")
            f.write(f"Best original frame index: {best_original_idx}\n")
            f.write(f"Best video time sec: {best_video_time}\n")
            f.write(f"Best confidence: {float(best_det[4])}\n")
        else:
            f.write("No detections found.\n")


if __name__ == "__main__":
    main()