import time
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os

# ====== KONFIG ======
MODEL_PATH = "blocks_detector/yolov8n_3208/weights/best.pt"
IMG_SIZE = 320
NUM_RUNS = 200
DEVICE = "cpu"
# ====================

def load_random_image(folder="../DataGenerator/preview/images/test"):
    images = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".png"):
                images.append(os.path.join(root, f))
    img_path = np.random.choice(images)
    img = cv2.imread(img_path)
    return img

def main():
    print("Loading model...")
    model = YOLO(MODEL_PATH)
    model.to(DEVICE)

    # Wymuszenie CPU i wyłączenie half precision
    torch.set_grad_enabled(False)

    print("Preparing input...")
    img = load_random_image()

    # Warmup (bardzo ważne)
    print("Warming up...")
    for _ in range(20):
        model(img, imgsz=IMG_SIZE, verbose=False)

    print("Starting benchmark...")
    start = time.perf_counter()

    for _ in range(NUM_RUNS):
        model(img, imgsz=IMG_SIZE, verbose=False)

    end = time.perf_counter()

    avg_time = (end - start) / NUM_RUNS
    fps = 1 / avg_time

    print("\n===== CPU BENCHMARK RESULTS =====")
    print(f"Average inference time: {avg_time*1000:.3f} ms")
    print(f"FPS: {fps:.2f}")

if __name__ == "__main__":
    main()