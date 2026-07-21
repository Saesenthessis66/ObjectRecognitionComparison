# Object Recognition Comparison

This repository contains the pipeline for my master's degree project.

The goal of the project is to compare object recognition performance between:

- a PC CPU/GPU environment
- a Kria KV260 FPGA board using a compiled Vitis AI model

The pipeline covers:

1. Dataset generation
2. Model training
3. Model quantization
4. Compilation for the KV260 Kria FPGA
5. Deployment to the board
6. Benchmarking on PC and KV260

---


## Full Pipeline Overview

The full workflow is:

```text
Generate dataset
      ↓
Train YOLOv5 model
      ↓
Quantize trained model with Vitis AI
      ↓
Compile quantized model for KV260
      ↓
Copy compiled model to the KV260 board
      ↓
Run benchmark on PC
      ↓
Run benchmark on KV260
      ↓
Compare results
```

---

## 1. Generate Dataset and Train Model

Run these commands from the repository root.

```bash
docker compose down --remove-orphans
docker compose run --rm clean -d -m
docker compose run --build generator
docker compose run --build trainer
```

The trained model is expected to be saved here:

```text
source/yolov5/runs/train/yolov5n_640_noaug/weights/best.pt
```

---

## 2. Start the Vitis AI Docker Container

Run this command from the repository root, before entering the `source` directory.

```bash
docker run -it \
  -v $(pwd)/source:/workspace/source \
  xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106
```

Inside the container, go to the YOLOv5 directory:

```bash
cd /workspace/source/yolov5
```

---

## 3. Prepare YOLOv5 for Quantization

Before quantization, the YOLOv5 `forward` method must be changed. It is located in 
```text
source/yolov5/models/yolo.py
```

---

## 4. Quantize the Model

### Calibration

Run calibration mode first:

```bash
python quant.py \
  --quant_mode calib \
  --weights runs/train/yolov5n_640_noaug/weights/best.pt \
  --img_dir /workspace/source/workspace/calib_data/images
```

This uses calibration images from:

```text
/workspace/source/workspace/calib_data/images
```

### Test Quantized Model

After calibration, run test mode:

```bash
python quant.py \
  --quant_mode test \
  --weights runs/train/yolov5n_640_noaug/weights/best.pt \
  --img_dir /workspace/source/workspace/calib_data/images
```

The quantized model should be generated here:

```text
source/yolov5/quantize_result/DetectionModel_int.xmodel
```

## 5. Compile the Model for KV260

Compile the quantized model using `vai_c_xir`:

```bash
vai_c_xir \
  -x quantize_result/DetectionModel_int.xmodel \
  -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
  -o compiled_model \
  -n yolov5_kv260
```

This creates the compiled model:

```text
compiled_model/yolov5_kv260.xmodel
```

This model is compiled for the KV260 DPU architecture.

---

## 6. Copy Files to the KV260 Board

Copy the compiled model to the KV260 board.

Replace the IP address with the IP address of your board.

```bash
scp -O compiled_model/yolov5_kv260.xmodel root@{kria_IP}:/home/root/
```

Copy the test directory:

```bash
scp -O -r test root@{kria_IP}:/home/root/
```

Copy benchmark results back from the board:

```bash
scp -O root@{kria_IP}:/home/root/result.png .
```

```bash
KV260_IP=10.1.1.101

scp -O compiled_model/yolov5_kv260.xmodel root@$KV260_IP:/home/root/
scp -O -r test root@$KV260_IP:/home/root/
scp -O root@$KV260_IP:/home/root/result.png .
```

---

# Running Benchmarks

The project contains benchmarks for both PC and KV260.

---

## 8. PC Benchmark Environment

Start an NVIDIA CUDA Docker container:

```bash
docker run -it \
  --gpus all \
  -v $(pwd):/workspace \
  -w /workspace \
  nvcr.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
```

Install required packages inside the container:

```bash
apt update && apt install -y python3 python3-pip libgl1
```

Install Python dependencies:

```bash
pip install \
  numpy==1.26.4 \
  onnxruntime-gpu==1.18.0 \
  opencv-python-headless==4.8.1.78 \
  pandas \
  matplotlib
```
Before running data vizualization run DPU benchmark on KV260 and copy results to main directory.

Run benchmark and data visualization:
```bash
python benchmark_cpu.py
python benchmark_gpu.py
python plot.py
```

---

## 9. KV260 Camera Setup

Log in to the KV260 board and go to the home directory:

```bash
cd /home/root
```

Unload any currently running application:

```bash
xmutil unloadapp || true
```

Load the KV260 smart camera application:

```bash
xmutil loadapp kv260-smartcam
```

Check that the application is loaded:

```bash
xmutil listapps
```

Wait for the camera pipeline to initialize:

```bash
sleep 5
```

Dump media device information for debugging:

```bash
media-ctl -p -d /dev/media0 > /tmp/media0.txt 2>&1 || true
```

---

## 10. Record Test Video on KV260

### Standard Recording

Remove any previous video file:

```bash
rm -f /home/root/clip.mp4
```

Record video from the camera:

```bash
gst-launch-1.0 -e -v \
  mediasrcbin media-device=/dev/media0 v4l2src0::io-mode=4 ! \
  "video/x-raw,width=1920,height=1080,format=NV12,framerate=30/1" ! \
  queue ! \
  omxh264enc ! \
  h264parse ! \
  mp4mux ! \
  filesink location=/home/root/clip.mp4
```

Check that the video was created:

```bash
ls -lh /home/root/clip.mp4
```

### High-Quality Recording

Remove any previous high-quality video file:

```bash
rm -f /home/root/clip_hq25.mp4
```

Record high-quality video:

```bash
gst-launch-1.0 -e -v \
  mediasrcbin media-device=/dev/media0 v4l2src0::io-mode=4 ! \
  "video/x-raw,width=1920,height=1080,format=NV12,framerate=30/1" ! \
  queue ! \
  omxh264enc target-bitrate=25000 control-rate=low-latency qp-mode=auto gop-mode=basic gop-length=60 b-frames=0 num-slices=8 ! \
  h264parse ! \
  mp4mux ! \
  filesink location=/home/root/clip_hq25.mp4
```

Check that the video was created:

```bash
ls -lh /home/root/clip_hq25.mp4
```

---

## 11. Run KV260 Video Benchmark

Run the benchmark script:

```bash
python3 video_benchmark.py \
  --model yolov5_kv260.xmodel \
  --frames-dir video_frames \
  --source-fps 5 \
  --conf-thresh 0.1 \
  --obj-thresh 0.1 \
  --save-hit-images
```

Arguments:

| Argument | Description |
|---|---|
| `--model` | Compiled `.xmodel` file for KV260 |
| `--frames-dir` | Directory where extracted frames are stored |
| `--source-fps` | FPS used when sampling frames from the source video |
| `--conf-thresh` | Confidence threshold |
| `--obj-thresh` | Objectness threshold |
| `--save-hit-images` | Saves images where objects were detected |

---



## Notes

- Run Docker commands from the repository root unless stated otherwise.
- Run Vitis AI quantization and compilation commands inside the Vitis AI Docker container.
- Run KV260 benchmark commands directly on the KV260 board.
- Replace all example IP addresses with the actual KV260 board IP address.
- Keep the quantization calibration dataset separate from the training and validation datasets.
