# ObjectRecognitionComparison
My master's degree project. 
Comparison between object recognition using Kria FPGA system and PC CPU.

Run pipeline:

docker compose down --remove-orphans
docker compose run --rm clean -d -m
docker compose run --build generator
docker compose run --build trainer

docker run -it \
  -v $(pwd)/source:/workspace/source \
  xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106

# SWAP FORWARD METHOD IN YOLO.PY FOR QUANTIZATION!!!

python quant.py \
  --quant_mode calib \
  --weights runs/train/yolov5n_640_noaug-aug11/weights/best.pt \
  --img_dir /workspace/source/workspace/calib_data/images

python quant.py \
  --quant_mode test \
  --weights runs/train/yolov5n_640_noaug-aug11/weights/best.pt \
  --img_dir  /workspace/source/workspace/calib_data/images

vai_c_xir \
  -x quantize_result/DetectionModel_int.xmodel \
  -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
  -o compiled_model \
  -n yolov5_kv260

scp -O compiled_model/yolov5_kv260.xmodel root@10.1.1.101:/home/root/
scp -O -r test root@10.1.1.102:/home/root/
scp -O root@10.1.1.106:/home/root/result.png .

Can't add new modules after the interpreter has been initialized

docker run -it \
    --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    nvcr.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

apt update && apt install -y python3 python3-pip libgl1

    pip install \
  numpy==1.26.4 \
  onnxruntime-gpu==1.18.0 \
  opencv-python-headless==4.8.1.78 \
  pandas \
  matplotlib


  cd /home/root

xmutil unloadapp || true
xmutil loadapp kv260-smartcam
xmutil listapps

sleep 5

media-ctl -p -d /dev/media0 > /tmp/media0.txt 2>&1 || true

rm -f /home/root/clip.mp4

gst-launch-1.0 -e -v \
  mediasrcbin media-device=/dev/media0 v4l2src0::io-mode=4 ! \
  "video/x-raw,width=1920,height=1080,format=NV12,framerate=30/1" ! \
  queue ! \
  omxh264enc ! \
  h264parse ! \
  mp4mux ! \
  filesink location=/home/root/clip.mp4

ls -lh /home/root/clip.mp4

rm -f /home/root/clip_hq25.mp4

gst-launch-1.0 -e -v \
  mediasrcbin media-device=/dev/media0 v4l2src0::io-mode=4 ! \
  "video/x-raw,width=1920,height=1080,format=NV12,framerate=30/1" ! \
  queue ! \
  omxh264enc target-bitrate=25000 control-rate=low-latency qp-mode=auto gop-mode=basic gop-length=60 b-frames=0 num-slices=8 ! \
  h264parse ! \
  mp4mux ! \
  filesink location=/home/root/clip_hq25.mp4

ls -lh /home/root/clip_hq25.mp4


python3 video_benchmark.py \
  --model yolov5_kv260.xmodel \
  --frames-dir video_frames \
  --source-fps 5 \
  --conf-thresh 0.1 \
  --obj-thresh 0.1 \
  --save-hit-images