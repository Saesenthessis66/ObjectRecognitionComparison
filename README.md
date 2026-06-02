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
  -v $(pwd)/calib_data:/workspace/calib_data \
  xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106

# SWAP FORWARD METHOD IN YOLO.PY FOR QUANTIZATION!!!

python quant.py \
  --quant_mode calib \
  --weights runs/train/yolov5n_640_noaug/weights/best.pt \
  --img_dir /workspace/source/workspace/calib_data/images

python quant.py \
  --quant_mode test \
  --weights runs/train/yolov5n_640_noaug/weights/best.pt \
  --img_dir  /workspace/source/workspace/calib_data/images

vai_c_xir \
  -x quantize_result/DetectionModel_int.xmodel \
  -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
  -o compiled_model \
  -n yolov5_kv260

scp -O compiled_model/yolov5_kv260.xmodel root@10.1.1.106:/home/root/
scp -O test.png root@10.1.1.106:/home/root/
scp -O root@10.1.1.106:/home/root/result.png .

Can't add new modules after the interpreter has been initialized