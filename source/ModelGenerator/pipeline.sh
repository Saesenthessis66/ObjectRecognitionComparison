#!/bin/bash
set -e  # stop on error
echo "===== STEP 0: PREPARE CALIBRATION DATA ====="

CALIB_SRC="/workspace/preview/images/train"
CALIB_DST="/workspace/calib_data/images"
CALIB_COUNT=200   # adjust if needed

mkdir -p "$CALIB_DST"

echo "Cleaning old calibration data..."
rm -rf ${CALIB_DST:?}/*

echo "Selecting calibration images..."
ls "$CALIB_SRC"/*.png | head -n $CALIB_COUNT | while read img; do
    cp "$img" "$CALIB_DST/"
done

echo "Calibration dataset size:"
ls "$CALIB_DST" | wc -l

echo "===== STEP 1: TRAIN ====="
cd /app/yolov5

python train.py \
  --img 320 \
  --batch 128 \
  --epochs 140 \
  --data /workspace/preview/data.yaml \
  --cfg models/yolov5n.yaml \
  --weights yolov5n.pt \
  --project runs/train \
  --hyp /app/ModelGenerator/hyp.yaml \
  --save-period 1 \
  --name yolov5n_320_noaug \
  --exist-ok

echo "===== STEP 2: EVALUATE ====="
cd /app/ModelGenerator

python main.py

echo "===== PIPELINE DONE ====="