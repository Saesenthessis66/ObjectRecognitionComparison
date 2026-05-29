#!/bin/bash
# usage: ./clean.sh -d -m
# -d clears dataset
# -m clears model
DATA_DIR="/workspace"
MODEL_DIR="/app/ModelGenerator"
YOLO_DIR="/app/yolov5"

CLEAR_DATA=false
CLEAR_MODEL=false

while getopts "dm" opt; do
  case $opt in
    d) CLEAR_DATA=true ;;
    m) CLEAR_MODEL=true ;;
    *) 
       echo "Usage: $0 -d (clears dataset) -m (clears model)"
       exit 1 ;;
  esac
done

if [ "$CLEAR_DATA" = false ] && [ "$CLEAR_MODEL" = false ]; then
    echo "Usage: $0 -d (clears dataset) -m (clears model)"
    exit 1
fi

if [ "$CLEAR_DATA" = true ]; then
    echo "Clearing dataset in $DATA_DIR ..."
    rm -rf "${DATA_DIR}/preview"/*
    rm -rf "${DATA_DIR}/calib_data"/*
fi

if [ "$CLEAR_MODEL" = true ]; then
    echo "Clearing model in $MODEL_DIR ..."
    rm -rf "$MODEL_DIR/eval_results"/* 
    rm -rf "${YOLO_DIR}/runs/"*
    rm -rf "${YOLO_DIR}/quantize_result/"*
    rm -rf "${YOLO_DIR}/compiled_model/"*
fi

echo "Done."