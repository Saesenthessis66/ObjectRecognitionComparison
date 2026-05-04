#!/bin/bash

# Usage: ./clean.sh -d -m
# -d clears dataset
# -m clears model

DATA_DIR="/workspace/preview"
MODEL_DIR="/app/ModelGenerator"

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
    rm -rf "$DATA_DIR"/*
fi

if [ "$CLEAR_MODEL" = true ]; then
    echo "Clearing model in $MODEL_DIR ..."
    rm -rf "$MODEL_DIR/runs"/*
    rm -rf "$MODEL_DIR/yolo*" 
    rm -rf "$MODEL_DIR/eval_results"/* 
fi

echo "Done."