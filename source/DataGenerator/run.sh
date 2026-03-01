#!/bin/bash
set -e

docker run --rm \
  -u $(id -u):$(id -g) \
  -v $(pwd):/app \
  linuxserver/blender  \
  blender --background --python /app/main.py

echo "Dataset generated."