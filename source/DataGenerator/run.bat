@echo off

docker run --rm ^
  --gpus all ^
  -v %cd%:/app ^
  linuxserver/blender ^
  blender --background --python /app/main.py
echo Dataset generated.
pause