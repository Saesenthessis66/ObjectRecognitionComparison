@echo off

docker run --rm ^
  -v %cd%:/app ^
  linuxserver/blender ^
  --background --python /app/main.py

echo Dataset generated.
pause