docker run --rm `
  -v ${PWD}:/app `
  linuxserver/blender  `
  blender --background --python /app/main.py

Write-Host "Dataset generated."