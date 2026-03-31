# ObjectRecognitionComparison
My master's degree project. 
Comparison between object recognition using Kria FPGA system and PC CPU.

Run pipeline:

docker compose down --remove-orphans
.\clear.ps1 -d
.\clear.ps1 -m
docker compose run -- build generator
docker compose run -- build trainer