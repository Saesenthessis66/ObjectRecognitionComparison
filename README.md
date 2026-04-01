# ObjectRecognitionComparison
My master's degree project. 
Comparison between object recognition using Kria FPGA system and PC CPU.

Run pipeline:

docker compose down --remove-orphans
docker compose run --rm clean -d -m
docker compose run --build generator
docker compose run --build trainer