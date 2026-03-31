param(
    [switch]$d,
    [switch]$m
)

$DATA_DIR = ".\workspace\preview"
$MODEL_DIR = ".\ModelGenerator"

if (-not ($d -or $m)) {
    Write-Host "Usage: .\clean.ps1 -d (clears dataset) -m (clears model)"
    exit
}

if ($d) {
    Remove-Item "$DATA_DIR\*" -Recurse -Force -ErrorAction SilentlyContinue
}

if ($m) {
    Remove-Item "$MODEL_DIR\runs\*" -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item "$MODEL_DIR\yolo*" -Recurse -Force -ErrorAction SilentlyContinue
}