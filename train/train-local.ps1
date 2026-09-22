param(
  [int]$MaxMinutes = 30,
  [int]$Epochs = 1000,
  [int]$BatchSize = 1,
  [double]$LearningRate = 0.001
)

$ErrorActionPreference = "Stop"

$TrainDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $TrainDir

Write-Host ""
Write-Host "tiny-imgai local trainer" -ForegroundColor Cyan
Write-Host "256x256 U-Net · laptop mode · batch $BatchSize"
Write-Host ""

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
  throw "Node.js is not installed or not on PATH."
}

$nodeVersion = node --version
Write-Host "Node: $nodeVersion"

if (-not (Test-Path ".\node_modules")) {
  Write-Host "Installing training dependencies..." -ForegroundColor Yellow
  npm install
}

if (-not (Test-Path ".\data\originals")) {
  New-Item -ItemType Directory -Force ".\data\originals" | Out-Null
}
if (-not (Test-Path ".\data\processed")) {
  New-Item -ItemType Directory -Force ".\data\processed" | Out-Null
}

Write-Host "Building dataset manifest..." -ForegroundColor Yellow
node .\build-manifest.js

$manifestPath = Join-Path $TrainDir "data\manifest.json"
if (-not (Test-Path $manifestPath)) {
  throw "Manifest was not created: $manifestPath"
}

$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
if ([int]$manifest.pairCount -le 0) {
  throw "No matched PDF pairs found. Put matching PDFs in data\originals and data\processed."
}

Write-Host ("Matched pairs: {0}" -f $manifest.pairCount) -ForegroundColor Green
Write-Host "Starting local training. Checkpoints are resumable." -ForegroundColor Green
Write-Host ""

$env:DATA_DIR = Join-Path $TrainDir "data"
$env:CHECKPOINT_DIR = Join-Path $TrainDir "checkpoints\model"
$env:INPUT_SIZE = "256"
$env:BATCH_SIZE = "$BatchSize"
$env:MAX_MINUTES = "$MaxMinutes"
$env:EPOCHS = "$Epochs"
$env:LEARNING_RATE = "$LearningRate"

node .\train.js

$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
  throw "Training exited with code $exitCode."
}

Write-Host ""
Write-Host "Training run finished. Checkpoint:" -ForegroundColor Green
Write-Host (Join-Path $TrainDir "checkpoints\model")
Write-Host ""
