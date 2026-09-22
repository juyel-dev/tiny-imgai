param(
  [int]$MaxMinutes = 30,
  [int]$Epochs = 1,
  [int]$MaxPages = 100,
  [int]$BatchSize = 1,
  [double]$LearningRate = 0.001
)

$ErrorActionPreference = "Stop"

$TrainDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $TrainDir

Write-Host ""
Write-Host "tiny-imgai Python local trainer" -ForegroundColor Cyan
Write-Host "256x256 U-Net | CPU | batch $BatchSize"
Write-Host ""

$Python = Join-Path $TrainDir ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
  if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python is not installed or not on PATH."
  }

  Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
  python -m venv .venv
  $venvExit = $LASTEXITCODE
  if ($venvExit -ne 0) {
    throw "Could not create Python virtual environment."
  }
}

Write-Host "Python:" -NoNewline
& $Python --version

Write-Host "Installing/updating Python dependencies..." -ForegroundColor Yellow
& $Python -m pip install --upgrade pip

& $Python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) {
  throw "PyTorch CPU installation failed."
}

& $Python -m pip install -r .\requirements-python.txt
if ($LASTEXITCODE -ne 0) {
  throw "Python dependency installation failed."
}

New-Item -ItemType Directory -Force ".\data\originals" | Out-Null
New-Item -ItemType Directory -Force ".\data\processed" | Out-Null

$Args = @(
  ".\train.py",
  "--max-minutes", "$MaxMinutes",
  "--epochs", "$Epochs",
  "--max-pages", "$MaxPages",
  "--batch-size", "$BatchSize",
  "--learning-rate", "$LearningRate"
)

Write-Host ""
Write-Host "Starting Python training..." -ForegroundColor Green
Write-Host "Selected pages: $MaxPages (0 = all)"
Write-Host "Epochs: $Epochs"
Write-Host "Time budget: $MaxMinutes minutes"
Write-Host ""

& $Python @Args

if ($LASTEXITCODE -ne 0) {
  throw "Python training exited with code $LASTEXITCODE."
}
