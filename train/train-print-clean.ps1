param(
  [int]$MaxMinutes = 120,
  [int]$Epochs = 3,
  [int]$MaxPages = 0,
  [double]$LearningRate = 0.001
)

$ErrorActionPreference = "Stop"

$TrainDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $TrainDir
$Python = Join-Path $TrainDir ".venv\Scripts\python.exe"
$CheckpointDir = Join-Path $TrainDir "checkpoints\python-model-print-clean"

if (-not (Test-Path $Python)) {
  throw "Python venv not found. Run .\train-python.ps1 once first."
}

& $Python -m pip install -r .\requirements-python.txt
if ($LASTEXITCODE -ne 0) {
  throw "Dependency check failed with code $LASTEXITCODE."
}

$Args = @(
  ".\train.py",
  "--max-minutes", "$MaxMinutes",
  "--epochs", "$Epochs",
  "--max-pages", "$MaxPages",
  "--batch-size", "1",
  "--learning-rate", "$LearningRate",
  "--loss-profile", "print-clean-v1",
  "--checkpoint-dir", "$CheckpointDir"
)

Write-Host ""
Write-Host "tiny-imgai print-clean training" -ForegroundColor Cyan
Write-Host "256x256 U-Net | CPU | batch 1 | print-clean-v1"
Write-Host "Pages: $MaxPages (0 = all) | Epochs: $Epochs | Time: $MaxMinutes min"
Write-Host "Checkpoint: $CheckpointDir"
Write-Host ""

& $Python @Args
if ($LASTEXITCODE -ne 0) {
  throw "Print-clean training exited with code $LASTEXITCODE."
}
