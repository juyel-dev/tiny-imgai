param(
  [int]$Epochs = 3,
  [int]$MaxPages = 0,
  [int]$MaxMinutes = 120,
  [int]$Threads = 6,
  [double]$LearningRate = 0.001,
  [int]$MinFreeRamMiB = 768,
  [int]$MaxRssMiB = 2048,
  [int]$PreviewPages = 12
)

$ErrorActionPreference = "Stop"
$TrainDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = Split-Path -Parent $TrainDir
Set-Location $TrainDir
$Python = Join-Path $TrainDir ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
  throw "Python venv not found. Run .\train-python.ps1 first."
}

Write-Host ""
Write-Host "tiny-imgai PRODUCTION PIPELINE" -ForegroundColor Cyan
Write-Host "512x512 | U-Net b48 | print-clean-v1 | CPU"
Write-Host ""

& .\train-scale.ps1 -BaseChannels 48 -Epochs $Epochs -MaxPages $MaxPages -MaxMinutes $MaxMinutes -LearningRate $LearningRate -Threads $Threads -MinFreeRamMiB $MinFreeRamMiB -MaxRssMiB $MaxRssMiB
if ($LASTEXITCODE -ne 0) { throw "Training launcher failed." }

$CheckpointDir = Join-Path $TrainDir "checkpoints\scaled-512-b48-eager"
$Checkpoint = Join-Path $CheckpointDir "checkpoint.pt"
if (-not (Test-Path $Checkpoint)) { throw "Training finished without a checkpoint: $Checkpoint" }

& $Python -c "import torch,sys; p=sys.argv[1]; target=int(sys.argv[2]); s=torch.load(p,map_location='cpu',weights_only=False); ok=int(s.get('epoch',0)) >= target and bool(s.get('bn_calibrated',False)); print('epoch='+str(s.get('epoch',0))); print('bn_calibrated='+str(bool(s.get('bn_calibrated',False)))); sys.exit(0 if ok else 2)" $Checkpoint $Epochs
if ($LASTEXITCODE -ne 0) {
  Write-Host ""
  Write-Host "Training is resumable but has not reached the requested target yet." -ForegroundColor Yellow
  Write-Host "Run .\run-production.ps1 again to continue from the saved checkpoint."
  exit 0
}

Write-Host ""
Write-Host "Training target reached and BatchNorm is finalized." -ForegroundColor Green

$checkpointState = & $Python -c "import torch,sys; s=torch.load(sys.argv[1],map_location='cpu',weights_only=False); print(str(s.get('epoch',0))+'|'+str(s.get('global_step',0)))" $Checkpoint
if ($LASTEXITCODE -ne 0) { throw "Could not read checkpoint metadata." }
$checkpointParts = $checkpointState.Split("|")
$checkpointEpoch = [int]$checkpointParts[0]
$checkpointStep = [int]$checkpointParts[1]

$evaluationRoot = Join-Path $TrainDir "evaluation"
$matchingEvaluation = $null
if (Test-Path $evaluationRoot) {
  foreach ($metricsPath in (Get-ChildItem $evaluationRoot -Directory -Filter "scale512-*" | Sort-Object LastWriteTime -Descending | ForEach-Object { Join-Path $_.FullName "metrics.json" })) {
    if (-not (Test-Path $metricsPath)) { continue }
    try {
      $metrics = Get-Content $metricsPath -Raw | ConvertFrom-Json
      $summary = $metrics.summary
      if (
        [int]$summary.checkpoint_epoch -eq $checkpointEpoch -and
        [int]$summary.checkpoint_global_step -eq $checkpointStep -and
        [int]$summary.input_size -eq 512 -and
        [int]$summary.base_channels -eq 48
      ) {
        $matchingEvaluation = $metricsPath
        break
      }
    } catch {
      continue
    }
  }
}

if ($matchingEvaluation) {
  Write-Host "Matching 512px/b48 evaluation already exists. Skipping repeat evaluation." -ForegroundColor Green
  Write-Host "Metrics: $matchingEvaluation"
} else {
  Write-Host "Running full 512px evaluation..." -ForegroundColor Cyan
  & .\evaluate-scale.ps1 -BaseChannels 48 -MaxPages $MaxPages -PreviewPages $PreviewPages -MaxRssMiB $MaxRssMiB -MinFreeRamMiB $MinFreeRamMiB
  if ($LASTEXITCODE -ne 0) { throw "Evaluation failed." }
}

Write-Host ""
Write-Host "Exporting production model for tf.js..." -ForegroundColor Cyan
& .\export-tfjs.ps1 -BaseChannels 48
if ($LASTEXITCODE -ne 0) { throw "tf.js export failed." }

Write-Host ""
Write-Host "Installing production model into web/models..." -ForegroundColor Cyan
& .\install-browser-model.ps1
if ($LASTEXITCODE -ne 0) { throw "Browser model installation failed." }

Write-Host ""
Write-Host "PRODUCTION PIPELINE COMPLETE" -ForegroundColor Green
Write-Host "Checkpoint: $Checkpoint"
Write-Host ("tf.js model: " + (Join-Path $RepoDir "web\models\tfjs-512-b48"))