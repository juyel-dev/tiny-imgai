param(
  [int]$BaseChannels = 32,
  [int]$Epochs = 3,
  [int]$MaxPages = 0,
  [int]$MaxMinutes = 120,
  [double]$LearningRate = 0.001,
  [int]$Threads = 6,
  [switch]$ActivationCheckpointing,
  [int]$MinFreeRamMiB = 768,
  [int]$MaxRssMiB = 3072
)

$ErrorActionPreference = "Stop"
$TrainDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $TrainDir
$Python = Join-Path $TrainDir ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Python venv not found. Run .\train-python.ps1 first." }
& $Python -m pip install -r .\requirements-python.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency check failed with code $LASTEXITCODE." }
$Args = @(".\train-scale.py","--base-channels",$BaseChannels,"--epochs",$Epochs,"--max-pages",$MaxPages,"--max-minutes",$MaxMinutes,"--learning-rate",$LearningRate,"--threads",$Threads,"--min-free-ram-mib",$MinFreeRamMiB,"--max-rss-mib",$MaxRssMiB)
if ($ActivationCheckpointing) { $Args += "--activation-checkpointing" }
& $Python @Args
if ($LASTEXITCODE -ne 0) { throw "Scalable training exited with code $LASTEXITCODE." }