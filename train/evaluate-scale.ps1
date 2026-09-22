param(
  [int]$BaseChannels = 48,
  [int]$MaxPages = 0,
  [int]$PreviewPages = 12,
  [int]$MaxRssMiB = 2048,
  [int]$MinFreeRamMiB = 768
)

$ErrorActionPreference = "Stop"
$TrainDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $TrainDir
$Python = Join-Path $TrainDir ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
  throw "Python venv not found. Run .\train-python.ps1 first."
}

& $Python -m pip install -r .\requirements-python.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency check failed." }

$CheckpointDir = Join-Path $TrainDir ("checkpoints\scaled-512-b{0}-eager" -f $BaseChannels)
$Args = @(".\evaluate-scale.py", "--base-channels", $BaseChannels, "--max-pages", $MaxPages, "--preview-pages", $PreviewPages, "--max-rss-mib", $MaxRssMiB, "--min-free-ram-mib", $MinFreeRamMiB, "--checkpoint-dir", $CheckpointDir)

& $Python @Args
if ($LASTEXITCODE -ne 0) { throw "Scalable evaluation exited with code $LASTEXITCODE." }
