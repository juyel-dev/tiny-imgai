param(
  [int]$MaxPages = 0
)

$ErrorActionPreference = "Stop"

$TrainDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $TrainDir
$Python = Join-Path $TrainDir ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
  throw "Python venv not found. Run .\train-python.ps1 first."
}

& $Python .\finalize-checkpoint.py --max-pages $MaxPages
if ($LASTEXITCODE -ne 0) {
  throw "Checkpoint finalization exited with code $LASTEXITCODE."
}
