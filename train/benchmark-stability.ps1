param(
  [int]$Pages = 10,
  [int]$Steps = 120,
  [int]$Warmup = 5
)

$ErrorActionPreference = "Stop"

$TrainDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $TrainDir
$Python = Join-Path $TrainDir ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
  throw "Python venv not found. Run .\train-python.ps1 once first."
}

& $Python -m pip install psutil

& $Python .\benchmark-stability.py --pages $Pages --steps $Steps --warmup $Warmup
if ($LASTEXITCODE -ne 0) {
  throw "Stability benchmark exited with code $LASTEXITCODE."
}
