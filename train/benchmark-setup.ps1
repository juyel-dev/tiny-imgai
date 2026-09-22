param(
  [int]$Pages = 4,
  [int]$Steps = 3,
  [int]$Warmup = 1
)

$ErrorActionPreference = "Stop"

$TrainDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $TrainDir
$Python = Join-Path $TrainDir ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
  throw "Python venv not found. Run .\train-python.ps1 once first."
}

& $Python .\benchmark-setup.py --pages $Pages --steps $Steps --warmup $Warmup
if ($LASTEXITCODE -ne 0) {
  throw "Benchmark exited with code $LASTEXITCODE."
}
