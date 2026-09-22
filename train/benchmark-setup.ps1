param(
  [int]$Pages = 4,
  [int]$Steps = 15,
  [int]$Warmup = 3
)

$ErrorActionPreference = "Stop"

$TrainDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $TrainDir
$Python = Join-Path $TrainDir ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
  throw "Python venv not found. Run .\train-python.ps1 once first."
}

Write-Host "Benchmark: $Pages pages · $Warmup warmup · $Steps measured steps"
& $Python .\benchmark-setup.py --pages $Pages --steps $Steps --warmup $Warmup
if ($LASTEXITCODE -ne 0) {
  throw "Benchmark exited with code $LASTEXITCODE."
}
