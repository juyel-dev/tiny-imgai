param(
  [int]$Pages = 10,
  [int]$Rounds = 4,
  [int]$StepsPerRound = 50,
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

& $Python .\benchmark-finalists.py --pages $Pages --rounds $Rounds --steps-per-round $StepsPerRound --warmup $Warmup
if ($LASTEXITCODE -ne 0) {
  throw "Finalist benchmark exited with code $LASTEXITCODE."
}
