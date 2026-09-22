param(
  [int]$MaxPages = 0,
  [int]$PreviewPages = 12
)

$ErrorActionPreference = "Stop"

$TrainDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $TrainDir
$Python = Join-Path $TrainDir ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
  throw "Python venv not found. Run .\train-python.ps1 once first."
}

& $Python -m pip install "Pillow>=11,<13"
if ($LASTEXITCODE -ne 0) {
  throw "Pillow installation failed."
}

& $Python .\evaluate.py --max-pages $MaxPages --preview-pages $PreviewPages
if ($LASTEXITCODE -ne 0) {
  throw "Evaluation exited with code $LASTEXITCODE."
}
