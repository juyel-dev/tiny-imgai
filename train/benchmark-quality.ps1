param(
  [int]$TrainPages = 16,
  [int]$ValPages = 8,
  [int]$Epochs = 1,
  [int]$Threads = 6,
  [int]$MinFreeRamMiB = 768,
  [int]$MaxRssMiB = 1536,
  [int]$PreviewPages = 2
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

& $Python .\benchmark-quality.py --train-pages $TrainPages --val-pages $ValPages --epochs $Epochs --threads $Threads --min-free-ram-mib $MinFreeRamMiB --max-rss-mib $MaxRssMiB --preview-pages $PreviewPages
if ($LASTEXITCODE -ne 0) { throw "Quality screen failed." }