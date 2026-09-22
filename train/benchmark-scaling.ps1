param(
  [int]$Pages = 8,
  [int]$Warmup = 2,
  [int]$Steps = 8,
  [int]$Threads = 6,
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
& $Python .\benchmark-scaling.py --pages $Pages --warmup $Warmup --steps $Steps --threads $Threads --min-free-ram-mib $MinFreeRamMiB --max-rss-mib $MaxRssMiB
if ($LASTEXITCODE -ne 0) { throw "Scaling benchmark exited with code $LASTEXITCODE." }