param(
  [int]$BaseChannels = 48
)

$ErrorActionPreference = "Stop"

if ($BaseChannels -ne 48) {
  throw "The production tf.js exporter is locked to base width 48."
}

$TrainDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $TrainDir
$Python = Join-Path $TrainDir ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
  throw "Python venv not found. Run .\train-python.ps1 first."
}

& $Python -m pip install -r .\requirements-python.txt
if ($LASTEXITCODE -ne 0) {
  throw "Dependency check failed."
}

& $Python .\export-tfjs.py --base-channels $BaseChannels
if ($LASTEXITCODE -ne 0) {
  throw "PyTorch export staging failed."
}

if (-not (Test-Path .\node_modules)) {
  npm ci
  if ($LASTEXITCODE -ne 0) {
    throw "npm dependency installation failed."
  }
}

& node .\export-tfjs.mjs --staging-dir .\evaluation\tfjs-export-staging --output-dir .\checkpoints\tfjs-512-b48
if ($LASTEXITCODE -ne 0) {
  throw "tf.js conversion failed."
}
