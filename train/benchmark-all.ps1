param(
  [int]$Pages = 100,
  [int]$Rounds = 2,
  [int]$StepsPerConfig = 25,
  [int]$Warmup = 5,
  [switch]$SkipCompile,
  [switch]$SkipBf16,
  [int]$EnduranceSteps = 300,
  [int]$EnduranceCandidates = 4
)

$ErrorActionPreference = "Stop"

$TrainDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $TrainDir
$Python = Join-Path $TrainDir ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
  throw "Python venv not found. Run .\train-python.ps1 once first."
}

& $Python -m pip install -r .\requirements-python.txt
if ($LASTEXITCODE -ne 0) {
  throw "Dependency check failed with code $LASTEXITCODE."
}

$Args = @(
  ".\benchmark-all.py",
  "--pages", $Pages,
  "--rounds", $Rounds,
  "--steps-per-config", $StepsPerConfig,
  "--warmup", $Warmup,
  "--endurance-steps", $EnduranceSteps,
  "--endurance-candidates", $EnduranceCandidates
)

if ($SkipCompile) {
  $Args += "--skip-compile"
}

if ($SkipBf16) {
  $Args += "--skip-bf16"
}

& $Python $Args
if ($LASTEXITCODE -ne 0) {
  throw "Exhaustive benchmark exited with code $LASTEXITCODE."
}
