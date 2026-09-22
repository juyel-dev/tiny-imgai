param(
  [string]$SourceDir = $null,
  [string]$DestinationDir = $null
)

$ErrorActionPreference = "Stop"

$TrainDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = Split-Path -Parent $TrainDir

if (-not $SourceDir) {
  $SourceDir = Join-Path $TrainDir "checkpoints\tfjs-512-b48"
}
if (-not $DestinationDir) {
  $DestinationDir = Join-Path $RepoDir "web\models\tfjs-512-b48"
}

$required = @("model.json", "weights.bin", "metadata.json")
foreach ($name in $required) {
  $path = Join-Path $SourceDir $name
  if (-not (Test-Path $path)) {
    throw "Missing exported browser model file: $path"
  }
}

New-Item -ItemType Directory -Force -Path $DestinationDir | Out-Null
Copy-Item (Join-Path $SourceDir "model.json") (Join-Path $DestinationDir "model.json") -Force
Copy-Item (Join-Path $SourceDir "weights.bin") (Join-Path $DestinationDir "weights.bin") -Force
Copy-Item (Join-Path $SourceDir "metadata.json") (Join-Path $DestinationDir "metadata.json") -Force

Write-Host ""
Write-Host "Browser production model installed." -ForegroundColor Green
Write-Host "Source:      $SourceDir"
Write-Host "Destination: $DestinationDir"
Write-Host ""
Write-Host "Files:" -ForegroundColor Cyan
Get-ChildItem $DestinationDir -File |
  Select-Object Name, @{Name="MiB";Expression={[math]::Round($_.Length / 1MB, 2)}} |
  Format-Table -AutoSize
