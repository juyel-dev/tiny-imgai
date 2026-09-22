# Compatibility wrapper: the old finalist command now runs the full benchmark matrix.
& .\benchmark-all.ps1 @args
if ($LASTEXITCODE -ne 0) {
  throw "Exhaustive benchmark exited with code $LASTEXITCODE."
}
