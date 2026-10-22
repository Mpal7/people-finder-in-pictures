# Launch the local web UI (127.0.0.1 only). Stop with Ctrl-C.
Set-Location $PSScriptRoot

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Run .\setup.ps1 first."
}

& $VenvPython (Join-Path $PSScriptRoot "webapp.py") @args
exit $LASTEXITCODE
