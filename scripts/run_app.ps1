# Launch the local web UI (127.0.0.1 only). Stop with Ctrl-C.
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location $Repo

$VenvPython = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Run .\scripts\setup.ps1 first."
}

& $VenvPython (Join-Path $Repo "webapp.py") @args
exit $LASTEXITCODE
