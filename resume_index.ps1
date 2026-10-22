# Resume indexing on Windows from any folder.
#   .\resume_index.ps1 --root "D:/Photos"
#   .\resume_index.ps1 --root "D:/Photos" --check
Set-Location $PSScriptRoot

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Run .\setup.ps1 first."
}

& $VenvPython (Join-Path $PSScriptRoot "resume_index.py") @args
exit $LASTEXITCODE
