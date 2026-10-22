# Resume indexing on Windows from any folder.
#   .\scripts\resume_index.ps1 --root "D:/Photos"
#   .\scripts\resume_index.ps1 --root "D:/Photos" --check
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location $Repo

$VenvPython = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Run .\scripts\setup.ps1 first."
}

& $VenvPython -m cli.resume_index @args
exit $LASTEXITCODE
