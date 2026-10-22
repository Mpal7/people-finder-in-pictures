# One-time Windows setup: virtualenv + Python dependencies + model weights.
# Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
param(
    [switch]$NoModels
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location $Repo

$Candidates = @()
if ($env:PYTHON) {
    $Candidates += @{ Exe = $env:PYTHON; Args = @() }
}
if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($Version in @("3.12", "3.11", "3.10")) {
        $Candidates += @{ Exe = "py"; Args = @("-$Version") }
    }
}
$Candidates += @{ Exe = "python"; Args = @() }

$Python = $null
$PythonArgs = @()
foreach ($Candidate in $Candidates) {
    $CandidateArgs = @($Candidate.Args)
    $Usable = $false
    try {
        & $Candidate.Exe @CandidateArgs -c "import struct, sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) and struct.calcsize('P') == 8 else 1)" 2>$null | Out-Null
        $Usable = ($LASTEXITCODE -eq 0)
    } catch {
        $Usable = $false
    }
    if ($Usable) {
        $Python = $Candidate.Exe
        $PythonArgs = $CandidateArgs
        break
    }
}
if (-not $Python) {
    throw "Python 3.10-3.12 (64-bit) is required. Install it from https://www.python.org/downloads/ and rerun."
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "creating virtualenv (.venv)..."
    & $Python @PythonArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "could not create .venv" }
}

$VenvPython = Join-Path $Repo ".venv\Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
& $VenvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }

if (-not $NoModels) {
    Write-Host ""
    Write-Host "downloading model weights (one time, ~1 GB)..."
    & $VenvPython recon\setup_models.py
    if ($LASTEXITCODE -ne 0) { throw "model download failed" }
}

Write-Host ""
Write-Host "Setup complete. Start the UI with:"
Write-Host "  .\scripts\run_app.ps1"
