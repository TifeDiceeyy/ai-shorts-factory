# Phase 4 review dashboard: .\dashboard.ps1  (serves on http://127.0.0.1:8420)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "no .venv found — run: python -m venv .venv; .venv\Scripts\pip install -r requirements.txt"
    exit 1
}

$env:PYTHONPATH = Join-Path $PSScriptRoot "src"
& $python -m uvicorn shorts_factory.dashboard.app:app --host 127.0.0.1 --port 8420 --reload
exit $LASTEXITCODE
