# Analytics report: .\analytics.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "no .venv found — run: python -m venv .venv; .venv\Scripts\pip install -r requirements.txt"
    exit 1
}

$env:PYTHONPATH = Join-Path $PSScriptRoot "src"
& $python -m shorts_factory.analytics
exit $LASTEXITCODE
