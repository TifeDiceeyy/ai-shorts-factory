# One command, end to end: .\run.ps1 soap
# Produces artifacts/<topic>/{<topic>.mp4, <topic>.script.json, captions.*,
# cost-report.json, verification-report.json}. Exit code is non-zero if any
# verification criterion fails or the topic is safety-blocked.
param(
    [Parameter(Mandatory = $true)]
    [string]$Topic
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "no .venv found — run: python -m venv .venv; .venv\Scripts\pip install -r requirements.txt"
    exit 1
}

$env:PYTHONPATH = Join-Path $PSScriptRoot "src"
& $python -m shorts_factory.pipeline $Topic
exit $LASTEXITCODE
