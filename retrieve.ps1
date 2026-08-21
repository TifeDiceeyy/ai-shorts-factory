# Phase 1 retrieval: .\retrieve.ps1 soap   or   .\retrieve.ps1 "roman concrete"
# Requires SEARCH_PROVIDER=tavily + SEARCH_API_KEY + an explicit BUDGET_CAP_USD
# in .env — there is no stub for search, so this refuses to run without them.
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
& $python -m shorts_factory.retrieval $Topic
exit $LASTEXITCODE
