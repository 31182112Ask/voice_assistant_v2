param(
    [switch]$CpuOllama
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

if (!(Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
}

if ($CpuOllama) {
    $env:OLLAMA_NUM_GPU = "0"
}

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

& ".\.venv\Scripts\python.exe" -m server.main
