param(
    [switch]$CpuOllama,
    [int]$Port = 8000,
    [switch]$NoLlamaCpp,
    [string]$LlamaModel = "",
    [int]$LlamaPort = 8080,
    [int]$LlamaNgl = 99,
    [int]$LlamaCtx = 4096
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

& ".\scripts\stop_server.ps1" -Port $Port

$llamaProcess = $null

function Test-PortListening([int]$CheckPort) {
    $lines = netstat -ano | Select-String (":$CheckPort\s")
    foreach ($line in $lines) {
        if ($line.Line -match "\sLISTENING\s+\d+\s*$") {
            return $true
        }
    }
    return $false
}

function Find-LlamaServer {
    $cmd = Get-Command llama-server -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    $wingetRoot = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
    if (Test-Path $wingetRoot) {
        $exe = Get-ChildItem -Path $wingetRoot -Recurse -Filter llama-server.exe -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($exe) { return $exe.FullName }
    }

    $ollamaExe = Join-Path $env:LOCALAPPDATA "Programs\Ollama\lib\ollama\llama-server.exe"
    if (Test-Path $ollamaExe) { return $ollamaExe }

    return $null
}

if (-not $NoLlamaCpp) {
    if (-not (Test-PortListening $LlamaPort)) {
        if (-not $LlamaModel) {
            $modelFile = Get-ChildItem -Path ".\models\llm" -Filter "*.gguf" -ErrorAction SilentlyContinue |
                Sort-Object Length -Descending |
                Select-Object -First 1
            if ($modelFile) {
                $LlamaModel = $modelFile.FullName
            }
        }

        if ($LlamaModel) {
            $llamaServer = Find-LlamaServer
            if (-not $llamaServer) {
                throw "llama-server not found. Install llama.cpp or pass -NoLlamaCpp."
            }
            Write-Host "Starting llama-server on port $LlamaPort with $LlamaModel"
            $llamaProcess = Start-Process -FilePath $llamaServer `
                -ArgumentList @(
                    "-m", $LlamaModel,
                    "-ngl", "$LlamaNgl",
                    "-c", "$LlamaCtx",
                    "-fa", "on",
                    "--no-mmap",
                    "--port", "$LlamaPort",
                    "--host", "127.0.0.1"
                ) `
                -WorkingDirectory $Root `
                -RedirectStandardOutput (Join-Path $Root "llamacpp.log") `
                -RedirectStandardError (Join-Path $Root "llamacpp.err.log") `
                -WindowStyle Hidden `
                -PassThru

            $deadline = (Get-Date).AddSeconds(30)
            while ((Get-Date) -lt $deadline) {
                if (Test-PortListening $LlamaPort) { break }
                Start-Sleep -Milliseconds 500
            }
            if (-not (Test-PortListening $LlamaPort)) {
                throw "llama-server did not start on port $LlamaPort. Check llamacpp.err.log."
            }
        }
    }
}

try {
    & ".\.venv\Scripts\python.exe" -m server.main
} finally {
    & ".\scripts\stop_server.ps1" -Port $Port
    if ($llamaProcess -and -not $llamaProcess.HasExited) {
        Write-Host "Stopping llama-server PID $($llamaProcess.Id)"
        Stop-Process -Id $llamaProcess.Id -Force -ErrorAction SilentlyContinue
    }
}
