param(
    [int]$Port = 8000,
    [int[]]$ExcludePid = @($PID),
    [switch]$SkipCommandLineScan
)

$ErrorActionPreference = "SilentlyContinue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$escapedRoot = [regex]::Escape($Root)
$pids = New-Object System.Collections.Generic.HashSet[int]

if (-not $SkipCommandLineScan) {
try {
    Get-CimInstance Win32_Process |
        Where-Object {
            ($ExcludePid -notcontains [int]$_.ProcessId) -and
            $_.Name -match "python(\.exe)?$" -and
            $_.CommandLine -match "server\.main" -and
            $_.CommandLine -match $escapedRoot
        } |
        ForEach-Object { [void]$pids.Add([int]$_.ProcessId) }
} catch {
    Write-Host "Process command-line scan unavailable; falling back to port scan."
}
}

try {
    $lines = netstat -ano | Select-String (":$Port\s")
    foreach ($line in $lines) {
        if ($line.Line -match "\sLISTENING\s+(\d+)\s*$") {
            $id = [int]$Matches[1]
            if ($ExcludePid -notcontains $id) {
                [void]$pids.Add($id)
            }
        }
    }
} catch {
}

foreach ($id in $pids) {
    try {
        $proc = Get-Process -Id $id -ErrorAction Stop
        Write-Host "Stopping voice_assistant_v2 process PID $id ($($proc.ProcessName))"
        Stop-Process -Id $id -Force -ErrorAction Stop
    } catch {
    }
}
