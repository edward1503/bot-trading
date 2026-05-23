# Show local bot + dashboard status.

$ROOT = Split-Path -Parent $PSScriptRoot
$PidFile = Join-Path $ROOT "logs\local_pids.txt"

Write-Output "=== Process status ==="
if (Test-Path $PidFile) {
    foreach ($line in Get-Content $PidFile) {
        if ($line -match '^(\w+)=(\d+)$') {
            $name = $Matches[1]; $pidNum = [int]$Matches[2]
            $proc = Get-Process -Id $pidNum -ErrorAction SilentlyContinue
            if ($proc) {
                $children = Get-CimInstance Win32_Process -Filter "ParentProcessId=$pidNum" -ErrorAction SilentlyContinue
                $childInfo = if ($children) { " (child python PID $(($children | Select-Object -First 1).ProcessId))" } else { " (no child python!)" }
                Write-Output "  ✓ $name wrapper PID $pidNum  uptime $((New-TimeSpan $proc.StartTime (Get-Date)).ToString('dd\.hh\:mm\:ss'))$childInfo"
            } else {
                Write-Output "  ✗ $name wrapper PID $pidNum NOT RUNNING"
            }
        }
    }
} else {
    Write-Output "  No PID file. Not started, or stopped cleanly."
}

Write-Output ""
Write-Output "=== Health check ==="
try {
    $h = Invoke-RestMethod "http://localhost:8080/api/health" -TimeoutSec 3
    Write-Output ("  status: {0}  |  last loop: {1}  |  age: {2}s" -f $h.status, $h.last_loop, $h.age_seconds)
} catch {
    Write-Output "  ✗ Dashboard not responding on :8080"
}

Write-Output ""
Write-Output "=== Last 5 bot log lines ==="
$botLog = Join-Path $ROOT "logs\bot.log"
if (Test-Path $botLog) {
    Get-Content $botLog -Tail 5 | ForEach-Object { Write-Output "  $_" }
} else {
    Write-Output "  (no bot.log yet)"
}
