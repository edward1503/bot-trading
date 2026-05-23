# Stop bot + dashboard started by start_local.ps1.
# Kills the wrapper PowerShell + its child python process tree.

$ROOT = Split-Path -Parent $PSScriptRoot
$PidFile = Join-Path $ROOT "logs\local_pids.txt"

if (-not (Test-Path $PidFile)) {
    Write-Output "No PID file found at $PidFile — nothing to stop."
    Write-Output "If processes are running anyway: Get-Process python,powershell | Stop-Process -Force"
    exit 0
}

$content = Get-Content $PidFile
foreach ($line in $content) {
    if ($line -match '^(\w+)=(\d+)$') {
        $name = $Matches[1]; $pidNum = [int]$Matches[2]
        # Kill wrapper + its children (the python process running inside)
        try {
            $children = Get-CimInstance Win32_Process -Filter "ParentProcessId=$pidNum" -ErrorAction SilentlyContinue
            foreach ($c in $children) {
                Stop-Process -Id $c.ProcessId -Force -ErrorAction SilentlyContinue
                Write-Output "  killed child python PID $($c.ProcessId) ($name)"
            }
            Stop-Process -Id $pidNum -Force -ErrorAction SilentlyContinue
            Write-Output "✓ stopped $name wrapper PID $pidNum"
        } catch {
            Write-Output "⚠  $name PID $pidNum already gone"
        }
    }
}

Remove-Item $PidFile -Force
Write-Output "Done."
