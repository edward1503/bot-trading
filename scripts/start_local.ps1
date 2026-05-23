# Start bot + dashboard as detached background processes.
# Usage:  .\scripts\start_local.ps1
#
# Writes PIDs to logs\local_pids.txt for stop_local.ps1.
# Each process auto-restarts on crash (via inner _run_*_loop.ps1).

$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

$LogDir = Join-Path $ROOT "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

$PidFile = Join-Path $LogDir "local_pids.txt"
if (Test-Path $PidFile) {
    Write-Output "⚠  Existing PID file found at $PidFile"
    Write-Output "   Run .\scripts\stop_local.ps1 first to avoid duplicates."
    exit 1
}

$botScript  = Join-Path $PSScriptRoot "_run_bot_loop.ps1"
$dashScript = Join-Path $PSScriptRoot "_run_dashboard_loop.ps1"

$botProc = Start-Process powershell.exe `
    -ArgumentList "-NoProfile","-NonInteractive","-ExecutionPolicy","Bypass","-File",$botScript `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $LogDir "bot_wrapper.log") `
    -RedirectStandardError  (Join-Path $LogDir "bot_wrapper_err.log") `
    -PassThru

$dashProc = Start-Process powershell.exe `
    -ArgumentList "-NoProfile","-NonInteractive","-ExecutionPolicy","Bypass","-File",$dashScript `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $LogDir "dashboard_wrapper.log") `
    -RedirectStandardError  (Join-Path $LogDir "dashboard_wrapper_err.log") `
    -PassThru

"bot=$($botProc.Id)`ndashboard=$($dashProc.Id)" | Out-File $PidFile -Encoding utf8

Write-Output "✓ Bot wrapper PID:       $($botProc.Id)"
Write-Output "✓ Dashboard wrapper PID: $($dashProc.Id)"
Write-Output ""
Write-Output "Dashboard:  http://localhost:8080"
Write-Output "Health:     http://localhost:8080/api/health"
Write-Output "Bot log:    Get-Content logs\bot.log -Wait -Tail 20"
Write-Output "Stop:       .\scripts\stop_local.ps1"
