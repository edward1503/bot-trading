# Inner wrapper: auto-restart dashboard on crash.
# Don't run directly — invoked by start_local.ps1.
Set-Location $PSScriptRoot\..
$ErrorActionPreference = "Continue"
while ($true) {
    Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting dashboard..."
    python -m src.dashboard.server
    Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Dashboard exited with code $LASTEXITCODE — restart in 10s"
    Start-Sleep -Seconds 10
}
