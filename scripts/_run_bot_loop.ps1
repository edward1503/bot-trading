# Inner wrapper: auto-restart scheduler on crash.
# Don't run directly — invoked by start_local.ps1.
Set-Location $PSScriptRoot\..
$ErrorActionPreference = "Continue"
while ($true) {
    Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting scheduler..."
    python -m src.scheduler
    Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Scheduler exited with code $LASTEXITCODE — restart in 30s"
    Start-Sleep -Seconds 30
}
