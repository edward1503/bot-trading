# XAUUSD bot CLI. Single entry point for daily operations.
#
# Usage:  bot <command>
#
# Commands:
#   start     Start bot + dashboard (background, auto-restart)
#   stop      Stop both
#   restart   Stop + start
#   status    Process + health + last log lines
#   logs      Tail bot.log live (Ctrl+C to exit)
#   dash      Open dashboard in browser
#   health    Print /api/health JSON
#   pos       Print current position
#   trades    Print last 10 trades from DB
#   reset     Reset paper portfolio to fresh $100k (asks confirm)
#   help      Show this help

param(
    [Parameter(Position = 0)][string]$Command = "status",
    [Parameter(ValueFromRemainingArguments = $true)]$Rest
)

$ROOT = Split-Path -Parent $PSScriptRoot
Push-Location $ROOT
try {

function Write-Cmd($text) { Write-Host $text -ForegroundColor Cyan }
function Write-Ok($text)  { Write-Host $text -ForegroundColor Green }
function Write-Warn($text){ Write-Host $text -ForegroundColor Yellow }
function Write-Err($text) { Write-Host $text -ForegroundColor Red }

switch ($Command.ToLower()) {

    "start" {
        Write-Cmd "▶ Starting bot + dashboard..."
        & "$PSScriptRoot\start_local.ps1"
    }

    "stop" {
        Write-Cmd "■ Stopping bot + dashboard..."
        & "$PSScriptRoot\stop_local.ps1"
    }

    "restart" {
        Write-Cmd "↻ Restarting..."
        & "$PSScriptRoot\stop_local.ps1"
        Start-Sleep -Seconds 2
        & "$PSScriptRoot\start_local.ps1"
    }

    "status" {
        & "$PSScriptRoot\status_local.ps1"
    }

    "logs" {
        $log = Join-Path $ROOT "logs\bot.log"
        if (-not (Test-Path $log)) { Write-Err "No log file yet — run 'bot start' first"; break }
        Write-Cmd "Tailing $log  (Ctrl+C to exit)"
        Get-Content $log -Wait -Tail 30
    }

    "dash" {
        Write-Cmd "→ Opening http://localhost:8080"
        Start-Process "http://localhost:8080"
    }

    "health" {
        try {
            $h = Invoke-RestMethod "http://localhost:8080/api/health" -TimeoutSec 5
            $color = if ($h.status -eq "ok") { "Green" } else { "Yellow" }
            Write-Host "Status:    $($h.status)" -ForegroundColor $color
            Write-Host "Last loop: $($h.last_loop)"
            Write-Host "Age:       $($h.age_seconds)s ago"
        } catch { Write-Err "Dashboard not responding on :8080" }
    }

    "pos" {
        try {
            $p = Invoke-RestMethod "http://localhost:8080/api/position" -TimeoutSec 5
            if ($p.side -eq "flat") {
                Write-Host "Position: FLAT" -ForegroundColor Yellow
            } else {
                $pnlColor = if ($p.unrealized_pnl -ge 0) { "Green" } else { "Red" }
                Write-Host ('Side:        {0}' -f $p.side.ToUpper())
                Write-Host ('Size:        {0:N3} oz  (notional ${1:N0})' -f $p.position_size, $p.notional)
                Write-Host ('Avg entry:   ${0:N2}' -f $p.avg_entry_price)
                Write-Host ('Current px:  ${0:N2}' -f $p.current_price)
                Write-Host ('Unrealized:  ${0:N4} ({1:N3}%)' -f $p.unrealized_pnl, $p.unrealized_pct) -ForegroundColor $pnlColor
                Write-Host ('Exposure:    {0:N2}%' -f $p.exposure_pct)
            }
            Write-Host ""
            Write-Host ('Balance:     ${0:N2}' -f $p.balance)
            Write-Host ('Realized:    ${0:N2}' -f $p.realized_pnl)
        } catch { Write-Err "Dashboard not running. Try 'bot start'." }
    }

    "trades" {
        try {
            $t = Invoke-RestMethod "http://localhost:8080/api/trades" -TimeoutSec 5
            if (-not $t -or $t.Count -eq 0) { Write-Warn "No trades yet"; break }
            $t | Select-Object -First 10 timestamp, action, volume_oz, price, llm_signal, llm_confidence, rl_action |
                 Format-Table -AutoSize
        } catch { Write-Err "Dashboard not running. Try 'bot start'." }
    }

    "reset" {
        Write-Warn "This will WIPE all trades + portfolio history and reset balance to \$100,000."
        $confirm = Read-Host "Type 'yes' to confirm"
        if ($confirm -ne "yes") { Write-Cmd "Cancelled."; break }

        # Auto-stop if running
        $PidFile = Join-Path $ROOT "logs\local_pids.txt"
        $wasRunning = Test-Path $PidFile
        if ($wasRunning) {
            Write-Cmd "Stopping bot..."
            & "$PSScriptRoot\stop_local.ps1" | Out-Null
        }

        python -c "from sqlalchemy import text; from src.db import get_engine; conn=get_engine().connect();`
[conn.execute(text(f'DELETE FROM {t}')) for t in ['paper_portfolio','trades','portfolio_snapshots','position_tracker']]; conn.commit(); conn.close();`
from src.execution.paper_trader import PaperTrader; PaperTrader(); print('Reset to \$100,000')"

        if ($wasRunning) {
            Write-Cmd "Restarting bot..."
            & "$PSScriptRoot\start_local.ps1"
        }
    }

    { $_ -in @("help", "-h", "--help", "?") } {
        Write-Host ""
        Write-Host "XAUUSD bot CLI" -ForegroundColor Cyan
        Write-Host "Single entry point for daily operations of the XAUUSD trading bot."
        Write-Host ""
        Write-Host 'Usage:  bot <command>' -ForegroundColor Yellow
        Write-Host '        (no command = ' -NoNewline; Write-Host 'status' -ForegroundColor Green -NoNewline; Write-Host ')'
        Write-Host ""

        Write-Host 'Lifecycle:' -ForegroundColor Magenta
        Write-Host '  start     ' -NoNewline -ForegroundColor Green; Write-Host 'Start bot + dashboard in background (auto-restart on crash)'
        Write-Host '  stop      ' -NoNewline -ForegroundColor Green; Write-Host 'Stop bot + dashboard processes'
        Write-Host '  restart   ' -NoNewline -ForegroundColor Green; Write-Host 'Stop then start (useful after config/code changes)'
        Write-Host ""

        Write-Host 'Monitoring:' -ForegroundColor Magenta
        Write-Host '  status    ' -NoNewline -ForegroundColor Green; Write-Host 'Process state + /api/health + last log lines'
        Write-Host '  logs      ' -NoNewline -ForegroundColor Green; Write-Host 'Tail logs/bot.log live (Ctrl+C to exit)'
        Write-Host '  health    ' -NoNewline -ForegroundColor Green; Write-Host 'Print /api/health JSON (last loop time, age)'
        Write-Host '  dash      ' -NoNewline -ForegroundColor Green; Write-Host 'Open dashboard at http://localhost:8080 in browser'
        Write-Host ""

        Write-Host 'Trading data:' -ForegroundColor Magenta
        Write-Host '  pos       ' -NoNewline -ForegroundColor Green; Write-Host 'Print current position (side, size, entry, unrealized PnL, balance)'
        Write-Host '  trades    ' -NoNewline -ForegroundColor Green; Write-Host 'Print last 10 trades from DB with LLM/RL signals'
        Write-Host ""

        Write-Host 'Maintenance:' -ForegroundColor Magenta
        Write-Host '  reset     ' -NoNewline -ForegroundColor Green; Write-Host 'Wipe trades + portfolio history, reset to $100k (asks confirm)'
        Write-Host '  help      ' -NoNewline -ForegroundColor Green; Write-Host 'Show this help (aliases: -h, --help, ?)'
        Write-Host ""

        Write-Host 'Examples:' -ForegroundColor Yellow
        Write-Host '  bot start            # boot everything'
        Write-Host '  bot                  # quick status check'
        Write-Host '  bot logs             # watch live activity'
        Write-Host '  bot pos              # check open position + PnL'
        Write-Host ""
    }

    default {
        Write-Err "Unknown command: $Command"
        Write-Host "Run 'bot help' for available commands."
        exit 1
    }
}

} finally { Pop-Location }
