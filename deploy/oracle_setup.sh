#!/bin/bash
# Oracle Cloud Always Free (Ubuntu 22.04 ARM, 4 OCPU / 24 GB) setup.
# Idempotent — safe to re-run after pulling new code.
#
#   bash deploy/oracle_setup.sh

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/edward1503/bot-trading.git}"
PROJECT_DIR="${PROJECT_DIR:-/home/ubuntu/bot-trading}"
SERVICE_USER="${SERVICE_USER:-ubuntu}"
DASH_PORT="${DASH_PORT:-8080}"

echo "=== XAUUSD Bot · Oracle Cloud Setup ==="
echo "Project dir: $PROJECT_DIR"
echo "Dashboard port: $DASH_PORT"

# ─── 1. System dependencies ──────────────────────────────────────────────────
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3-pip git \
    build-essential iptables-persistent logrotate curl

# ─── 2. Clone or pull the repo ───────────────────────────────────────────────
if [ ! -d "$PROJECT_DIR/.git" ]; then
    git clone "$REPO_URL" "$PROJECT_DIR"
else
    git -C "$PROJECT_DIR" pull --ff-only || echo "WARN: git pull failed (uncommitted changes?)"
fi
cd "$PROJECT_DIR"

# ─── 3. Virtualenv + deps ────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    python3.11 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel
pip install --no-cache-dir -r requirements.txt

# ─── 4. .env scaffolding ─────────────────────────────────────────────────────
if [ ! -f config/.env ]; then
    cp config/.env.example config/.env
    echo "⚠  config/.env created from template. Fill in BYBIT_API_KEY / BYBIT_API_SECRET / GROQ_API_KEY before starting!"
fi

mkdir -p logs models data/historical

# ─── 5. Open dashboard port in iptables (Oracle Ubuntu blocks by default) ────
if ! sudo iptables -C INPUT -p tcp --dport "$DASH_PORT" -j ACCEPT 2>/dev/null; then
    # Insert before the REJECT rule at the bottom of INPUT chain
    sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport "$DASH_PORT" -j ACCEPT
    sudo netfilter-persistent save
    echo "✓ iptables: opened tcp/$DASH_PORT"
fi

# ─── 6. Systemd: trading bot ─────────────────────────────────────────────────
sudo tee /etc/systemd/system/trading-bot.service > /dev/null <<EOF
[Unit]
Description=XAUUSD 24/7 Paper Trading Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$PROJECT_DIR
Environment=PYTHONUNBUFFERED=1
ExecStart=$PROJECT_DIR/.venv/bin/python -m src.scheduler
Restart=always
RestartSec=30
StandardOutput=append:$PROJECT_DIR/logs/bot.log
StandardError=append:$PROJECT_DIR/logs/bot.log

[Install]
WantedBy=multi-user.target
EOF

# ─── 7. Systemd: dashboard ───────────────────────────────────────────────────
sudo tee /etc/systemd/system/trading-dashboard.service > /dev/null <<EOF
[Unit]
Description=XAUUSD Trading Dashboard (FastAPI)
After=network-online.target trading-bot.service
Wants=network-online.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$PROJECT_DIR
Environment=PYTHONUNBUFFERED=1
ExecStart=$PROJECT_DIR/.venv/bin/python -m src.dashboard.server
Restart=always
RestartSec=10
StandardOutput=append:$PROJECT_DIR/logs/dashboard.log
StandardError=append:$PROJECT_DIR/logs/dashboard.log

[Install]
WantedBy=multi-user.target
EOF

# ─── 8. logrotate: keep logs ≤ ~50 MB total ──────────────────────────────────
sudo tee /etc/logrotate.d/trading-bot > /dev/null <<EOF
$PROJECT_DIR/logs/*.log {
    daily
    rotate 14
    size 10M
    missingok
    notifempty
    compress
    delaycompress
    copytruncate
}
EOF

# ─── 9. Enable + start services ──────────────────────────────────────────────
sudo systemctl daemon-reload
sudo systemctl enable trading-bot trading-dashboard
sudo systemctl restart trading-bot trading-dashboard

PUBLIC_IP=$(curl -s ifconfig.me || echo "<your-oracle-ip>")
cat <<EOF

=== Setup Complete ===

  Bot status:        sudo systemctl status trading-bot
  Dashboard status:  sudo systemctl status trading-dashboard
  Bot logs:          tail -f $PROJECT_DIR/logs/bot.log
  Dashboard logs:    tail -f $PROJECT_DIR/logs/dashboard.log
  Dashboard URL:     http://$PUBLIC_IP:$DASH_PORT
  Health check:      curl http://$PUBLIC_IP:$DASH_PORT/api/health

IMPORTANT: also open tcp/$DASH_PORT in Oracle Cloud Security List (VCN → Subnet → Security List → Ingress).

EOF
