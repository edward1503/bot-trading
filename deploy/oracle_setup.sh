#!/bin/bash
# Oracle Cloud Always Free (Ubuntu 22.04 ARM) setup script
# Run as: bash deploy/oracle_setup.sh

set -e

echo "=== XAUUSD Bot Oracle Cloud Setup ==="

# System deps
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3-pip git

# Clone / pull project (adjust path as needed)
cd /home/ubuntu
git clone https://github.com/edward1503/bot-trading.git || (cd bot-trading && git pull)
cd bot-trading

# Virtual env + deps
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Copy env file (do this manually with your actual keys)
if [ ! -f config/.env ]; then
    cp config/.env.example config/.env
    echo "⚠  Edit config/.env with your OANDA and Groq API keys before starting!"
fi

# Create systemd service for trading bot
sudo tee /etc/systemd/system/trading-bot.service > /dev/null <<EOF
[Unit]
Description=XAUUSD 24/7 Paper Trading Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/bot-trading
ExecStart=/home/ubuntu/bot-trading/.venv/bin/python src/scheduler.py
Restart=always
RestartSec=30
StandardOutput=append:/home/ubuntu/bot-trading/logs/bot.log
StandardError=append:/home/ubuntu/bot-trading/logs/bot.log

[Install]
WantedBy=multi-user.target
EOF

# Create systemd service for dashboard
sudo tee /etc/systemd/system/trading-dashboard.service > /dev/null <<EOF
[Unit]
Description=XAUUSD Trading Dashboard (Streamlit)
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/bot-trading
ExecStart=/home/ubuntu/bot-trading/.venv/bin/streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start services
sudo systemctl daemon-reload
sudo systemctl enable trading-bot trading-dashboard
sudo systemctl start trading-bot trading-dashboard

echo ""
echo "=== Setup Complete ==="
echo "Bot status:       sudo systemctl status trading-bot"
echo "Dashboard status: sudo systemctl status trading-dashboard"
echo "Bot logs:         tail -f logs/bot.log"
echo "Dashboard URL:    http://$(curl -s ifconfig.me):8501"
echo ""
echo "IMPORTANT: Open port 8501 in Oracle Cloud Security List for dashboard access."
