# XAUUSD 24/7 Paper Trading Bot

**Groq Llama 3.3 70B** + **PPO Reinforcement Learning** + **Evolutionary RL (CMA-ES)**  
Paper trading XAU/USD via OANDA Practice API — runs 24/7 on Oracle Cloud Always Free.

---

## Architecture

```
LLM Agents (Groq)          Evolutionary RL (EvoTorch)
  • Technical Analyst   ──►   Population of PPO policies
  • Sentiment Analyst         Weekly fitness = Sharpe ratio
  • Risk Manager              CMA-ES evolves → deploy best
        │                              │
        └──────────── Signal Router ───┘
                            │
                     OANDA Practice API
                       (XAU_USD orders)
                            │
                     SQLite + Streamlit Dashboard
```

---

## Quick Start

### 1. Get free API keys
- **OANDA Practice**: https://www.oanda.com/demo-account/ (free demo account)
- **Groq**: https://console.groq.com/ (free tier: 30 RPM, 1000 RPD)

### 2. Configure
```bash
cp config/.env.example config/.env
# Edit config/.env with your OANDA_API_KEY, OANDA_ACCOUNT_ID, GROQ_API_KEY
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Test connections
```bash
python scripts/test_connection.py
```

### 5. Train baseline RL model (on Gold Futures historical data)
```bash
python src/rl/train.py --train-start 2022-01-01 --train-end 2023-12-31
```

### 6. Backtest out-of-sample
```bash
python src/backtest/run_backtest.py --start 2024-01-01 --end 2024-06-01
# Report: results/backtest_2024-01-01_2024-06-01.html
```

### 7. Run 24/7 paper trading bot
```bash
python src/scheduler.py
```

### 8. View dashboard
```bash
streamlit run src/dashboard/app.py
# Open: http://localhost:8501
```

---

## Deploy to Oracle Cloud (Always Free)

```bash
bash deploy/oracle_setup.sh
# Then open port 8501 in Oracle Security List for dashboard
```

---

## Config (`config/config.yaml`)

Key settings to customize:
- `trading.units` — position size in XAU units
- `risk.max_daily_drawdown` — circuit breaker threshold (default 3%)
- `rl.population_size` — number of policies in evolutionary pool (default 20)
- `rl.evolution_interval_days` — how often to evolve (default 7 days)
- `llm.model` — Groq model name

---

## Stack (All Free)

| Component | Technology | Cost |
|-----------|-----------|------|
| Paper Broker | OANDA v20 Practice API | Free |
| LLM | Groq Llama 3.3 70B | Free tier (30 RPM) |
| RL | Stable-Baselines3 PPO | Open source |
| Evolutionary | EvoTorch CMA-ES | Open source |
| Hosting | Oracle Cloud ARM 4 OCPU / 24GB | Always Free |
| Dashboard | Streamlit + Plotly | Open source |
| Backtest | backtesting.py | Open source |

---

## Project Structure

```
src/
├── agents/         # LLM agents (technical, sentiment, risk)
├── rl/             # Gym env, PPO training, EvoTorch evolution
├── backtest/       # backtesting.py wrapper + walk-forward validation
├── data/           # OANDA candle fetcher, news headline fetcher
├── execution/      # OANDA paper broker client
├── dashboard/      # Streamlit 4-tab dashboard
├── router.py       # Signal fusion: LLM + RL → order decision
├── scheduler.py    # 24/7 APScheduler: 5min loop + daily + weekly
└── db.py           # SQLite persistence (trades, PnL, fitness)
config/
├── config.yaml     # All tunable parameters
└── .env.example    # API key template
scripts/
└── test_connection.py  # Smoke test Phase 1
deploy/
└── oracle_setup.sh     # Oracle Cloud systemd setup
```