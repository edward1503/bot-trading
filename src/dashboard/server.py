"""FastAPI backend for the trading dashboard."""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="XAUUSD Bot Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/portfolio")
def get_portfolio():
    from src.db import get_portfolio_df
    df = get_portfolio_df()
    if df.empty:
        return []
    df["timestamp"] = df["timestamp"].astype(str)
    return df.to_dict(orient="records")


@app.get("/api/trades")
def get_trades():
    from src.db import get_trades_df
    df = get_trades_df()
    if df.empty:
        return []
    df["timestamp"] = df["timestamp"].astype(str)
    return df.fillna("").to_dict(orient="records")


@app.get("/api/fitness")
def get_fitness():
    from src.db import get_fitness_df
    df = get_fitness_df()
    if df.empty:
        return []
    df["timestamp"] = df["timestamp"].astype(str)
    return df.to_dict(orient="records")


@app.get("/api/live")
def get_live():
    from src.data.bybit_fetcher import fetch_candles
    try:
        df = fetch_candles("XAUUSDT", "5", 200)
        df = df.reset_index()
        df["time"] = df["time"].astype(str)
        cols = ["time", "open", "high", "low", "close", "volume",
                "ema50", "ema200", "bb_upper", "bb_lower", "rsi", "macd", "atr"]
        cols = [c for c in cols if c in df.columns]
        return df[cols].to_dict(orient="records")
    except Exception as exc:
        return {"error": str(exc)}


@app.get("/api/price")
def get_price():
    from src.data.bybit_fetcher import fetch_current_price
    try:
        return fetch_current_price("XAUUSDT")
    except Exception as exc:
        return {"error": str(exc)}


@app.get("/api/status")
def get_status():
    try:
        from src.db import get_trades_df, get_portfolio_df
        trades = get_trades_df()
        portfolio = get_portfolio_df()
        last_trade = trades.iloc[-1].to_dict() if not trades.empty else {}
        last_snap = portfolio.iloc[-1].to_dict() if not portfolio.empty else {}
        return {
            "total_trades": len(trades),
            "last_action": last_trade.get("action", "—"),
            "last_signal": last_trade.get("llm_signal", "—"),
            "last_price": last_trade.get("price", 0),
            "nav": last_snap.get("nav", 0),
            "balance": last_snap.get("balance", 0),
            "unrealized_pnl": last_snap.get("unrealized_pnl", 0),
            "server_time": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as exc:
        return {"error": str(exc)}


@app.get("/api/position")
def get_position():
    """Current open position from paper_portfolio + live unrealized PnL."""
    try:
        from sqlalchemy import text
        from src.db import get_engine
        from src.data.bybit_fetcher import fetch_current_price

        with get_engine().connect() as conn:
            row = conn.execute(text(
                "SELECT balance, position_size, avg_entry_price, realized_pnl, updated_at "
                "FROM paper_portfolio ORDER BY id DESC LIMIT 1"
            )).fetchone()
        if row is None:
            return {"position_size": 0, "side": "flat"}

        balance, pos_size, entry, realized, updated_at = row
        price = fetch_current_price("XAUUSDT").get("last", 0)
        unrealized = (price - entry) * pos_size if pos_size != 0 and entry > 0 else 0.0
        side = "long" if pos_size > 0 else ("short" if pos_size < 0 else "flat")
        notional = abs(pos_size) * price
        return {
            "side":            side,
            "position_size":   pos_size,
            "avg_entry_price": entry,
            "current_price":   price,
            "unrealized_pnl":  unrealized,
            "unrealized_pct":  (unrealized / (entry * abs(pos_size)) * 100) if (entry > 0 and pos_size != 0) else 0.0,
            "notional":        notional,
            "balance":         balance,
            "realized_pnl":    realized,
            "exposure_pct":    (notional / balance * 100) if balance > 0 else 0.0,
            "updated_at":      str(updated_at),
        }
    except Exception as exc:
        return {"error": str(exc)}


# Serve static files (HTML/JS/CSS)
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.dashboard.server:app", host="0.0.0.0", port=8080, reload=False)
