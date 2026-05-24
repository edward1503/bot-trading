"""FastAPI backend for the trading dashboard."""

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.config import PROJECT_ROOT

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


_ALLOWED_INTERVALS = {"1", "5", "15", "30", "60", "240", "D", "W", "M"}


@app.get("/api/live")
def get_live(interval: str = "5", limit: int = 200):
    from src.data.bybit_fetcher import fetch_candles
    iv = interval.upper() if interval.lower() in ("d", "w", "m") else interval
    if iv not in _ALLOWED_INTERVALS:
        return {"error": f"invalid interval '{interval}'; allowed: {sorted(_ALLOWED_INTERVALS)}"}
    limit = max(50, min(int(limit), 1000))
    try:
        df = fetch_candles("XAUUSDT", iv, limit)
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


@app.get("/api/health")
def get_health():
    """Liveness probe: returns last loop timestamp + staleness flag (stale if >15 min)."""
    from sqlalchemy import text
    from src.db import get_engine

    try:
        with get_engine().connect() as conn:
            last_trade_ts = conn.execute(text("SELECT MAX(timestamp) FROM trades")).scalar()
            last_snap_ts = conn.execute(text("SELECT MAX(timestamp) FROM portfolio_snapshots")).scalar()
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}

    now = datetime.now(timezone.utc)
    last_loop_ts = max(filter(None, [last_trade_ts, last_snap_ts]), default=None)
    if last_loop_ts is None:
        return {"status": "unknown", "reason": "no data yet", "now": now.isoformat()}

    last_loop_dt = last_loop_ts if isinstance(last_loop_ts, datetime) else datetime.fromisoformat(str(last_loop_ts))
    if last_loop_dt.tzinfo is None:
        last_loop_dt = last_loop_dt.replace(tzinfo=timezone.utc)
    age_sec = (now - last_loop_dt).total_seconds()
    stale = age_sec > 15 * 60
    return {
        "status": "stale" if stale else "ok",
        "last_loop": last_loop_dt.isoformat(),
        "age_seconds": int(age_sec),
        "now": now.isoformat(),
    }


# Serve static files (HTML/JS/CSS)
static_dir = PROJECT_ROOT / "src" / "dashboard" / "static"
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.dashboard.server:app", host="0.0.0.0", port=8080, reload=False)
