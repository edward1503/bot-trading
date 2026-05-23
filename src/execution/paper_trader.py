"""
Local paper trading simulator — no broker account needed.
Uses Bybit public API (no auth) for real-time XAUUSD prices.
Tracks virtual portfolio in SQLite.
"""

import logging
from datetime import datetime, timezone

from sqlalchemy import text
from src.db import get_engine

logger = logging.getLogger(__name__)

INITIAL_BALANCE = 100_000.0   # virtual USD
SPREAD_PIPS     = 0.30        # simulated spread (gold ~$0.30)
COMMISSION_PCT  = 0.00005     # 0.005% per side (Bybit-like)


class PaperTrader:
    """Simulates order execution with a virtual $100,000 portfolio."""

    def __init__(self, symbol: str = "XAUUSDT"):
        self.symbol = symbol
        self._ensure_tables()

    # ── Internal DB helpers ───────────────────────────────────────────────────

    def _ensure_tables(self):
        with get_engine().begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS paper_portfolio (
                    id INTEGER PRIMARY KEY,
                    balance REAL NOT NULL DEFAULT 100000.0,
                    position_size REAL NOT NULL DEFAULT 0.0,
                    avg_entry_price REAL NOT NULL DEFAULT 0.0,
                    realized_pnl REAL NOT NULL DEFAULT 0.0,
                    updated_at TEXT NOT NULL
                )
            """))
            # Insert initial row if empty
            row = conn.execute(text("SELECT COUNT(*) FROM paper_portfolio")).scalar()
            if row == 0:
                conn.execute(text("""
                    INSERT INTO paper_portfolio (balance, position_size, avg_entry_price, realized_pnl, updated_at)
                    VALUES (:bal, 0.0, 0.0, 0.0, :ts)
                """), {"bal": INITIAL_BALANCE, "ts": _now()})

    def _load(self) -> dict:
        with get_engine().connect() as conn:
            row = conn.execute(text(
                "SELECT balance, position_size, avg_entry_price, realized_pnl FROM paper_portfolio ORDER BY id DESC LIMIT 1"
            )).fetchone()
        return {
            "balance":       row[0],
            "position_size": row[1],   # positive=long oz, negative=short oz
            "avg_entry":     row[2],
            "realized_pnl":  row[3],
        }

    def _save(self, state: dict):
        with get_engine().begin() as conn:
            conn.execute(text("""
                UPDATE paper_portfolio SET
                    balance=:bal, position_size=:pos, avg_entry_price=:entry,
                    realized_pnl=:rpnl, updated_at=:ts
                WHERE id = (SELECT MAX(id) FROM paper_portfolio)
            """), {
                "bal":   state["balance"],
                "pos":   state["position_size"],
                "entry": state["avg_entry"],
                "rpnl":  state["realized_pnl"],
                "ts":    _now(),
            })

    # ── Public API (mirrors BybitBroker interface) ────────────────────────────

    def get_account_summary(self) -> dict:
        state  = self._load()
        price  = self._mid_price()
        pos    = state["position_size"]
        entry  = state["avg_entry"]
        unreal = (price - entry) * pos if pos != 0 and entry > 0 else 0.0
        nav    = state["balance"] + unreal
        return {
            "balance":         state["balance"],
            "nav":             nav,
            "unrealized_pnl":  unreal,
            "realized_pnl":    state["realized_pnl"],
            "open_trade_count": 1 if pos != 0 else 0,
        }

    def get_open_position(self, symbol: str = None) -> dict:
        state = self._load()
        price = self._mid_price()
        pos   = state["position_size"]
        entry = state["avg_entry"]
        unreal = (price - entry) * pos if pos != 0 and entry > 0 else 0.0
        return {
            "symbol":         self.symbol,
            "size":           pos,
            "units":          pos,
            "long_size":      pos if pos > 0 else 0.0,
            "short_size":     pos if pos < 0 else 0.0,
            "avg_price":      entry,
            "unrealized_pnl": unreal,
        }

    def adjust_position(
        self,
        symbol: str,
        target_size: float,    # -1.0 to 1.0 from router
        current_size: float,
        base_qty: float = 0.01,
    ) -> dict | None:
        target_oz = round(target_size * base_qty * 10, 3)
        delta_oz  = round(target_oz - current_size, 3)
        if abs(delta_oz) < 0.001:
            return None
        side = "Buy" if delta_oz > 0 else "Sell"
        return self._execute(side, abs(delta_oz))

    def close_position(self, symbol: str = None) -> dict | None:
        state = self._load()
        pos   = state["position_size"]
        if pos == 0:
            return None
        side = "Sell" if pos > 0 else "Buy"
        return self._execute(side, abs(pos))

    # ── Order simulation ──────────────────────────────────────────────────────

    def _execute(self, side: str, qty_oz: float) -> dict:
        """Simulate market order fill with spread + commission."""
        state    = self._load()
        mid      = self._mid_price()
        fill_px  = mid + SPREAD_PIPS / 2 if side == "Buy" else mid - SPREAD_PIPS / 2
        notional = fill_px * qty_oz
        fee      = notional * COMMISSION_PCT
        pos      = state["position_size"]

        if side == "Buy":
            new_pos   = pos + qty_oz
            avg_entry = (
                (state["avg_entry"] * abs(pos) + fill_px * qty_oz) / abs(new_pos)
                if new_pos != 0 else 0.0
            )
            state["balance"] -= fee
        else:  # Sell
            new_pos = pos - qty_oz
            # Realize PnL if reducing/flipping long
            if pos > 0:
                closed_qty = min(qty_oz, pos)
                realized   = (fill_px - state["avg_entry"]) * closed_qty - fee
                state["balance"]      += realized
                state["realized_pnl"] += realized
            else:
                state["balance"] -= fee
            avg_entry = state["avg_entry"] if new_pos != 0 else 0.0

        state["position_size"] = round(new_pos, 4)
        state["avg_entry"]     = round(avg_entry, 4)
        self._save(state)

        logger.info("PAPER %s %.4f oz @ %.2f | fee=%.4f | pos=%.4f",
                    side, qty_oz, fill_px, fee, new_pos)
        return {"side": side, "qty": qty_oz, "fill_price": fill_px, "fee": fee}

    def _mid_price(self) -> float:
        from src.data.bybit_fetcher import fetch_current_price
        try:
            return fetch_current_price(self.symbol)["last"]
        except Exception:
            # Fallback: last close from candles
            from src.data.bybit_fetcher import fetch_candles
            df = fetch_candles(self.symbol, "1", 1)
            return float(df["close"].iloc[-1]) if not df.empty else 2600.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
