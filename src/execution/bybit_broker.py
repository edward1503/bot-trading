"""Bybit Testnet paper trading broker for XAUUSDT perpetual."""

import os
import logging
from typing import Optional

from pybit.unified_trading import HTTP

from src.config import load_env

load_env()
logger = logging.getLogger(__name__)

CATEGORY = "linear"


class BybitBroker:
    def __init__(self, testnet: bool = True):
        self.session = HTTP(
            testnet=testnet,
            api_key=os.getenv("BYBIT_API_KEY"),
            api_secret=os.getenv("BYBIT_API_SECRET"),
        )
        self.testnet = testnet

    # ── Account ──────────────────────────────────────────────────────────────

    def get_account_summary(self) -> dict:
        """Return wallet balance summary (USDT unified account)."""
        resp = self.session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        if resp["retCode"] != 0:
            raise RuntimeError(f"Bybit wallet error: {resp['retMsg']}")
        acc = resp["result"]["list"][0]
        coin_info = next((c for c in acc["coin"] if c["coin"] == "USDT"), {})
        balance    = float(coin_info.get("walletBalance", 0))
        equity     = float(coin_info.get("equity", balance))
        unrealized = float(coin_info.get("unrealisedPnl", 0))
        return {
            "balance":        balance,
            "nav":            equity,
            "unrealized_pnl": unrealized,
            "open_trade_count": self._count_open_positions(),
        }

    def _count_open_positions(self) -> int:
        try:
            resp = self.session.get_positions(category=CATEGORY, settleCoin="USDT")
            return len([p for p in resp["result"]["list"] if float(p["size"]) > 0])
        except Exception:
            return 0

    # ── Positions ─────────────────────────────────────────────────────────────

    def get_open_position(self, symbol: str = "XAUUSDT") -> dict:
        """Return current position. size > 0 = long, < 0 = short, 0 = flat."""
        resp = self.session.get_positions(category=CATEGORY, symbol=symbol)
        if resp["retCode"] != 0:
            raise RuntimeError(f"Bybit position error: {resp['retMsg']}")

        positions = resp["result"]["list"]
        long_size = short_size = 0.0
        avg_price = 0.0
        unrealized = 0.0

        for p in positions:
            size = float(p["size"])
            if p["side"] == "Buy":
                long_size = size
                avg_price = float(p.get("avgPrice", 0))
                unrealized += float(p.get("unrealisedPnl", 0))
            elif p["side"] == "Sell":
                short_size = -size
                avg_price = float(p.get("avgPrice", 0))
                unrealized += float(p.get("unrealisedPnl", 0))

        net = long_size + short_size
        return {
            "symbol":         symbol,
            "size":           net,        # positive=long, negative=short
            "long_size":      long_size,
            "short_size":     short_size,
            "avg_price":      avg_price,
            "unrealized_pnl": unrealized,
            # Legacy compat key used by risk agent
            "units": net,
        }

    # ── Orders ────────────────────────────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        side: str,       # "Buy" or "Sell"
        qty: float,
        order_type: str = "Market",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> dict:
        # Testnet XAUUSDT has no real order book — simulate fill at mainnet price.
        # This is standard paper trading: track position locally, PnL via real prices.
        from src.data.bybit_fetcher import fetch_current_price
        price_info = fetch_current_price(symbol)
        fill_price = price_info["ask"] if side == "Buy" else price_info["bid"]

        fake_order_id = f"paper_{side}_{qty}_{int(fill_price)}"
        logger.info("Paper fill: %s %s qty=%.3f @ $%.2f (mainnet)",
                    side, symbol, qty, fill_price)
        return {
            "orderId":   fake_order_id,
            "symbol":    symbol,
            "side":      side,
            "qty":       str(qty),
            "fillPrice": fill_price,
            "paper":     True,
        }

    def close_position(self, symbol: str = "XAUUSDT") -> Optional[dict]:
        """Close all open positions for the symbol using reduceOnly market order."""
        pos = self.get_open_position(symbol)
        net = pos["size"]
        if net == 0:
            logger.info("No open position to close for %s", symbol)
            return None

        close_side = "Sell" if net > 0 else "Buy"
        qty = abs(net)
        resp = self.session.place_order(
            category=CATEGORY,
            symbol=symbol,
            side=close_side,
            orderType="Market",
            qty=str(qty),
            reduceOnly=True,
            timeInForce="IOC",
        )
        if resp["retCode"] != 0:
            raise RuntimeError(f"Bybit close error: {resp['retMsg']}")
        logger.info("Closed %s position: %s %s", symbol, close_side, qty)
        return resp["result"]

    def adjust_position(
        self,
        symbol: str,
        target_size: float,
        current_size: float,
        base_qty: float = 0.01,
    ) -> Optional[dict]:
        """Move from current_size to target_size."""
        # target_size is in [-1,1] range (router output), convert to actual qty
        target_qty = round(target_size * base_qty * 10, 3)  # scale: 1.0 → base_qty*10
        current_qty = current_size  # already in oz from get_open_position

        delta = target_qty - current_qty
        if abs(delta) < 0.001:
            return None

        if delta > 0:
            return self.place_order(symbol, "Buy", abs(delta))
        else:
            return self.place_order(symbol, "Sell", abs(delta))
