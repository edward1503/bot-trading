"""OANDA v20 paper trading broker: place orders, manage positions, track account."""

import os
import logging
from typing import Optional

import oandapyV20
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades_ep
import oandapyV20.endpoints.positions as positions_ep
import oandapyV20.endpoints.accounts as accounts_ep
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails
from dotenv import load_dotenv

load_dotenv("config/.env")
logger = logging.getLogger(__name__)


class OandaBroker:
    def __init__(self):
        self.account_id = os.getenv("OANDA_ACCOUNT_ID")
        self.client = oandapyV20.API(
            access_token=os.getenv("OANDA_API_KEY"),
            environment="practice",
        )

    # ── Account ──────────────────────────────────────────────────────────────

    def get_account_summary(self) -> dict:
        r = accounts_ep.AccountSummary(self.account_id)
        self.client.request(r)
        acc = r.response["account"]
        return {
            "balance": float(acc["balance"]),
            "nav": float(acc["NAV"]),
            "unrealized_pnl": float(acc["unrealizedPL"]),
            "open_trade_count": int(acc["openTradeCount"]),
            "margin_used": float(acc.get("marginUsed", 0)),
        }

    def get_daily_pnl(self) -> float:
        """Approximate daily PnL as unrealizedPL + realized from today's closed trades."""
        summary = self.get_account_summary()
        return summary["unrealized_pnl"]

    # ── Positions ─────────────────────────────────────────────────────────────

    def get_open_position(self, instrument: str = "XAU_USD") -> dict:
        """Return current position for instrument. units > 0 = long, < 0 = short, 0 = flat."""
        try:
            r = positions_ep.PositionDetails(self.account_id, instrument)
            self.client.request(r)
            pos = r.response["position"]
            long_units = int(pos["long"]["units"])
            short_units = int(pos["short"]["units"])
            net_units = long_units + short_units
            avg_price = float(pos["long"]["averagePrice"]) if long_units > 0 else (
                float(pos["short"]["averagePrice"]) if short_units < 0 else 0.0
            )
            return {
                "instrument": instrument,
                "units": net_units,
                "long_units": long_units,
                "short_units": short_units,
                "avg_price": avg_price,
                "unrealized_pnl": float(pos["unrealizedPL"]),
            }
        except oandapyV20.exceptions.V20Error as e:
            if "POSITION_NOT_FOUND" in str(e):
                return {"instrument": instrument, "units": 0, "long_units": 0,
                        "short_units": 0, "avg_price": 0.0, "unrealized_pnl": 0.0}
            raise

    # ── Orders ────────────────────────────────────────────────────────────────

    def place_market_order(
        self,
        instrument: str,
        units: int,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
    ) -> dict:
        """Place a market order. units > 0 = buy, units < 0 = sell."""
        order_data: dict = {"instrument": instrument, "units": units}

        if stop_loss_price is not None:
            order_data["stopLossOnFill"] = StopLossDetails(
                price=str(round(stop_loss_price, 2))
            ).data

        if take_profit_price is not None:
            order_data["takeProfitOnFill"] = TakeProfitDetails(
                price=str(round(take_profit_price, 2))
            ).data

        r = orders.OrderCreate(self.account_id, data=MarketOrderRequest(**order_data).data)
        self.client.request(r)
        result = r.response.get("orderFillTransaction", r.response.get("orderCreateTransaction", {}))
        logger.info("Market order placed: %s %s units → %s", units, instrument, result.get("price", "pending"))
        return result

    def close_position(self, instrument: str = "XAU_USD") -> dict:
        """Close all open units for the given instrument."""
        pos = self.get_open_position(instrument)
        if pos["units"] == 0:
            logger.info("No position to close for %s", instrument)
            return {}

        body = {}
        if pos["long_units"] > 0:
            body["longUnits"] = "ALL"
        if pos["short_units"] < 0:
            body["shortUnits"] = "ALL"

        r = positions_ep.PositionClose(self.account_id, instrument, data=body)
        self.client.request(r)
        logger.info("Closed position for %s", instrument)
        return r.response

    def adjust_position(
        self,
        instrument: str,
        target_units: int,
        current_units: int,
    ) -> Optional[dict]:
        """Move from current_units to target_units with a single market order."""
        delta = target_units - current_units
        if delta == 0:
            return None
        return self.place_market_order(instrument, delta)
