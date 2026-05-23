"""SQLite persistence layer for trades, PnL snapshots, and fitness history."""

import os
import logging
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.orm import DeclarativeBase, Session

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "logs/trades.db")


class Base(DeclarativeBase):
    pass


class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    instrument = Column(String(20), default="XAU_USD")
    action = Column(String(10))
    units = Column(Integer)
    price = Column(Float)
    llm_signal = Column(String(10))
    llm_reasoning = Column(Text)
    llm_confidence = Column(Float)
    rl_action = Column(Float)
    volume_oz = Column(Float, nullable=True)   # actual oz traded
    pnl = Column(Float, nullable=True)
    portfolio_value = Column(Float, nullable=True)


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    balance = Column(Float)
    nav = Column(Float)
    unrealized_pnl = Column(Float)
    daily_pnl = Column(Float, nullable=True)


class PositionTracker(Base):
    """Tracks current open position with mainnet entry price for real PnL."""
    __tablename__ = "position_tracker"
    id         = Column(Integer, primary_key=True)
    symbol     = Column(String(20), unique=True)
    side       = Column(String(10))   # "long", "short", "flat"
    size       = Column(Float, default=0.0)
    entry_price = Column(Float, nullable=True)   # mainnet price at entry
    updated_at = Column(DateTime, default=datetime.utcnow)


class FitnessLog(Base):
    __tablename__ = "fitness_logs"
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    generation = Column(Integer)
    best_sharpe = Column(Float)
    mean_sharpe = Column(Float)
    worst_sharpe = Column(Float)


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
        _engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
        Base.metadata.create_all(_engine)
    return _engine


def update_position(symbol: str, action: str, size: float, mainnet_price: float):
    """Called after every buy/sell execution to track position with real entry price."""
    from sqlalchemy import text
    with Session(get_engine()) as session:
        row = session.execute(
            text("SELECT * FROM position_tracker WHERE symbol=:s"), {"s": symbol}
        ).fetchone()

        if action == "buy":
            side = "long"
        elif action == "sell":
            side = "short"
        else:
            return  # hold/close handled separately

        if row is None:
            session.execute(
                text("INSERT INTO position_tracker (symbol, side, size, entry_price, updated_at) VALUES (:sym,:side,:sz,:ep,:ts)"),
                {"sym": symbol, "side": side, "sz": size, "ep": mainnet_price, "ts": datetime.utcnow()},
            )
        else:
            # Average-in if same direction, reset if flipping
            existing_side = row[2] if hasattr(row, '__getitem__') else getattr(row, 'side', 'flat')
            existing_size = float(row[3] if hasattr(row, '__getitem__') else getattr(row, 'size', 0))
            existing_entry = float(row[4] if hasattr(row, '__getitem__') else getattr(row, 'entry_price', mainnet_price) or mainnet_price)

            if side == existing_side and existing_size > 0:
                # Average entry price
                total = existing_size + size
                avg_entry = (existing_entry * existing_size + mainnet_price * size) / total
                session.execute(
                    text("UPDATE position_tracker SET side=:side, size=:sz, entry_price=:ep, updated_at=:ts WHERE symbol=:sym"),
                    {"side": side, "sz": total, "ep": avg_entry, "ts": datetime.utcnow(), "sym": symbol},
                )
            else:
                # New direction — reset
                session.execute(
                    text("UPDATE position_tracker SET side=:side, size=:sz, entry_price=:ep, updated_at=:ts WHERE symbol=:sym"),
                    {"side": side, "sz": size, "ep": mainnet_price, "ts": datetime.utcnow(), "sym": symbol},
                )
        session.commit()


def close_position_tracker(symbol: str):
    from sqlalchemy import text
    with Session(get_engine()) as session:
        session.execute(
            text("UPDATE position_tracker SET side='flat', size=0, entry_price=NULL, updated_at=:ts WHERE symbol=:sym"),
            {"ts": datetime.utcnow(), "sym": symbol},
        )
        session.commit()


def calc_unrealized_pnl(symbol: str, current_mainnet_price: float, contract_size: float = 0.01) -> float:
    """PnL = size × (current_price - entry_price) × contract_size. Shorts are negated."""
    from sqlalchemy import text
    with Session(get_engine()) as session:
        row = session.execute(
            text("SELECT side, size, entry_price FROM position_tracker WHERE symbol=:s"), {"s": symbol}
        ).fetchone()
    if row is None or row[0] == "flat" or row[2] is None or row[1] == 0:
        return 0.0
    side, size, entry = row[0], float(row[1]), float(row[2])
    pnl = size * (current_mainnet_price - entry) * contract_size
    return round(pnl if side == "long" else -pnl, 4)


def log_trade(trade_data: dict):
    with Session(get_engine()) as session:
        trade = Trade(**{k: v for k, v in trade_data.items() if hasattr(Trade, k)})
        session.add(trade)
        session.commit()


def log_portfolio_snapshot(summary: dict, daily_pnl: float = None):
    with Session(get_engine()) as session:
        snap = PortfolioSnapshot(
            balance=summary.get("balance"),
            nav=summary.get("nav"),
            unrealized_pnl=summary.get("unrealized_pnl"),
            daily_pnl=daily_pnl,
        )
        session.add(snap)
        session.commit()


def log_fitness(cycle: dict):
    with Session(get_engine()) as session:
        fl = FitnessLog(
            generation=cycle.get("generation"),
            best_sharpe=cycle.get("best_sharpe"),
            mean_sharpe=cycle.get("mean_sharpe"),
            worst_sharpe=cycle.get("worst_sharpe"),
        )
        session.add(fl)
        session.commit()


def get_trades_df():
    import pandas as pd
    with Session(get_engine()) as session:
        from sqlalchemy import text
        result = session.execute(text("SELECT * FROM trades ORDER BY timestamp DESC"))
        rows = result.fetchall()
        cols = result.keys()
    return pd.DataFrame(rows, columns=list(cols)) if rows else pd.DataFrame()


def get_portfolio_df():
    import pandas as pd
    with Session(get_engine()) as session:
        from sqlalchemy import text
        result = session.execute(text("SELECT * FROM portfolio_snapshots ORDER BY timestamp"))
        rows = result.fetchall()
        cols = result.keys()
    return pd.DataFrame(rows, columns=list(cols)) if rows else pd.DataFrame()


def get_fitness_df():
    import pandas as pd
    with Session(get_engine()) as session:
        from sqlalchemy import text
        result = session.execute(text("SELECT * FROM fitness_logs ORDER BY timestamp"))
        rows = result.fetchall()
        cols = result.keys()
    return pd.DataFrame(rows, columns=list(cols)) if rows else pd.DataFrame()
