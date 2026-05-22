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
