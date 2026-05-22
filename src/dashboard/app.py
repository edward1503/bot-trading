"""
Streamlit dashboard — 4 tabs:
  1. PnL Curve + metrics
  2. Trade Log + signal debug
  3. RL Training metrics (fitness history)
  4. Live XAUUSD chart with indicators
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

load_dotenv("config/.env")

st.set_page_config(
    page_title="XAUUSD Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

AUTO_REFRESH_SECS = 60


def load_data():
    from src.db import get_trades_df, get_portfolio_df, get_fitness_df
    return get_trades_df(), get_portfolio_df(), get_fitness_df()


def fetch_live_chart_data():
    try:
        from src.data.oanda_fetcher import fetch_candles
        return fetch_candles("XAU_USD", "M5", 100)
    except Exception as exc:
        st.warning(f"Live data unavailable: {exc}")
        return pd.DataFrame()


# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 XAUUSD 24/7 Paper Trading Bot")
st.caption("Groq Llama 3.3 70B + PPO + Evolutionary RL (CMA-ES) | OANDA Practice")

tab1, tab2, tab3, tab4 = st.tabs(["PnL & Metrics", "Trade Log", "RL Fitness", "Live Chart"])

trades_df, portfolio_df, fitness_df = load_data()

# ── Tab 1: PnL Curve ──────────────────────────────────────────────────────────
with tab1:
    if portfolio_df.empty:
        st.info("No portfolio data yet. Start the trading bot to see PnL.")
    else:
        portfolio_df["timestamp"] = pd.to_datetime(portfolio_df["timestamp"])
        portfolio_df = portfolio_df.sort_values("timestamp")

        # KPI row
        latest = portfolio_df.iloc[-1]
        initial = portfolio_df.iloc[0]["balance"]
        total_return_pct = ((latest["nav"] - initial) / initial) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("NAV", f"${latest['nav']:,.2f}", f"{total_return_pct:+.2f}%")
        col2.metric("Balance", f"${latest['balance']:,.2f}")
        col3.metric("Unrealized PnL", f"${latest['unrealized_pnl']:+.2f}")
        if not trades_df.empty:
            col4.metric("Total Trades", len(trades_df))

        # Equity curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=portfolio_df["timestamp"], y=portfolio_df["nav"],
            mode="lines", name="NAV", line=dict(color="#00CC96", width=2)
        ))
        fig.add_hline(y=initial, line_dash="dash", line_color="gray", annotation_text="Initial")
        fig.update_layout(
            title="Equity Curve (NAV over time)",
            xaxis_title="Time", yaxis_title="USD",
            template="plotly_dark", height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Daily PnL bar chart
        if "daily_pnl" in portfolio_df.columns:
            daily = portfolio_df.dropna(subset=["daily_pnl"])
            if not daily.empty:
                fig2 = px.bar(
                    daily, x="timestamp", y="daily_pnl",
                    color=daily["daily_pnl"].apply(lambda x: "profit" if x >= 0 else "loss"),
                    color_discrete_map={"profit": "#00CC96", "loss": "#EF553B"},
                    title="Daily PnL",
                    template="plotly_dark",
                )
                st.plotly_chart(fig2, use_container_width=True)


# ── Tab 2: Trade Log ──────────────────────────────────────────────────────────
with tab2:
    if trades_df.empty:
        st.info("No trades yet.")
    else:
        # Summary metrics
        if "pnl" in trades_df.columns:
            closed = trades_df.dropna(subset=["pnl"])
            if not closed.empty:
                wins = (closed["pnl"] > 0).sum()
                win_rate = wins / len(closed) * 100
                total_pnl = closed["pnl"].sum()
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Closed Trades", len(closed))
                col2.metric("Win Rate", f"{win_rate:.1f}%")
                col3.metric("Total Realized PnL", f"${total_pnl:+,.2f}")

        # Filter options
        signal_filter = st.multiselect(
            "Filter by LLM signal", ["buy", "sell", "hold"],
            default=["buy", "sell", "hold"]
        )
        display_df = trades_df[trades_df["llm_signal"].isin(signal_filter)].copy()

        # Style: color rows by signal
        def row_color(row):
            if row.get("llm_signal") == "buy":
                return ["background-color: rgba(0,204,150,0.15)"] * len(row)
            elif row.get("llm_signal") == "sell":
                return ["background-color: rgba(239,85,59,0.15)"] * len(row)
            return [""] * len(row)

        cols_to_show = ["timestamp", "action", "units", "price", "llm_signal",
                        "llm_confidence", "rl_action", "llm_reasoning", "pnl"]
        show_cols = [c for c in cols_to_show if c in display_df.columns]

        st.dataframe(
            display_df[show_cols].style.apply(row_color, axis=1),
            use_container_width=True,
            height=500,
        )


# ── Tab 3: RL Fitness ─────────────────────────────────────────────────────────
with tab3:
    if fitness_df.empty:
        st.info("No evolutionary RL fitness data yet. Fitness is computed weekly.")
    else:
        fitness_df["timestamp"] = pd.to_datetime(fitness_df["timestamp"])

        col1, col2 = st.columns(2)
        latest_f = fitness_df.iloc[-1]
        col1.metric("Latest Best Sharpe", f"{latest_f['best_sharpe']:.3f}")
        col2.metric("Generation", int(latest_f["generation"]))

        # Fitness over generations
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fitness_df["generation"], y=fitness_df["best_sharpe"],
            mode="lines+markers", name="Best Sharpe", line=dict(color="#AB63FA", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=fitness_df["generation"], y=fitness_df["mean_sharpe"],
            mode="lines", name="Mean Sharpe", line=dict(color="#FFA15A", dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=fitness_df["generation"], y=fitness_df["worst_sharpe"],
            mode="lines", name="Worst Sharpe", line=dict(color="#636EFA", dash="dot")
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Sharpe=0")
        fig.update_layout(
            title="Evolutionary Fitness per Generation (Sharpe Ratio)",
            xaxis_title="Generation", yaxis_title="Sharpe Ratio",
            template="plotly_dark", height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(fitness_df, use_container_width=True)


# ── Tab 4: Live Chart ─────────────────────────────────────────────────────────
with tab4:
    st.caption("Live XAUUSD M5 chart (refreshes every 60s)")

    if st.button("Refresh Now"):
        st.rerun()

    df_live = fetch_live_chart_data()

    if df_live.empty:
        st.info("Live data unavailable.")
    else:
        df_live = df_live.reset_index()

        fig = go.Figure()

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df_live["time"], open=df_live["open"], high=df_live["high"],
            low=df_live["low"], close=df_live["close"],
            name="XAU/USD", increasing_line_color="#00CC96", decreasing_line_color="#EF553B"
        ))

        # EMA lines
        if "ema50" in df_live.columns:
            fig.add_trace(go.Scatter(
                x=df_live["time"], y=df_live["ema50"],
                mode="lines", name="EMA50", line=dict(color="#FFA15A", width=1)
            ))
        if "ema200" in df_live.columns:
            fig.add_trace(go.Scatter(
                x=df_live["time"], y=df_live["ema200"],
                mode="lines", name="EMA200", line=dict(color="#AB63FA", width=1)
            ))

        # BB bands
        if "bb_upper" in df_live.columns:
            fig.add_trace(go.Scatter(
                x=df_live["time"], y=df_live["bb_upper"],
                mode="lines", name="BB Upper", line=dict(color="rgba(99,110,250,0.5)", width=1)
            ))
            fig.add_trace(go.Scatter(
                x=df_live["time"], y=df_live["bb_lower"],
                mode="lines", name="BB Lower", line=dict(color="rgba(99,110,250,0.5)", width=1),
                fill="tonexty", fillcolor="rgba(99,110,250,0.05)"
            ))

        fig.update_layout(
            title="XAU/USD M5 Live Chart",
            xaxis_title="Time", yaxis_title="Price (USD)",
            template="plotly_dark", height=500,
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # RSI subplot
        if "rsi" in df_live.columns:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=df_live["time"], y=df_live["rsi"],
                mode="lines", name="RSI(14)", line=dict(color="#19D3F3", width=1.5)
            ))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray")
            fig_rsi.update_layout(
                title="RSI(14)", template="plotly_dark", height=200,
                yaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig_rsi, use_container_width=True)

# ── Auto-refresh ──────────────────────────────────────────────────────────────
st.markdown(f"*Auto-refreshes every {AUTO_REFRESH_SECS}s*")
time.sleep(AUTO_REFRESH_SECS)
st.rerun()
