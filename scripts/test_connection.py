"""
Quick smoke test for Phase 1.
Run: python scripts/test_connection.py

Checks:
  1. OANDA practice account reachable, can fetch XAU_USD candles
  2. OANDA account balance readable
  3. Groq API reachable, Llama 3.3 70B responds
  4. News fetcher returns headlines
"""

import sys
import os
import json
import logging

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check(label: str, fn):
    try:
        result = fn()
        print(f"  ✓ {label}")
        return result
    except Exception as exc:
        print(f"  ✗ {label}: {exc}")
        return None


def main():
    from dotenv import load_dotenv
    load_dotenv("config/.env")

    print("\n=== Phase 1 Connection Test ===\n")

    # ── 1. OANDA candles ──────────────────────────────────────────────────────
    print("[ OANDA Data ]")
    from src.data.oanda_fetcher import fetch_candles, fetch_current_price
    df = check("Fetch 50 M5 XAU_USD candles", lambda: fetch_candles("XAU_USD", "M5", 50))
    if df is not None and not df.empty:
        print(f"     → {len(df)} bars, last close: {df['close'].iloc[-1]:.2f}")
        print(f"     → columns: {list(df.columns)}")

    price = check("Fetch current bid/ask price", lambda: fetch_current_price("XAU_USD"))
    if price:
        print(f"     → bid={price['bid']}, ask={price['ask']}, spread={price['spread']:.4f}")

    # ── 2. OANDA account ─────────────────────────────────────────────────────
    print("\n[ OANDA Account ]")
    from src.execution.oanda_broker import OandaBroker
    broker = OandaBroker()
    summary = check("Get account summary", broker.get_account_summary)
    if summary:
        print(f"     → balance: ${summary['balance']:,.2f}, NAV: ${summary['nav']:,.2f}")
    pos = check("Get XAU_USD position", lambda: broker.get_open_position("XAU_USD"))
    if pos:
        print(f"     → units: {pos['units']} (0 = flat)")

    # ── 3. Groq LLM ──────────────────────────────────────────────────────────
    print("\n[ Groq Llama 3.3 70B ]")
    from groq import Groq
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def test_groq():
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": 'XAU/USD RSI=62, MACD positive, price above EMA200. Return JSON: {"signal":"buy|sell|hold","confidence":0.0}'
            }],
            response_format={"type": "json_object"},
            max_tokens=60,
            temperature=0.1,
        )
        return json.loads(resp.choices[0].message.content)

    groq_result = check("Groq JSON signal response", test_groq)
    if groq_result:
        print(f"     → {groq_result}")

    # ── 4. News headlines ─────────────────────────────────────────────────────
    print("\n[ News Fetcher ]")
    from src.data.news_fetcher import fetch_headlines
    headlines = check("Fetch gold headlines (GNews RSS fallback)", fetch_headlines)
    if headlines:
        print(f"     → {len(headlines)} headlines")
        print(f"     → Sample: {headlines[0][:80]}...")

    print("\n=== Done ===\n")


if __name__ == "__main__":
    main()
