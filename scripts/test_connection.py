"""
Smoke test — run: python scripts/test_connection.py

Checks:
  1. Bybit Testnet: fetch XAUUSDT candles + current price
  2. Bybit Testnet: account balance + open positions
  3. Groq: Llama 3.3 70B responds with valid JSON signal
  4. News fetcher: returns gold headlines
"""

import sys, os, json, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def check(label, fn):
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

    print("\n=== Connection Smoke Test ===\n")

    # ── 1. Bybit candles ──────────────────────────────────────────────────────
    print("[ Bybit Testnet — Market Data ]")
    from src.data.bybit_fetcher import fetch_candles, fetch_current_price

    df = check("Fetch 50 M5 XAUUSDT candles", lambda: fetch_candles("XAUUSDT", "5", 50))
    if df is not None and not df.empty:
        print(f"     → {len(df)} bars  |  last close: {df['close'].iloc[-1]:.2f}")
        print(f"     → indicators: {[c for c in df.columns if c not in ('open','high','low','close','volume')]}")

    price = check("Fetch current bid/ask price", lambda: fetch_current_price("XAUUSDT"))
    if price:
        print(f"     → last={price['last']}, bid={price['bid']}, ask={price['ask']}, spread={price['spread']:.2f}")

    # ── 2. Bybit Testnet — Unified Trading Account ────────────────────────────
    print("\n[ Bybit Testnet — Unified Trading Account ]")
    from src.execution.bybit_broker import BybitBroker
    broker = BybitBroker(testnet=True)

    summary = check("Get wallet balance (USDT)", broker.get_account_summary)
    if summary:
        print(f"     → balance: ${summary['balance']:,.2f}  NAV: ${summary['nav']:,.2f}  "
              f"unrealized PnL: ${summary['unrealized_pnl']:+.2f}")

    pos = check("Get XAUUSDT position", lambda: broker.get_open_position("XAUUSDT"))
    if pos:
        print(f"     → net size: {pos['size']} oz  (0 = flat)")

    # ── 3. Groq ───────────────────────────────────────────────────────────────
    print("\n[ Groq — Llama 3.3 70B ]")
    from groq import Groq
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def test_groq():
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": 'XAU/USD RSI=58, MACD positive, price above EMA200. Return JSON: {"signal":"buy|sell|hold","confidence":0.0,"reasoning":""}'
            }],
            response_format={"type": "json_object"},
            max_tokens=200,
            temperature=0.1,
        )
        return json.loads(resp.choices[0].message.content)

    result = check("JSON signal from Llama 3.3 70B", test_groq)
    if result:
        print(f"     → {result}")

    # ── 4. News ───────────────────────────────────────────────────────────────
    print("\n[ News Fetcher ]")
    from src.data.news_fetcher import fetch_headlines
    headlines = check("Fetch gold headlines (Google News RSS)", fetch_headlines)
    if headlines:
        print(f"     → {len(headlines)} headlines")
        if headlines:
            print(f"     → Sample: {headlines[0][:80]}")

    print("\n=== Done ===\n")
    print("Next step: fill in config/.env then run:")
    print("  python scripts/test_connection.py")
    print("  python src/rl/train.py --train-start 2022-01-01 --train-end 2023-12-31")
    print("  python src/scheduler.py")


if __name__ == "__main__":
    main()
