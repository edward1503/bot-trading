"""Fetch gold/XAUUSD-related news headlines for LLM sentiment analysis."""

import os
import logging
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

load_dotenv("config/.env")
logger = logging.getLogger(__name__)

_GOLD_KEYWORDS = "gold OR XAUUSD OR \"gold price\" OR \"safe haven\""
_RSS_URL = "https://gnews.io/api/v4/search?q=gold+price&lang=en&max=10&apikey={key}"


def fetch_headlines(max_articles: int = 10) -> list[str]:
    """Return list of recent gold-related headlines. Falls back to RSS if NewsAPI unavailable."""
    headlines = _fetch_newsapi(max_articles)
    if not headlines:
        headlines = _fetch_gnews_rss(max_articles)
    if not headlines:
        logger.warning("No headlines fetched; using empty list")
    return headlines[:max_articles]


def _fetch_newsapi(max_articles: int) -> list[str]:
    api_key = os.getenv("NEWS_API_KEY", "")
    if not api_key:
        return []
    try:
        from_date = (datetime.now(timezone.utc) - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": _GOLD_KEYWORDS,
            "from": from_date,
            "sortBy": "publishedAt",
            "pageSize": max_articles,
            "language": "en",
            "apiKey": api_key,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        return [a["title"] for a in articles if a.get("title")]
    except Exception as exc:
        logger.warning("NewsAPI fetch failed: %s", exc)
        return []


def _fetch_gnews_rss(max_articles: int) -> list[str]:
    """Parse Google News RSS feed as free fallback (no API key required)."""
    try:
        import xml.etree.ElementTree as ET
        url = "https://news.google.com/rss/search?q=gold+price+XAUUSD&hl=en-US&gl=US&ceid=US:en"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        titles = [item.find("title").text for item in root.iter("item") if item.find("title") is not None]
        return titles[:max_articles]
    except Exception as exc:
        logger.warning("GNews RSS fetch failed: %s", exc)
        return []
