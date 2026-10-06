"""RSS collector. Fetches feeds and returns a normalised DataFrame."""
from __future__ import annotations

import feedparser
import pandas as pd

DEFAULT_FEEDS = {
    "BBC World": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "BBC Technology": "http://feeds.bbci.co.uk/news/technology/rss.xml",
    "BBC Business": "http://feeds.bbci.co.uk/news/business/rss.xml",
    "The Guardian World": "https://www.theguardian.com/world/rss",
    "The Guardian Tech": "https://www.theguardian.com/technology/rss",
    "TechCrunch": "https://techcrunch.com/feed/",
    "NDTV": "https://feeds.feedburner.com/ndtvnews-top-stories",
    "Times of India": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
}


def fetch_feed(name: str, url: str, limit: int = 100) -> list[dict]:
    parsed = feedparser.parse(url)
    rows = []
    for e in parsed.entries[:limit]:
        rows.append({
            "title": e.get("title", ""),
            "description": e.get("summary", "") or e.get("description", ""),
            "source": name,
            "url": e.get("link", ""),
            "published": e.get("published", e.get("updated", "")),
            "author": e.get("author", ""),
            "category": (e.get("tags") or [{}])[0].get("term", "") if e.get("tags") else "",
        })
    return rows


def collect(feeds: dict[str, str] | None = None, limit: int = 100) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fetch several feeds. Returns (articles, {feed_name: status_message})."""
    feeds = feeds or DEFAULT_FEEDS
    all_rows, status = [], {}
    for name, url in feeds.items():
        try:
            rows = fetch_feed(name, url, limit)
            all_rows.extend(rows)
            status[name] = f"{len(rows)} articles" if rows else "no entries (feed empty or unreachable)"
        except Exception as exc:  # network errors must not crash the app
            status[name] = f"error: {exc}"
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True).dt.tz_localize(None)
    return df, status
