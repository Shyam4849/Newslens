"""Load & validate news datasets (CSV / JSON)."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

REQUIRED = ["title"]
OPTIONAL = ["description", "source", "url", "published", "author", "category"]

# Common alternative column names -> canonical names
ALIASES = {
    "headline": "title", "text": "description", "content": "description",
    "summary": "description", "body": "description", "short_description": "description",
    "publisher": "source", "link": "url", "date": "published", "pubdate": "published",
    "publication_date": "published", "published_at": "published", "authors": "author",
}


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={k: v for k, v in ALIASES.items() if k in df.columns and v not in df.columns})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {missing}. Found: {list(df.columns)}")
    for c in OPTIONAL:
        if c not in df.columns:
            df[c] = ""
    df["title"] = df["title"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["source"] = df["source"].replace("", "Unknown").fillna("Unknown").astype(str)
    df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True).dt.tz_localize(None)
    if df["published"].isna().all():
        df["published"] = pd.Timestamp.today().normalize()
    df["published"] = df["published"].fillna(df["published"].max())
    df = df[df["title"].str.strip() != ""]
    df = df.drop_duplicates(subset=["title", "source"]).reset_index(drop=True)
    df.insert(0, "article_id", range(len(df)))
    return df[["article_id", "title", "description", "source", "url", "published", "author", "category"]]


def load_csv(path_or_buffer) -> pd.DataFrame:
    return _normalise(pd.read_csv(path_or_buffer))


def load_json(path_or_buffer) -> pd.DataFrame:
    # Supports both JSON arrays and JSON-lines (e.g. Kaggle News Category dataset)
    try:
        df = pd.read_json(path_or_buffer)
    except ValueError:
        if hasattr(path_or_buffer, "seek"):
            path_or_buffer.seek(0)
        df = pd.read_json(path_or_buffer, lines=True)
    return _normalise(df)


def load_any(name: str, data: bytes) -> pd.DataFrame:
    buf = io.BytesIO(data)
    return load_json(buf) if name.lower().endswith((".json", ".jsonl")) else load_csv(buf)


def load_sample(path: str | Path = "data/news.csv") -> pd.DataFrame:
    return load_csv(path)
