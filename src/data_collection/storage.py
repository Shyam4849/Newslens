"""SQLite persistence for collected articles (used by the RSS collector)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/newslens.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    source TEXT,
    url TEXT UNIQUE,
    published TEXT,
    author TEXT,
    category TEXT,
    fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def _conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute(SCHEMA)
    return con


def save_articles(df: pd.DataFrame, db_path: Path = DB_PATH) -> int:
    """Insert new articles (deduplicated by URL). Returns number inserted."""
    if df.empty:
        return 0
    con = _conn(db_path)
    before = con.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    rows = [
        (r.title, r.description, r.source, r.url or None, str(r.published), r.author, r.category)
        for r in df.itertuples()
    ]
    con.executemany(
        "INSERT OR IGNORE INTO articles (title, description, source, url, published, author, category) "
        "VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()
    after = con.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    con.close()
    return after - before


def load_articles(db_path: Path = DB_PATH) -> pd.DataFrame:
    if not Path(db_path).exists():
        return pd.DataFrame()
    con = _conn(db_path)
    df = pd.read_sql_query(
        "SELECT title, description, source, url, published, author, category FROM articles", con)
    con.close()
    return df


def count_articles(db_path: Path = DB_PATH) -> int:
    if not Path(db_path).exists():
        return 0
    con = _conn(db_path)
    n = con.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    con.close()
    return n
