"""Text preprocessing. Two representations are maintained:

* tfidf_text      -> aggressively cleaned (lowercase, no punctuation, no stop-words)
* embedding_text  -> lightly cleaned (HTML/URLs removed only) so transformer
                     embeddings keep natural sentence structure.
"""
from __future__ import annotations

import html
import re

import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_URL = re.compile(r"https?://\S+|www\.\S+")
_NON_ALNUM = re.compile(r"[^a-z0-9\s\-]")
_WS = re.compile(r"\s+")

# A few news-specific filler words that add no topical signal.
NEWS_STOP_WORDS = {
    "said", "says", "say", "new", "report", "reports", "reported", "according",
    "news", "read", "updated", "just", "also", "will", "one", "two", "year",
    "years", "week", "today", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
}
STOP_WORDS = set(ENGLISH_STOP_WORDS) | NEWS_STOP_WORDS


def strip_html(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    text = html.unescape(text)
    if "<" in text and ">" in text:
        text = BeautifulSoup(text, "html.parser").get_text(" ")
    return text


def light_clean(text: str) -> str:
    """Minimal cleaning suitable for sentence-embedding models."""
    text = strip_html(text)
    text = _URL.sub(" ", text)
    return _WS.sub(" ", text).strip()


def _simple_lemma(tok: str) -> str:
    """Very small suffix-stripping lemmatiser (dependency-free)."""
    if len(tok) > 5 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def heavy_clean(text: str) -> str:
    """Aggressive cleaning for TF-IDF / keyword extraction."""
    text = light_clean(text).lower()
    text = _NON_ALNUM.sub(" ", text)
    toks = [_simple_lemma(t) for t in text.split()]
    toks = [t for t in toks if len(t) > 2 and t not in STOP_WORDS and not t.isdigit()]
    return " ".join(toks)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    title = df["title"].fillna("").astype(str)
    desc = df["description"].fillna("").astype(str)
    raw = (title + ". " + desc).str.strip()
    df["embedding_text"] = raw.map(light_clean)
    df["tfidf_text"] = raw.map(heavy_clean)
    return df
