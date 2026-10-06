"""Emerging-topic detection.

Emerging Score = 0.35*growth + 0.25*recent_activity + 0.20*source_diversity + 0.20*velocity
(each component normalised to 0..1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHTS = {"growth": 0.35, "recent_activity": 0.25, "source_diversity": 0.20, "velocity": 0.20}


def daily_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Rows = day, columns = topic_id, values = article count (noise excluded)."""
    d = df[df["cluster"] != -1].copy()
    d["day"] = d["published"].dt.normalize()
    pivot = d.pivot_table(index="day", columns="cluster", values="article_id", aggfunc="count", fill_value=0)
    if pivot.empty:
        return pivot
    full = pd.date_range(pivot.index.min(), pivot.index.max(), freq="D")
    return pivot.reindex(full, fill_value=0)


def label_for(score: float, growth: float) -> str:
    if score >= 0.60:
        return "🔥 Rising"
    if score >= 0.40 or growth > 0.15:
        return "↑ Growing"
    if growth < -0.15:
        return "↓ Declining"
    return "→ Stable"


def compute_trends(df: pd.DataFrame, topic_names: dict[int, str], window_days: int = 3) -> pd.DataFrame:
    counts = daily_counts(df)
    cols = ["topic_id", "topic", "articles", "recent", "previous", "growth_rate", "recent_activity",
            "source_diversity", "velocity", "emerging_score", "trend"]
    if counts.empty or len(counts) < 2:
        return pd.DataFrame(columns=cols)

    w = max(1, min(window_days, len(counts) // 2))
    recent_c = counts.iloc[-w:].sum()
    prev_c = counts.iloc[-2 * w:-w].sum()
    total_recent = max(recent_c.sum(), 1)

    end = counts.index.max()
    recent_mask = df["published"].dt.normalize() > (end - pd.Timedelta(days=w))
    x = np.arange(len(counts))

    rows = []
    for tid in counts.columns:
        r, p = float(recent_c[tid]), float(prev_c[tid])
        growth = (r - p) / max(p, 1.0)                       # raw growth rate
        sub = df[(df["cluster"] == tid) & recent_mask]
        all_sub = df[df["cluster"] == tid]
        diversity = sub["source"].nunique() / max(all_sub["source"].nunique(), 1) if len(sub) else 0.0
        slope = np.polyfit(x, counts[tid].values, 1)[0] if counts[tid].sum() else 0.0
        velocity_raw = slope / max(counts[tid].mean(), 1e-9)  # relative daily change
        rows.append(dict(topic_id=int(tid), topic=topic_names.get(int(tid), f"Topic {tid}"),
                         articles=int(counts[tid].sum()), recent=int(r), previous=int(p),
                         growth_rate=growth, recent_activity=r / total_recent,
                         source_diversity=diversity, velocity=velocity_raw))
    t = pd.DataFrame(rows)

    # normalise components to 0..1
    growth_n = np.clip(t["growth_rate"], 0, 2) / 2                       # +200% => 1.0
    activity_n = t["recent_activity"] / max(t["recent_activity"].max(), 1e-9)
    velocity_n = np.clip(t["velocity"], 0, 0.5) / 0.5
    t["emerging_score"] = (WEIGHTS["growth"] * growth_n + WEIGHTS["recent_activity"] * activity_n +
                           WEIGHTS["source_diversity"] * t["source_diversity"] +
                           WEIGHTS["velocity"] * velocity_n).round(3)
    t["trend"] = [label_for(s, g) for s, g in zip(t["emerging_score"], t["growth_rate"])]
    return t.sort_values("emerging_score", ascending=False).reset_index(drop=True)[cols]
