"""Near-duplicate / same-story detection using cosine similarity."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


def find_story_groups(df: pd.DataFrame, emb: np.ndarray, threshold: float = 0.80,
                      max_days_apart: int = 3) -> pd.DataFrame:
    """Groups articles whose cosine similarity exceeds `threshold` (connected components),
    restricted to articles published close in time. Adds a `story_id` column (-1 = unique)."""
    n = len(df)
    E = normalize(emb)
    days = df["published"].dt.normalize().values.astype("datetime64[D]").astype(int)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    B = 512
    for s in range(0, n, B):
        sims = E[s:s + B] @ E.T
        for i in range(sims.shape[0]):
            gi = s + i
            cand = np.where((sims[i] >= threshold) & (np.abs(days - days[gi]) <= max_days_apart))[0]
            for j in cand:
                if j > gi:
                    ra, rb = find(gi), find(j)
                    if ra != rb:
                        parent[rb] = ra
    roots = np.array([find(i) for i in range(n)])
    sizes = pd.Series(roots).map(pd.Series(roots).value_counts()).values
    story = np.where(sizes >= 2, roots, -1)
    # relabel to 1..k ordered by group size
    out = df.copy()
    out["story_id"] = -1
    mask = story != -1
    if mask.any():
        s = pd.Series(story[mask])
        order = s.value_counts().index.tolist()
        mapping = {r: i + 1 for i, r in enumerate(order)}
        out.loc[mask, "story_id"] = s.map(mapping).values
    return out
