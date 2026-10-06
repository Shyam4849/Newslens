"""Quantitative evaluation of the clustering."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score


def evaluate(X: np.ndarray, labels: np.ndarray) -> dict:
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    out = {
        "n_articles": int(len(labels)),
        "n_clusters": int(n_clusters),
        "n_noise": int((~mask).sum()),
        "noise_pct": float((~mask).mean() * 100),
        "silhouette": None,
        "davies_bouldin": None,
    }
    if n_clusters >= 2 and mask.sum() > n_clusters:
        try:
            out["silhouette"] = float(silhouette_score(X[mask], labels[mask]))
            out["davies_bouldin"] = float(davies_bouldin_score(X[mask], labels[mask]))
        except Exception:
            pass
    return out
