"""Dimensionality reduction + clustering (HDBSCAN default, K-Means optional)."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def umap_available() -> bool:
    try:
        import umap  # noqa: F401
        return True
    except Exception:
        return False


def reduce_dims(emb: np.ndarray, n_components: int, seed: int = 42, n_neighbors: int = 15,
                use_umap: bool = True) -> tuple[np.ndarray, str]:
    n = emb.shape[0]
    n_components = max(2, min(n_components, emb.shape[1], n - 2))
    if use_umap and umap_available() and n > 20:
        import umap
        red = umap.UMAP(n_components=n_components, n_neighbors=min(n_neighbors, n - 1),
                        min_dist=0.0 if n_components > 2 else 0.1, metric="cosine", random_state=seed)
        return red.fit_transform(emb), "umap"
    return PCA(n_components=n_components, random_state=seed).fit_transform(emb), "pca"


def run_hdbscan(X: np.ndarray, min_cluster_size: int = 10, min_samples: int | None = None) -> np.ndarray:
    model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                    cluster_selection_method="eom")
    return model.fit_predict(X)


def run_kmeans(X: np.ndarray, k: int = 8, seed: int = 42) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)


def cluster(X: np.ndarray, algorithm: str = "hdbscan", min_cluster_size: int = 10,
            min_samples: int | None = None, k: int = 8) -> np.ndarray:
    if algorithm == "kmeans":
        return run_kmeans(X, k)
    return run_hdbscan(X, min_cluster_size, min_samples)


def assign_new_articles(new_emb: np.ndarray, old_emb: np.ndarray, old_labels: np.ndarray,
                        threshold: float = 0.45) -> np.ndarray:
    """Phase-6 helper: assign fresh articles to an existing cluster by centroid cosine
    similarity, or -1 (candidate for a *new* cluster) if nothing is close enough."""
    ids = [c for c in np.unique(old_labels) if c != -1]
    if not ids:
        return np.full(len(new_emb), -1)
    cents = normalize(np.vstack([old_emb[old_labels == c].mean(axis=0) for c in ids]))
    sims = normalize(new_emb) @ cents.T
    best = sims.argmax(axis=1)
    return np.where(sims.max(axis=1) >= threshold, np.array(ids)[best], -1)
