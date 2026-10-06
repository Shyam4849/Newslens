"""End-to-end NewsLens pipeline:
articles -> preprocessing -> embeddings -> UMAP -> HDBSCAN -> topics -> trends -> duplicates
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.clustering.cluster_evaluation import evaluate
from src.clustering.cluster_model import cluster, reduce_dims
from src.embeddings.embedding_generator import generate_embeddings
from src.preprocessing.text_cleaner import preprocess_dataframe
from src.similarity.duplicate_detector import find_story_groups
from src.topics.keyword_extractor import cluster_keywords, npmi_coherence
from src.topics.topic_namer import name_topics
from src.trends.emerging_topics import compute_trends, daily_counts


@dataclass
class Config:
    embedding_backend: str = "auto"       # auto | sbert | tfidf
    algorithm: str = "hdbscan"            # hdbscan | kmeans
    min_cluster_size: int = 40
    min_samples: int | None = None
    k: int = 8                            # only for kmeans
    cluster_dims: int = 5
    use_umap: bool = True
    trend_window_days: int = 3
    dup_threshold: float = 0.80
    use_llm_names: bool = False
    seed: int = 42


@dataclass
class Result:
    df: pd.DataFrame
    topics: pd.DataFrame
    trends: pd.DataFrame
    counts: pd.DataFrame
    metrics: dict
    keywords: dict
    topic_names: dict
    notes: list[str] = field(default_factory=list)
    embeddings: np.ndarray | None = None


def run_pipeline(articles: pd.DataFrame, cfg: Config, progress=None) -> Result:
    def step(msg, frac):
        if progress:
            progress(frac, msg)

    notes: list[str] = []
    step("Cleaning text…", 0.05)
    df = preprocess_dataframe(articles)
    if len(df) < 30:
        raise ValueError("Need at least 30 articles to cluster meaningfully.")

    step("Generating embeddings…", 0.15)
    backend = "tfidf" if cfg.embedding_backend == "tfidf" else cfg.embedding_backend
    if backend == "tfidf":
        from src.embeddings.embedding_generator import embed_tfidf_svd
        emb, used, note = embed_tfidf_svd(df["tfidf_text"]), "tfidf-svd", ""
    else:
        emb, used, note = generate_embeddings(df, backend)
    if note:
        notes.append(note)

    step("Reducing dimensions (UMAP)…", 0.45)
    Xc, red_c = reduce_dims(emb, cfg.cluster_dims, cfg.seed, use_umap=cfg.use_umap)
    X2, red_2 = reduce_dims(emb, 2, cfg.seed, use_umap=cfg.use_umap)
    if red_c == "pca":
        notes.append("umap-learn not available (or disabled); used PCA for dimensionality reduction.")

    step("Clustering…", 0.60)
    labels = cluster(Xc, cfg.algorithm, cfg.min_cluster_size, cfg.min_samples, cfg.k)
    df["cluster"] = labels
    df["x"], df["y"] = X2[:, 0], X2[:, 1]

    step("Discovering topics…", 0.72)
    kw = cluster_keywords(df["tfidf_text"], labels, top_n=10)
    titles = {c: df.loc[df.cluster == c, "title"].head(5).tolist() for c in kw}
    names = name_topics(kw, titles, cfg.use_llm_names)
    # make names unique
    seen: dict[str, int] = {}
    for cid, n in list(names.items()):
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            names[cid] = f"{n} ({seen[n]})"
    df["topic_name"] = df["cluster"].map(names)
    coh = npmi_coherence(kw, df["tfidf_text"])

    step("Evaluating & detecting trends…", 0.82)
    metrics = evaluate(Xc, labels)
    metrics["avg_coherence"] = float(np.mean(list(coh.values()))) if coh else None
    metrics["embedding_backend"] = used
    metrics["reduction"] = red_c
    metrics["algorithm"] = cfg.algorithm

    trends = compute_trends(df, names, cfg.trend_window_days)
    counts = daily_counts(df)

    step("Detecting duplicate stories…", 0.92)
    df = find_story_groups(df, emb, cfg.dup_threshold)

    # per-topic summary table
    total = max(len(df), 1)
    rows = []
    for cid, g in df[df.cluster != -1].groupby("cluster"):
        rows.append(dict(topic_id=int(cid), topic=names[int(cid)], articles=len(g),
                         share_pct=100 * len(g) / total, sources=g["source"].nunique(),
                         keywords=", ".join(kw[int(cid)][:8]), coherence=coh.get(int(cid), 0.0),
                         first_seen=g["published"].min(), last_seen=g["published"].max()))
    topics = pd.DataFrame(rows).sort_values("articles", ascending=False).reset_index(drop=True)
    if not trends.empty and not topics.empty:
        topics = topics.merge(trends[["topic_id", "trend", "emerging_score", "growth_rate"]],
                              on="topic_id", how="left")
    step("Done", 1.0)
    return Result(df=df, topics=topics, trends=trends, counts=counts, metrics=metrics,
                  keywords=kw, topic_names=names, notes=notes, embeddings=emb)
