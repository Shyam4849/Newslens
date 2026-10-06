"""Cluster -> keywords (class-based TF-IDF) and topic coherence (NPMI)."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def cluster_keywords(texts, labels, top_n: int = 10) -> dict[int, list[str]]:
    """Treat every cluster as ONE document, then TF-IDF across clusters (c-TF-IDF style).
    Words frequent in a cluster but rare elsewhere rank highest."""
    labels = np.asarray(labels)
    ids = sorted(c for c in set(labels) if c != -1)
    if not ids:
        return {}
    docs = [" ".join(np.asarray(texts)[labels == c]) for c in ids]
    cv = CountVectorizer(ngram_range=(1, 2), min_df=1, max_df=1.0)
    counts = cv.fit_transform(docs)
    tfidf = TfidfTransformer(sublinear_tf=True).fit_transform(counts)
    vocab = np.array(cv.get_feature_names_out())
    result = {}
    for row, cid in enumerate(ids):
        scores = tfidf[row].toarray().ravel()
        order = scores.argsort()[::-1]
        picked: list[str] = []
        for i in order:
            w = vocab[i]
            # skip a bigram if it is made of words already picked (reduces redundancy)
            if any(w in p or p in w for p in picked):
                continue
            picked.append(w)
            if len(picked) == top_n:
                break
        result[int(cid)] = picked
    return result


def keybert_keywords(texts, labels, top_n: int = 8) -> dict[int, list[str]] | None:
    """Optional semantic keywords via KeyBERT (returns None if not installed)."""
    try:
        from keybert import KeyBERT
    except Exception:
        return None
    kb = KeyBERT()
    labels = np.asarray(labels)
    out = {}
    for c in sorted(set(labels) - {-1}):
        doc = " ".join(np.asarray(texts)[labels == c][:60])
        out[int(c)] = [k for k, _ in kb.extract_keywords(doc, keyphrase_ngram_range=(1, 2), top_n=top_n)]
    return out


def npmi_coherence(keywords: dict[int, list[str]], texts, top_k: int = 8) -> dict[int, float]:
    """Topic coherence (NPMI) using document co-occurrence over the whole corpus."""
    texts = list(texts)
    n_docs = len(texts)
    cv = CountVectorizer(ngram_range=(1, 2), binary=True, min_df=1)
    D = cv.fit_transform(texts).tocsc()
    vocab = cv.vocabulary_
    eps = 1e-12
    out = {}
    for cid, words in keywords.items():
        idx = [vocab[w] for w in words[:top_k] if w in vocab]
        if len(idx) < 2:
            out[cid] = 0.0
            continue
        scores = []
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                da, db = D[:, idx[a]], D[:, idx[b]]
                p_a, p_b = da.sum() / n_docs, db.sum() / n_docs
                p_ab = da.multiply(db).sum() / n_docs
                if p_ab <= 0:
                    scores.append(-1.0)
                    continue
                pmi = np.log(p_ab / (p_a * p_b + eps) + eps)
                scores.append(float(pmi / (-np.log(p_ab + eps))) if p_ab < 1 else 1.0)
        out[cid] = float(np.mean(scores))
    return out
