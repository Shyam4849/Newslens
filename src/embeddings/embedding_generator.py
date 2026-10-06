"""Feature extraction: Sentence-Transformer embeddings, with a TF-IDF+SVD fallback."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

DEFAULT_SBERT = "all-MiniLM-L6-v2"


def sbert_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


def tfidf_matrix(texts, max_features: int = 20000, ngram_range=(1, 2), min_df: int = 2):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range,
                          min_df=min_df if len(texts) > 50 else 1, sublinear_tf=True)
    X = vec.fit_transform(texts)
    return X, vec


def embed_tfidf_svd(tfidf_texts, dim: int = 100, seed: int = 42) -> np.ndarray:
    """Dense 'semantic-ish' vectors via LSA (TF-IDF -> truncated SVD)."""
    X, _ = tfidf_matrix(list(tfidf_texts))
    k = max(2, min(dim, X.shape[1] - 1, X.shape[0] - 1))
    Z = TruncatedSVD(n_components=k, random_state=seed).fit_transform(X)
    return normalize(Z)


def embed_sbert(texts, model_name: str = DEFAULT_SBERT, batch_size: int = 64, progress=None) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    emb = model.encode(list(texts), batch_size=batch_size, show_progress_bar=False,
                       normalize_embeddings=True, convert_to_numpy=True)
    return emb


def generate_embeddings(df, backend: str = "auto", model_name: str = DEFAULT_SBERT):
    """Returns (embeddings, backend_actually_used, note)."""
    note = ""
    if backend in ("auto", "sbert"):
        if sbert_available():
            try:
                return embed_sbert(df["embedding_text"], model_name), f"sbert:{model_name}", note
            except Exception as exc:
                note = f"Sentence-Transformers failed ({type(exc).__name__}: {exc}); used TF-IDF+SVD fallback."
        else:
            note = "sentence-transformers not installed; used TF-IDF+SVD fallback."
        if backend == "sbert" and not note:
            note = "Fell back to TF-IDF+SVD."
    return embed_tfidf_svd(df["tfidf_text"]), "tfidf-svd", note
