"""NewsLens — Intelligent News Article Clustering & Emerging Topic Discovery (Streamlit frontend)."""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

from src.clustering.cluster_model import umap_available
from src.data_collection import rss_collector, storage
from src.data_collection.data_loader import load_any, load_sample
from src.embeddings.embedding_generator import sbert_available
from src.pipeline import Config, run_pipeline
from src.visualization.topic_map import distribution_chart, source_bar, topic_map, trend_lines

st.set_page_config(page_title="NewsLens", page_icon="📰", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
div[data-testid="stMetric"] {background:#1a1f2b; border:1px solid #2a3142; padding:14px 16px; border-radius:12px;}
.hero h1 {margin-bottom:0; font-size:2.1rem;}
.hero p {color:#9aa3b5; margin-top:2px;}
.pill {display:inline-block; padding:2px 10px; border-radius:999px; background:#2a3142; font-size:.78rem; margin-right:6px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>📰 NewsLens</h1>'
            '<p>Intelligent news clustering &amp; emerging-topic discovery</p></div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ sidebar
sb = st.sidebar
sb.header("1 · Data")
source_choice = sb.radio("Source", ["Sample dataset", "Upload CSV / JSON", "Collected (RSS → SQLite)"], label_visibility="collapsed")

articles: pd.DataFrame | None = None
try:
    if source_choice == "Sample dataset":
        if Path("data/news.csv").exists():
            articles = load_sample("data/news.csv")
        else:
            sb.error("data/news.csv not found. Run: python scripts/generate_sample_data.py")
    elif source_choice == "Upload CSV / JSON":
        up = sb.file_uploader("Needs a `title` column (plus description, source, published, url…)", type=["csv", "json", "jsonl"])
        if up is not None:
            articles = load_any(up.name, up.getvalue())
    else:
        raw = storage.load_articles()
        if raw.empty:
            sb.info("No collected articles yet — fetch some in the **Data sources** tab.")
        else:
            from src.data_collection.data_loader import _normalise
            articles = _normalise(raw)
except Exception as exc:
    sb.error(f"Could not load data: {exc}")

sb.header("2 · Model")
backend_opts = {"Auto (Sentence-Transformers if installed)": "auto", "Sentence-Transformers (all-MiniLM-L6-v2)": "sbert",
                "TF-IDF + SVD (fast, offline)": "tfidf"}
backend = backend_opts[sb.selectbox("Embeddings", list(backend_opts))]
algorithm = sb.radio("Clustering", ["HDBSCAN", "K-Means"], horizontal=True).lower().replace("-", "")
if algorithm == "hdbscan":
    mcs = sb.slider("Min cluster size", 5, 100, 40, help="Smaller → more, finer topics.")
    k = 8
else:
    k = sb.slider("K (clusters)", 2, 30, 8)
    mcs = 40
use_umap = sb.checkbox("Use UMAP for reduction", value=True, disabled=not umap_available(),
                       help="Falls back to PCA if umap-learn isn't installed.")
sb.header("3 · Trends & duplicates")
window = sb.slider("Trend window (days)", 1, 7, 3, help="Recent window vs the window right before it.")
dup_thr = sb.slider("Duplicate similarity", 0.60, 0.99, 0.80, 0.01)
use_llm = sb.checkbox("LLM topic names (needs ANTHROPIC_API_KEY)", value=False)

run = sb.button("▶ Run analysis", type="primary", width="stretch")
sb.caption(f"Sentence-Transformers: {'✅' if sbert_available() else '❌'} · UMAP: {'✅' if umap_available() else '❌'}")


@st.cache_data(show_spinner=False)
def _run(df: pd.DataFrame, cfg_dict: dict):
    return run_pipeline(df, Config(**cfg_dict))


cfg_dict = dict(embedding_backend=backend, algorithm=algorithm, min_cluster_size=mcs, k=k, use_umap=use_umap,
                trend_window_days=window, dup_threshold=dup_thr, use_llm_names=use_llm)

# ------------------------------------------------------------------ run
tab_names = ["📊 Overview", "🧩 Topics", "🔥 Emerging", "🗺️ Topic map", "📰 Articles", "🔁 Duplicates", "📈 Model metrics", "🛰️ Data sources"]
tabs = st.tabs(tab_names)

if run and articles is not None:
    bar = st.progress(0.0, text="Starting…")
    try:
        # progress can't be used inside cache_data reliably -> run directly, then store
        res = run_pipeline(articles, Config(**cfg_dict), progress=lambda f, m: bar.progress(f, text=m))
        st.session_state["result"] = res
        st.session_state["cfg"] = cfg_dict
    except Exception as exc:
        st.session_state.pop("result", None)
        bar.empty()
        st.error(f"Pipeline failed: {exc}")
    else:
        bar.empty()

res = st.session_state.get("result")

# ------------------------------------------------------------------ data-sources tab (always available)
with tabs[7]:
    st.subheader("Collect live news via RSS")
    st.caption("Articles are stored in a local SQLite DB (`data/newslens.db`). Then pick **Collected** in the sidebar.")
    chosen = st.multiselect("Feeds", list(rss_collector.DEFAULT_FEEDS), default=list(rss_collector.DEFAULT_FEEDS)[:4])
    custom = st.text_input("Custom feed (Name|URL)", placeholder="My Feed|https://example.com/rss.xml")
    c1, c2 = st.columns([1, 3])
    if c1.button("Fetch feeds"):
        feeds = {n: rss_collector.DEFAULT_FEEDS[n] for n in chosen}
        if "|" in custom:
            n, u = custom.split("|", 1)
            feeds[n.strip()] = u.strip()
        with st.spinner("Fetching…"):
            new_df, status = rss_collector.collect(feeds)
            added = storage.save_articles(new_df) if not new_df.empty else 0
        st.success(f"Stored {added} new articles. Database now holds {storage.count_articles()}.")
        st.json(status)
    c2.metric("Articles in database", storage.count_articles())
    if source_choice != "Collected (RSS → SQLite)":
        st.info("Switch the sidebar data source to **Collected (RSS → SQLite)** to analyse them.")
    st.markdown("**Expected dataset format** (CSV/JSON): `title`, `description`, `source`, `url`, `published`, `author`, `category`. "
                "Common aliases such as `headline`, `content`, `date` are mapped automatically.")

if res is None:
    for i in range(7):
        with tabs[i]:
            if articles is None:
                st.info("Choose or upload a dataset in the sidebar, then press **Run analysis**.")
            else:
                st.info(f"Loaded **{len(articles):,}** articles from **{articles['source'].nunique()}** sources. "
                        "Press **▶ Run analysis** in the sidebar to discover topics.")
                st.dataframe(articles.head(20)[["title", "source", "published"]], width="stretch", hide_index=True)
    st.stop()

df, topics, trends, m = res.df, res.topics, res.trends, res.metrics
for n in res.notes:
    st.warning(n)

# ------------------------------------------------------------------ overview
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Articles", f"{m['n_articles']:,}")
    c[1].metric("Topics", m["n_clusters"])
    c[2].metric("Sources", df["source"].nunique())
    n_emerg = int((trends["trend"] == "🔥 Rising").sum()) if not trends.empty else 0
    c[3].metric("Emerging", n_emerg)
    c[4].metric("Noise", f"{m['n_noise']} ({m['noise_pct']:.0f}%)")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("#### 🔥 Emerging topics")
        if trends.empty:
            st.info("Not enough date range to compute trends.")
        else:
            for _, r in trends.head(4).iterrows():
                st.markdown(f"**{r['trend']}  {r['topic']}** — {r['growth_rate']*100:+.0f}% "
                            f"· score `{r['emerging_score']:.2f}` · {r['recent']} recent articles")
                st.progress(float(min(max(r["emerging_score"], 0), 1)))
        st.markdown("#### Trend over time")
        if not res.counts.empty:
            top_ids = trends["topic_id"].head(4).tolist() if not trends.empty else list(res.counts.columns[:4])
            st.plotly_chart(trend_lines(res.counts, res.topic_names, top_ids), width="stretch")
    with right:
        st.markdown("#### Topic distribution")
        st.plotly_chart(distribution_chart(topics), width="stretch")

# ------------------------------------------------------------------ topics
with tabs[1]:
    st.subheader("Discovered topics")
    show = topics.copy()
    show["trend"] = show.get("trend", "")
    st.dataframe(
        show[["topic_id", "topic", "articles", "share_pct", "sources", "trend", "keywords"]].rename(columns={"share_pct": "share %"}),
        width="stretch", hide_index=True,
        column_config={"share %": st.column_config.ProgressColumn("share %", min_value=0, max_value=float(max(show["share_pct"].max(), 1)), format="%.1f%%")})
    st.divider()
    pick = st.selectbox("Inspect a topic", topics["topic_id"], format_func=lambda i: res.topic_names[int(i)])
    g = df[df.cluster == pick].sort_values("published", ascending=False)
    st.markdown("".join(f'<span class="pill">{k}</span>' for k in res.keywords.get(int(pick), [])), unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"**Latest articles ({len(g)})**")
        st.dataframe(g[["published", "source", "title"]].head(50), width="stretch", hide_index=True)
    with c2:
        st.markdown("**Source mix**")
        st.plotly_chart(source_bar(g), width="stretch")

# ------------------------------------------------------------------ emerging
with tabs[2]:
    st.subheader("Emerging topic detection")
    st.caption("Score = 0.35·growth + 0.25·recent activity + 0.20·source diversity + 0.20·velocity (each normalised 0–1).")
    if trends.empty:
        st.info("Need at least 2 days of data.")
    else:
        st.dataframe(
            trends.assign(growth_rate=trends["growth_rate"] * 100)[["trend", "topic", "articles", "previous", "recent", "growth_rate", "source_diversity", "velocity", "emerging_score"]]
            .rename(columns={"growth_rate": "growth %", "previous": "prev window", "recent": "recent window"}),
            width="stretch", hide_index=True,
            column_config={"emerging_score": st.column_config.ProgressColumn("score", min_value=0, max_value=1, format="%.2f"),
                           "growth %": st.column_config.NumberColumn(format="%+.0f%%"),
                           "source_diversity": st.column_config.NumberColumn("source diversity", format="%.2f"),
                           "velocity": st.column_config.NumberColumn(format="%.3f")})
        sel = st.multiselect("Compare topics", trends["topic_id"], default=trends["topic_id"].head(3).tolist(),
                             format_func=lambda i: res.topic_names[int(i)])
        st.plotly_chart(trend_lines(res.counts, res.topic_names, sel), width="stretch")

# ------------------------------------------------------------------ map
with tabs[3]:
    st.subheader("Topic map")
    st.caption("Each point is an article, projected to 2-D with UMAP (PCA fallback). Hover for details; click legend items to isolate topics.")
    hide_noise = st.checkbox("Hide noise / unclustered", value=False)
    st.plotly_chart(topic_map(df[df.cluster != -1] if hide_noise else df), width="stretch")

# ------------------------------------------------------------------ articles
with tabs[4]:
    st.subheader("Article explorer")
    f1, f2, f3, f4 = st.columns([2, 1.5, 1.5, 1.5])
    q = f1.text_input("Search title / description")
    tsel = f2.multiselect("Topic", sorted(topics["topic"]))
    ssel = f3.multiselect("Source", sorted(df["source"].unique()))
    dmin, dmax = df["published"].min().date(), df["published"].max().date()
    drange = f4.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
    v = df.copy()
    if q:
        v = v[v["title"].str.contains(q, case=False, na=False) | v["description"].str.contains(q, case=False, na=False)]
    if tsel:
        v = v[v["topic_name"].isin(tsel)]
    if ssel:
        v = v[v["source"].isin(ssel)]
    if isinstance(drange, tuple) and len(drange) == 2:
        v = v[(v["published"].dt.date >= drange[0]) & (v["published"].dt.date <= drange[1])]
    st.caption(f"{len(v):,} articles")
    cols = ["published", "source", "topic_name", "title", "url"]
    st.dataframe(v[cols].sort_values("published", ascending=False), width="stretch", hide_index=True,
                 column_config={"url": st.column_config.LinkColumn("link", display_text="open")})
    st.download_button("⬇ Export CSV", v.drop(columns=["embedding_text", "tfidf_text"]).to_csv(index=False).encode(),
                       "newslens_articles.csv", "text/csv")
    if len(v):
        idx = st.selectbox("Article detail", v.index, format_func=lambda i: v.loc[i, "title"][:100])
        a = v.loc[idx]
        st.markdown(f"### {a['title']}")
        st.caption(f"{a['source']} · {a['published']:%d %b %Y %H:%M} · Topic: **{a['topic_name'] or 'Noise'}**")
        st.write(a["description"])
        if a["url"]:
            st.markdown(f"[Open source]({a['url']})")

# ------------------------------------------------------------------ duplicates
with tabs[5]:
    st.subheader("Same-story groups")
    st.caption("Articles with high cosine similarity published within 3 days of each other are treated as one story.")
    d = df[df.story_id != -1]
    if d.empty:
        st.info("No near-duplicate stories at this threshold. Try lowering it in the sidebar.")
    else:
        sizes = d.groupby("story_id").agg(n=("article_id", "count"), outlets=("source", "nunique"), title=("title", "first")).sort_values("n", ascending=False)
        st.metric("Stories covered by multiple articles", len(sizes))
        for sid, r in sizes.head(15).iterrows():
            with st.expander(f"📰 Story #{sid} · {r['n']} related articles · {r['outlets']} outlets — {r['title'][:90]}"):
                st.dataframe(d[d.story_id == sid][["source", "published", "title"]], hide_index=True, width="stretch")

# ------------------------------------------------------------------ metrics
with tabs[6]:
    st.subheader("Model performance")
    c = st.columns(4)
    c[0].metric("Silhouette ↑", f"{m['silhouette']:.3f}" if m["silhouette"] is not None else "n/a")
    c[1].metric("Davies-Bouldin ↓", f"{m['davies_bouldin']:.3f}" if m["davies_bouldin"] is not None else "n/a")
    c[2].metric("Avg coherence (NPMI) ↑", f"{m['avg_coherence']:.3f}" if m["avg_coherence"] is not None else "n/a")
    c[3].metric("Clusters / noise", f"{m['n_clusters']} / {m['n_noise']}")
    st.markdown(f"**Pipeline:** `{m['embedding_backend']}` embeddings → `{m['reduction']}` → `{m['algorithm']}`")
    st.caption("Silhouette & Davies-Bouldin are computed on the reduced space, excluding noise points.")
    st.dataframe(topics[["topic", "articles", "coherence"]], hide_index=True, width="stretch")
    with st.expander("How to read these"):
        st.markdown("- **Silhouette** (−1…1): closer to 1 = tighter, better-separated clusters.\n"
                    "- **Davies-Bouldin**: lower is better.\n"
                    "- **NPMI coherence** (−1…1): how often a topic's top keywords co-occur in the same articles.\n"
                    "- Always sanity-check by reading representative articles per topic.")
