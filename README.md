# 📰 NewsLens
**Intelligent News Article Clustering & Emerging Topic Discovery System** — Streamlit app.

```
News → preprocessing → embeddings → UMAP → HDBSCAN → topics → emerging-topic score → dashboard
```

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/generate_sample_data.py                  # optional: regenerates data/news.csv
streamlit run app.py
```
Open http://localhost:8501, then press **▶ Run analysis** in the sidebar.

* First run with Sentence-Transformers downloads `all-MiniLM-L6-v2` (~90 MB, needs internet).
* No internet / no GPU? Choose **TF-IDF + SVD** in the sidebar. If `sentence-transformers` or `umap-learn`
  aren't installed the app falls back automatically (TF-IDF+SVD, PCA) and tells you.

## Using your own data
Upload a CSV/JSON in the sidebar. Needs a `title` column; `description`, `source`, `url`, `published`,
`author`, `category` are optional. Aliases like `headline`, `content`, `date` are mapped automatically
(works with the Kaggle *News Category Dataset*). Aim for 1,000–5,000 articles spread over several days.

## Live news (Phase 6)
**Data sources** tab → pick RSS feeds → *Fetch feeds*. Articles are stored in SQLite (`data/newslens.db`,
deduplicated by URL). Choose **Collected (RSS → SQLite)** in the sidebar and run. Fetch daily
(cron / Task Scheduler running a small script that calls `rss_collector.collect` + `storage.save_articles`)
to build trend history. `cluster_model.assign_new_articles()` assigns fresh articles to existing clusters.

## App tabs
Overview · Topics · Emerging · Topic map · Articles (search/filter/export) · Duplicates · Model metrics · Data sources

## Emerging score
`0.35·growth + 0.25·recent activity + 0.20·source diversity + 0.20·velocity` (each normalised 0–1),
comparing the last *N* days with the previous *N* days (sidebar). Labels: 🔥 Rising ≥ 0.60, ↑ Growing, → Stable, ↓ Declining.
The reference "today" is the newest article date in the dataset.

## Optional LLM topic names
`pip install anthropic`, set `ANTHROPIC_API_KEY`, tick the sidebar box. The LLM only turns keywords into a name;
the ML pipeline never depends on it.

## Layout
```
app.py                      Streamlit UI
src/pipeline.py             orchestration
src/data_collection/        data_loader, rss_collector, storage (SQLite)
src/preprocessing/          text_cleaner (TF-IDF text vs embedding text)
src/embeddings/             Sentence-Transformers + TF-IDF/SVD fallback
src/clustering/             UMAP + HDBSCAN/K-Means, evaluation
src/topics/                 c-TF-IDF keywords, NPMI coherence, naming
src/trends/                 emerging-topic scoring
src/similarity/             duplicate-story detection
src/visualization/          Plotly charts
scripts/generate_sample_data.py
```
Note: the bundled sample data is synthetic (for demo/testing). Use real news for meaningful results.

## Tuning tips
* Too many tiny topics → raise **Min cluster size**; too few / merged → lower it.
* Lots of noise is normal with HDBSCAN; noise = articles that don't belong to a dense topic.


