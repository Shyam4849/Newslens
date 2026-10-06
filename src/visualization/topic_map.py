"""Plotly visualisations."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

TEMPLATE = "plotly_dark"


def topic_map(df: pd.DataFrame, height: int = 640) -> go.Figure:
    d = df.copy()
    d["topic_name"] = d["topic_name"].fillna("Noise / Unclustered")
    d["published_str"] = d["published"].dt.strftime("%d %b %Y")
    fig = px.scatter(d, x="x", y="y", color="topic_name", template=TEMPLATE, height=height,
                     hover_data={"title": True, "source": True, "published_str": True,
                                 "x": False, "y": False, "topic_name": True},
                     labels={"topic_name": "Topic", "published_str": "Published"})
    fig.update_traces(marker=dict(size=7, opacity=0.8, line=dict(width=0)))
    for tid, g in d[d["cluster"] != -1].groupby("topic_name"):
        fig.add_annotation(x=g["x"].median(), y=g["y"].median(), text=f"<b>{tid}</b>", showarrow=False,
                           font=dict(size=11, color="white"), bgcolor="rgba(0,0,0,0.45)")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(legend=dict(orientation="v", font=dict(size=10)), margin=dict(l=0, r=0, t=10, b=0))
    return fig


def distribution_chart(topics: pd.DataFrame) -> go.Figure:
    d = topics.sort_values("articles")
    fig = px.bar(d, x="articles", y="topic", orientation="h", template=TEMPLATE, text="share_pct",
                 color="articles", color_continuous_scale="Sunset")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(coloraxis_showscale=False, height=max(320, 34 * len(d) + 80),
                      margin=dict(l=0, r=30, t=10, b=0), yaxis_title=None, xaxis_title="Articles")
    return fig


def trend_lines(counts: pd.DataFrame, names: dict[int, str], topic_ids: list[int]) -> go.Figure:
    fig = go.Figure()
    for tid in topic_ids:
        if tid in counts.columns:
            fig.add_trace(go.Scatter(x=counts.index, y=counts[tid], mode="lines+markers",
                                     name=names.get(tid, str(tid)), line=dict(width=2.5)))
    fig.update_layout(template=TEMPLATE, height=380, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title=None, yaxis_title="Articles / day", legend=dict(orientation="h", y=-0.2))
    return fig


def source_bar(df: pd.DataFrame) -> go.Figure:
    s = df["source"].value_counts().head(15).sort_values()
    fig = px.bar(x=s.values, y=s.index, orientation="h", template=TEMPLATE)
    fig.update_layout(height=max(260, 26 * len(s) + 60), margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="Articles", yaxis_title=None)
    return fig
