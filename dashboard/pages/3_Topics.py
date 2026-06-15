"""
SentimentCompass — Page 3: Topic Intelligence
"""
import sys, os
_D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _D not in sys.path: sys.path.insert(0, _D)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.data_loader import load_data, apply_sidebar_filters, APP_COLORS, APP_ORDER, APP_LABELS
from utils.helpers     import get_topic_sentiment
from utils.charts      import (platform_topic_heatmap, topic_sentiment_bars,
                                 platform_bar)

st.set_page_config(page_title="Topics · SentimentCompass",
                   page_icon="🔍", layout="wide")

df_raw = load_data()
if df_raw is None:
    st.error("Dataset not found."); st.stop()

df, sel_apps = apply_sidebar_filters(df_raw)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-hero">
  <h1>🔍 Topic Intelligence</h1>
  <p>LDA + BERTopic topic discovery — what are users really talking about?</p>
</div>""", unsafe_allow_html=True)

# Check topic columns
HAS_LDA    = "lda_topic_label"  in df.columns
HAS_BERT   = "bertopic_label"   in df.columns

if not HAS_LDA and not HAS_BERT:
    st.warning("No topic columns found. Please run **Notebook 04** first.", icon="⚠️")
    st.stop()

# ── LDA section ───────────────────────────────────────────────────────────────
if HAS_LDA:
    st.markdown("## 📚 LDA Topic Modeling Results")

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    valid_lda = df[df["lda_topic_label"].notna() & (df["lda_topic_id"] >= 0)] \
                if "lda_topic_id" in df.columns else df[df["lda_topic_label"].notna()]
    n_topics  = df["lda_topic_id"].nunique() if "lda_topic_id" in df.columns else "—"
    k1.metric("Distinct LDA Topics",   str(n_topics))
    k2.metric("Reviews with Topic",     f"{len(valid_lda):,}")
    if "lda_topic_prob" in df.columns:
        k3.metric("Avg Topic Probability", f"{df['lda_topic_prob'].mean():.3f}")
    most_disc = df["lda_topic_label"].value_counts().idxmax() if HAS_LDA else "—"
    k4.metric("Most Discussed Topic",  most_disc[:28] + "…"
              if len(most_disc) > 28 else most_disc)

    st.divider()

    # Topic frequency bar
    st.markdown('<div class="sec-title">LDA Topic Frequency</div>',
                unsafe_allow_html=True)
    lda_counts = df["lda_topic_label"].value_counts()
    fig_lda_freq = go.Figure()
    colors_cycle = [APP_COLORS[a] for a in APP_ORDER]
    fig_lda_freq.add_trace(go.Bar(
        x=lda_counts.index,
        y=lda_counts.values,
        marker_color=[colors_cycle[i % 5] for i in range(len(lda_counts))],
        text=[f"{v:,}" for v in lda_counts.values],
        textposition="outside",
        hovertemplate="%{x}<br>%{y:,} reviews<extra></extra>",
    ))
    fig_lda_freq.update_layout(
        height=380, paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=140),
        xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
        yaxis_title="Number of Reviews", showlegend=False,
    )
    st.plotly_chart(fig_lda_freq, use_container_width=True)

    st.divider()

    # Platform × Topic heatmap (full width)
    st.markdown('<div class="sec-title">Platform × Topic Distribution Heatmap</div>',
                unsafe_allow_html=True)
    st.caption("Cell values show what % of each platform's reviews belong to each topic.")
    flt_apps = df[df["App"].isin(sel_apps)] if sel_apps else df
    st.plotly_chart(platform_topic_heatmap(flt_apps), use_container_width=True)

    st.divider()

    # Topic sentiment: positive vs negative side by side
    topic_sent = get_topic_sentiment(df)
    ts1, ts2 = st.columns(2)
    with ts1:
        st.markdown('<div class="sec-title">🟢 Most Praised Topics</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            topic_sentiment_bars(topic_sent, top_n=8, positive=True),
            use_container_width=True
        )
    with ts2:
        st.markdown('<div class="sec-title">🔴 Most Complained-About Topics</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            topic_sentiment_bars(topic_sent, top_n=8, positive=False),
            use_container_width=True
        )

    st.divider()

    # Topic sentiment heatmap (topic × sentiment stats)
    st.markdown('<div class="sec-title">Topic Sentiment Overview Table</div>',
                unsafe_allow_html=True)
    if topic_sent is not None and len(topic_sent) > 0:
        ts_disp = topic_sent.rename(columns={
            "lda_topic_label": "Topic",
            "count":           "Reviews",
            "mean_vader":      "VADER Mean",
            "pct_pos":         "Positive %",
            "pct_neg":         "Negative %",
        })
        if "Topic" in ts_disp.columns:
            ts_disp = ts_disp.set_index("Topic")
        st.dataframe(
            ts_disp.style
            .format({"Reviews": "{:,}", "VADER Mean": "{:.4f}",
                     "Positive %": "{:.1f}%", "Negative %": "{:.1f}%"})
            .background_gradient(subset=["VADER Mean"],
                                  cmap="RdYlGn", vmin=-0.3, vmax=0.7),
            use_container_width=True,
        )

    # Per-platform top topic breakdown
    st.divider()
    st.markdown('<div class="sec-title">Top 3 LDA Topics per Platform</div>',
                unsafe_allow_html=True)
    pcols = st.columns(len(sel_apps) if sel_apps else 5)
    for i, app in enumerate([a for a in APP_ORDER if a in sel_apps]):
        sub = df[df["App"] == app]
        if len(sub) == 0:
            continue
        top3 = sub["lda_topic_label"].value_counts().head(3)
        color = APP_COLORS[app]
        with pcols[i % len(pcols)]:
            st.markdown(f"**{APP_LABELS[app]}**")
            for rank, (topic, cnt) in enumerate(top3.items(), 1):
                pct = cnt / len(sub) * 100
                st.markdown(
                    f"<small><span style='color:{color};font-weight:600'>#{rank}</span>"
                    f" {topic[:30]}<br>"
                    f"<span style='color:#888'>{cnt:,} reviews ({pct:.1f}%)</span></small>",
                    unsafe_allow_html=True,
                )

# ── BERTopic section ──────────────────────────────────────────────────────────
if HAS_BERT:
    st.divider()
    st.markdown("## 🤖 BERTopic Semantic Clustering")

    bert_counts = df[df["bertopic_id"] != -1]["bertopic_label"].value_counts() \
                  if "bertopic_id" in df.columns else df["bertopic_label"].value_counts()
    bert_top    = bert_counts.head(15)
    outlier_n   = (df["bertopic_id"] == -1).sum() if "bertopic_id" in df.columns else 0

    st.caption(
        f"BERTopic discovered **{bert_counts.nunique()}** semantic clusters "
        f"across {len(df):,} reviews. "
        f"{outlier_n:,} reviews are in the outlier cluster (topic −1)."
    )

    fig_bert = go.Figure()
    bert_colors = [f"hsl({int(220 - i*14)},60%,55%)" for i in range(len(bert_top))]
    fig_bert.add_trace(go.Bar(
        y=bert_top.index[::-1],
        x=bert_top.values[::-1],
        orientation="h",
        marker_color=bert_colors[::-1],
        text=[f"{v:,}" for v in bert_top.values[::-1]],
        textposition="outside",
        hovertemplate="%{y}<br>%{x:,} reviews<extra></extra>",
    ))
    fig_bert.update_layout(
        title="Top 15 BERTopic Semantic Clusters",
        height=max(380, len(bert_top) * 38 + 80),
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=10, r=80, t=55, b=10),
        xaxis_title="Review Count",
        xaxis_range=[0, bert_top.max() * 1.3],
        font=dict(family="Inter, Arial", size=12),
    )
    st.plotly_chart(fig_bert, use_container_width=True)
