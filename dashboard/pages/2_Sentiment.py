"""
SentimentCompass — Page 2: Sentiment Intelligence
"""
import sys, os
_D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _D not in sys.path: sys.path.insert(0, _D)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.data_loader import load_data, apply_sidebar_filters, APP_COLORS, APP_ORDER, APP_LABELS
from utils.helpers     import get_platform_stats, compute_twsi
from utils.charts      import (sentiment_violin, model_correlation_heatmap,
                                 twsi_ranking, platform_bar, sentiment_stacked_bar)

st.set_page_config(page_title="Sentiment · SentimentCompass",
                   page_icon="🧠", layout="wide")

df_raw = load_data()
if df_raw is None:
    st.error("Dataset not found."); st.stop()

df, sel_apps = apply_sidebar_filters(df_raw)
stats        = get_platform_stats(df)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-hero">
  <h1>🧠 Sentiment Intelligence</h1>
  <p>Multi-model sentiment analysis — VADER · TextBlob · RoBERTa · TWSI</p>
</div>""", unsafe_allow_html=True)

# ── Model selector ────────────────────────────────────────────────────────────
MODEL_MAP = {
    "VADER Compound"         : "vader_compound",
    "TextBlob Polarity"      : "tb_polarity",
    "RoBERTa Score"          : "roberta_score",
    "Pre-computed (Baseline)": "Sentiment_Polarity",
}
available_models = {k: v for k, v in MODEL_MAP.items() if v in df.columns}
if not available_models:
    st.error("No sentiment columns found. Please run Notebook 03."); st.stop()

col_sel, col_info = st.columns([2, 3])
with col_sel:
    chosen_model_label = st.selectbox("Primary Sentiment Model", list(available_models.keys()))
    chosen_col = available_models[chosen_model_label]
with col_info:
    desc = {
        "VADER Compound"         : "Rule-based lexicon. Range [−1, 1]. Fast, handles punctuation/caps emphasis. Run on all 50K reviews.",
        "TextBlob Polarity"      : "Pattern-based. Range [−1, 1]. Also provides subjectivity score. Run on all 50K reviews.",
        "RoBERTa Score"          : "Transformer model (cardiffnlp). Net score = P(positive) − P(negative). Most contextually accurate.",
        "Pre-computed (Baseline)": "Pre-labeled sentiment in the original dataset. Serves as a validation baseline.",
    }
    st.info(desc.get(chosen_model_label, ""))

st.divider()

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
if chosen_col in df.columns:
    valid = df[chosen_col].dropna()
    k1.metric(f"{chosen_model_label} Mean",   f"{valid.mean():.4f}")
    k2.metric("Median",                        f"{valid.median():.4f}")
    pct_pos = (valid > 0.05).mean() * 100
    pct_neg = (valid < -0.05).mean() * 100
    k3.metric("% Positive (>0.05)",           f"{pct_pos:.1f}%")
    k4.metric("% Negative (<−0.05)",          f"{pct_neg:.1f}%")

st.divider()

# ── Row 1: Violin + TWSI ranking ──────────────────────────────────────────────
r1, r2 = st.columns([3, 2])
with r1:
    st.markdown('<div class="sec-title">Score Distribution by Platform (Violin)</div>',
                unsafe_allow_html=True)
    fig_v = sentiment_violin(df, chosen_col, f"{chosen_model_label} Distribution")
    st.plotly_chart(fig_v, use_container_width=True)

with r2:
    st.markdown('<div class="sec-title">Thumbs-Weighted Sentiment (TWSI)</div>',
                unsafe_allow_html=True)
    twsi = compute_twsi(df, score_col=chosen_col)
    mean_sent = pd.Series({app: df[df["App"]==app][chosen_col].mean()
                            for app in APP_ORDER
                            if app in df["App"].values and chosen_col in df.columns})
    fig_t = twsi_ranking(twsi, mean_sent, "TWSI vs Simple Mean")
    st.plotly_chart(fig_t, use_container_width=True)
    with st.expander("ℹ️ What is TWSI?"):
        st.markdown("""
        **TWSI = Thumbs-Weighted Sentiment Index**

        `TWSI = Σ(log(1 + thumbs_up) × sentiment) / Σ(log(1 + thumbs_up))`

        Reviews with more thumbs-up carry more weight.
        TWSI captures *community-endorsed* sentiment rather than treating every review equally.
        A positive TWSI gap (vs simple mean) means endorsed reviews are more positive.
        """)

st.divider()

# ── Row 2: Stacked sentiment bars + platform ranking ─────────────────────────
r3, r4 = st.columns(2)
with r3:
    st.markdown('<div class="sec-title">Positive / Neutral / Negative Split (VADER)</div>',
                unsafe_allow_html=True)
    st.plotly_chart(sentiment_stacked_bar(df), use_container_width=True)

with r4:
    st.markdown(f'<div class="sec-title">Platform Ranking: {chosen_model_label}</div>',
                unsafe_allow_html=True)
    if len(stats) > 0 and chosen_col in df.columns:
        # Compute mean for chosen model
        model_means = pd.DataFrame([
            {"app": app, "label": APP_LABELS[app], "color": APP_COLORS[app],
             "score": df[df["App"]==app][chosen_col].mean()}
            for app in APP_ORDER if app in df["App"].values
        ])
        fig_rank = go.Figure()
        for _, row in model_means.sort_values("score").iterrows():
            color = "#1D9E75" if row["score"] > 0.05 else (
                    "#E24B4A" if row["score"] < -0.05 else "#F5C518")
            fig_rank.add_trace(go.Bar(
                y=[row["label"]], x=[row["score"]], orientation="h",
                marker_color=color,
                text=[f"{row['score']:.4f}"], textposition="outside",
                showlegend=False,
                hovertemplate=f"{row['label']}: %{{x:.4f}}<extra></extra>",
            ))
        fig_rank.add_vline(x=0, line_dash="dash", line_color="#999999", opacity=0.6)
        fig_rank.update_layout(
            height=360, paper_bgcolor="white", plot_bgcolor="white",
            margin=dict(l=10,r=80,t=20,b=10),
            xaxis_title=chosen_model_label, yaxis_title="",
            font=dict(family="Inter, Arial", size=12),
        )
        st.plotly_chart(fig_rank, use_container_width=True)

st.divider()

# ── Row 3: Model correlation heatmap ─────────────────────────────────────────
st.markdown('<div class="sec-title">Multi-Model Score Correlation Matrix</div>',
            unsafe_allow_html=True)
st.plotly_chart(model_correlation_heatmap(df), use_container_width=True)

st.divider()

# ── Row 4: TextBlob subjectivity ──────────────────────────────────────────────
if "tb_subjectivity" in df.columns:
    st.markdown('<div class="sec-title">TextBlob Subjectivity Score by Platform</div>',
                unsafe_allow_html=True)
    st.caption("Subjectivity [0=objective/factual, 1=highly subjective/opinionated] — "
               "useful for distinguishing factual feedback from emotional reactions.")

    sc1, sc2 = st.columns(2)
    with sc1:
        # Box plot of subjectivity
        fig_subj = go.Figure()
        for app in APP_ORDER:
            sub_data = df[df["App"] == app]["tb_subjectivity"].dropna()
            if len(sub_data) < 10:
                continue
            fig_subj.add_trace(go.Box(
                y=sub_data, name=APP_LABELS[app],
                marker_color=APP_COLORS[app],
                boxmean=True,
                hovertemplate=f"{APP_LABELS[app]}<br>%{{y:.3f}}<extra></extra>",
            ))
        fig_subj.update_layout(
            title="Subjectivity Score Distribution",
            yaxis_title="Subjectivity Score",
            height=380, paper_bgcolor="white", plot_bgcolor="#FAFAFA",
            margin=dict(l=10,r=10,t=45,b=10), showlegend=False,
        )
        st.plotly_chart(fig_subj, use_container_width=True)

    with sc2:
        # Subjectivity vs Polarity scatter
        sample = df.sample(min(3000, len(df)), random_state=42)
        fig_scatter = go.Figure()
        for app in APP_ORDER:
            sub_s = sample[sample["App"] == app]
            if len(sub_s) < 5:
                continue
            fig_scatter.add_trace(go.Scatter(
                x=sub_s["tb_polarity"], y=sub_s["tb_subjectivity"],
                mode="markers",
                marker=dict(color=APP_COLORS[app], size=4, opacity=0.5),
                name=APP_LABELS[app],
                hovertemplate=(f"Polarity: %{{x:.3f}}<br>"
                               f"Subjectivity: %{{y:.3f}}<extra>{APP_LABELS[app]}</extra>"),
            ))
        fig_scatter.update_layout(
            title="TextBlob: Polarity vs Subjectivity (3K sample)",
            xaxis_title="Polarity", yaxis_title="Subjectivity",
            height=380, paper_bgcolor="white", plot_bgcolor="#FAFAFA",
            margin=dict(l=10,r=10,t=45,b=10),
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        )
        fig_scatter.add_vline(x=0, line_dash="dash", line_color="#cccccc", opacity=0.5)
        st.plotly_chart(fig_scatter, use_container_width=True)

# ── RoBERTa coverage note ────────────────────────────────────────────────────
if "roberta_processed" in df.columns:
    n_proc = int(df["roberta_processed"].sum())
    n_tot  = len(df)
    if n_proc < n_tot:
        st.info(
            f"ℹ️ RoBERTa was run on **{n_proc:,} / {n_tot:,} reviews** "
            f"({n_proc/n_tot*100:.1f}%). "
            "Re-run Notebook 03 with a GPU to process all reviews.",
            icon="💡",
        )
