"""
SentimentCompass — Page 1: Executive Overview
"""
import sys, os
_D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _D not in sys.path: sys.path.insert(0, _D)

import streamlit as st
import plotly.graph_objects as go

from utils.data_loader import load_data, apply_sidebar_filters, APP_COLORS, APP_ORDER, APP_LABELS
from utils.helpers     import get_platform_stats, compute_twsi
from utils.charts      import (rating_distribution, sentiment_stacked_bar,
                                 platform_bar, satisfaction_scatter)

st.set_page_config(page_title="Overview · SentimentCompass",
                   page_icon="📊", layout="wide")

# ── Load & filter ─────────────────────────────────────────────────────────────
df_raw = load_data()
if df_raw is None:
    st.error("Dataset not found. Please run all five notebooks first.")
    st.stop()

df, sel_apps = apply_sidebar_filters(df_raw)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-hero">
  <h1>📊 Executive Overview</h1>
  <p>High-level satisfaction metrics and cross-platform comparisons across
     all five Generative AI platforms.</p>
</div>""", unsafe_allow_html=True)

# ── KPI cards ─────────────────────────────────────────────────────────────────
stats = get_platform_stats(df)

total_reviews = len(df)
avg_stars     = df["Star_Rating"].mean()
avg_vader     = df["vader_compound"].mean() if "vader_compound" in df.columns else 0
best_plat     = stats.loc[stats["mean_vader"].idxmax(), "label"] if len(stats) else "—"
worst_plat    = stats.loc[stats["mean_vader"].idxmin(), "label"] if len(stats) else "—"

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Reviews",        f"{total_reviews:,}")
k2.metric("Platforms Selected",   str(len(sel_apps)))
k3.metric("Avg Star Rating",      f"{avg_stars:.2f} ★")
k4.metric("Avg VADER Sentiment",  f"{avg_vader:.3f}",
          help="VADER compound score: +1=most positive, −1=most negative")
k5.metric("Most Satisfied",       best_plat)

st.divider()

# ── Platform cards ────────────────────────────────────────────────────────────
st.markdown('<div class="sec-title">Platform-by-Platform Snapshot</div>',
            unsafe_allow_html=True)
card_cols = st.columns(len(sel_apps) if sel_apps else 5)
for i, app in enumerate([a for a in APP_ORDER if a in sel_apps]):
    sub   = df[df["App"] == app]
    if len(sub) == 0:
        continue
    color   = APP_COLORS[app]
    stars   = sub["Star_Rating"].mean()
    vader   = sub["vader_compound"].mean() if "vader_compound" in sub.columns else 0
    pct_pos = (sub["is_positive_review"].mean()*100
               if "is_positive_review" in sub.columns else 0)
    pct_neg = (sub["is_negative_review"].mean()*100
               if "is_negative_review" in sub.columns else 0)
    sent_arrow = "⬆" if vader > 0.1 else ("⬇" if vader < -0.05 else "➡")
    with card_cols[i % len(card_cols)]:
        st.markdown(f"""
        <div style="background:white;border-radius:14px;padding:18px 16px;
                    box-shadow:0 2px 10px rgba(0,0,0,0.08);
                    border-top:5px solid {color};margin-bottom:8px;">
          <div style="font-weight:700;font-size:1rem;color:{color};margin-bottom:10px">
            {APP_LABELS[app]}
          </div>
          <div style="display:flex;justify-content:space-between;margin-bottom:6px">
            <span style="font-size:0.78rem;color:#888">Avg Stars</span>
            <span style="font-weight:600">{stars:.2f} ★</span>
          </div>
          <div style="display:flex;justify-content:space-between;margin-bottom:6px">
            <span style="font-size:0.78rem;color:#888">Sentiment</span>
            <span style="font-weight:600">{sent_arrow} {vader:.3f}</span>
          </div>
          <div style="display:flex;justify-content:space-between;margin-bottom:6px">
            <span style="font-size:0.78rem;color:#888">Positive %</span>
            <span style="font-weight:600;color:#1D9E75">{pct_pos:.1f}%</span>
          </div>
          <div style="display:flex;justify-content:space-between">
            <span style="font-size:0.78rem;color:#888">Negative %</span>
            <span style="font-weight:600;color:#E24B4A">{pct_neg:.1f}%</span>
          </div>
          <div style="margin-top:10px;font-size:0.72rem;color:#aaa;
                      text-align:right">{len(sub):,} reviews</div>
        </div>""", unsafe_allow_html=True)

st.divider()

# ── Row: Rating distribution + Sentiment stacked bar ─────────────────────────
r1, r2 = st.columns(2)
with r1:
    st.markdown('<div class="sec-title">Star Rating Distribution</div>',
                unsafe_allow_html=True)
    st.plotly_chart(rating_distribution(df), use_container_width=True)

with r2:
    st.markdown('<div class="sec-title">Sentiment Category by Platform</div>',
                unsafe_allow_html=True)
    st.plotly_chart(sentiment_stacked_bar(df), use_container_width=True)

# ── Row: Satisfaction ranking + Scatter ──────────────────────────────────────
r3, r4 = st.columns(2)
with r3:
    st.markdown('<div class="sec-title">Platform Satisfaction Ranking (VADER Mean)</div>',
                unsafe_allow_html=True)
    if len(stats) > 0:
        st.plotly_chart(platform_bar(stats, "mean_vader",
                                      "Platform Satisfaction (VADER mean)"),
                        use_container_width=True)
    else:
        st.info("No data available.")

with r4:
    st.markdown('<div class="sec-title">Stars vs Sentiment — Platform Scatter</div>',
                unsafe_allow_html=True)
    st.plotly_chart(satisfaction_scatter(stats), use_container_width=True)

# ── Full-width: Star rating per platform heatmap ──────────────────────────────
st.divider()
st.markdown('<div class="sec-title">Star Rating % Breakdown per Platform</div>',
            unsafe_allow_html=True)

STAR_COLORS = ["#E24B4A","#F5923E","#F5C518","#8DC265","#1D9E75"]
fig_stars = go.Figure()
for star, color in zip([1,2,3,4,5], STAR_COLORS):
    vals  = []
    for app in [a for a in APP_ORDER if a in sel_apps]:
        sub = df[df["App"] == app]
        vals.append(round((sub["Star_Rating"] == star).mean()*100, 1) if len(sub) else 0)
    fig_stars.add_trace(go.Bar(
        name=f"{star} ★",
        x=[APP_LABELS[a] for a in APP_ORDER if a in sel_apps],
        y=vals,
        marker_color=color,
        text=[f"{v:.0f}%" for v in vals], textposition="inside",
        textfont=dict(color="white", size=10),
    ))
fig_stars.update_layout(
    barmode="stack", height=320,
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
    paper_bgcolor="white", plot_bgcolor="white",
    yaxis_title="Percentage (%)",
)
st.plotly_chart(fig_stars, use_container_width=True)

# ── Summary table ─────────────────────────────────────────────────────────────
st.divider()
st.markdown('<div class="sec-title">Platform Summary Statistics</div>',
            unsafe_allow_html=True)
if len(stats) > 0:
    display_cols = [c for c in ["label","n_reviews","mean_stars","pct_positive",
                                 "pct_negative","mean_vader","mean_tb"] if c in stats.columns]
    display_df   = stats[display_cols].rename(columns={
        "label":"Platform","n_reviews":"Reviews","mean_stars":"Avg ★",
        "pct_positive":"Positive %","pct_negative":"Negative %",
        "mean_vader":"VADER Mean","mean_tb":"TextBlob Mean",
    }).set_index("Platform")
    st.dataframe(display_df.style.format({
        "Reviews":"{:,}","Avg ★":"{:.3f}","Positive %":"{:.1f}%",
        "Negative %":"{:.1f}%","VADER Mean":"{:.4f}","TextBlob Mean":"{:.4f}",
    }).background_gradient(subset=["VADER Mean"], cmap="RdYlGn", vmin=-0.3, vmax=0.7),
    use_container_width=True)
