"""
SentimentCompass — Data Loader
Handles parquet loading with caching, sidebar filters, and shared constants.
"""
import os
import streamlit as st
import pandas as pd
import numpy as np

# ── Platform palette (consistent with all notebooks) ──────────────────────────
APP_COLORS = {
    "ChatGPT":           "#1D9E75",
    "Google_Gemini":     "#378ADD",
    "Claude":            "#534AB7",
    "Microsoft_Copilot": "#BA7517",
    "Perplexity":        "#D85A30",
}
APP_ORDER  = ["ChatGPT", "Google_Gemini", "Claude", "Microsoft_Copilot", "Perplexity"]
APP_LABELS = {
    "ChatGPT":           "ChatGPT",
    "Google_Gemini":     "Gemini",
    "Claude":            "Claude",
    "Microsoft_Copilot": "MS Copilot",
    "Perplexity":        "Perplexity",
}
PALETTE    = [APP_COLORS[a] for a in APP_ORDER]
CAT_COLORS = {"Positive": "#1D9E75", "Neutral": "#F5C518", "Negative": "#E24B4A"}

# ── Candidate data paths (works from any CWD) ─────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_PATHS = [
    "data/processed/reviews_migration.parquet",
    "../data/processed/reviews_migration.parquet",
    os.path.join(_THIS_DIR, "../../data/processed/reviews_migration.parquet"),
    os.path.join(_THIS_DIR, "../../../data/processed/reviews_migration.parquet"),
]


@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame | None:
    """Load reviews_migration.parquet with multi-path fallback."""
    for path in _CANDIDATE_PATHS:
        norm = os.path.normpath(path)
        if os.path.exists(norm):
            try:
                df = pd.read_parquet(norm)
                if "Review_Date" in df.columns:
                    df["Review_Date"] = pd.to_datetime(df["Review_Date"], errors="coerce")
                # Ensure required base columns exist
                for col in ["Star_Rating", "App"]:
                    if col not in df.columns:
                        raise ValueError(f"Missing column: {col}")
                return df
            except Exception as exc:
                st.warning(f"Could not load {norm}: {exc}")
                continue
    return None


def col(df: pd.DataFrame, name: str, default=None):
    """Safely retrieve a column; return default Series if missing."""
    if name in df.columns:
        return df[name]
    if default is not None:
        return pd.Series([default] * len(df), index=df.index)
    return None


# ── Global custom CSS ─────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    #MainMenu  {visibility: hidden;}
    footer     {visibility: hidden;}
    header     {visibility: hidden;}

    /* Sidebar */
    section[data-testid="stSidebar"] {background: #1a1a2e; color: white;}
    section[data-testid="stSidebar"] * {color: white !important;}
    section[data-testid="stSidebar"] .stMultiSelect > label,
    section[data-testid="stSidebar"] .stSlider > label {font-size:0.82rem;}

    /* KPI metric cards */
    .kpi-card {
        background: white;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07);
        text-align: center;
        border-top: 4px solid #1D9E75;
    }
    .kpi-value {font-size: 2rem; font-weight: 700; color: #1a1a2e; line-height: 1.1;}
    .kpi-label {font-size: 0.75rem; color: #888; text-transform: uppercase;
                letter-spacing: 0.06em; margin-top: 6px;}
    .kpi-delta {font-size: 0.8rem; margin-top: 4px;}

    /* Platform chip */
    .plat-chip {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        color: white;
        margin: 2px;
    }

    /* Section headings */
    .sec-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1a1a2e;
        padding-bottom: 6px;
        border-bottom: 2px solid #f0f0f0;
        margin-bottom: 12px;
    }

    /* Page hero */
    .page-hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 28px 32px;
        color: white;
        margin-bottom: 24px;
    }
    .page-hero h1 {font-size: 1.8rem; font-weight: 700; margin: 0; color: white;}
    .page-hero p  {font-size: 0.95rem; color: #aab4c8; margin: 8px 0 0; line-height: 1.6;}
</style>
"""


def inject_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def apply_sidebar_filters(df: pd.DataFrame):
    """
    Render sidebar filter controls and return (filtered_df, selected_apps).
    Call this at the top of every page.
    """
    inject_css()

    st.sidebar.markdown(
        "<div style='text-align:center;padding:12px 0 4px'>"
        "<span style='font-size:1.6rem'>🧭</span>"
        "<div style='font-size:1.05rem;font-weight:700;margin-top:4px'>SentimentCompass</div>"
        "<div style='font-size:0.72rem;color:#aab4c8;margin-top:2px'>GenAI Review Analytics</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.divider()
    st.sidebar.markdown("### ⚙️ Filters")

    # ── Platform ──────────────────────────────────────────────────────────────
    available = [a for a in APP_ORDER if a in df["App"].unique()]
    selected_apps = st.sidebar.multiselect(
        "Platform",
        options=available,
        default=available,
        format_func=lambda x: APP_LABELS.get(x, x),
        key="sb_apps",
    )
    if not selected_apps:
        selected_apps = available

    # ── Star rating ───────────────────────────────────────────────────────────
    star_min, star_max = int(df["Star_Rating"].min()), int(df["Star_Rating"].max())
    stars = st.sidebar.slider("Star Rating", star_min, star_max,
                               (star_min, star_max), key="sb_stars")

    # ── Sentiment label ───────────────────────────────────────────────────────
    if "vader_label" in df.columns:
        sentiments = st.sidebar.multiselect(
            "Sentiment",
            options=["Positive", "Neutral", "Negative"],
            default=["Positive", "Neutral", "Negative"],
            key="sb_sent",
        )
        if not sentiments:
            sentiments = ["Positive", "Neutral", "Negative"]
    else:
        sentiments = ["Positive", "Neutral", "Negative"]

    # ── Date range ────────────────────────────────────────────────────────────
    date_range = None
    if "Review_Date" in df.columns:
        d_min = df["Review_Date"].min().date()
        d_max = df["Review_Date"].max().date()
        date_range = st.sidebar.date_input(
            "Date Range", value=(d_min, d_max),
            min_value=d_min, max_value=d_max, key="sb_dates",
        )

    st.sidebar.divider()

    # ── Apply ─────────────────────────────────────────────────────────────────
    flt = df[df["App"].isin(selected_apps)].copy()
    flt = flt[(flt["Star_Rating"] >= stars[0]) & (flt["Star_Rating"] <= stars[1])]
    if "vader_label" in flt.columns:
        flt = flt[flt["vader_label"].isin(sentiments)]
    if date_range and len(date_range) == 2 and "Review_Date" in flt.columns:
        flt = flt[
            (flt["Review_Date"].dt.date >= date_range[0]) &
            (flt["Review_Date"].dt.date <= date_range[1])
        ]

    n_filtered = len(flt)
    pct        = n_filtered / max(len(df), 1) * 100
    st.sidebar.metric("Filtered Reviews", f"{n_filtered:,}", f"{pct:.1f}% of total")

    return flt, selected_apps
