"""
SentimentCompass — Main App Entry Point
Run: streamlit run dashboard/app.py
"""
import sys, os
# Ensure dashboard/ is on the path so pages can import utils
_DASH = os.path.dirname(os.path.abspath(__file__))
if _DASH not in sys.path:
    sys.path.insert(0, _DASH)

import streamlit as st
from utils.data_loader import load_data, inject_css, APP_COLORS, APP_ORDER, APP_LABELS

st.set_page_config(
    page_title="SentimentCompass",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ── Sidebar brand ─────────────────────────────────────────────────────────────
st.sidebar.markdown(
    "<div style='text-align:center;padding:12px 0 4px'>"
    "<span style='font-size:1.6rem'>🧭</span>"
    "<div style='font-size:1.05rem;font-weight:700;margin-top:4px'>SentimentCompass</div>"
    "<div style='font-size:0.72rem;color:#aab4c8;margin-top:2px'>GenAI Review Analytics</div>"
    "</div>",
    unsafe_allow_html=True,
)
st.sidebar.divider()
st.sidebar.markdown("### 📄 Navigation")
st.sidebar.page_link("app.py",              label="🏠 Home")
st.sidebar.page_link("pages/1_Overview.py", label="📊 Overview")
st.sidebar.page_link("pages/2_Sentiment.py",label="🧠 Sentiment")
st.sidebar.page_link("pages/3_Topics.py",   label="🔍 Topics")
st.sidebar.page_link("pages/4_Migration.py",label="🔄 Migration")
st.sidebar.page_link("pages/5_Data_Explorer.py", label="🗄️ Data Explorer")

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-hero">
  <h1>🧭 SentimentCompass</h1>
  <p>
    A multi-model NLP analytics platform exploring user satisfaction, topic trends,
    and migration signals across <strong>50,000 reviews</strong> of five leading
    Generative AI tools.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Quick stats from loaded data ──────────────────────────────────────────────
df = load_data()
if df is not None:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Reviews",     f"{len(df):,}")
    with c2:
        st.metric("Platforms",         "5")
    with c3:
        mean_s = df["Star_Rating"].mean()
        st.metric("Avg Star Rating",   f"{mean_s:.2f} ★")
    with c4:
        if "vader_compound" in df.columns:
            st.metric("Avg Sentiment (VADER)", f"{df['vader_compound'].mean():.3f}")
        else:
            st.metric("Avg Sentiment", "—")
    with c5:
        if "migration_flag" in df.columns:
            n_mig = df["migration_flag"].sum()
            st.metric("Migration Events", f"{int(n_mig):,}")
        else:
            st.metric("Migration Events", "—")

    st.divider()

    # ── Platform quick-view cards ─────────────────────────────────────────────
    st.markdown("### Platform Overview")
    cols = st.columns(5)
    for i, app in enumerate(APP_ORDER):
        sub   = df[df["App"] == app]
        stars = sub["Star_Rating"].mean() if len(sub) else 0
        color = APP_COLORS[app]
        with cols[i]:
            st.markdown(f"""
            <div style="background:white;border-radius:12px;padding:16px;
                        box-shadow:0 2px 8px rgba(0,0,0,0.07);
                        border-top:4px solid {color};text-align:center;">
              <div style="font-weight:700;font-size:1rem;color:{color}">{APP_LABELS[app]}</div>
              <div style="font-size:1.5rem;font-weight:700;margin:6px 0">{stars:.2f} ★</div>
              <div style="font-size:0.78rem;color:#888">{len(sub):,} reviews</div>
            </div>""", unsafe_allow_html=True)

else:
    st.error(
        "⚠️ Dataset not found. Make sure you have run Notebooks 01–05 "
        "and the file `data/processed/reviews_migration.parquet` exists.",
        icon="🚨",
    )
    st.info("Expected location: `sentimentcompass/data/processed/reviews_migration.parquet`")
    st.stop()

st.divider()

# ── Project pipeline ──────────────────────────────────────────────────────────
st.markdown("### 📐 Project Pipeline")
p1, p2, p3, p4, p5 = st.columns(5)
pipeline = [
    ("📊", "Notebook 01", "EDA & Data Quality", "#378ADD"),
    ("🛠️", "Notebook 02", "Cleaning & Feature Engineering", "#1D9E75"),
    ("🧠", "Notebook 03", "Sentiment Analysis (VADER · TextBlob · RoBERTa)", "#534AB7"),
    ("🔍", "Notebook 04", "LDA + BERTopic Modeling", "#BA7517"),
    ("🔄", "Notebook 05", "Migration & Competitive Intelligence", "#D85A30"),
]
for col_, (icon, nb, desc, color) in zip([p1,p2,p3,p4,p5], pipeline):
    with col_:
        st.markdown(f"""
        <div style="background:white;border-radius:10px;padding:14px;
                    box-shadow:0 1px 6px rgba(0,0,0,0.06);
                    border-left:4px solid {color};height:100px;">
          <div style="font-size:1.2rem">{icon}</div>
          <div style="font-weight:700;font-size:0.85rem;color:{color}">{nb}</div>
          <div style="font-size:0.75rem;color:#666;margin-top:4px">{desc}</div>
        </div>""", unsafe_allow_html=True)

st.divider()

# ── Tech stack ────────────────────────────────────────────────────────────────
st.markdown("### 🛠️ Technology Stack")
tc1, tc2, tc3, tc4 = st.columns(4)
with tc1:
    st.markdown("**NLP Models**")
    st.markdown("- VADER Sentiment\n- TextBlob\n- RoBERTa (cardiffnlp)\n- LDA (gensim)\n- BERTopic")
with tc2:
    st.markdown("**Analytics**")
    st.markdown("- pandas · NumPy\n- NetworkX\n- scikit-learn\n- scipy")
with tc3:
    st.markdown("**Visualisation**")
    st.markdown("- Plotly (interactive)\n- Seaborn\n- Matplotlib\n- Streamlit")
with tc4:
    st.markdown("**Data**")
    st.markdown("- 50,000 app reviews\n- 5 AI platforms\n- 65 engineered columns\n- Parquet format")
