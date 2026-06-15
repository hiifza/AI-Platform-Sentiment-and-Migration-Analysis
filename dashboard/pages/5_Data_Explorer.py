"""
SentimentCompass — Page 5: Data Explorer
"""
import sys, os
_D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _D not in sys.path: sys.path.insert(0, _D)

import streamlit as st
import pandas as pd

from utils.data_loader import load_data, apply_sidebar_filters, APP_LABELS, APP_ORDER

st.set_page_config(page_title="Data Explorer · SentimentCompass",
                   page_icon="🗄️", layout="wide")

df_raw = load_data()
if df_raw is None:
    st.error("Dataset not found."); st.stop()

df, sel_apps = apply_sidebar_filters(df_raw)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-hero">
  <h1>🗄️ Data Explorer</h1>
  <p>Search, filter, and export the full review dataset with all engineered features.</p>
</div>""", unsafe_allow_html=True)

# ── Extra filters (beyond sidebar) ───────────────────────────────────────────
st.markdown("### 🔧 Additional Filters")
xf1, xf2, xf3, xf4 = st.columns(4)

with xf1:
    search_text = st.text_input("🔍 Search review text", placeholder="e.g. pricing, crash, slow…")

with xf2:
    topic_opts = ["All topics"]
    if "lda_topic_label" in df.columns:
        topic_opts += sorted(df["lda_topic_label"].dropna().unique().tolist())
    chosen_topic = st.selectbox("LDA Topic", topic_opts)

with xf3:
    mig_opts = ["All reviews", "Migration reviews only", "Non-migration reviews"]
    if "migration_flag" not in df.columns:
        mig_opts = ["All reviews"]
    chosen_mig = st.selectbox("Migration Flag", mig_opts)

with xf4:
    inf_opts = ["All reviews", "Influential only (thumbs > 100)", "Non-influential"]
    if "is_influential_review" not in df.columns:
        inf_opts = ["All reviews"]
    chosen_inf = st.selectbox("Influence", inf_opts)

# ── Apply extra filters ───────────────────────────────────────────────────────
flt = df.copy()

if search_text.strip():
    col_to_search = "review_text_clean" if "review_text_clean" in flt.columns else "Review_Text"
    mask = flt[col_to_search].str.contains(search_text, case=False, na=False)
    flt  = flt[mask]

if chosen_topic != "All topics" and "lda_topic_label" in flt.columns:
    flt = flt[flt["lda_topic_label"] == chosen_topic]

if chosen_mig == "Migration reviews only" and "migration_flag" in flt.columns:
    flt = flt[flt["migration_flag"] == True]
elif chosen_mig == "Non-migration reviews" and "migration_flag" in flt.columns:
    flt = flt[flt["migration_flag"] == False]

if chosen_inf == "Influential only (thumbs > 100)" and "is_influential_review" in flt.columns:
    flt = flt[flt["is_influential_review"] == 1]
elif chosen_inf == "Non-influential" and "is_influential_review" in flt.columns:
    flt = flt[flt["is_influential_review"] == 0]

st.divider()

# ── Results header + download ─────────────────────────────────────────────────
rc1, rc2, rc3 = st.columns([2, 1, 1])
with rc1:
    st.markdown(f"**{len(flt):,} reviews** match current filters "
                f"({len(flt)/max(len(df_raw),1)*100:.1f}% of total dataset)")
with rc2:
    csv_data = flt.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Download Filtered CSV",
        data=csv_data,
        file_name="sentimentcompass_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )
with rc3:
    # Column selector
    all_cols = flt.columns.tolist()
    default_cols = [c for c in [
        "App", "Star_Rating", "Review_Date", "vader_label", "vader_compound",
        "roberta_label", "lda_topic_label", "migration_flag",
        "review_text_clean", "Thumbs_Up_Count",
    ] if c in all_cols]
    selected_cols = st.multiselect("Columns to show", all_cols,
                                    default=default_cols, key="col_select",
                                    label_visibility="collapsed")

# ── Stats strip for current selection ─────────────────────────────────────────
if len(flt) > 0:
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Avg Stars",    f"{flt['Star_Rating'].mean():.2f} ★")
    s2.metric("Avg Sentiment (VADER)",
              f"{flt['vader_compound'].mean():.4f}" if "vader_compound" in flt.columns else "—")
    pct_pos = (flt["is_positive_review"].mean()*100
               if "is_positive_review" in flt.columns else 0)
    s3.metric("Positive %",   f"{pct_pos:.1f}%")
    mig_n = int(flt["migration_flag"].sum()) if "migration_flag" in flt.columns else 0
    s4.metric("Migration Events", f"{mig_n:,}")

st.divider()

# ── Main data table ───────────────────────────────────────────────────────────
if not selected_cols:
    selected_cols = default_cols if default_cols else all_cols[:8]

# Map App to readable label for display
display_df = flt[selected_cols].copy()
if "App" in display_df.columns:
    display_df["App"] = display_df["App"].map(APP_LABELS).fillna(display_df["App"])
if "Review_Date" in display_df.columns:
    display_df["Review_Date"] = pd.to_datetime(
        display_df["Review_Date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

# Truncate long text columns for display
TEXT_COLS = ["review_text_clean", "Review_Text", "migration_reason_text"]
for tc in TEXT_COLS:
    if tc in display_df.columns:
        display_df[tc] = display_df[tc].apply(
            lambda x: str(x)[:120] + "…" if isinstance(x, str) and len(x) > 120 else x
        )

MAX_ROWS = 5000
if len(display_df) > MAX_ROWS:
    st.info(f"Showing first {MAX_ROWS:,} of {len(display_df):,} rows for performance.")
    display_df = display_df.head(MAX_ROWS)

st.dataframe(display_df, use_container_width=True, height=480)

st.divider()

# ── Review detail expander ────────────────────────────────────────────────────
with st.expander("🔎 Full Review Detail — Click to expand"):
    if len(flt) == 0:
        st.info("No reviews match the current filters.")
    else:
        review_idx = st.number_input(
            f"Review index (0 – {len(flt)-1})", 0, max(0, len(flt)-1), 0
        )
        row = flt.iloc[int(review_idx)]
        d1, d2 = st.columns([3, 2])
        with d1:
            st.markdown(f"**Platform:** {APP_LABELS.get(row['App'], row['App'])}")
            if "review_text_clean" in row.index:
                st.markdown("**Cleaned Review Text:**")
                st.text_area("", str(row["review_text_clean"]),
                             height=160, disabled=True, key="detail_text")
            elif "Review_Text" in row.index:
                st.markdown("**Original Review Text:**")
                st.text_area("", str(row["Review_Text"]),
                             height=160, disabled=True, key="detail_orig")
        with d2:
            detail_fields = {
                "Star Rating"      : row.get("Star_Rating", "—"),
                "VADER Compound"   : f"{row['vader_compound']:.4f}" if "vader_compound" in row.index else "—",
                "VADER Label"      : row.get("vader_label", "—"),
                "RoBERTa Label"    : row.get("roberta_label", "—"),
                "TextBlob Polarity": f"{row['tb_polarity']:.4f}" if "tb_polarity" in row.index else "—",
                "LDA Topic"        : row.get("lda_topic_label", "—"),
                "Thumbs Up"        : row.get("Thumbs_Up_Count", "—"),
                "Migration Flag"   : row.get("migration_flag", "—"),
                "Migration: From"  : APP_LABELS.get(row.get("migration_source"), row.get("migration_source","—")),
                "Migration: To"    : APP_LABELS.get(row.get("migration_destination"), row.get("migration_destination","—")),
                "Review Date"      : str(row.get("Review_Date", "—"))[:10],
            }
            for field, val in detail_fields.items():
                st.markdown(f"**{field}:** {val}")

st.divider()

# ── Quick summary by platform ─────────────────────────────────────────────────
with st.expander("📊 Quick Summary — Filtered Results by Platform"):
    if len(flt) > 0:
        summary_cols = {
            "Count": ("App", "count"),
            "Avg Stars": ("Star_Rating", "mean"),
        }
        agg_dict = {"Star_Rating": "mean"}
        if "vader_compound" in flt.columns:
            agg_dict["vader_compound"] = "mean"
        if "is_positive_review" in flt.columns:
            agg_dict["is_positive_review"] = "mean"
        if "is_negative_review" in flt.columns:
            agg_dict["is_negative_review"] = "mean"

        grp = flt.groupby("App").agg({"Star_Rating": ["count","mean"], **{
            k: v for k, v in agg_dict.items() if k != "Star_Rating"
        }})
        grp.columns = ["_".join(c).strip("_") for c in grp.columns]
        grp.index   = [APP_LABELS.get(a, a) for a in grp.index]

        pct_cols = {c: "{:.1f}%" for c in grp.columns
                    if "positive" in c or "negative" in c}
        fmt = {"Star_Rating_mean": "{:.3f}", **pct_cols}
        if "vader_compound_mean" in grp.columns:
            fmt["vader_compound_mean"] = "{:.4f}"

        st.dataframe(
            grp.style.format(fmt)
               .background_gradient(
                   subset=[c for c in grp.columns if "mean" in c and "star" in c.lower()],
                   cmap="RdYlGn", vmin=1, vmax=5
               ),
            use_container_width=True,
        )
    else:
        st.info("No data in current filter selection.")
