"""
SentimentCompass — Page 4: Migration Intelligence
"""
import sys, os
_D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _D not in sys.path: sys.path.insert(0, _D)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.data_loader import load_data, apply_sidebar_filters, APP_COLORS, APP_ORDER, APP_LABELS
from utils.helpers     import (build_migration_matrix, build_networkx_graph,
                                 get_net_migration, get_mention_counts)
from utils.charts      import (migration_network, migration_matrix_heatmap,
                                 net_migration_bar, migration_reasons_bar,
                                 platform_mention_bar)

st.set_page_config(page_title="Migration · SentimentCompass",
                   page_icon="🔄", layout="wide")

df_raw = load_data()
if df_raw is None:
    st.error("Dataset not found."); st.stop()

df, sel_apps = apply_sidebar_filters(df_raw)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-hero">
  <h1>🔄 Migration Intelligence</h1>
  <p>Competitive mention detection · Directed migration network · Platform winners & losers</p>
</div>""", unsafe_allow_html=True)

# Check migration columns
HAS_MIG  = "migration_flag" in df.columns
HAS_COMP = "mentions_competitor" in df.columns

if not HAS_MIG and not HAS_COMP:
    st.warning("Migration columns not found. Please run **Notebook 05** first.", icon="⚠️")
    st.stop()

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
n_with_mentions = int(df["mentions_competitor"].sum()) if HAS_COMP else 0
n_migration     = int(df["migration_flag"].sum())      if HAS_MIG  else 0
mention_rate    = n_with_mentions / max(len(df), 1) * 100
migration_rate  = n_migration / max(len(df), 1) * 100

k1.metric("Reviews with Mentions", f"{n_with_mentions:,}", f"{mention_rate:.1f}%")
k2.metric("Migration Events",      f"{n_migration:,}",     f"{migration_rate:.2f}%")
if HAS_MIG and "migration_source" in df.columns:
    mig_df   = df[df["migration_flag"] == True]
    all_flows = mig_df["migration_source"].value_counts()
    k3.metric("Top Losing Platform",
              APP_LABELS.get(all_flows.idxmax(), "—") if len(all_flows) else "—")
    all_dest  = mig_df["migration_destination"].value_counts()
    k4.metric("Top Gaining Platform",
              APP_LABELS.get(all_dest.idxmax(), "—") if len(all_dest) else "—")
    if "migration_reason_category" in mig_df.columns:
        top_reason = mig_df["migration_reason_category"].value_counts().idxmax() \
                     if mig_df["migration_reason_category"].notna().any() else "—"
        k5.metric("Top Migration Reason", top_reason)
    else:
        k5.metric("Top Migration Reason", "—")
else:
    k3.metric("Top Loser", "—"); k4.metric("Top Gainer", "—"); k5.metric("Top Reason", "—")

st.divider()

# ── Build migration matrix + graph ────────────────────────────────────────────
matrix  = build_migration_matrix(df)
G       = build_networkx_graph(matrix, df)
net_df  = get_net_migration(matrix)

# ── Network graph (full width) ────────────────────────────────────────────────
st.markdown('<div class="sec-title">Platform Migration Network (Directed Graph)</div>',
            unsafe_allow_html=True)
st.caption(
    "**Nodes** = platforms, sized by review count. "
    "**Arrows** = migration direction with edge weight. "
    "Hover over nodes for in/out-flow stats."
)
fig_net = migration_network(G, "User Migration Network — SentimentCompass")
st.plotly_chart(fig_net, use_container_width=True)

with st.expander("📐 Graph Metrics Table"):
    import networkx as nx
    in_deg   = dict(G.in_degree(weight="weight"))
    out_deg  = dict(G.out_degree(weight="weight"))
    try:
        between = nx.betweenness_centrality(G, weight="weight")
    except Exception:
        between = {n: 0.0 for n in G.nodes()}
    gm_rows = []
    for app in APP_ORDER:
        lbl = APP_LABELS[app]
        gm_rows.append({
            "Platform"          : lbl,
            "In-flow (arrivals)": in_deg.get(lbl, 0),
            "Out-flow (departures)": out_deg.get(lbl, 0),
            "Net Migration"     : in_deg.get(lbl,0) - out_deg.get(lbl,0),
            "Betweenness"       : round(between.get(lbl, 0), 4),
        })
    gm_df = pd.DataFrame(gm_rows).set_index("Platform")
    st.dataframe(
        gm_df.style
        .background_gradient(subset=["Net Migration"],
                              cmap="RdYlGn", vmin=-10, vmax=10),
        use_container_width=True,
    )

st.divider()

# ── Row: migration matrix + net migration bar ─────────────────────────────────
r1, r2 = st.columns(2)
with r1:
    st.markdown('<div class="sec-title">Migration Matrix (Source → Destination)</div>',
                unsafe_allow_html=True)
    st.plotly_chart(migration_matrix_heatmap(matrix), use_container_width=True)

with r2:
    st.markdown('<div class="sec-title">Net Migration Score by Platform</div>',
                unsafe_allow_html=True)
    st.caption("Positive = gaining users · Negative = losing users")
    st.plotly_chart(net_migration_bar(net_df), use_container_width=True)
    # Winner / Loser callout
    if len(net_df) > 0:
        winner = net_df.iloc[0]
        loser  = net_df.iloc[-1]
        wc, lc = st.columns(2)
        wc.success(f"🏆 **Net Winner:** {winner['label']}  ({winner['net']:+d})")
        lc.error(  f"📉 **Net Loser:**  {loser['label']}   ({loser['net']:+d})")

st.divider()

# ── Row: mention bars + migration reasons ─────────────────────────────────────
r3, r4 = st.columns(2)
with r3:
    st.markdown('<div class="sec-title">Competitor Mention Frequency</div>',
                unsafe_allow_html=True)
    st.plotly_chart(platform_mention_bar(df), use_container_width=True)

with r4:
    st.markdown('<div class="sec-title">Migration Reason Categories</div>',
                unsafe_allow_html=True)
    st.plotly_chart(migration_reasons_bar(df), use_container_width=True)

st.divider()

# ── Competitive Intelligence Scorecards ───────────────────────────────────────
st.markdown('<div class="sec-title">Platform Competitive Intelligence Scorecards</div>',
            unsafe_allow_html=True)

ci_path = None
for p in ["data/processed/platform_competitive_intelligence.csv",
          "../data/processed/platform_competitive_intelligence.csv"]:
    if os.path.exists(p):
        ci_path = p; break

if ci_path:
    ci_df = pd.read_csv(ci_path, index_col=0)
    st.dataframe(
        ci_df.style.format({
            col: "{:.2f}%" for col in ci_df.columns
            if "Rate" in col or "%" in col
        }).background_gradient(subset=[c for c in ci_df.columns if "Net" in c],
                                cmap="RdYlGn", vmin=-20, vmax=20),
        use_container_width=True,
    )
else:
    # Build scorecards on the fly
    ci_rows = []
    for app in [a for a in APP_ORDER if a in sel_apps]:
        sub     = df[df["App"] == app]
        mig_sub = df[df["migration_flag"] == True] if HAS_MIG else pd.DataFrame()
        gains   = int((mig_sub["migration_destination"] == app).sum()) if HAS_MIG and "migration_destination" in mig_sub.columns else 0
        losses  = int((mig_sub["migration_source"]      == app).sum()) if HAS_MIG and "migration_source"      in mig_sub.columns else 0
        mention_n = int(sub["mentions_competitor"].sum()) if HAS_COMP else 0
        m_rate    = round(mention_n / max(len(sub), 1) * 100, 2)
        all_comps = []
        if "mentioned_platforms" in sub.columns:
            for m in sub["mentioned_platforms"].dropna():
                all_comps.extend([x for x in str(m).split("|") if x and x != app])
        top_rival = APP_LABELS.get(
            pd.Series(all_comps).value_counts().idxmax(), "—"
        ) if all_comps else "—"
        ci_rows.append({
            "Platform"           : APP_LABELS[app],
            "Reviews"            : len(sub),
            "Mention Rate %"     : m_rate,
            "Top Rival Mentioned": top_rival,
            "Gains"              : gains,
            "Losses"             : losses,
            "Net"                : gains - losses,
        })
    ci_built = pd.DataFrame(ci_rows).set_index("Platform")
    st.dataframe(
        ci_built.style
        .background_gradient(subset=["Net"], cmap="RdYlGn", vmin=-15, vmax=15)
        .format({"Reviews":"{:,}", "Mention Rate %":"{:.2f}%"}),
        use_container_width=True,
    )

st.divider()

# ── Migration review samples ──────────────────────────────────────────────────
if HAS_MIG:
    with st.expander("📋 Sample Migration Reviews"):
        mig_sample = df[df["migration_flag"] == True].copy()
        if len(mig_sample) > 0:
            mig_sample["src_lbl"] = mig_sample["migration_source"].map(
                lambda x: APP_LABELS.get(x, x) if x else "—")
            mig_sample["dst_lbl"] = mig_sample["migration_destination"].map(
                lambda x: APP_LABELS.get(x, x) if x else "—")
            show_cols = [c for c in ["App","src_lbl","dst_lbl","migration_phrase",
                                      "migration_reason_category","Star_Rating",
                                      "vader_compound","migration_reason_text"]
                         if c in mig_sample.columns]
            st.dataframe(
                mig_sample[show_cols].rename(columns={
                    "App":"Platform","src_lbl":"From","dst_lbl":"To",
                    "migration_phrase":"Trigger Phrase",
                    "migration_reason_category":"Reason",
                    "Star_Rating":"Stars","vader_compound":"VADER",
                    "migration_reason_text":"Context Snippet",
                }).head(50),
                use_container_width=True,
            )
        else:
            st.info("No migration events detected in the current filter.")

# ── Download migration summary ────────────────────────────────────────────────
st.divider()
if HAS_MIG:
    mig_export = df[df["migration_flag"] == True]
    csv_bytes  = mig_export.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Download Migration Events CSV",
        data=csv_bytes,
        file_name="sentimentcompass_migration_events.csv",
        mime="text/csv",
    )
