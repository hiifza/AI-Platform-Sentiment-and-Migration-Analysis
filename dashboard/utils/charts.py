"""
SentimentCompass — Plotly Chart Library
All chart functions return go.Figure objects for st.plotly_chart().
"""
import math
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from utils.data_loader import APP_COLORS, APP_ORDER, APP_LABELS, PALETTE, CAT_COLORS

# ── Shared layout defaults ─────────────────────────────────────────────────────
_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="#FAFAFA",
    font=dict(family="Inter, Arial, sans-serif", size=12, color="#333333"),
    margin=dict(l=10, r=10, t=45, b=10),
    hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="#cccccc"),
    legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#e0e0e0", borderwidth=1),
)


def _fig(title="", height=400, **kwargs) -> go.Figure:
    layout = {**_LAYOUT, "title": dict(text=title, font=dict(size=15, color="#1a1a2e")),
              "height": height, **kwargs}
    return go.Figure(layout=go.Layout(**layout))


# ─────────────────────────────────────────────────────────────────────────────
# 1. RATING DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
def rating_distribution(df: pd.DataFrame, title="Star Rating Distribution") -> go.Figure:
    STAR_COLORS = ["#E24B4A", "#F5923E", "#F5C518", "#8DC265", "#1D9E75"]
    counts = df["Star_Rating"].value_counts().sort_index()
    fig = _fig(title, height=380)
    fig.add_trace(go.Bar(
        x=[f"{s} ★" for s in counts.index],
        y=counts.values,
        marker_color=STAR_COLORS[:len(counts)],
        text=[f"{v/len(df)*100:.1f}%" for v in counts.values],
        textposition="outside",
        hovertemplate="%{x}: %{y:,} reviews<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Star Rating", yaxis_title="Number of Reviews",
        showlegend=False, plot_bgcolor="white",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. SENTIMENT STACKED BAR
# ─────────────────────────────────────────────────────────────────────────────
def sentiment_stacked_bar(df: pd.DataFrame, label_col="vader_label",
                           title="Sentiment Distribution by Platform") -> go.Figure:
    fig = _fig(title, height=400)
    if label_col not in df.columns:
        fig.add_annotation(text="Sentiment column not available", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    for cat, color in CAT_COLORS.items():
        vals = []
        for app in APP_ORDER:
            sub = df[df["App"] == app]
            if len(sub) == 0:
                vals.append(0)
                continue
            vals.append(round((sub[label_col] == cat).mean() * 100, 1))
        fig.add_trace(go.Bar(
            name=cat, x=[APP_LABELS[a] for a in APP_ORDER],
            y=vals, marker_color=color,
            text=[f"{v:.0f}%" for v in vals], textposition="inside",
            textfont=dict(color="white", size=11),
            hovertemplate=f"{cat}: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack", xaxis_title="",
        yaxis_title="Percentage (%)", yaxis_range=[0, 105],
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. PLATFORM BAR (horizontal ranking)
# ─────────────────────────────────────────────────────────────────────────────
def platform_bar(stats: pd.DataFrame, y_col: str, title: str,
                 ascending=True, fmt=".3f") -> go.Figure:
    df_s = stats.sort_values(y_col, ascending=ascending)
    labels = df_s["label"].tolist()
    values = df_s[y_col].tolist()
    colors = df_s["color"].tolist()

    fig = _fig(title, height=max(300, len(labels) * 55 + 80))
    fig.add_trace(go.Bar(
        y=labels, x=values, orientation="h",
        marker_color=colors,
        text=[f"{v:{fmt}}" for v in values],
        textposition="outside",
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="", yaxis_title="",
        showlegend=False, plot_bgcolor="white",
        xaxis_range=[min(0, min(values) * 1.3), max(values) * 1.25],
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. SENTIMENT VIOLIN
# ─────────────────────────────────────────────────────────────────────────────
def sentiment_violin(df: pd.DataFrame, score_col="vader_compound",
                     title="Sentiment Score Distribution") -> go.Figure:
    fig = _fig(title, height=440)
    if score_col not in df.columns:
        fig.add_annotation(text=f"Column '{score_col}' not available",
                           showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    for app in APP_ORDER:
        sub = df[df["App"] == app][score_col].dropna()
        if len(sub) < 10:
            continue
        fig.add_trace(go.Violin(
            y=sub, name=APP_LABELS[app],
            box_visible=True, meanline_visible=True,
            fillcolor=APP_COLORS[app], opacity=0.72,
            line_color=APP_COLORS[app],
            hovertemplate=f"{APP_LABELS[app]}<br>%{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        yaxis_title="Sentiment Score",
        xaxis_title="Platform",
        showlegend=False,
        yaxis=dict(zeroline=True, zerolinecolor="#cccccc", zerolinewidth=1.5),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.4,
                  annotation_text="Neutral", annotation_position="bottom right")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
def model_correlation_heatmap(df: pd.DataFrame,
                               title="Model Score Correlation Matrix") -> go.Figure:
    COLS = {
        "Baseline (Pre-computed)": "Sentiment_Polarity",
        "VADER Compound"         : "vader_compound",
        "TextBlob Polarity"      : "tb_polarity",
        "RoBERTa Score"          : "roberta_score",
    }
    available = {k: v for k, v in COLS.items()
                 if v in df.columns and df[v].notna().sum() > 100}
    if len(available) < 2:
        fig = _fig(title)
        fig.add_annotation(text="Insufficient model data", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    sub = df[[v for v in available.values()]].dropna()
    sub.columns = list(available.keys())
    corr = sub.corr().round(3)

    mask  = np.triu(np.ones_like(corr, dtype=bool), k=1)
    z     = corr.values.copy()
    z[mask] = None

    fig = _fig(title, height=420)
    fig.add_trace(go.Heatmap(
        z=z, x=list(available.keys()), y=list(available.keys()),
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=corr.values.round(3),
        texttemplate="%{text}",
        textfont=dict(size=12, color="black"),
        hovertemplate="%{y} ↔ %{x}<br>r = %{z:.3f}<extra></extra>",
        colorbar=dict(title="Pearson r", thickness=14),
    ))
    fig.update_layout(
        xaxis=dict(tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11)),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. TWSI RANKING
# ─────────────────────────────────────────────────────────────────────────────
def twsi_ranking(twsi: pd.Series, mean_sent: pd.Series,
                 title="TWSI vs Mean Sentiment") -> go.Figure:
    apps   = [a for a in APP_ORDER if a in twsi.index]
    labels = [APP_LABELS[a] for a in apps]
    colors = [APP_COLORS[a] for a in apps]

    fig = _fig(title, height=380)
    x = list(range(len(apps)))
    w = 0.36
    fig.add_trace(go.Bar(
        x=[i - w/2 for i in x], y=[twsi[a] for a in apps],
        name="TWSI (weighted)", marker_color=colors,
        text=[f"{twsi[a]:.3f}" for a in apps],
        textposition="outside",
        width=w,
        hovertemplate="%{text}<extra>TWSI</extra>",
    ))
    fig.add_trace(go.Bar(
        x=[i + w/2 for i in x], y=[mean_sent.get(a, 0) for a in apps],
        name="Simple Mean", marker_color=[c + "88" for c in colors],
        text=[f"{mean_sent.get(a,0):.3f}" for a in apps],
        textposition="outside",
        width=w,
        hovertemplate="%{text}<extra>Mean</extra>",
    ))
    fig.update_layout(
        xaxis=dict(tickvals=x, ticktext=labels),
        yaxis_title="Score", barmode="group",
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#cccccc", opacity=0.7)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. PLATFORM × TOPIC HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
def platform_topic_heatmap(df: pd.DataFrame, topic_col="lda_topic_label",
                            title="Platform × Topic Distribution (%)") -> go.Figure:
    if topic_col not in df.columns:
        fig = _fig(title)
        fig.add_annotation(text="Topic data not available — run Notebook 04",
                           showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    pivot = (
        df[df[topic_col].notna()]
        .groupby(["App", topic_col])
        .size()
        .reset_index(name="Count")
    )
    pivot["Pct"] = pivot.groupby("App")["Count"].transform(lambda x: x / x.sum() * 100)
    wide = (pivot.pivot(index=topic_col, columns="App", values="Pct")
                 .fillna(0)
                 .reindex(columns=APP_ORDER, fill_value=0))
    wide.columns = [APP_LABELS[a] for a in APP_ORDER]

    fig = _fig(title, height=max(380, len(wide) * 40 + 100))
    fig.add_trace(go.Heatmap(
        z=wide.values.round(1), x=wide.columns.tolist(), y=wide.index.tolist(),
        colorscale="YlOrRd",
        text=wide.values.round(1),
        texttemplate="%{text:.1f}%",
        textfont=dict(size=10),
        hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="% of platform", thickness=14),
    ))
    fig.update_layout(
        xaxis_title="", yaxis_title="",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=180, r=20, t=55, b=30),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8. TOPIC SENTIMENT RANKING
# ─────────────────────────────────────────────────────────────────────────────
def topic_sentiment_bars(topic_sent_df: pd.DataFrame, top_n=10,
                          positive=True, title=None) -> go.Figure:
    if topic_sent_df is None or len(topic_sent_df) == 0:
        fig = _fig(title or "")
        fig.add_annotation(text="No topic sentiment data available.",
                           showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    sort_col = "pct_pos" if positive else "pct_neg"
    if sort_col not in topic_sent_df.columns:
        sort_col = "mean_vader"

    df_plot = topic_sent_df.sort_values(sort_col, ascending=False).head(top_n)
    color   = "#1D9E75" if positive else "#E24B4A"
    label   = "% Positive Reviews" if positive else "% Negative Reviews"
    if title is None:
        title = f"Top {top_n} {'Praised' if positive else 'Complained-About'} Topics"

    fig = _fig(title, height=max(300, top_n * 38 + 80))
    fig.add_trace(go.Bar(
        y=df_plot["lda_topic_label"] if "lda_topic_label" in df_plot.columns
          else df_plot.iloc[:, 0],
        x=df_plot[sort_col],
        orientation="h",
        marker_color=color, opacity=0.85,
        text=[f"{v:.1f}%" for v in df_plot[sort_col]],
        textposition="outside",
        hovertemplate="%{y}<br>" + label + ": %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title=label, yaxis_title="",
        showlegend=False, plot_bgcolor="white",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 9. MIGRATION MATRIX HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
def migration_matrix_heatmap(matrix: pd.DataFrame,
                              title="Migration Matrix (Source → Destination)") -> go.Figure:
    display = matrix.copy().astype(float)
    display.index   = [APP_LABELS[a] for a in APP_ORDER if a in display.index]
    display.columns = [APP_LABELS[a] for a in APP_ORDER if a in display.columns]
    # Mask diagonal
    np_diag = np.diag_indices_from(display.values)
    z = display.values.copy().astype(float)
    for i in range(len(z)):
        z[i][i] = None

    fig = _fig(title, height=420)
    fig.add_trace(go.Heatmap(
        z=z, x=display.columns.tolist(), y=display.index.tolist(),
        colorscale="Blues",
        text=display.values.astype(int),
        texttemplate="%{text}",
        textfont=dict(size=13, color="black"),
        hovertemplate="FROM %{y} → TO %{x}<br>Count: %{z:.0f}<extra></extra>",
        colorbar=dict(title="Count", thickness=14),
    ))
    fig.update_layout(
        xaxis_title="Destination Platform",
        yaxis_title="Source Platform",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 10. MIGRATION NET SCORE BAR
# ─────────────────────────────────────────────────────────────────────────────
def net_migration_bar(net_df: pd.DataFrame,
                      title="Net Migration Score by Platform") -> go.Figure:
    df_s   = net_df.sort_values("net")
    labels = df_s["label"].tolist()
    nets   = df_s["net"].tolist()
    colors = ["#1D9E75" if v >= 0 else "#E24B4A" for v in nets]

    fig = _fig(title, height=380)
    fig.add_trace(go.Bar(
        y=labels, x=nets, orientation="h",
        marker_color=colors,
        text=[f"{v:+d}" for v in nets],
        textposition="outside",
        hovertemplate="%{y}<br>Net migration: %{x:+d}<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="solid", line_color="#333333", line_width=1.5)
    fig.update_layout(
        xaxis_title="Net Migration (Arrivals − Departures)",
        yaxis_title="", showlegend=False, plot_bgcolor="white",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 11. MIGRATION NETWORK GRAPH (Plotly + NetworkX)
# ─────────────────────────────────────────────────────────────────────────────
def migration_network(G: nx.DiGraph,
                      title="Platform Migration Network") -> go.Figure:
    fig = _fig(title, height=580)
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#f0f2f6",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    if G.number_of_nodes() == 0:
        fig.add_annotation(text="No network data available.",
                           showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    pos = nx.spring_layout(G, seed=42, k=2.5)

    # Edge weights for scaling
    edges_data = list(G.edges(data=True))
    max_w      = max((d.get("weight", 1) for _, _, d in edges_data), default=1)

    # Draw edges as lines
    for u, v, data in edges_data:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        w       = data.get("weight", 1)
        opacity = 0.30 + 0.55 * (w / max_w)
        lw      = max(1.0, (w / max_w) * 5.5)
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=lw, color=f"rgba(90,90,90,{opacity:.2f})"),
            hoverinfo="skip",
            showlegend=False,
        ))
        # Arrow annotation (slightly before destination to clear node marker)
        dx, dy  = x1 - x0, y1 - y0
        length  = math.sqrt(dx * dx + dy * dy)
        if length > 0:
            ex = x0 + dx * 0.82
            ey = y0 + dy * 0.82
        else:
            ex, ey = x1, y1
        fig.add_annotation(
            ax=x0, ay=y0, x=ex, y=ey,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.8,
            arrowwidth=max(1.2, (w / max_w) * 3.5),
            arrowcolor=f"rgba(60,60,60,{opacity:.2f})",
            text=f"  {w}", font=dict(size=9, color="#333333"),
            bgcolor="rgba(255,255,255,0.6)", borderpad=1,
        )

    # Draw nodes
    in_deg  = dict(G.in_degree(weight="weight"))
    out_deg = dict(G.out_degree(weight="weight"))
    node_xs, node_ys, node_cols, node_szs, hover_txts = [], [], [], [], []
    for node in G.nodes():
        attrs   = G.nodes[node]
        nx_, ny = pos[node]
        reviews = attrs.get("reviews", 10000)
        node_xs.append(nx_)
        node_ys.append(ny)
        node_cols.append(attrs.get("color", "#888888"))
        node_szs.append(max(35, reviews / 180))
        node_net = in_deg.get(node, 0) - out_deg.get(node, 0)
        hover_txts.append(
            f"<b>{node}</b><br>"
            f"In-flow  : {in_deg.get(node,0)}<br>"
            f"Out-flow : {out_deg.get(node,0)}<br>"
            f"Net      : {node_net:+d}"
        )

    fig.add_trace(go.Scatter(
        x=node_xs, y=node_ys,
        mode="markers+text",
        marker=dict(size=node_szs, color=node_cols,
                    line=dict(width=3, color="white")),
        text=list(G.nodes()),
        textposition="top center",
        textfont=dict(size=12, color="#1a1a2e"),
        hovertext=hover_txts,
        hoverinfo="text",
        showlegend=False,
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 12. MIGRATION REASONS BAR
# ─────────────────────────────────────────────────────────────────────────────
def migration_reasons_bar(df: pd.DataFrame,
                           title="Migration Reason Categories") -> go.Figure:
    if "migration_reason_category" not in df.columns:
        fig = _fig(title)
        fig.add_annotation(text="Migration reason data not available.",
                           showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    mig = df[df.get("migration_flag", pd.Series(False, index=df.index)) == True]
    if len(mig) == 0:
        fig = _fig(title)
        fig.add_annotation(text="No migration events detected in current filter.",
                           showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    counts = mig["migration_reason_category"].value_counts().sort_values()
    colors = [f"hsl({int(230 - i*20)},65%,55%)" for i in range(len(counts))]

    fig = _fig(title, height=max(320, len(counts) * 40 + 80))
    fig.add_trace(go.Bar(
        y=counts.index, x=counts.values, orientation="h",
        marker_color=colors,
        text=[f"{v} ({v/len(mig)*100:.1f}%)" for v in counts.values],
        textposition="outside",
        hovertemplate="%{y}: %{x} events<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Migration Events", yaxis_title="",
        showlegend=False, plot_bgcolor="white",
        xaxis_range=[0, counts.max() * 1.3],
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 13. PLATFORM MENTION BARS
# ─────────────────────────────────────────────────────────────────────────────
def platform_mention_bar(df: pd.DataFrame,
                          title="Competitor Mention Frequency") -> go.Figure:
    fig = _fig(title, height=380)
    if "mentioned_platforms" not in df.columns:
        fig.add_annotation(text="Mention data not available — run Notebook 05",
                           showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    # Outward: how often each platform's reviews mention competitors
    outward = []
    inward  = []
    for app in APP_ORDER:
        sub     = df[df["App"] == app]
        out_cnt = sub.get("mentions_competitor",
                          pd.Series(False, index=sub.index)).sum() \
                  if "mentions_competitor" in sub.columns else 0
        in_cnt  = sum(
            app in str(row.get("mentioned_platforms", "")).split("|")
            for _, row in df[df["App"] != app].iterrows()
        ) if "mentioned_platforms" in df.columns else 0
        outward.append(int(out_cnt))
        inward.append(int(in_cnt))

    labels = [APP_LABELS[a] for a in APP_ORDER]
    x_pos  = list(range(len(APP_ORDER)))
    w      = 0.38

    fig.add_trace(go.Bar(
        x=[i - w/2 for i in x_pos], y=outward, width=w,
        name="Outward (mentions rival)", marker_color=PALETTE, opacity=0.9,
        text=outward, textposition="outside",
        hovertemplate="%{text} mentions<extra>Outward</extra>",
    ))
    fig.add_trace(go.Bar(
        x=[i + w/2 for i in x_pos], y=inward, width=w,
        name="Inward (mentioned by rivals)", marker_color="#378ADD", opacity=0.55,
        text=inward, textposition="outside",
        hovertemplate="%{text} mentions<extra>Inward</extra>",
    ))
    fig.update_layout(
        xaxis=dict(tickvals=x_pos, ticktext=labels),
        yaxis_title="Mention Count", barmode="group",
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 14. SATISFACTION RADAR / SCATTER (overview)
# ─────────────────────────────────────────────────────────────────────────────
def satisfaction_scatter(stats: pd.DataFrame,
                          title="Satisfaction: Stars vs Sentiment") -> go.Figure:
    fig = _fig(title, height=420)
    if len(stats) == 0:
        return fig

    for _, row in stats.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["mean_stars"]], y=[row.get("mean_vader", 0)],
            mode="markers+text",
            marker=dict(size=max(20, row["n_reviews"] / 700),
                        color=row["color"], line=dict(width=2.5, color="white")),
            text=[row["label"]],
            textposition="top center",
            textfont=dict(size=11, color=row["color"]),
            name=row["label"],
            hovertemplate=(
                f"<b>{row['label']}</b><br>"
                f"Avg Stars: {row['mean_stars']:.2f}<br>"
                f"VADER Mean: {row.get('mean_vader',0):.4f}<br>"
                f"Reviews: {row['n_reviews']:,}<extra></extra>"
            ),
        ))
    fig.update_layout(
        xaxis_title="Mean Star Rating",
        yaxis_title="Mean VADER Sentiment",
        showlegend=False,
        xaxis=dict(range=[1, 5.5]),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#cccccc", opacity=0.6)
    return fig
