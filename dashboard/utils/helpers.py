"""
SentimentCompass — Analytics Helpers
Pure computation functions (no Streamlit calls).
"""
import math
import numpy as np
import pandas as pd
import networkx as nx
from utils.data_loader import APP_COLORS, APP_ORDER, APP_LABELS


def compute_twsi(df: pd.DataFrame,
                 score_col: str = "vader_compound",
                 weight_col: str = "thumbs_log") -> pd.Series:
    """
    Thumbs-Weighted Sentiment Index per platform.
    TWSI = Σ(log(1+thumbs) × sentiment) / Σ(log(1+thumbs))
    Falls back to simple mean if denominator is 0.
    """
    results = {}
    for app in APP_ORDER:
        sub = df[df["App"] == app]
        if len(sub) == 0:
            results[app] = 0.0
            continue
        if score_col not in sub.columns:
            results[app] = 0.0
            continue
        scores  = sub[score_col].fillna(0)
        weights = sub[weight_col].fillna(0) if weight_col in sub.columns \
                  else pd.Series(1.0, index=sub.index)
        denom   = weights.sum()
        results[app] = float((weights * scores).sum() / denom) \
                       if denom > 0 else float(scores.mean())
    return pd.Series(results)


def get_platform_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-platform summary DataFrame with all key metrics."""
    rows = []
    for app in APP_ORDER:
        sub = df[df["App"] == app]
        if len(sub) == 0:
            continue
        row = {
            "app"      : app,
            "label"    : APP_LABELS[app],
            "color"    : APP_COLORS[app],
            "n_reviews": len(sub),
            "mean_stars": round(sub["Star_Rating"].mean(), 3),
            "pct_5star" : round((sub["Star_Rating"] == 5).mean() * 100, 1),
            "pct_1star" : round((sub["Star_Rating"] == 1).mean() * 100, 1),
        }
        for src_col, key in [
            ("vader_compound",  "mean_vader"),
            ("tb_polarity",     "mean_tb"),
            ("roberta_score",   "mean_roberta"),
            ("tb_subjectivity", "mean_subjectivity"),
            ("Sentiment_Polarity", "mean_baseline"),
        ]:
            row[key] = round(float(sub[src_col].dropna().mean()), 4) \
                       if src_col in sub.columns and sub[src_col].notna().any() else 0.0

        for src_col, key in [
            ("is_positive_review", "pct_positive"),
            ("is_negative_review", "pct_negative"),
            ("is_neutral_review",  "pct_neutral"),
        ]:
            row[key] = round(float(sub[src_col].mean()) * 100, 1) \
                       if src_col in sub.columns else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_migration_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build 5×5 directed migration count matrix."""
    matrix = pd.DataFrame(0, index=APP_ORDER, columns=APP_ORDER)
    if "migration_flag" not in df.columns:
        return matrix
    mig = df[df["migration_flag"] == True].dropna(
        subset=["migration_source", "migration_destination"]
    )
    for _, row in mig.iterrows():
        src, dst = row["migration_source"], row["migration_destination"]
        if src in APP_ORDER and dst in APP_ORDER and src != dst:
            matrix.loc[src, dst] += 1
    return matrix


def build_networkx_graph(matrix: pd.DataFrame, df: pd.DataFrame) -> nx.DiGraph:
    """Build a NetworkX DiGraph from the migration matrix."""
    G = nx.DiGraph()
    for app in APP_ORDER:
        G.add_node(
            APP_LABELS[app],
            color   = APP_COLORS[app],
            reviews = int((df["App"] == app).sum()),
            app_key = app,
        )
    for src in APP_ORDER:
        for dst in APP_ORDER:
            if src != dst:
                w = int(matrix.loc[src, dst])
                if w > 0:
                    G.add_edge(APP_LABELS[src], APP_LABELS[dst], weight=w)
    return G


def get_net_migration(matrix: pd.DataFrame) -> pd.DataFrame:
    """Return in-flow, out-flow, and net migration per platform."""
    rows = []
    for app in APP_ORDER:
        in_flow  = int(matrix[app].sum())
        out_flow = int(matrix.loc[app].sum())
        rows.append({
            "app"     : app,
            "label"   : APP_LABELS[app],
            "color"   : APP_COLORS[app],
            "in_flow" : in_flow,
            "out_flow": out_flow,
            "net"     : in_flow - out_flow,
        })
    return pd.DataFrame(rows).sort_values("net", ascending=False).reset_index(drop=True)


def get_mention_counts(df: pd.DataFrame) -> pd.Series:
    """Count how often each platform is mentioned in competitors' reviews."""
    counts = {app: 0 for app in APP_ORDER}
    if "mentioned_platforms" not in df.columns:
        return pd.Series(counts)
    for _, row in df.iterrows():
        mentioned = [m for m in str(row.get("mentioned_platforms", "")).split("|") if m]
        for m in mentioned:
            if m in counts and m != row["App"]:
                counts[m] += 1
    return pd.Series(counts)


def get_topic_sentiment(df: pd.DataFrame, topic_col: str = "lda_topic_label") -> pd.DataFrame:
    """Return per-topic sentiment statistics."""
    if topic_col not in df.columns or "vader_compound" not in df.columns:
        return pd.DataFrame()
    ts = (
        df[df[topic_col].notna()]
        .groupby(topic_col)
        .agg(
            count      = (topic_col,       "count"),
            mean_vader = ("vader_compound", "mean"),
            pct_pos    = ("is_positive_review", "mean") if "is_positive_review" in df.columns
                          else ("vader_compound", lambda x: (x > 0.05).mean()),
            pct_neg    = ("is_negative_review", "mean") if "is_negative_review" in df.columns
                          else ("vader_compound", lambda x: (x < -0.05).mean()),
        )
        .reset_index()
    )
    ts["pct_pos"] = (ts["pct_pos"] * 100).round(1)
    ts["pct_neg"] = (ts["pct_neg"] * 100).round(1)
    ts["mean_vader"] = ts["mean_vader"].round(4)
    return ts.sort_values("count", ascending=False).reset_index(drop=True)
