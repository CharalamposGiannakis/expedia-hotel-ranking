"""
Shared utilities for the hotel ranking pipeline.
Reusable functions that multiple scripts depend on.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_raw(path, sample_frac=None, random_state=42):
    """Load raw CSV. Optionally sample by srch_id groups."""
    df = pd.read_csv(path)
    if sample_frac and sample_frac < 1.0:
        srch_ids = df["srch_id"].unique()
        n = int(len(srch_ids) * sample_frac)
        rng = np.random.RandomState(random_state)
        keep = rng.choice(srch_ids, size=n, replace=False)
        df = df[df["srch_id"].isin(keep)].reset_index(drop=True)
        print(f"  Sampled {n}/{len(srch_ids)} searches → {len(df):,} rows")
    return df


def make_relevance(df):
    """Create relevance target from click_bool and booking_bool."""
    df = df.copy()
    df["relevance"] = 0
    df.loc[df["click_bool"] == 1, "relevance"] = 1
    df.loc[df["booking_bool"] == 1, "relevance"] = 5
    return df


def ndcg_at_k(relevances, k=5):
    """Compute NDCG@k for a single query's ranked list of relevances."""
    relevances = np.asarray(relevances)

    # DCG from the model's top-k ranking
    top_k = relevances[:k]
    dcg = np.sum(top_k / np.log2(np.arange(2, len(top_k) + 2)))

    # IDCG from the ideal top-k ranking over the FULL query
    ideal_top_k = np.sort(relevances)[::-1][:k]
    idcg = np.sum(ideal_top_k / np.log2(np.arange(2, len(ideal_top_k) + 2)))

    return dcg / idcg if idcg > 0 else 0.0


def eval_ndcg(df, score_col, k=5):
    """
    Compute mean NDCG@k across all searches.
    df must contain: srch_id, relevance, and score_col (higher = better).
    """
    results = []
    for _, group in df.groupby("srch_id"):
        ranked = group.sort_values(score_col, ascending=False)
        results.append(ndcg_at_k(ranked["relevance"].values, k))
    return np.mean(results)


def save_fig(fig, name, figure_dir):
    """Save matplotlib figure to the figures directory."""
    path = Path(figure_dir) / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved figure: {path}")


def print_header(title):
    """Print a visible section header to stdout."""
    line = "=" * 60
    print(f"\n{line}\n  {title}\n{line}\n")
