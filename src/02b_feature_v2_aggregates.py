"""
02b — Fold-Safe Historical Aggregate Features (v2)

MUST RUN AFTER 03_split.py (requires train/val split to exist).

These features encode "how popular is this hotel historically?" using
target-based aggregations. They are the single biggest expected NDCG gain
because the v1 model has no way to distinguish a consistently-booked
hotel from one that never converts.

Leakage prevention:
  - Training rows: leave-one-out encoding with Bayesian smoothing
  - Validation rows: computed from full training set with smoothing
  - Test rows: computed from full training set with smoothing

Aggregation levels:
  A. Property-level (prop_id)
  B. Property × destination (prop_id × srch_destination_id)
  C. Destination-level (srch_destination_id)
  D. Property × site (prop_id × site_id)

Pipeline order:
  00 → 01 → 02 → 03 → 02b → 04 → 05 → 06 → 07

Run time: ~2 min on sampled data.
"""


import pandas as pd
import numpy as np

from config import TRAIN_SPLIT, VAL_SPLIT, TRAIN_FE, TEST_FE, OUTPUT_DIR
from src.utils import print_header

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Smoothing parameter ───────────────────────────────────────────
# Controls how much we shrink toward the global mean.
# Higher = more conservative (better for rare hotels).
# 100 is a standard starting point for ~5M rows.
SMOOTHING_FACTOR = 500


# ════════════════════════════════════════════════════════════════════
#  Core encoding functions
# ════════════════════════════════════════════════════════════════════

def smoothed_aggregate(group_stats, global_mean, smoothing=SMOOTHING_FACTOR):
    """
    Apply Bayesian smoothing: blend group mean toward global mean
    based on group size.
    
    smoothed = (n * group_mean + smoothing * global_mean) / (n + smoothing)
    
    Hotels with many observations → close to their own mean.
    Hotels with few observations → pulled toward global mean.
    """
    n = group_stats["count"]
    mean = group_stats["mean"]
    return (n * mean + smoothing * global_mean) / (n + smoothing)


def leave_one_out_mean(df, group_col, target_col, global_mean, smoothing=SMOOTHING_FACTOR):
    """
    Leave-one-out target encoding for training rows.
    For each row, the aggregate excludes that row's own target value.
    
    loo_mean_i = (group_sum - y_i) / (group_count - 1)
    Then smoothed toward global mean.
    """
    group_sum = df.groupby(group_col)[target_col].transform("sum")
    group_count = df.groupby(group_col)[target_col].transform("count")
    
    # Leave-one-out: subtract this row's contribution
    loo_sum = group_sum - df[target_col]
    loo_count = group_count - 1
    
    # For groups with only 1 member, fall back to global mean
    loo_mean = np.where(loo_count > 0, loo_sum / loo_count, global_mean)
    
    # Smooth toward global mean (using loo_count, not original count)
    smoothed = (loo_count * loo_mean + smoothing * global_mean) / (loo_count + smoothing)
    
    return smoothed


def compute_group_stats(train_df, group_cols, target_col):
    """Compute group-level stats from training data for applying to val/test."""
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    
    stats = train_df.groupby(group_cols)[target_col].agg(["mean", "count"]).reset_index()
    stats.columns = group_cols + ["mean", "count"]
    return stats


def apply_smoothed_to_df(df, stats, group_cols, global_mean, feature_name, smoothing=SMOOTHING_FACTOR):
    """Merge group stats onto df and compute smoothed feature."""
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    
    merged = df[group_cols].merge(stats, on=group_cols, how="left")
    
    # Hotels/groups not seen in training → use global mean
    merged["mean"] = merged["mean"].fillna(global_mean)
    merged["count"] = merged["count"].fillna(0)
    
    df[feature_name] = smoothed_aggregate(merged, global_mean, smoothing)
    return df


# ════════════════════════════════════════════════════════════════════
#  Feature definitions
# ════════════════════════════════════════════════════════════════════

AGGREGATE_CONFIGS = [
    # --- A. Property-level ---
    {
        "group_cols": "prop_id",
        "target": "booking_bool",
        "feature_name": "prop_booking_rate",
    },
    {
        "group_cols": "prop_id",
        "target": "click_bool",
        "feature_name": "prop_click_rate",
    },
    {
        "group_cols": "prop_id",
        "target": "relevance",
        "feature_name": "prop_mean_relevance",
    },
    # --- B. Property × destination ---
    {
        "group_cols": ["prop_id", "srch_destination_id"],
        "target": "booking_bool",
        "feature_name": "prop_dest_booking_rate",
    },
    # --- C. Destination-level ---
    {
        "group_cols": "srch_destination_id",
        "target": "booking_bool",
        "feature_name": "dest_booking_rate",
    },
    {
        "group_cols": "srch_destination_id",
        "target": "relevance",
        "feature_name": "dest_mean_relevance",
    },
    # --- D. Property × site ---
    {
        "group_cols": ["prop_id", "site_id"],
        "target": "booking_bool",
        "feature_name": "prop_site_booking_rate",
    },
]


# ════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════

def main():
    print_header("02b — FOLD-SAFE HISTORICAL AGGREGATES (v2)")
    
    # ── Load split data ──
    train = pd.read_parquet(TRAIN_SPLIT)
    val = pd.read_parquet(VAL_SPLIT)
    
    print(f"  Train: {len(train):,} rows, {train['srch_id'].nunique():,} searches")
    print(f"  Val:   {len(val):,} rows, {val['srch_id'].nunique():,} searches")
    
    # ── Also need full train for test encoding ──
    train_full = pd.read_parquet(TRAIN_FE)
    print(f"  Full train (for test encoding): {len(train_full):,} rows")
    
    new_features = []
    
    for cfg in AGGREGATE_CONFIGS:
        group_cols = cfg["group_cols"]
        target = cfg["target"]
        feat_name = cfg["feature_name"]
        
        group_label = group_cols if isinstance(group_cols, str) else " × ".join(group_cols)
        print(f"\n  Building: {feat_name} (group={group_label}, target={target})")
        
        global_mean = train[target].mean()
        print(f"    Global mean ({target}): {global_mean:.6f}")
        
        # --- Training: leave-one-out encoding ---
        if isinstance(group_cols, str):
            train[feat_name] = leave_one_out_mean(
                train, group_cols, target, global_mean, SMOOTHING_FACTOR
            )
        else:
            # For multi-key groups, create a temporary composite key
            key = "_".join(group_cols)
            train[key] = train[group_cols[0]].astype(str) + "_" + train[group_cols[1]].astype(str)
            train[feat_name] = leave_one_out_mean(
                train, key, target, global_mean, SMOOTHING_FACTOR
            )
            train.drop(columns=[key], inplace=True)
        
        # --- Validation: full training set stats ---
        stats = compute_group_stats(train, group_cols, target)
        val = apply_smoothed_to_df(val, stats, group_cols, global_mean, feat_name)
        
        # Coverage stats
        if isinstance(group_cols, str):
            val_groups = val[group_cols].nunique()
            train_groups = train[group_cols].nunique()
            coverage = val[group_cols].isin(train[group_cols].unique()).mean() * 100
        else:
            val_keys = val[group_cols].apply(tuple, axis=1)
            train_keys = train[group_cols].apply(tuple, axis=1)
            coverage = val_keys.isin(train_keys.unique()).mean() * 100
        
        print(f"    Val coverage: {coverage:.1f}% of val rows have matching train groups")
        print(f"    Train {feat_name}: mean={train[feat_name].mean():.6f}, std={train[feat_name].std():.6f}")
        print(f"    Val {feat_name}:   mean={val[feat_name].mean():.6f}, std={val[feat_name].std():.6f}")
        
        new_features.append(feat_name)
    
    # ── Property count feature (not target-based, so no LOO needed) ──
    print(f"\n  Building: prop_count (non-target, safe without LOO)")
    prop_counts = train.groupby("prop_id").size().reset_index(name="prop_count")
    train = train.merge(prop_counts, on="prop_id", how="left")
    val = val.merge(prop_counts, on="prop_id", how="left")
    val["prop_count"] = val["prop_count"].fillna(0)
    new_features.append("prop_count")
    print(f"    Train prop_count: mean={train['prop_count'].mean():.1f}")
    print(f"    Val prop_count:   mean={val['prop_count'].mean():.1f}")
    
    # ── Save updated splits ──
    train.to_parquet(TRAIN_SPLIT, index=False)
    val.to_parquet(VAL_SPLIT, index=False)
    print(f"\n  Updated: {TRAIN_SPLIT}")
    print(f"  Updated: {VAL_SPLIT}")
    
    # ── Build test features ──
    print(f"\n  Building test aggregate features from full training data...")
    test = pd.read_parquet(TEST_FE)
    
    for cfg in AGGREGATE_CONFIGS:
        group_cols = cfg["group_cols"]
        target = cfg["target"]
        feat_name = cfg["feature_name"]
        
        global_mean = train_full[target].mean()
        stats = compute_group_stats(train_full, group_cols, target)
        test = apply_smoothed_to_df(test, stats, group_cols, global_mean, feat_name)
    
    # Property count from full train
    prop_counts_full = train_full.groupby("prop_id").size().reset_index(name="prop_count")
    test = test.merge(prop_counts_full, on="prop_id", how="left")
    test["prop_count"] = test["prop_count"].fillna(0)
    
    test.to_parquet(TEST_FE, index=False)
    print(f"  Updated: {TEST_FE}")
    
    # ── Summary ──
    print(f"\n  New features added ({len(new_features)}):")
    for f in new_features:
        print(f"    - {f}")
    
    # Verify no NaN in new features
    for f in new_features:
        train_nan = train[f].isna().sum()
        val_nan = val[f].isna().sum()
        if train_nan > 0 or val_nan > 0:
            print(f"  ⚠ {f} has NaN: train={train_nan}, val={val_nan}")
    
    # Get clean feature count
    drop = {
        "srch_id", "date_time", "position", "click_bool", "booking_bool",
        "gross_bookings_usd", "gross_booking_usd", "relevance", "prop_id",
    }
    for i in range(1, 9):
        drop.update({f"comp{i}_rate", f"comp{i}_inv", f"comp{i}_rate_percent_diff"})
    total_features = len([c for c in train.columns if c not in drop])
    print(f"\n  Total features now: {total_features} (was 79 in v1)")
    
    print("\n✓ v2 aggregate features complete.")
    print("  Pipeline: rerun 04 → 05 → 06 to see impact.")


if __name__ == "__main__":
    main()
