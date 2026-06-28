"""
02c — Non-Target Feature Blocks (v3)

MUST RUN AFTER 03_split.py (requires train/val split to exist).

Two independent feature blocks, each toggleable:

  F1: Destination/country-normalized price features
      Computed from training split prices only, applied to val/test.
      Zero leakage risk — uses only price_usd, never targets.

  F2: Non-target property profile aggregates
      Hotel-level stats computed from feature columns only.
      Captures "is this hotel running a deal" and "how widely shown is it"
      without using click/booking targets.

Pipeline order:
  00 → 01 → 02 → 03 → 02c → 04 → 05 → 06

Usage:
  python src/02c_feature_v3_nontarget.py                # both F1 + F2
  python src/02c_feature_v3_nontarget.py --only f1      # F1 only
  python src/02c_feature_v3_nontarget.py --only f2      # F2 only

Run time: ~2 min on full data.
"""


import argparse
import pandas as pd
import numpy as np

from config import TRAIN_SPLIT, VAL_SPLIT, TRAIN_FE, TEST_FE, OUTPUT_DIR
from src.utils import print_header

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════
#  F1: Destination/country-normalized price
# ════════════════════════════════════════════════════════════════════

def compute_price_norms(train):
    """Compute destination-level and country-level price statistics from train."""
    dest_stats = train.groupby("srch_destination_id")["price_usd"].agg(
        dest_price_median="median",
        dest_price_mean="mean",
    ).reset_index()

    country_stats = train.groupby("prop_country_id")["price_usd"].agg(
        country_price_median="median",
        country_price_mean="mean",
    ).reset_index()

    return dest_stats, country_stats


def apply_f1(df, dest_stats, country_stats):
    """Apply F1 price normalization features to a dataframe."""
    # Drop any pre-existing F1/intermediate columns to avoid merge conflicts
    f1_cols = [
        "dest_price_median", "dest_price_mean",
        "country_price_median", "country_price_mean",
        "price_over_dest_median", "price_over_dest_mean",
        "price_minus_dest_median", "log_price_minus_dest_log_median",
        "price_over_country_median", "price_over_country_mean",
    ]
    existing = [c for c in f1_cols if c in df.columns]
    if existing:
        df = df.drop(columns=existing)

    # Merge destination stats
    df = df.merge(dest_stats, on="srch_destination_id", how="left")

    # Price relative to destination
    df["price_over_dest_median"] = df["price_usd"] / df["dest_price_median"].replace(0, np.nan)
    df["price_over_dest_mean"] = df["price_usd"] / df["dest_price_mean"].replace(0, np.nan)
    df["price_minus_dest_median"] = df["price_usd"] - df["dest_price_median"]

    # Log-scale version
    df["log_price_minus_dest_log_median"] = (
        np.log1p(df["price_usd"]) - np.log1p(df["dest_price_median"])
    )

    # Drop intermediate columns
    df.drop(columns=["dest_price_median", "dest_price_mean"], inplace=True)

    # Merge country stats
    df = df.merge(country_stats, on="prop_country_id", how="left")

    # Price relative to country
    df["price_over_country_median"] = df["price_usd"] / df["country_price_median"].replace(0, np.nan)
    df["price_over_country_mean"] = df["price_usd"] / df["country_price_mean"].replace(0, np.nan)

    # Drop intermediate columns
    df.drop(columns=["country_price_median", "country_price_mean"], inplace=True)

    new_features = [
        "price_over_dest_median",
        "price_over_dest_mean",
        "price_minus_dest_median",
        "log_price_minus_dest_log_median",
        "price_over_country_median",
        "price_over_country_mean",
    ]

    return df, new_features


def build_f1(train, val, test, train_full):
    """Build F1 features for all splits."""
    print("\n  --- F1: Destination/Country Price Normalization ---")

    # Compute stats from training split only
    dest_stats, country_stats = compute_price_norms(train)

    print(f"  Destination price stats: {len(dest_stats)} destinations")
    print(f"  Country price stats: {len(country_stats)} countries")

    train, feat_names = apply_f1(train, dest_stats, country_stats)
    val, _ = apply_f1(val, dest_stats, country_stats)

    # For test: compute from full training data
    dest_stats_full, country_stats_full = compute_price_norms(train_full)
    test, _ = apply_f1(test, dest_stats_full, country_stats_full)

    # Coverage
    for name in feat_names:
        t_null = train[name].isna().mean() * 100
        v_null = val[name].isna().mean() * 100
        print(f"  {name}: train null={t_null:.2f}%, val null={v_null:.2f}%")

    return train, val, test, feat_names


# ════════════════════════════════════════════════════════════════════
#  F2: Non-target property profile aggregates
# ════════════════════════════════════════════════════════════════════

def compute_property_profiles(train):
    """Compute property-level feature aggregates from train (no targets)."""
    profiles = train.groupby("prop_id").agg(
        prop_seen_count=("price_usd", "size"),
        prop_avg_price=("price_usd", "mean"),
        prop_median_price=("price_usd", "median"),
        prop_price_std=("price_usd", "std"),
        prop_promotion_rate=("promotion_flag", "mean"),
        prop_distinct_destinations=("srch_destination_id", "nunique"),
        prop_distinct_sites=("site_id", "nunique"),
    ).reset_index()

    # Fill std NaN (hotels appearing once) with 0
    profiles["prop_price_std"] = profiles["prop_price_std"].fillna(0)

    return profiles


def apply_f2(df, profiles):
    """Apply F2 property profile features to a dataframe."""
    # Drop any pre-existing profile columns to avoid merge conflicts
    # (can happen if test_fe.parquet was modified by a prior experiment)
    profile_cols = [c for c in profiles.columns if c != "prop_id"]
    existing = [c for c in profile_cols if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
    
    df = df.merge(profiles, on="prop_id", how="left")

    # Hotels not seen in training → fill with sensible defaults
    df["prop_seen_count"] = df["prop_seen_count"].fillna(0)
    df["prop_avg_price"] = df["prop_avg_price"].fillna(df["price_usd"])
    df["prop_median_price"] = df["prop_median_price"].fillna(df["price_usd"])
    df["prop_price_std"] = df["prop_price_std"].fillna(0)
    df["prop_promotion_rate"] = df["prop_promotion_rate"].fillna(0)
    df["prop_distinct_destinations"] = df["prop_distinct_destinations"].fillna(0)
    df["prop_distinct_sites"] = df["prop_distinct_sites"].fillna(0)

    # Derived: is current price a deal vs this hotel's own average?
    df["price_vs_prop_avg"] = df["price_usd"] / df["prop_avg_price"].replace(0, np.nan)
    df["price_vs_prop_median"] = df["price_usd"] / df["prop_median_price"].replace(0, np.nan)
    df["price_diff_from_prop_avg"] = df["price_usd"] - df["prop_avg_price"]

    new_features = [
        "prop_seen_count",
        "prop_avg_price",
        "prop_median_price",
        "prop_price_std",
        "prop_promotion_rate",
        "prop_distinct_destinations",
        "prop_distinct_sites",
        "price_vs_prop_avg",
        "price_vs_prop_median",
        "price_diff_from_prop_avg",
    ]

    return df, new_features


def build_f2(train, val, test, train_full):
    """Build F2 features for all splits."""
    print("\n  --- F2: Non-Target Property Profiles ---")

    # Compute from training split only
    profiles = compute_property_profiles(train)

    print(f"  Property profiles: {len(profiles)} unique hotels")

    train, feat_names = apply_f2(train, profiles)
    val, _ = apply_f2(val, profiles)

    # For test: compute from full training data
    profiles_full = compute_property_profiles(train_full)
    test, _ = apply_f2(test, profiles_full)

    # Coverage
    val_coverage = val["prop_id"].isin(profiles["prop_id"]).mean() * 100
    print(f"  Val coverage: {val_coverage:.1f}% of val hotels seen in train")

    for name in feat_names:
        t_mean = train[name].mean()
        v_mean = val[name].mean()
        print(f"  {name}: train mean={t_mean:.4f}, val mean={v_mean:.4f}")

    return train, val, test, feat_names


# ════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["f1", "f2"], default=None,
                        help="Run only one feature block")
    args = parser.parse_args()

    run_f1 = args.only in (None, "f1")
    run_f2 = args.only in (None, "f2")

    print_header("02c — NON-TARGET FEATURES (v3)")

    # Load data
    train = pd.read_parquet(TRAIN_SPLIT)
    val = pd.read_parquet(VAL_SPLIT)
    train_full = pd.read_parquet(TRAIN_FE)
    test = pd.read_parquet(TEST_FE)

    print(f"  Train: {len(train):,} rows, {train['srch_id'].nunique():,} searches")
    print(f"  Val:   {len(val):,} rows, {val['srch_id'].nunique():,} searches")
    print(f"  Full train (for test encoding): {len(train_full):,} rows")
    print(f"  Test: {len(test):,} rows")
    print(f"  Blocks: F1={'ON' if run_f1 else 'OFF'}, F2={'ON' if run_f2 else 'OFF'}")

    all_new_features = []

    if run_f1:
        train, val, test, f1_features = build_f1(train, val, test, train_full)
        all_new_features.extend(f1_features)

    if run_f2:
        train, val, test, f2_features = build_f2(train, val, test, train_full)
        all_new_features.extend(f2_features)

    # Save
    train.to_parquet(TRAIN_SPLIT, index=False)
    val.to_parquet(VAL_SPLIT, index=False)
    test.to_parquet(TEST_FE, index=False)

    print(f"\n  Updated: {TRAIN_SPLIT}")
    print(f"  Updated: {VAL_SPLIT}")
    print(f"  Updated: {TEST_FE}")

    # Feature count
    drop = {
        "srch_id", "date_time", "position", "click_bool", "booking_bool",
        "gross_bookings_usd", "gross_booking_usd", "relevance", "prop_id",
    }
    for i in range(1, 9):
        drop.update({f"comp{i}_rate", f"comp{i}_inv", f"comp{i}_rate_percent_diff"})
    total = len([c for c in train.columns if c not in drop])

    print(f"\n  New features added ({len(all_new_features)}):")
    for f in all_new_features:
        print(f"    - {f}")
    print(f"\n  Total features now: {total} (was 79 in v1)")

    print("\n✓ v3 non-target features complete.")
    print("  Pipeline: rerun 04 → 06 to see impact.")


if __name__ == "__main__":
    main()
