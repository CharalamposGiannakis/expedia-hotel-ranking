"""
02 — Feature Engineering
This is where the competition is won.
Features are grouped into categories for clarity.

Feature categories:
  A. Raw features (pass-through)
  B. Missing-value indicators
  C. Within-search normalization (rank / z-score relative to other hotels in same search)
  D. Price competitiveness features
  E. Visitor–property match features
  F. Competitor aggregation features
  G. Property historical quality signals
  H. Search context features

Run time: ~5 min on full data.
"""


import pandas as pd
import numpy as np

from config import (
    TRAIN_RAW, TEST_RAW, TRAIN_FE, TEST_FE, OUTPUT_DIR,
    SAMPLE_FRAC, RANDOM_STATE,
    RAW_FEATURES, COMP_RATE_COLS, COMP_INV_COLS, COMP_DIFF_COLS,
)
from src.utils import load_raw, make_relevance, print_header

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════
#  Feature construction functions
# ════════════════════════════════════════════════════════════════════

def add_missing_indicators(df):
    """B. Binary flags for strategically important missing values and semantic zeros."""
    # NaN flags
    high_null_cols = [
        "visitor_hist_starrating", "visitor_hist_adr_usd",
        "prop_review_score", "prop_location_score2",
        "srch_query_affinity_score", "orig_destination_distance",
    ]
    for col in high_null_cols:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isnull().astype(int)
    
    # Semantic-zero flags (per competition data dictionary):
    #   prop_starrating=0     → "no stars, unknown, or cannot be publicized"
    #   prop_review_score=0   → "no reviews"
    #   prop_log_historical_price=0 → "hotel was not sold in that period"
    df["prop_starrating_zero"] = (df["prop_starrating"] == 0).astype(int)
    df["prop_review_score_zero"] = (df["prop_review_score"] == 0).astype(int)
    df["prop_log_historical_price_zero"] = (df["prop_log_historical_price"] == 0).astype(int)
    
    return df


def add_within_search_features(df):
    """C. Rank and z-score features relative to the search group."""
    grp = df.groupby("srch_id")
    
    # Price rank within search (lower = cheaper = better)
    df["price_rank"] = grp["price_usd"].rank(method="min", ascending=True)
    df["price_rank_pct"] = grp["price_usd"].rank(pct=True)
    
    # Star rating rank
    df["star_rank"] = grp["prop_starrating"].rank(method="min", ascending=False)
    
    # Review score rank
    df["review_rank"] = grp["prop_review_score"].rank(method="min", ascending=False)
    
    # Location score ranks (keep loc1 for model, but loc2 is the primary signal)
    df["loc1_rank"] = grp["prop_location_score1"].rank(method="min", ascending=False)
    df["loc2_rank"] = grp["prop_location_score2"].rank(method="min", ascending=False)
    
    # Z-score normalization within search for key numeric features
    for col in ["price_usd", "prop_starrating", "prop_review_score",
                "prop_location_score1", "prop_location_score2"]:
        mean = grp[col].transform("mean")
        std = grp[col].transform("std").replace(0, 1)
        df[f"{col}_zscore"] = (df[col] - mean) / std
    
    # Number of properties in search (search competitiveness)
    df["search_n_props"] = grp["prop_id"].transform("count")
    
    # Price relative to search mean
    search_mean_price = grp["price_usd"].transform("mean")
    df["price_vs_search_mean"] = df["price_usd"] / search_mean_price.replace(0, np.nan)
    
    # Best in search flags (loc uses score2, the stronger signal per EDA)
    df["is_cheapest"] = (df["price_rank"] == 1).astype(int)
    df["is_best_review"] = (df["review_rank"] == 1).astype(int)
    df["is_best_location"] = (df["loc2_rank"] == 1).astype(int)
    df["is_best_star"] = (df["star_rank"] == 1).astype(int)
    
    # Gap-to-best features: how far is this hotel from the best option in the search
    df["price_gap_to_cheapest"] = df["price_usd"] - grp["price_usd"].transform("min")
    df["star_gap_to_best"] = grp["prop_starrating"].transform("max") - df["prop_starrating"]
    df["review_gap_to_best"] = grp["prop_review_score"].transform("max") - df["prop_review_score"]
    df["loc2_gap_to_best"] = grp["prop_location_score2"].transform("max") - df["prop_location_score2"]
    
    return df


def add_price_features(df):
    """D. Price competitiveness and value features."""
    # Price per person
    total_guests = (df["srch_adults_count"] + df["srch_children_count"]).replace(0, 1)
    df["price_per_person"] = df["price_usd"] / total_guests
    
    # Log price (handles extreme outliers: max=205k vs mean=174)
    df["log_price"] = np.log1p(df["price_usd"])
    
    # Price vs historical price (value signal)
    # Note: prop_log_historical_price=0 means "not sold in prior period", not price=1
    hist_log = df["prop_log_historical_price"]
    hist_price = np.exp(hist_log)
    hist_price = hist_price.where(hist_log > 0, np.nan)  # mask semantic zeros
    df["price_vs_historical"] = df["price_usd"] / hist_price
    
    # Promotion + price interaction
    df["promo_price_discount"] = df["promotion_flag"] * df["price_rank_pct"]
    
    return df


def add_visitor_match_features(df):
    """E. How well does this property match visitor history?"""
    # Star rating gap: visitor preference vs property
    df["star_gap"] = (df["visitor_hist_starrating"] - df["prop_starrating"]).abs()
    
    # Price gap: visitor typical spend vs this property
    df["price_gap_abs"] = (df["visitor_hist_adr_usd"] - df["price_usd"]).abs()
    df["price_ratio_to_hist"] = df["price_usd"] / df["visitor_hist_adr_usd"].replace(0, np.nan)
    
    # Has history at all (returning vs new customer)
    df["has_visitor_history"] = df["visitor_hist_starrating"].notna().astype(int)
    
    return df


def add_competitor_features(df):
    """F. Aggregate competitor signals into usable features."""
    # How many competitors have lower price?
    rate_cols = [c for c in COMP_RATE_COLS if c in df.columns]
    inv_cols = [c for c in COMP_INV_COLS if c in df.columns]
    diff_cols = [c for c in COMP_DIFF_COLS if c in df.columns]
    
    # Count competitors cheaper / same / more expensive
    if rate_cols:
        rates = df[rate_cols]
        df["comp_cheaper_count"] = (rates == -1).sum(axis=1)    # Expedia more expensive
        df["comp_pricier_count"] = (rates == 1).sum(axis=1)     # Expedia cheaper
        df["comp_rate_available"] = rates.notna().sum(axis=1)    # how many comparisons exist
        
        # Net competitive advantage (-1 to +1 scale)
        total = df["comp_rate_available"].replace(0, np.nan)
        df["comp_net_advantage"] = (df["comp_pricier_count"] - df["comp_cheaper_count"]) / total
    
    # Count competitors out of stock
    if inv_cols:
        invs = df[inv_cols]
        df["comp_out_of_stock"] = (invs == 1).sum(axis=1)
        df["comp_inv_available"] = invs.notna().sum(axis=1)
    
    # Mean competitor price difference
    if diff_cols:
        df["comp_mean_diff"] = df[diff_cols].mean(axis=1)
        df["comp_max_diff"] = df[diff_cols].max(axis=1)
    
    return df


def add_property_quality_features(df):
    """G. Property quality signals."""
    # Star–review agreement (quality consistency: inflated stars vs actual reviews)
    df["star_review_diff"] = df["prop_starrating"] - df["prop_review_score"]
    
    return df


def add_search_context_features(df):
    """H. Derived search-level context."""
    # Is family trip?
    df["is_family"] = (df["srch_children_count"] > 0).astype(int)
    
    # Total guests
    df["total_guests"] = df["srch_adults_count"] + df["srch_children_count"]
    
    # Is long stay?
    df["is_long_stay"] = (df["srch_length_of_stay"] >= 7).astype(int)
    
    # Is last minute?
    df["is_last_minute"] = (df["srch_booking_window"] <= 1).astype(int)
    
    # Is advance booking?
    df["is_advance_booking"] = (df["srch_booking_window"] >= 30).astype(int)
    
    # Same country (domestic travel)?
    df["is_domestic"] = (df["visitor_location_country_id"] == df["prop_country_id"]).astype(int)
    
    # Date features
    if "date_time" in df.columns:
        dt = pd.to_datetime(df["date_time"], errors="coerce")
        df["search_month"] = dt.dt.month
        df["search_dayofweek"] = dt.dt.dayofweek
        df["search_hour"] = dt.dt.hour
    
    return df


def build_features(df):
    """Run full feature pipeline."""
    df = add_missing_indicators(df)
    df = add_within_search_features(df)
    df = add_price_features(df)
    df = add_visitor_match_features(df)
    df = add_competitor_features(df)
    df = add_property_quality_features(df)
    df = add_search_context_features(df)
    return df


def get_feature_columns(df):
    """Return list of all feature columns (exclude IDs, targets, raw comp cols)."""
    drop = {
        "srch_id", "date_time", "position", "click_bool", "booking_bool",
        "gross_bookings_usd", "gross_booking_usd",  # both spellings to be safe
        "relevance", "prop_id",
    }
    # Also drop raw competitor columns (replaced by aggregations)
    for i in range(1, 9):
        drop.update({f"comp{i}_rate", f"comp{i}_inv", f"comp{i}_rate_percent_diff"})
    
    return [c for c in df.columns if c not in drop]


# ════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════

def main():
    print_header("02 — FEATURE ENGINEERING")
    
    # ── Train ──
    print("Loading and engineering train features...")
    train = load_raw(TRAIN_RAW, sample_frac=SAMPLE_FRAC, random_state=RANDOM_STATE)
    train = make_relevance(train)
    train = build_features(train)
    
    feature_cols = get_feature_columns(train)
    print(f"  Total features: {len(feature_cols)}")
    
    train.to_parquet(TRAIN_FE, index=False)
    print(f"  Saved: {TRAIN_FE}")
    
    # ── Test ──
    print("Loading and engineering test features...")
    test = pd.read_csv(TEST_RAW)
    test = build_features(test)
    test.to_parquet(TEST_FE, index=False)
    print(f"  Saved: {TEST_FE}")
    
    # ── Feature list ──
    feat_summary = pd.DataFrame({
        "feature": feature_cols,
        "dtype": [str(train[c].dtype) for c in feature_cols],
        "null_pct": [train[c].isnull().mean() * 100 for c in feature_cols],
    })
    feat_summary.to_csv(OUTPUT_DIR / "feature_list.csv", index=False)
    print(f"  Feature list saved: {OUTPUT_DIR / 'feature_list.csv'}")
    
    print("\n✓ Feature engineering complete.")


if __name__ == "__main__":
    main()
