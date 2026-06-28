"""
01 — Exploratory Data Analysis
Key distributions, correlations with target, position bias analysis.
Run time: ~3 min on sampled data.
"""


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    TRAIN_RAW, FIGURE_DIR, OUTPUT_DIR,
    SAMPLE_FRAC, RANDOM_STATE, RAW_FEATURES
)
from src.utils import load_raw, make_relevance, print_header, save_fig

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print_header("01 — EXPLORATORY DATA ANALYSIS")
    
    df = load_raw(TRAIN_RAW, sample_frac=SAMPLE_FRAC, random_state=RANDOM_STATE)
    df = make_relevance(df)
    
    # ── 1. Target distribution ──────────────────────────────────
    print("1. Target distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    df["click_bool"].value_counts().plot.bar(ax=axes[0], title="Click distribution")
    df["booking_bool"].value_counts().plot.bar(ax=axes[1], title="Booking distribution")
    plt.tight_layout()
    save_fig(fig, "target_distribution.png", FIGURE_DIR)
    plt.close()
    
    # ── 2. Position bias (critical for this competition) ────────
    print("2. Position bias analysis...")
    # Only non-random results show position bias
    normal = df[df["random_bool"] == 0]
    random = df[df["random_bool"] == 1]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # CTR by position (normal sort)
    pos_ctr_normal = normal.groupby("position")["click_bool"].mean().head(40)
    pos_ctr_random = random.groupby("position")["click_bool"].mean().head(40)
    axes[0].plot(pos_ctr_normal.index, pos_ctr_normal.values, label="Normal sort", marker=".")
    axes[0].plot(pos_ctr_random.index, pos_ctr_random.values, label="Random sort", marker=".")
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("Click-through rate")
    axes[0].set_title("CTR by Position")
    axes[0].legend()
    
    # Book rate by position
    pos_book_normal = normal.groupby("position")["booking_bool"].mean().head(40)
    pos_book_random = random.groupby("position")["booking_bool"].mean().head(40)
    axes[1].plot(pos_book_normal.index, pos_book_normal.values, label="Normal sort", marker=".")
    axes[1].plot(pos_book_random.index, pos_book_random.values, label="Random sort", marker=".")
    axes[1].set_xlabel("Position")
    axes[1].set_ylabel("Booking rate")
    axes[1].set_title("Booking Rate by Position")
    axes[1].legend()
    
    plt.tight_layout()
    save_fig(fig, "position_bias.png", FIGURE_DIR)
    plt.close()
    
    # ── 2b. Search size distribution ────────────────────────────
    print("2b. Search size distribution...")
    search_sizes = df.groupby("srch_id").size()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    axes[0].hist(search_sizes, bins=60, color="steelblue", alpha=0.7, edgecolor="white")
    axes[0].axvline(search_sizes.mean(), color="red", linestyle="--", label=f"Mean={search_sizes.mean():.1f}")
    axes[0].axvline(search_sizes.median(), color="orange", linestyle="--", label=f"Median={search_sizes.median():.0f}")
    axes[0].set_title("Properties per Search")
    axes[0].set_xlabel("Number of properties")
    axes[0].set_ylabel("Number of searches")
    axes[0].legend()
    
    # Booking rate by search size (do larger searches dilute signal?)
    search_stats = df.groupby("srch_id").agg(
        n_props=("prop_id", "size"),
        has_booking=("booking_bool", "max"),
        has_click=("click_bool", "max"),
    )
    size_bins = pd.qcut(search_stats["n_props"], 10, duplicates="drop")
    book_by_size = search_stats.groupby(size_bins)["has_booking"].mean()
    click_by_size = search_stats.groupby(size_bins)["has_click"].mean()
    
    axes[1].plot(range(len(book_by_size)), book_by_size.values, marker="o", label="Any booking")
    axes[1].plot(range(len(click_by_size)), click_by_size.values, marker="s", label="Any click")
    axes[1].set_xticks(range(len(book_by_size)))
    axes[1].set_xticklabels([str(b) for b in book_by_size.index], rotation=45, fontsize=7)
    axes[1].set_title("Engagement Rate by Search Size")
    axes[1].set_xlabel("Properties per search (binned)")
    axes[1].set_ylabel("Fraction of searches")
    axes[1].legend()
    
    plt.tight_layout()
    save_fig(fig, "search_size_distribution.png", FIGURE_DIR)
    plt.close()
    
    # Print search size stats for notes
    print(f"  Search size — min: {search_sizes.min()}, max: {search_sizes.max()}, "
          f"mean: {search_sizes.mean():.1f}, median: {search_sizes.median():.0f}")
    print(f"  P5={search_sizes.quantile(0.05):.0f}, P25={search_sizes.quantile(0.25):.0f}, "
          f"P75={search_sizes.quantile(0.75):.0f}, P95={search_sizes.quantile(0.95):.0f}")
    
    # ── 3. Feature distributions: booked vs not ─────────────────
    print("3. Feature distributions by booking status...")
    plot_features = [
        "prop_starrating", "prop_review_score", "price_usd",
        "prop_location_score1", "prop_location_score2",
        "srch_length_of_stay", "srch_booking_window",
        "orig_destination_distance", "promotion_flag"
    ]
    n = len(plot_features)
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, feat in enumerate(plot_features):
        if feat in df.columns:
            booked = df[df["booking_bool"] == 1][feat].dropna()
            not_booked = df[df["booking_bool"] == 0][feat].dropna()
            axes[i].hist(not_booked, bins=50, alpha=0.5, label="Not booked", density=True)
            axes[i].hist(booked, bins=50, alpha=0.5, label="Booked", density=True)
            axes[i].set_title(feat, fontsize=9)
            axes[i].legend(fontsize=7)
    
    plt.tight_layout()
    save_fig(fig, "feature_distributions_by_booking.png", FIGURE_DIR)
    plt.close()
    
    # ── 4. Correlation with relevance ───────────────────────────
    print("4. Feature correlations with relevance...")
    numeric_cols = [c for c in RAW_FEATURES if c in df.columns]
    corrs = df[numeric_cols + ["relevance"]].corr()["relevance"].drop("relevance").sort_values()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    corrs.plot.barh(ax=ax, color=["red" if v < 0 else "steelblue" for v in corrs])
    ax.set_title("Feature Correlation with Relevance")
    ax.set_xlabel("Pearson correlation")
    plt.tight_layout()
    save_fig(fig, "feature_correlations.png", FIGURE_DIR)
    plt.close()
    
    # ── 5. Missing value analysis ───────────────────────────────
    print("5. Missing values...")
    null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    null_pct = null_pct[null_pct > 0]
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(null_pct) * 0.3)))
    null_pct.plot.barh(ax=ax, color="coral")
    ax.set_title("Missing Value % (features with nulls)")
    ax.set_xlabel("% missing")
    plt.tight_layout()
    save_fig(fig, "missing_values.png", FIGURE_DIR)
    plt.close()
    
    # ── 6. Price analysis ───────────────────────────────────────
    print("6. Price analysis...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    df["price_usd"].clip(upper=df["price_usd"].quantile(0.99)).hist(
        bins=100, ax=axes[0], alpha=0.7
    )
    axes[0].set_title("Price Distribution (clipped 99th pct)")
    axes[0].set_xlabel("price_usd")
    
    # Price vs booking
    price_bins = pd.qcut(df["price_usd"].clip(upper=5000), 20, duplicates="drop")
    book_by_price = df.groupby(price_bins)["booking_bool"].mean()
    book_by_price.plot(ax=axes[1], marker=".")
    axes[1].set_title("Booking Rate by Price Bin")
    axes[1].set_xlabel("Price bin")
    axes[1].set_ylabel("Booking rate")
    axes[1].tick_params(axis="x", rotation=45)
    
    plt.tight_layout()
    save_fig(fig, "price_analysis.png", FIGURE_DIR)
    plt.close()
    
    # ── 7. Random_bool debiasing analysis ─────────────────────
    print("7. Random_bool debiasing analysis...")
    normal = df[df["random_bool"] == 0]
    random = df[df["random_bool"] == 1]
    
    report_lines = []
    report_lines.append("RANDOM_BOOL DEBIASING ANALYSIS")
    report_lines.append("=" * 50)
    report_lines.append(f"\nTotal rows:        {len(df):,}")
    report_lines.append(f"Normal sort rows:  {len(normal):,} ({len(normal)/len(df):.1%})")
    report_lines.append(f"Random sort rows:  {len(random):,} ({len(random)/len(df):.1%})")
    report_lines.append(f"\nNormal searches:   {normal['srch_id'].nunique():,}")
    report_lines.append(f"Random searches:   {random['srch_id'].nunique():,}")
    
    # Check if random_bool is per-search or per-row
    mixed = df.groupby("srch_id")["random_bool"].nunique()
    n_mixed = (mixed > 1).sum()
    report_lines.append(f"\nSearches with mixed random_bool: {n_mixed}")
    report_lines.append("  → " + ("Per-search flag (good)" if n_mixed == 0 
                                   else f"{n_mixed} searches have mixed values (unexpected)"))
    
    # Target rates comparison
    report_lines.append(f"\n--- Target rates ---")
    report_lines.append(f"{'':20s} {'Normal':>10s} {'Random':>10s} {'Ratio':>10s}")
    for col in ["click_bool", "booking_bool"]:
        r_norm = normal[col].mean()
        r_rand = random[col].mean()
        ratio = r_rand / r_norm if r_norm > 0 else float("nan")
        report_lines.append(f"{col:20s} {r_norm:10.4f} {r_rand:10.4f} {ratio:10.3f}")
    
    # Feature-target correlations: random vs normal
    # This shows which features are real predictors vs position-confounded
    report_lines.append(f"\n--- Feature–booking correlations by subset ---")
    report_lines.append(f"{'Feature':35s} {'Normal':>10s} {'Random':>10s} {'Δ':>10s} {'Note':>20s}")
    
    check_features = [
        "prop_starrating", "prop_review_score", "price_usd",
        "prop_location_score1", "prop_location_score2",
        "prop_brand_bool", "promotion_flag",
        "orig_destination_distance", "srch_query_affinity_score",
    ]
    
    corr_rows = []
    for feat in check_features:
        if feat not in df.columns:
            continue
        c_norm = normal[[feat, "booking_bool"]].dropna().corr().iloc[0, 1]
        c_rand = random[[feat, "booking_bool"]].dropna().corr().iloc[0, 1]
        delta = c_rand - c_norm
        note = ""
        if abs(delta) > 0.02:
            note = "POSITION-CONFOUNDED" if abs(c_norm) > abs(c_rand) else "STRONGER IN RANDOM"
        report_lines.append(f"{feat:35s} {c_norm:10.4f} {c_rand:10.4f} {delta:+10.4f} {note:>20s}")
        corr_rows.append({"feature": feat, "corr_normal": c_norm, "corr_random": c_rand, "delta": delta})
    
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUTPUT_DIR / "random_bool_correlations.csv", index=False)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 7a. Target rates comparison
    labels = ["Click Rate", "Booking Rate"]
    norm_rates = [normal["click_bool"].mean(), normal["booking_bool"].mean()]
    rand_rates = [random["click_bool"].mean(), random["booking_bool"].mean()]
    x = np.arange(len(labels))
    axes[0, 0].bar(x - 0.15, norm_rates, 0.3, label="Normal sort", color="steelblue")
    axes[0, 0].bar(x + 0.15, rand_rates, 0.3, label="Random sort", color="coral")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels)
    axes[0, 0].set_ylabel("Rate")
    axes[0, 0].set_title("Target Rates: Normal vs Random Sort")
    axes[0, 0].legend()
    
    # 7b. Correlation comparison (paired bar chart)
    if len(corr_df) > 0:
        x = np.arange(len(corr_df))
        axes[0, 1].barh(x - 0.2, corr_df["corr_normal"], 0.35, label="Normal", color="steelblue")
        axes[0, 1].barh(x + 0.2, corr_df["corr_random"], 0.35, label="Random", color="coral")
        axes[0, 1].set_yticks(x)
        axes[0, 1].set_yticklabels(corr_df["feature"], fontsize=8)
        axes[0, 1].set_xlabel("Correlation with booking_bool")
        axes[0, 1].set_title("Feature–Booking Correlation by Subset")
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].axvline(0, color="gray", linewidth=0.5)
    
    # 7c. Position effect on booking (normal vs random)
    pos_book_normal = normal.groupby("position")["booking_bool"].mean().head(40)
    pos_book_random = random.groupby("position")["booking_bool"].mean().head(40)
    axes[1, 0].plot(pos_book_normal.index, pos_book_normal.values, label="Normal sort", marker=".", markersize=3)
    axes[1, 0].plot(pos_book_random.index, pos_book_random.values, label="Random sort", marker=".", markersize=3)
    axes[1, 0].set_xlabel("Position")
    axes[1, 0].set_ylabel("Booking rate")
    axes[1, 0].set_title("Booking Rate by Position (bias check)")
    axes[1, 0].legend()
    
    # 7d. Position effect on clicks (normal vs random)
    pos_ctr_normal = normal.groupby("position")["click_bool"].mean().head(40)
    pos_ctr_random = random.groupby("position")["click_bool"].mean().head(40)
    axes[1, 1].plot(pos_ctr_normal.index, pos_ctr_normal.values, label="Normal sort", marker=".", markersize=3)
    axes[1, 1].plot(pos_ctr_random.index, pos_ctr_random.values, label="Random sort", marker=".", markersize=3)
    axes[1, 1].set_xlabel("Position")
    axes[1, 1].set_ylabel("Click-through rate")
    axes[1, 1].set_title("CTR by Position (bias check)")
    axes[1, 1].legend()
    
    plt.tight_layout()
    save_fig(fig, "random_bool_analysis.png", FIGURE_DIR)
    plt.close()
    
    # Training strategy recommendation
    report_lines.append(f"\n--- Training strategy implications ---")
    if norm_rates[1] > rand_rates[1] * 1.1:
        report_lines.append("Normal-sort data has higher booking rate → position bias inflates targets.")
        report_lines.append("RECOMMENDATION: Consider training on random_bool=1 only, or upweighting it.")
    else:
        report_lines.append("Booking rates similar across subsets → position bias is mild for bookings.")
        report_lines.append("RECOMMENDATION: Use all data; position bias mainly affects clicks not bookings.")
    
    report_lines.append(f"\nTraining options to test:")
    report_lines.append(f"  A. All data (max volume: {len(df):,} rows)")
    report_lines.append(f"  B. Random only (clean signal: {len(random):,} rows)")
    report_lines.append(f"  C. All data, weight random_bool=1 rows 2-3x higher")
    report_lines.append(f"  D. All data, add random_bool as feature (let model learn the distinction)")
    
    # Save report
    (OUTPUT_DIR / "random_bool_analysis.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  Saved: {OUTPUT_DIR / 'random_bool_analysis.txt'}")
    
    # ── 8. Save numeric summary ─────────────────────────────────
    print("8. Saving numeric summary...")
    summary = df.describe(include="all").T
    summary.to_csv(OUTPUT_DIR / "eda_summary.csv")
    
    # Correlation matrix
    corrs_full = df[numeric_cols].corr()
    corrs_full.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
    
    print("\n✓ EDA complete. Check figures/ and outputs/")


if __name__ == "__main__":
    main()
