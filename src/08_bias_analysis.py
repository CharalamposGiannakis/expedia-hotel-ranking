"""
08 — Bias Analysis and Mitigation

Approach:
  1. DETECT: Measure NDCG@5 across traveler segments (family vs non-family)
     using a baseline model with uniform weights.
  2. MITIGATE: Pre-processing — retrain LightGBM with higher sample weights
     for the underperforming segment so the model learns their patterns better.
  3. EVALUATE: Compare per-segment NDCG@5 before and after mitigation.

Why training-time reweighting (not post-processing score boost):
  Family/non-family is a search-level attribute — all hotels in the same search
  share the same segment. A uniform post-processing score boost would shift all
  hotels in a family search equally, leaving the within-search ranking unchanged.
  NDCG is a within-query metric, so post-processing boosts are a no-op.
  Reweighting during training actually changes how the model splits and learns,
  producing different within-search rankings for the disadvantaged segment.

Prerequisites:
  - train_split.parquet and val_split.parquet with F2 features
  - Best model params (T4) defined in this script

Usage:
  python src/08_bias_analysis.py
  python src/08_bias_analysis.py --weights 1.5 2.0 3.0 5.0

Run time: ~15-20 min (trains multiple models).
"""


import argparse
import json
import pandas as pd
import numpy as np
import lightgbm as lgb

from config import TRAIN_SPLIT, VAL_SPLIT, OUTPUT_DIR, FIGURE_DIR, RANDOM_STATE
from src.utils import print_header, eval_ndcg, ndcg_at_k

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ── Best model params (T4) ────────────────────────────────────────
BEST_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [5],
    "learning_rate": 0.02,
    "num_leaves": 127,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 3000,
    "early_stopping_rounds": 150,
    "verbose": -1,
    "random_state": RANDOM_STATE,
}

DEFAULT_WEIGHTS = [1.5, 2.0, 3.0, 5.0]


def get_feature_columns(df):
    drop = {
        "srch_id", "date_time", "position", "click_bool", "booking_bool",
        "gross_bookings_usd", "gross_booking_usd", "relevance", "prop_id",
        "_sample_weight", "segment",
    }
    for i in range(1, 9):
        drop.update({f"comp{i}_rate", f"comp{i}_inv", f"comp{i}_rate_percent_diff"})
    return [c for c in df.columns if c not in drop]


def segment_ndcg(df, score_col, segment_col, k=5):
    results = {}
    for seg, group in df.groupby(segment_col):
        results[seg] = eval_ndcg(group, score_col, k=k)
    return results


def segment_booking_recall_at_k(df, score_col, segment_col, k=5):
    results = {}
    for seg_name, seg_data in df.groupby(segment_col):
        bookings = seg_data[seg_data["relevance"] == 5]
        if len(bookings) == 0:
            results[seg_name] = float("nan")
            continue
        recalled = 0
        for srch_id in bookings["srch_id"].unique():
            search = seg_data[seg_data["srch_id"] == srch_id]
            top_k_props = search.nlargest(k, score_col)["prop_id"].values
            booked_props = bookings[bookings["srch_id"] == srch_id]["prop_id"].values
            if any(b in top_k_props for b in booked_props):
                recalled += 1
        results[seg_name] = recalled / bookings["srch_id"].nunique()
    return results


def train_weighted_model(train, val, feature_cols, sample_weight):
    train = train.sort_values("srch_id").reset_index(drop=True)
    val = val.sort_values("srch_id").reset_index(drop=True)
    sample_weight = sample_weight[train.index] if hasattr(sample_weight, 'iloc') else sample_weight

    train_groups = train.groupby("srch_id", sort=False).size().values
    val_groups = val.groupby("srch_id", sort=False).size().values

    train_ds = lgb.Dataset(
        train[feature_cols], label=train["relevance"],
        group=train_groups, weight=sample_weight, free_raw_data=False
    )
    val_ds = lgb.Dataset(
        val[feature_cols], label=val["relevance"],
        group=val_groups, reference=train_ds, free_raw_data=False
    )

    params = {k: v for k, v in BEST_PARAMS.items()
              if k not in ["n_estimators", "early_stopping_rounds"]}

    callbacks = [
        lgb.log_evaluation(period=500),
        lgb.early_stopping(stopping_rounds=BEST_PARAMS["early_stopping_rounds"]),
    ]

    model = lgb.train(
        params, train_ds,
        num_boost_round=BEST_PARAMS["n_estimators"],
        valid_sets=[val_ds],
        valid_names=["val"],
        callbacks=callbacks,
    )

    val["score"] = model.predict(val[feature_cols], num_iteration=model.best_iteration)
    return model, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="*", type=float, default=DEFAULT_WEIGHTS,
                        help="Weight multipliers to test for disadvantaged segment")
    args = parser.parse_args()

    print_header("08 — BIAS ANALYSIS")

    train = pd.read_parquet(TRAIN_SPLIT)
    val = pd.read_parquet(VAL_SPLIT)
    feature_cols = get_feature_columns(train)

    train["segment"] = np.where(train["srch_children_count"] > 0, "Family", "Non-Family")
    val["segment"] = np.where(val["srch_children_count"] > 0, "Family", "Non-Family")

    print(f"  Train: {len(train):,} rows, {train['srch_id'].nunique():,} searches")
    print(f"  Val:   {len(val):,} rows, {val['srch_id'].nunique():,} searches")
    print(f"  Features: {len(feature_cols)}")

    for seg in ["Family", "Non-Family"]:
        t_count = train[train["segment"] == seg]["srch_id"].nunique()
        v_count = val[val["segment"] == seg]["srch_id"].nunique()
        print(f"  {seg}: {t_count:,} train / {v_count:,} val searches")

    # ── 1. DETECTION ──────────────────────────────────────────
    print_header("1. BIAS DETECTION (baseline, uniform weights)")

    uniform_weight = np.ones(len(train))
    baseline_model, val_baseline = train_weighted_model(
        train.copy(), val.copy(), feature_cols, uniform_weight
    )

    overall_baseline = eval_ndcg(val_baseline, "score", k=5)
    seg_baseline = segment_ndcg(val_baseline, "score", "segment", k=5)
    recall_baseline = segment_booking_recall_at_k(val_baseline, "score", "segment", k=5)

    print(f"\n  Overall NDCG@5: {overall_baseline:.5f}")
    for seg in ["Family", "Non-Family"]:
        print(f"  {seg}: NDCG@5 = {seg_baseline.get(seg, 0):.5f}, "
              f"Booking recall@5 = {recall_baseline.get(seg, 0):.4f}")

    gap_baseline = abs(seg_baseline.get("Family", 0) - seg_baseline.get("Non-Family", 0))
    disadvantaged = "Family" if seg_baseline.get("Family", 0) < seg_baseline.get("Non-Family", 0) else "Non-Family"
    disparity_ratio = seg_baseline.get("Family", 0) / max(seg_baseline.get("Non-Family", 0), 1e-10)

    print(f"\n  Disparity ratio (Family / Non-Family): {disparity_ratio:.4f}")
    print(f"  Gap: {gap_baseline:.5f}")
    print(f"  Disadvantaged segment: {disadvantaged}")

    # ── 2. MITIGATION ─────────────────────────────────────────
    print_header("2. BIAS MITIGATION (training-time reweighting)")

    is_disadvantaged = (train["segment"] == disadvantaged).values
    weight_results = []

    for w in args.weights:
        print(f"\n  Testing weight = {w}x for {disadvantaged}...")
        sample_weight = np.where(is_disadvantaged, w, 1.0)

        _, val_w = train_weighted_model(
            train.copy(), val.copy(), feature_cols, sample_weight
        )

        overall_w = eval_ndcg(val_w, "score", k=5)
        seg_w = segment_ndcg(val_w, "score", "segment", k=5)
        recall_w = segment_booking_recall_at_k(val_w, "score", "segment", k=5)
        gap_w = abs(seg_w.get("Family", 0) - seg_w.get("Non-Family", 0))

        weight_results.append({
            "weight": w,
            "overall_ndcg5": round(overall_w, 5),
            "family_ndcg5": round(seg_w.get("Family", 0), 5),
            "nonfamily_ndcg5": round(seg_w.get("Non-Family", 0), 5),
            "gap": round(gap_w, 5),
            "family_recall5": round(recall_w.get("Family", 0), 4),
            "nonfamily_recall5": round(recall_w.get("Non-Family", 0), 4),
        })

        print(f"    Overall: {overall_w:.5f}  Family: {seg_w.get('Family', 0):.5f}  "
              f"Non-Family: {seg_w.get('Non-Family', 0):.5f}  Gap: {gap_w:.5f}")

    # Select best: smallest gap with acceptable overall performance
    threshold = overall_baseline * 0.98
    viable = [r for r in weight_results if r["overall_ndcg5"] >= threshold]

    if viable:
        best = min(viable, key=lambda r: r["gap"])
    else:
        print("\n  WARNING: No weight met the 98% threshold. Using smallest gap.")
        best = min(weight_results, key=lambda r: r["gap"])

    print(f"\n  Best weight: {best['weight']}x")
    print(f"  Constraint: overall NDCG@5 >= {threshold:.5f} (98% of baseline)")

    # ── 3. EVALUATION ─────────────────────────────────────────
    print_header("3. MITIGATION EVALUATION")

    rows = [
        ("Overall NDCG@5", overall_baseline, best["overall_ndcg5"]),
        ("Family NDCG@5", seg_baseline.get("Family", 0), best["family_ndcg5"]),
        ("Non-Family NDCG@5", seg_baseline.get("Non-Family", 0), best["nonfamily_ndcg5"]),
        ("Gap", gap_baseline, best["gap"]),
    ]

    print(f"  {'':25s} {'Before':>10s} {'After':>10s} {'Change':>10s}")
    print(f"  {'':25s} {'':>10s} {'':>10s} {'':>10s}")
    for label, before, after in rows:
        change = after - before
        print(f"  {label:25s} {before:10.5f} {after:10.5f} {change:+10.5f}")

    gap_reduction = (1 - best["gap"] / gap_baseline) * 100 if gap_baseline > 0 else 0
    print(f"\n  Gap reduction: {gap_reduction:.1f}%")

    # ── 4. VISUALIZATION ──────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    segments = ["Family", "Non-Family"]
    before_vals = [seg_baseline.get(s, 0) for s in segments]
    after_vals = [best["family_ndcg5"], best["nonfamily_ndcg5"]]
    x = np.arange(len(segments))
    axes[0, 0].bar(x - 0.15, before_vals, 0.3, label="Before (uniform)", color="coral")
    axes[0, 0].bar(x + 0.15, after_vals, 0.3, label=f"After (weight={best['weight']}x)", color="steelblue")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(segments)
    axes[0, 0].set_ylabel("NDCG@5")
    axes[0, 0].set_title("Per-Segment NDCG@5: Before vs After Mitigation")
    axes[0, 0].legend()

    axes[0, 1].bar(["Before", "After"], [gap_baseline, best["gap"]],
                    color=["coral", "steelblue"])
    axes[0, 1].set_ylabel("NDCG@5 Gap")
    axes[0, 1].set_title("Fairness Gap: Before vs After")

    weights = [r["weight"] for r in weight_results]
    overalls = [r["overall_ndcg5"] for r in weight_results]
    gaps = [r["gap"] for r in weight_results]

    axes[1, 0].plot(weights, overalls, "o-", color="steelblue", label="Overall NDCG@5")
    axes[1, 0].axhline(overall_baseline, color="gray", linestyle="--", label="Baseline")
    axes[1, 0].axhline(threshold, color="red", linestyle=":", label="98% threshold")
    axes[1, 0].set_xlabel(f"Weight for {disadvantaged}")
    axes[1, 0].set_ylabel("Overall NDCG@5")
    axes[1, 0].set_title("Weight vs Overall Performance")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(weights, gaps, "o-", color="coral", label="NDCG@5 gap")
    axes[1, 1].axhline(gap_baseline, color="gray", linestyle="--", label="Baseline gap")
    axes[1, 1].set_xlabel(f"Weight for {disadvantaged}")
    axes[1, 1].set_ylabel("NDCG@5 Gap (|Family - Non-Family|)")
    axes[1, 1].set_title("Weight vs Fairness Gap")
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "bias_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {FIGURE_DIR / 'bias_analysis.png'}")

    # ── 5. SAVE REPORT ────────────────────────────────────────
    report = {
        "segment_definition": "Family (srch_children_count > 0) vs Non-Family",
        "detection": {
            "overall_ndcg5": round(overall_baseline, 5),
            "per_segment_ndcg5": {k: round(v, 5) for k, v in seg_baseline.items()},
            "per_segment_recall5": {k: round(v, 4) for k, v in recall_baseline.items()},
            "disparity_ratio": round(disparity_ratio, 4),
            "gap": round(gap_baseline, 5),
            "disadvantaged_segment": disadvantaged,
        },
        "mitigation": {
            "method": "Pre-processing: training-time sample reweighting",
            "description": (
                f"Rows belonging to {disadvantaged} searches were assigned a weight of "
                f"{best['weight']}x during LightGBM training, while other rows kept weight 1.0. "
                f"This increases the model's attention to the underperforming segment."
            ),
            "constraint": "Overall NDCG@5 must remain within 98% of baseline",
            "weights_tested": [r["weight"] for r in weight_results],
            "best_weight": best["weight"],
        },
        "evaluation": {
            "before": {
                "overall_ndcg5": round(overall_baseline, 5),
                "family_ndcg5": round(seg_baseline.get("Family", 0), 5),
                "nonfamily_ndcg5": round(seg_baseline.get("Non-Family", 0), 5),
                "gap": round(gap_baseline, 5),
            },
            "after": best,
            "gap_reduction_pct": round(gap_reduction, 1),
        },
        "weight_search_results": weight_results,
    }

    with open(OUTPUT_DIR / "bias_report.json", "w") as f:
        json.dump(report, f, indent=2)

    text_lines = [
        "BIAS ANALYSIS REPORT",
        "=" * 50,
        "",
        f"Segment: {report['segment_definition']}",
        f"Disadvantaged: {disadvantaged}",
        "",
        "--- DETECTION ---",
        f"Overall NDCG@5: {overall_baseline:.5f}",
    ]
    for seg in segments:
        text_lines.append(f"{seg}: NDCG@5={seg_baseline.get(seg,0):.5f}, "
                          f"Recall@5={recall_baseline.get(seg,0):.4f}")
    text_lines.extend([
        f"Disparity ratio: {disparity_ratio:.4f}",
        f"Gap: {gap_baseline:.5f}",
        "",
        "--- MITIGATION ---",
        f"Method: Training-time sample reweighting",
        f"Best weight: {best['weight']}x for {disadvantaged}",
        f"Constraint: overall >= {threshold:.5f} (98% of baseline)",
        "",
        "--- EVALUATION ---",
    ])
    for label, before, after in rows:
        change = after - before
        text_lines.append(f"{label:25s} {before:10.5f} {after:10.5f} {change:+10.5f}")
    text_lines.append(f"\nGap reduction: {gap_reduction:.1f}%")

    (OUTPUT_DIR / "bias_report.txt").write_text("\n".join(text_lines), encoding="utf-8")

    print(f"  Saved: {OUTPUT_DIR / 'bias_report.json'}")
    print(f"  Saved: {OUTPUT_DIR / 'bias_report.txt'}")
    print("\n  Bias analysis complete.")


if __name__ == "__main__":
    main()
