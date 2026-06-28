"""
04 — LightGBM LambdaRank
Primary model. Directly optimizes NDCG using listwise learning-to-rank.

Why LambdaRank:
  - The task is ranking, not classification. Pointwise models miss inter-item relationships.
  - LambdaRank computes gradients based on NDCG swap deltas, directly optimizing what we're scored on.
  - LightGBM's native lambdarank handles query groups natively.

Run time: ~10-15 min on full data.
"""


import pandas as pd
import numpy as np
import lightgbm as lgb
import json

from config import (
    TRAIN_SPLIT, VAL_SPLIT, OUTPUT_DIR, FIGURE_DIR,
    LGBM_PARAMS, RANDOM_STATE
)
from src.utils import print_header, eval_ndcg

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print_header("04 — LightGBM LambdaRank")
    
    # ── Load ──
    train = pd.read_parquet(TRAIN_SPLIT)
    val = pd.read_parquet(VAL_SPLIT)
    
    # ── Feature columns ──
    def get_feature_columns(df):
        drop = {
            "srch_id", "date_time", "position", "click_bool", "booking_bool",
            "gross_bookings_usd", "gross_booking_usd",
            "relevance", "prop_id",
        }
        for i in range(1, 9):
            drop.update({f"comp{i}_rate", f"comp{i}_inv", f"comp{i}_rate_percent_diff"})
        return [c for c in df.columns if c not in drop]
    
    feature_cols = get_feature_columns(train)
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train: {len(train):,} rows")
    print(f"  Val:   {len(val):,} rows")
    
    # ── Sort by srch_id first (required for LightGBM ranking) ──
    train = train.sort_values("srch_id").reset_index(drop=True)
    val = val.sort_values("srch_id").reset_index(drop=True)
    
    # ── Build query groups (number of items per search, after sorting) ──
    train_groups = train.groupby("srch_id", sort=False).size().values
    val_groups = val.groupby("srch_id", sort=False).size().values
    
    # ── Create datasets ──
    X_train = train[feature_cols]
    y_train = train["relevance"]
    X_val = val[feature_cols]
    y_val = val["relevance"]
    
    train_ds = lgb.Dataset(X_train, label=y_train, group=train_groups, free_raw_data=False)
    val_ds = lgb.Dataset(X_val, label=y_val, group=val_groups, reference=train_ds, free_raw_data=False)
    
    # ── Train ──
    print("\nTraining...")
    callbacks = [
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=LGBM_PARAMS.get("early_stopping_rounds", 50)),
    ]
    
    params = {k: v for k, v in LGBM_PARAMS.items() 
              if k not in ["n_estimators", "early_stopping_rounds"]}
    
    model = lgb.train(
        params,
        train_ds,
        num_boost_round=LGBM_PARAMS["n_estimators"],
        valid_sets=[train_ds, val_ds],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    
    # ── Save model ──
    model_path = OUTPUT_DIR / "lgbm_model.txt"
    model.save_model(str(model_path))
    print(f"\n  Model saved: {model_path}")
    print(f"  Best iteration: {model.best_iteration}")
    
    # ── Evaluate ──
    val["lgbm_score"] = model.predict(X_val)
    ndcg5 = eval_ndcg(val, "lgbm_score", k=5)
    print(f"\n  Validation NDCG@5: {ndcg5:.5f}")
    
    # Save val predictions for comparison
    val[["srch_id", "prop_id", "relevance", "lgbm_score"]].to_csv(
        OUTPUT_DIR / "lgbm_val_results.csv", index=False
    )
    
    # ── Feature importance ──
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    
    importance.to_csv(OUTPUT_DIR / "lgbm_feature_importance.csv", index=False)
    
    fig, ax = plt.subplots(figsize=(8, max(6, len(feature_cols) * 0.2)))
    top_n = importance.head(30)
    ax.barh(range(len(top_n)), top_n["importance"].values)
    ax.set_yticks(range(len(top_n)))
    ax.set_yticklabels(top_n["feature"].values, fontsize=8)
    ax.set_title("LightGBM Feature Importance (top 30, gain)")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "lgbm_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # ── Save results summary ──
    results = {
        "model": "LightGBM LambdaRank",
        "n_features": len(feature_cols),
        "best_iteration": model.best_iteration,
        "val_ndcg5": round(ndcg5, 5),
        "train_rows": len(train),
        "val_rows": len(val),
        "top_10_features": importance.head(10)["feature"].tolist(),
    }
    with open(OUTPUT_DIR / "lgbm_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ LightGBM training complete.")


if __name__ == "__main__":
    main()
