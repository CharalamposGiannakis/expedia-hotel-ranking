"""
05 — Logistic Regression Baseline
Pointwise approach: predict P(booking) for each property, rank by probability.

Why this as second model:
  - Serves as interpretable baseline to measure LambdaRank improvement.
  - Fast to train, easy to interpret.
  - If LGBMs advantage over LR is small, features matter more than model — good insight.

Run time: ~2 min.
"""


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import json
import joblib

from config import TRAIN_SPLIT, VAL_SPLIT, OUTPUT_DIR, LR_MAX_ITER, RANDOM_STATE
from src.utils import print_header, eval_ndcg

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_feature_columns(df):
    """Same exclusion logic as feature engineering."""
    drop = {
        "srch_id", "date_time", "position", "click_bool", "booking_bool",
        "gross_bookings_usd", "gross_booking_usd",
        "relevance", "prop_id",
    }
    for i in range(1, 9):
        drop.update({f"comp{i}_rate", f"comp{i}_inv", f"comp{i}_rate_percent_diff"})
    return [c for c in df.columns if c not in drop]


def main():
    print_header("05 — LOGISTIC REGRESSION BASELINE")
    
    train = pd.read_parquet(TRAIN_SPLIT)
    val = pd.read_parquet(VAL_SPLIT)
    
    feature_cols = get_feature_columns(train)
    
    # Binary target: booked or not
    y_train = train["booking_bool"]
    y_val = val["booking_bool"]
    
    X_train = train[feature_cols].fillna(-999)
    X_val = val[feature_cols].fillna(-999)
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train: {len(train):,} rows ({y_train.sum():,} bookings)")
    print(f"  Val:   {len(val):,} rows ({y_val.sum():,} bookings)")
    
    # ── Train ──
    print("\nTraining logistic regression...")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=LR_MAX_ITER,
            random_state=RANDOM_STATE,
            class_weight="balanced",   # handle class imbalance
            solver="lbfgs",
        ))
    ])
    model.fit(X_train, y_train)
    
    # ── Predict booking probability ──
    val["lr_score"] = model.predict_proba(X_val)[:, 1]
    
    # Need relevance column for NDCG
    from src.utils import make_relevance
    if "relevance" not in val.columns:
        val = make_relevance(val)
    
    ndcg5 = eval_ndcg(val, "lr_score", k=5)
    print(f"\n  Validation NDCG@5: {ndcg5:.5f}")
    
    # ── Save ──
    joblib.dump(model, OUTPUT_DIR / "baseline_model.pkl")
    
    val[["srch_id", "prop_id", "relevance", "lr_score"]].to_csv(
        OUTPUT_DIR / "baseline_val_results.csv", index=False
    )
    
    # Feature coefficients
    lr = model.named_steps["lr"]
    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": lr.coef_[0],
    }).sort_values("coefficient", ascending=False)
    coef_df.to_csv(OUTPUT_DIR / "baseline_coefficients.csv", index=False)
    
    results = {
        "model": "Logistic Regression (balanced)",
        "n_features": len(feature_cols),
        "val_ndcg5": round(ndcg5, 5),
        "top_5_positive": coef_df.head(5)["feature"].tolist(),
        "top_5_negative": coef_df.tail(5)["feature"].tolist(),
    }
    with open(OUTPUT_DIR / "baseline_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Baseline training complete.")


if __name__ == "__main__":
    main()
