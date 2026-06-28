"""
07 — Predict (Kaggle Submission)
Generate the final ranking file for Kaggle upload.
Format: srch_id, prop_id (sorted by predicted relevance descending within each search).

Usage:
  python src/07_predict.py                                          # default model
  python src/07_predict.py --model outputs/exp_t4_lr02_leaves127    # specific experiment

Run time: ~2 min.
"""


import argparse
import pandas as pd
import lightgbm as lgb

from config import TEST_FE, OUTPUT_DIR, SUBMISSION
from src.utils import print_header

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_feature_columns(df):
    """Same exclusion logic."""
    drop = {
        "srch_id", "date_time", "position", "click_bool", "booking_bool",
        "gross_bookings_usd", "gross_booking_usd",
        "relevance", "prop_id",
    }
    for i in range(1, 9):
        drop.update({f"comp{i}_rate", f"comp{i}_inv", f"comp{i}_rate_percent_diff"})
    return [c for c in df.columns if c not in drop]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Path to experiment directory containing lgbm_model.txt "
                             "(e.g. outputs/exp_t4_lr02_leaves127)")
    args = parser.parse_args()

    print_header("07 — GENERATE SUBMISSION")
    
    # ── Load model ──
    if args.model:
        from pathlib import Path
        model_dir = Path(args.model)
        model_path = model_dir / "lgbm_model.txt"
        print(f"  Model source: {model_path}")
    else:
        model_path = OUTPUT_DIR / "lgbm_model.txt"
        print(f"  Model source: {model_path} (default)")
    
    model = lgb.Booster(model_file=str(model_path))
    test = pd.read_parquet(TEST_FE)
    
    # Use the model's own feature names to avoid mismatch from dirty test_fe
    feature_cols = model.feature_name()
    missing = [c for c in feature_cols if c not in test.columns]
    if missing:
        raise ValueError(f"Test data missing features the model expects: {missing}")
    
    print(f"  Test rows: {len(test):,}")
    print(f"  Test searches: {test['srch_id'].nunique():,}")
    print(f"  Features: {len(feature_cols)} (from model)")
    
    # ── Predict ──
    test["score"] = model.predict(test[feature_cols])
    
    # ── Rank within each search (descending by score) ──
    test = test.sort_values(["srch_id", "score"], ascending=[True, False])
    
    # ── Format submission ──
    submission = test[["srch_id", "prop_id"]]
    
    submission.to_csv(SUBMISSION, index=False)
    print(f"\n  Submission saved: {SUBMISSION}")
    print(f"  Rows: {len(submission):,}")
    print(f"  Unique searches: {submission['srch_id'].nunique():,}")
    
    # Sanity check
    props_per_search = submission.groupby("srch_id").size()
    print(f"  Props per search: min={props_per_search.min()}, "
          f"max={props_per_search.max()}, mean={props_per_search.mean():.1f}")
    
    print("\n✓ Submission ready for Kaggle upload.")


if __name__ == "__main__":
    main()
