"""
03 — Train/Validation Split
Split by srch_id groups to avoid data leakage.
Properties from the same search must stay together.

Run time: ~1 min.
"""


import pandas as pd
import numpy as np

from config import TRAIN_FE, TRAIN_SPLIT, VAL_SPLIT, VAL_FRAC, RANDOM_STATE, OUTPUT_DIR
from src.utils import print_header

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print_header("03 — TRAIN/VAL SPLIT")
    
    df = pd.read_parquet(TRAIN_FE)
    print(f"  Loaded {len(df):,} rows, {df['srch_id'].nunique():,} searches")

    # Downcast floats to save memory (optional but can speed up training)
    float_cols = df.select_dtypes(include="float64").columns
    df[float_cols] = df[float_cols].astype("float32")

    # Group-level split
    srch_ids = df["srch_id"].unique()
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(srch_ids)
    
    n_val = int(len(srch_ids) * VAL_FRAC)
    val_ids = set(srch_ids[:n_val])
    train_ids = set(srch_ids[n_val:])
    
    train = df[df["srch_id"].isin(train_ids)].reset_index(drop=True)
    val = df[df["srch_id"].isin(val_ids)].reset_index(drop=True)
    
    print(f"  Train: {len(train):,} rows, {len(train_ids):,} searches")
    print(f"  Val:   {len(val):,} rows, {len(val_ids):,} searches")
    
    # Verify no leakage
    overlap = train_ids & val_ids
    assert len(overlap) == 0, f"Leakage! {len(overlap)} shared srch_ids"
    print("  ✓ No search ID leakage")
    
    # Check target rates are similar
    print(f"  Train book rate: {train['booking_bool'].mean():.4f}")
    print(f"  Val book rate:   {val['booking_bool'].mean():.4f}")
    
    train.to_parquet(TRAIN_SPLIT, index=False)
    val.to_parquet(VAL_SPLIT, index=False)
    print(f"\n  Saved: {TRAIN_SPLIT}")
    print(f"  Saved: {VAL_SPLIT}")
    
    print("\n✓ Split complete.")


if __name__ == "__main__":
    main()
