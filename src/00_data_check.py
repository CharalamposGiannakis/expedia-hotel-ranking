"""
00 — Data check
Quick inspection of raw data: shape, dtypes, nulls, target distributions.
Run time: ~1 min on full data.
"""


import pandas as pd
from config import TRAIN_RAW, TEST_RAW, OUTPUT_DIR
from src.utils import print_header

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print_header("00 — DATA CHECK")
    
    # ── Load ──
    print("Loading train...")
    train = pd.read_csv(TRAIN_RAW)
    print(f"  Train shape: {train.shape}")
    
    print("Loading test...")
    test = pd.read_csv(TEST_RAW)
    print(f"  Test shape:  {test.shape}")
    
    report = []
    report.append(f"Train shape: {train.shape}")
    report.append(f"Test shape:  {test.shape}")
    report.append(f"Train columns: {list(train.columns)}")
    report.append(f"Test columns:  {list(test.columns)}")
    
    # ── Columns only in train ──
    train_only = set(train.columns) - set(test.columns)
    report.append(f"\nTrain-only columns: {train_only}")
    
    # ── Dtypes ──
    report.append(f"\n--- DTYPES ---\n{train.dtypes.to_string()}")
    
    # ── Nulls ──
    null_pct = (train.isnull().sum() / len(train) * 100).sort_values(ascending=False)
    report.append(f"\n--- NULL % (train) ---\n{null_pct.to_string()}")
    
    null_pct_test = (test.isnull().sum() / len(test) * 100).sort_values(ascending=False)
    report.append(f"\n--- NULL % (test) ---\n{null_pct_test.to_string()}")
    
    # ── Target distribution ──
    report.append(f"\n--- TARGET (train) ---")
    report.append(f"click_bool:   {train['click_bool'].value_counts().to_dict()}")
    report.append(f"booking_bool: {train['booking_bool'].value_counts().to_dict()}")
    report.append(f"Click rate:   {train['click_bool'].mean():.4f}")
    report.append(f"Book rate:    {train['booking_bool'].mean():.4f}")
    
    # ── Searches ──
    n_searches_train = train["srch_id"].nunique()
    n_searches_test = test["srch_id"].nunique()
    report.append(f"\nUnique searches (train): {n_searches_train:,}")
    report.append(f"Unique searches (test):  {n_searches_test:,}")
    report.append(f"Avg properties per search (train): {len(train)/n_searches_train:.1f}")
    report.append(f"Avg properties per search (test):  {len(test)/n_searches_test:.1f}")
    
    # ── Random bool distribution ──
    report.append(f"\nrandom_bool distribution (train): {train['random_bool'].value_counts().to_dict()}")
    
    # ── Write ──
    out = OUTPUT_DIR / "data_summary.txt"
    out.write_text("\n".join(report))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
