"""
Central configuration for the hotel ranking pipeline.
All paths, hyperparameters, and constants live here.
Every script imports from this file — nothing is hardcoded elsewhere.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
OUTPUT_DIR  = ROOT / "outputs"
FIGURE_DIR  = ROOT / "figures"
REPORT_DIR  = ROOT / "reports"

TRAIN_RAW   = DATA_DIR / "train.csv"
TEST_RAW    = DATA_DIR / "test.csv"

TRAIN_FE    = OUTPUT_DIR / "train_fe.parquet"
TEST_FE     = OUTPUT_DIR / "test_fe.parquet"
TRAIN_SPLIT = OUTPUT_DIR / "train_split.parquet"
VAL_SPLIT   = OUTPUT_DIR / "val_split.parquet"
SUBMISSION  = OUTPUT_DIR / "submission.csv"

# ── Sampling ───────────────────────────────────────────────────────
# Work on a sample during development; set to 1.0 for final run.
SAMPLE_FRAC  = 1.0        # fraction of unique srch_ids to keep
RANDOM_STATE = 42

# ── Target / relevance ────────────────────────────────────────────
# NDCG relevance grades per competition specification
REL_BOOK  = 5
REL_CLICK = 1
REL_NONE  = 0

# ── Feature engineering ───────────────────────────────────────────
# Competitor columns (1-8)
COMP_NUMS = list(range(1, 9))
COMP_RATE_COLS = [f"comp{i}_rate" for i in COMP_NUMS]
COMP_INV_COLS  = [f"comp{i}_inv" for i in COMP_NUMS]
COMP_DIFF_COLS = [f"comp{i}_rate_percent_diff" for i in COMP_NUMS]

# Raw numeric features to keep as-is
RAW_FEATURES = [
    "prop_starrating",
    "prop_review_score",
    "prop_brand_bool",
    "prop_location_score1",
    "prop_location_score2",
    "prop_log_historical_price",
    "price_usd",
    "promotion_flag",
    "srch_length_of_stay",
    "srch_booking_window",
    "srch_adults_count",
    "srch_children_count",
    "srch_room_count",
    "srch_saturday_night_bool",
    "srch_query_affinity_score",
    "orig_destination_distance",
    "random_bool",
]

# ── Train/val split ───────────────────────────────────────────────
VAL_FRAC = 0.2   # fraction of srch_ids held out for validation

# ── LightGBM hyperparameters ──────────────────────────────────────
LGBM_PARAMS = {
    "objective":        "lambdarank",
    "metric":           "ndcg",
    "eval_at":          [5],
    "learning_rate":    0.05,
    "num_leaves":       127,
    "min_child_samples": 50,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "n_estimators":     1000,
    "early_stopping_rounds": 50,
    "verbose":          -1,
    "random_state":     RANDOM_STATE,
}

# ── Baseline (logistic regression) ────────────────────────────────
LR_MAX_ITER = 1000

# ── Bias analysis ─────────────────────────────────────────────────
# Segment travelers: families (children > 0) vs non-families
BIAS_SEGMENT_COL   = "is_family"       # created in feature engineering
BIAS_SEGMENT_LABEL = "Family vs Non-Family"
