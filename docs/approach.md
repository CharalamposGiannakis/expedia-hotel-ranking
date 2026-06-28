# Pipeline Approach

> **Best configuration**: v1 + F2 features, LightGBM LambdaRank, lr=0.02, leaves=127
> **Kaggle NDCG@5: 0.40533**

---

## Pipeline diagram

```
train.csv ──→ 02_feature_engineering.py ──→ train_fe.parquet ──→ 03_split.py ──→ train_split.parquet
                                                                                  val_split.parquet
                                                                                       │
                                                                                       ▼
                                                                            02c_feature_v3_nontarget.py (F2 only)
                                                                                       │
                                                                                       ▼
                                                                            04_train_lgbm.py (best params)
                                                                                       │
                                                                                       ▼
                                                                            06_evaluate.py
                                                                                       │
test.csv ──→ 02_feature_engineering.py ──→ test_fe.parquet                              │
                       │                                                               │
                       ▼                                                               │
              02c (F2 applied to test) ──→ test_fe.parquet (updated)                    │
                       │                                                               │
                       ▼                                                               │
              07_predict.py ──→ submission.csv                                         │
                                                                                       │
                                                                            08_bias_analysis.py
                                                                                       │
                                                                                       ▼
                                                                            bias_report.json + figures
```

## Scripts in execution order

| # | Script | Purpose | Input | Output |
|---|--------|---------|-------|--------|
| 1 | `00_data_check.py` | Shape, types, nulls, targets | raw CSVs | `data_summary.txt` |
| 2 | `01_eda.py` | Distributions, position bias, random_bool analysis | raw train | `figures/`, `random_bool_analysis.txt` |
| 3 | `02_feature_engineering.py` | Build 79 v1 features | raw CSVs | `train_fe.parquet`, `test_fe.parquet` |
| 4 | `03_split.py` | Group-aware train/val split by srch_id | `train_fe.parquet` | `train_split.parquet`, `val_split.parquet` |
| 5 | `02c_feature_v3_nontarget.py --only f2` | Add 10 property profile features | split parquets + `train_fe` | updated split parquets + `test_fe` |
| 6 | `04_train_lgbm.py` | Train LightGBM LambdaRank | split parquets | `lgbm_model.txt`, importance CSV |
| 7 | `05_train_baseline.py` | Train logistic regression baseline | split parquets | `baseline_model.pkl`, coefficients |
| 8 | `06_evaluate.py` | Compare models, NDCG@k table | val predictions | `evaluation_report.txt`, comparison figure |
| 9 | `07_predict.py --model outputs/exp_t4_lr02_leaves127` | Generate submission | best model + `test_fe` | `submission.csv` |
| 10 | `08_bias_analysis.py` | Detect and mitigate fairness disparity | split parquets | `bias_report.json`, bias figures |

## Key config settings (config.py)

```python
SAMPLE_FRAC = 1.0          # Full data for final run
VAL_FRAC = 0.2             # 20% validation
RANDOM_STATE = 42           # Reproducibility

# Best model params (lr=0.02, leaves=127)
LGBM_PARAMS = {
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
}
```

## Feature count progression

| Stage | Features | Script |
|-------|---------|--------|
| Raw data | 54 columns | — |
| v1 feature engineering | 79 | `02_feature_engineering.py` |
| + F2 property profiles | 89 | `02c_feature_v3_nontarget.py --only f2` |

## Experiment history

### Full-data model experiments

| Experiment | Features | Val NDCG@5 | Verdict |
|------------|---------|-----------|---------|
| v2 (target aggregates + regularized) | 87 | 0.38535 | Rejected — overfitting |
| **v1 (no aggregates)** | **79** | **0.39426** | **Best at this stage** |
| v1 + random_bool weighting | 79 | 0.39078 | Rejected — hurt performance |

### Feature block experiments

| Experiment | Features | Val NDCG@5 | Delta vs v1 | Verdict |
|------------|---------|-----------|------------|---------|
| F1 (price norms) | 85 | 0.39436 | +0.00010 | Rejected — no gain |
| **F2 (property profiles)** | **89** | **0.40576** | **+0.01150** | **Winner** |
| F4 (F1 + F2) | 95 | 0.40527 | +0.01101 | Rejected — F1 adds noise |

### Hyperparameter tuning

| Config | lr | leaves | Val NDCG@5 | Gap | Verdict |
|--------|-----|--------|-----------|-----|---------|
| F2 base | 0.05 | 127 | 0.40576 | 0.103 | Starting point |
| Config 1 | 0.03 | 63 | 0.40677 | 0.070 | Good gap, less val |
| Config 2 | 0.02 | 63 | 0.40666 | 0.061 | Underfits slightly |
| Config 3 | 0.02 | 31 | 0.40511 | 0.034 | Underfits clearly |
| **Best** | **0.02** | **127** | **0.40766** | **0.095** | **Winner** |
| Config 5 | 0.02 | 63 | 0.40583 | 0.045 | Subsampling hurts |

## Leakage prevention checklist

- [x] `position` excluded from features (train-only column)
- [x] `gross_bookings_usd` excluded (both spellings in drop set)
- [x] `click_bool`, `booking_bool` excluded (targets)
- [x] Train/val split by `srch_id` groups (no within-search leakage)
- [x] F2 property profiles computed from training split only, applied to val/test
- [x] `price_vs_historical` set to NaN when `prop_log_historical_price=0`
- [x] v2 target aggregates tested and rejected (overfitting)
