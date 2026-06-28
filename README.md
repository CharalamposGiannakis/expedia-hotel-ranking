# Expedia Hotel Search Ranking

A learning-to-rank system that predicts which hotel a user will book from Expedia search results. Built on ~10 million real search impressions from the [ICDM 2013 Personalized Sort challenge](https://www.kaggle.com/c/expedia-personalized-sort) dataset. LightGBM LambdaRank with 89 engineered features, evaluated at **NDCG@5 = 0.408**.

| Metric | Score |
|--------|-------|
| Validation NDCG@5 | 0.4077 |
| Held-out test NDCG@5 | 0.4053 |
| Baseline (logistic regression) | 0.3497 |
| Improvement over baseline | +16.6% |

## The problem

Each row in the dataset is a hotel shown to a user as part of a search. A single search produces ~25 hotel impressions. The task: rank these hotels so the one the user actually books appears at the top. Relevance grades are 5 (booked), 1 (clicked), and 0 (ignored), scored via NDCG@5 averaged across all queries.

Two properties make this harder than a standard classification problem. First, position bias: under Expedia's normal sort, hotels shown higher get clicked more *regardless of quality*. The dataset includes a `random_bool` flag marking searches where results were shuffled randomly — these unbiased observations anchor the feature analysis. Second, the data is extremely sparse: 97% of impressions are ignored, visitor purchase history exists for only ~5% of rows, and competitor pricing columns are 55–98% null.

## Approach

The core idea is that users compare hotels *within* their search results, not against some absolute standard. "Is this the cheapest option?" matters more than "is this $150/night." Most of the feature engineering effort went into capturing these within-search relative signals.

### Feature engineering (54 raw → 89 features)

The 89 features fall into nine categories. Within-search normalization turned out to be the most impactful — 7 of the top 10 features by model importance are engineered relative features, not raw columns.

| Category | Count | Examples | Rationale |
|----------|-------|----------|-----------|
| Raw pass-through | 17 | Star rating, review score, price, promotion flag | Baseline property and search attributes |
| Missing-value flags | 9 | `visitor_hist_starrating_missing`, `prop_review_score_zero` | Missingness is informative: no visitor history = new customer, star rating of 0 = unknown |
| Within-search ranks | 10 | `price_rank`, `review_rank`, `loc2_rank`, `is_cheapest` | Relative position among competing hotels in the same search |
| Within-search z-scores | 5 | `price_usd_zscore`, `prop_starrating_zscore` | How far each property deviates from the search-level mean |
| Gap-to-best | 4 | `price_gap_to_cheapest`, `review_gap_to_best` | Distance to the best option — captures "how much worse" a hotel is |
| Price features | 5 | `log_price`, `price_per_person`, `price_vs_historical` | Price normalization and value signals |
| Visitor–property match | 4 | `star_gap`, `price_ratio_to_hist`, `has_visitor_history` | Content-based filtering: does this hotel match the user's past preferences? |
| Competitor aggregation | 7 | `comp_net_advantage`, `comp_out_of_stock` | Compressed 24 sparse competitor columns (55–98% null) into usable signals |
| Property profiles | 10 | `prop_avg_price`, `prop_promotion_rate`, `price_vs_prop_median` | Hotel-level statistics computed from training data only — captures "is this hotel running a deal?" |
| Search context | 8 | `is_family`, `is_domestic`, `search_hour` | Derived traveler intent signals |

Property profiles (the "F2" block) deserve a note: these are aggregate statistics about each hotel computed *without* using click/booking targets. They capture signals like "this hotel is priced 20% below its own median" — a deal indicator that doesn't leak the outcome. These 10 features alone improved NDCG@5 by +0.012 over the base feature set.

### Model

**Primary — LightGBM LambdaRank.** The task is ranking, not classification. LambdaRank computes gradients based on NDCG swap deltas, so it directly optimizes the evaluation metric. It also handles missing values natively (no imputation needed, which matters when 95% of visitor history is null).

**Baseline — logistic regression.** Predicts P(booking) and ranks by probability. Serves as a sanity check and demonstrates the gap between pointwise classification and listwise ranking: the LambdaRank model outperforms it by 16.6%.

Final hyperparameters were found through a five-configuration grid search over learning rate and tree complexity. Lower learning rate (0.02) with patient early stopping consistently beat faster convergence, while 127 leaves was the right complexity for 4M training rows.

### Fairness analysis

The model was audited for bias between family travelers (`srch_children_count > 0`) and non-family travelers. Detection measured per-segment NDCG@5; mitigation used training-time sample reweighting for the disadvantaged segment.

A subtle point drove the mitigation design: family/non-family is a *search-level* attribute, so all hotels in the same search share the same segment. Post-processing score adjustments shift all hotels equally and leave the within-search ranking unchanged — they're a no-op for NDCG, which is a within-query metric. Reweighting during training actually changes how the model splits, producing different rankings for family searches.

## Experiment progression

The model went through 11 systematic experiments. This table shows the key milestones.

| Stage | Val NDCG@5 | What changed |
|-------|-----------|-------------|
| First working model (10% sample) | 0.3688 | v1 features, default LightGBM params |
| Full training data | 0.3943 | Same features, 10× more data |
| + Property profiles (F2) | 0.4058 | +0.012 from non-target hotel statistics |
| + Tuned hyperparameters | 0.4077 | lr=0.02, patience=150, leaves=127 |
| Held-out test evaluation | 0.4053 | Separate test set, not seen during training |

**What didn't work** (and why it matters that we tried):

- **Target-encoded property aggregates** — hotel booking rate, click rate, mean relevance with leave-one-out encoding and Bayesian smoothing. Caused severe overfitting (train NDCG 0.95+) even on full data. Dropped entirely.
- **Destination-normalized pricing** — redundant with within-search price features, which already capture relative value at a finer granularity. Net gain: +0.0001 NDCG.
- **Upweighting random_bool=1 rows** — the "unbiased" observations. Hurt overall performance (0.3908 vs 0.3943). The model apparently learns useful patterns from position-biased data too.
- **Heavy regularization** (31 leaves, row subsampling) — consistently underfit. At 4M rows, the model had headroom for 127-leaf trees.

## Project structure

```
expedia-hotel-ranking/
├── config.py                           # All paths, hyperparameters, feature lists
├── run_pipeline.sh                     # End-to-end pipeline runner
├── pyproject.toml
├── src/
│   ├── 00_data_check.py                # Dataset shape, types, null audit
│   ├── 01_eda.py                       # Distributions, position bias, random_bool analysis
│   ├── 02_feature_engineering.py       # 79 base features (categories A–H)
│   ├── 02b_feature_v2_aggregates.py    # Target-encoded aggregates (rejected)
│   ├── 02c_feature_v3_nontarget.py     # Property profiles (F2, +10 features)
│   ├── 03_split.py                     # Group-aware train/val split by srch_id
│   ├── 04_train_lgbm.py                # LightGBM LambdaRank training
│   ├── 05_train_baseline.py            # Logistic regression baseline
│   ├── 06_evaluate.py                  # Model comparison, NDCG@k tables
│   ├── 07_predict.py                   # Generate test set predictions
│   ├── 08_bias_analysis.py             # Fairness detection + mitigation
│   └── utils.py                        # NDCG computation, data loading, plotting
├── figures/                            # EDA and evaluation plots
├── docs/
│   ├── approach.md                     # Design decisions and experiment log
│   └── decisions.md                    # Frozen design choices
├── data/                               # Raw CSVs (not tracked, see below)
└── outputs/                            # Generated artifacts (not tracked)
```

## Quick start

```bash
git clone https://github.com/CharalamposGiannakis/expedia-hotel-ranking.git
cd expedia-hotel-ranking

pip install -r requirements.txt

# Place train.csv and test.csv in data/
# Then run the full pipeline:
bash run_pipeline.sh
```

The pipeline expects `data/train.csv` and `data/test.csv` from the [Kaggle competition page](https://www.kaggle.com/c/expedia-personalized-sort/data). All intermediate artifacts (parquet files, models, predictions) are written to `outputs/` and regenerated on each run.

**Requirements:** Python ≥ 3.10, LightGBM, pandas, numpy, scikit-learn, matplotlib. See `pyproject.toml` for the full dependency list.

## Key findings

Three insights that shaped the final approach:

**Within-search normalization is the single most impactful feature category.** The top 10 features by model importance are dominated by ranks, z-scores, and gap-to-best metrics — not raw property attributes. This makes sense: a 4-star hotel means something different when the other results are all 5-star versus all 3-star.

**Position bias is real but asymmetric.** Under normal sort, booking rate is 3.73%; under random sort, it drops to 0.51% — a 7× difference. But the bias primarily affects click behavior, not booking decisions. Users click what's shown first, but they book what fits their needs. This is why excluding `position` from features (it's train-only anyway) doesn't cripple the model.

**Missingness carries signal.** 95% of visitor history fields are null, and competitor columns are 55–98% missing. Rather than imputing these, the model uses dedicated missing-value flags and lets LightGBM's native NaN handling do the rest. A null `visitor_hist_starrating` means "new customer" — that's information, not noise.

## License

MIT
