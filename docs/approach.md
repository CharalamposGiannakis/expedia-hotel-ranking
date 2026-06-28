# Approach and design decisions

This document records the design choices, experiment progression, and lessons learned while building the hotel ranking system. It's written for anyone who wants to understand *why* the pipeline looks the way it does, not just *what* it does.

## Problem framing

The raw task is: given a user's hotel search and the list of properties returned, predict which hotel the user will book. The evaluation metric is NDCG@5 — a ranking metric that heavily rewards placing the booked hotel in the top 5 positions.

This framing matters because it rules out pointwise classification as the primary approach. NDCG doesn't care about predicted probabilities — it cares about the *ordering* of hotels within each search. Two models can have identical AUC but very different NDCG@5, because NDCG penalizes misranking at the top of the list more than at the bottom.

### Dataset characteristics

The training set contains 4,958,347 rows across 199,795 searches (roughly 25 hotels per search). The test set is comparably sized at ~5M rows. Each row describes one hotel shown in one search, with 54 columns covering the hotel's attributes (star rating, review score, price, location scores), the search context (destination, dates, guest counts), visitor history (average past spending and star preference, available for ~5% of rows), and competitive positioning (pricing and availability versus 8 unnamed competitors, 55–98% null).

Three rows in the training set have special significance: `click_bool`, `booking_bool`, and `position`. The first two form the target; the third is the hotel's display position, available only in training data. Position is the source of the most important bias in the dataset.

## Position bias: the central challenge

The dataset includes a `random_bool` flag. When `random_bool=1`, search results were displayed in random order; when `random_bool=0`, Expedia's own ranking algorithm determined display order. About 30% of the training data is randomly sorted.

Under normal sort, the booking rate is 3.73%. Under random sort, it drops to 0.51%. This 7× gap isn't because random sort shows worse hotels — the hotel pool is the same. It's because users are biased toward whatever appears first: they click what's visible and only reach lower positions if the top results don't fit.

This creates a problem for model training. Features that correlate with position (like `prop_location_score2`, which Expedia's own sort algorithm likely uses) will show inflated correlations with booking under normal sort. The feature *looks* predictive, but part of that correlation is just "hotels with high location scores get shown first, and whatever's shown first gets booked."

We verified this by comparing feature–booking correlations between the two subsets. Some features showed stable correlations (trustworthy), while others shifted substantially:

- `srch_query_affinity_score`: correlation *flips sign* from +0.038 (normal) to −0.035 (random)
- `promotion_flag`: drops from 0.041 to 0.010
- `prop_location_score2`: drops from 0.080 to 0.033

The analysis guided two decisions. First, `position` itself is excluded from features (it's train-only anyway, but even if it weren't, using it would teach the model position bias, not hotel quality). Second, we tested training on random-only data and upweighting random rows, but neither improved validation NDCG — the model apparently extracts useful signal from biased data too, presumably because within-search relative features are less position-confounded than raw features.

## Feature engineering

### Design philosophy

The central insight is that hotel search is *comparative*. A user doesn't evaluate a hotel in isolation — they compare it to the other options in their search. A $200/night hotel is expensive when the other results are $80–120, but cheap when they're $300+. This observation drove the most impactful feature category: within-search normalization.

### Feature categories

**Within-search normalization (categories C, D, E in the code).** For each key attribute (price, star rating, review score, location scores), we compute:
- Rank within the search (e.g., `price_rank` = 1 for cheapest)
- Percentile rank (`price_rank_pct`)
- Z-score relative to search mean and std
- Gap-to-best (distance from the top option in that dimension)
- Superlative flags (`is_cheapest`, `is_best_review`, etc.)

These 24 features dominate the model's importance rankings. The z-score of price within a search (`price_usd_zscore`) is more predictive than raw price, and the location score rank (`loc2_rank`) outperforms the raw location score.

**Missing-value indicators.** Rather than imputing nulls, we created binary flags for strategically important missing values. The rationale: in this dataset, missingness is informative. A null `visitor_hist_starrating` means the user has no purchase history — that's a "new customer" signal, not random noise. Similarly, `prop_starrating=0` doesn't mean zero stars; per the data dictionary, it means "unknown or cannot be publicized." These semantic zeros get their own flags.

**Competitor aggregation.** The raw data has 24 competitor columns (`comp1_rate`, `comp1_inv`, `comp1_rate_percent_diff` × 8 competitors), all 55–98% null. We compressed these into 7 features: how many competitors are cheaper, how many are more expensive, net competitive advantage (normalized −1 to +1), how many competitors are out of stock, and mean/max price difference. The raw columns are dropped.

**Property profiles (F2, added late in the pipeline).** These are hotel-level aggregate statistics computed from the *training split only*: how often the hotel appears in searches, its median price, price standard deviation, promotion rate, and how many distinct destinations and booking sites it's shown on. The most valuable derived feature is `price_vs_prop_median` — whether the hotel's current price is above or below its own typical price. This captures a "deal" signal without using click/booking targets.

Leakage prevention for F2: statistics are computed from training data, then joined to validation and test sets by `prop_id`. Hotels unseen in training get null values (LightGBM handles these natively).

### What we tried and dropped

**Target-encoded property aggregates (v2).** We computed per-hotel booking rate, click rate, and mean relevance using leave-one-out encoding with Bayesian smoothing (prior = global rate, smoothing factor = 100). Despite the regularization, these features caused severe overfitting — training NDCG jumped to 0.95+ while validation barely moved. The problem: with ~200K unique hotels and only ~4M training rows, most hotels appear fewer than 20 times. The smoothed rates converge to the global mean for rare hotels but memorize outcomes for frequent ones. We dropped these entirely.

**Destination-normalized pricing (F1).** Price divided by destination-median price, to capture "is this hotel expensive *for this destination*?" In practice, it added +0.0001 NDCG — essentially zero. The within-search price features already capture relative value at a finer granularity (per-search, not per-destination), making destination-level normalization redundant.

## Model selection and training

### Why LambdaRank

We chose LightGBM's `lambdarank` objective for three reasons:

1. It directly optimizes NDCG. The gradient computation uses swap deltas — for each pair of items in a query, it asks "how much would NDCG change if these two swapped ranks?" and uses that as the loss signal. This is more aligned with the evaluation metric than binary cross-entropy (which treats each row independently) or pairwise losses (which don't weight by rank position).

2. LightGBM handles missing values natively. With 95% null rates in visitor history and 55–98% in competitor columns, imputation would either destroy signal (mean imputation) or introduce complexity (model-based imputation). LightGBM learns optimal split directions for missing values during training.

3. It handles query groups naturally. The `group` parameter tells LightGBM which rows belong to the same search, so the ranking loss is computed within queries, not across the entire dataset.

### Training setup

- **Split:** 80/20 by `srch_id`. All hotels from the same search stay together — splitting within a search would leak information about the user's behavior to the validation set.
- **Training rows:** ~3.97M (159,836 searches)
- **Validation rows:** ~992K (39,959 searches)
- **Early stopping:** patience of 150 rounds on validation NDCG@5

### Hyperparameter tuning

We ran five configurations, varying learning rate and tree complexity:

| Config | Learning rate | Leaves | Val NDCG@5 | Train-val gap | Notes |
|--------|-------------|--------|-----------|--------------|-------|
| F2 baseline | 0.05 | 127 | 0.40576 | 0.103 | Starting point |
| T1 | 0.03 | 63 | 0.40677 | 0.070 | Good gap reduction |
| T2 | 0.02 | 63 | 0.40666 | 0.061 | Marginal gain |
| T3 | 0.02 | 31 | 0.40511 | 0.034 | Underfits |
| **T4** | **0.02** | **127** | **0.40766** | **0.095** | **Best validation** |
| T5 | 0.02 | 63 + subsample | 0.40583 | 0.045 | Subsampling hurts |

T4 won: the lower learning rate (0.02 vs 0.05) allows finer gradient steps, while keeping 127 leaves preserves model capacity. The train-val gap of 0.095 is acceptable — reducing it further (T3, T5) consistently hurt validation performance, suggesting these configurations underfit. The best iteration landed at 1049 out of 3000 allowed rounds.

### Baseline comparison

A logistic regression baseline (StandardScaler, balanced class weights, predicting P(booking)) scored NDCG@5 = 0.3497 on full data. The LambdaRank model beats it by 16.6% relative — a gap that comes from two sources: (1) gradient boosted trees capture feature interactions that linear models miss, and (2) the ranking objective directly optimizes what we're scored on, while the logistic baseline optimizes binary cross-entropy and only uses predicted probabilities as a proxy for ranking.

## Fairness analysis

We segmented users into families (`srch_children_count > 0`) and non-families, then measured per-segment NDCG@5.

### Why this segment

Family travelers are a natural minority in the dataset and plausibly underserved: they have different preferences (multi-room, kid-friendly amenities, proximity to attractions), and their search patterns may be underrepresented in training. If the model ranks worse for family searches, it creates a real-world equity problem — families see less relevant results and waste more time.

### Mitigation

The key insight: post-processing score adjustments don't work here. Family/non-family is a search-level attribute — all hotels in the same search share the segment. Adding a uniform bonus to all hotel scores in family searches shifts the distribution but leaves the *within-search ranking* unchanged. NDCG is a within-query metric, so this is a no-op.

Instead, we used pre-processing: training-time sample reweighting. Rows from the disadvantaged segment received a weight multiplier during LightGBM training, increasing the model's attention to their patterns. We searched over multipliers [1.5, 2.0, 3.0, 5.0], constrained to keep overall NDCG@5 within 98% of the unweighted baseline.

Results are stored in `outputs/bias_report.json` when the pipeline runs.

## Leakage prevention

A checklist we followed throughout:

- `position` excluded from features (train-only column, and it *is* the bias we're trying to avoid)
- `gross_bookings_usd` excluded (post-outcome variable — you only know this after the booking happens)
- `click_bool` and `booking_bool` excluded (they form the target)
- Train/val split by `srch_id` groups, not random row sampling
- Property profiles (F2) computed from training split only, then joined to validation and test
- `price_vs_historical` set to NaN when `prop_log_historical_price=0` (semantic zero = "not sold in prior period," not "price was $1")
- Target-encoded aggregates tested and rejected after observing overfitting

## Reproducing the results

The pipeline runs end-to-end via `bash run_pipeline.sh`. Each script is numbered and reads from the outputs of the previous step. The full pipeline on 5M training rows takes about 30–40 minutes on a machine with 16 GB RAM.

To reproduce the exact experiment progression, you would need to run the intermediate experiment scripts (`02b`, `02c` with different flags) and manually compare results — the frozen pipeline only runs the winning configuration.

All random seeds are fixed at 42 via `config.py`.
