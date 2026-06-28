# Decisions Log

All key decisions for the hotel ranking pipeline. These reflect the final pipeline state.

---

## D1: Target variable
- **Relevance grades**: booking=5, click=1, neither=0 (per competition specification)
- **Primary model target**: relevance (for LambdaRank)
- **Baseline target**: booking_bool (binary classification)

## D2: Evaluation metric
- **NDCG@5** (per Kaggle competition rules)
- Validation: group-aware split by srch_id (20% held out)
- Custom `eval_ndcg()` computes ideal from full query, not just top-k

## D3: Train/val split
- **Group split by srch_id** — all properties from the same search stay together
- Train: ~3.97M rows (159,836 searches), Val: ~992K rows (39,959 searches)

## D4: Missing values
- **Competitor columns**: aggregated into 7 summary features, raw columns dropped
- **Visitor history**: NaN = no history, LightGBM handles NaN natively
- **Semantic zeros**: `prop_starrating=0` (unknown), `prop_review_score=0` (no reviews), `prop_log_historical_price=0` (not sold) — each gets a dedicated flag
- **price_vs_historical**: set to NaN when historical log price is 0

## D5: Primary model (best config)
- **LightGBM LambdaRank**, lr=0.02, leaves=127, min_child=50, subsample=0.8
- Best iteration: 1049 | Val NDCG@5: 0.40766 | Kaggle: 0.40533

## D6: Feature set (89 features)
- v1 base (79): raw, missing flags, semantic zeros, within-search norms, price, visitor match, competitor aggs, search context
- F2 addition (10): non-target property profiles (seen count, price stats, promotion rate, distinct destinations/sites, price vs own average)

## D6b: Recommender systems integration
- The pipeline implements a **hybrid recommender system** combining:
  - Content-based filtering (visitor-property match features)
  - Item popularity signals (property profiles from F2)
  - Learning-to-rank recommendation engine (LambdaRank)
- Pure collaborative filtering was impractical: no persistent user IDs, 95% missing visitor history, extreme sparsity
- The cold start problem is addressed via property-level popularity priors (F2 features)

## D7: Rejected approaches
- v2 target aggregates: severe overfitting even on full data
- F1 price normalization: zero gain, redundant with within-search features
- Random_bool upweighting: hurt performance
- Heavy regularization: underfit
- price_per_night: ambiguous per competition specification

## D8: Fairness analysis
- Segment: Family vs Non-Family (srch_children_count > 0)
- Mitigation: training-time sample reweighting (not post-processing — explained why)
- Constraint: overall NDCG@5 within 98% of baseline
