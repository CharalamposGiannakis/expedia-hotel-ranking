#!/bin/bash
# Full pipeline: data → features → model → evaluation → submission
#
# Prerequisites:
#   - data/train.csv and data/test.csv in place
#   - Python environment with dependencies from requirements.txt
#
# For development (fast iteration on a 10% sample):
#   Set SAMPLE_FRAC=0.1 in config.py, then run this script.
# For the final submission:
#   Set SAMPLE_FRAC=1.0 in config.py. Expect ~30-40 min on 16 GB RAM.

set -e

echo "================================================"
echo "  Expedia Hotel Ranking — Full Pipeline"
echo "================================================"

cd "$(dirname "$0")"

echo ""
echo "[1/9] Data check..."
python src/00_data_check.py

echo ""
echo "[2/9] Exploratory data analysis..."
python src/01_eda.py

echo ""
echo "[3/9] Base feature engineering (79 features)..."
python src/02_feature_engineering.py

echo ""
echo "[4/9] Train/val split (group-aware by srch_id)..."
python src/03_split.py

echo ""
echo "[5/9] Property profile features (F2, +10 features)..."
python src/02c_feature_v3_nontarget.py --only f2

echo ""
echo "[6/9] LightGBM LambdaRank training..."
python src/04_train_lgbm.py

echo ""
echo "[7/9] Logistic regression baseline..."
python src/05_train_baseline.py

echo ""
echo "[8/9] Model evaluation and comparison..."
python src/06_evaluate.py

echo ""
echo "[9/9] Generate submission..."
python src/07_predict.py

echo ""
echo "================================================"
echo "  Pipeline complete!"
echo "  Submission: outputs/submission.csv"
echo "  Figures:    figures/"
echo "  Model:      outputs/lgbm_model.txt"
echo "================================================"
echo ""
echo "Optional — run fairness analysis separately:"
echo "  python src/08_bias_analysis.py"
