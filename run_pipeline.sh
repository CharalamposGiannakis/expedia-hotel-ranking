#!/bin/bash
# Run the full pipeline end-to-end.
# Usage: bash run_pipeline.sh
# For development (sampled data), set SAMPLE_FRAC=0.1 in config.py
# For final submission, set SAMPLE_FRAC=1.0

set -e  # Exit on any error

echo "============================================"
echo "  Hotel Search Ranking Pipeline"
echo "============================================"

cd "$(dirname "$0")"

echo ""
echo "[1/10] Data check..."
python src/00_data_check.py

echo ""
echo "[2/10] EDA..."
python src/01_eda.py

echo ""
echo "[3/10] Feature engineering (v1)..."
python src/02_feature_engineering.py

echo ""
echo "[4/10] Train/val split..."
python src/03_split.py

echo ""
echo "[5/10] Non-target property profile features (F2)..."
python src/02c_feature_v3_nontarget.py --only f2

echo ""
echo "[6/10] LightGBM LambdaRank..."
python src/04_train_lgbm.py

echo ""
echo "[7/10] Logistic regression baseline..."
python src/05_train_baseline.py

echo ""
echo "[8/10] Evaluation..."
python src/06_evaluate.py

echo ""
echo "[9/10] Generate submission..."
python src/07_predict.py --model outputs/exp_t4_lr02_leaves127

echo ""
echo "[10/10] Bias analysis..."
python src/08_bias_analysis.py

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "  Submission file: outputs/submission.csv"
echo "  Bias report:     outputs/bias_report.json"
echo "============================================"
