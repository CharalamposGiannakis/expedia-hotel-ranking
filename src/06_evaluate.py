"""
06 — Evaluation
Compare LightGBM vs baseline on validation set.
Produce evaluation numbers and comparison tables.

Run time: ~1 min.
"""


import pandas as pd
import numpy as np
import json

from config import OUTPUT_DIR, FIGURE_DIR
from src.utils import print_header, eval_ndcg

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print_header("06 — EVALUATION")
    
    # ── Load predictions ──
    lgbm = pd.read_csv(OUTPUT_DIR / "lgbm_val_results.csv")
    baseline = pd.read_csv(OUTPUT_DIR / "baseline_val_results.csv")
    
    # Merge
    val = lgbm.merge(baseline[["srch_id", "prop_id", "lr_score"]], on=["srch_id", "prop_id"])
    
    # ── NDCG@k for multiple k values ──
    results = {}
    for k in [1, 3, 5, 10]:
        ndcg_lgbm = eval_ndcg(val, "lgbm_score", k=k)
        ndcg_lr = eval_ndcg(val, "lr_score", k=k)
        results[f"NDCG@{k}"] = {"LightGBM": round(ndcg_lgbm, 5), "LogReg": round(ndcg_lr, 5)}
        print(f"  NDCG@{k}:  LightGBM={ndcg_lgbm:.5f}  LogReg={ndcg_lr:.5f}  "
              f"Δ={ndcg_lgbm - ndcg_lr:+.5f}")
    
    # ── Per-query analysis ──
    print("\nPer-query comparison...")
    query_results = []
    for srch_id, group in val.groupby("srch_id"):
        from src.utils import ndcg_at_k
        lgbm_ranked = group.sort_values("lgbm_score", ascending=False)
        lr_ranked = group.sort_values("lr_score", ascending=False)
        query_results.append({
            "srch_id": srch_id,
            "n_props": len(group),
            "has_booking": group["relevance"].max() == 5,
            "ndcg5_lgbm": ndcg_at_k(lgbm_ranked["relevance"].values, 5),
            "ndcg5_lr": ndcg_at_k(lr_ranked["relevance"].values, 5),
        })
    
    qdf = pd.DataFrame(query_results)
    qdf["lgbm_wins"] = qdf["ndcg5_lgbm"] > qdf["ndcg5_lr"]
    qdf["lr_wins"] = qdf["ndcg5_lr"] > qdf["ndcg5_lgbm"]
    qdf["tie"] = qdf["ndcg5_lgbm"] == qdf["ndcg5_lr"]
    
    print(f"  LightGBM wins: {qdf['lgbm_wins'].sum():,} ({qdf['lgbm_wins'].mean():.1%})")
    print(f"  LogReg wins:   {qdf['lr_wins'].sum():,} ({qdf['lr_wins'].mean():.1%})")
    print(f"  Ties:          {qdf['tie'].sum():,} ({qdf['tie'].mean():.1%})")
    
    # ── Visualization ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart of NDCG@k
    ks = list(results.keys())
    lgbm_vals = [results[k]["LightGBM"] for k in ks]
    lr_vals = [results[k]["LogReg"] for k in ks]
    x = np.arange(len(ks))
    axes[0].bar(x - 0.15, lgbm_vals, 0.3, label="LightGBM", color="steelblue")
    axes[0].bar(x + 0.15, lr_vals, 0.3, label="LogReg", color="coral")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ks)
    axes[0].set_ylabel("NDCG")
    axes[0].set_title("Model Comparison: NDCG@k")
    axes[0].legend()
    
    # Distribution of per-query NDCG@5 differences
    diff = qdf["ndcg5_lgbm"] - qdf["ndcg5_lr"]
    axes[1].hist(diff, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_title("Per-Query NDCG@5 Difference (LightGBM − LogReg)")
    axes[1].set_xlabel("NDCG@5 difference")
    axes[1].set_ylabel("Number of queries")
    
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # ── Save report ──
    report_lines = [
        "EVALUATION REPORT",
        "=" * 50,
        "",
        "Model Comparison (Validation Set)",
        "-" * 40,
    ]
    for k, vals in results.items():
        report_lines.append(f"  {k}: LightGBM={vals['LightGBM']:.5f}  LogReg={vals['LogReg']:.5f}")
    
    report_lines.extend([
        "",
        "Per-Query Wins",
        "-" * 40,
        f"  LightGBM better: {qdf['lgbm_wins'].sum():,} queries ({qdf['lgbm_wins'].mean():.1%})",
        f"  LogReg better:   {qdf['lr_wins'].sum():,} queries ({qdf['lr_wins'].mean():.1%})",
        f"  Ties:            {qdf['tie'].sum():,} queries ({qdf['tie'].mean():.1%})",
    ])
    
    (OUTPUT_DIR / "evaluation_report.txt").write_text("\n".join(report_lines))
    
    print("\n✓ Evaluation complete.")


if __name__ == "__main__":
    main()
