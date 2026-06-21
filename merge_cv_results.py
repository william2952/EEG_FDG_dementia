#!/usr/bin/env python3
"""
Aggregate per-fold prediction files produced by train_cv.py --fold-idx
into a single predictions.parquet and print overall CV metrics.

Usage:
    python merge_cv_results.py
    python merge_cv_results.py --output-dir model_data/cv_results_stratified
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="model_data/cv_results_stratified")
    parser.add_argument("--n-folds",    type=int, default=10)
    parser.add_argument("--suffix",     type=str, default="1",
                        help="Model suffix to aggregate (e.g. '2' reads fold_N_preds_2.parquet).")
    args = parser.parse_args()

    out = Path(args.output_dir)
    sfx = f"_{args.suffix}"
    dfs = []
    missing = []
    for i in range(args.n_folds):
        p = out / f"fold_{i}_preds{sfx}.parquet"
        if p.exists():
            dfs.append(pd.read_parquet(p))
        else:
            missing.append(i)

    if missing:
        print(f"WARNING: missing fold files for folds {missing} — they will be excluded from metrics.")

    if not dfs:
        print("No fold prediction files found.")
        return

    combined = pd.concat(dfs, ignore_index=True)
    merged_path = out / f"predictions{sfx}.parquet"
    combined.to_parquet(merged_path, index=False)
    print(f"Saved {len(combined)} rows → {merged_path}")

    # Per-fold R²
    fold_r2s = []
    for fold_idx, grp in combined.groupby("fold"):
        r2 = max(0.0, r2_score(grp["true_pc1"], grp["pred_pc1"]))
        fold_r2s.append((fold_idx, r2))
        print(f"  fold {fold_idx:2d}  R² = {r2:.4f}  (n={len(grp)})")

    r2_arr = np.array([r for _, r in fold_r2s])
    print(f"\nMean R² = {r2_arr.mean():.4f}  ±  {r2_arr.std():.4f}  (SD across {len(r2_arr)} folds)")


if __name__ == "__main__":
    main()
