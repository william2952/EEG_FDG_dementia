#!/usr/bin/env python3
"""
Train 5 frequency-band-ablation CV models (one per band: delta, theta, alpha,
beta, gamma), on a single PC (default PC1, override with --pc).

For each band, this bandstop-filters that frequency range out of every EEG
channel (4th-order zero-phase Butterworth, via scipy sosfiltfilt) across
every window in train/val/holdout, then runs the full 10-fold stratified CV
(top-3 ensemble per fold), same as the main PC-model CV pipeline and the same
retraining philosophy as run_channel_ablation.py. Unlike the notebook's
"Experiment 1" (which applies the bandstop post-hoc to windows fed through a
single already-trained checkpoint), this retrains a full model from scratch
per band, with the filter baked in from the start.

Each band gets its own suffix (the band name itself), so outputs are:
  checkpoints_delta/, logs_delta/, fold_N_preds_delta.parquet, predictions_delta.parquet

Usage:
    python run_band_ablation.py --pc 5
    python run_band_ablation.py --bands delta theta   # only these
    python run_band_ablation.py --skip-done
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

# Standard EEG band edges (Hz). Identical to BANDS in
# interpretability_analysis.ipynb (Experiment 1 cell) — hardcoded here so
# this script has no notebook dependency on the cluster.
BANDS = {
    "delta": (0.5, 4),
    "theta": (4,   8),
    "alpha": (8,  12),
    "beta":  (12, 30),
    "gamma": (30, 45),
}


def band_is_done(output_dir: str, band: str, n_folds: int) -> bool:
    out = Path(output_dir)
    return all((out / f"fold_{i}_preds_{band}.parquet").exists() for i in range(n_folds))


def run_band(band: str, shared_args: list[str]) -> tuple[str, int]:
    lo, hi = BANDS[band]
    cmd = [
        sys.executable, "run_cv_parallel.py",
        "--suffix", band,
        "--mask-band-lo", str(lo),
        "--mask-band-hi", str(hi),
    ] + shared_args
    print(f"\n{'#'*60}")
    print(f"# Band {band}  ({lo}-{hi} Hz)")
    print(f"{'#'*60}")
    print(f"Command: {' '.join(cmd)}\n")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0
    status = "✓" if result.returncode == 0 else "✗"
    print(f"\n[band-ablation] {band} {status}  ({elapsed/60:.1f} min total)")
    return band, result.returncode


def main():
    parser = argparse.ArgumentParser(description="Train 5 band-ablation CV models on a single PC")

    # Forwarded to run_cv_parallel.py -> train_cv.py
    parser.add_argument("--data-dir",    default="model_data/non_ica_19_channels_thresh")
    parser.add_argument("--pca-parquet", default="model_data/matched_pca_vectors.parquet")
    parser.add_argument("--output-dir",  default="model_data/band_ablation_results")
    parser.add_argument("--n-folds",     type=int, default=10)
    parser.add_argument("--n-bins",      type=int, default=20)
    parser.add_argument("--n-seg",       type=int, default=1)
    parser.add_argument("--n-samples",   type=int, default=100)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--max-epochs",  type=int, default=50)
    parser.add_argument("--patience",    type=int, default=10)
    parser.add_argument("--batch-size",  type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--accelerator", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--ensemble-k",  type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Parallel folds within each band. >1 requires --accelerator cpu.")
    parser.add_argument("--pc", type=int, default=1,
                        help="Which PC to run this ablation study on (1-indexed, e.g. 5 for PC5). "
                             "Default PC1.")

    # Runner-only args
    parser.add_argument("--bands",       type=str, nargs="+", default=None,
                        help="Only run these bands, e.g. --bands delta theta. Default: all 5.")
    parser.add_argument("--skip-done",   action="store_true",
                        help="Skip bands where all fold_N_preds_{band}.parquet already exist.")

    args = parser.parse_args()

    for b in (args.bands or []):
        if b not in BANDS:
            raise SystemExit(f"Unknown band '{b}'. Valid: {sorted(BANDS)}")

    band_list = args.bands if args.bands is not None else list(BANDS.keys())

    if args.skip_done:
        pending = [b for b in band_list if not band_is_done(args.output_dir, b, args.n_folds)]
        skipped = [b for b in band_list if band_is_done(args.output_dir, b, args.n_folds)]
        if skipped:
            print(f"[band-ablation] Skipping already-done bands: {skipped}")
    else:
        pending = band_list

    if not pending:
        print("[band-ablation] All bands already done.")
        return

    print(f"[band-ablation] Will train bands: {pending}")

    # Single PC, fixed per run — this study is about band importance for one PC at a time.
    pca_offset = args.pc - 1
    print(f"[band-ablation] Running on PC{args.pc} (--n-pca 1 --pca-offset {pca_offset})")
    shared = [
        "--data-dir",    args.data_dir,
        "--pca-parquet", args.pca_parquet,
        "--output-dir",  args.output_dir,
        "--n-folds",     str(args.n_folds),
        "--n-bins",      str(args.n_bins),
        "--n-pca",       "1",
        "--pca-offset",  str(pca_offset),
        "--n-seg",       str(args.n_seg),
        "--n-samples",   str(args.n_samples),
        "--seed",        str(args.seed),
        "--max-epochs",  str(args.max_epochs),
        "--patience",    str(args.patience),
        "--batch-size",  str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--accelerator", args.accelerator,
        "--ensemble-k",  str(args.ensemble_k),
        "--max-workers", str(args.max_workers),
        "--skip-done",
    ]

    failed = []
    t_total = time.time()
    for band in pending:
        _, rc = run_band(band, shared)
        if rc != 0:
            failed.append(band)
            print(f"[band-ablation] {band} FAILED. Continuing with remaining bands.")

    elapsed_total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"[band-ablation] Finished in {elapsed_total/3600:.1f} h")
    if failed:
        print(f"  Failed bands: {sorted(failed)}")
        print(f"  Retry:  python run_band_ablation.py --bands {' '.join(sorted(failed))} [same flags]")
    else:
        print("  All bands succeeded.")
        print("  To merge per-band results:")
        for b in band_list:
            print(f"    python merge_cv_results.py --suffix {b} --output-dir {args.output_dir}")


if __name__ == "__main__":
    main()
