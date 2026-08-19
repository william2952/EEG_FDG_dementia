#!/usr/bin/env python3
"""
Scan all channel- and band-ablation output directories and report, per
suffix (channel or band name), whether:
  - training is complete (all N fold_{i}_preds_{suffix}.parquet files exist)
  - it has already been merged (predictions_{suffix}.parquet exists, written
    by merge_cv_results.py)

Pure stdlib (pathlib only) — no need to activate the conda env to run this,
just needs to be run from the project root (where model_data/ lives).

Usage:
    python check_ablation_status.py
    python check_ablation_status.py --model-data-dir model_data
"""
import argparse
import subprocess
import sys
from pathlib import Path

CHANNELS = ["C3", "C4", "Cz", "F3", "F4", "F7", "F8", "Fp1", "Fp2", "Fz",
            "O1", "O2", "P3", "P4", "P7", "P8", "Pz", "T7", "T8"]
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

# (output-dir name, list of suffixes, n_folds) — every ablation run set up so far.
STUDIES = [
    # ── Channel ablation (19 channels) ──
    ("channel_ablation_thresh_results",          CHANNELS, "Channel ablation — non-ICA PC1"),
    ("channel_ablation_thresh_pc2_results",      CHANNELS, "Channel ablation — non-ICA PC2"),
    ("channel_ablation_thresh_pc5_results",      CHANNELS, "Channel ablation — non-ICA PC5"),
    ("channel_ablation_thresh_pc8_results",      CHANNELS, "Channel ablation — non-ICA PC8"),
    ("channel_ablation_thresh_ica_results",      CHANNELS, "Channel ablation — ICA PC1"),
    ("channel_ablation_thresh_ica_pc5_results",  CHANNELS, "Channel ablation — ICA PC5"),
    ("channel_ablation_thresh_ica_pc2_results",  CHANNELS, "Channel ablation — ICA PC2"),
    ("channel_ablation_thresh_ica_pc8_results",  CHANNELS, "Channel ablation — ICA PC8"),
    ("channel_ablation_thresh_pc5_no_bands_results", CHANNELS, "Channel ablation — non-ICA PC5, all bands removed"),
    # ── Band ablation (5 bands) ──
    ("band_ablation_thresh_pc1_results",         BANDS,    "Band ablation — non-ICA PC1"),
    ("band_ablation_thresh_pc2_results",         BANDS,    "Band ablation — non-ICA PC2"),
    ("band_ablation_thresh_pc5_results",         BANDS,    "Band ablation — non-ICA PC5"),
    ("band_ablation_thresh_pc8_results",         BANDS,    "Band ablation — non-ICA PC8"),
    ("band_ablation_thresh_ica_pc1_results",     BANDS,    "Band ablation — ICA PC1"),
    ("band_ablation_thresh_ica_pc2_results",     BANDS,    "Band ablation — ICA PC2"),
    ("band_ablation_thresh_ica_pc5_results",     BANDS,    "Band ablation — ICA PC5"),
    ("band_ablation_thresh_ica_pc8_results",     BANDS,    "Band ablation — ICA PC8"),
]


def suffix_status(out_dir: Path, suffix: str, n_folds: int) -> tuple[int, bool]:
    n_done = sum((out_dir / f"fold_{i}_preds_{suffix}.parquet").exists() for i in range(n_folds))
    merged = (out_dir / f"predictions_{suffix}.parquet").exists()
    return n_done, merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-data-dir", default="model_data")
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--run-merges", action="store_true",
                        help="Actually invoke merge_cv_results.py for every fully-trained, "
                             "not-yet-merged suffix, instead of just printing the commands.")
    parser.add_argument("--exclude-dir", type=str, nargs="+", default=[],
                        help="Output-dir name(s) to skip entirely, e.g. still-running studies: "
                             "--exclude-dir band_ablation_thresh_ica_pc5_results band_ablation_thresh_ica_pc8_results")
    args = parser.parse_args()

    base = Path(args.model_data_dir)
    excluded = set(args.exclude_dir)
    still_training = []   # (dir, suffix, n_done, n_folds)
    ready_to_merge = []   # (dir, suffix)
    already_merged = []   # (dir, suffix)

    for dirname, suffixes, label in STUDIES:
        if dirname in excluded:
            print(f"[excluded] {label:38s}  {dirname}")
            continue
        out_dir = base / dirname
        if not out_dir.exists():
            print(f"[skip] {label:38s}  {dirname}  (directory not found — not started or not synced here)")
            continue

        print(f"\n{label}  ({dirname})")
        for sfx in suffixes:
            n_done, merged = suffix_status(out_dir, sfx, args.n_folds)
            if merged:
                tag = "merged"
                already_merged.append((dirname, sfx))
            elif n_done == args.n_folds:
                tag = "READY TO MERGE"
                ready_to_merge.append((dirname, sfx))
            else:
                tag = f"training ({n_done}/{args.n_folds} folds)"
                still_training.append((dirname, sfx, n_done, args.n_folds))
            print(f"  {sfx:8s}  {tag}")

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Already merged:    {len(already_merged)}")
    print(f"  Ready to merge:    {len(ready_to_merge)}")
    print(f"  Still training:    {len(still_training)}")

    if ready_to_merge and not args.run_merges:
        print(f"\n--- Fully trained but NOT yet merged — run these (or rerun with --run-merges): ---")
        for dirname, sfx in ready_to_merge:
            print(f"  python merge_cv_results.py --suffix {sfx} --output-dir model_data/{dirname}")

    if ready_to_merge and args.run_merges:
        print(f"\n--- Running {len(ready_to_merge)} merges ---")
        failed = []
        for i, (dirname, sfx) in enumerate(ready_to_merge, 1):
            out_dir = f"model_data/{dirname}"
            print(f"\n[{i}/{len(ready_to_merge)}] merge --suffix {sfx} --output-dir {out_dir}")
            cmd = [sys.executable, "merge_cv_results.py",
                   "--suffix", sfx, "--output-dir", out_dir, "--n-folds", str(args.n_folds)]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                failed.append((dirname, sfx))
        print(f"\n{'='*70}")
        print(f"Merged {len(ready_to_merge) - len(failed)}/{len(ready_to_merge)} successfully.")
        if failed:
            print("Failed:")
            for dirname, sfx in failed:
                print(f"  {dirname}  {sfx}")

    if still_training:
        print(f"\n--- Still training (check squeue) ---")
        for dirname, sfx, n_done, n_folds in still_training:
            print(f"  {dirname:38s} {sfx:8s} {n_done}/{n_folds} folds")


if __name__ == "__main__":
    main()
