#!/usr/bin/env python3
"""
Train 19 channel-ablation CV models (one per central channel), on a single PC
(default PC1, override with --pc).

For each of the 19 EEG channels, this ablates that channel plus its 3 nearest
neighbours (by standard 10-20 montage position) — same neighbour map already
computed in interpretability_analysis.ipynb's "Experiment 2: Channel
Importance" cell — by zeroing them in every window across train/val/holdout,
then runs the full 10-fold stratified CV (top-3 ensemble per fold), same as
the main PC-model CV pipeline. Unlike the notebook (which applies this mask
post-hoc to a single already-trained checkpoint), this retrains a full model
from scratch per channel, with the mask baked in from the start.

Each channel gets its own suffix (the channel name itself), so outputs are:
  checkpoints_C3/, logs_C3/, fold_N_preds_C3.parquet, predictions_C3.parquet

Usage:
    python run_channel_ablation.py
    python run_channel_ablation.py --channels C3 Cz F4   # only these
    python run_channel_ablation.py --skip-done
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

# Central channel -> [central] + 3 nearest neighbours, by Euclidean distance
# in the MNE standard_1020 montage. Identical to the neighbour_map printed in
# interpretability_analysis.ipynb (Experiment 2 cell) — hardcoded here so this
# script has no mne dependency on the cluster.
NEIGHBOUR_MAP = {
    "C3":  ["C3",  "P3",  "F3",  "Cz"],
    "C4":  ["C4",  "P4",  "F4",  "T8"],
    "Cz":  ["Cz",  "Pz",  "C3",  "Fz"],
    "F3":  ["F3",  "Fz",  "F7",  "Fp1"],
    "F4":  ["F4",  "Fz",  "F8",  "Fp2"],
    "F7":  ["F7",  "F3",  "Fp1", "T7"],
    "F8":  ["F8",  "F4",  "Fp2", "T8"],
    "Fp1": ["Fp1", "F7",  "Fp2", "F3"],
    "Fp2": ["Fp2", "Fp1", "F8",  "F4"],
    "Fz":  ["Fz",  "F3",  "F4",  "Cz"],
    "O1":  ["O1",  "P7",  "O2",  "P3"],
    "O2":  ["O2",  "O1",  "P8",  "P4"],
    "P3":  ["P3",  "Pz",  "P7",  "O1"],
    "P4":  ["P4",  "Pz",  "P8",  "O2"],
    "P7":  ["P7",  "T7",  "O1",  "P3"],
    "P8":  ["P8",  "O2",  "T8",  "P4"],
    "Pz":  ["Pz",  "P3",  "P4",  "Cz"],
    "T7":  ["T7",  "P7",  "F7",  "C3"],
    "T8":  ["T8",  "P8",  "F8",  "C4"],
}


def channel_is_done(output_dir: str, channel: str, n_folds: int) -> bool:
    out = Path(output_dir)
    return all((out / f"fold_{i}_preds_{channel}.parquet").exists() for i in range(n_folds))


def run_channel(channel: str, shared_args: list[str]) -> tuple[str, int]:
    mask = ",".join(NEIGHBOUR_MAP[channel])
    cmd = [
        sys.executable, "run_cv_parallel.py",
        "--suffix", channel,
        "--mask-channels", mask,
    ] + shared_args
    print(f"\n{'#'*60}")
    print(f"# Channel {channel}  (masking {NEIGHBOUR_MAP[channel]})")
    print(f"{'#'*60}")
    print(f"Command: {' '.join(cmd)}\n")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0
    status = "✓" if result.returncode == 0 else "✗"
    print(f"\n[channel-ablation] {channel} {status}  ({elapsed/60:.1f} min total)")
    return channel, result.returncode


def main():
    parser = argparse.ArgumentParser(description="Train 19 channel-ablation CV models on PC1")

    # Forwarded to run_cv_parallel.py -> train_cv.py
    parser.add_argument("--data-dir",    default="model_data/non_ica_19_channels_thresh")
    parser.add_argument("--pca-parquet", default="model_data/matched_pca_vectors.parquet")
    parser.add_argument("--output-dir",  default="model_data/channel_ablation_results")
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
                        help="Parallel folds within each channel. >1 requires --accelerator cpu.")
    parser.add_argument("--pc", type=int, default=1,
                        help="Which PC to run this ablation study on (1-indexed, e.g. 5 for PC5). "
                             "Default PC1.")
    parser.add_argument("--mask-band-lo", type=float, default=None,
                        help="Optional: also bandstop-filter [lo, hi] Hz out of every channel, "
                             "on top of the per-channel neighbour mask (e.g. 0.5 45 to strip all "
                             "5 standard EEG bands at once, since they're contiguous). Requires "
                             "--mask-band-hi too.")
    parser.add_argument("--mask-band-hi", type=float, default=None)

    # Runner-only args
    parser.add_argument("--channels",    type=str, nargs="+", default=None,
                        help="Only run these central channels, e.g. --channels C3 Cz F4. "
                             "Default: all 19.")
    parser.add_argument("--skip-done",   action="store_true",
                        help="Skip channels where all fold_N_preds_{channel}.parquet already exist.")

    args = parser.parse_args()

    if (args.mask_band_lo is None) != (args.mask_band_hi is None):
        raise SystemExit("--mask-band-lo and --mask-band-hi must be given together.")

    for c in (args.channels or []):
        if c not in NEIGHBOUR_MAP:
            raise SystemExit(f"Unknown channel '{c}'. Valid: {sorted(NEIGHBOUR_MAP)}")

    channel_list = args.channels if args.channels is not None else list(NEIGHBOUR_MAP.keys())

    if args.skip_done:
        pending = [c for c in channel_list if not channel_is_done(args.output_dir, c, args.n_folds)]
        skipped = [c for c in channel_list if channel_is_done(args.output_dir, c, args.n_folds)]
        if skipped:
            print(f"[channel-ablation] Skipping already-done channels: {skipped}")
    else:
        pending = channel_list

    if not pending:
        print("[channel-ablation] All channels already done.")
        return

    print(f"[channel-ablation] Will train channels: {pending}")

    # Single PC, fixed per run — this study is about channel importance for one PC at a time.
    pca_offset = args.pc - 1
    print(f"[channel-ablation] Running on PC{args.pc} (--n-pca 1 --pca-offset {pca_offset})")
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
    if args.mask_band_lo is not None:
        shared += ["--mask-band-lo", str(args.mask_band_lo), "--mask-band-hi", str(args.mask_band_hi)]
        print(f"[channel-ablation] ALSO bandstop-filtering [{args.mask_band_lo}, {args.mask_band_hi}] Hz "
              f"out of every channel, on top of each channel's neighbour mask.")

    failed = []
    t_total = time.time()
    for channel in pending:
        _, rc = run_channel(channel, shared)
        if rc != 0:
            failed.append(channel)
            print(f"[channel-ablation] {channel} FAILED. Continuing with remaining channels.")

    elapsed_total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"[channel-ablation] Finished in {elapsed_total/3600:.1f} h")
    if failed:
        print(f"  Failed channels: {sorted(failed)}")
        print(f"  Retry:  python run_channel_ablation.py --channels {' '.join(sorted(failed))} [same flags]")
    else:
        print("  All channels succeeded.")
        print("  To merge per-channel results:")
        for c in channel_list:
            print(f"    python merge_cv_results.py --suffix {c} --output-dir {args.output_dir}")


if __name__ == "__main__":
    main()
