#!/usr/bin/env python3
"""
Launch all CV folds in parallel (CPU) or sequentially (MPS/CUDA).

Each fold runs as a subprocess, so memory is fully released between folds
and a crash in one fold never kills the others.

Usage examples
--------------
# Parallel on CPU (fastest overall if you have many cores):
python run_cv_parallel.py --accelerator cpu --max-workers 4

# Sequential on MPS (one fold at a time, clean process per fold):
python run_cv_parallel.py --accelerator mps

# Resume: skip folds that already have a fold_N_preds.parquet:
python run_cv_parallel.py --accelerator mps --skip-done

# Then aggregate results when all folds are done:
python merge_cv_results.py
"""
import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def fold_is_done(output_dir: str, fold_idx: int, suffix: str) -> bool:
    return (Path(output_dir) / f"fold_{fold_idx}_preds_{suffix}.parquet").exists()


def run_fold(fold_idx: int, base_args: list[str]) -> tuple[int, int, str]:
    """Run one fold as a subprocess. Returns (fold_idx, returncode, stderr)."""
    cmd = [sys.executable, "train_cv.py", "--fold-idx", str(fold_idx)] + base_args
    print(f"[launcher] Starting fold {fold_idx}: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0
    status = "✓" if result.returncode == 0 else "✗"
    print(f"[launcher] Fold {fold_idx} {status}  ({elapsed/60:.1f} min)")
    return fold_idx, result.returncode, ""


def main():
    parser = argparse.ArgumentParser(description="Parallel/sequential CV fold launcher")

    # ── Forwarded args (must match train_cv.py) ───────────────────────────────
    parser.add_argument("--data-dir",    default="model_data/non_ica_19_channels")
    parser.add_argument("--pca-parquet", default="model_data/matched_pca_vectors.parquet")
    parser.add_argument("--output-dir",  default="model_data/cv_results_stratified")
    parser.add_argument("--n-folds",     type=int, default=10)
    parser.add_argument("--n-bins",      type=int, default=20)
    parser.add_argument("--n-pca",       type=int, default=1)
    parser.add_argument("--pca-offset",  type=int, default=0)
    parser.add_argument("--n-seg",       type=int, default=1)
    parser.add_argument("--n-samples",   type=int, default=100)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--max-epochs",  type=int, default=50)
    parser.add_argument("--patience",    type=int, default=10)
    parser.add_argument("--batch-size",  type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--accelerator", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--ensemble-k",  type=int, default=3)
    parser.add_argument("--suffix",      type=str, default="1",
                        help="Output suffix passed to train_cv.py (e.g. '2' for PC2 model).")
    parser.add_argument("--mask-channels", type=str, default=None,
                        help="Comma-separated channel names to ablate, forwarded to train_cv.py.")
    parser.add_argument("--mask-band-lo", type=float, default=None,
                        help="Low edge (Hz) of a frequency band to ablate, forwarded to train_cv.py.")
    parser.add_argument("--mask-band-hi", type=float, default=None,
                        help="High edge (Hz) of a frequency band to ablate, forwarded to train_cv.py.")
    parser.add_argument("--keep-band-lo", type=float, default=None,
                        help="Low edge (Hz) of a frequency band to isolate (bandpass), forwarded to train_cv.py.")
    parser.add_argument("--keep-band-hi", type=float, default=None,
                        help="High edge (Hz) of a frequency band to isolate (bandpass), forwarded to train_cv.py.")

    # ── Launcher-only args ────────────────────────────────────────────────────
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Max parallel folds. Use 1 for MPS/CUDA (sequential), "
                             ">1 only with --accelerator cpu.")
    parser.add_argument("--skip-done",   action="store_true",
                        help="Skip folds that already have fold_N_preds.parquet.")
    parser.add_argument("--folds",       type=int, nargs="+", default=None,
                        help="Run only these fold indices, e.g. --folds 3 7 9.")

    args = parser.parse_args()

    if args.max_workers > 1 and args.accelerator != "cpu":
        print(f"WARNING: --max-workers {args.max_workers} with --accelerator {args.accelerator} "
              f"will share the device across processes. Consider --accelerator cpu for true parallelism.")

    # Build the forwarded arg list (everything except launcher-only flags)
    forward = [
        "--data-dir",    args.data_dir,
        "--pca-parquet", args.pca_parquet,
        "--output-dir",  args.output_dir,
        "--n-folds",     str(args.n_folds),
        "--n-bins",      str(args.n_bins),
        "--n-pca",       str(args.n_pca),
        "--pca-offset",  str(args.pca_offset),
        "--n-seg",       str(args.n_seg),
        "--n-samples",   str(args.n_samples),
        "--seed",        str(args.seed),
        "--max-epochs",  str(args.max_epochs),
        "--patience",    str(args.patience),
        "--batch-size",  str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--accelerator", args.accelerator,
        "--ensemble-k",  str(args.ensemble_k),
        "--suffix",      args.suffix,
    ]
    if args.mask_channels:
        forward += ["--mask-channels", args.mask_channels]
    if args.mask_band_lo is not None:
        forward += ["--mask-band-lo", str(args.mask_band_lo), "--mask-band-hi", str(args.mask_band_hi)]
    if args.keep_band_lo is not None:
        forward += ["--keep-band-lo", str(args.keep_band_lo), "--keep-band-hi", str(args.keep_band_hi)]

    fold_indices = args.folds if args.folds is not None else list(range(args.n_folds))

    if args.skip_done:
        pending = [i for i in fold_indices if not fold_is_done(args.output_dir, i, args.suffix)]
        skipped = [i for i in fold_indices if fold_is_done(args.output_dir, i, args.suffix)]
        if skipped:
            print(f"[launcher] Skipping already-done folds: {skipped}")
    else:
        pending = fold_indices

    if not pending:
        print("[launcher] All folds already done. Run merge_cv_results.py to aggregate.")
        return

    print(f"[launcher] Running {len(pending)} fold(s) with max_workers={args.max_workers}: {pending}")

    failed = []
    if args.max_workers == 1:
        # Sequential — simplest, avoids any concurrency issues
        for fold_idx in pending:
            _, rc, _ = run_fold(fold_idx, forward)
            if rc != 0:
                failed.append(fold_idx)
                print(f"[launcher] Fold {fold_idx} FAILED (exit code {rc}). Continuing with remaining folds.")
    else:
        # Parallel on CPU
        with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {pool.submit(run_fold, i, forward): i for i in pending}
            for fut in as_completed(futures):
                fold_idx, rc, _ = fut.result()
                if rc != 0:
                    failed.append(fold_idx)
                    print(f"[launcher] Fold {fold_idx} FAILED (exit code {rc}).")

    print("\n[launcher] Done.")
    if failed:
        print(f"  Failed folds: {sorted(failed)}")
        print(f"  Re-run with:  python run_cv_parallel.py --folds {' '.join(map(str, sorted(failed)))} [same flags]")
    else:
        print("  All folds succeeded. Run: python merge_cv_results.py --output-dir", args.output_dir)


if __name__ == "__main__":
    main()
