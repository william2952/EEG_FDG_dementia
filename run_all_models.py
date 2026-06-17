#!/usr/bin/env python3
"""
Train all 9 PC models sequentially. Each model trains all 10 folds via
run_cv_parallel.py (which runs folds sequentially on MPS, or in parallel on CPU).

Model N  →  --pca-offset N-1  →  suffix N
  Model 1: PC1  (pca-offset 0, suffix 1)  → checkpoints_1/, logs_1/, predictions_1.parquet
  Model 2: PC2  (pca-offset 1, suffix 2)  → checkpoints_2/, logs_2/, predictions_2.parquet
  ...
  Model 9: PC9  (pca-offset 8, suffix 9)  → checkpoints_9/, logs_9/, predictions_9.parquet

Usage:
    # Run all 9 models on MPS (sequential folds):
    python run_all_models.py

    # Parallel folds on CPU:
    python run_all_models.py --accelerator cpu --max-workers 4

    # Skip models whose predictions_N.parquet already exist:
    python run_all_models.py --skip-done

    # Run only specific models (e.g. models 2, 3):
    python run_all_models.py --models 2 3

    # After everything finishes, merge per-model results (run separately per model):
    python merge_cv_results.py --suffix 2
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path


def model_is_done(output_dir: str, suffix: str, n_folds: int) -> bool:
    """All folds present for this model."""
    out = Path(output_dir)
    return all((out / f"fold_{i}_preds_{suffix}.parquet").exists() for i in range(n_folds))


def run_model(suffix: str, pca_offset: int, shared_args: list[str]) -> tuple[str, int]:
    """Run one model (all folds) via run_cv_parallel.py."""
    cmd = [
        sys.executable, "run_cv_parallel.py",
        "--pca-offset", str(pca_offset),
        "--suffix",     suffix,
    ] + shared_args
    print(f"\n{'#'*60}")
    print(f"# Model {suffix}  (PC{suffix}, pca-offset={pca_offset})")
    print(f"{'#'*60}")
    print(f"Command: {' '.join(cmd)}\n")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0
    status = "✓" if result.returncode == 0 else "✗"
    print(f"\n[all-models] Model {suffix} {status}  ({elapsed/60:.1f} min total)")
    return suffix, result.returncode


def main():
    parser = argparse.ArgumentParser(description="Train all 9 CV models (one per PC)")

    # ── Forwarded to run_cv_parallel.py ──────────────────────────────────────
    parser.add_argument("--data-dir",    default="model_data/non_ica_19_channels")
    parser.add_argument("--pca-parquet", default="model_data/matched_pca_vectors.parquet")
    parser.add_argument("--output-dir",  default="model_data/cv_results_stratified")
    parser.add_argument("--n-folds",     type=int, default=10)
    parser.add_argument("--n-bins",      type=int, default=20)
    parser.add_argument("--n-pca",       type=int, default=1)
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
                        help="Parallel folds within each model. >1 requires --accelerator cpu.")

    # ── Runner-only args ──────────────────────────────────────────────────────
    parser.add_argument("--n-models",    type=int, default=9,
                        help="Number of PC models to train (1 through N).")
    parser.add_argument("--models",      type=int, nargs="+", default=None,
                        help="Run only these model numbers, e.g. --models 2 3 7.")
    parser.add_argument("--skip-done",   action="store_true",
                        help="Skip models where all fold_N_preds_{suffix}.parquet files already exist.")

    args = parser.parse_args()

    model_nums = args.models if args.models is not None else list(range(1, args.n_models + 1))

    if args.skip_done:
        pending = [m for m in model_nums
                   if not model_is_done(args.output_dir, str(m), args.n_folds)]
        skipped = [m for m in model_nums
                   if model_is_done(args.output_dir, str(m), args.n_folds)]
        if skipped:
            print(f"[all-models] Skipping already-done models: {skipped}")
    else:
        pending = model_nums

    if not pending:
        print("[all-models] All models already done.")
        return

    print(f"[all-models] Will train models: {pending}")

    # Args forwarded to run_cv_parallel.py (excluding model-specific --pca-offset / --suffix)
    shared = [
        "--data-dir",    args.data_dir,
        "--pca-parquet", args.pca_parquet,
        "--output-dir",  args.output_dir,
        "--n-folds",     str(args.n_folds),
        "--n-bins",      str(args.n_bins),
        "--n-pca",       str(args.n_pca),
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
        "--skip-done",   # always pass so individual fold reruns are safe
    ]

    failed = []
    t_total = time.time()
    for model_num in pending:
        suffix     = str(model_num)
        pca_offset = model_num - 1   # model 1 → PC1 (offset 0), model 2 → PC2 (offset 1), …
        _, rc = run_model(suffix, pca_offset, shared)
        if rc != 0:
            failed.append(model_num)
            print(f"[all-models] Model {model_num} FAILED. Continuing with remaining models.")

    elapsed_total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"[all-models] Finished in {elapsed_total/3600:.1f} h")
    if failed:
        print(f"  Failed models: {sorted(failed)}")
        print(f"  Retry:  python run_all_models.py --models {' '.join(map(str, sorted(failed)))} [same flags]")
    else:
        print("  All models succeeded.")
        print("  To merge per-model results:")
        for m in range(1, args.n_models + 1):
            print(f"    python merge_cv_results.py --suffix {m}")


if __name__ == "__main__":
    main()
