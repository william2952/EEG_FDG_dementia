#!/bin/bash
# ── EEG CV training on Bouchet GPU nodes ────────────────────────────────────
# Account confirmed via `slurm_checkup.sh`: pi_am2359 (Adam Mecca), default account.
#
# Usage:
#   sbatch train_fold.sbatch            # runs all 10 folds sequentially inside one job
#   sbatch --array=0-9 train_fold.sbatch  # runs each fold as its own array task (faster wall-clock)

#SBATCH --job-name=eeg_cv
#SBATCH --account=pi_am2359
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

module load miniconda
conda activate eeg

# Persistent storage (never auto-purged) — code, input data, and FINAL outputs live here.
PROJECT_DIR="$HOME/project/eeg_project"
# 60-day scratch (auto-deleted after 60 days) — bulky/regenerable per-epoch checkpoints go here.
SCRATCH_DIR="$HOME/project_pi_am2359/eeg_scratch"

mkdir -p "$SCRATCH_DIR/checkpoints" logs

cd "$PROJECT_DIR"

# If submitted as a job array (--array=0-9), SLURM_ARRAY_TASK_ID selects one fold.
# Otherwise (no --array), run every fold sequentially in this single job.
FOLD_ARGS=()
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  FOLD_ARGS=(--fold-idx "$SLURM_ARRAY_TASK_ID")
fi

python train_cv.py \
  --data-dir      "$PROJECT_DIR/model_data/non_ica_19_channels" \
  --pca-parquet   "$PROJECT_DIR/model_data/matched_pca_vectors.parquet" \
  --output-dir    "$SCRATCH_DIR/cv_results_stratified" \
  --accelerator   cuda \
  --num-workers   8 \
  --batch-size    64 \
  "${FOLD_ARGS[@]}"

# Copy the small, durable outputs (predictions/coefficients) to persistent project storage.
# Skip the bulky per-epoch checkpoint files — those stay in scratch and can be regenerated.
mkdir -p "$PROJECT_DIR/model_data/cv_results_stratified"
cp "$SCRATCH_DIR"/cv_results_stratified/*preds*.parquet \
   "$PROJECT_DIR/model_data/cv_results_stratified/" 2>/dev/null || true
cp "$SCRATCH_DIR"/cv_results_stratified/predictions_*.parquet \
   "$PROJECT_DIR/model_data/cv_results_stratified/" 2>/dev/null || true
