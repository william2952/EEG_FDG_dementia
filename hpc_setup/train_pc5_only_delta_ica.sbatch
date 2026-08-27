#!/bin/bash
# ── Original 10-fold CV on PC5 (ICA-thresholded), ALL channels, but only ──
# ── the delta band (0.5-4 Hz) kept — everything else bandpass-filtered out ──
# Same as train_pc5_only_delta.sbatch, but on post_ica_19_channels_var_thresh.
# One array task per fold (0-9), each writing fold_{i}_preds_5_only_delta_ica.parquet.
# After all 10 finish, merge with:
#   python merge_cv_results.py --suffix 5_only_delta_ica --output-dir model_data/cv_pc5_only_delta_ica
#
# Submit with:
#   sbatch --array=0-9 train_pc5_only_delta_ica.sbatch

#SBATCH --job-name=eeg_pc5_only_delta_ica
#SBATCH --account=pi_am2359
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --exclude=a1125u14n01

set -euo pipefail

module load miniconda
eval "$(conda shell.bash hook)"
conda activate eeg

# Persistent storage (never auto-purged) — code, input data, and FINAL outputs live here.
PROJECT_DIR="$HOME/project_pi_am2359/ww488/eeg_project"
# 60-day scratch (auto-deleted after 60 days) — bulky/regenerable per-epoch checkpoints go here.
SCRATCH_DIR="$HOME/scratch_pi_am2359/ww488/eeg_scratch"

mkdir -p "$SCRATCH_DIR/checkpoints" logs
cd "$PROJECT_DIR"

# If submitted as a job array (--array=0-9), SLURM_ARRAY_TASK_ID selects one fold.
# Otherwise (no --array), run every fold sequentially in this single job.
FOLD_ARGS=()
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  FOLD_ARGS=(--fold-idx "$SLURM_ARRAY_TASK_ID")
fi

python train_cv.py \
  --data-dir      "$PROJECT_DIR/model_data/post_ica_19_channels_var_thresh" \
  --pca-parquet   "$PROJECT_DIR/model_data/matched_pca_vectors.parquet" \
  --output-dir    "$SCRATCH_DIR/cv_pc5_only_delta_ica" \
  --n-pca         1 \
  --pca-offset    4 \
  --suffix        5_only_delta_ica \
  --keep-band-lo  0.5 \
  --keep-band-hi  4 \
  --accelerator   cuda \
  --num-workers   8 \
  --batch-size    64 \
  "${FOLD_ARGS[@]}"

# Copy the small, durable outputs (predictions) to persistent project storage.
# Skip the bulky per-epoch checkpoint files — those stay in scratch and can be regenerated.
mkdir -p "$PROJECT_DIR/model_data/cv_pc5_only_delta_ica"
cp "$SCRATCH_DIR"/cv_pc5_only_delta_ica/*_5_only_delta_ica.parquet \
   "$PROJECT_DIR/model_data/cv_pc5_only_delta_ica/" 2>/dev/null || true
