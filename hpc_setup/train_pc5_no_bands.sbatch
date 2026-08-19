#!/bin/bash
# ── Original 10-fold CV on PC5 (non-ICA, thresholded), with all 5 standard ──
# ── EEG bands (delta+theta+alpha+beta+gamma) removed from every channel ─────
# The 5 standard bands are contiguous (delta 0.5-4, theta 4-8, alpha 8-12,
# beta 12-30, gamma 30-45 Hz), so removing all 5 at once is just a single
# bandstop filter over [0.5, 45] Hz -- no need to apply 5 separate filters.
# What's left in the signal is only <0.5 Hz drift and >45 Hz content.
# One array task per fold (0-9), each writing fold_{i}_preds_5_no_bands.parquet.
# After all 10 finish, merge with:
#   python merge_cv_results.py --suffix 5_no_bands --output-dir model_data/cv_pc5_no_bands
#
# Submit with:
#   sbatch train_pc5_no_bands.sbatch            # all 10 folds sequentially, one job
#   sbatch --array=0-9 train_pc5_no_bands.sbatch  # each fold as its own array task

#SBATCH --job-name=eeg_pc5_no_bands
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
  --data-dir      "$PROJECT_DIR/model_data/non_ica_19_channels_thresh" \
  --pca-parquet   "$PROJECT_DIR/model_data/matched_pca_vectors.parquet" \
  --output-dir    "$SCRATCH_DIR/cv_pc5_no_bands" \
  --n-pca         1 \
  --pca-offset    4 \
  --suffix        5_no_bands \
  --mask-band-lo  0.5 \
  --mask-band-hi  45 \
  --accelerator   cuda \
  --num-workers   8 \
  --batch-size    64 \
  "${FOLD_ARGS[@]}"

# Copy the small, durable outputs (predictions) to persistent project storage.
# Skip the bulky per-epoch checkpoint files — those stay in scratch and can be regenerated.
mkdir -p "$PROJECT_DIR/model_data/cv_pc5_no_bands"
cp "$SCRATCH_DIR"/cv_pc5_no_bands/*_5_no_bands.parquet \
   "$PROJECT_DIR/model_data/cv_pc5_no_bands/" 2>/dev/null || true
