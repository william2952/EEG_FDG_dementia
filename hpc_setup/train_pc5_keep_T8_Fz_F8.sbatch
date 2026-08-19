#!/bin/bash
# ── Original 10-fold CV on PC5 (non-ICA, thresholded), keeping ONLY T8/Fz/F8 ──
# Every channel except T8, Fz, F8 is zero-masked throughout train/val/holdout
# (via train_cv.py's --mask-channels, same mechanism as the channel-ablation
# studies, just inverted: mask everything EXCEPT the 3 channels we want to keep).
# One array task per fold (0-9), each writing fold_{i}_preds_5_keep_T8_Fz_F8.parquet.
# After all 10 finish, merge with:
#   python merge_cv_results.py --suffix 5_keep_T8_Fz_F8 --output-dir model_data/cv_pc5_keep_T8_Fz_F8
#
# Submit with:
#   sbatch train_pc5_keep_T8_Fz_F8.sbatch            # all 10 folds sequentially, one job
#   sbatch --array=0-9 train_pc5_keep_T8_Fz_F8.sbatch  # each fold as its own array task

#SBATCH --job-name=eeg_pc5_keep3ch
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

# All 19 channels EXCEPT T8, Fz, F8 -> these get zero-masked (i.e. only
# T8/Fz/F8 survive).
python train_cv.py \
  --data-dir      "$PROJECT_DIR/model_data/non_ica_19_channels_thresh" \
  --pca-parquet   "$PROJECT_DIR/model_data/matched_pca_vectors.parquet" \
  --output-dir    "$SCRATCH_DIR/cv_pc5_keep_T8_Fz_F8" \
  --n-pca         1 \
  --pca-offset    4 \
  --suffix        5_keep_T8_Fz_F8 \
  --mask-channels "C3,C4,Cz,F3,F4,F7,Fp1,Fp2,O1,O2,P3,P4,P7,P8,Pz,T7" \
  --accelerator   cuda \
  --num-workers   8 \
  --batch-size    64 \
  "${FOLD_ARGS[@]}"

# Copy the small, durable outputs (predictions) to persistent project storage.
# Skip the bulky per-epoch checkpoint files — those stay in scratch and can be regenerated.
mkdir -p "$PROJECT_DIR/model_data/cv_pc5_keep_T8_Fz_F8"
cp "$SCRATCH_DIR"/cv_pc5_keep_T8_Fz_F8/*_5_keep_T8_Fz_F8.parquet \
   "$PROJECT_DIR/model_data/cv_pc5_keep_T8_Fz_F8/" 2>/dev/null || true
