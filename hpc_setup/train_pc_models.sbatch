#!/bin/bash
# ── Train PC2-PC10 CV models on Bouchet GPU nodes, on the thresholded dataset ──
# Mirrors the PC1 run already done (10 folds, top-3 ensemble), just pointed at
# non_ica_19_channels_thresh instead of non_ica_19_channels, and looping models
# 2 through 10 instead of just 1.
#
# One array task per PC model. Each task trains all 10 folds for that model
# sequentially (via run_cv_parallel.py, called internally by run_all_models.py),
# so 9 models can run concurrently as 9 separate GPU jobs if the partition has
# capacity, rather than 90 individual fold-jobs.
#
# Submit with:
#   sbatch train_pc_models.sbatch

#SBATCH --job-name=eeg_pc_cv
#SBATCH --account=pi_am2359
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=2-10
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

module load miniconda
conda activate eeg

# Persistent storage (never auto-purged) — code, input data, and FINAL outputs live here.
PROJECT_DIR="$HOME/project_pi_am2359/ww488/eeg_project"
# 60-day scratch (auto-deleted after 60 days) — bulky/regenerable per-epoch checkpoints go here.
SCRATCH_DIR="$HOME/scratch_pi_am2359/eeg_scratch"

mkdir -p "$SCRATCH_DIR/checkpoints" logs

cd "$PROJECT_DIR"

# SLURM_ARRAY_TASK_ID (2..10) IS the PC/model number here:
#   model N -> pca-offset N-1 -> suffix N   (see run_all_models.py)
MODEL_NUM="$SLURM_ARRAY_TASK_ID"

python run_all_models.py \
  --data-dir      "$PROJECT_DIR/model_data/non_ica_19_channels_thresh" \
  --pca-parquet   "$PROJECT_DIR/model_data/matched_pca_vectors.parquet" \
  --output-dir    "$SCRATCH_DIR/cv_results_stratified_ica_thresh" \
  --models        "$MODEL_NUM" \
  --n-folds       10 \
  --ensemble-k    3 \
  --accelerator   cuda \
  --num-workers   8 \
  --batch-size    64 \
  --skip-done

# Copy this model's small, durable outputs (predictions/coefficients) to
# persistent project storage. Skip the bulky per-epoch checkpoints — those
# stay in scratch and can be regenerated if needed.
mkdir -p "$PROJECT_DIR/model_data/cv_results_stratified_ica_thresh"
cp "$SCRATCH_DIR"/cv_results_stratified_ica_thresh/*_"${MODEL_NUM}".parquet \
   "$PROJECT_DIR/model_data/cv_results_stratified_ica_thresh/" 2>/dev/null || true
