#!/bin/bash
# ── Train 5 frequency-band-ablation CV models on Bouchet GPU nodes, PC5, ICA-thresh data ──
# Same as train_band_ablation_ica_pc1.sbatch, but targets PC5 and writes to a
# distinct output folder so it can't collide with the ICA PC1 band-ablation results.
#
# Submit with:
#   sbatch train_band_ablation_ica_pc5.sbatch

#SBATCH --job-name=eeg_band_abl_ica_pc5
#SBATCH --account=pi_am2359
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=1-5
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

# Map array index (1-5) -> band name, same order as BANDS in
# run_band_ablation.py / interpretability_analysis.ipynb.
BANDS=(delta theta alpha beta gamma)
BAND="${BANDS[$((SLURM_ARRAY_TASK_ID - 1))]}"

python run_band_ablation.py \
  --data-dir      "$PROJECT_DIR/model_data/post_ica_19_channels_var_thresh" \
  --pca-parquet   "$PROJECT_DIR/model_data/matched_pca_vectors.parquet" \
  --output-dir    "$SCRATCH_DIR/band_ablation_thresh_ica_pc5_results" \
  --pc            5 \
  --bands         "$BAND" \
  --n-folds       10 \
  --ensemble-k    3 \
  --accelerator   cuda \
  --num-workers   8 \
  --batch-size    64 \
  --skip-done

# Copy this band's small, durable outputs (predictions/R² summary) to
# persistent project storage. Skip the bulky per-epoch checkpoints — those
# stay in scratch and can be regenerated if needed.
mkdir -p "$PROJECT_DIR/model_data/band_ablation_thresh_ica_pc5_results"
cp "$SCRATCH_DIR"/band_ablation_thresh_ica_pc5_results/*_"${BAND}".parquet \
   "$PROJECT_DIR/model_data/band_ablation_thresh_ica_pc5_results/" 2>/dev/null || true
