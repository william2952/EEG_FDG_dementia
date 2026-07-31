#!/bin/bash
# ── Train 19 channel-ablation CV models on Bouchet GPU nodes, PC1 only ──
# Each array task ablates one central channel + its 3 nearest neighbours
# (hardcoded neighbour map — see run_channel_ablation.py) by zeroing them in
# every window across train/val/holdout, then runs the full 10-fold
# stratified CV (top-3 ensemble) on non_ica_19_channels_thresh data.
#
# Submit with:
#   sbatch train_channel_ablation.sbatch

#SBATCH --job-name=eeg_ch_abl
#SBATCH --account=pi_am2359
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=1-19
#SBATCH --output=logs/%x_%A_%a.out

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

# Map array index (1-19) -> central channel name, alphabetical order
# (matches CH_NAMES in interpretability_analysis.ipynb).
CHANNELS=(C3 C4 Cz F3 F4 F7 F8 Fp1 Fp2 Fz O1 O2 P3 P4 P7 P8 Pz T7 T8)
CHANNEL="${CHANNELS[$((SLURM_ARRAY_TASK_ID - 1))]}"

python run_channel_ablation.py \
  --data-dir      "$PROJECT_DIR/model_data/non_ica_19_channels_thresh" \
  --pca-parquet   "$PROJECT_DIR/model_data/matched_pca_vectors.parquet" \
  --output-dir    "$SCRATCH_DIR/channel_ablation_results" \
  --channels      "$CHANNEL" \
  --n-folds       10 \
  --ensemble-k    3 \
  --accelerator   cuda \
  --num-workers   8 \
  --batch-size    64 \
  --skip-done

# Copy this channel's small, durable outputs (predictions/R² summary) to
# persistent project storage. Skip the bulky per-epoch checkpoints — those
# stay in scratch and can be regenerated if needed.
mkdir -p "$PROJECT_DIR/model_data/channel_ablation_results"
cp "$SCRATCH_DIR"/channel_ablation_results/*_"${CHANNEL}".parquet \
   "$PROJECT_DIR/model_data/channel_ablation_results/" 2>/dev/null || true
