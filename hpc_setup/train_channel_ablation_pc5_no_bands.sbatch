#!/bin/bash
# ── Channel-importance ablation, PC5 non-ICA, with all 5 standard bands removed ──
# Same 19-channel retrained-ablation study as train_channel_ablation_pc5.sbatch,
# but every run ALSO has all 5 standard EEG bands (0.5-45 Hz, contiguous so it's
# a single bandstop) stripped out first, via run_channel_ablation.py's new
# --mask-band-lo/--mask-band-hi passthrough. So each channel's model sees:
# that channel + its 3 neighbours zeroed out, AND the entire standard EEG
# spectrum bandstopped, everywhere (train/val/holdout).
#
# Submit with:
#   sbatch train_channel_ablation_pc5_no_bands.sbatch

#SBATCH --job-name=eeg_ch_abl_pc5_no_bands
#SBATCH --account=pi_am2359
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=1-19
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

# Map array index (1-19) -> central channel name, alphabetical order
# (matches CH_NAMES in interpretability_analysis.ipynb).
CHANNELS=(C3 C4 Cz F3 F4 F7 F8 Fp1 Fp2 Fz O1 O2 P3 P4 P7 P8 Pz T7 T8)
CHANNEL="${CHANNELS[$((SLURM_ARRAY_TASK_ID - 1))]}"

python run_channel_ablation.py \
  --data-dir      "$PROJECT_DIR/model_data/non_ica_19_channels_thresh" \
  --pca-parquet   "$PROJECT_DIR/model_data/matched_pca_vectors.parquet" \
  --output-dir    "$SCRATCH_DIR/channel_ablation_thresh_pc5_no_bands_results" \
  --pc            5 \
  --channels      "$CHANNEL" \
  --mask-band-lo  0.5 \
  --mask-band-hi  45 \
  --n-folds       10 \
  --ensemble-k    3 \
  --accelerator   cuda \
  --num-workers   8 \
  --batch-size    64 \
  --skip-done

# Copy this channel's small, durable outputs (predictions/R² summary) to
# persistent project storage. Skip the bulky per-epoch checkpoints — those
# stay in scratch and can be regenerated if needed.
mkdir -p "$PROJECT_DIR/model_data/channel_ablation_thresh_pc5_no_bands_results"
cp "$SCRATCH_DIR"/channel_ablation_thresh_pc5_no_bands_results/*_"${CHANNEL}".parquet \
   "$PROJECT_DIR/model_data/channel_ablation_thresh_pc5_no_bands_results/" 2>/dev/null || true
