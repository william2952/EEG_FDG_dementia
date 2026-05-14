# EEG_PET_dementia

Multimodal deep-learning project combining **EEG** and **FDG-PET** imaging for dementia-related analysis, developed with data from Mayo Clinic.

## Goal

Predict subject-level FDG-PET PCA components from minimally-preprocessed EEG.

## Pipeline

- **Preprocessing**: FIR filtering, average referencing, 10-second non-overlapping epochs, per-recording/per-channel z-scoring.
- **Targets**: FDG-PET PCA vectors, standardized per-PC using train-set stats.
- **Training**:
  - Multi-window: each subject contributes every non-overlapping `n_segments`-long window (~120 windows/subject for `n_segments=1`).
  - 80/20 subject-level train/test split — no subject appears in both.
  - Model: 1D temporal CNN, ~493K params (channels 64/128/256, dropout 0.2).
  - Loss: MSE on standardized PCA targets. Optimizer: AdamW (lr 3e-4, weight decay 1e-2).
  - Early stopping on aggregated val R² (patience 10).
- **Evaluation**: per-window predictions are averaged within each test subject before computing R² across subjects — the standard subject-level metric for this kind of task.

## Result

Aggregated subject-level val R² ≈ **0.28** on PC1 (`n_segments=1`, 10s windows). Reproduces the prior baseline with cleaner methodology (reflect-padded segments, subject-aggregated metric, early stopping).

## Layout

```
model_data/
  non_ica/                       preprocessed EEG parquets (one per subject)
  matched_pca_vectors.parquet    FDG-PET PCA targets
  lightning_logs/version_N/      TensorBoard logs per run
  models/best-vN.ckpt            best (max val_r2_mean) checkpoint per run
  models/last-vN.ckpt            final-epoch checkpoint per run
data_processing.ipynb            EEG preprocessing
model_lightning.ipynb            training + evaluation
old_notebooks/                   archived earlier experiments
```

## Run

Launch training cells in `model_lightning.ipynb`. Monitor with:

```bash
tensorboard --logdir model_data/lightning_logs --port 6006
```

