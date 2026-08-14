# EEG_PET_dementia

Multimodal deep-learning project combining **EEG** and **FDG-PET** imaging for dementia-related analysis, developed with data from Mayo Clinic.

## Goal

Predict subject-level FDG-PET PCA components from minimally-preprocessed EEG.

## Pipeline

- **Preprocessing**: FIR filtering, average referencing, 10-second non-overlapping epochs, per-recording/per-channel z-scoring. An ICA-thresholded variant of the pipeline is also available (`*_thresh` / `*_ica*` data directories) for comparison against the non-ICA baseline.
- **Targets**: FDG-PET PCA vectors, standardized per-PC using train-set stats.
- **Model**: 1D temporal CNN (`EEGTemporalCNN`, ~493K params, channels 64/128/256, dropout 0.2), defined in [train_cv.py](train_cv.py). See [architecture_schematic.ipynb](architecture_schematic.ipynb) for a block-diagram walkthrough.
- **Training**:
  - Multi-window: each subject contributes every non-overlapping `n_segments`-long window (~120 windows/subject for `n_segments=1`).
  - Stratified 10-fold CV: subjects are quantile-binned on PC1 and one subject per bin is held out per fold, so folds are subject-level and target-balanced (`--n-bins`, `--n-folds`).
  - Loss: MSE on standardized PCA targets. Optimizer: AdamW (lr 3e-4, weight decay 1e-2).
  - Early stopping on aggregated val R² (patience 10), top-k checkpoint ensembling at holdout time (`--ensemble-k`).
- **Evaluation**: per-window predictions are averaged within each test subject before computing R² across subjects — the standard subject-level metric for this kind of task.
- **Ablation studies**: channel-ablation (`run_channel_ablation.py`) zeroes each of the 19 electrodes plus its 3 nearest 10-20 neighbours; band-ablation (`run_band_ablation.py`) bandstop-filters each of delta/theta/alpha/beta/gamma out of every channel. Both re-run the full stratified CV per ablation and support the `_ica`/`_thresh` data variants and arbitrary target PCs (`--pc`).

## Result

Aggregated subject-level val R² ≈ **0.28** on PC1 (`n_segments=1`, 10s windows). Reproduces the prior baseline with cleaner methodology (reflect-padded segments, subject-aggregated metric, early stopping).

## Layout

```
train_cv.py                    single-fold CV training entry point (also used by ablation scripts)
run_cv_parallel.py              orchestrates all folds for one PC model (parallel on CPU, sequential on MPS/CUDA)
run_all_models.py               trains all PC models (PC1-PC9) sequentially via run_cv_parallel.py
run_channel_ablation.py         channel-ablation study (19 channels x 10-fold CV) for a given PC
run_band_ablation.py            frequency-band-ablation study (5 bands x 10-fold CV) for a given PC
eeg_dataset.py                  EEGSegmentDataset — windowing, channel masking, band masking
merge_cv_results.py             aggregates per-fold prediction parquets into overall CV metrics
check_ablation_status.py        scans ablation output dirs, reports completion/merge status per suffix
baseline_ML_model.ipynb         classical baseline: band-power features + Ridge regression
data_processing.ipynb           EEG preprocessing
model_lightning.ipynb           single-split training + evaluation (non-CV)
model_lightning_cv.ipynb        CV training + evaluation
model_lightning_ica.ipynb       CV training on ICA-thresholded data
model_lightning_separate_pc.ipynb  per-PC model variant
interpretability_analysis.ipynb channel/band importance analysis (source of the ablation neighbour maps)
architecture_schematic.ipynb    EEGTemporalCNN block-diagram schematic
hpc_setup/                      SLURM (Yale Bouchet cluster) submission scripts + SETUP.md
model_data/
  non_ica_19_channels/           preprocessed EEG parquets, non-ICA, 19-channel montage
  post_ica_19_channels*/         ICA-cleaned EEG variants (thresholded/variance-filtered)
  matched_pca_vectors.parquet    FDG-PET PCA targets
  cv_results_stratified*/        per-fold + merged CV predictions, non-ICA / ICA-thresh
  channel_ablation_*_results/    channel-ablation CV outputs
  band_ablation_*_results/       band-ablation CV outputs
  pc_r2_analysis/                aggregated R² across PC models
  models*/, lightning_logs*/     checkpoints and TensorBoard logs per run
old_notebooks/                   archived earlier experiments
```

## Run

Single model, all folds:

```bash
python run_cv_parallel.py --suffix 1 --pca-offset 0   # PC1
```

All PC models (PC1-PC9):

```bash
python run_all_models.py
```

Aggregate results and check status:

```bash
python merge_cv_results.py --output-dir model_data/cv_results_stratified
python check_ablation_status.py
```

Ablation studies:

```bash
python run_channel_ablation.py --pc 1
python run_band_ablation.py --pc 1
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir model_data/lightning_logs --port 6006
```

For training on Yale's Bouchet GPU cluster, see [hpc_setup/SETUP.md](hpc_setup/SETUP.md).
