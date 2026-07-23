# Running train_cv.py on Yale's Bouchet GPU cluster

One correction before starting: `model_data/` on your Mac is 477GB total, but
`train_cv.py`'s defaults only touch two things — `model_data/non_ica_19_channels`
(51GB) and `model_data/matched_pca_vectors.parquet` (2MB). Only transfer those,
not the whole `model_data/` tree, unless you specifically need the other variants
(ICA versions, older lightning_logs, etc.) on the cluster too.

## 0. Group setup — confirmed

```bash
groups              # → ww488 pi_am2359
slurm_checkup.sh     # → pi_am2359 [ adam mecca ] (default)
```

Your PI account is `pi_am2359`. That's already filled into `train_fold.sbatch`
below — no more placeholders to substitute.

## 1. Globus — confirmed

Two collections are set up and visible in the Globus File Manager:
- `ww488_mac` — your Mac (Private Mapped Collection, GCP)
- `Yale CRC Bouchet HA` — Bouchet's cluster-side collection

## 2. Transfer data and code to Bouchet

In the Globus File Manager, put `ww488_mac` in the left panel and
`Yale CRC Bouchet HA` in the right panel. On the Bouchet side, navigate into
`project` (your home directory has a `project` shortcut pointing at
`/nfs/roberts/project/pi_am2359/...`) and create a working folder, e.g.
`eeg_project`.

Select and transfer from `ww488_mac` into `~/project/eeg_project/` on Bouchet:
- `model_data/non_ica_19_channels/` (51GB — this is the bulk of the transfer)
- `model_data/matched_pca_vectors.parquet`
- `train_cv.py`, `eeg_dataset.py`, `run_cv_parallel.py`, `merge_cv_results.py`

Select the files/folders on the left, click "Transfer or Sync to..." in the
right-hand panel menu, confirm the destination path on the right, and hit
Start. For the 51GB data folder this will take a while — Globus runs it in
the background, so the browser tab doesn't need to stay open; check progress
under "Activity" in the left nav.

Code files are tiny — you can also just drag-and-drop them via the Open
OnDemand file browser instead, if that's easier once Globus is busy moving
the big data transfer.

**Why `project` and not `scratch`:** `project` (4TiB/group) is never
auto-purged. `scratch` — confusingly, its symlink is named
`project_pi_am2359` — silently deletes anything untouched for 60 days. Put
your input data and code in `project`; only use `scratch` for disposable
intermediate output (checkpoints) as shown in the sbatch script below.

## 3. Build the conda environment (one-time)

Never build on the login node — grab an interactive GPU session first:

```bash
salloc -p gpu_devel --gpus=1 -t 1:00:00
module load miniconda
conda create -n eeg python=3.11 -y
conda activate eeg
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning scikit-learn pandas numpy pyarrow
```

Verify the GPU is visible before exiting the session:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 4. Submit the training job

The script at `hpc_setup/train_fold.sbatch` (in this repo) is set up to:
- read code + input data from `project` (persistent)
- write checkpoints to `scratch` (disposable, 60-day purge — fine, since
  they're regenerable)
- copy the small final prediction/coefficient parquet files back to
  `project` when the job finishes, so nothing you actually need is ever at
  risk of the scratch purge

Edit the two `<PI_NETID>` placeholders in the script, then transfer it to
`~/project/eeg_project/` on Bouchet and submit:

```bash
# All 10 folds sequentially in one job:
sbatch train_fold.sbatch

# Or, faster wall-clock: one job per fold, running in parallel:
sbatch --array=0-9 train_fold.sbatch
```

The array form is worth using if your PI group has enough GPU allocation for
several jobs to run concurrently — check with `sinfo -p gpu` for node
availability.

## 5. Monitor

```bash
squeue --me                 # job status
tail -f logs/eeg_cv_*.out   # live training output
jobstats <jobid>            # after completion, confirms the GPU was actually used
```

## 6. Retrieve the trained model outputs

Once folds finish, `~/project/eeg_project/model_data/cv_results_stratified/`
holds the final `fold_*_preds_*.parquet` and `predictions_*.parquet` files —
these are your model coefficients/predictions, and they're small (KBs–MBs,
not GBs). Pull just these back to your Mac via the Open OnDemand file
browser (drag-and-drop) or Globus, then run `merge_cv_results.py` and your
local analysis notebooks against them as usual.

You do **not** need to bring the 51GB of raw EEG data or the per-epoch
checkpoints back — those stay on the cluster.
