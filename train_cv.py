#!/usr/bin/env python3
import argparse
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from eeg_dataset import EEGSegmentDataset


# ── Model ─────────────────────────────────────────────────────────────────────

class CatPool1d(nn.Module):
    def __init__(self, kernel_size, channels):
        super().__init__()
        self.avg  = nn.AvgPool1d(kernel_size)
        self.max  = nn.MaxPool1d(kernel_size)
        self.proj = nn.Conv1d(channels * 2, channels, 1, bias=False)

    def forward(self, x):
        return self.proj(torch.cat([self.avg(x), self.max(x)], dim=1))


class CatAdaptivePool1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg  = nn.AdaptiveAvgPool1d(1)
        self.max  = nn.AdaptiveMaxPool1d(1)
        self.proj = nn.Conv1d(channels * 2, channels, 1, bias=False)

    def forward(self, x):
        return self.proj(torch.cat([self.avg(x), self.max(x)], dim=1))


class EEGTemporalCNN(nn.Module):
    def __init__(self, n_channels, n_outputs, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 64, 128, padding=64, bias=False),
            nn.BatchNorm1d(64), nn.GELU(), CatPool1d(4, 64), nn.Dropout(dropout),
            nn.Conv1d(64, 128, 16, padding=8, bias=False),
            nn.BatchNorm1d(128), nn.GELU(), CatPool1d(4, 128), nn.Dropout(dropout),
            nn.Conv1d(128, 256, 8, padding=4, bias=False),
            nn.BatchNorm1d(256), nn.GELU(),
            CatAdaptivePool1d(256), nn.Flatten(),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, n_outputs),
        )

    def forward(self, x):
        return self.net(x)


class EEGRegressor(pl.LightningModule):
    def __init__(self, n_channels, n_outputs, lr=3e-4):
        super().__init__()
        self.save_hyperparameters()
        self.net  = EEGTemporalCNN(n_channels, n_outputs)
        self.loss = nn.MSELoss()
        self._train_buf = []
        self._val_buf   = []

    @staticmethod
    def _agg_r2(buf):
        preds   = torch.cat([b[0] for b in buf]).numpy()
        targets = torch.cat([b[1] for b in buf]).numpy()
        subj    = torch.cat([b[2] for b in buf]).numpy()
        agg_p, agg_t = [], []
        for s in np.unique(subj):
            m = subj == s
            agg_p.append(preds[m].mean(0))
            agg_t.append(targets[m].mean(0))
        agg_p, agg_t = np.stack(agg_p), np.stack(agg_t)
        if agg_p.shape[0] < 2:
            return np.full(agg_p.shape[1], np.nan)
        return r2_score(agg_t, agg_p, multioutput="raw_values")

    def training_step(self, batch, _):
        x, y, subj = batch
        pred = self.net(x)
        loss = self.loss(pred, y)
        self._train_buf.append((pred.detach().cpu(), y.detach().cpu(), subj.detach().cpu()))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        r2 = self._agg_r2(self._train_buf)
        for i, v in enumerate(r2):
            self.log(f"train_r2_pc{i+1}", float(v))
        self.log("train_r2_mean", float(r2.mean()), prog_bar=True)
        self._train_buf.clear()

    def validation_step(self, batch, _):
        x, y, subj = batch
        pred = self.net(x)
        self._val_buf.append((pred.detach().cpu(), y.detach().cpu(), subj.detach().cpu()))
        self.log("val_loss", self.loss(pred, y), on_step=True, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        r2 = self._agg_r2(self._val_buf)
        for i, v in enumerate(r2):
            self.log(f"val_r2_pc{i+1}", float(v))
        self.log("val_r2_mean", float(r2.mean()), prog_bar=True)
        self._val_buf.clear()

    def predict_step(self, batch, _):
        x, y, subj = batch
        return self.net(x).cpu(), y.cpu(), subj.cpu()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-2)


# ── Stratified fold creation ──────────────────────────────────────────────────

def make_stratified_folds(subjects, pca_vals, n_folds, n_bins, rng):
    """
    Build n_folds holdout sets by sampling 1 subject per PC1 quantile bin.
    Subjects sort into n_bins equal-count bins; the first n_folds subjects drawn
    from each bin (one per fold) become holdout; the rest go to always_train.
    Returns (folds, always_train) where folds[i] is an ndarray of subject ids.
    """
    pc1 = np.array([float(pca_vals.loc[s].iloc[0]) for s in subjects])
    sorted_idx = np.argsort(pc1)
    bins = np.array_split(sorted_idx, n_bins)

    fold_lists = [[] for _ in range(n_folds)]
    always_train_idx = []
    for bin_idx in bins:
        perm = rng.permutation(len(bin_idx))
        shuffled_bin = bin_idx[perm]
        for fold_i in range(min(n_folds, len(shuffled_bin))):
            fold_lists[fold_i].append(subjects[shuffled_bin[fold_i]])
        always_train_idx.extend(shuffled_bin[n_folds:].tolist())

    folds = [np.array(f) for f in fold_lists]
    always_train = subjects[np.array(always_train_idx, dtype=int)] if always_train_idx else np.array([])
    return folds, always_train


# ── CV loop ───────────────────────────────────────────────────────────────────

def _load_pca_and_subjects(args):
    pca_vals_raw = pd.read_parquet(args.pca_parquet)
    pca_vals = pd.DataFrame(
        np.stack(pca_vals_raw["vector"].to_numpy()),
        index=pca_vals_raw["subject"].astype(str),
    )
    all_subjects = np.array(sorted(
        set(pca_vals.index) & {p.stem for p in Path(args.data_dir).glob("*.parquet")}
    ))
    return pca_vals, all_subjects


def _build_folds(args, pca_vals, all_subjects):
    rng = np.random.default_rng(args.seed)
    return make_stratified_folds(all_subjects, pca_vals, args.n_folds, args.n_bins, rng)


def train_one_fold(args, fold_idx, folds, always_train, pca_vals):
    """Train and evaluate a single fold. Saves to fold_{fold_idx}_preds_{suffix}.parquet."""
    sfx = f"_{args.suffix}"
    ckpt_dir = Path(args.output_dir) / f"checkpoints{sfx}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Fold {fold_idx}  |  holdout n={len(folds[fold_idx])}")
    print(f"{'='*60}")

    test_subjects = folds[fold_idx]
    other_holdout = np.concatenate([folds[i] for i in range(args.n_folds) if i != fold_idx])
    non_test = np.concatenate([other_holdout, always_train])
    print(f"  train+val pool: {len(non_test)} subjects")

    rng_fold = np.random.default_rng(args.seed + fold_idx)
    perm = rng_fold.permutation(len(non_test))
    n_val = max(1, int(0.1 * len(non_test)))
    val_subjects   = non_test[perm[:n_val]]
    train_subjects = non_test[perm[n_val:]]

    train_ds = EEGSegmentDataset(
        args.data_dir, pca_vals,
        n_pca_components=args.n_pca, n_segments=args.n_seg, n_samples=args.n_samples,
        pca_offset=args.pca_offset,
        subject_ids=train_subjects,
        mask_channels=args.mask_channels,
        mask_band=args.mask_band,
    )
    val_ds = EEGSegmentDataset(
        args.data_dir, pca_vals,
        n_pca_components=args.n_pca, n_segments=args.n_seg, n_samples=args.n_samples,
        pca_offset=args.pca_offset,
        subject_ids=val_subjects,
        target_mean=train_ds.target_mean, target_std=train_ds.target_std,
        mask_channels=args.mask_channels,
        mask_band=args.mask_band,
    )
    test_ds = EEGSegmentDataset(
        args.data_dir, pca_vals,
        n_pca_components=args.n_pca, n_segments=args.n_seg, n_samples=args.n_samples,
        pca_offset=args.pca_offset,
        subject_ids=test_subjects,
        target_mean=train_ds.target_mean, target_std=train_ds.target_std,
        mask_channels=args.mask_channels,
        mask_band=args.mask_band,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = EEGRegressor(n_channels=19, n_outputs=args.n_pca)

    callbacks = [
        EarlyStopping(monitor="val_r2_mean", mode="max", patience=args.patience),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="val_r2_mean",
            mode="max",
            filename=f"fold{fold_idx}-top{{epoch:02d}}-{{val_r2_mean:.4f}}",
            save_top_k=args.ensemble_k,
        ),
    ]
    logger = TensorBoardLogger(
        save_dir=str(Path(args.output_dir) / f"logs{sfx}"),
        name=f"fold{fold_idx}",
        version=0,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=1,
        log_every_n_steps=1,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_loader, val_loader)

    ckpt_paths = sorted(callbacks[1].best_k_models.keys())
    pred_trainer = pl.Trainer(accelerator=args.accelerator, devices=1,
                              logger=False, enable_progress_bar=False)
    all_preds = []
    for ckpt_path in ckpt_paths:
        m = EEGRegressor.load_from_checkpoint(ckpt_path)
        b = pred_trainer.predict(m, test_loader)
        all_preds.append(torch.cat([x[0] for x in b]).numpy())
    print(f"  Ensemble of {len(ckpt_paths)} checkpoints")
    preds   = np.mean(all_preds, axis=0)
    targets = torch.cat([x[1] for x in b]).numpy()
    subjs   = torch.cat([x[2] for x in b]).numpy()

    fold_records = []
    agg_p, agg_t = [], []
    for s in np.unique(subjs):
        mask = subjs == s
        pred_denorm   = preds[mask].mean(0)   * train_ds.target_std + train_ds.target_mean
        target_denorm = targets[mask].mean(0) * train_ds.target_std + train_ds.target_mean
        fold_records.append({
            "fold":     fold_idx,
            "subject":  str(s),
            "pred_pc1": float(pred_denorm[0]),
            "true_pc1": float(target_denorm[0]),
        })
        agg_p.append(preds[mask].mean(0))
        agg_t.append(targets[mask].mean(0))

    fold_r2 = r2_score(np.stack(agg_t), np.stack(agg_p), multioutput="raw_values")
    print(f"  Fold {fold_idx} holdout R²: {fold_r2}")

    pred_path = Path(args.output_dir) / f"fold_{fold_idx}_preds{sfx}.parquet"
    pd.DataFrame(fold_records).to_parquet(pred_path, index=False)
    print(f"  Saved predictions → {pred_path}")
    return fold_r2


def run_cv(args):
    sfx = f"_{args.suffix}"
    ckpt_dir = Path(args.output_dir) / f"checkpoints{sfx}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    pca_vals, all_subjects = _load_pca_and_subjects(args)
    print(f"Total subjects: {len(all_subjects)}")

    folds, always_train = _build_folds(args, pca_vals, all_subjects)
    holdout_sizes = [len(f) for f in folds]
    print(f"Stratified folds: {args.n_folds} folds × {args.n_bins} PC1 bins → "
          f"holdout sizes {holdout_sizes}  (always-train: {len(always_train)})")
    for i, f in enumerate(folds):
        print(f"  fold {i}: {len(f)} subjects")

    # ── Single-fold mode (used by run_cv_parallel.py) ─────────────────────────
    if args.fold_idx is not None:
        train_one_fold(args, args.fold_idx, folds, always_train, pca_vals)
        return

    # Load existing predictions if resuming
    pred_path = Path(args.output_dir) / f"predictions{sfx}.parquet"
    if args.start_fold > 0 and pred_path.exists():
        fold_records = pd.read_parquet(pred_path).to_dict("records")
        print(f"Loaded {len(fold_records)} existing predictions from {pred_path}")
    else:
        fold_records = []

    fold_r2s = []

    # Recover R² for skipped folds without loading EEG data
    for fold_idx in range(args.start_fold):
        test_subjects = folds[fold_idx]
        other_holdout = np.concatenate([folds[i] for i in range(args.n_folds) if i != fold_idx])
        non_test = np.concatenate([other_holdout, always_train])

        rng_fold = np.random.default_rng(args.seed + fold_idx)
        perm = rng_fold.permutation(len(non_test))
        n_val = max(1, int(0.1 * len(non_test)))
        train_subjects = non_test[perm[n_val:]]

        train_ids = sorted(set(pca_vals.index) & {str(s) for s in train_subjects})
        raw = np.stack([pca_vals.loc[s].to_numpy(dtype=np.float32)[args.pca_offset:args.pca_offset + args.n_pca] for s in train_ids])
        target_mean = raw.mean(0).astype(np.float32)
        target_std  = (raw.std(0) + 1e-8).astype(np.float32)

        test_ds = EEGSegmentDataset(
            args.data_dir, pca_vals,
            n_pca_components=args.n_pca, n_segments=args.n_seg, n_samples=args.n_samples,
            pca_offset=args.pca_offset,
            subject_ids=test_subjects, target_mean=target_mean, target_std=target_std,
            mask_channels=args.mask_channels,
            mask_band=args.mask_band,
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        ckpt_paths = sorted(ckpt_dir.glob(f"fold{fold_idx}-top*.ckpt"))
        pred_trainer = pl.Trainer(accelerator=args.accelerator, devices=1,
                                  logger=False, enable_progress_bar=False)
        all_preds = []
        for ckpt_path in ckpt_paths:
            m = EEGRegressor.load_from_checkpoint(ckpt_path)
            b = pred_trainer.predict(m, test_loader)
            all_preds.append(torch.cat([x[0] for x in b]).numpy())
        preds   = np.mean(all_preds, axis=0)
        targets = torch.cat([x[1] for x in b]).numpy()
        subjs   = torch.cat([x[2] for x in b]).numpy()

        agg_p, agg_t = [], []
        for s in np.unique(subjs):
            mask = subjs == s
            agg_p.append(preds[mask].mean(0))
            agg_t.append(targets[mask].mean(0))

        fold_r2 = r2_score(np.stack(agg_t), np.stack(agg_p), multioutput="raw_values")
        fold_r2s.append(fold_r2)
        print(f"Recovered fold {fold_idx} holdout R² (ensemble {len(ckpt_paths)}): {fold_r2}")

    # Train remaining folds sequentially
    for fold_idx in range(args.start_fold, args.n_folds):
        fold_r2 = train_one_fold(args, fold_idx, folds, always_train, pca_vals)
        fold_r2s.append(fold_r2)

        # Merge this fold's per-fold file into the combined predictions.parquet
        fold_pred_path = Path(args.output_dir) / f"fold_{fold_idx}_preds{sfx}.parquet"
        fold_records.extend(pd.read_parquet(fold_pred_path).to_dict("records"))
        pd.DataFrame(fold_records).to_parquet(pred_path, index=False)

        gc.collect()
        if args.accelerator == "mps":
            torch.mps.empty_cache()

    print("\nAll folds complete.")
    fold_r2_arr = np.array([r[0] for r in fold_r2s])
    print(f"Per-fold R²: {np.round(fold_r2_arr, 4)}")
    print(f"Mean ± SD:   {fold_r2_arr.mean():.4f} ± {fold_r2_arr.std():.4f}")


def main():
    parser = argparse.ArgumentParser(description="EEG cross-validation training")
    parser.add_argument("--data-dir",    default="model_data/non_ica_19_channels")
    parser.add_argument("--pca-parquet", default="model_data/matched_pca_vectors.parquet")
    parser.add_argument("--output-dir",  default="model_data/cv_results_stratified")
    parser.add_argument("--n-folds",     type=int, default=10)
    parser.add_argument("--n-bins",      type=int, default=20,
                        help="PC1 quantile bins; 1 subject per bin is sampled into each holdout set.")
    parser.add_argument("--n-pca",       type=int, default=1)
    parser.add_argument("--pca-offset",  type=int, default=0,
                        help="Index of the first PCA component to use as target (0=PC1, 1=PC2, ...).")
    parser.add_argument("--n-seg",       type=int, default=1)
    parser.add_argument("--n-samples",   type=int, default=100)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--max-epochs",  type=int, default=50)
    parser.add_argument("--patience",    type=int, default=10)
    parser.add_argument("--batch-size",  type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--accelerator", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--ensemble-k",  type=int, default=3,
                        help="Number of top-k checkpoints to save and average at holdout time.")
    parser.add_argument("--start-fold",  type=int, default=0,
                        help="Resume from this fold (0-indexed). Loads predictions_{suffix}.parquet for prior folds.")
    parser.add_argument("--fold-idx",    type=int, default=None,
                        help="Run exactly this one fold and exit. Used by run_cv_parallel.py.")
    parser.add_argument("--suffix",      type=str, default="1",
                        help="Output name suffix. Model 1 (PC1) → '1', model 2 (PC2) → '2', etc. "
                             "Controls names of checkpoints_N, logs_N, predictions_N.parquet.")
    parser.add_argument("--mask-channels", type=str, default=None,
                        help="Comma-separated channel names to zero out (ablate) in every "
                             "window, e.g. 'C3,P3,F3,Cz'. Applied identically to train/val/holdout.")
    parser.add_argument("--mask-band-lo", type=float, default=None,
                        help="Low edge (Hz) of a frequency band to bandstop-filter out of "
                             "every channel, e.g. 0.5 for delta. Requires --mask-band-hi too.")
    parser.add_argument("--mask-band-hi", type=float, default=None,
                        help="High edge (Hz) of a frequency band to bandstop-filter out of "
                             "every channel, e.g. 4 for delta. Requires --mask-band-lo too.")
    parser.add_argument("--sfreq", type=float, default=256,
                        help="Sampling rate (Hz) of the cached segments, used for --mask-band-*.")
    args = parser.parse_args()
    args.mask_channels = ([c.strip() for c in args.mask_channels.split(",")]
                          if args.mask_channels else None)
    if (args.mask_band_lo is None) != (args.mask_band_hi is None):
        raise SystemExit("--mask-band-lo and --mask-band-hi must be given together.")
    args.mask_band = ((args.mask_band_lo, args.mask_band_hi)
                      if args.mask_band_lo is not None else None)
    run_cv(args)


if __name__ == "__main__":
    main()
