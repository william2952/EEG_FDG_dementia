from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class ResNet1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        n_outputs: int = 1,
        base_channels: int = 32,
        blocks_per_stage: Sequence[int] = (2, 2, 2),
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_ch = base_channels

        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels,
                base_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(base_channels, blocks_per_stage[0], stride=1, kernel_size=kernel_size)
        self.stage2 = self._make_stage(base_channels * 2, blocks_per_stage[1], stride=2, kernel_size=kernel_size)
        self.stage3 = self._make_stage(base_channels * 4, blocks_per_stage[2], stride=2, kernel_size=kernel_size)

        self.head_bn = nn.BatchNorm1d(base_channels * 4)
        self.head_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(base_channels * 4, n_outputs)

    def _make_stage(self, out_ch: int, n_blocks: int, stride: int, kernel_size: int) -> nn.Sequential:
        blocks = [ResBlock1D(self.in_ch, out_ch, kernel_size=kernel_size, stride=stride)]
        self.in_ch = out_ch
        for _ in range(1, n_blocks):
            blocks.append(ResBlock1D(self.in_ch, out_ch, kernel_size=kernel_size, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = x.mean(dim=-1)
        x = self.head_bn(x)
        x = self.head_dropout(x)
        return self.fc(x)


class EEGRegressionLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        target_mean: Optional[np.ndarray] = None,
        target_std: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = nn.MSELoss()

        self.target_mean = None
        self.target_std = None
        if target_mean is not None and target_std is not None:
            self.target_mean = torch.tensor(target_mean, dtype=torch.float32).view(1, -1)
            self.target_std = torch.tensor(target_std, dtype=torch.float32).view(1, -1)

        self.save_hyperparameters(ignore=["model", "target_mean", "target_std"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        if self.target_mean is None or self.target_std is None:
            return x
        mean = self.target_mean.to(device=x.device, dtype=x.dtype)
        std = self.target_std.to(device=x.device, dtype=x.dtype)
        return x * std + mean

    def _shared_step(self, batch, stage: str):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))

        y_hat_eval = self._denorm(y_hat)
        y_eval = self._denorm(y)
        ss_res = torch.sum((y_eval - y_hat_eval) ** 2)
        ss_tot = torch.sum((y_eval - torch.mean(y_eval)) ** 2) + 1e-8
        r2 = 1.0 - (ss_res / ss_tot)
        self.log(f"{stage}_r2", r2, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))

        return loss

    def training_step(self, batch, batch_idx: int):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx: int):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx: int):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class EEGDataModule(pl.LightningDataModule):
    """
    Wraps your existing Dataset (e.g., CustomDataLoader from model_v1.ipynb)
    and applies the same stratified split strategy you use in the notebook.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        train_frac: float = 0.8,
        random_state: int = 42,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last_train: bool = True,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_frac = train_frac
        self.random_state = random_state
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last_train = drop_last_train

        self.train_ds: Optional[Subset] = None
        self.val_ds: Optional[Subset] = None

    def setup(self, stage: Optional[str] = None):
        if self.train_ds is not None and self.val_ds is not None:
            return

        if not hasattr(self.dataset, "subjects") or not hasattr(self.dataset, "subject_to_pca"):
            raise ValueError(
                "Dataset must expose `subjects` and `subject_to_pca` for stratified split "
                "to match model_v1.ipynb behavior."
            )

        targets = np.array([
            self.dataset.subject_to_pca[self.dataset.subjects[i]][0] for i in range(len(self.dataset))
        ])

        bins = np.percentile(targets, [25, 50, 75])
        strata = np.digitize(targets, bins)
        indices = np.arange(len(self.dataset))

        train_idx, val_idx = train_test_split(
            indices,
            train_size=self.train_frac,
            stratify=strata,
            random_state=self.random_state,
        )

        self.train_ds = Subset(self.dataset, train_idx.tolist())
        self.val_ds = Subset(self.dataset, val_idx.tolist())

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError("Call setup() before requesting train_dataloader().")
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=self.drop_last_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise RuntimeError("Call setup() before requesting val_dataloader().")
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()


class MetricsCollector(pl.Callback):
    """Collects per-epoch metrics into lists for post-training plotting."""

    def __init__(self):
        super().__init__()
        self.train_loss: list[float] = []
        self.val_loss: list[float] = []
        self.train_r2: list[float] = []
        self.val_r2: list[float] = []

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        metrics = trainer.callback_metrics
        if "train_loss" in metrics:
            self.train_loss.append(metrics["train_loss"].item())
        if "train_r2" in metrics:
            self.train_r2.append(metrics["train_r2"].item())

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        if "val_loss" in metrics:
            self.val_loss.append(metrics["val_loss"].item())
        if "val_r2" in metrics:
            self.val_r2.append(metrics["val_r2"].item())


@dataclass
class TrainerBundle:
    trainer: pl.Trainer
    logger: TensorBoardLogger
    checkpoint_callback: ModelCheckpoint
    metrics_collector: MetricsCollector


def build_trainer(
    run_name: str = "resnet1d_v1",
    tb_root_dir: str = "model_data/tb_logs",
    ckpt_dir: str = "model_data/models/lightning",
    max_epochs: int = 50,
    monitor: str = "val_loss",
    monitor_mode: str = "min",
    early_stopping_patience: int = 10,
) -> TrainerBundle:
    logger = TensorBoardLogger(save_dir=tb_root_dir, name=run_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:02d}-{val_loss:.4f}-{val_r2:.4f}",
        monitor=monitor,
        mode=monitor_mode,
        save_top_k=1,
        save_last=True,
    )

    metrics_collector = MetricsCollector()

    callbacks = [
        checkpoint_callback,
        metrics_collector,
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(monitor=monitor, mode=monitor_mode, patience=early_stopping_patience),
    ]

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=max_epochs,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        deterministic=False,
    )

    return TrainerBundle(
        trainer=trainer,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        metrics_collector=metrics_collector,
    )


def tensorboard_logdir(tb_root_dir: str = "model_data/tb_logs") -> str:
    return str(Path(tb_root_dir).resolve())
