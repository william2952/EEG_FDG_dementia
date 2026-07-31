import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset


class EEGSegmentDataset(Dataset):
    def __init__(self, parquet_dir, pca_targets, n_pca_components, n_segments,
                 n_samples, subject_ids, target_mean=None, target_std=None,
                 pca_offset=0, mask_channels=None):
        """
        mask_channels: optional list of channel-name strings (e.g. ["C3","P3","F3","Cz"])
                       to zero out in every window, applied uniformly to every subject
                       and every split (train/val/holdout) that uses this dataset.
        """
        self.n_segments = n_segments
        self.n_samples = n_samples
        paths = {p.stem: p for p in Path(parquet_dir).glob("*.parquet")}
        ids = sorted(set(paths) & set(pca_targets.index.astype(str)) & {str(s) for s in subject_ids})
        self.subjects = ids
        self.subj_to_idx = {s: i for i, s in enumerate(ids)}

        raw = np.stack([pca_targets.loc[s].to_numpy(dtype=np.float32)[pca_offset:pca_offset + n_pca_components]
                        for s in ids])
        if target_mean is None:
            target_mean = raw.mean(0).astype(np.float32)
            target_std = (raw.std(0) + 1e-8).astype(np.float32)
        self.target_mean, self.target_std = target_mean, target_std
        self.targets = {s: ((raw[i] - target_mean) / target_std).astype(np.float32)
                        for i, s in enumerate(ids)}

        # Preload each subject once into memory as (n_seg, n_ch, 2560) float32
        self.cache = {}
        mask_idx = None
        for i, s in enumerate(ids):
            df = (pd.read_parquet(paths[s],
                                  columns=["channel_name", "segment", "segment_index"])
                    .sort_values(["segment_index", "channel_name"]))
            n_seg = df["segment_index"].nunique()
            n_ch = df["channel_name"].nunique()

            if mask_channels and mask_idx is None:
                # Row order after the sort above is alphabetical by channel_name
                # within each segment_index block — derive mask indices once,
                # from the first subject (channel set is fixed across subjects).
                ch_names = sorted(df["channel_name"].unique())
                mask_idx = [ch_names.index(c) for c in mask_channels]

            arr = np.stack([np.pad(np.asarray(a, dtype=np.float32)[:2560],
                                   (0, max(0, 2560 - len(a))), mode="reflect")
                            for a in df["segment"]])
            arr = arr.reshape(n_seg, n_ch, 2560)
            if mask_idx:
                arr[:, mask_idx, :] = 0.0
            self.cache[s] = arr
            if (i + 1) % 100 == 0 or i + 1 == len(ids):
                print(f"  loaded {i+1}/{len(ids)}", flush=True)

        usable = [s for s in ids if len(self.cache[s]) >= n_segments]
        self._index = [(s, i) for s in usable for i in range(n_samples)]

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        subject_id, _ = self._index[idx]
        n_avail = len(self.cache[subject_id])
        start = np.random.randint(0, n_avail - self.n_segments + 1)
        arr = self.cache[subject_id][start:start + self.n_segments]  # (n_seg, n_ch, L)
        window = arr.transpose(1, 0, 2).reshape(arr.shape[1], -1)    # (n_ch, n_seg*L)
        return (torch.from_numpy(np.ascontiguousarray(window)),
                torch.from_numpy(self.targets[subject_id]),
                self.subj_to_idx[subject_id])
