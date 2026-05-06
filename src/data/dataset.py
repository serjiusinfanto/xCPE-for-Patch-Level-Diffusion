"""
TimeSeriesDataset: sliding-window patch dataset for time series forecasting.

Each item is a (context_patches, target) pair:
  context_patches : (num_patches, patch_length, n_variates)
  target          : (horizon, n_variates)

Patch stride controls overlap:
  Pre-training  : stride = patch_length  (non-overlapping)
  Fine-tuning   : stride = 8             (overlapping, per CLAUDE.md §1)

Normalization is fit on the training split only and reused for val/test,
preventing any leakage of future statistics.

Reference: Wang et al., ICLR 2025. github.com/Melmaphother/TimeDART
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from src.data.preprocess import split_dataset, normalize, apply_normalization
from src.utils.config import DataConfig


class TimeSeriesDataset(Dataset):
    """Sliding-window dataset that returns patch tensors.

    Args:
        csv_path:    Path to the raw CSV file.
        data_config: DataConfig from the YAML (context_length, horizon,
                     patch_length, train_split, val_split, dataset name).
        split:       One of 'train', 'val', 'test'.
        stride:      Patch stride. Defaults to patch_length (non-overlapping).
                     Pass 8 for fine-tuning (overlapping).
    """

    def __init__(
        self,
        csv_path: str,
        data_config: DataConfig,
        split: str = "train",
        stride: int = None,
    ):
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got '{split}'"

        self.patch_length = data_config.patch_length
        self.stride = stride if stride is not None else data_config.patch_length
        self.context_length = data_config.context_length
        self.horizon = data_config.horizon

        # ── Load CSV and drop the date/timestamp column ──────────────────────
        df = pd.read_csv(csv_path)
        # First column is always a date string in all ETT / Weather datasets.
        df = df.drop(columns=[df.columns[0]])

        # ── Time-ordered split ───────────────────────────────────────────────
        train_df, val_df, test_df = split_dataset(
            df,
            data_config.dataset,
            data_config.train_split,
            data_config.val_split,
        )

        # ── Z-score normalization — fit ONLY on train ────────────────────────
        train_arr = train_df.values.astype(np.float32)
        val_arr = val_df.values.astype(np.float32)
        test_arr = test_df.values.astype(np.float32)

        train_arr, mean, std = normalize(train_arr)
        val_arr = apply_normalization(val_arr, mean, std)
        test_arr = apply_normalization(test_arr, mean, std)

        # Store stats so the training loop can invert normalization if needed.
        self.mean = torch.tensor(mean, dtype=torch.float32)  # (1, n_var)
        self.std = torch.tensor(std, dtype=torch.float32)    # (1, n_var)

        # ── Select the requested split ───────────────────────────────────────
        split_map = {"train": train_arr, "val": val_arr, "test": test_arr}
        arr = split_map[split]
        self.data = torch.tensor(arr, dtype=torch.float32)   # (T, n_var)

        # ── Pre-compute number of valid sliding windows ──────────────────────
        window = self.context_length + self.horizon
        self.n_windows = max(0, len(self.data) - window + 1)

    # ── num_patches for this dataset instance ──────────────────────────────
    @property
    def num_patches(self) -> int:
        """Number of patches per context window with the current stride."""
        return (self.context_length - self.patch_length) // self.stride + 1

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int):
        """Return (context_patches, target).

        context_patches : (num_patches, patch_length, n_variates)
        target          : (horizon, n_variates)
        """
        # Slice the context window and the target horizon
        x = self.data[idx : idx + self.context_length]                         # (L, n_var)
        y = self.data[idx + self.context_length :
                      idx + self.context_length + self.horizon]                 # (H, n_var)

        # Build patches via unfold on the time dimension.
        # unfold(dim=0, size=patch_length, step=stride):
        #   (L, n_var) → (num_patches, n_var, patch_length)
        patches = x.unfold(0, self.patch_length, self.stride)                  # (P, n_var, patch_len)
        patches = patches.permute(0, 2, 1)                                     # (P, patch_len, n_var)

        return patches, y
