"""
Data preprocessing utilities: z-score normalization and time-ordered splitting.

Normalization is always fit on the training split only. The stored mean/std
must be reused for val and test to prevent data leakage.

Split ratios follow the TimeDART paper convention:
  ETT datasets  : 60 / 20 / 20  (train / val / test)
  Weather       : 70 / 10 / 20

Reference: Wang et al., ICLR 2025. github.com/Melmaphother/TimeDART
"""

import numpy as np
import pandas as pd
from typing import Tuple


def split_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    train_split: float,
    val_split: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train / val / test by time order (no shuffling).

    Args:
        df:           Full dataset with the date column already dropped.
        dataset_name: Used only to document intent; actual ratios come from
                      train_split / val_split (set in the YAML config).
        train_split:  Fraction of rows for training (e.g. 0.6).
        val_split:    Fraction of rows for validation (e.g. 0.2).
                      test fraction = 1 - train_split - val_split.

    Returns:
        (train_df, val_df, test_df) — non-overlapping, time-ordered.
    """
    n = len(df)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    return train_df, val_df, test_df


def normalize(
    train_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit z-score normalization on the training array.

    Args:
        train_arr: shape (T_train, n_variates), float.

    Returns:
        (normalized_arr, mean, std) where mean and std have shape
        (1, n_variates) and are computed solely from train_arr.
        std values below 1e-8 are clamped to 1.0 to avoid division by zero.
    """
    mean = train_arr.mean(axis=0, keepdims=True)   # (1, n_var)
    std = train_arr.std(axis=0, keepdims=True)      # (1, n_var)
    std = np.where(std < 1e-8, 1.0, std)           # guard near-constant channels
    normalized = (train_arr - mean) / std
    return normalized, mean, std


def apply_normalization(
    arr: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply pre-computed normalization stats to val or test arrays.

    Args:
        arr:  shape (T, n_variates).
        mean: shape (1, n_variates), from normalize() on the train split.
        std:  shape (1, n_variates), from normalize() on the train split.

    Returns:
        Normalized array of the same shape as arr.
    """
    return (arr - mean) / std


def inverse_normalize(
    arr: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Undo z-score normalization (for metric reporting in original units).

    Args:
        arr:  Normalized array, shape (T, n_variates).
        mean: shape (1, n_variates).
        std:  shape (1, n_variates).

    Returns:
        Array in original scale.
    """
    return arr * std + mean
