"""
Phase 1 validation tests for the data pipeline.

Run from the project root:
    pytest tests/test_data.py -v

All tests use the ETTh1 dataset via the baseline config. They verify:
  1. No data leakage: val/test normalization uses ONLY train mean/std.
  2. Patch shape is exactly (num_patches, 16, n_variates).
  3. No NaN or Inf values in any split.
  4. Split sizes match expected percentages (±1%).
  5. Reproducibility: same seed → same batch every time.
"""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.data.dataset import TimeSeriesDataset
from src.data.preprocess import split_dataset, normalize, apply_normalization
import pandas as pd

CONFIG_PATH = "configs/baseline_etth1.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def datasets(cfg):
    path = cfg.data.path
    train_ds = TimeSeriesDataset(path, cfg.data, split="train")
    val_ds = TimeSeriesDataset(path, cfg.data, split="val")
    test_ds = TimeSeriesDataset(path, cfg.data, split="test")
    return train_ds, val_ds, test_ds


# ── Test 1: No data leakage ──────────────────────────────────────────────────

def test_no_leakage_val_uses_train_stats(cfg):
    """Val normalization must use train mean/std, not its own."""
    path = cfg.data.path
    df = pd.read_csv(path).drop(columns=[pd.read_csv(path).columns[0]])

    train_df, val_df, _ = split_dataset(
        df, cfg.data.dataset, cfg.data.train_split, cfg.data.val_split
    )
    train_arr = train_df.values.astype(np.float32)
    val_arr = val_df.values.astype(np.float32)

    _, train_mean, train_std = normalize(train_arr)
    _, val_mean, val_std = normalize(val_arr)

    # Train and val statistics must differ (val is a different time period).
    assert not np.allclose(train_mean, val_mean, atol=1e-4), \
        "Train and val means are suspiciously identical — possible leakage."
    assert not np.allclose(train_std, val_std, atol=1e-4), \
        "Train and val stds are suspiciously identical — possible leakage."

    # The dataset must use train stats, not val stats.
    val_ds = TimeSeriesDataset(path, cfg.data, split="val")
    patches, _ = val_ds[0]
    # Manually compute what val[0] should look like with train stats.
    val_normalized_correctly = apply_normalization(val_arr, train_mean, train_std)
    context = cfg.data.context_length
    patch_len = cfg.data.patch_length
    expected_patches = torch.tensor(val_normalized_correctly[:context], dtype=torch.float32)
    expected_patches = expected_patches.unfold(0, patch_len, patch_len).permute(0, 2, 1)
    assert torch.allclose(patches, expected_patches, atol=1e-5), \
        "Val dataset is not using train normalization stats."


# ── Test 2: Patch shape ──────────────────────────────────────────────────────

def test_patch_shape(datasets, cfg):
    """Each item must have patches of shape (num_patches, 16, n_variates)."""
    train_ds, val_ds, test_ds = datasets
    patch_len = cfg.data.patch_length  # 16
    context = cfg.data.context_length  # 336
    horizon = cfg.data.horizon         # 96

    # Expected number of non-overlapping patches
    expected_n_patches = (context - patch_len) // patch_len + 1  # 21

    for ds_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        patches, target = ds[0]
        assert patches.dim() == 3, \
            f"{ds_name}: patches should be 3D, got {patches.dim()}D"
        n_patches, p_len, n_var = patches.shape
        assert p_len == patch_len, \
            f"{ds_name}: patch_length should be {patch_len}, got {p_len}"
        assert n_patches == expected_n_patches, \
            f"{ds_name}: expected {expected_n_patches} patches, got {n_patches}"
        assert target.shape == (horizon, n_var), \
            f"{ds_name}: target shape should be ({horizon}, {n_var}), got {target.shape}"


# ── Test 3: No NaN or Inf ────────────────────────────────────────────────────

def test_no_nan_inf(datasets):
    """No NaN or Inf values in any patch or target across all splits."""
    train_ds, val_ds, test_ds = datasets
    for ds_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        # Check first, middle, and last items
        indices = [0, len(ds) // 2, len(ds) - 1]
        for idx in indices:
            patches, target = ds[idx]
            assert not torch.isnan(patches).any(), \
                f"{ds_name}[{idx}]: NaN found in patches"
            assert not torch.isinf(patches).any(), \
                f"{ds_name}[{idx}]: Inf found in patches"
            assert not torch.isnan(target).any(), \
                f"{ds_name}[{idx}]: NaN found in target"
            assert not torch.isinf(target).any(), \
                f"{ds_name}[{idx}]: Inf found in target"


# ── Test 4: Split sizes ──────────────────────────────────────────────────────

def test_split_sizes(cfg):
    """Train/val/test split sizes must match configured fractions (±1%)."""
    path = cfg.data.path
    df = pd.read_csv(path)
    df = df.drop(columns=[df.columns[0]])
    n_total = len(df)

    train_df, val_df, test_df = split_dataset(
        df, cfg.data.dataset, cfg.data.train_split, cfg.data.val_split
    )
    test_split = 1.0 - cfg.data.train_split - cfg.data.val_split

    for name, split_df, expected_frac in [
        ("train", train_df, cfg.data.train_split),
        ("val",   val_df,   cfg.data.val_split),
        ("test",  test_df,  test_split),
    ]:
        actual_frac = len(split_df) / n_total
        assert abs(actual_frac - expected_frac) <= 0.01, (
            f"{name} split: expected {expected_frac:.2f}, "
            f"got {actual_frac:.4f} (diff > 1%)"
        )


# ── Test 5: Reproducibility ──────────────────────────────────────────────────

def test_reproducibility(cfg):
    """Same seed must produce the same batch ordering every run."""
    path = cfg.data.path
    seed = cfg.training.seed

    def get_first_batch(seed_val):
        torch.manual_seed(seed_val)
        np.random.seed(seed_val)
        ds = TimeSeriesDataset(path, cfg.data, split="train")
        loader = DataLoader(ds, batch_size=4, shuffle=True,
                            generator=torch.Generator().manual_seed(seed_val))
        patches, _ = next(iter(loader))
        return patches

    batch_a = get_first_batch(seed)
    batch_b = get_first_batch(seed)

    assert torch.allclose(batch_a, batch_b), \
        "Same seed produced different batches — dataset is not reproducible."


# ── Test 6: Dataset length sanity ────────────────────────────────────────────

def test_dataset_length(datasets, cfg):
    """Every split must have a positive number of windows."""
    train_ds, val_ds, test_ds = datasets
    window = cfg.data.context_length + cfg.data.horizon  # 432
    for ds_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        assert len(ds) > 0, f"{ds_name} dataset has zero windows"


# ── Test 7: No temporal overlap between splits ───────────────────────────────

def test_no_temporal_overlap(cfg):
    """Train end index must be strictly before val start, and val before test."""
    path = cfg.data.path
    df = pd.read_csv(path)
    df = df.drop(columns=[df.columns[0]])
    n = len(df)

    train_end = int(n * cfg.data.train_split)
    val_end = int(n * (cfg.data.train_split + cfg.data.val_split))

    assert train_end < val_end < n, \
        "Split boundaries overlap or are out of order."
    assert val_end - train_end > 0, "Val split is empty."
    assert n - val_end > 0, "Test split is empty."
