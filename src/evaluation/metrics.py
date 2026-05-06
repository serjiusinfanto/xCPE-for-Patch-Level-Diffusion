"""
Evaluation metrics for time series forecasting.

Both functions operate on raw tensors and return Python floats.
They are used by finetune.py to compute per-batch metrics and by
scripts/collect_results.py to aggregate final numbers.
"""

import torch


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Squared Error.

    Args:
        pred:   (B, horizon, n_var) — model predictions.
        target: (B, horizon, n_var) — ground truth.

    Returns:
        Scalar MSE averaged over all elements.
    """
    return torch.mean((pred - target) ** 2).item()


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error.

    Args:
        pred:   (B, horizon, n_var)
        target: (B, horizon, n_var)

    Returns:
        Scalar MAE averaged over all elements.
    """
    return torch.mean(torch.abs(pred - target)).item()
