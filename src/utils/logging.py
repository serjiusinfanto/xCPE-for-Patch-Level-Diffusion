"""
Lightweight CSV logger for training metrics.

Writes one row per epoch to a CSV file.  Use this as the default logger;
swap for W&B by passing use_wandb=True (requires `wandb login` first).
"""

import csv
import os
from pathlib import Path


class CSVLogger:
    """Append-mode CSV logger.

    Args:
        log_dir:    Directory where the CSV file is written.
        filename:   CSV filename (default 'training_log.csv').
        use_wandb:  If True, also log to Weights & Biases.
        run_name:   W&B run name (only used when use_wandb=True).
    """

    def __init__(
        self,
        log_dir: str,
        filename: str = "training_log.csv",
        use_wandb: bool = False,
        run_name: str | None = None,
    ):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.path      = os.path.join(log_dir, filename)
        self.use_wandb = use_wandb
        self._header_written = False
        # Truncate any existing file so re-runs start clean.
        open(self.path, "w").close()

        if use_wandb:
            import wandb
            wandb.init(project="xcpe-timedart", name=run_name, resume="allow")
            self._wandb = wandb
        else:
            self._wandb = None

    def log(self, metrics: dict, step: int | None = None) -> None:
        """Write a row of metrics.

        Args:
            metrics: Dict mapping metric names → float values.
            step:    Optional epoch / step number (added as 'step' column).
        """
        if step is not None:
            metrics = {"step": step, **metrics}

        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(metrics)

        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def close(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
