"""
Supervised fine-tuning loop for TimeDART.

The encoder is initialised from a pre-trained checkpoint.  The encoder is
frozen for the first 5 epochs, then unfrozen for joint fine-tuning.

Fine-tuning uses overlapping patches (stride = finetune_stride = 8) to give
the forecast head denser context coverage.

Usage (always via accelerate):
    accelerate launch scripts/run_finetune.py --config configs/baseline_etth1.yaml

Reference: Wang et al., "TimeDART", ICLR 2025. github.com/Melmaphother/TimeDART
"""

import os
import sys
import random
import argparse

# Ensure the project root (parent of src/) is on sys.path when the script is
# launched directly via `accelerate launch src/training/finetune.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator

from src.utils.config import load_config
from src.utils.logging import CSVLogger
from src.data.dataset import TimeSeriesDataset
from src.models import build_model
from src.evaluation.metrics import compute_mse, compute_mae

# Freeze epochs before unfreezing encoder (CLAUDE.md §2.5)
_FREEZE_EPOCHS = 5


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Encoder parameter group helpers ───────────────────────────────────────────

def _encoder_params(model: torch.nn.Module):
    """Return all parameters belonging to the encoder (not the forecast head)."""
    exclude = {"forecast_head"}
    return [p for n, p in model.named_parameters() if n.split(".")[0] not in exclude]


def _set_encoder_grad(model: torch.nn.Module, requires_grad: bool) -> None:
    for n, p in model.named_parameters():
        if n.split(".")[0] != "forecast_head":
            p.requires_grad_(requires_grad)


# ── One epoch ─────────────────────────────────────────────────────────────────

def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    accelerator: Accelerator,
    is_train: bool,
) -> tuple[float, float, float]:
    """Return (avg_loss, avg_mse, avg_mae)."""
    model.train(is_train)
    total_loss = total_mse = total_mae = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for patches, targets in loader:
            # patches:  (B, L, patch_len, n_var)
            # targets:  (B, horizon, n_var)
            preds = model.forecast(patches)                    # (B, horizon, n_var)
            loss  = F.mse_loss(preds, targets)

            if is_train:
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()

            # Gather metrics (detached)
            p_cpu = preds.detach().cpu()
            t_cpu = targets.detach().cpu()
            total_loss += loss.item()
            total_mse  += compute_mse(p_cpu, t_cpu)
            total_mae  += compute_mae(p_cpu, t_cpu)
            n_batches  += 1

    n = max(n_batches, 1)
    return total_loss / n, total_mse / n, total_mae / n


# ── Main fine-tuning loop ──────────────────────────────────────────────────────

def finetune(
    config_path: str,
    horizon_override: int | None = None,
    seed_override: int | None = None,
) -> None:
    cfg = load_config(config_path)
    if horizon_override is not None:
        cfg.data.horizon = horizon_override
    if seed_override is not None:
        cfg.training.seed = seed_override

    set_seed(cfg.training.seed)

    accelerator = Accelerator(mixed_precision="fp16")
    device      = accelerator.device

    # ── Data (overlapping patches via finetune_stride) ───────────────────────
    train_ds = TimeSeriesDataset(
        cfg.data.path, cfg.data, split="train",
        stride=cfg.data.finetune_stride,
    )
    val_ds = TimeSeriesDataset(
        cfg.data.path, cfg.data, split="val",
        stride=cfg.data.finetune_stride,
    )
    test_ds = TimeSeriesDataset(
        cfg.data.path, cfg.data, split="test",
        stride=cfg.data.finetune_stride,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
        generator=torch.Generator().manual_seed(cfg.training.seed),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg.model, cfg.data)

    # Load pre-trained encoder weights if checkpoint exists.
    run_tag_pretrain = (
        f"{cfg.data.dataset}_h{cfg.data.horizon}_{cfg.model.variant}_seed{cfg.training.seed}_pretrain"
    )
    pretrain_ckpt = f"results/checkpoints/{run_tag_pretrain}_best.pt"
    if os.path.exists(pretrain_ckpt):
        state = torch.load(pretrain_ckpt, map_location="cpu", weights_only=True)
        # Load only encoder weights (skip forecast_head which has different size).
        encoder_state = {k: v for k, v in state.items()
                         if not k.startswith("forecast_head")}
        missing, unexpected = model.load_state_dict(encoder_state, strict=False)
        if accelerator.is_main_process:
            print(f"[finetune] loaded pretrain checkpoint: {pretrain_ckpt}")
            if missing:
                print(f"[finetune] missing keys: {missing}")
    else:
        if accelerator.is_main_process:
            print(f"[finetune] WARNING: no pretrain checkpoint found at {pretrain_ckpt}. "
                  "Training from scratch.")

    # Freeze encoder for first _FREEZE_EPOCHS epochs.
    _set_encoder_grad(model, requires_grad=False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.finetune_lr,
        weight_decay=1e-5,
    )

    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    run_tag  = f"{cfg.data.dataset}_h{cfg.data.horizon}_{cfg.model.variant}_seed{cfg.training.seed}_finetune"
    logger   = CSVLogger("results/logs", filename=f"{run_tag}.csv")
    ckpt_dir = "results/checkpoints"

    best_val_mse = float("inf")
    patience_ctr = 0

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(1, cfg.training.finetune_epochs + 1):

        # Unfreeze encoder after _FREEZE_EPOCHS.
        if epoch == _FREEZE_EPOCHS + 1:
            _set_encoder_grad(accelerator.unwrap_model(model), requires_grad=True)
            # Rebuild optimizer to include newly unfrozen params.
            optimizer = torch.optim.AdamW(
                accelerator.unwrap_model(model).parameters(),
                lr=cfg.training.finetune_lr,
                weight_decay=1e-5,
            )
            optimizer = accelerator.prepare(optimizer)
            if accelerator.is_main_process:
                print(f"[finetune] epoch {epoch}: encoder unfrozen")

        train_loss, train_mse, train_mae = _run_epoch(
            model, train_loader, optimizer, accelerator, is_train=True
        )
        val_loss, val_mse, val_mae = _run_epoch(
            model, val_loader, None, accelerator, is_train=False
        )

        if accelerator.is_main_process:
            logger.log({
                "train_loss": train_loss, "train_mse": train_mse, "train_mae": train_mae,
                "val_loss":   val_loss,   "val_mse":   val_mse,   "val_mae":   val_mae,
            }, step=epoch)
            print(
                f"[finetune] epoch {epoch:>3}  "
                f"train_mse={train_mse:.4f}  val_mse={val_mse:.4f}  val_mae={val_mae:.4f}"
            )

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                patience_ctr = 0
                torch.save(
                    accelerator.unwrap_model(model).state_dict(),
                    os.path.join(ckpt_dir, f"{run_tag}_best.pt"),
                )
            else:
                patience_ctr += 1
                if patience_ctr >= cfg.training.patience:
                    print(f"[finetune] early stopping at epoch {epoch}")
                    break

    # ── Final evaluation on test set ──────────────────────────────────────────
    if accelerator.is_main_process:
        best_ckpt = os.path.join(ckpt_dir, f"{run_tag}_best.pt")
        if os.path.exists(best_ckpt):
            state = torch.load(best_ckpt, map_location=device, weights_only=True)
            accelerator.unwrap_model(model).load_state_dict(state)

    _, test_mse, test_mae = _run_epoch(
        model, test_loader, None, accelerator, is_train=False
    )

    if accelerator.is_main_process:
        logger.log({"test_mse": test_mse, "test_mae": test_mae}, step=0)
        logger.close()
        print(f"\n[finetune] TEST  MSE={test_mse:.4f}  MAE={test_mae:.4f}")
        print(f"[finetune] checkpoint: {best_ckpt}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  required=True,          help="Path to YAML config")
    parser.add_argument("--horizon", type=int, default=None, help="Override horizon")
    parser.add_argument("--seed",    type=int, default=None, help="Override random seed")
    args = parser.parse_args()
    finetune(args.config, args.horizon, args.seed)
