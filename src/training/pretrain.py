"""
Self-supervised diffusion pre-training loop for TimeDART.

Training objective: given a context window with ~50 % of its patch tokens
replaced by DDPM-noised versions, the encoder must predict the added noise.

Usage (always via accelerate for FP16 / multi-GPU):
    accelerate launch scripts/run_pretrain.py --config configs/baseline_etth1.yaml

Key hyper-parameters (all in YAML):
    pretrain_lr    = 1e-4
    weight_decay   = 1e-5  (hardcoded — not a tunable HP)
    pretrain_epochs
    batch_size
    patience       (early stopping on val loss)

Reference: Wang et al., "TimeDART", ICLR 2025. github.com/Melmaphother/TimeDART
"""

import os
import sys
import random
import argparse

# Ensure the project root (parent of src/) is on sys.path when the script is
# launched directly via `accelerate launch src/training/pretrain.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator

from src.utils.config import load_config
from src.utils.logging import CSVLogger
from src.data.dataset import TimeSeriesDataset
from src.diffusion.noise_scheduler import LinearBetaSchedule
from src.diffusion.denoising import corrupt_patches
from src.models import build_model


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── One epoch ─────────────────────────────────────────────────────────────────

def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    noise_scheduler: LinearBetaSchedule,
    optimizer: torch.optim.Optimizer | None,
    accelerator: Accelerator,
    is_train: bool,
) -> float:
    model.train(is_train)
    total_loss, n_batches = 0.0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for patches, _ in loader:
            # patches: (B, L, patch_len, n_var) — target not used in pretrain
            x_input, noise_target, t, mask = corrupt_patches(
                patches, noise_scheduler, corrupt_ratio=0.5
            )

            pred_noise = model.denoise(x_input, t)          # (B, L, patch_len, n_var)

            # Compute loss only on the corrupted patches.
            mask_4d = mask.unsqueeze(-1).unsqueeze(-1).expand_as(pred_noise)
            loss = F.mse_loss(pred_noise[mask_4d], noise_target[mask_4d])

            if is_train:
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)


# ── Main training loop ─────────────────────────────────────────────────────────

def pretrain(
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

    # ── Data ──────────────────────────────────────────────────────────────────
    # Pre-training uses non-overlapping patches (stride = patch_length).
    train_ds = TimeSeriesDataset(cfg.data.path, cfg.data, split="train")
    val_ds   = TimeSeriesDataset(cfg.data.path, cfg.data, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
        generator=torch.Generator().manual_seed(cfg.training.seed),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # ── Model + schedule + optimiser ──────────────────────────────────────────
    model            = build_model(cfg.model, cfg.data)
    noise_scheduler  = LinearBetaSchedule().to(device)
    optimizer        = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.pretrain_lr,
        weight_decay=1e-5,
    )

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    noise_scheduler = noise_scheduler.to(device)

    # ── Logging + checkpointing ────────────────────────────────────────────────
    run_tag  = f"{cfg.data.dataset}_h{cfg.data.horizon}_{cfg.model.variant}_seed{cfg.training.seed}_pretrain"
    logger   = CSVLogger("results/logs", filename=f"{run_tag}.csv")
    ckpt_dir = "results/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_loss = float("inf")
    patience_ctr  = 0

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(1, cfg.training.pretrain_epochs + 1):
        train_loss = _run_epoch(
            model, train_loader, noise_scheduler, optimizer, accelerator, is_train=True
        )
        val_loss = _run_epoch(
            model, val_loader, noise_scheduler, None, accelerator, is_train=False
        )

        if accelerator.is_main_process:
            logger.log({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
            print(f"[pretrain] epoch {epoch:>3}  train={train_loss:.4f}  val={val_loss:.4f}")

            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                ckpt_path = os.path.join(ckpt_dir, f"{run_tag}_ep{epoch}.pt")
                torch.save(accelerator.unwrap_model(model).state_dict(), ckpt_path)

            # Save best checkpoint + early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_ctr  = 0
                torch.save(
                    accelerator.unwrap_model(model).state_dict(),
                    os.path.join(ckpt_dir, f"{run_tag}_best.pt"),
                )
            else:
                patience_ctr += 1
                if patience_ctr >= cfg.training.patience:
                    print(f"[pretrain] early stopping at epoch {epoch}")
                    break

    if accelerator.is_main_process:
        logger.close()
        print(f"[pretrain] done. Best val loss: {best_val_loss:.4f}")
        print(f"[pretrain] checkpoint: results/checkpoints/{run_tag}_best.pt")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  required=True,        help="Path to YAML config")
    parser.add_argument("--horizon", type=int, default=None, help="Override horizon")
    parser.add_argument("--seed",    type=int, default=None, help="Override random seed")
    args = parser.parse_args()
    pretrain(args.config, args.horizon, args.seed)
