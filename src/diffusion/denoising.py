"""
Patch-level forward and reverse diffusion utilities.

This module provides a thin convenience wrapper around LinearBetaSchedule
for use inside the pre-training loop.  The core maths live in
noise_scheduler.py; this module handles the patch-selection logic that
corrupts only a random subset of patches per sequence (CLAUDE.md §2.4).

Reference: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
"""

import torch
from src.diffusion.noise_scheduler import LinearBetaSchedule


def corrupt_patches(
    patches: torch.Tensor,
    noise_scheduler: LinearBetaSchedule,
    corrupt_ratio: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Corrupt a random subset of patches with DDPM forward noise.

    For each sample in the batch:
      1. Sample a single noise level  t ~ Uniform(1, T).
      2. Select  ceil(L * corrupt_ratio)  patch positions at random.
      3. Replace those patches with noisy versions; leave the rest clean.

    Args:
        patches:        (B, L, patch_len, n_var) — clean patch tokens.
        noise_scheduler: Pre-built LinearBetaSchedule (already on device).
        corrupt_ratio:  Fraction of patches to corrupt (default 0.5).

    Returns:
        x_input   : (B, L, patch_len, n_var) — mixed clean/noisy patches.
        noise     : (B, L, patch_len, n_var) — actual noise added (zero for
                     clean patches).
        t         : (B,) — noise levels used per sample.
        mask      : (B, L) bool — True where a patch was corrupted.
    """
    B, L, patch_len, n_var = patches.shape
    device = patches.device

    # ── 1. Sample noise level t (0-indexed: valid range [0, T-1]) ──────────
    t = torch.randint(0, noise_scheduler.T, (B,), device=device)

    # ── 2. Add noise to ALL patches (we'll only keep a subset) ──────────────
    x_t, noise_all = noise_scheduler.add_noise(patches, t)
    # x_t, noise_all: both (B, L, patch_len, n_var)

    # ── 3. Build corruption mask ─────────────────────────────────────────────
    num_corrupt = max(1, round(L * corrupt_ratio))
    mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    for b in range(B):
        idx = torch.randperm(L, device=device)[:num_corrupt]
        mask[b, idx] = True

    # ── 4. Compose the mixed input ───────────────────────────────────────────
    mask_4d = mask.unsqueeze(-1).unsqueeze(-1)            # (B, L, 1, 1)
    x_input = torch.where(mask_4d, x_t, patches)          # noisy where mask

    # Zero out noise for non-corrupted patches so the loss target is correct.
    noise_target = noise_all * mask_4d.float()

    return x_input, noise_target, t, mask
