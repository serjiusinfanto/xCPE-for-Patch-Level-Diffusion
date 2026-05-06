"""
DDPM linear beta schedule for patch-level diffusion pre-training.

Implements the forward noising process  q(x_t | x_0) = N(√ᾱ_t x_0, (1−ᾱ_t)I)
and the posterior mean estimator used to recover x_0 from a predicted noise.

Parameters follow CLAUDE.md §2:
  β_1 = 1e-4,  β_T = 0.02,  T = 1000  (linear schedule)

All schedule tensors are stored as nn.Module buffers so that a single
.to(device) call moves everything to GPU without extra bookkeeping.

Reference: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
"""

import torch
import torch.nn as nn


class LinearBetaSchedule(nn.Module):
    """Linear DDPM beta schedule with pre-computed forward-process constants.

    Args:
        beta_start: β at step 1  (default 1e-4).
        beta_end:   β at step T  (default 0.02).
        T:          Total diffusion steps (default 1000).
    """

    def __init__(
        self,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        T: int = 1000,
    ):
        super().__init__()
        self.T = T

        betas                    = torch.linspace(beta_start, beta_end, T)
        alphas                   = 1.0 - betas
        alpha_bars               = torch.cumprod(alphas, dim=0)
        sqrt_alpha_bars          = alpha_bars.sqrt()
        sqrt_one_minus_alpha_bars = (1.0 - alpha_bars).sqrt()

        # Register as buffers: moved to GPU automatically with .to(device).
        self.register_buffer("betas",                     betas)
        self.register_buffer("alphas",                    alphas)
        self.register_buffer("alpha_bars",                alpha_bars)
        self.register_buffer("sqrt_alpha_bars",           sqrt_alpha_bars)
        self.register_buffer("sqrt_one_minus_alpha_bars", sqrt_one_minus_alpha_bars)

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t ~ q(x_t | x_0) for a batch of noise levels t.

        x_t = √ᾱ_t · x_0  +  √(1−ᾱ_t) · ε,   ε ~ N(0, I)

        Args:
            x_0: Clean input of any shape (B, *dims).
            t:   Noise levels, shape (B,), values in [0, T-1].

        Returns:
            (x_t, noise) — both the same shape as x_0.
        """
        noise = torch.randn_like(x_0)

        # Index schedule tensors and reshape for broadcasting over *dims.
        extra_dims = x_0.dim() - 1                                # number of non-batch dims
        view_shape = (-1,) + (1,) * extra_dims

        sqrt_ab   = self.sqrt_alpha_bars[t].view(view_shape)           # (B, 1, ...)
        sqrt_1mab = self.sqrt_one_minus_alpha_bars[t].view(view_shape) # (B, 1, ...)

        x_t = sqrt_ab * x_0 + sqrt_1mab * noise
        return x_t, noise

    # ------------------------------------------------------------------
    # Reverse process helper
    # ------------------------------------------------------------------

    def predict_x0(
        self,
        x_t: torch.Tensor,
        predicted_noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate the clean signal x_0 from x_t and the predicted noise.

        x_0_hat = (x_t  −  √(1−ᾱ_t) · ε_hat) / √ᾱ_t

        Args:
            x_t:              Noisy input, same shape as x_0.
            predicted_noise:  Model's noise prediction, same shape as x_0.
            t:                Noise levels, shape (B,).

        Returns:
            Estimated x_0, same shape as x_t.
        """
        extra_dims = x_t.dim() - 1
        view_shape = (-1,) + (1,) * extra_dims

        sqrt_ab   = self.sqrt_alpha_bars[t].view(view_shape)
        sqrt_1mab = self.sqrt_one_minus_alpha_bars[t].view(view_shape)

        return (x_t - sqrt_1mab * predicted_noise) / sqrt_ab.clamp(min=1e-8)
