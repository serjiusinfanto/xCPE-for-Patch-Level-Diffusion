"""
TimeDART: Causal Transformer encoder with patch-level diffusion pre-training.

Architecture summary
--------------------
Patch tokens are processed channel-independently (one variate = one stream),
following the reference implementation.  This makes the model agnostic to the
number of variates and allows weights to be shared across all channels.

Forward pipeline (both tasks):
  (B, L, patch_len, n_var)
      → channel-independence reshape  → (B·n_var, L, patch_len)
      → patch embedding               → (B·n_var, L, d_model)
      → positional encoding           → (B·n_var, L, d_model)
      → Transformer encoder           → (B·n_var, L, d_model)

Pre-training head (denoise):
      → diffusion head  Linear(d_model → patch_len)
      → (B·n_var, L, patch_len) → reshape → (B, L, patch_len, n_var)

Fine-tuning head (forecast):
      → reshape  (B, n_var, L·d_model)
      → forecast head  Linear(L_ft·d_model → horizon)  [one head per channel]
      → permute  → (B, horizon, n_var)

num_patches at fine-tuning  L_ft = (context_length − patch_len) // finetune_stride + 1
num_patches at pre-training L_pt = (context_length − patch_len) // patch_len + 1

The forecast head is initialised with L_ft (stride=8) because it is only used
during fine-tuning.  The encoder carries no such dependency.

Reference: Wang et al., "TimeDART", ICLR 2025. github.com/Melmaphother/TimeDART
"""

import torch
import torch.nn as nn

from src.models.positional_encoding import FixedPositionalEmbedding
from src.utils.config import ModelConfig, DataConfig


# ── Transformer building block ────────────────────────────────────────────────

class _TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block with Multi-Head Self-Attention + FFN.

    Adapted from: github.com/Melmaphother/TimeDART — layers/TimeDART_EncDec.py
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn  = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-norm self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + self.drop(attn_out)
        # Pre-norm FFN
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


# ── TimeDART ─────────────────────────────────────────────────────────────────

class TimeDART(nn.Module):
    """TimeDART with swappable positional encoding.

    Args:
        model_config: Hyperparameters (d_model, n_heads, n_layers, d_ff,
                      patch_length, dropout, variant).
        data_config:  Data parameters (context_length, horizon, finetune_stride).
                      Used to size the forecast head.
        pos_enc:      Optional pre-built positional encoding module.  Defaults
                      to FixedPositionalEmbedding when None.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataConfig,
        pos_enc: nn.Module | None = None,
    ):
        super().__init__()

        d      = model_config.d_model
        p      = model_config.patch_length
        h      = data_config.horizon
        L_ft   = (data_config.context_length - p) // data_config.finetune_stride + 1

        # ── 1. Patch embedding (channel-independent: Linear(patch_len → d_model)) ──
        self.patch_embed = nn.Linear(p, d)

        # ── 2. Positional encoding ───────────────────────────────────────────
        self.pos_enc = pos_enc if pos_enc is not None else FixedPositionalEmbedding(d)

        # ── 3. Transformer encoder (nn.ModuleList for per-layer xCPE injection) ──
        self.layers = nn.ModuleList([
            _TransformerBlock(d, model_config.n_heads, model_config.d_ff, model_config.dropout)
            for _ in range(model_config.n_layers)
        ])
        self.norm = nn.LayerNorm(d)

        # ── 4. Diffusion head (pre-training: predict added noise) ────────────
        self.diffusion_head = nn.Linear(d, p)

        # ── 5. Forecast head (fine-tuning: L_ft patches × d_model → horizon) ─
        self.forecast_head = nn.Linear(L_ft * d, h)

        # Store for use in encode() / forward helpers
        self.d_model = d
        self.patch_length = p

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _causal_mask(L: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask that prevents attending to future positions."""
        mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        # nn.MultiheadAttention expects float mask where True positions are -inf
        return mask.float().masked_fill(mask, float("-inf"))

    def _transformer_forward(
        self, x: torch.Tensor, causal: bool
    ) -> torch.Tensor:
        """Run the Transformer stack.

        Args:
            x:      (B, L, d_model)
            causal: Whether to apply a causal mask.

        Returns:
            (B, L, d_model)
        """
        mask = self._causal_mask(x.size(1), x.device) if causal else None
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

    # ── Public API ─────────────────────────────────────────────────────────────

    def encode(self, patches: torch.Tensor, causal: bool = False) -> torch.Tensor:
        """Embed and encode a batch of patch sequences.

        Args:
            patches: (B, L, patch_len, n_var)
            causal:  Apply causal masking (True during pre-training).

        Returns:
            (B, L, d_model) — one hidden vector per patch.
        """
        B, L, p, n_var = patches.shape

        # Channel independence: treat each variate as a separate batch element.
        x = patches.permute(0, 3, 1, 2)              # (B, n_var, L, p)
        x = x.reshape(B * n_var, L, p)               # (B·n_var, L, p)

        x = self.patch_embed(x)                       # (B·n_var, L, d_model)
        x = self.pos_enc(x)                           # (B·n_var, L, d_model)
        x = self._transformer_forward(x, causal)      # (B·n_var, L, d_model)

        # Reshape back to (B, L, d_model) by averaging across variates.
        # Note: averaging here collapses per-variate representations into a
        # single sequence representation used only for the forecast head.
        # The diffusion head operates before this collapse (see denoise()).
        x = x.reshape(B, n_var, L, self.d_model)      # (B, n_var, L, d_model)
        return x                                       # keep n_var for downstream use

    def denoise(
        self, noisy_patches: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Predict the noise added to noisy_patches (pre-training).

        Args:
            noisy_patches: (B, L, patch_len, n_var) — mixed clean/noisy input.
            t:             (B,) — noise levels (not used in encoder; kept for
                           API compatibility with schedulers that expect it).

        Returns:
            (B, L, patch_len, n_var) — predicted noise for every patch.
        """
        B, L, p, n_var = noisy_patches.shape

        x = noisy_patches.permute(0, 3, 1, 2).reshape(B * n_var, L, p)  # CI
        x = self.patch_embed(x)                                           # (B·n_var, L, d)
        x = self.pos_enc(x)
        x = self._transformer_forward(x, causal=True)                    # causal during pretrain
        x = self.diffusion_head(x)                                        # (B·n_var, L, p)

        # Reshape back to (B, L, p, n_var)
        x = x.reshape(B, n_var, L, p).permute(0, 2, 3, 1)               # (B, L, p, n_var)
        return x

    def forecast(self, patches: torch.Tensor) -> torch.Tensor:
        """Predict the forecast horizon from context patches (fine-tuning).

        Args:
            patches: (B, L, patch_len, n_var) — context window patches.

        Returns:
            (B, horizon, n_var) — predicted future values.
        """
        B, L, p, n_var = patches.shape

        x = patches.permute(0, 3, 1, 2).reshape(B * n_var, L, p)  # CI
        x = self.patch_embed(x)                                     # (B·n_var, L, d)
        x = self.pos_enc(x)
        x = self._transformer_forward(x, causal=False)             # no mask at finetune
        # x: (B·n_var, L, d_model)

        # Flatten patch dimension and project to horizon.
        x = x.reshape(B, n_var, L * self.d_model)                  # (B, n_var, L·d)
        x = self.forecast_head(x)                                   # (B, n_var, horizon)
        x = x.permute(0, 2, 1)                                     # (B, horizon, n_var)
        return x

    def forward(
        self, patches: torch.Tensor, mode: str = "forecast"
    ) -> torch.Tensor:
        """Dispatch to denoise (pre-training) or forecast (fine-tuning).

        Args:
            patches: (B, L, patch_len, n_var)
            mode:    'denoise' | 'forecast'
        """
        if mode == "denoise":
            # t is unused here; handled externally in the training loop.
            dummy_t = torch.zeros(patches.size(0), dtype=torch.long, device=patches.device)
            return self.denoise(patches, dummy_t)
        return self.forecast(patches)
