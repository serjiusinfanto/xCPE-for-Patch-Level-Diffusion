"""
Positional encoding modules for the xCPE-TimeDART project.

Two classes coexist here:
  FixedPositionalEmbedding — learned absolute lookup table (baseline)
  xCPE                     — content-conditioned positional encoding (our contribution)

FixedPositionalEmbedding is adapted from:
  github.com/Melmaphother/TimeDART — layers/Embed.py :: LearnablePositionEncoding

xCPE is adapted from:
  github.com/Pointcept/PointTransformerV3 — model.py :: Block.cpe
  The original CPE uses a 3-D sparse convolution (SubMConv3d, kernel=3) to aggregate
  local point-cloud geometry. We replace that with a temporal neighbourhood statistic
  MLP: for each patch token we compute [mean, variance, linear_trend_slope] of the
  three-token neighbourhood and project those 3 scalars to d_model.

Reference: Xiaoyang Wu et al., "Point Transformer V3", CVPR 2024 (Oral).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedPositionalEmbedding(nn.Module):
    """Learned absolute positional embedding.

    A simple lookup table of shape (max_len, d_model). The embedding for
    position i is looked up and added to the token at position i.

    Adapted from: github.com/Melmaphother/TimeDART — layers/Embed.py

    Args:
        d_model:  Token dimension.
        max_len:  Maximum sequence length supported (default 512 is >> any
                  num_patches we use in this project).
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embedding to x.

        Args:
            x: (B, L, d_model)

        Returns:
            (B, L, d_model) — x + positional embeddings for positions 0 … L-1.
        """
        L = x.size(1)
        positions = torch.arange(L, device=x.device)     # (L,)
        pos_emb = self.embedding(positions)               # (L, d_model)
        return x + pos_emb.unsqueeze(0)                   # broadcast over B


class xCPE(nn.Module):
    """Content-Conditioned Positional Encoding for patch-level Transformers.

    For each patch token x_i we gather its immediate temporal neighbourhood
    [x_{i-1}, x_i, x_{i+1}] (zero-padded at the sequence edges) and compute
    three scalar statistics that characterise the local dynamics:

        mean  — average activation level across the three neighbours
        var   — spread / volatility of those activations
        slope — linear trend direction:  (x_{i+1} − x_{i-1}) / 2

    These three scalars are projected to d_model via a small 2-layer GELU MLP
    and added to x as positional information. Crucially, no absolute index is
    used, so tokens with identical local content receive identical positional
    embeddings regardless of where they appear in the sequence.

    Adapted from: github.com/Pointcept/PointTransformerV3 — model.py :: Block.cpe
    Original: sparse 3-D convolution over point-cloud neighbours →
    Ours:     MLP over temporal neighbourhood statistics.

    Reference: Xiaoyang Wu et al., "Point Transformer V3", CVPR 2024 (Oral).

    Args:
        d_model:           Token dimension.
        dropout:           Dropout inside the MLP (default 0.1 per CLAUDE.md).
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        # MLP: 3 scalars → d_model → d_model  (2 layers, GELU)
        self.mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def _compute_conditioning(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a 3-scalar conditioning signal per position.

        Args:
            x: (B, L, d_model) — token embeddings.

        Returns:
            (B, L, 3) — [mean, var, slope] of the neighbourhood for each token.
        """
        # Pad the sequence: one zero vector on each end so that edge tokens
        # still have a full three-token neighbourhood.
        # F.pad pads the last dimension by default; we need to pad dimension 1.
        # Shape trick: pad (left=1, right=1) on the sequence dimension.
        x_pad = F.pad(x, (0, 0, 1, 1))           # (B, L+2, d_model)

        left   = x_pad[:, :-2, :]                 # (B, L, d_model)  x_{i-1}
        center = x_pad[:, 1:-1, :]                # (B, L, d_model)  x_i
        right  = x_pad[:, 2:,  :]                 # (B, L, d_model)  x_{i+1}

        # Collapse d_model → scalar per token by averaging over the feature dim.
        # This gives one "activation magnitude" per token in the neighbourhood.
        left_s   = left.mean(dim=-1)              # (B, L)
        center_s = center.mean(dim=-1)            # (B, L)
        right_s  = right.mean(dim=-1)             # (B, L)

        neighborhood = torch.stack([left_s, center_s, right_s], dim=-1)  # (B, L, 3)

        mean  = neighborhood.mean(dim=-1, keepdim=True)   # (B, L, 1)
        var   = neighborhood.var(dim=-1, keepdim=True,
                                 unbiased=False)           # (B, L, 1)
        slope = ((right_s - left_s) / 2).unsqueeze(-1)    # (B, L, 1)  central diff

        return torch.cat([mean, var, slope], dim=-1)       # (B, L, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add content-conditioned positional embedding to x.

        Args:
            x: (B, L, d_model)

        Returns:
            (B, L, d_model) — x + xCPE positional embeddings.
        """
        c       = self._compute_conditioning(x)   # (B, L, 3)
        pos_emb = self.mlp(c)                      # (B, L, d_model)
        return x + pos_emb


class RoPEEmbedding(nn.Module):
    """Rotary Positional Encoding (RoPE).

    Applied *inside* self-attention by rotating Q and K vectors.
    The token embeddings themselves are not modified — call apply_rope()
    on Q and K tensors after splitting into heads.

    Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary
    Position Embedding", arXiv:2104.09864, 2021.

    Args:
        head_dim: Dimension of each attention head (d_model // n_heads).
                  Must be even.
        max_len:  Maximum sequence length to precompute (default 512).
    """

    def __init__(self, head_dim: int, max_len: int = 512):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        half = head_dim // 2
        # theta_i = 10000^(-2i / head_dim)
        theta = 1.0 / (10_000 ** (torch.arange(0, half, dtype=torch.float32) / half))
        t     = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, theta)                        # (max_len, half)
        emb   = torch.cat([freqs, freqs], dim=-1)            # (max_len, head_dim)
        # Register as buffers so they move with .to(device) and are not parameters.
        self.register_buffer("rope_cos", emb.cos().unsqueeze(0).unsqueeze(0))  # (1,1,max_len,hd)
        self.register_buffer("rope_sin", emb.sin().unsqueeze(0).unsqueeze(0))  # (1,1,max_len,hd)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate pairs of dimensions by 90°: [x1, x2] → [-x2, x1]."""
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rope(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors.

        Args:
            q: (B, n_heads, L, head_dim)
            k: (B, n_heads, L, head_dim)

        Returns:
            Rotated q and k of the same shape.
        """
        L   = q.shape[2]
        cos = self.rope_cos[:, :, :L, :]   # (1, 1, L, head_dim)
        sin = self.rope_sin[:, :, :L, :]
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot
