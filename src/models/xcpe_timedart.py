"""
xCPETimeDART and RoPETimeDART — TimeDART subclasses for Phase 3 ablations.

xCPETimeDART swaps the positional encoding module with xCPE (content-conditioned).
Three placement modes are supported:
  'all'   — xCPE replaces FixedPositionalEmbedding globally (applied once before all layers).
  'early' — No global pos_enc; xCPE injected before the first 2 Transformer layers.
  'late'  — No global pos_enc; xCPE injected before the last 2 Transformer layers.

RoPETimeDART replaces the standard nn.MultiheadAttention blocks with a custom
attention implementation that applies Rotary Position Embeddings to Q and K.
The global pos_enc is disabled (nn.Identity) since RoPE carries position info
inside attention.

Adapted from:
  TimeDART — github.com/Melmaphother/TimeDART (ICLR 2025)
  xCPE     — github.com/Pointcept/PointTransformerV3, model.py::Block (CVPR 2024)
  RoPE     — Su et al., arXiv:2104.09864 (2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.positional_encoding import xCPE, RoPEEmbedding
from src.models.timedart import TimeDART, _TransformerBlock
from src.utils.config import ModelConfig, DataConfig


# ── RoPE-aware Transformer block ──────────────────────────────────────────────

class _RoPETransformerBlock(nn.Module):
    """Pre-norm Transformer block with manual attention that applies RoPE to Q and K.

    Mirrors _TransformerBlock in timedart.py but replaces nn.MultiheadAttention
    with a manual scaled dot-product attention so we can inject RoPE rotations.

    Args:
        d_model:  Token dimension.
        n_heads:  Number of attention heads (d_model must be divisible by n_heads).
        d_ff:     Feed-forward hidden dimension.
        dropout:  Dropout probability.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = self.head_dim ** -0.5

        # Q, K, V projections and output projection
        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rope = RoPEEmbedding(self.head_dim)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1    = nn.LayerNorm(d_model)
        self.norm2    = nn.LayerNorm(d_model)
        self.drop     = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, D) → (B, n_heads, L, head_dim)."""
        B, L, D = x.shape
        return x.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_heads, L, head_dim) → (B, L, D)."""
        B, H, L, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * hd)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-norm self-attention with RoPE
        x_norm = self.norm1(x)
        q = self._split_heads(self.q_proj(x_norm))   # (B, H, L, hd)
        k = self._split_heads(self.k_proj(x_norm))
        v = self._split_heads(self.v_proj(x_norm))

        q, k = self.rope.apply_rope(q, k)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, L, L)
        if mask is not None:
            # mask shape: (L, L) — broadcasts to (B, H, L, L)
            attn = attn + mask
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = self.out_proj(self._merge_heads(torch.matmul(attn, v)))
        x   = x + self.drop(out)

        # Pre-norm FFN
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


# ── xCPETimeDART ─────────────────────────────────────────────────────────────

class xCPETimeDART(TimeDART):
    """TimeDART with xCPE replacing FixedPositionalEmbedding.

    The ONLY structural change from the parent class is in how positional
    information is injected.  All other components (patch embedding, Transformer
    backbone, diffusion head, forecast head) are inherited unchanged.

    Args:
        model_config: Hyperparameters — same as TimeDART.
        data_config:  Data parameters — same as TimeDART.
        xcpe_layers:  Where to apply xCPE:
                      'all'   — global replacement of pos_enc (before all layers).
                      'early' — injected before layers 0 and 1 (no global pos_enc).
                      'late'  — injected before layers 1 and 2 (no global pos_enc).
    """

    VALID_MODES = {"all", "early", "late"}

    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataConfig,
        xcpe_layers: str = "all",
    ):
        if xcpe_layers not in self.VALID_MODES:
            raise ValueError(
                f"xcpe_layers must be one of {self.VALID_MODES}, got {xcpe_layers!r}"
            )
        super().__init__(model_config, data_config)

        d = model_config.d_model
        n = model_config.n_layers
        self.xcpe_layers = xcpe_layers

        if xcpe_layers == "all":
            # Simple case: xCPE is the global positional encoding.
            # _transformer_forward() is inherited unchanged.
            self.pos_enc = xCPE(d, dropout=model_config.dropout)

        else:
            # For early/late, positional encoding is applied per-layer inside
            # _transformer_forward(), so disable the global pos_enc.
            self.pos_enc       = nn.Identity()
            self.xcpe          = xCPE(d, dropout=model_config.dropout)
            # Determine which layer *indices* (0-based) get xCPE applied before them.
            if xcpe_layers == "early":
                # First two layers: indices 0, 1
                self.xcpe_indices = frozenset(range(min(2, n)))
            else:  # "late"
                # Last two layers: indices n-2, n-1
                self.xcpe_indices = frozenset(range(max(0, n - 2), n))

    def _transformer_forward(
        self, x: torch.Tensor, causal: bool
    ) -> torch.Tensor:
        if self.xcpe_layers == "all":
            # Delegate to parent — pos_enc is already xCPE
            return super()._transformer_forward(x, causal)

        # early / late: inject xCPE before the designated layer indices.
        mask = self._causal_mask(x.size(1), x.device) if causal else None
        for i, layer in enumerate(self.layers):
            if i in self.xcpe_indices:
                x = self.xcpe(x)
            x = layer(x, mask)
        return self.norm(x)


# ── RoPETimeDART ─────────────────────────────────────────────────────────────

class RoPETimeDART(TimeDART):
    """TimeDART with Rotary Position Embeddings applied inside attention.

    The standard nn.MultiheadAttention blocks are replaced with
    _RoPETransformerBlock, which applies RoPE to Q and K before computing
    attention scores.  The global pos_enc is set to nn.Identity() because
    RoPE encodes position directly inside the attention operation.

    Reference: Su et al., "RoFormer", arXiv:2104.09864, 2021.

    Args:
        model_config: Hyperparameters — same as TimeDART.
        data_config:  Data parameters — same as TimeDART.
    """

    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super().__init__(model_config, data_config)

        # Disable the global positional encoding — RoPE operates inside attention.
        self.pos_enc = nn.Identity()

        # Replace all Transformer blocks with RoPE-aware versions.
        self.layers = nn.ModuleList([
            _RoPETransformerBlock(
                d_model  = model_config.d_model,
                n_heads  = model_config.n_heads,
                d_ff     = model_config.d_ff,
                dropout  = model_config.dropout,
            )
            for _ in range(model_config.n_layers)
        ])
