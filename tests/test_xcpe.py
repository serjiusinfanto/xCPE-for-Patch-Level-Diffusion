"""
Isolation tests for the xCPE positional encoding module.

Run from the project root:
    pytest tests/test_xcpe.py -v

Tests verify three critical properties (CLAUDE.md §2.1):
  1. Output shape:  forward() returns (B, L, d_model).
  2. Position-invariance: two tokens with identical local neighbourhoods
     receive identical positional embeddings regardless of their position
     in the sequence.
  3. Gradient flow: gradients propagate through the MLP back to the input.
"""

import pytest
import torch
from src.models.positional_encoding import xCPE, FixedPositionalEmbedding


D_MODEL = 64
B       = 2
L       = 21   # typical num_patches


# ── Test 1: Output shape ──────────────────────────────────────────────────────

def test_xcpe_output_shape():
    """xCPE.forward() must return (B, L, d_model)."""
    model = xCPE(d_model=D_MODEL)
    x = torch.randn(B, L, D_MODEL)
    out = model(x)
    assert out.shape == (B, L, D_MODEL), (
        f"Expected shape ({B}, {L}, {D_MODEL}), got {out.shape}"
    )


# ── Test 2: Position-invariance ───────────────────────────────────────────────

def test_xcpe_position_invariant():
    """Tokens with identical local neighbourhoods must receive identical
    positional embeddings regardless of their absolute position in the sequence.

    Construction: build two long sequences where the central region is identical
    but shifted by some offset.  Extract the positional embeddings at those
    positions and compare.
    """
    model = xCPE(d_model=D_MODEL)
    model.eval()

    torch.manual_seed(0)
    # A shared neighbourhood of 3 tokens
    neighborhood = torch.randn(1, 3, D_MODEL)   # (1, 3, d_model)

    # Sequence A: neighborhood at positions 5, 6, 7
    seq_a = torch.randn(1, 15, D_MODEL)
    seq_a[0, 5:8, :] = neighborhood

    # Sequence B: same neighborhood but at positions 1, 2, 3
    seq_b = torch.randn(1, 15, D_MODEL)
    seq_b[0, 1:4, :] = neighborhood

    with torch.no_grad():
        # xCPE adds the positional embedding to x; we want just the PE.
        # Compute output − input to isolate the embedding.
        pe_a = model(seq_a) - seq_a   # (1, 15, d_model) — positional embeddings
        pe_b = model(seq_b) - seq_b

    # The PE at the centre position of the neighbourhood should be identical.
    # Centre in A is index 6; centre in B is index 2.
    centre_a = pe_a[0, 6, :]   # (d_model,)
    centre_b = pe_b[0, 2, :]   # (d_model,)

    assert torch.allclose(centre_a, centre_b, atol=1e-5), (
        "xCPE is NOT position-invariant: identical neighbourhoods produced "
        f"different embeddings.\n  max_diff={( centre_a - centre_b).abs().max():.2e}"
    )


# ── Test 3: Gradient flow ─────────────────────────────────────────────────────

def test_xcpe_gradient_flow():
    """Gradients must flow through the MLP back to the input tensor."""
    model = xCPE(d_model=D_MODEL)
    x = torch.randn(B, L, D_MODEL, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "No gradient on input — backward pass is broken."
    assert not torch.isnan(x.grad).any(), "NaN gradient detected."
    assert not torch.isinf(x.grad).any(), "Inf gradient detected."
    # The MLP has trainable weights; verify they also received gradients.
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for MLP param: {name}"


# ── Bonus: FixedPositionalEmbedding shape ─────────────────────────────────────

def test_fixed_pe_shape():
    """FixedPositionalEmbedding must also return (B, L, d_model)."""
    pe = FixedPositionalEmbedding(d_model=D_MODEL)
    x  = torch.randn(B, L, D_MODEL)
    out = pe(x)
    assert out.shape == (B, L, D_MODEL), (
        f"FixedPE: expected ({B}, {L}, {D_MODEL}), got {out.shape}"
    )


# ── Bonus: xCPE edge-padding correctness ─────────────────────────────────────

def test_xcpe_edge_padding():
    """Positions 0 and L-1 must produce finite outputs (no NaN from padding)."""
    model = xCPE(d_model=D_MODEL)
    x = torch.randn(1, L, D_MODEL)
    out = model(x)
    assert not torch.isnan(out[:, 0, :]).any(),   "NaN at position 0 (left edge)"
    assert not torch.isnan(out[:, -1, :]).any(),  "NaN at position L-1 (right edge)"
    assert not torch.isinf(out[:, 0, :]).any(),   "Inf at position 0 (left edge)"
    assert not torch.isinf(out[:, -1, :]).any(),  "Inf at position L-1 (right edge)"
