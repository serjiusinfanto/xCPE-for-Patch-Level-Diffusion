"""
xCPE-TimeDART — Live Presentation Demo
Streamlit app: run from project root with `streamlit run demo/app.py`

Tab 1 — Data Explorer    : explore ETTh1 or Weather test set with patch overlay
Tab 2 — Live Forecasting : real inference on ETTh1 H=96 using trained checkpoints
Tab 3 — xCPE Internals   : content-conditioned vs fixed positional embeddings
Tab 4 — Results          : aggregate MSE / MAE tables and charts for both datasets
"""

import os
import sys

# Allow `from src.*` imports when launched from project root or demo/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Suppress OpenMP conflict between PyTorch (libiomp5md) and NumPy/matplotlib (libomp)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch

from src.data.preprocess import normalize, apply_normalization
from src.models.timedart import TimeDART
from src.models.xcpe_timedart import xCPETimeDART
from src.utils.config import DataConfig, ModelConfig

# ── Project-wide constants ────────────────────────────────────────────────────

CONTEXT_LEN = 336
HORIZON     = 96
PATCH_LEN   = 16
STRIDE      = 8                                              # finetune (overlapping)
N_PATCHES   = (CONTEXT_LEN - PATCH_LEN) // STRIDE + 1      # 41

CHECKPOINT_DIR = os.path.join(ROOT, "results", "checkpoints")

DATASETS = {
    "ETTh1": {
        "path":        os.path.join(ROOT, "data", "raw", "data", "raw",
                                    "ETDataset", "ETT-small", "ETTh1.csv"),
        "train_split": 0.6,
        "val_split":   0.2,
        "freq":        "Hourly (1 h)",
        "domain":      "Electricity Transformer",
        "n_rows":      17_420,
        "n_var":       7,
    },
    "Weather": {
        "path":        os.path.join(ROOT, "data", "raw", "weather", "weather.csv"),
        "train_split": 0.7,
        "val_split":   0.1,
        "freq":        "10-minute",
        "domain":      "Meteorology",
        "n_rows":      52_696,
        "n_var":       21,
    },
}

# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset…")
def load_dataset(name: str) -> dict:
    info = DATASETS[name]
    df   = pd.read_csv(info["path"])

    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        dates = df[date_col].values
        df    = df.drop(columns=[date_col])
    else:
        dates = np.arange(len(df))

    cols   = list(df.columns)
    values = df.values.astype(np.float32)
    n      = len(values)

    train_end = int(n * info["train_split"])
    val_end   = int(n * (info["train_split"] + info["val_split"]))

    _, mean, std = normalize(values[:train_end])
    norm         = apply_normalization(values, mean, std)

    return {
        "raw": values, "norm": norm, "mean": mean, "std": std,
        "dates": dates, "cols": cols,
        "n": n, "train_end": train_end, "val_end": val_end,
    }


@st.cache_resource(show_spinner="Loading trained models…")
def load_models():
    data_cfg = DataConfig(
        dataset="ETTh1", path="",
        context_length=CONTEXT_LEN, horizon=HORIZON, patch_length=PATCH_LEN,
        train_split=0.6, val_split=0.2, finetune_stride=STRIDE,
    )
    base_cfg = ModelConfig(variant="baseline", d_model=64, n_heads=4, n_layers=3,
                           d_ff=256, patch_length=PATCH_LEN, dropout=0.1)
    xcpe_cfg = ModelConfig(variant="xcpe_all", d_model=64, n_heads=4, n_layers=3,
                           d_ff=256, patch_length=PATCH_LEN, dropout=0.1)

    baseline = TimeDART(base_cfg, data_cfg)
    baseline.load_state_dict(
        torch.load(
            os.path.join(CHECKPOINT_DIR, "ETTh1_h96_baseline_finetune_best.pt"),
            map_location="cpu", weights_only=False,
        )
    )
    baseline.eval()

    xcpe = xCPETimeDART(xcpe_cfg, data_cfg, xcpe_layers="all")
    xcpe.load_state_dict(
        torch.load(
            os.path.join(CHECKPOINT_DIR, "ETTh1_h96_xcpe_all_finetune_best.pt"),
            map_location="cpu", weights_only=False,
        )
    )
    xcpe.eval()

    return baseline, xcpe


def make_patches(context: np.ndarray, patch_len: int, stride: int) -> np.ndarray:
    """(context_len, n_var) → (n_patches, patch_len, n_var)"""
    return np.array([
        context[i : i + patch_len]
        for i in range(0, len(context) - patch_len + 1, stride)
    ])

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="xCPE-TimeDART Demo",
    page_icon="📡",
    layout="wide",
)
st.title("xCPE-TimeDART — Live Demo")
st.caption(
    "Content-Conditioned Positional Encoding for Patch-Level Diffusion in "
    "Time Series Forecasting · Spring 2026"
)

tab1, tab2, tab3 = st.tabs([
    "Data Explorer",
    "Live Forecasting",
    "xCPE Internals",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Data Explorer
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Dataset Explorer")

    # Side-by-side dataset comparison cards
    st.markdown("#### Dataset Comparison")
    cmp_cols = st.columns(2)
    for col, ds_name in zip(cmp_cols, ["ETTh1", "Weather"]):
        info = DATASETS[ds_name]
        d    = load_dataset(ds_name)
        test_rows = d["n"] - d["val_end"]
        with col:
            st.markdown(f"**{ds_name}**")
            st.dataframe(
                pd.DataFrame({
                    "Property": [
                        "Timesteps", "Variates", "Frequency", "Domain",
                        "Train / Val / Test",
                        "Context length", "Patch length", "Patches / window",
                        "Horizons evaluated",
                    ],
                    "Value": [
                        f"{info['n_rows']:,}", str(info["n_var"]), info["freq"],
                        info["domain"],
                        f"{int(info['train_split']*100)} / "
                        f"{int(info['val_split']*100)} / "
                        f"{int((1-info['train_split']-info['val_split'])*100)} %",
                        str(CONTEXT_LEN), str(PATCH_LEN),
                        str((CONTEXT_LEN - PATCH_LEN) // PATCH_LEN + 1),
                        "96 · 192 · 336 · 720",
                    ],
                }),
                width='stretch',
                hide_index=True,
            )

    st.divider()
    st.markdown("#### Explore a Dataset")

    ds_sel = st.radio("Select dataset", list(DATASETS.keys()), horizontal=True)
    data   = load_dataset(ds_sel)

    test_start = data["val_end"]
    test_norm  = data["norm"][test_start:]
    all_cols   = data["cols"]

    # Variate selector
    default_vars = all_cols[:min(3, len(all_cols))]
    sel_vars = st.multiselect("Variates to display", all_cols, default=default_vars)

    if not sel_vars:
        st.info("Select at least one variate above.")
        st.stop()

    var_idx = [all_cols.index(v) for v in sel_vars]

    # Test-set overview
    st.markdown("**Full test set — normalised scale**")
    fig_ov, ax_ov = plt.subplots(figsize=(14, 3))
    for vi, vn in zip(var_idx, sel_vars):
        ax_ov.plot(test_norm[:, vi], lw=0.7, label=vn)
    ax_ov.set_xlabel("Test timestep")
    ax_ov.set_ylabel("z-score")
    ax_ov.set_title(
        f"{ds_sel} test set  ·  {len(test_norm):,} timesteps  ·  "
        f"{len(all_cols)} variates total"
    )
    ax_ov.legend(fontsize=8, ncol=6, loc="upper right")
    st.pyplot(fig_ov, width='stretch')
    plt.close(fig_ov)

    # Window slider
    st.markdown(
        f"**Context window explorer** — {CONTEXT_LEN} timesteps → "
        f"{(CONTEXT_LEN - PATCH_LEN) // PATCH_LEN + 1} non-overlapping patches of P={PATCH_LEN}"
    )
    max_win = max(1, len(test_norm) - CONTEXT_LEN - HORIZON)
    win_pos_tab1 = st.slider(
        "Window start (test offset)", 0, max_win,
        min(300, max_win), key="tab1_win",
    )

    ctx  = data["norm"][test_start + win_pos_tab1 : test_start + win_pos_tab1 + CONTEXT_LEN]
    gt   = data["norm"][test_start + win_pos_tab1 + CONTEXT_LEN :
                        test_start + win_pos_tab1 + CONTEXT_LEN + HORIZON]

    fig_w, ax_w = plt.subplots(figsize=(14, 3.5))

    # Alternating patch shading (non-overlapping, stride=16)
    for pi in range(CONTEXT_LEN // PATCH_LEN):
        color = "#ddeeff" if pi % 2 == 0 else "#ffeedd"
        ax_w.axvspan(pi * PATCH_LEN, (pi + 1) * PATCH_LEN, alpha=0.35, color=color, lw=0)

    for vi, vn in zip(var_idx, sel_vars):
        ax_w.plot(ctx[:, vi], lw=1.3, label=vn)

    # Forecast zone
    ax_w.axvspan(CONTEXT_LEN, CONTEXT_LEN + HORIZON, alpha=0.08, color="green", lw=0)
    for vi, vn in zip(var_idx, sel_vars):
        ax_w.plot(
            range(CONTEXT_LEN, CONTEXT_LEN + HORIZON), gt[:, vi],
            lw=1.0, linestyle="--", alpha=0.6,
        )

    ax_w.axvline(CONTEXT_LEN, color="gray", lw=1.5, linestyle=":")
    ax_w.set_xlabel("Timestep within window")
    ax_w.set_ylabel("z-score")
    ax_w.set_title(
        f"Context (alternating patch colours, P={PATCH_LEN})  |  "
        f"Forecast zone: dashed, green shading (H={HORIZON})"
    )
    ax_w.legend(fontsize=8)
    st.pyplot(fig_w, width='stretch')
    plt.close(fig_w)

    # Patch statistics table (non-overlapping patches, first selected variate)
    patches_table = make_patches(ctx, PATCH_LEN, PATCH_LEN)  # non-overlapping
    vi0 = var_idx[0]
    rows = []
    for pi, patch in enumerate(patches_table):
        p = patch[:, vi0]
        slope = float(np.polyfit(np.arange(PATCH_LEN), p, 1)[0])
        rows.append({
            "Patch": pi + 1,
            "Mean (z)": round(float(p.mean()), 4),
            "Std (z)":  round(float(p.std()),  4),
            "Slope":    round(slope,             4),
            "Min (z)":  round(float(p.min()),   4),
            "Max (z)":  round(float(p.max()),   4),
        })

    with st.expander(
        f"Patch statistics table — {len(rows)} patches · variate: {sel_vars[0]}",
        expanded=False,
    ):
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Forecasting
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Live Forecasting — ETTh1, H = 96")
    st.info(
        "Running **real trained checkpoints** — baseline and xCPE-all. "
        "Predictions computed live from test-set patches.",
        #icon="",
    )

    etth1 = load_dataset("ETTh1")
    baseline_m, xcpe_m = load_models()

    t_start = etth1["val_end"]
    max_off  = etth1["n"] - t_start - CONTEXT_LEN - HORIZON

    col_sl, col_var = st.columns([3, 1])
    with col_sl:
        win_pos = st.slider(
            "Test window position  (drag to explore different regimes)",
            0, max(1, max_off), 500, key="tab2_win",
        )
    with col_var:
        var_name = st.selectbox(
            "Variate", etth1["cols"],
            index=etth1["cols"].index("OT") if "OT" in etth1["cols"] else 0,
        )

    vi = etth1["cols"].index(var_name)

    ctx_norm = etth1["norm"][t_start + win_pos : t_start + win_pos + CONTEXT_LEN]
    gt_norm  = etth1["norm"][t_start + win_pos + CONTEXT_LEN :
                             t_start + win_pos + CONTEXT_LEN + HORIZON]

    patches_np = make_patches(ctx_norm, PATCH_LEN, STRIDE)          # (41, 16, 7)
    patches_t  = torch.from_numpy(patches_np).unsqueeze(0)          # (1, 41, 16, 7)

    baseline_m.eval()
    xcpe_m.eval()
    with torch.no_grad():
        pred_base = baseline_m.forecast(patches_t).squeeze(0).numpy()  # (96, 7)
        pred_xcpe = xcpe_m.forecast(patches_t).squeeze(0).numpy()      # (96, 7)

    mse_base = float(np.mean((pred_base[:, vi] - gt_norm[:, vi]) ** 2))
    mse_xcpe = float(np.mean((pred_xcpe[:, vi] - gt_norm[:, vi]) ** 2))
    improvement = (mse_base - mse_xcpe) / mse_base * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline MSE", f"{mse_base:.4f}")
    m2.metric(
        "xCPE-all MSE", f"{mse_xcpe:.4f}",
        delta=f"{mse_xcpe - mse_base:+.4f}", delta_color="inverse",
    )
    m3.metric(
        "Improvement", f"{improvement:.1f}%",
        delta="xCPE wins" if improvement > 0 else "Baseline wins",
        delta_color="normal" if improvement > 0 else "inverse",
    )

    # Forecast plot
    show_last = 64  # context steps to show for visual continuity
    x_ctx  = np.arange(-show_last, 0)
    x_pred = np.arange(0, HORIZON)

    fig_fc, ax_fc = plt.subplots(figsize=(14, 4))
    ax_fc.plot(x_ctx,  ctx_norm[-show_last:, vi], color="steelblue",  lw=1.5,
               label=f"Context — {var_name} (last {show_last} steps)")
    ax_fc.plot(x_pred, gt_norm[:, vi],            color="black",      lw=2.0,
               label="Ground truth")
    ax_fc.plot(x_pred, pred_base[:, vi],          color="tomato",     lw=1.5,
               linestyle="--", label=f"Baseline  (MSE = {mse_base:.4f})")
    ax_fc.plot(x_pred, pred_xcpe[:, vi],          color="seagreen",   lw=1.5,
               linestyle="--", label=f"xCPE-all  (MSE = {mse_xcpe:.4f})")

    ax_fc.axvline(0, color="gray", lw=1.2, linestyle=":")
    ax_fc.axvspan(0, HORIZON, alpha=0.04, color="green")
    ax_fc.set_xlabel("Timestep  (0 = forecast start)")
    ax_fc.set_ylabel("z-score")
    ax_fc.set_title(f"ETTh1 · {var_name} · H={HORIZON}  (test window offset = {win_pos})")
    ax_fc.legend(fontsize=9)
    st.pyplot(fig_fc, width='stretch')
    plt.close(fig_fc)

    st.caption(
        "Drag the slider to different test windows. "
        "High-variance regimes (around offsets 800–1200) tend to show the largest xCPE advantage."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — xCPE Internals
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("xCPE Internals — Content-Conditioned vs Fixed Positional Embeddings")
    st.markdown(
        "**Fixed PE** assigns each position a fixed learned vector — "
        "identical regardless of what the patch contains. "
        "It will never change when you drag the slider. "
        "**xCPE** computes each positional embedding from the local neighbourhood's "
        "[mean, variance, linear slope], so it produces *different embeddings for different content*."
    )

    etth1_3 = load_dataset("ETTh1")
    bm3, xm3 = load_models()

    ctx_s3    = etth1_3["val_end"] + win_pos
    ctx_norm3 = etth1_3["norm"][ctx_s3 : ctx_s3 + CONTEXT_LEN]

    patches3   = make_patches(ctx_norm3, PATCH_LEN, STRIDE)            # (41, 16, 7)
    patches_t3 = torch.from_numpy(patches3.copy()).unsqueeze(0)        # (1, 41, 16, 7)

    B, L, p, n_var = patches_t3.shape
    x_ci  = patches_t3.permute(0, 3, 1, 2).reshape(B * n_var, L, p)   # (7, 41, 16)
    x_one = x_ci[:1].clone()                                           # (1, 41, 16) — variate 0

    bm3.eval()
    xm3.eval()
    with torch.no_grad():
        embedded = xm3.patch_embed(x_one)                              # (1, 41, 64)
        cond_3   = xm3.pos_enc._compute_conditioning(embedded)         # (1, 41, 3)
        xcpe_emb = xm3.pos_enc.mlp(cond_3).squeeze(0).numpy()         # (41, 64)
        cond_np  = cond_3.squeeze(0).numpy()                           # (41, 3)

        positions = torch.arange(L)
        fixed_emb = bm3.pos_enc.embedding(positions).numpy()           # (41, 64)

    # Window info + stats banner
    st.info(
        f"Showing window offset **{win_pos}**  ·  "
        f"xCPE embedding range: **{xcpe_emb.min():.3f}** → **{xcpe_emb.max():.3f}**  ·  "
        f"Fixed PE range: **{fixed_emb.min():.3f}** → **{fixed_emb.max():.3f}** (never changes)",
        #icon="",
    )

    # ── Heatmaps — independent colour scales so xCPE variation is visible ─────
    h_col1, h_col2 = st.columns(2)

    with h_col1:
        st.markdown("**Fixed PE** — same for every window (by design)")
        vf = np.abs(fixed_emb).max() * 0.9
        fig_fp, ax_fp = plt.subplots(figsize=(6, 4.5))
        im_fp = ax_fp.imshow(fixed_emb.T, aspect="auto", cmap="RdBu_r", vmin=-vf, vmax=vf)
        ax_fp.set_xlabel("Patch position (0 → 40)")
        ax_fp.set_ylabel("d_model dimension (0 → 63)")
        ax_fp.set_title("Fixed PE")
        plt.colorbar(im_fp, ax=ax_fp, shrink=0.75)
        st.pyplot(fig_fp, width='stretch')
        plt.close(fig_fp)

    with h_col2:
        st.markdown("**xCPE** — different for every window (reflects content)")
        vx = np.abs(xcpe_emb).max() * 0.9
        fig_xp, ax_xp = plt.subplots(figsize=(6, 4.5))
        im_xp = ax_xp.imshow(xcpe_emb.T, aspect="auto", cmap="RdBu_r", vmin=-vx, vmax=vx)
        ax_xp.set_xlabel("Patch position (0 → 40)")
        ax_xp.set_ylabel("d_model dimension (0 → 63)")
        ax_xp.set_title(f"xCPE  ·  window offset = {win_pos} ")
        plt.colorbar(im_xp, ax=ax_xp, shrink=0.75)
        st.pyplot(fig_xp, width='stretch')
        plt.close(fig_xp)

    # ── The 3 conditioning signals — most visually dramatic change ────────────
    st.markdown(
        "#### The 3 scalars that drive xCPE — these change with every window"
    )
    st.caption(
        "Each bar below is computed live from the current window's patch embeddings. "
        "Drag Tab 2's slider to a very different offset (e.g. 0 vs 1000) "
        "and return here — the bar shapes will be noticeably different."
    )

    fig_sig, axes_sig = plt.subplots(1, 3, figsize=(14, 3.0), sharey=False)
    sig_info = [
        ("Neighbourhood Mean",            "#4878cf", "Avg activation level per patch"),
        ("Neighbourhood Variance",         "#e07b54", "Volatility / spread per patch"),
        ("Slope  (central difference)",    "#2ca02c", "Linear trend direction per patch"),
    ]
    patch_x = np.arange(N_PATCHES)
    for j, (title, color, xlabel) in enumerate(sig_info):
        vals = cond_np[:, j]
        axes_sig[j].bar(patch_x, vals, color=color, alpha=0.75, width=0.8)
        axes_sig[j].axhline(0, color="black", lw=0.7, linestyle="--")
        axes_sig[j].set_title(f"{title}\n(window offset = {win_pos})",
                               fontsize=9, fontweight="bold")
        axes_sig[j].set_xlabel(xlabel, fontsize=8)
        axes_sig[j].set_ylabel("Value")
        axes_sig[j].set_xlim(-0.5, N_PATCHES - 0.5)
    plt.tight_layout()
    st.pyplot(fig_sig, width='stretch')
    plt.close(fig_sig)

    st.caption(
        "These 3 scalars per patch feed a 2-layer GELU MLP (the only new parameters "
        "added by this project, ~200 weights) to produce the positional embedding. "
        "No absolute position index is used anywhere."
    )
