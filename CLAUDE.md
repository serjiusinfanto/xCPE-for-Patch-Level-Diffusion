# CLAUDE.md — Project Implementation Guide
## Content-Conditioned Positional Encoding for Patch-Level Diffusion in Time Series Forecasting

> **Goal:** Transplant xCPE (extended Conditional Positional Encoding) from PTv3 (via DiPT) into TimeDART's
> Transformer encoder and evaluate whether content-aware positional encoding improves forecasting
> accuracy on non-stationary time series benchmarks (ETT, Weather).
>
> **Hardware target:** NVIDIA RTX 4070 · 8 GB VRAM · PyTorch 2.x · FP16 mixed precision

---

## Repository Structure

```
project/
├── CLAUDE.md                        ← this file
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                         ← downloaded CSVs (ETTh1, ETTh2, ETTm1, ETTm2, Weather)
│   └── processed/                   ← normalized, patched tensors cached as .pt files
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py               ← TimeSeriesDataset, patch collation
│   │   └── preprocess.py            ← z-score normalization, train/val/test splitting
│   ├── models/
│   │   ├── __init__.py
│   │   ├── timedart.py              ← original TimeDART (causal Transformer + diffusion)
│   │   ├── positional_encoding.py   ← BOTH fixed embedding AND xCPE module live here
│   │   └── xcpe_timedart.py         ← TimeDART subclass with xCPE swapped in
│   ├── diffusion/
│   │   ├── __init__.py
│   │   ├── noise_scheduler.py       ← DDPM linear/cosine beta schedule
│   │   └── denoising.py             ← forward/reverse diffusion at patch level
│   ├── training/
│   │   ├── __init__.py
│   │   ├── pretrain.py              ← self-supervised diffusion pre-training loop
│   │   └── finetune.py              ← supervised forecasting fine-tuning loop
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py               ← MSE, MAE across horizons
│   └── utils/
│       ├── __init__.py
│       ├── config.py                ← dataclass configs for model, training, data
│       └── logging.py               ← Weights & Biases or CSV logger
├── scripts/
│   ├── download_data.sh             ← downloads all datasets from HuggingFace (thuml/Time-Series-Library)
│   ├── run_pretrain.sh              ← launches pre-training for a given config
│   └── run_finetune.sh              ← launches fine-tuning for a given config
├── configs/
│   ├── baseline_etth1.yaml          ← vanilla TimeDART on ETTh1
│   ├── xcpe_etth1.yaml              ← xCPE-TimeDART on ETTh1
│   └── ablations/
│       ├── xcpe_early_layers.yaml
│       ├── xcpe_late_layers.yaml
│       └── rope_etth1.yaml
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_positional_encoding_viz.ipynb   ← visualise fixed vs xCPE embeddings
│   └── 03_results_analysis.ipynb
└── results/
    ├── logs/
    └── tables/                      ← CSV files of MSE/MAE per dataset/horizon
```

---

## Source Repositories to Clone

```bash
# TimeDART — base model
git clone https://github.com/Melmaphother/TimeDART.git reference/TimeDART

# PTv3 — xCPE original source (CVPR 2024 Oral)
# xCPE was introduced in PTv3, not DiPT. DiPT uses xCPE from PTv3.
# The xCPE implementation lives in model.py → Block class
git clone https://github.com/Pointcept/PointTransformerV3.git reference/PTv3
# Full framework (alternative, contains same code):
# git clone https://github.com/Pointcept/Pointcept.git reference/Pointcept

# Data


# Download all datasets from HuggingFace (CC BY 4.0)
# pip install huggingface_hub
# python scripts/download_data.py
# See scripts/download_data.py:
#   from huggingface_hub import hf_hub_download
#   for f in ["ETT-small/ETTh1.csv","ETT-small/ETTh2.csv",
#             "ETT-small/ETTm1.csv","ETT-small/ETTm2.csv","weather/weather.csv"]:
#       hf_hub_download("thuml/Time-Series-Library", f, repo_type="dataset", local_dir="data/raw")
```

> **Do not copy-paste TimeDART or PTv3 code wholesale into src/.** Read the reference
> implementations to understand them, then write clean, well-commented versions in src/.
> Cite both repos in every file that adapts their logic.

---

## Environment Setup

```bash
python -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install accelerate transformers einops pyyaml wandb pandas numpy scikit-learn
pip install matplotlib seaborn jupyter
```

**requirements.txt** must pin exact versions after setup:
```
torch==2.2.*
accelerate==0.28.*
einops==0.7.*
```

FP16 mixed precision is enabled via `accelerate` — always use `accelerate launch` for
training scripts, never plain `python`, to ensure VRAM efficiency on the RTX 4070.

---

## Key Design Decisions (read before writing any code)

### 1. Patching
- Patch length **P = 16** timesteps (matches TimeDART paper)
- Patches are non-overlapping during pre-training; overlapping with stride S=8 during fine-tuning
- Each patch becomes one token fed to the Transformer

### 2. Diffusion (pre-training only)
- **Linear beta schedule**: β₁=1e-4 → β_T=0.02, T=1000 steps
- At each pre-training step: sample a random noise level t, corrupt a random subset of patches, train the encoder to denoise them back
- The diffusion loss is the standard MSE between predicted and actual clean patch

### 3. Positional Encoding — the core transplant
Two classes must coexist in `src/models/positional_encoding.py`:

**Class A — FixedPositionalEmbedding (baseline)**
```python
# Standard learned absolute embedding: lookup table indexed by patch position
# shape: (1, max_seq_len, d_model)
# Identical to TimeDART's original implementation
```

**Class B — xCPE (our contribution)**
```python
# Content-conditioned positional encoding adapted from PTv3 (CVPR 2024)
# For each patch token x_i (shape: d_model):
#   1. Gather local neighborhood: [x_{i-1}, x_i, x_{i+1}] (pad edges with zeros)
#   2. Compute conditioning signal c_i from neighborhood:
#      c_i = [mean(neighborhood), var(neighborhood), linear_trend_slope(neighborhood)]
#      → project c_i to d_model via a small MLP (2 layers, GELU)
#   3. Positional embedding = c_i  (NOT added to x_i here — the Transformer adds it)
# No absolute index is used anywhere in xCPE
```

The xCPE MLP is the only trainable component added by this project.
It has ~3 × d_model parameters — negligible relative to the Transformer.

### 4. Model variants (all sharing the same Transformer backbone)
| Variant | Positional Encoding | Purpose |
|---------|-------------------|---------|
| `baseline` | FixedPositionalEmbedding | Primary comparison |
| `xcpe_all` | xCPE at all layers | Full proposal |
| `xcpe_early` | xCPE at layers 1–2 only, fixed elsewhere | Ablation A2 |
| `xcpe_late` | xCPE at last 2 layers only, fixed elsewhere | Ablation A3 |
| `rope` | RoPE (rotary) | Ablation B — sanity check |

### 5. Transformer backbone (keep small — VRAM constraint)
- `d_model = 64`
- `n_heads = 4`
- `n_layers = 3`
- `d_ff = 256`
- `dropout = 0.1`
- Causal masking during pre-training; no masking during fine-tuning

---

## Evaluation Protocol

For every model variant × dataset × horizon:
- Report **MSE** and **MAE** on the held-out test set
- Horizons: H ∈ {96, 192, 336, 720}
- Datasets: ETTh1, ETTm1, Weather (minimum); ETTh2, ETTm2 if time allows
- Run each experiment with **3 random seeds** and report mean ± std

Final results table shape:
```
| Model      | ETTh1-96 MSE | ETTh1-96 MAE | ... | Weather-720 MSE | Weather-720 MAE |
|------------|-------------|-------------|-----|----------------|----------------|
| Baseline   |             |             |     |                |                |
| xCPE (all) |             |             |     |                |                |
| xCPE early |             |             |     |                |                |
| xCPE late  |             |             |     |                |                |
| RoPE       |             |             |     |                |                |
```

---

---

# PHASE 1 — Foundation and Data Pipeline (25%)

**Objective:** A clean, validated data pipeline and project scaffold.
**Done when:** You can load any dataset, produce correctly shaped patch tensors, and
confirm train/val/test splits match TimeDART paper statistics.

## Tasks

### 1.1 — Project scaffold
- [ ] Create the full directory structure listed above
- [ ] Set up virtual environment and install all dependencies
- [ ] Clone all reference repositories
- [ ] Write `configs/baseline_etth1.yaml` with all hyperparameters as the single source of truth

### 1.2 — Download and inspect data
- [ ] Run `scripts/download_data.sh` to fetch ETT and Weather CSVs into `data/raw/`
- [ ] Open `notebooks/01_data_exploration.ipynb`
- [ ] Plot each series; note obvious non-stationarity (seasonal shifts, regime changes)
- [ ] Record: number of timesteps, number of variates, presence of missing values per dataset

### 1.3 — Implement `src/data/preprocess.py`
```python
# Must implement:
def normalize(df, split='train'):
    # z-score per channel using ONLY training-set mean/std
    # returns normalized df + stored mean/std for inverse transform

def split_dataset(df, dataset_name):
    # ETT: 60/20/20 by time order (no shuffling)
    # Weather: 70/10/20 by time order
    # returns train_df, val_df, test_df
```

### 1.4 — Implement `src/data/dataset.py`
```python
class TimeSeriesDataset(torch.utils.data.Dataset):
    # __init__: accepts df, patch_length=16, horizon, stride, split
    # __getitem__: returns (context_patches, target) where
    #   context_patches: shape (num_patches, patch_length, n_variates)
    #   target: shape (horizon, n_variates)
    # __len__: number of valid windows in the split
```

### 1.5 — Validation checks (write these as pytest tests in `tests/test_data.py`)
- [ ] No data leakage: val/test normalization uses ONLY train mean/std
- [ ] Patch shape is exactly `(num_patches, 16, n_variates)`
- [ ] No NaN or Inf values in any split
- [ ] Split sizes match expected percentages (±1%)
- [ ] Reproducibility: same seed → same batch every time

### 1.6 — Config system
- [ ] Implement `src/utils/config.py` using Python dataclasses
- [ ] Config must cover: `DataConfig`, `ModelConfig`, `TrainingConfig`
- [ ] All YAML configs load into these dataclasses via a `load_config(path)` function

## Phase 1 Deliverable
Running this command must work without errors:
```bash
python -c "
from src.data.dataset import TimeSeriesDataset
from src.utils.config import load_config
cfg = load_config('configs/baseline_etth1.yaml')
ds = TimeSeriesDataset('data/raw/ETDataset/ETT-small/ETTh1.csv', cfg.data)
print(ds[0][0].shape, ds[0][1].shape)  # expect: (seq_len//16, 16, 7) and (96, 7)
"
```

---

# PHASE 2 — Baseline Model: TimeDART Reproduction (25%)

**Objective:** A fully working TimeDART baseline that reproduces (within ~5%) the MSE/MAE
numbers reported in the original paper on ETTh1 at H=96.
**Done when:** Baseline pre-training converges and fine-tuned MSE on ETTh1-96 is within
tolerance of the paper's reported 0.370 MSE.

## Tasks

### 2.1 — Implement `src/models/positional_encoding.py`

```python
class FixedPositionalEmbedding(nn.Module):
    """
    Standard learned absolute positional embedding.
    Adapted from: github.com/Melmaphother/TimeDART
    """
    def __init__(self, d_model: int, max_len: int = 512):
        # lookup table: nn.Embedding(max_len, d_model)
        # forward(x): x shape (B, L, d_model) → add embedding for positions 0..L-1

class xCPE(nn.Module):
    """
    Content-conditioned positional encoding.
    Adapted from: github.com/Pointcept/PointTransformerV3 — PTv3 Block class (spatial → temporal domain)
    
    For each patch token, computes a positional embedding conditioned on
    the local neighborhood's statistics rather than the absolute index.
    """
    def __init__(self, d_model: int, neighborhood_size: int = 3):
        # MLP: input_dim=3 (mean, var, slope) → hidden=d_model → output=d_model
        # Activation: GELU

    def _compute_conditioning(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        # For each position i, gather [x_{i-1}, x_i, x_{i+1}], pad edges
        # Compute: mean across neighborhood dim, var across neighborhood dim,
        #          linear trend slope via least-squares over the 3 positions
        # Returns: (B, L, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns positional embeddings of shape (B, L, d_model)
        # These are ADDED to x inside the Transformer, not returned separately
```

> **Critical:** xCPE must be tested in isolation before plugging into the Transformer.
> Write `tests/test_xcpe.py` that verifies: (1) output shape is correct, (2) two tokens
> with identical local neighborhoods produce identical embeddings regardless of position,
> (3) gradients flow through the MLP.

### 2.2 — Implement `src/diffusion/noise_scheduler.py`
```python
class LinearBetaSchedule:
    # beta_start=1e-4, beta_end=0.02, T=1000
    # Precompute: alphas, alpha_bars, sqrt_alpha_bars, sqrt_one_minus_alpha_bars
    
    def add_noise(self, x_0, t):
        # x_0: clean patches (B, num_patches, patch_len, d)
        # t: noise level per sample (B,)
        # returns: x_t (noisy), noise (the actual noise added)
    
    def predict_x0(self, x_t, predicted_noise, t):
        # reverse step: given predicted noise, recover x_0 estimate
```

### 2.3 — Implement `src/models/timedart.py`
```python
class TimeDART(nn.Module):
    """
    Causal Transformer encoder with patch-level diffusion pre-training.
    Reference: Wang et al., ICLR 2025. github.com/Melmaphother/TimeDART
    """
    def __init__(self, config: ModelConfig):
        # Components:
        # 1. Patch embedding: Linear(patch_length * n_variates → d_model)
        # 2. Positional encoding: FixedPositionalEmbedding (swappable)
        # 3. Causal Transformer encoder: nn.TransformerEncoder with causal mask
        # 4. Diffusion head: Linear(d_model → patch_length * n_variates) for denoising
        # 5. Forecast head: Linear(d_model * num_patches → horizon * n_variates) for finetuning

    def encode(self, patches):
        # patches: (B, L, patch_len, n_var) → embed → add pos enc → transformer → (B, L, d_model)

    def denoise(self, noisy_patches, t):
        # Used during pre-training: encode noisy patches, predict clean patches

    def forecast(self, patches):
        # Used during fine-tuning: encode patches, flatten, project to horizon
```

### 2.4 — Implement `src/training/pretrain.py`
```python
# Pre-training loop:
# for each batch:
#   1. Sample random noise level t ~ Uniform(1, T)
#   2. Select random subset of patches to corrupt (50% of patches)
#   3. Add noise: x_t = noise_scheduler.add_noise(x_0, t)
#   4. Forward pass: predicted_noise = model.denoise(x_t, t)
#   5. Loss: MSE(predicted_noise, actual_noise)
#   6. Backward + optimizer step (AdamW, lr=1e-4, weight_decay=1e-5)
# Save checkpoint every 10 epochs
# Log: train_loss, val_loss per epoch
```

### 2.5 — Implement `src/training/finetune.py`
```python
# Fine-tuning loop:
# Load pre-trained encoder weights (freeze for first 5 epochs, then unfreeze)
# for each batch:
#   1. Forward pass: predictions = model.forecast(patches)
#   2. Loss: MSE(predictions, targets)
#   3. Backward + step (AdamW, lr=5e-5)
# Save best checkpoint by val MSE
# Evaluate on test set using best checkpoint
```

### 2.6 — Run baseline experiment
```bash
accelerate launch scripts/run_pretrain.sh --config configs/baseline_etth1.yaml
accelerate launch scripts/run_finetune.sh --config configs/baseline_etth1.yaml
```

## Phase 2 Deliverables
- [ ] Pre-training loss curve converges (monotonically decreasing over 50 epochs)
- [ ] ETTh1-96 test MSE within 5% of paper's reported 0.370
- [ ] ETTh1-96 test MAE within 5% of paper's reported 0.400
- [ ] Checkpoint saved to `results/checkpoints/baseline_etth1_h96.pt`
- [ ] `tests/test_xcpe.py` all passing

---

# PHASE 3 — xCPE Integration and Ablations (25%)

**Objective:** Swap xCPE into TimeDART and run the full ablation matrix.
**Done when:** All five model variants (baseline, xcpe_all, xcpe_early, xcpe_late, rope)
have been trained and evaluated on ETTh1 across all four horizons.

## Tasks

### 3.1 — Implement `src/models/xcpe_timedart.py`
```python
class xCPETimeDART(TimeDART):
    """
    TimeDART with xCPE replacing FixedPositionalEmbedding.
    The ONLY change from the parent class is the positional encoding module.
    Everything else (Transformer, diffusion head, forecast head) is inherited unchanged.
    """
    def __init__(self, config: ModelConfig, xcpe_layers: str = 'all'):
        # xcpe_layers: 'all' | 'early' | 'late'
        # 'all':   replace pos enc globally before the Transformer
        # 'early': apply xCPE only before layers 1-2, fixed embedding before layers 3+
        # 'late':  apply xCPE only before the last 2 layers, fixed elsewhere
        super().__init__(config)
        self.pos_enc = xCPE(config.d_model)   # replaces FixedPositionalEmbedding
```

> **Note on 'early'/'late' variants:** To apply encoding per-layer, you need to expose
> the Transformer's layer stack and insert the encoding at intermediate points. Use
> `nn.ModuleList` instead of `nn.TransformerEncoder` so you can iterate manually
> and inject xCPE between specific layers.

### 3.2 — Implement RoPE variant (ablation B)
```python
class RoPEEmbedding(nn.Module):
    """
    Rotary Positional Encoding.
    Reference: Su et al., 2024. arXiv:2104.09864
    Applied inside attention (modify Q and K, not the token embeddings directly).
    """
    def __init__(self, d_model: int):
        # Precompute rotation frequencies: theta_i = 10000^(-2i/d_model)
    
    def rotate_half(self, x): ...
    
    def apply_rope(self, q, k, seq_len):
        # q, k: (B, heads, L, head_dim)
        # returns rotated q, k
```

Integrate RoPE by subclassing the Transformer's attention module.

### 3.3 — Config files for all variants
Create one YAML per variant in `configs/`. Each must specify:
```yaml
model:
  variant: xcpe_all        # baseline | xcpe_all | xcpe_early | xcpe_late | rope
  d_model: 64
  n_heads: 4
  n_layers: 3
  d_ff: 256
  patch_length: 16

training:
  pretrain_epochs: 50
  finetune_epochs: 30
  pretrain_lr: 1e-4
  finetune_lr: 5e-5
  batch_size: 32
  seed: 42

data:
  dataset: ETTh1
  horizon: 96
  context_length: 336
```

### 3.4 — Run all ablations
For each variant, run pre-training + fine-tuning at all four horizons:
```bash
for VARIANT in baseline xcpe_all xcpe_early xcpe_late rope; do
  for H in 96 192 336 720; do
    accelerate launch scripts/run_pretrain.sh --config configs/${VARIANT}_etth1.yaml --horizon $H
    accelerate launch scripts/run_finetune.sh --config configs/${VARIANT}_etth1.yaml --horizon $H
  done
done
```

> **VRAM tip:** If any run OOMs, reduce batch_size from 32 → 16 before any other change.
> Do NOT reduce d_model — it will invalidate comparison against the baseline.

### 3.5 — Results collection
- [ ] Implement `src/evaluation/metrics.py`: `compute_mse(pred, target)`, `compute_mae(pred, target)`
- [ ] Implement `scripts/collect_results.py`: reads all checkpoint logs, writes `results/tables/etth1_results.csv`
- [ ] Run with 3 seeds (42, 123, 456) per variant × horizon combination
- [ ] Populate the results table defined in the Evaluation Protocol section above

### 3.6 — Sanity checks before moving to Phase 4
- [ ] xCPE model has more parameters than baseline? (expected: ~3×d_model extra, i.e., ~200 params)
- [ ] xCPE positional embeddings visually differ from fixed embeddings (check notebook 02)
- [ ] No variant took longer than 2× baseline training time (if so, investigate bottleneck)
- [ ] All variants produce finite (non-NaN) losses throughout training

## Phase 3 Deliverable
`results/tables/etth1_results.csv` populated with MSE and MAE for all 5 variants × 4 horizons × 3 seeds.

---

# PHASE 4 — Extended Evaluation, Analysis, and Write-Up (25%)

**Objective:** Extend results to Weather dataset, produce all visualizations, interpret
findings, and deliver the final report.
**Done when:** Full results table is populated, all figures are generated, and the report
draft is complete.

## Tasks

### 4.1 — Extend experiments to Weather dataset
```bash
# Repeat Phase 3 experiments with dataset: Weather
# Use the same hyperparameters — do NOT tune separately per dataset
for VARIANT in baseline xcpe_all xcpe_early xcpe_late rope; do
  for H in 96 192 336 720; do
    accelerate launch scripts/run_pretrain.sh --config configs/${VARIANT}_weather.yaml --horizon $H
    accelerate launch scripts/run_finetune.sh --config configs/${VARIANT}_weather.yaml --horizon $H
  done
done
```

### 4.2 — Visualizations (all in `notebooks/`)

**Notebook 02 — Positional Encoding Analysis:**
- [ ] Plot fixed embedding matrix as heatmap (rows = positions, cols = d_model dims)
- [ ] Plot xCPE embedding matrix for the same sequence — do similar dynamics get similar rows?
- [ ] t-SNE of positional embeddings: fixed (colored by position) vs xCPE (colored by local variance)
  - Hypothesis to visualize: xCPE clusters by regime, fixed clusters by index

**Notebook 03 — Results Analysis:**
- [ ] Bar chart: MSE per model variant at H=96 for ETTh1 and Weather side by side
- [ ] Line chart: MSE vs horizon (96→720) per variant — does xCPE's advantage grow with horizon?
- [ ] Error bar plot showing mean ± std across 3 seeds
- [ ] Improvement heatmap: (xCPE_all MSE − baseline MSE) / baseline MSE × 100% per dataset × horizon

### 4.3 — Interpretation (write these as comments/markdown in notebooks)

Answer each of the following questions with evidence from your results:

1. **Does xCPE improve over fixed embeddings overall?**
   Point to specific cells in the results table.

2. **Does the benefit vary by horizon?**
   A longer horizon requires the model to rely more on structural patterns vs. position —
   if xCPE helps more at H=720 than H=96, this supports the non-stationarity hypothesis.

3. **Is content-conditioning specifically responsible, or is any non-fixed encoding sufficient?**
   Compare xcpe_all vs. rope — if both improve, it's about flexibility; if only xCPE improves,
   it's specifically about content-conditioning.

4. **Where in the Transformer does xCPE matter most?**
   Compare xcpe_early vs. xcpe_late. Early layers handle local pattern detection;
   late layers handle sequence-level reasoning.

5. **What if results do not improve?**
   Acceptable explanations to investigate and document:
   - TimeDART's causal masking already implicitly encodes local content, making xCPE redundant
   - The ETT datasets are too stationary for content-conditioning to matter
   - The xCPE MLP is undertrained (try increasing its capacity: 2 → 3 layers)

### 4.4 — Final report structure
The report should be structured as follows (adapt the proposal document from Phase 0):

```
1. Introduction & Motivation         (~0.5 page)
2. Background                        (~1 page)  — TimeDART, PTv3/xCPE
3. Method                            (~1 page)  — xCPE adaptation, architecture diagram
4. Experiments                       (~1.5 pages) — datasets, baselines, ablation matrix
5. Results                           (~1 page)  — tables + key figures
6. Analysis & Discussion             (~0.5 page) — answer the 5 questions above
7. Conclusion                        (~0.25 page)
References
```

### 4.5 — Code cleanup checklist
- [ ] Every file has a module docstring citing its source paper/repo
- [ ] No dead code, commented-out experiments, or hardcoded paths
- [ ] All hyperparameters are in YAML configs — zero magic numbers in Python files
- [ ] `README.md` explains how to reproduce every number in the results table
- [ ] `requirements.txt` is pinned and tested on a fresh venv

## Phase 4 Deliverable
- `results/tables/etth1_results.csv` and `results/tables/weather_results.csv` complete
- All notebooks rendered with outputs
- Final report PDF submitted

---

## Common Pitfalls and How to Avoid Them

| Pitfall | Prevention |
|---------|-----------|
| Data leakage through normalization | Always compute mean/std on train split only; store and reuse for val/test |
| Comparing against reproduced baseline, not paper numbers | Report both: your reproduction AND the paper's numbers |
| xCPE MLP overfitting (it's tiny but the signal is weak) | Keep MLP at 2 layers max; apply dropout=0.1 inside MLP |
| OOM on 8GB VRAM | Use FP16 always; reduce batch size before anything else; use gradient checkpointing if needed: `model.gradient_checkpointing_enable()` |
| Non-reproducible results | Set `torch.manual_seed`, `numpy.random.seed`, and `random.seed` at the top of every training script; use `torch.backends.cudnn.deterministic = True` |
| Slow data loading bottleneck | Use `DataLoader(num_workers=4, pin_memory=True)` |
| xCPE neighborhood padding edge case | For i=0: pad left with zeros. For i=L-1: pad right with zeros. Test explicitly. |

---

## Quick Reference — Key Paper Numbers to Reproduce

| Dataset | Horizon | TimeDART MSE (paper) | TimeDART MAE (paper) |
|---------|---------|---------------------|---------------------|
| ETTh1   | 96      | 0.370               | 0.400               |
| ETTh1   | 192     | 0.413               | 0.427               |
| ETTh1   | 336     | 0.422               | 0.440               |
| ETTh1   | 720     | 0.447               | 0.468               |
| Weather | 96      | 0.149               | 0.198               |
| Weather | 192     | 0.194               | 0.241               |
| Weather | 336     | 0.245               | 0.282               |
| Weather | 720     | 0.315               | 0.334               |

> If your baseline reproduction is >10% above these numbers, stop and debug before
> proceeding to Phase 3. A broken baseline makes the xCPE comparison meaningless.
