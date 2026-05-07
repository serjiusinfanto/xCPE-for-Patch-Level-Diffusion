# xCPE-TimeDART: Content-Conditioned Positional Encoding for Time Series Forecasting

This project adapts **xCPE** (Content-Conditioned Positional Encoding) from [Point Transformer V3](https://github.com/Pointcept/PointTransformerV3) (CVPR 2024) into [TimeDART](https://github.com/Melmaphother/TimeDART) (ICLR 2025), a diffusion-based patch-level Transformer for time series forecasting.

The core idea: instead of assigning each patch token a positional embedding based only on its index, xCPE computes the embedding from the local neighbourhood's **mean, variance, and slope** — making the positional signal content-aware. This is especially useful for non-stationary time series where the same position can carry very different dynamics across different windows.

---

## Repository Structure

```
├── src/
│   ├── models/
│   │   ├── positional_encoding.py   ← FixedPositionalEmbedding, xCPE, RoPEEmbedding
│   │   ├── timedart.py              ← baseline TimeDART
│   │   └── xcpe_timedart.py         ← xCPETimeDART and RoPETimeDART variants
│   ├── data/
│   │   ├── dataset.py               ← TimeSeriesDataset with patch collation
│   │   └── preprocess.py            ← z-score normalization, train/val/test splitting
│   ├── diffusion/
│   │   ├── noise_scheduler.py       ← linear beta schedule DDPM
│   │   └── denoising.py             ← forward/reverse diffusion
│   ├── training/
│   │   ├── pretrain.py              ← self-supervised diffusion pre-training
│   │   └── finetune.py              ← supervised forecasting fine-tuning
│   ├── evaluation/
│   │   └── metrics.py               ← MSE and MAE
│   └── utils/
│       └── config.py                ← dataclass configs and YAML loader
├── configs/                         ← YAML configs for all variants and datasets
├── scripts/                         ← download data, run pretrain/finetune
├── notebooks/                       ← data exploration and results analysis
├── demo/
│   └── app.py                       ← Streamlit live demo
├── results/
│   └── tables/                      ← MSE/MAE CSVs for all experiments
└── reference/                       ← TimeDART and PTv3 reference implementations
```

---

## Model Variants

| Variant | Positional Encoding | Description |
|---|---|---|
| `baseline` | Fixed (learned lookup) | Standard TimeDART |
| `xcpe_all` | xCPE at all layers | Full proposal |
| `xcpe_early` | xCPE at layers 1–2 only | Ablation: early layers |
| `xcpe_late` | xCPE at layers 2–3 only | Ablation: late layers |
| `rope` | Rotary PE inside attention | Ablation: alternative flexible PE |

---

## Results

### ETTh1 (MSE, mean ± std over 3 seeds)

| Variant | H=96 | H=192 | H=336 | H=720 |
|---|---|---|---|---|
| Baseline | 0.4185 ± 0.0025 | 0.4656 ± 0.0033 | 0.5207 ± 0.0015 | 0.6385 ± 0.0100 |
| xCPE (all) | 0.4218 ± 0.0058 | **0.4573 ± 0.0029** | **0.5077 ± 0.0038** | **0.6130 ± 0.0031** |
| xCPE (early) | 0.4330 ± 0.0104 | 0.4873 ± 0.0020 | 0.5379 ± 0.0098 | 0.6580 ± 0.0021 |
| xCPE (late) | 0.4340 ± 0.0061 | 0.4824 ± 0.0046 | 0.5204 ± 0.0100 | 0.6460 ± 0.0055 |
| RoPE | 0.4249 ± 0.0071 | 0.4766 ± 0.0023 | 0.5286 ± 0.0028 | 0.6498 ± 0.0031 |

### Weather (MSE, mean ± std over 3 seeds)

| Variant | H=96 | H=192 | H=336 | H=720 |
|---|---|---|---|---|
| Baseline | 0.1581 ± 0.0018 | 0.2043 ± 0.0022 | 0.2564 ± 0.0019 | 0.3312 ± 0.0031 |
| xCPE (all) | **0.1546 ± 0.0024** | **0.1964 ± 0.0027** | **0.2444 ± 0.0031** | **0.3130 ± 0.0029** |
| xCPE (early) | 0.1598 ± 0.0042 | 0.2071 ± 0.0038 | 0.2521 ± 0.0044 | 0.3268 ± 0.0033 |
| xCPE (late) | 0.1558 ± 0.0029 | 0.2009 ± 0.0034 | 0.2511 ± 0.0028 | 0.3289 ± 0.0041 |
| RoPE | 0.1612 ± 0.0031 | 0.2089 ± 0.0025 | 0.2618 ± 0.0033 | 0.3378 ± 0.0028 |

xCPE (all) wins at H=192, 336, 720 on ETTh1 and at all horizons on Weather. The improvement grows with forecast horizon.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install accelerate transformers einops pyyaml wandb pandas numpy scikit-learn
pip install matplotlib seaborn jupyter streamlit
```

---

## Data

Download all datasets from HuggingFace (CC BY 4.0):

```bash
pip install huggingface_hub
python scripts/download_data.py
```

This fetches ETTh1, ETTh2, ETTm1, ETTm2, and Weather into `data/raw/`.

---

## Training

```bash
# Pre-train
accelerate launch scripts/run_pretrain.sh --config configs/xcpe_etth1.yaml

# Fine-tune
accelerate launch scripts/run_finetune.sh --config configs/xcpe_etth1.yaml
```

Each run takes approximately 25–30 minutes on an RTX 4070. The full experiment grid (5 variants × 4 horizons × 3 seeds) takes roughly 30 hours.

---

## Live Demo

```bash
streamlit run demo/app.py
```

Loads real trained checkpoints and runs live inference on the ETTh1 test set. Three tabs: data explorer, live forecasting (baseline vs xCPE side by side), and xCPE internals (positional embedding heatmaps and conditioning signals).

---

## Architecture

- **d_model**: 64 — **n_heads**: 4 — **n_layers**: 3 — **d_ff**: 256
- **Patch length**: 16 timesteps — **Context length**: 336 timesteps
- **Pre-training**: linear beta schedule DDPM, T=1000, 50 epochs
- **Fine-tuning**: AdamW, lr=5e-5, up to 30 epochs with early stopping
- **Hardware**: NVIDIA RTX 4070 8GB, FP16 mixed precision

---

## References

- Wang et al., *TimeDART*, ICLR 2025. [GitHub](https://github.com/Melmaphother/TimeDART)
- Wu et al., *Point Transformer V3*, CVPR 2024 (Oral). [GitHub](https://github.com/Pointcept/PointTransformerV3)
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, arXiv:2104.09864, 2021.

