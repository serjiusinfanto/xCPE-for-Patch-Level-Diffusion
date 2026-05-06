"""
Configuration dataclasses and YAML loader for the xCPE-TimeDART project.

All hyperparameters live in YAML files under configs/. This module loads
them into typed dataclasses so the rest of the codebase never touches raw dicts.
"""

from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class DataConfig:
    dataset: str           # ETTh1 | ETTh2 | ETTm1 | ETTm2 | Weather
    path: str              # relative path to the CSV from project root
    context_length: int    # input window in timesteps (e.g. 336)
    horizon: int           # forecast horizon in timesteps (e.g. 96)
    patch_length: int      # timesteps per patch token (e.g. 16)
    train_split: float     # fraction of data used for training (e.g. 0.6)
    val_split: float       # fraction used for validation (e.g. 0.2)
    finetune_stride: int   # patch stride during fine-tuning (overlapping, e.g. 8)
    # test split is inferred as 1 - train_split - val_split


@dataclass
class ModelConfig:
    variant: str        # baseline | xcpe_all | xcpe_early | xcpe_late | rope
    d_model: int        # token embedding dimension (e.g. 64)
    n_heads: int        # number of attention heads (e.g. 4)
    n_layers: int       # number of Transformer encoder layers (e.g. 3)
    d_ff: int           # feedforward dimension (e.g. 256)
    patch_length: int   # used by the model to size its patch embedding
    dropout: float      # dropout rate (e.g. 0.1)


@dataclass
class TrainingConfig:
    pretrain_epochs: int    # number of self-supervised pre-training epochs
    finetune_epochs: int    # number of supervised fine-tuning epochs
    pretrain_lr: float      # AdamW learning rate during pre-training
    finetune_lr: float      # AdamW learning rate during fine-tuning
    batch_size: int         # samples per batch
    seed: int               # random seed for reproducibility
    patience: int           # early stopping patience (val MSE)


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig


def load_config(path: str) -> Config:
    """Load a YAML config file and return a typed Config object.

    Args:
        path: Path to the YAML file (e.g. 'configs/baseline_etth1.yaml').

    Returns:
        Config with .data, .model, and .training sub-configs.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    data_cfg = DataConfig(**raw["data"])
    model_cfg = ModelConfig(**raw["model"])
    training_cfg = TrainingConfig(**raw["training"])

    return Config(data=data_cfg, model=model_cfg, training=training_cfg)
