"""
src/models — Model factory and public API.

Use build_model() to instantiate the correct model class based on
cfg.model.variant.  Training scripts should call this instead of
importing individual classes directly.
"""

from src.models.timedart import TimeDART
from src.models.xcpe_timedart import xCPETimeDART, RoPETimeDART
from src.utils.config import ModelConfig, DataConfig

import torch.nn as nn

_VARIANT_MAP = {
    "baseline":   lambda mc, dc: TimeDART(mc, dc),
    "xcpe_all":   lambda mc, dc: xCPETimeDART(mc, dc, xcpe_layers="all"),
    "xcpe_early": lambda mc, dc: xCPETimeDART(mc, dc, xcpe_layers="early"),
    "xcpe_late":  lambda mc, dc: xCPETimeDART(mc, dc, xcpe_layers="late"),
    "rope":       lambda mc, dc: RoPETimeDART(mc, dc),
}


def build_model(model_config: ModelConfig, data_config: DataConfig) -> nn.Module:
    """Instantiate the correct model class from cfg.model.variant.

    Args:
        model_config: ModelConfig with a .variant field.
        data_config:  DataConfig (used to size the forecast head).

    Returns:
        An nn.Module subclass of TimeDART.

    Raises:
        ValueError: If model_config.variant is not recognised.
    """
    variant = model_config.variant
    if variant not in _VARIANT_MAP:
        raise ValueError(
            f"Unknown model variant {variant!r}. "
            f"Valid options: {sorted(_VARIANT_MAP)}"
        )
    return _VARIANT_MAP[variant](model_config, data_config)
