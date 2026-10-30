"""Torch-free parameter initialisation so the aggregation server never imports torch.

Key names and shapes match ``ThreatMLP.state_dict()`` (dropout > 0 layout:
Linear, LayerNorm, ReLU, Dropout per hidden block).
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def init_mlp_params(
    input_dim: int, hidden_dims: Sequence[int] = (128, 64, 32), num_classes: int = 2, seed: int | None = None
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    params: Dict[str, np.ndarray] = {}

    def linear(prefix: str, fan_in: int, fan_out: int) -> None:
        bound = 1.0 / np.sqrt(fan_in)
        params[f"{prefix}.weight"] = rng.uniform(-bound, bound, (fan_out, fan_in)).astype(np.float32)
        params[f"{prefix}.bias"] = rng.uniform(-bound, bound, (fan_out,)).astype(np.float32)

    d = input_dim
    for i, h in enumerate(hidden_dims):
        linear(f"backbone.{4 * i}", d, h)
        params[f"backbone.{4 * i + 1}.weight"] = np.ones(h, np.float32)
        params[f"backbone.{4 * i + 1}.bias"] = np.zeros(h, np.float32)
        d = h
    linear("head", d, num_classes)
    return params
