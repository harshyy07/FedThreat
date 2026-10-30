"""Lightweight MLP threat detection model designed for DP-SGD compatibility."""

import torch
import torch.nn as nn


class ThreatMLP(nn.Module):
    """3-layer MLP classifier using LayerNorm for DP-SGD compatibility (no BatchNorm)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers: list[nn.Module] = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        return logits
