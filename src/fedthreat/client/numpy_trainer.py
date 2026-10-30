"""Dependency-light local trainer (softmax regression) for demos, CI and constrained edge hosts.

The production trainer is ``ThreatMLP`` + ``ClientTrainer`` (torch). Both return the same thing:
a parameter DELTA dict, so the agent/server stack is model-agnostic.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

Params = Dict[str, np.ndarray]


def _prep(X: np.ndarray) -> np.ndarray:
    return np.sign(X) * np.log1p(np.abs(X))  # tames heavy-tailed byte/packet counts


def init_linear_params(dim: int, num_classes: int = 2) -> Params:
    return {"linear.weight": np.zeros((num_classes, dim), np.float32), "linear.bias": np.zeros(num_classes, np.float32)}


def predict_proba(params: Params, X: np.ndarray) -> np.ndarray:
    z = _prep(X) @ params["linear.weight"].T + params["linear.bias"]
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_delta(params: Params, X: np.ndarray, y: np.ndarray, epochs: int = 5, lr: float = 0.1, batch: int = 128,
                seed: int = 0) -> Params:
    rng = np.random.default_rng(seed)
    W, b = params["linear.weight"].astype(np.float64).copy(), params["linear.bias"].astype(np.float64).copy()
    Xp = _prep(X).astype(np.float64)
    n, k = len(y), W.shape[0]
    onehot = np.eye(k)[y]
    for _ in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            z = Xp[idx] @ W.T + b
            z -= z.max(axis=1, keepdims=True)
            p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
            g = (p - onehot[idx]) / len(idx)
            W -= lr * g.T @ Xp[idx]
            b -= lr * g.sum(axis=0)
    return {"linear.weight": (W - params["linear.weight"]).astype(np.float32),
            "linear.bias": (b - params["linear.bias"]).astype(np.float32)}
