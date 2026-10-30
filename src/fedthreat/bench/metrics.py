"""Detection and attack-success metrics (binary: 1 = attack, 0 = benign)."""

from __future__ import annotations

import numpy as np


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """PR-AUC as average precision. Accuracy is meaningless on ~15% positives; use this."""
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    ys = y[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / ys.sum())


def threshold_at_fpr(y: np.ndarray, score: np.ndarray, fpr: float) -> float:
    neg = np.sort(score[y == 0])
    if len(neg) == 0:
        return float("inf")
    k = int(np.ceil((1.0 - fpr) * len(neg))) - 1
    return float(neg[min(max(k, 0), len(neg) - 1)])


def recall_at_fpr(y: np.ndarray, score: np.ndarray, fpr: float, pos_mask: np.ndarray | None = None) -> float:
    """Recall at a fixed false-positive rate chosen on ALL benign data.
    ``pos_mask`` restricts which positives count (e.g. only the held-out attack family)."""
    thr = threshold_at_fpr(y, score, fpr)
    pos = (y == 1) if pos_mask is None else pos_mask
    return float((score[pos] > thr).mean()) if pos.any() else float("nan")
