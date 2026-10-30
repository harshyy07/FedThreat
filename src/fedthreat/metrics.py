"""Evaluation metrics for threat detection models under class imbalance."""

from typing import Dict
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    precision_recall_curve,
    auc,
    roc_curve,
)


def compute_threat_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    y_pred: np.ndarray,
    target_fpr: float = 0.001,
) -> Dict[str, float]:
    """Computes security-focused metrics: Macro-F1, PR-AUC, and Recall at fixed FPR (e.g. 0.1%).
    
    Args:
        y_true: Ground truth binary labels (0 = benign, 1 = attack).
        y_scores: Predicted probabilities for the positive attack class.
        y_pred: Predicted class labels.
        target_fpr: False positive rate target (default 0.1% for realistic SOC alert thresholds).
    """
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    
    # Precision-Recall AUC
    p_curve, r_curve, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = float(auc(r_curve, p_curve))
    
    # Recall at fixed FPR
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    # Find maximum TPR where FPR <= target_fpr
    valid_indices = np.where(fpr <= target_fpr)[0]
    if len(valid_indices) > 0:
        recall_at_fpr = float(tpr[valid_indices[-1]])
    else:
        recall_at_fpr = 0.0

    return {
        "macro_f1": float(f1),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "pr_auc": pr_auc,
        f"recall_at_{target_fpr*100:.2f}%_fpr": recall_at_fpr,
    }
