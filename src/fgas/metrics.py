"""AUC (Eq. 13) and ROC points (Eqs. 11-12)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auc(y: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC with ties counted as 0.5 (identical to Eq. 13)."""
    return float(roc_auc_score(y, scores))


def auc_eq13(y: np.ndarray, scores: np.ndarray) -> float:
    """Direct implementation of Eq. 13: mean over (outlier, inlier) pairs of 1 / 0.5 / 0."""
    s_out = scores[y == 1][:, None]
    s_in = scores[y == 0][None, :]
    return float(((s_out > s_in) + 0.5 * (s_out == s_in)).mean())


def roc_points(y: np.ndarray, scores: np.ndarray):
    """(FPR, TPR) after each rank position, samples sorted by descending score (Eqs. 11-12).

    Tied scores are kept as a single step (they cannot be ordered), which matches the
    trapezoid used by roc_auc_score.
    """
    order = np.argsort(-scores, kind="mergesort")
    s, yy = scores[order], y[order]
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(yy)[last]
    fp = np.cumsum(1 - yy)[last]
    tpr = np.r_[0.0, tp / max(yy.sum(), 1)]
    fpr = np.r_[0.0, fp / max((1 - yy).sum(), 1)]
    return fpr, tpr
