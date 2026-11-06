"""Baseline detectors re-run with standard libraries (NOTES A#14).

Each entry maps a method name to score(X, k, seed) -> outlier scores (higher = more anomalous).
Methods the paper tunes over k use the paper's grid; parameter-free methods ignore k.
"""
from __future__ import annotations

import warnings

import numpy as np


def _knn(X, k, seed):
    # Ramaswamy et al. [27]: distance to the k-th nearest neighbour.
    from pyod.models.knn import KNN
    return KNN(n_neighbors=k, method="largest").fit(X).decision_scores_


def _cof(X, k, seed):
    from pyod.models.cof import COF
    return COF(n_neighbors=k).fit(X).decision_scores_


def _fastabod(X, k, seed):
    from pyod.models.abod import ABOD
    return ABOD(n_neighbors=k, method="fast").fit(X).decision_scores_


def _loop(X, k, seed):
    from PyNomaly import loop
    m = loop.LocalOutlierProbability(X, n_neighbors=k, use_numba=False, progress_bar=False).fit()
    return np.asarray(m.local_outlier_probabilities, dtype=float)


def _ecod(X, k, seed):
    from pyod.models.ecod import ECOD
    return ECOD().fit(X).decision_scores_


def _iforest(X, k, seed):
    from pyod.models.iforest import IForest
    return IForest(n_estimators=100, random_state=seed).fit(X).decision_scores_


METHODS = {
    "kNN": _knn,
    "COF": _cof,
    "FastABOD": _fastabod,
    "LoOP": _loop,
    "ECOD": _ecod,
    "IForest": _iforest,
}


def score(method: str, X: np.ndarray, k: int | None = None, seed: int = 0):
    """Return (scores, note). Non-finite scores: +inf -> above the finite max; NaN -> score is None."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            s = np.asarray(METHODS[method](X, k, seed), dtype=float)
        except Exception as e:  # e.g. k too small for the method
            return None, f"error: {type(e).__name__}: {e}"[:200]
    note = None
    if np.isnan(s).any():
        return None, f"{int(np.isnan(s).sum())} NaN scores"
    if np.isinf(s).any():
        fin = s[np.isfinite(s)]
        top = fin.max() if fin.size else 0.0
        note = f"{int(np.isinf(s).sum())} inf scores mapped to max+1"
        s = np.where(np.isposinf(s), top + 1, np.where(np.isneginf(s), fin.min() - 1, s))
    return s, note
