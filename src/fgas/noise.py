"""Attribute noise (Section 4.3, following Zhu & Wu [43]); protocol details are NOTES A#7.

mode="all_attributes": for every attribute independently, ceil(x*n) distinct samples (outliers included)
  get that attribute replaced by a value drawn uniformly between the attribute's min and max (raw data).
mode="one_attribute": one attribute is chosen at random, and ceil(x*n) distinct samples get that
  attribute replaced the same way ("the sample ceil(xn) with a specific attribute value is replaced").
"""
from __future__ import annotations

import math

import numpy as np


def add_attribute_noise(X: np.ndarray, level: float, rng: np.random.Generator,
                        mode: str = "all_attributes") -> np.ndarray:
    X = np.array(X, dtype=np.float64, copy=True)
    n, m = X.shape
    c = math.ceil(round(level * n, 9))
    if c == 0:
        return X
    lo, hi = X.min(axis=0), X.max(axis=0)
    if mode == "all_attributes":
        cols = range(m)
    elif mode == "one_attribute":
        cols = [int(rng.integers(m))]
    else:
        raise ValueError(mode)
    for j in cols:
        idx = rng.choice(n, size=c, replace=False)
        X[idx, j] = rng.uniform(lo[j], hi[j], size=c)
    return X


def noise_rng(seed: int, level_idx: int, dataset_id: int) -> np.random.Generator:
    """Independent, reproducible stream per (seed, noise level, dataset)."""
    return np.random.default_rng([seed, level_idx, dataset_id])
