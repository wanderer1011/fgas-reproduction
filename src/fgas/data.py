"""Dataset loading and preprocessing (NOTES.md A#8-A#11)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Dataset:
    abbr: str
    X_raw: np.ndarray  # (n, m) features as stored, NaNs possible
    y: np.ndarray      # (n,) 1 = outlier, 0 = inlier
    meta: dict


def dataset_specs(path: str | Path = ROOT / "configs" / "datasets.yaml") -> dict[str, dict]:
    cfg = yaml.safe_load(Path(path).read_text())
    return {d["abbr"]: d for d in cfg["datasets"]}


def load_dataset(abbr: str, data_dir: str | Path = ROOT / "data" / "raw") -> Dataset:
    """Load a repo .mat file. Layout (verified in M1): key 'trandata', last column = label."""
    spec = dataset_specs()[abbr]
    mat = sio.loadmat(Path(data_dir) / spec["file"])
    arr = np.asarray(mat["trandata"], dtype=np.float64)
    X, y = arr[:, :-1], arr[:, -1]
    labels = np.unique(y[~np.isnan(y)])
    if not set(labels.tolist()) <= {0.0, 1.0}:
        raise ValueError(f"{abbr}: label column is not binary 0/1: {labels[:10]}")
    return Dataset(abbr, X, y.astype(int), spec)


def impute_mode(X: np.ndarray) -> np.ndarray:
    """'Maximum frequency method': replace NaN by the column's most frequent value."""
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        miss = np.isnan(col)
        if miss.any():
            vals, counts = np.unique(col[~miss], return_counts=True)
            col[miss] = vals[np.argmax(counts)]
    return X


def minmax(X: np.ndarray) -> np.ndarray:
    """Per-attribute min-max scaling to [0,1]; constant attributes map to 0."""
    lo, hi = X.min(axis=0), X.max(axis=0)
    rng = hi - lo
    out = np.zeros_like(X, dtype=np.float64)
    nz = rng > 0
    out[:, nz] = (X[:, nz] - lo[nz]) / rng[nz]
    return out


def preprocess(X_raw: np.ndarray, impute: str = "mode", scaling: str = "minmax") -> np.ndarray:
    X = impute_mode(X_raw) if impute == "mode" else X_raw.copy()
    if scaling == "minmax":
        X = minmax(X)
    elif scaling != "none":
        raise ValueError(scaling)
    return X
