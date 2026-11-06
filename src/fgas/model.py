"""FGAS: Fuzzy Granular Anomaly Score (Liu et al., Inf. Sci. 646 (2023) 119400).

Equation references are to the paper. One code path serves numpy (CPU) and cupy (GPU)
through the array module `xp`.

Pipeline for one sigma:
  T[o, a]   = |[o]_a| / |O|                      (Eq. 2-3, granule relative cardinality)
  D[o, p]   = sum_a (T[o,a] - T[p,a])^2          (Eq. 7, fgd="squared")   |  sum_a |.| (fgd="abs", official code)
  P         = E^-1 D,  E = diag(row sums of D)   (Eq. 8)
  pi(t+1)   = d + (1-d) pi(t) P,  pi(0) = 1/n    (Eq. 9, until ||pi(t+1)-pi(t)||_1 <= tol)
  AS        = (pi - min pi) / (max pi - min pi)  (Eq. 10)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def get_xp(device: str):
    if device == "cpu":
        return np
    if device == "cuda":
        import cupy as cp
        return cp
    raise ValueError(f"unknown device {device!r}")


def to_numpy(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


@dataclass
class FGASResult:
    sigma: float
    scores: np.ndarray | None        # AS in [0,1]; None if degenerate
    pi: np.ndarray | None             # stationary vector before Eq. 10
    n_iter: int
    residual: float                   # last ||pi(t+1)-pi(t)||_1
    converged: bool
    degenerate: str | None = None     # reason when scores is None
    extra: dict = field(default_factory=dict)


def granule_cardinalities(X, sigmas, xp=np, chunk_rows: int = 1024):
    """Relative granule cardinalities |[o]_a|/|O| for every sigma.

    X: (n, m) array in [0,1] on the xp device. Returns (S, n, m).
    Semantics match the official code exactly: sim = 1 - |a(o)-a(p)|; sim[sim < sigma] = 0.
    """
    n, m = X.shape
    S = len(sigmas)
    out = xp.zeros((S, n, m), dtype=X.dtype)
    sig = [X.dtype.type(s) for s in sigmas]
    for j in range(m):
        col = X[:, j]
        for i0 in range(0, n, chunk_rows):
            i1 = min(i0 + chunk_rows, n)
            sim = 1 - xp.abs(col[i0:i1, None] - col[None, :])      # (c, n)
            for k, s in enumerate(sig):
                out[k, i0:i1, j] = xp.where(sim < s, 0, sim).sum(axis=1)
    return out / n


def fgd_matrix(T, fgd: str = "squared", xp=np, chunk_rows: int = 1024):
    """Pairwise fuzzy granular distance (Eq. 7) from T (n, m), accumulated attribute by attribute."""
    n, m = T.shape
    D = xp.zeros((n, n), dtype=T.dtype)
    for j in range(m):
        t = T[:, j]
        for i0 in range(0, n, chunk_rows):
            i1 = min(i0 + chunk_rows, n)
            diff = t[i0:i1, None] - t[None, :]
            if fgd == "squared":
                D[i0:i1] += diff * diff
            elif fgd == "abs":
                D[i0:i1] += xp.abs(diff)
            else:
                raise ValueError(f"unknown fgd {fgd!r}")
    return D


def stationary(D, d: float, tol: float, max_iter: int, xp=np):
    """Eq. 8 + Eq. 9 power iteration. Returns (pi, n_iter, residual, converged, degenerate_reason)."""
    n = D.shape[0]
    rowsum = D.sum(axis=1)
    if bool((rowsum <= 0).any()):
        return None, 0, float("nan"), False, f"{int((rowsum <= 0).sum())} zero-distance row(s)"
    P = D / rowsum[:, None]
    dd = D.dtype.type(d)
    pi = xp.full(n, 1.0 / n, dtype=D.dtype)
    res, it = float("inf"), 0
    while it < max_iter:
        new = dd + (1 - dd) * (pi @ P)
        res = float(xp.abs(new - pi).sum())
        pi = new
        it += 1
        if res <= tol:
            return pi, it, res, True, None
    return pi, it, res, False, None


def minmax_scores(pi, xp=np):
    lo, hi = pi.min(), pi.max()
    if float(hi - lo) == 0.0:
        return None
    return (pi - lo) / (hi - lo)


def fgas_sweep(X, sigmas, *, fgd="squared", d=0.1, tol=1e-3, max_iter=10000,
               device="cpu", dtype="float64", chunk_rows=1024) -> list[FGASResult]:
    """Run FGAS for every sigma. Granule cardinalities are computed for all sigmas in one pass."""
    xp = get_xp(device)
    Xd = xp.asarray(X, dtype=dtype)
    T_all = granule_cardinalities(Xd, list(sigmas), xp, chunk_rows)
    results = []
    for k, s in enumerate(sigmas):
        D = fgd_matrix(T_all[k], fgd, xp, chunk_rows)
        pi, it, res, conv, degen = stationary(D, d, tol, max_iter, xp)
        scores = None
        if pi is not None:
            sc = minmax_scores(pi, xp)
            if sc is None:
                degen = "constant stationary vector"
            else:
                scores = to_numpy(sc).astype(np.float64)
        results.append(FGASResult(float(s), scores, None if pi is None else to_numpy(pi),
                                  it, res, conv, degen))
        del D
    if device == "cuda":
        xp.get_default_memory_pool().free_all_blocks()
    return results


def fgas(X, sigma, **kw) -> np.ndarray:
    """Convenience: anomaly scores for a single sigma (raises if degenerate)."""
    r = fgas_sweep(X, [sigma], **kw)[0]
    if r.scores is None:
        raise ValueError(f"degenerate FGAS at sigma={sigma}: {r.degenerate}")
    return r.scores
