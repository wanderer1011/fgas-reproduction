import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fgas.data import load_dataset, minmax, preprocess  # noqa: E402
from fgas.model import fgas, fgas_sweep, fgd_matrix, granule_cardinalities, stationary  # noqa: E402

try:
    import cupy  # noqa: F401
    HAS_GPU = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    HAS_GPU = False
gpu = pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU / cupy")

SIGMAS = [0.0, 0.3, 0.6, 0.85]


# ---------------------------------------------------------------- reference (literal equations)
def naive_fgas(X, sigma, fgd="squared", d=0.1, tol=1e-12, max_iter=100000):
    n, m = X.shape
    T = np.zeros((n, m))
    for a in range(m):
        for o in range(n):
            card = 0.0
            for p in range(n):
                r = 1 - abs(X[o, a] - X[p, a])          # R_a(o,p), Section 4.1
                card += 0.0 if r < sigma else r           # Eq. 3
            T[o, a] = card / n                            # |[o]_a| / |O|
    D = np.zeros((n, n))
    for o in range(n):
        for p in range(n):
            diff = T[o] - T[p]
            D[o, p] = (diff ** 2).sum() if fgd == "squared" else np.abs(diff).sum()   # Eq. 7 / code
    P = D / D.sum(axis=1, keepdims=True)                  # Eq. 8
    pi = np.full(n, 1 / n)
    for _ in range(max_iter):                             # Eq. 9
        new = d + (1 - d) * pi @ P
        done = np.abs(new - pi).sum() <= tol
        pi = new
        if done:
            break
    return (pi - pi.min()) / (pi.max() - pi.min())        # Eq. 10


@pytest.fixture(scope="module")
def rand_X():
    return np.random.default_rng(0).random((40, 5))


# ---------------------------------------------------------------- parity with official code
def test_parity_official_readme_example():
    """Official README: Example.mat, normalize 'range', sigma=0.6, code FGD (abs), tol 1e-4."""
    expected = np.array([0.1390, 0.0539, 0.3785, 0.0011, 0.3655, 0.0, 0.4530, 1.0000, 0.1325, 0.1692])
    X = minmax(sio.loadmat(ROOT / "data" / "raw" / "Example.mat")["Example"].astype(float))
    got = fgas(X, 0.6, fgd="abs", tol=1e-4)
    assert np.max(np.abs(got - expected)) <= 1e-4, got.round(4)


# ---------------------------------------------------------------- equations
@pytest.mark.parametrize("fgd", ["squared", "abs"])
@pytest.mark.parametrize("sigma", SIGMAS)
def test_efficient_equals_naive(rand_X, fgd, sigma):
    got = fgas(rand_X, sigma, fgd=fgd, tol=1e-12, max_iter=100000, chunk_rows=7)
    np.testing.assert_allclose(got, naive_fgas(rand_X, sigma, fgd), atol=1e-10)


def test_sweep_equals_single_sigma(rand_X):
    sweep = fgas_sweep(rand_X, SIGMAS, tol=1e-10)
    for r in sweep:
        np.testing.assert_allclose(r.scores, fgas(rand_X, r.sigma, tol=1e-10), atol=1e-12)


def test_damping_scalar_equals_uniform_restart(rand_X):
    """A#3: pi = d + (1-d) pi P (code) vs pi = d*u + (1-d) pi P with u = 1/n give the same AS."""
    T = granule_cardinalities(rand_X, [0.5])[0]
    D = fgd_matrix(T)
    pi_scalar = stationary(D, 0.1, 1e-14, 100000)[0]
    P = D / D.sum(1, keepdims=True)
    n = len(D)
    pi = np.full(n, 1 / n)
    for _ in range(100000):
        new = 0.1 / n + 0.9 * pi @ P
        if np.abs(new - pi).sum() <= 1e-16:
            break
        pi = new
    mm = lambda v: (v - v.min()) / (v.max() - v.min())  # noqa: E731
    np.testing.assert_allclose(mm(pi_scalar), mm(new), atol=1e-6)
    assert abs(new.sum() - 1) < 1e-9  # the uniform-restart version is a probability vector


def test_transition_rows_and_score_range(rand_X):
    T = granule_cardinalities(rand_X, [0.4])[0]
    D = fgd_matrix(T)
    assert np.allclose(D, D.T) and np.allclose(np.diag(D), 0)
    P = D / D.sum(1, keepdims=True)
    np.testing.assert_allclose(P.sum(1), 1.0)
    s = fgas(rand_X, 0.4)
    assert s.min() == 0.0 and s.max() == 1.0


def test_degenerate_all_identical_rows():
    X = np.ones((6, 3)) * 0.5
    r = fgas_sweep(X, [0.5])[0]
    assert r.scores is None and "zero-distance" in r.degenerate


def test_preprocess_minmax_constant_column():
    X = np.array([[1.0, 5.0], [3.0, 5.0], [2.0, 5.0]])
    Z = preprocess(X)
    np.testing.assert_allclose(Z[:, 0], [0, 1, 0.5])
    np.testing.assert_allclose(Z[:, 1], 0)


# ---------------------------------------------------------------- GPU vs CPU
@gpu
@pytest.mark.parametrize("fgd", ["squared", "abs"])
def test_gpu_float64_equals_cpu(rand_X, fgd):
    cpu = fgas_sweep(rand_X, SIGMAS, fgd=fgd, device="cpu")
    gpu_ = fgas_sweep(rand_X, SIGMAS, fgd=fgd, device="cuda", dtype="float64")
    for a, b in zip(cpu, gpu_):
        np.testing.assert_allclose(b.scores, a.scores, atol=1e-8)


@gpu
@pytest.mark.parametrize("abbr", ["Iris", "Wine"])
def test_gpu_on_real_data(abbr):
    X = preprocess(load_dataset(abbr).X_raw)
    cpu = fgas_sweep(X, SIGMAS, device="cpu")
    g64 = fgas_sweep(X, SIGMAS, device="cuda", dtype="float64")
    g32 = fgas_sweep(X, SIGMAS, device="cuda", dtype="float32")
    for a, b, c in zip(cpu, g64, g32):
        np.testing.assert_allclose(b.scores, a.scores, atol=1e-8)
        np.testing.assert_allclose(c.scores, a.scores, atol=1e-4)
