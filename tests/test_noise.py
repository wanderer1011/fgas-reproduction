import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fgas.noise import add_attribute_noise, noise_rng  # noqa: E402


def test_counts_and_range():
    X = np.random.default_rng(0).random((101, 4)) * 5 + 2
    for lvl in [0.0, 0.05, 0.3, 0.5]:
        Z = add_attribute_noise(X, lvl, noise_rng(0, 1, 1))
        changed = (Z != X).sum(axis=0)
        assert (changed <= math.ceil(lvl * 101)).all()
        assert (changed >= math.ceil(lvl * 101) - 1).all()   # a redraw can equal the old value only with prob ~0
        assert (Z.min(0) >= X.min(0)).all() and (Z.max(0) <= X.max(0)).all()


def test_reproducible_and_input_untouched():
    X = np.random.default_rng(1).random((50, 3))
    X0 = X.copy()
    a = add_attribute_noise(X, 0.2, noise_rng(3, 4, 5))
    b = add_attribute_noise(X, 0.2, noise_rng(3, 4, 5))
    assert np.array_equal(a, b) and np.array_equal(X, X0)
    c = add_attribute_noise(X, 0.2, noise_rng(4, 4, 5))
    assert not np.array_equal(a, c)


def test_one_attribute_mode():
    X = np.random.default_rng(2).random((80, 5))
    Z = add_attribute_noise(X, 0.25, noise_rng(0, 2, 3), mode="one_attribute")
    changed_cols = np.nonzero((Z != X).any(axis=0))[0]
    assert len(changed_cols) == 1 and (Z != X).sum() == math.ceil(0.25 * 80)
