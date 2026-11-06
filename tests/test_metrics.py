import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fgas.metrics import auc, auc_eq13, roc_points  # noqa: E402


@pytest.mark.parametrize("seed", range(5))
def test_sklearn_auc_equals_eq13_with_ties(seed):
    rng = np.random.default_rng(seed)
    y = (rng.random(200) < 0.15).astype(int)
    s = rng.integers(0, 12, 200).astype(float)  # many ties
    assert abs(auc(y, s) - auc_eq13(y, s)) < 1e-12


def test_roc_points_area_matches_auc():
    rng = np.random.default_rng(1)
    y = (rng.random(300) < 0.2).astype(int)
    s = rng.random(300) + 0.5 * y
    fpr, tpr = roc_points(y, s)
    assert fpr[0] == 0 and tpr[-1] == 1 and fpr[-1] == 1
    assert abs(np.trapezoid(tpr, fpr) - auc(y, s)) < 1e-12
