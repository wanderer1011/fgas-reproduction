"""Friedman test + Nemenyi post-hoc (Demsar 2006), as used in Section 4.4 / Fig. 3."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from scipy.stats import rankdata

# q_alpha for alpha = 0.10 (Studentized range / sqrt(2)). k <= 10 from Demsar (2006) Table 5(b);
# k = 14 value 3.12 reproduces the paper's CD = 4.7659 exactly (NOTES A#16).
Q_010 = {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920,
         14: 3.12}


@dataclass
class FriedmanResult:
    avg_rank: pd.Series     # 1 = best (higher AUC)
    chi2_F: float
    tau_F: float            # Iman-Davenport F statistic
    df: tuple[int, int]
    F_crit: float
    CD: float
    alpha: float


def friedman_nemenyi(table: pd.DataFrame, alpha: float = 0.10) -> FriedmanResult:
    """table: rows = datasets, columns = algorithms, values = AUC (higher is better)."""
    A = table.to_numpy(dtype=float)
    N, k = A.shape
    R = np.vstack([rankdata(-row, method="average") for row in A])  # ties share the average rank
    Rm = R.mean(axis=0)
    chi2 = 12 * N / (k * (k + 1)) * (np.sum(Rm ** 2) - k * (k + 1) ** 2 / 4)
    tau = (N - 1) * chi2 / (N * (k - 1) - chi2)
    df = (k - 1, (k - 1) * (N - 1))
    if alpha != 0.10 or k not in Q_010:
        raise ValueError("q_alpha only tabulated for alpha=0.10 and k in Q_010")
    cd = Q_010[k] * np.sqrt(k * (k + 1) / (6 * N))
    return FriedmanResult(pd.Series(Rm, index=table.columns), float(chi2), float(tau), df,
                          float(f_dist.ppf(1 - alpha, *df)), float(cd), alpha)
