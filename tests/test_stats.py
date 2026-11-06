import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fgas.stats import friedman_nemenyi  # noqa: E402


def test_reproduces_paper_section_4_4():
    t = pd.read_csv(ROOT / "paper" / "table2_paper.csv", index_col=0)
    r = friedman_nemenyi(t, alpha=0.10)
    assert r.df == (13, 182)
    assert round(r.F_crit, 4) == 1.5610          # paper: 1.5610
    assert round(r.CD, 4) == 4.7659              # paper: 4.7659
    assert abs(r.tau_F - 15.1347) / 15.1347 < 0.005  # paper: 15.1347 (we get 15.07 from 3-dp table)
    assert r.avg_rank.idxmin() == "FGAS" and r.avg_rank.idxmax() == "FastABOD"
