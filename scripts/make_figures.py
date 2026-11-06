"""Regenerate the paper's figures from run outputs.

Usage: python scripts/make_figures.py fig4 [--run runs/<id>_table2]
"""
import argparse
import glob
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fgas.plots import fig4_sigma  # noqa: E402


def latest(pattern: str) -> str:
    runs = sorted(glob.glob(str(ROOT / "runs" / pattern)))
    if not runs:
        raise SystemExit(f"no run matching {pattern}")
    return runs[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("figure", choices=["fig4"])
    ap.add_argument("--run")
    args = ap.parse_args()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    if args.figure == "fig4":
        run = Path(args.run or latest("*_table2"))
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        df = pd.read_csv(run / "auc.csv")
        stop = cfg["sigma_grid"].get("plot_stop", 0.95)
        for fgd in ["abs", "squared"]:
            p = out / f"fig4_sigma_{fgd}.png"
            fig4_sigma(df, fgd, p, plot_stop=stop)
            print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
