"""Table 2 (FGAS column): ours vs paper, both FGD variants.

Usage: python scripts/make_table2.py [--run runs/<id>_table2]
Writes results/table2_fgas.csv, results/table2_fgas.md, results/table2_full_ours.csv
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fgas.data import dataset_specs  # noqa: E402

TOL = yaml.safe_load("""
match: 0.005
close: 0.02
""")


def status(delta: float) -> str:
    a = abs(delta)
    return "match" if a <= TOL["match"] else "close" if a <= TOL["close"] else "GAP"


def best_per_dataset(df: pd.DataFrame, fgd: str, sigmas=None) -> pd.DataFrame:
    d = df[df.fgd == fgd]
    if sigmas is not None:
        d = d[d.sigma.round(2).isin(np.round(sigmas, 2))]
    idx = d.groupby("dataset")["auc"].idxmax()
    return d.loc[idx].set_index("dataset")[["auc", "sigma"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=sorted(glob.glob(str(ROOT / "runs" / "*_table2")))[-1])
    args = ap.parse_args()
    run = Path(args.run)
    df = pd.read_csv(run / "auc.csv")
    order = list(dataset_specs())
    paper = pd.read_csv(ROOT / "paper" / "table2_paper.csv", index_col=0)

    a = best_per_dataset(df, "abs").reindex(order)
    s = best_per_dataset(df, "squared").reindex(order)
    t = pd.DataFrame({
        "paper": paper.loc[order, "FGAS"],
        "abs_auc": a["auc"], "abs_sigma": a["sigma"],
        "sq_auc": s["auc"], "sq_sigma": s["sigma"],
    })
    t["abs_delta"] = t.abs_auc - t.paper
    t["abs_pct"] = 100 * t.abs_delta / t.paper
    t["abs_status"] = t.abs_delta.map(status)
    t["sq_delta"] = t.sq_auc - t.paper
    t["sq_pct"] = 100 * t.sq_delta / t.paper
    t["sq_status"] = t.sq_delta.map(status)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    t.to_csv(out / "table2_fgas.csv", float_format="%.6f")

    avg = {c: t[c].mean() for c in ["paper", "abs_auc", "sq_auc"]}
    # Paper-format Table 2 with our (abs, 3-dp) FGAS column swapped in; baselines as reported.
    full = paper.loc[order].copy()
    full["FGAS"] = t.abs_auc.round(3)
    wins = full.eq(full.max(axis=1), axis=0).sum()
    full.to_csv(out / "table2_full_ours.csv", float_format="%.3f")

    lines = [
        "# Table 2 — FGAS column: reproduction vs paper",
        "",
        f"Run: `{run.name}` (σ ∈ {{0, 0.05, …, 1.0}}, best AUC over the grid as in the paper; d = 0.1, "
        "tol = 1e-3, cuda float64).",
        f"Status: |Δ| ≤ {TOL['match']} match, ≤ {TOL['close']} close, else GAP. "
        "Paper values are rounded to 3 dp, so ±0.0005 is rounding.",
        "",
        "| Dataset | Paper | **Ours abs (headline)** | best σ | Δ | % dev | status | Ours squared (Eq. 7) | best σ | Δ | status |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for ds, r in t.iterrows():
        lines.append(
            f"| {ds} | {r.paper:.3f} | **{r.abs_auc:.4f}** | {r.abs_sigma:.2f} | {r.abs_delta:+.4f} | "
            f"{r.abs_pct:+.2f}% | {r.abs_status} | {r.sq_auc:.4f} | {r.sq_sigma:.2f} | {r.sq_delta:+.4f} | "
            f"{r.sq_status} |")
    lines.append(
        f"| **Average** | {avg['paper']:.3f} (printed 0.934) | **{avg['abs_auc']:.4f}** | | "
        f"{avg['abs_auc'] - avg['paper']:+.4f} | {100 * (avg['abs_auc'] / avg['paper'] - 1):+.2f}% | "
        f"{status(avg['abs_auc'] - avg['paper'])} | {avg['sq_auc']:.4f} | | "
        f"{avg['sq_auc'] - avg['paper']:+.4f} | {status(avg['sq_auc'] - avg['paper'])} |")
    lines += [
        "",
        f"abs: {sum(t.abs_status == 'match')}/15 match, {sum(t.abs_status == 'close')} close, "
        f"{sum(t.abs_status == 'GAP')} gap. squared: {sum(t.sq_status == 'match')}/15 match, "
        f"{sum(t.sq_status == 'close')} close, {sum(t.sq_status == 'GAP')} gap.",
        "",
        "## '1st order' row with our FGAS column (abs, rounded to 3 dp) and the paper's baseline numbers",
        "",
        "| " + " | ".join(wins.index) + " |",
        "|" + "---|" * len(wins),
        "| " + " | ".join(str(int(v)) for v in wins.values) + " |",
        "",
        "Paper: FGAS first on 12 datasets.",
    ]
    (out / "table2_fgas.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
