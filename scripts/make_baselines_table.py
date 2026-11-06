"""Compare re-run baselines with the paper and build the mixed Table 2 used for Fig. 3 (ours_rerun).

Usage: python scripts/make_baselines_table.py [--run runs/<id>_baselines]
Writes results/baselines_vs_paper.{csv,md} and results/table2_mixed_baselines.csv
"""
import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fgas.data import dataset_specs  # noqa: E402

MATCH, CLOSE = 0.01, 0.03  # PLAN §5 tolerances for re-run baselines


def status(d):
    return "match" if abs(d) <= MATCH else "close" if abs(d) <= CLOSE else "GAP"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    args = ap.parse_args()
    runs = [r for r in sorted(glob.glob(str(ROOT / "runs" / "*_baselines")))
            if len(pd.read_csv(Path(r) / "auc.csv").dataset.unique()) == 15] if not args.run else [args.run]
    run = Path(runs[-1])
    df = pd.read_csv(run / "auc.csv")
    order = list(dataset_specs())
    paper = pd.read_csv(ROOT / "paper" / "table2_paper.csv", index_col=0).loc[order]
    ours_fgas = pd.read_csv(ROOT / "results" / "table2_fgas.csv", index_col=0)["abs_auc"]

    # best over k (per seed), then mean +- std over seeds (only IForest has >1 seed)
    best_k = df.groupby(["dataset", "method", "seed"])["auc"].max().reset_index()
    agg = best_k.groupby(["dataset", "method"])["auc"].agg(["mean", "std", "count"]).reset_index()
    argk = (df.loc[df.groupby(["dataset", "method", "seed"])["auc"].idxmax()]
            .groupby(["dataset", "method"])["k"].first())
    methods = list(dict.fromkeys(df.method))

    rows = []
    for _, r in agg.iterrows():
        p = paper.loc[r.dataset, r.method]
        rows.append(dict(dataset=r.dataset, method=r.method, paper=p, ours=r["mean"],
                         ours_std=r["std"] if r["count"] > 1 else None,
                         best_k=argk.get((r.dataset, r.method)), delta=r["mean"] - p,
                         pct=100 * (r["mean"] - p) / p, status=status(r["mean"] - p)))
    cmp_ = pd.DataFrame(rows)
    cmp_["dataset"] = pd.Categorical(cmp_.dataset, order)
    cmp_ = cmp_.sort_values(["method", "dataset"])
    out = ROOT / "results"
    cmp_.to_csv(out / "baselines_vs_paper.csv", index=False, float_format="%.6f")

    mixed = paper.copy()
    for m in methods:
        mixed[m] = agg[agg.method == m].set_index("dataset").loc[order, "mean"]
    mixed["FGAS"] = ours_fgas.loc[order]
    mixed.to_csv(out / "table2_mixed_baselines.csv", float_format="%.6f")

    wide_o = mixed[methods + ["FGAS"]]
    lines = [f"# Re-run baselines vs paper (Table 2)", "",
             f"Run: `{run.name}`. Best AUC over k = 1..60 (kNN, COF, FastABOD, LoOP); ECOD has no "
             "parameter; IForest = mean ± std over 5 seeds (100 trees). "
             f"Status: |Δ| ≤ {MATCH} match, ≤ {CLOSE} close, else GAP.",
             "Libraries: pyod 3.6.7 (kNN 'largest', COF, ABOD method='fast', ECOD, IForest), "
             "PyNomaly 0.4.0 (LoOP).", "",
             "## Per-method summary", "",
             "| Method | Paper avg | Ours avg | Δ avg | match / close / GAP |", "|---|---|---|---|---|"]
    for m in methods:
        c = cmp_[cmp_.method == m]
        lines.append(f"| {m} | {c.paper.mean():.3f} | {c.ours.mean():.3f} | {c.ours.mean() - c.paper.mean():+.3f} | "
                     f"{(c.status == 'match').sum()} / {(c.status == 'close').sum()} / {(c.status == 'GAP').sum()} |")
    lines += ["", "## Per dataset (ours / paper)", "",
              "| Dataset | " + " | ".join(methods) + " | FGAS (ours abs) |",
              "|---|" + "---|" * (len(methods) + 1)]
    for ds in order:
        cells = []
        for m in methods:
            r = cmp_[(cmp_.method == m) & (cmp_.dataset == ds)].iloc[0]
            sd = f"±{r.ours_std:.3f}" if pd.notna(r.ours_std) else ""
            cells.append(f"{r.ours:.3f}{sd} / {r.paper:.3f}")
        lines.append(f"| {ds} | " + " | ".join(cells) + f" | {ours_fgas[ds]:.3f} |")
    lines.append("| **Average** | " + " | ".join(f"{wide_o[m].mean():.3f} / {paper[m].mean():.3f}" for m in methods)
                 + f" | {ours_fgas.mean():.3f} |")
    wins = mixed.round(3).eq(mixed.round(3).max(axis=1), axis=0).sum()
    lines += ["", "## Mixed Table 2 (our FGAS + our re-run baselines + paper-reported DIS, INFLO, LDOF, "
              "ODIN, DCROD, OutRanka, WNINOD): '1st order' (3-dp ties count for all)", "",
              "| " + " | ".join(wins.index) + " |", "|" + "---|" * len(wins),
              "| " + " | ".join(str(int(v)) for v in wins) + " |",
              "", f"Mixed-table averages: " + ", ".join(f"{c} {mixed[c].mean():.3f}" for c in mixed.columns)]
    (out / "baselines_vs_paper.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
