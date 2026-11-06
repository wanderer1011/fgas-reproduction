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
from fgas.plots import fig1_roc, fig2_noise, fig3_cd, fig4_sigma  # noqa: E402
from fgas.data import dataset_specs, load_dataset  # noqa: E402
from fgas.metrics import roc_points  # noqa: E402
import numpy as np  # noqa: E402
from fgas.stats import friedman_nemenyi  # noqa: E402


def latest(pattern: str) -> str:
    runs = sorted(glob.glob(str(ROOT / "runs" / pattern)))
    if not runs:
        raise SystemExit(f"no run matching {pattern}")
    return runs[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("figure", choices=["fig1", "fig2", "fig3", "fig4"])
    ap.add_argument("--run")
    args = ap.parse_args()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    if args.figure == "fig1":
        t2 = Path(latest("*_table2"))
        best = pd.read_csv(out / "table2_fgas.csv", index_col=0)
        fg = np.load(t2 / "scores.npz")
        bl_run = Path(args.run) if args.run else Path([r for r in sorted(glob.glob(str(ROOT / "runs" / "*_baselines")))
                                                      if len(pd.read_csv(Path(r) / "auc.csv").dataset.unique()) == 15][-1])
        bl = np.load(bl_run / "scores.npz")
        order = list(dataset_specs())
        curves, methods = {}, []
        for key in bl.files:
            ds, m, _ = key.split("|")
            y = load_dataset(ds).y
            curves[(ds, m)] = roc_points(y, bl[key])
            if m not in methods:
                methods.append(m)
        for ds in order:
            y = load_dataset(ds).y
            s = fg[f"{ds}|abs|cuda|float64|{best.loc[ds, 'abs_sigma']:.2f}"]
            curves[(ds, "FGAS")] = roc_points(y, s)
        p = out / "fig1_roc.png"
        fig1_roc(curves, order, p, methods + ["FGAS"])
        print("wrote", p, "| baselines from", bl_run.name)
    if args.figure == "fig2":
        run = Path(args.run or latest("*_noise"))
        df = pd.read_csv(run / "auc.csv")
        df["level"] = df.level.round(2)
        order = list(dataset_specs())
        # main protocol: sigma re-tuned per (level, seed)
        best = df.groupby(["dataset", "level", "seed"]).auc.max().reset_index()
        # secondary: sigma fixed at the clean (level 0) best on the same step-0.1 grid
        clean = df[df.level == 0].loc[lambda d: d.groupby("dataset").auc.idxmax()].set_index("dataset").sigma
        fixed = df[df.apply(lambda r: abs(r.sigma - clean[r.dataset]) < 1e-9, axis=1)]
        fixed = fixed.groupby(["dataset", "level", "seed"]).auc.max().reset_index()
        read = pd.read_csv(ROOT / "paper" / "fig2_readings.csv", comment="#", index_col=0)
        read.columns = [round(float(c), 2) for c in read.columns]
        summary = []
        for name, d in [("retuned", best), ("fixed", fixed)]:
            g = d.groupby(["level", "dataset"]).auc
            mean, lo, hi = (g.mean().unstack(), g.min().unstack(), g.max().unstack())
            fig2_noise(mean, out / f"fig2_noise_{name}.png", lo, hi,
                       title=f"Fig. 2 (reproduced; one-attribute noise, sigma {name}, mean of seeds, band = min-max).")
            s0 = d[d.seed == 0].groupby(["level", "dataset"]).auc.mean().unstack()
            fig2_noise(s0, out / f"fig2_noise_{name}_seed0.png",
                       title=f"Fig. 2 (reproduced; one-attribute noise, sigma {name}, single seed 0).")
            for ds in order:
                ours, pap = mean[ds], read.loc[ds]
                summary.append(dict(protocol=name, dataset=ds, clean=ours.loc[0.0], paper_clean=pap.loc[0.0],
                                    ours_at_0_5=ours.loc[0.5], paper_at_0_5=pap.loc[0.5],
                                    ours_min=ours.min(), paper_min=pap.min(),
                                    ours_max_drop=ours.loc[0.0] - ours.min(), paper_max_drop=pap.loc[0.0] - pap.min(),
                                    mae_vs_paper=(ours - pap).abs().mean(),
                                    seed_spread=(hi[ds] - lo[ds]).max()))
        sm = pd.DataFrame(summary)
        sm.to_csv(out / "fig2_vs_paper.csv", index=False, float_format="%.4f")
        lines = ["# Fig. 2 — attribute noise: ours vs paper (read off the figure, +-0.003)", "",
                 f"Run: `{run.name}`; one-attribute noise; sigma grid 0..1 step 0.1; mean over seeds.", ""]
        for name in ["retuned", "fixed"]:
            t = sm[sm.protocol == name]
            lines += [f"## sigma {name}", "",
                      "| Dataset | clean ours/paper | x=0.5 ours/paper | max drop ours/paper | MAE vs paper | seed spread |",
                      "|---|---|---|---|---|---|"]
            for _, r in t.iterrows():
                lines.append(f"| {r.dataset} | {r.clean:.3f} / {r.paper_clean:.3f} | {r.ours_at_0_5:.3f} / "
                             f"{r.paper_at_0_5:.3f} | {r.ours_max_drop:.3f} / {r.paper_max_drop:.3f} | "
                             f"{r.mae_vs_paper:.4f} | {r.seed_spread:.3f} |")
            lines += ["", f"Mean MAE vs paper: {t.mae_vs_paper.mean():.4f}; datasets with MAE <= 0.01: "
                          f"{(t.mae_vs_paper <= 0.01).sum()}/15", ""]
        (out / "fig2_vs_paper.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
    if args.figure == "fig3":
        paper = pd.read_csv(ROOT / "paper" / "table2_paper.csv", index_col=0)
        ours = pd.read_csv(out / "table2_fgas.csv", index_col=0)
        tables = {"paper": paper,
                  "ours_fgas": paper.assign(FGAS=ours["abs_auc"])}  # our FGAS (full precision), paper baselines
        mixed = out / "table2_mixed_baselines.csv"
        if mixed.exists():
            tables["ours_rerun"] = pd.read_csv(mixed, index_col=0)
        lines = ["# Friedman / Nemenyi (alpha = 0.10)", "",
                 "| Table | tau_F | df | F_crit | CD | FGAS avg rank | FGAS not sig. different from |",
                 "|---|---|---|---|---|---|---|"]
        for name, t in tables.items():
            r = friedman_nemenyi(t)
            p = out / f"fig3_cd_{name}.png"
            fig3_cd(r.avg_rank, r.CD, p)
            close = [a for a in r.avg_rank.index if a != "FGAS"
                     and abs(r.avg_rank[a] - r.avg_rank["FGAS"]) < r.CD]
            lines.append(f"| {name} | {r.tau_F:.4f} | {r.df} | {r.F_crit:.4f} | {r.CD:.4f} | "
                         f"{r.avg_rank['FGAS']:.3f} | {', '.join(close)} |")
            (out / f"fig3_ranks_{name}.csv").write_text(r.avg_rank.sort_values().to_csv())
            print("wrote", p)
        lines += ["", "Paper (Section 4.4): tau_F = 15.1347, F(13,182), F_crit = 1.5610, CD = 4.7659; "
                  "FGAS not significantly different from kNN, WNINOD, IForest, DCROD."]
        (out / "fig3_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
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
