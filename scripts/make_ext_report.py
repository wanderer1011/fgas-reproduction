"""Report for the real-world extension datasets (Mammo, Annthy, SECOM): FGAS vs re-run baselines.

Usage: python scripts/make_ext_report.py
Reads the latest runs/*_ext_fgas and all runs/*_ext_baselines; writes results/EXTENSIONS_table.{md,csv},
results/ext_roc.png, results/ext_sigma.png.
"""
import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fgas.data import ext_specs, load_dataset  # noqa: E402
from fgas.metrics import roc_points  # noqa: E402
from fgas.plots import ROC_COLORS  # noqa: E402

FIXED_SIGMA = 0.8   # best single global sigma on the 15 paper datasets (RESULTS.md §8)


def main() -> int:
    specs = ext_specs()
    order = list(specs)
    fg_run = Path(sorted(glob.glob(str(ROOT / "runs" / "*_ext_fgas")))[-1])
    fg = pd.read_csv(fg_run / "auc.csv")
    bl_runs = [r for r in sorted(glob.glob(str(ROOT / "runs" / "*_ext_baselines")))
               if (Path(r) / "auc.csv").exists()]           # skip runs still in progress
    pending = [Path(r).name for r in glob.glob(str(ROOT / "runs" / "*_ext_baselines"))
               if not (Path(r) / "auc.csv").exists()]
    if pending:
        print("NOTE: still running, not included:", ", ".join(pending))
    bl = pd.concat([pd.read_csv(Path(r) / "auc.csv").assign(run=Path(r).name) for r in bl_runs])
    # if a method was run twice, keep the latest run for it
    last = bl.groupby("method").run.max()
    bl = bl[bl.apply(lambda r: r.run == last[r.method], axis=1)]

    rows = []
    for ds in order:
        for fgd in ["abs", "squared"]:
            d = fg[(fg.dataset == ds) & (fg.fgd == fgd)]
            b = d.loc[d.auc.idxmax()]
            fx = d[np.isclose(d.sigma, FIXED_SIGMA)].auc.iloc[0]
            rows.append(dict(dataset=ds, method=f"FGAS ({fgd}, best σ)", auc=b.auc, std=np.nan,
                             setting=f"σ={b.sigma:.2f}"))
            rows.append(dict(dataset=ds, method=f"FGAS ({fgd}, σ={FIXED_SIGMA} fixed)", auc=fx, std=np.nan,
                             setting="no labels"))
        for m, g in bl[bl.dataset == ds].groupby("method"):
            per_seed = g.groupby("seed").auc.max()
            k = g.loc[g.auc.idxmax(), "k"] if g.auc.notna().any() else np.nan
            rows.append(dict(dataset=ds, method=m, auc=per_seed.mean(),
                             std=per_seed.std() if len(per_seed) > 1 else np.nan,
                             setting=(f"k={int(k)}" if pd.notna(k) else ("5 seeds" if m == "IForest" else "-"))
                             + (f", {int(g.auc.isna().sum())} NaN fits" if g.auc.isna().any() else "")))
    t = pd.DataFrame(rows)
    t["rank"] = t.groupby("dataset").auc.rank(ascending=False, method="min")
    out = ROOT / "results"
    t.to_csv(out / "EXTENSIONS_table.csv", index=False, float_format="%.4f")

    lines = ["| Method | " + " | ".join(f"{d} ({specs[d]['application']})" for d in order) + " |",
             "|---|" + "---|" * len(order)]
    for m in t.method.unique():
        cells = []
        for ds in order:
            r = t[(t.dataset == ds) & (t.method == m)]
            if r.empty or pd.isna(r.auc.iloc[0]):
                cells.append("n/a")
                continue
            r = r.iloc[0]
            sd = f" ± {r['std']:.3f}" if pd.notna(r["std"]) else ""
            cells.append(f"{'**' if r['rank'] == 1 else ''}{r.auc:.3f}{sd}{'**' if r['rank'] == 1 else ''} "
                         f"(#{int(r['rank'])}; {r.setting})")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    (out / "EXTENSIONS_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("runs:", fg_run.name, "|", ", ".join(Path(r).name for r in bl_runs))

    # ROC: FGAS (abs, best sigma) + baselines (best k) per dataset
    fs = np.load(fg_run / "scores.npz")
    fig, axes = plt.subplots(1, len(order), figsize=(5 * len(order), 4.6))
    for ax, ds in zip(axes, order):
        y = load_dataset(ds).y
        for r in bl_runs:
            z = np.load(Path(r) / "scores.npz")
            for key in z.files:
                d, m, _ = key.split("|")
                if d == ds and last[m] == Path(r).name:
                    fpr, tpr = roc_points(y, z[key])
                    ax.plot(100 * fpr, 100 * tpr, color=ROC_COLORS.get(m, "gray"), lw=0.9, label=m)
        best = fg[(fg.dataset == ds) & (fg.fgd == "abs")].sort_values("auc").iloc[-1]
        fpr, tpr = roc_points(y, fs[f"{ds}|abs|cuda|float64|{best.sigma:.2f}"])
        ax.plot(100 * fpr, 100 * tpr, color="black", lw=1.6, label="FGAS")
        ax.plot([0, 100], [0, 100], ":", color="gray", lw=0.6)
        ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="FPR (%)", ylabel="TPR (%)",
               title=f"{ds} — {specs[ds]['application']}")
        ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "ext_roc.png", dpi=150)
    plt.close(fig)

    # sigma curves
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, ds in enumerate(order):
        for fgd, ls in [("abs", "-"), ("squared", "--")]:
            d = fg[(fg.dataset == ds) & (fg.fgd == fgd)].sort_values("sigma")
            ax.plot(d.sigma, d.auc, ls, marker="o", ms=3, color=f"C{i}", label=f"{ds} ({fgd})")
    ax.set(xlabel="σ", ylabel="AUC", title="FGAS AUC vs σ on application datasets")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "ext_sigma.png", dpi=150)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
