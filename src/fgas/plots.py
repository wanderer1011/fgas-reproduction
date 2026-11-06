"""Figures in the paper's (MATLAB-default) visual layout."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# MATLAB default line colours (R2014b+) and the per-panel marker order used in Figs. 2 and 4.
MATLAB_COLORS = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F"]
PANEL_MARKERS = ["v", "o", ">", "*", "s", "^", "D", "*"]  # 8th (Pima) re-uses blue with a star
PANELS = (["Card", "Diab", "Ecoli", "Ionos", "Iris", "Musk", "Page", "Pima"],
          ["Sate", "Sonar", "Thyro", "Wbc", "Wdbc", "Wine", "Wpbc"])

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.linewidth": 0.8})


def _style(i: int) -> dict:
    marker = PANEL_MARKERS[i]
    return dict(color=MATLAB_COLORS[i % 7], marker=marker, markersize=6 if marker != "*" else 8,
                markerfacecolor="none" if marker != "*" else MATLAB_COLORS[i % 7], linewidth=1.2)


def two_panel_lines(curves: pd.DataFrame, xlabel: str, ylims, xticks, title: str, path,
                    xlim=None, yticks=None, legend_loc=("lower right", "lower right")):
    """curves: index = x values, columns = dataset abbr. Draws (a) datasets 1-8, (b) 9-15."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, (ax, names) in enumerate(zip(axes, PANELS)):
        for i, ds in enumerate(names):
            if ds in curves:
                ax.plot(curves.index, curves[ds], label=ds, **_style(i))
        ax.set_xlabel(xlabel, style="italic" if xlabel == "σ" else "normal")
        ax.set_ylabel("AUC")
        ax.set_ylim(*ylims[k])
        if yticks:
            ax.set_yticks(yticks[k])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
        ax.set_xticks(xticks)
        if xlim:
            ax.set_xlim(*xlim)
        ax.tick_params(direction="in", top=True, right=True)
        ax.legend(loc=legend_loc[k], frameon=True, fancybox=False, edgecolor="black", fontsize=9)
        ax.set_title(f"({'ab'[k]})  Datasets {'1-8' if k == 0 else '9-15'}", y=-0.28, fontsize=11)
    fig.suptitle(title, y=0.02, fontsize=10, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig4_sigma(df: pd.DataFrame, fgd: str, path, plot_stop: float = 0.95):
    d = df[(df.fgd == fgd) & (df.sigma <= plot_stop + 1e-9)]
    curves = d.pivot_table(index="sigma", columns="dataset", values="auc")
    two_panel_lines(curves, "σ", ylims=[(0.85, 1.0), (0.4, 1.0)], xticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
                    xlim=(0, 1),
                    yticks=[[0.85, 0.9, 0.95, 1.0], [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]],
                    title=f"Fig. 4 (reproduced, FGD = {fgd}). The variation curve of AUC on σ.", path=path)
