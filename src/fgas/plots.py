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


def fig3_cd(avg_rank: "pd.Series", cd: float, path, title: str = "Nemenyi's test figure on AUC."):
    """Critical-difference diagram in the paper's orientation: axis k (left) -> 1 (right),
    i.e. position = k + 1 - average rank (best algorithm on the left)."""
    k = len(avg_rank)
    pos = (k + 1 - avg_rank).sort_values(ascending=False)          # best first
    names = list(pos.index)
    half = (k + 1) // 2
    left, right = names[:half], names[half:][::-1]                 # right side: worst first
    fig, ax = plt.subplots(figsize=(7.2, 0.9 + 0.32 * max(len(left), len(right))))
    X = lambda v: k + 1 - v  # noqa: E731  (data x so that 'k' is drawn on the left)
    ax.set_xlim(X(k + 0.3), X(0.7))
    top = 0.0
    ax.plot([X(k), X(1)], [top, top], color="black", lw=1)
    for t in range(1, k + 1):
        ax.plot([X(t), X(t)], [top, top + 0.06], color="black", lw=1)
        ax.text(X(t), top + 0.1, str(t), ha="center", va="bottom", fontsize=10)
    # CD bar above the axis, starting at the left end
    y_cd = top + 0.55
    ax.plot([X(k), X(k - cd)], [y_cd, y_cd], color="red", lw=1)
    for x in (X(k), X(k - cd)):
        ax.plot([x, x], [y_cd - 0.05, y_cd + 0.05], color="red", lw=1)
    ax.text((X(k) + X(k - cd)) / 2, y_cd + 0.08, f"CD={cd:.4f}", ha="center", va="bottom", fontsize=10)
    step = 0.3
    line_col = "#6A6ADB"
    for i, n in enumerate(left):
        y = -0.45 - step * (i + 1)
        x = X(pos[n])
        ax.plot([x, x, X(k + 0.1)], [top, y, y], color=line_col, lw=0.8)
        ax.text(X(k + 0.2), y, n, ha="right", va="center", fontsize=10)
    for i, n in enumerate(right):
        y = -0.45 - step * (i + 1)
        x = X(pos[n])
        ax.plot([x, x, X(0.9)], [top, y, y], color=line_col, lw=0.8)
        ax.text(X(0.8), y, n, ha="left", va="center", fontsize=10)
    # cliques: maximal groups whose rank range is < CD (not significantly different)
    r = pos.sort_values()  # ascending position
    vals = r.to_numpy()
    groups = []
    for i in range(len(vals)):
        j = max(jj for jj in range(i, len(vals)) if vals[jj] - vals[i] < cd)
        if j > i and not any(a <= i and j <= b for a, b in groups):
            groups.append((i, j))
    for g, (i, j) in enumerate(groups):
        y = -0.12 - 0.1 * g
        ax.plot([X(vals[i]) + 0.03, X(vals[j]) - 0.03], [y, y], color="red", lw=1.2)
    ax.set_ylim(-0.45 - step * (max(len(left), len(right)) + 1), y_cd + 0.4)
    ax.axis("off")
    fig.text(0.5, 0.01, title, ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# Colours approximating the paper's Fig. 1 legend; FGAS is black as in the paper.
ROC_COLORS = {"DIS": "#FF0000", "COF": "#00CC00", "FastABOD": "#00E5FF", "INFLO": "#FF00FF",
              "kNN": "#3333AA", "LDOF": "#E06A00", "LoOP": "#E0B000", "ODIN": "#8000A0",
              "DCROD": "#2E8B57", "ECOD": "#20B2AA", "IForest": "#1F3FBF", "OutRanka": "#3070B0",
              "WNINOD": "#0000FF", "FGAS": "#000000"}


def fig1_roc(curves: dict, order: list, path, methods: list):
    """curves[(dataset, method)] = (fpr, tpr). 5x3 grid, axes in %, FGAS drawn last (on top)."""
    from string import ascii_lowercase
    fig, axes = plt.subplots(5, 3, figsize=(10.5, 16))
    for i, (ax, ds) in enumerate(zip(axes.flat, order)):
        for m in methods:
            if (ds, m) in curves:
                fpr, tpr = curves[(ds, m)]
                ax.plot(100 * fpr, 100 * tpr, color=ROC_COLORS.get(m, "gray"), lw=1.6 if m == "FGAS" else 0.9,
                        label=m, zorder=3 if m == "FGAS" else 2, drawstyle="default")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(0, 101, 10))
        ax.set_yticks(range(0, 101, 10))
        ax.tick_params(labelsize=7, direction="in", top=True, right=True)
        ax.set_xlabel("FPR (%)", fontsize=8)
        ax.set_ylabel("TPR (%)", fontsize=8)
        ax.legend(loc="lower right", fontsize=6, frameon=True, fancybox=False, edgecolor="black",
                  handlelength=1.5, borderpad=0.3, labelspacing=0.15)
        ax.set_title(f"({ascii_lowercase[i]})  {ds}", y=-0.32, fontsize=11)
    fig.text(0.5, 0.005, "Fig. 1 (reproduced). Comparison results on ROC — FGAS and the re-run baselines "
             "only (paper-only baselines have no scores to plot).", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.015, 1, 1), h_pad=2.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig2_noise(mean: "pd.DataFrame", path, lo: "pd.DataFrame | None" = None, hi: "pd.DataFrame | None" = None,
               title: str = "Fig. 2 (reproduced). Effect of attribute noise on AUC."):
    """mean/lo/hi: index = noise level, columns = dataset. Optional band = min..max over seeds."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ylims = [(0.90, 1.0), (0.55, 1.0)]
    yt = [[0.90 + 0.01 * i for i in range(11)], [0.55 + 0.05 * i for i in range(10)]]
    for k, (ax, names) in enumerate(zip(axes, PANELS)):
        for i, ds in enumerate(names):
            if ds not in mean:
                continue
            st = _style(i)
            ax.plot(mean.index, mean[ds], label=ds, **st)
            if lo is not None:
                ax.fill_between(mean.index, lo[ds], hi[ds], color=st["color"], alpha=0.12, lw=0)
        ax.set_xlabel("Noise level")
        ax.set_ylabel("AUC")
        ax.set_ylim(*ylims[k])
        ax.set_yticks(yt[k])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
        ax.set_xticks([round(0.05 * i, 2) for i in range(11)])
        ax.set_xticklabels([f"{0.05 * i:g}" for i in range(11)])
        ax.set_xlim(0, 0.5)
        ax.tick_params(direction="in", top=True, right=True)
        ax.legend(loc="lower right" if k == 1 else "lower right", frameon=True, fancybox=False,
                  edgecolor="black", fontsize=9)
        ax.set_title(f"({'ab'[k]})  Datasets {'1-8' if k == 0 else '9-15'}", y=-0.28, fontsize=11)
    fig.suptitle(title, y=0.02, fontsize=10, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)
