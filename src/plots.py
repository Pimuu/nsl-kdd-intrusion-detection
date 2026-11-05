"""Figure generation (matplotlib, static PNG for the paper).

Colour rules: each model / split keeps one fixed colour across every figure
(colour follows the entity, never its rank); magnitude (confusion matrices)
uses a single-hue sequential ramp; text is always neutral ink, never the
series colour.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import config  # noqa: E402
from .models import MODEL_LABELS  # noqa: E402

# Validated categorical palette, assigned in fixed slot order.
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
NEUTRAL = "#8a8984"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CLASS_COLORS = dict(zip(config.CATEGORY_ORDER, SLOT))
MODEL_COLORS = {"majority": NEUTRAL, "logreg": SLOT[0], "random_forest": SLOT[1], "hist_gb": SLOT[2]}
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_class_distribution(train: pd.DataFrame, test: pd.DataFrame, path: Path):
    """Share of each category in KDDTrain+ vs KDDTest+ (exposes the shift)."""
    cats = config.CATEGORY_ORDER
    tr = train["category"].value_counts().reindex(cats, fill_value=0)
    te = test["category"].value_counts().reindex(cats, fill_value=0)
    x = np.arange(len(cats))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for off, counts, label, color in [(-w / 2, tr, f"Train (n={len(train):,})", SLOT[0]),
                                      (w / 2, te, f"Test (n={len(test):,})", SLOT[1])]:
        share = 100 * counts / counts.sum()
        bars = ax.bar(x + off, share, w - 0.03, color=color, label=label)
        for b, s, n in zip(bars, share, counts):
            ax.annotate(f"{s:.1f}%\n({n:,})" if s >= 1 else f"{s:.2f}%\n({n:,})", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=7.5, color=INK_2,
                        xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x, cats)
    ax.set_ylabel("Share of records (%)")
    ax.set_ylim(0, max(100 * tr.max() / tr.sum(), 100 * te.max() / te.sum()) * 1.22)
    ax.set_title("Class distribution: training vs. test split")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    _save(fig, path)


def plot_confusion(cm: np.ndarray, labels: list[str], title: str, path: Path):
    """Heatmap coloured by row-normalised recall, annotated with counts."""
    row = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row, out=np.zeros_like(cm, dtype=float), where=row > 0)
    size = 3.2 + 0.75 * len(labels)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(norm, cmap=SEQ_BLUE, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if norm[i, j] > 0.55 else INK
            ax.text(j, i, f"{cm[i, j]:,}\n{100 * norm[i, j]:.1f}%", ha="center", va="center",
                    fontsize=8.5, color=color)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Row-normalised rate")
    cb.outline.set_visible(False)
    _save(fig, path)


def plot_roc(curves: dict, path: Path, title: str, colors: dict | None = None,
             labels: dict | None = None):
    """curves: {panel_name: {series: (fpr, tpr, auc)}} -> one panel per entry.

    Series default to models (MODEL_COLORS / MODEL_LABELS); pass ``colors`` and
    ``labels`` to plot other series, e.g. classes in a one-vs-rest ROC.
    """
    colors = colors or MODEL_COLORS
    labels = labels or MODEL_LABELS
    splits = list(curves)
    fig, axes = plt.subplots(1, len(splits), figsize=(5.2 * len(splits), 4.8), squeeze=False)
    for ax, split in zip(axes[0], splits):
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color=NEUTRAL, label="Chance (AUC = 0.500)")
        for model, (fpr, tpr, auc) in curves[split].items():
            ax.plot(fpr, tpr, lw=2, color=colors[model],
                    label=f"{labels.get(model, model)} (AUC = {auc:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_xlabel("False alarm rate (FPR)")
        ax.set_ylabel("Detection rate (TPR)")
        ax.set_title(split)
        ax.legend(loc="lower right", fontsize=8.5)
        ax.set_aspect("equal")
    fig.suptitle(title, fontweight="bold", y=1.0 if len(splits) > 1 else 1.02)
    _save(fig, path)


def plot_feature_importance(imp: pd.DataFrame, path: Path, top: int = 15):
    """Two panels: impurity importance and permutation importance (test set)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    panels = [("impurity", "Impurity (MDI) importance\nrandom forest, training data", None),
              ("perm_mean", "Permutation importance (drop in ROC-AUC)\nrandom forest, KDDTest+", "perm_std")]
    for ax, (col, title, err) in zip(axes, panels):
        d = imp.sort_values(col, ascending=False).head(top)[::-1]
        ax.barh(d.index, d[col], xerr=None if err is None else d[err], color=SLOT[0], height=0.7,
                error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2})
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_xlabel("Mean decrease in impurity (summed over one-hot columns)")
    axes[1].set_xlabel("Mean decrease in ROC-AUC (5 repeats, ±1 sd)")
    fig.suptitle("Top-15 features for binary attack detection", fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def plot_grouped_bars(df: pd.DataFrame, path: Path, title: str, ylabel: str, color_map: dict,
                      label_map: dict | None = None, fmt: str = "{:.0f}"):
    """Grouped bar chart: rows of ``df`` = x groups, columns = series (in percent)."""
    label_map = label_map or {}
    n = len(df.columns)
    x = np.arange(len(df.index))
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(6.5, 1.5 * len(df.index) + 2), 4.2))
    for k, col in enumerate(df.columns):
        bars = ax.bar(x - 0.4 + w * (k + 0.5), df[col], w - 0.02, color=color_map[col],
                      label=label_map.get(col, col))
        for b, v in zip(bars, df[col]):
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=7.5, color=INK_2,
                        xytext=(0, 1.5), textcoords="offset points")
    ax.set_xticks(x, df.index)
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=n, fontsize=8.5)
    _save(fig, path)
