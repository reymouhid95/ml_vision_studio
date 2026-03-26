from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def make_confusion_figure(
    preds: list[int],
    actuals: list[int],
    class_names: list[str],
) -> plt.Figure:
    """
    Build a matplotlib figure with:
      - Top subplot: N×N confusion matrix heatmap
      - Bottom subplot: per-class F1 / Precision / Recall bar chart
    """
    n = len(class_names)
    matrix = np.zeros((n, n), dtype=int)
    for p, a in zip(preds, actuals):
        matrix[a][p] += 1

    # ── Metrics ────────────────────────────────────────────
    precisions, recalls, f1s = [], [], []
    for i in range(n):
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    # ── Figure ─────────────────────────────────────────────
    fig_h = max(6, n * 0.7 + 3)
    fig, (ax_cm, ax_bar) = plt.subplots(
        2, 1,
        figsize=(max(5, n * 0.9 + 1), fig_h),
        gridspec_kw={"height_ratios": [3, 1.5]},
    )
    fig.patch.set_facecolor("#1a1a2e")

    # Heatmap
    norm_mat = matrix.astype(float)
    row_sums = norm_mat.sum(axis=1, keepdims=True)
    norm_mat = norm_mat / (row_sums + 1e-8)

    for i in range(n):
        for j in range(n):
            val = norm_mat[i, j]
            if i == j:
                color = (0.063, 0.725, 0.506, 0.1 + val * 0.7)
            else:
                color = (0.937, 0.267, 0.267, 0.05 + val * 0.5)
            ax_cm.add_patch(mpatches.Rectangle((j, n - i - 1), 1, 1, color=color))
            txt = str(matrix[i, j]) if matrix[i, j] > 0 else "·"
            ax_cm.text(
                j + 0.5, n - i - 0.5, txt,
                ha="center", va="center",
                fontsize=9, color="white",
                fontweight="bold" if i == j else "normal",
            )

    ax_cm.set_xlim(0, n)
    ax_cm.set_ylim(0, n)
    ax_cm.set_xticks(np.arange(n) + 0.5)
    ax_cm.set_yticks(np.arange(n) + 0.5)
    short = [c[:10] for c in class_names]
    ax_cm.set_xticklabels(short, rotation=30, ha="right", color="white", fontsize=8)
    ax_cm.set_yticklabels(short[::-1], color="white", fontsize=8)
    ax_cm.set_title("Matrice de confusion", color="white", fontsize=11, pad=8)
    ax_cm.set_facecolor("#0f0f1a")
    ax_cm.tick_params(colors="white")
    for spine in ax_cm.spines.values():
        spine.set_edgecolor("#333")

    # Bar chart
    x = np.arange(n)
    width = 0.25
    colors_map = {"F1": "#a855f7", "Précision": "#10b981", "Rappel": "#f59e0b"}
    for k, (vals, label) in enumerate(
        zip([f1s, precisions, recalls], ["F1", "Précision", "Rappel"])
    ):
        ax_bar.bar(x + k * width, vals, width, label=label, color=colors_map[label], alpha=0.85)

    ax_bar.set_xticks(x + width)
    ax_bar.set_xticklabels(short, rotation=30, ha="right", color="white", fontsize=7)
    ax_bar.set_ylim(0, 1.1)
    ax_bar.set_yticks([0, 0.5, 1.0])
    ax_bar.tick_params(colors="white")
    ax_bar.set_facecolor("#0f0f1a")
    ax_bar.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white",
                  loc="upper right", framealpha=0.6)
    ax_bar.set_title("Par classe", color="white", fontsize=9, pad=4)
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#333")

    fig.tight_layout(pad=1.5)
    return fig
