"""
K-Means clustering non supervisé + méthode du coude + t-SNE.

Usage principal : clustering sur les embeddings MNIST déjà chargés.
Sous-échantillonne automatiquement pour garder des temps de calcul raisonnables.
"""
from __future__ import annotations

from typing import Generator

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# ─── Style ────────────────────────────────────────────────────────────────────

DARK_BG = "#1a1a2e"
PANEL_BG = "#0f0f1a"
PALETTE = [
    "#a855f7", "#22d3ee", "#f97316", "#4ade80", "#f43f5e",
    "#facc15", "#60a5fa", "#e879f9", "#34d399", "#fb923c",
    "#818cf8", "#f472b6", "#2dd4bf", "#fbbf24", "#c084fc",
]


# ─── Préparation des données ──────────────────────────────────────────────────

def subsample(X: np.ndarray, n_max: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (X_sub, idx) sous-échantillonné à n_max exemples max."""
    rng = np.random.default_rng(seed)
    n = min(n_max, len(X))
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx], idx


def flatten(X: np.ndarray) -> np.ndarray:
    """Aplatit (N, H, W, C) ou (N, H, W) → (N, H*W*C)."""
    return X.reshape(len(X), -1)


# ─── Méthode du coude ────────────────────────────────────────────────────────

def run_elbow(
    X_flat: np.ndarray,
    k_max: int = 12,
    n_elbow: int = 3000,
    seed: int = 42,
) -> tuple[list[int], list[float], list[float], int]:
    """
    Calcule inertie + silhouette pour K = 2 … k_max.

    Returns
    -------
    ks, inertias, silhouettes, best_k
    """
    X_sub, _ = subsample(X_flat, n_elbow, seed)
    X_scaled = StandardScaler().fit_transform(X_sub)

    ks, inertias, silhouettes = [], [], []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=200)
        labels = km.fit_predict(X_scaled)
        ks.append(k)
        inertias.append(float(km.inertia_))
        sil = float(silhouette_score(X_scaled, labels, sample_size=min(1500, len(X_sub))))
        silhouettes.append(sil)

    best_k = ks[int(np.argmax(silhouettes))]
    return ks, inertias, silhouettes, best_k


# ─── K-Means complet ─────────────────────────────────────────────────────────

def run_kmeans(
    X_flat: np.ndarray,
    k: int,
    true_labels: np.ndarray | None = None,
    n_kmeans: int = 5000,
    seed: int = 42,
) -> dict:
    """
    K-Means sur X_flat (sous-échantillonné à n_kmeans).

    Returns
    -------
    dict avec : labels, idx, inertia, silhouette, ari (si true_labels fournis),
                X_sub (données utilisées), scaler
    """
    X_sub, idx = subsample(X_flat, n_kmeans, seed)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)

    km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)
    labels = km.fit_predict(X_scaled)

    inertia = float(km.inertia_)
    sil = float(silhouette_score(X_scaled, labels, sample_size=min(2000, len(X_sub))))

    ari = None
    if true_labels is not None:
        true_sub = true_labels[idx]
        ari = float(adjusted_rand_score(true_sub, labels))

    return {
        "km": km,
        "scaler": scaler,
        "labels": labels,
        "idx": idx,
        "inertia": inertia,
        "silhouette": sil,
        "ari": ari,
        "X_sub": X_sub,
    }


# ─── t-SNE ───────────────────────────────────────────────────────────────────

def run_tsne(
    X_sub: np.ndarray,
    perplexity: int = 30,
    n_tsne: int = 2000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    t-SNE 2D sur un sous-ensemble de X_sub.

    Returns (X_2d, tsne_idx) — indices dans X_sub utilisés.
    """
    X_t, tsne_idx = subsample(X_sub, n_tsne, seed)
    X_scaled = StandardScaler().fit_transform(X_t)
    tsne = TSNE(
        n_components=2, random_state=seed,
        perplexity=perplexity, max_iter=500, verbose=0,
    )
    X_2d = tsne.fit_transform(X_scaled)
    return X_2d, tsne_idx


# ─── Figures ─────────────────────────────────────────────────────────────────

def _ax_style(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors="white")
    ax.grid(alpha=0.2, color="#444")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")


def make_elbow_figure(
    ks: list[int],
    inertias: list[float],
    silhouettes: list[float],
    best_k: int | None = None,
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor(DARK_BG)
    _ax_style(ax1)
    _ax_style(ax2)

    ax1.plot(ks, inertias, color="#a855f7", linewidth=2, marker="o", markersize=6)
    if best_k and best_k in ks:
        i = ks.index(best_k)
        ax1.scatter([best_k], [inertias[i]], color="#f97316", s=130, zorder=6,
                    label=f"Meilleur K = {best_k}")
        ax1.legend(facecolor=PANEL_BG, labelcolor="white", fontsize=9)
    ax1.set_xlabel("K (nombre de clusters)", color="white", fontsize=9)
    ax1.set_ylabel("Inertie (WCSS)", color="white", fontsize=9)
    ax1.set_title("Méthode du coude", color="white", fontsize=11, fontweight="bold")

    ax2.plot(ks, silhouettes, color="#22d3ee", linewidth=2, marker="s", markersize=6)
    if best_k and best_k in ks:
        i = ks.index(best_k)
        ax2.scatter([best_k], [silhouettes[i]], color="#f97316", s=130, zorder=6)
    ax2.set_xlabel("K (nombre de clusters)", color="white", fontsize=9)
    ax2.set_ylabel("Score silhouette ↑", color="white", fontsize=9)
    ax2.set_title("Silhouette (plus haut = mieux)", color="white", fontsize=11, fontweight="bold")

    fig.tight_layout(pad=1.5)
    return fig


def make_tsne_figure(
    X_2d: np.ndarray,
    labels: np.ndarray,
    true_labels: np.ndarray | None = None,
    k: int = 10,
) -> plt.Figure:
    """Projection t-SNE colorée par clusters et (optionnel) vraies étiquettes."""
    ncols = 2 if true_labels is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    if ncols == 1:
        axes = [axes]
    fig.patch.set_facecolor(DARK_BG)

    def _scatter(ax, color_labels, title, n_unique):
        _ax_style(ax)
        colors = [PALETTE[int(lbl) % len(PALETTE)] for lbl in color_labels]
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, alpha=0.55, s=5)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold")
        ax.set_xlabel("t-SNE dim 1", color="white", fontsize=9)
        ax.set_ylabel("t-SNE dim 2", color="white", fontsize=9)
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=PALETTE[i % len(PALETTE)],
                       markersize=7, label=str(i))
            for i in range(n_unique)
        ]
        ax.legend(handles=handles, facecolor=PANEL_BG, labelcolor="white",
                  fontsize=8, ncol=min(5, n_unique), loc="upper right",
                  framealpha=0.7)

    _scatter(axes[0], labels, f"Clusters K-Means (K = {k})", k)
    if true_labels is not None:
        n_true = len(np.unique(true_labels))
        _scatter(axes[1], true_labels,
                 f"Vraies étiquettes ({n_true} classes)", n_true)

    fig.tight_layout(pad=1.5)
    return fig
