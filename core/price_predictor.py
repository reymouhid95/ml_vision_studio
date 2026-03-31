"""Régression linéaire — prédiction de prix immobilier."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(n_samples: int = 100, noise_level: float = 30.0, seed: int = 42):
    """
    Génère un dataset synthétique : surface (m²) → prix (k€).

    Relation réelle : prix = 2.5 * surface + 50  (k€)
    Avec bruit gaussien contrôlé par noise_level.

    Returns
    -------
    X : ndarray (n, 1)  — surface en m²
    y : ndarray (n,)    — prix en k€
    """
    rng = np.random.default_rng(seed)
    surface = rng.uniform(20, 280, size=n_samples)
    noise   = rng.normal(0, noise_level, size=n_samples)
    price   = 2.5 * surface + 50.0 + noise
    price   = np.clip(price, 10, None)   # prix minimum 10 k€
    return surface.reshape(-1, 1), price


# ─────────────────────────────────────────────────────────────────────────────
#  Entraînement
# ─────────────────────────────────────────────────────────────────────────────

def train_price_model(n_samples: int = 100, noise_level: float = 30.0):
    """
    Entraîne une régression linéaire sur le dataset synthétique.

    Returns
    -------
    model    : LinearRegression entraîné
    metrics  : dict avec R², RMSE, MAE (train + test)
    X_all, y_all  : dataset complet (pour tracé)
    X_test, y_test, y_pred : données test (pour résidus)
    """
    X, y = generate_dataset(n_samples, noise_level)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    metrics = {
        "R²_train":  r2_score(y_train, y_pred_train),
        "R²_test":   r2_score(y_test,  y_pred_test),
        "RMSE_train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "RMSE_test":  np.sqrt(mean_squared_error(y_test,  y_pred_test)),
        "MAE_train":  mean_absolute_error(y_train, y_pred_train),
        "MAE_test":   mean_absolute_error(y_test,  y_pred_test),
        "coef":       float(model.coef_[0]),
        "intercept":  float(model.intercept_),
        "n_train":    len(X_train),
        "n_test":     len(X_test),
    }
    return model, metrics, X, y, X_test, y_test, y_pred_test


# ─────────────────────────────────────────────────────────────────────────────
#  Prédiction
# ─────────────────────────────────────────────────────────────────────────────

def predict_price(model: LinearRegression, surface_m2: float) -> float:
    """Prédit le prix (k€) pour une surface donnée (m²)."""
    return float(model.predict([[surface_m2]])[0])


# ─────────────────────────────────────────────────────────────────────────────
#  Figures
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG   = "#1a1a2e"
PANEL_BG  = "#0f0f1a"
COLOR_1   = "#a855f7"   # violet — données
COLOR_2   = "#22d3ee"   # cyan   — droite de régression
COLOR_3   = "#f97316"   # orange — prédiction
COLOR_4   = "#4ade80"   # vert   — test set


def make_dataset_figure(X: np.ndarray, y: np.ndarray) -> plt.Figure:
    """Nuage de points du dataset."""
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    ax.scatter(X.ravel(), y, color=COLOR_1, alpha=0.6, s=25, label="Données")
    ax.set_xlabel("Surface (m²)", color="white")
    ax.set_ylabel("Prix (k€)",     color="white")
    ax.set_title("Dataset : surface → prix",  color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    ax.legend(facecolor=PANEL_BG, labelcolor="white")
    fig.tight_layout()
    return fig


def make_regression_figure(
    model: LinearRegression,
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    highlight_x: float | None = None,
    highlight_y: float | None = None,
) -> plt.Figure:
    """Droite de régression + nuage + jeu test + prédiction mise en évidence."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(DARK_BG)

    # ── Subplot gauche : droite de régression ──
    ax = axes[0]
    ax.set_facecolor(PANEL_BG)

    ax.scatter(X.ravel(), y, color=COLOR_1, alpha=0.4, s=20, label="Toutes les données")
    ax.scatter(X_test.ravel(), y_test, color=COLOR_4, alpha=0.9, s=30, label="Jeu test")

    x_line = np.linspace(X.min(), X.max(), 200)
    y_line = model.predict(x_line.reshape(-1, 1))
    ax.plot(x_line, y_line, color=COLOR_2, linewidth=2, label="Droite de régression")

    if highlight_x is not None and highlight_y is not None:
        ax.scatter([highlight_x], [highlight_y], color=COLOR_3, s=120,
                   zorder=5, label=f"Prédiction ({highlight_x:.0f} m²)")
        ax.axvline(highlight_x, color=COLOR_3, linestyle="--", alpha=0.5)
        ax.axhline(highlight_y, color=COLOR_3, linestyle="--", alpha=0.5)

    ax.set_xlabel("Surface (m²)", color="white")
    ax.set_ylabel("Prix (k€)",     color="white")
    ax.set_title("Régression linéaire",  color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    ax.legend(facecolor=PANEL_BG, labelcolor="white", fontsize=8)

    # ── Subplot droit : résidus (test) ──
    ax2 = axes[1]
    ax2.set_facecolor(PANEL_BG)
    residuals = y_test - y_pred_test
    ax2.scatter(y_pred_test, residuals, color=COLOR_4, alpha=0.7, s=25)
    ax2.axhline(0, color=COLOR_2, linewidth=1.5, linestyle="--")
    ax2.set_xlabel("Prédiction (k€)", color="white")
    ax2.set_ylabel("Résidu (k€)",      color="white")
    ax2.set_title("Résidus — jeu test",       color="white")
    ax2.tick_params(colors="white")
    for sp in ax2.spines.values():
        sp.set_edgecolor("#333")

    fig.tight_layout()
    return fig
