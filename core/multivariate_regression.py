"""
Régression multivariée — California Housing (sklearn, aucun téléchargement requis).

Dataset : 20 640 quartiers californiens (1990).
Cible   : valeur médiane des maisons (convertie en k$).
Features: 8 variables numériques (revenu, âge, pièces, population…).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ─── Métadonnées des features ─────────────────────────────────────────────────

ALL_FEATURE_NAMES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]

FEATURE_LABELS_FR = {
    "MedInc":      "Revenu médian (×10 k$)",
    "HouseAge":    "Âge médian logement",
    "AveRooms":    "Nb pièces moyen",
    "AveBedrms":   "Nb chambres moyen",
    "Population":  "Population quartier",
    "AveOccup":    "Occupants moyens",
    "Latitude":    "Latitude",
    "Longitude":   "Longitude",
}

FEATURE_DESCRIPTIONS = {
    "MedInc":      "Revenu médian des ménages (en dizaines de milliers de $)",
    "HouseAge":    "Âge médian des logements du quartier",
    "AveRooms":    "Nombre moyen de pièces par logement",
    "AveBedrms":   "Nombre moyen de chambres par logement",
    "Population":  "Population totale du quartier",
    "AveOccup":    "Nombre moyen d'occupants par logement",
    "Latitude":    "Latitude géographique",
    "Longitude":   "Longitude géographique",
}

# ─── Style ───────────────────────────────────────────────────────────────────

DARK_BG  = "#1a1a2e"
PANEL_BG = "#0f0f1a"


def _ax_style(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors="white")
    ax.grid(alpha=0.2, color="#444")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")


# ─── Données ─────────────────────────────────────────────────────────────────

def load_california() -> tuple[np.ndarray, np.ndarray]:
    """
    Charge California Housing depuis sklearn (pas de téléchargement réseau).

    Returns
    -------
    X : (20640, 8) — features dans l'ordre ALL_FEATURE_NAMES
    y : (20640,)   — prix médian en k$ (×100 depuis la valeur sklearn en 100k$)
    """
    data = fetch_california_housing()
    X = data.data
    y = data.target * 100.0   # convertit en k$
    return X, y


# ─── Entraînement ────────────────────────────────────────────────────────────

def train_multivariate_model(
    feature_selection: list[str],
    model_type: str = "linear",
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple:
    """
    Entraîne une régression sur les features sélectionnées.

    Parameters
    ----------
    feature_selection : liste de noms parmi ALL_FEATURE_NAMES
    model_type        : "linear" ou "ridge"

    Returns
    -------
    pipe, metrics, X_all, y_all, X_test_raw, y_test, y_pred
    """
    X_all, y_all = load_california()
    col_idx = [ALL_FEATURE_NAMES.index(f) for f in feature_selection
               if f in ALL_FEATURE_NAMES]
    if not col_idx:
        col_idx = list(range(len(ALL_FEATURE_NAMES)))
    selected = [ALL_FEATURE_NAMES[i] for i in col_idx]

    X = X_all[:, col_idx]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_all, test_size=test_size, random_state=seed
    )

    reg = Ridge(alpha=1.0) if model_type == "ridge" else LinearRegression()
    pipe = Pipeline([("scaler", StandardScaler()), ("model", reg)])
    pipe.fit(X_train, y_train)

    y_pred_train = pipe.predict(X_train)
    y_pred_test  = pipe.predict(X_test)

    metrics = {
        "R²_train":   float(r2_score(y_train, y_pred_train)),
        "R²_test":    float(r2_score(y_test,  y_pred_test)),
        "RMSE_train": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        "RMSE_test":  float(np.sqrt(mean_squared_error(y_test,  y_pred_test))),
        "MAE_train":  float(mean_absolute_error(y_train, y_pred_train)),
        "MAE_test":   float(mean_absolute_error(y_test,  y_pred_test)),
        "n_train":    int(len(X_train)),
        "n_test":     int(len(X_test)),
        "features":   selected,
        "coefs":      pipe.named_steps["model"].coef_.tolist(),
        "model_type": model_type,
    }
    return pipe, metrics, X_all, y_all, X_test, y_test, y_pred_test


def predict_multivariate(pipe: Pipeline, feature_values: dict, selected_features: list[str]) -> float:
    """Prédit le prix (k$) à partir d'un dict feature→valeur."""
    x = np.array([[float(feature_values.get(f, 0.0)) for f in selected_features]])
    return float(pipe.predict(x)[0])


# ─── Figures ─────────────────────────────────────────────────────────────────

def make_importance_figure(coefs: list[float], names: list[str]) -> plt.Figure:
    """Diagramme en barres horizontales des coefficients normalisés."""
    coefs_arr = np.array(coefs)
    labels_fr = [FEATURE_LABELS_FR.get(n, n) for n in names]
    sorted_idx = np.argsort(np.abs(coefs_arr))
    colors = ["#4ade80" if c > 0 else "#f43f5e" for c in coefs_arr[sorted_idx]]

    height = max(3.0, 0.55 * len(names) + 1.0)
    fig, ax = plt.subplots(figsize=(8, height))
    fig.patch.set_facecolor(DARK_BG)
    _ax_style(ax)

    ax.barh(
        [labels_fr[i] for i in sorted_idx],
        coefs_arr[sorted_idx],
        color=colors, alpha=0.88, edgecolor="#333", linewidth=0.5,
    )
    ax.axvline(0, color="#888", linewidth=1)
    ax.set_xlabel("Coefficient (après normalisation)", color="white", fontsize=9)
    ax.set_title("Importance des features", color="white", fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([labels_fr[i] for i in sorted_idx], color="white", fontsize=9)

    legend_handles = [
        Patch(facecolor="#4ade80", label="Positif → ↑ prix"),
        Patch(facecolor="#f43f5e", label="Négatif → ↓ prix"),
    ]
    ax.legend(handles=legend_handles, facecolor=PANEL_BG, labelcolor="white", fontsize=8)
    fig.tight_layout()
    return fig


def make_scatter_residuals_figure(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    r2: float,
) -> plt.Figure:
    """Prédit vs Réel + Graphe des résidus."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor(DARK_BG)
    _ax_style(ax1)
    _ax_style(ax2)

    # Prédit vs Réel
    ax1.scatter(y_test, y_pred, alpha=0.12, s=5, color="#a855f7", rasterized=True)
    lims = [
        min(float(y_test.min()), float(y_pred.min())),
        max(float(y_test.max()), float(y_pred.max())),
    ]
    ax1.plot(lims, lims, "r--", linewidth=1.5, alpha=0.9, label="Parfait (R²=1)")
    ax1.set_xlabel("Prix réel (k$)", color="white", fontsize=9)
    ax1.set_ylabel("Prix prédit (k$)", color="white", fontsize=9)
    ax1.set_title(f"Prédit vs Réel — R² = {r2:.3f}", color="white", fontsize=11, fontweight="bold")
    ax1.legend(facecolor=PANEL_BG, labelcolor="white", fontsize=8)

    # Résidus
    residuals = y_pred - y_test
    ax2.scatter(y_pred, residuals, alpha=0.12, s=5, color="#22d3ee", rasterized=True)
    ax2.axhline(0, color="#f97316", linewidth=1.5)
    ax2.set_xlabel("Prix prédit (k$)", color="white", fontsize=9)
    ax2.set_ylabel("Résidu (prédit − réel)", color="white", fontsize=9)
    ax2.set_title("Graphe des résidus", color="white", fontsize=11, fontweight="bold")

    fig.tight_layout(pad=1.5)
    return fig
