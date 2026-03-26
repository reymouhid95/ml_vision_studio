"""
Approche ML classique — Chats vs Chiens.

Pipeline de features :
  - HOG (Histogram of Oriented Gradients) → forme / contours
  - Histogramme couleur (3 canaux × 32 bins) → couleur

Classifieurs comparés :
  - SVM (noyau RBF, C=10)
  - Random Forest (200 arbres)

Les deux modèles sont encapsulés dans un sklearn Pipeline (StandardScaler inclus).
"""
from __future__ import annotations

from pathlib import Path
from typing import Generator

import joblib
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_DIR   = Path(__file__).parent / "models"
SVM_PATH    = MODEL_DIR / "svm_model.joblib"
RF_PATH     = MODEL_DIR / "rf_model.joblib"
CLASS_NAMES = ["chat", "chien"]


# ── Extraction de features ───────────────────────────────────────────────────

def _hog_features(img: np.ndarray) -> np.ndarray:
    """HOG depuis une image (H, W, 3) float32 [0,1]."""
    gray = rgb2gray(img)
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )


def _color_hist(img: np.ndarray, bins: int = 32) -> np.ndarray:
    """Histogramme par canal RGB (3 × bins features)."""
    hists = [
        np.histogram(img[:, :, c], bins=bins, range=(0.0, 1.0))[0].astype(np.float32)
        for c in range(3)
    ]
    return np.concatenate(hists)


def extract_features(images: np.ndarray) -> np.ndarray:
    """
    images : (N, H, W, 3) float32.
    Retourne (N, n_features) float32 — HOG + histogramme couleur.
    """
    feats = []
    for img in images:
        feats.append(np.concatenate([_hog_features(img), _color_hist(img)]))
    return np.array(feats, dtype=np.float32)


# ── Entraînement ─────────────────────────────────────────────────────────────

def train_ml_models(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
) -> Generator:
    """
    Générateur — yield des messages de progression, puis un dict final.

    Séquence :
      yield str  → message de log
      yield dict → résultats finaux (toujours en dernier)
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Features ──────────────────────────────────────────────────────────────
    yield "Extraction des features HOG + histogramme couleur (train)…"
    X_tr = extract_features(X_train)
    yield f"  → {X_tr.shape[0]} exemples, {X_tr.shape[1]} features"

    yield "Extraction features val + test…"
    X_v  = extract_features(X_val)
    X_te = extract_features(X_test)

    results = {}

    # ── SVM ───────────────────────────────────────────────────────────────────
    yield "Entraînement SVM (RBF, C=10)…"
    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=0)),
    ])
    svm_pipe.fit(X_tr, y_train)
    joblib.dump(svm_pipe, SVM_PATH)

    svm_val_acc  = accuracy_score(y_val,  svm_pipe.predict(X_v))
    svm_pred     = svm_pipe.predict(X_te)
    svm_test_acc = accuracy_score(y_test, svm_pred)
    svm_report   = classification_report(y_test, svm_pred, target_names=CLASS_NAMES)
    svm_cm       = confusion_matrix(y_test, svm_pred).tolist()

    yield f"SVM — val : {svm_val_acc:.3f}  /  test : {svm_test_acc:.3f}"
    results["svm"] = {
        "val_acc":  float(svm_val_acc),
        "test_acc": float(svm_test_acc),
        "report":   svm_report,
        "cm":       svm_cm,
        "preds":    svm_pred.tolist(),
        "actuals":  y_test.tolist(),
    }

    # ── Random Forest ─────────────────────────────────────────────────────────
    yield "Entraînement Random Forest (200 arbres)…"
    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)),
    ])
    rf_pipe.fit(X_tr, y_train)
    joblib.dump(rf_pipe, RF_PATH)

    rf_val_acc  = accuracy_score(y_val,  rf_pipe.predict(X_v))
    rf_pred     = rf_pipe.predict(X_te)
    rf_test_acc = accuracy_score(y_test, rf_pred)
    rf_report   = classification_report(y_test, rf_pred, target_names=CLASS_NAMES)
    rf_cm       = confusion_matrix(y_test, rf_pred).tolist()

    yield f"RF  — val : {rf_val_acc:.3f}  /  test : {rf_test_acc:.3f}"
    results["rf"] = {
        "val_acc":  float(rf_val_acc),
        "test_acc": float(rf_test_acc),
        "report":   rf_report,
        "cm":       rf_cm,
        "preds":    rf_pred.tolist(),
        "actuals":  y_test.tolist(),
    }

    yield "✓ Entraînement ML terminé."
    yield results   # toujours en dernier


# ── Prédiction ────────────────────────────────────────────────────────────────

def predict_ml(img: np.ndarray, model: str = "svm") -> dict[str, float]:
    """
    img    : (H, W, 3) float32 [0, 1].
    model  : "svm" ou "rf".
    Retourne {class_name: probabilité}.
    """
    path = SVM_PATH if model == "svm" else RF_PATH
    if not path.exists():
        raise FileNotFoundError(f"Modèle ML non trouvé : {path}. Entraînez d'abord.")
    pipe  = joblib.load(path)
    feat  = extract_features(img[np.newaxis])   # (1, n_features)
    proba = pipe.predict_proba(feat)[0]
    return {CLASS_NAMES[i]: float(proba[i]) for i in range(len(CLASS_NAMES))}


def models_trained() -> dict[str, bool]:
    return {"svm": SVM_PATH.exists(), "rf": RF_PATH.exists()}
