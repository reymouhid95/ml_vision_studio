"""Reconnaissance de chiffres manuscrits — CNN sur MNIST."""

from __future__ import annotations

from typing import Generator

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

DARK_BG  = "#1a1a2e"
PANEL_BG = "#0f0f1a"
PALETTE  = ["#a855f7", "#22d3ee", "#f97316", "#4ade80",
            "#f43f5e", "#facc15", "#60a5fa", "#e879f9",
            "#34d399", "#fb923c"]


# ─────────────────────────────────────────────────────────────────────────────
#  Données
# ─────────────────────────────────────────────────────────────────────────────

def load_mnist():
    """
    Charge MNIST depuis tf.keras.

    Returns
    -------
    (X_train, y_train), (X_test, y_test)
      X : float32, shape (N, 28, 28, 1), normalisé [0, 1]
      y : int, shape (N,), valeurs 0–9
    """
    import tensorflow as tf
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype("float32")[..., np.newaxis] / 255.0
    X_test  = X_test .astype("float32")[..., np.newaxis] / 255.0
    return (X_train, y_train), (X_test, y_test)


# ─────────────────────────────────────────────────────────────────────────────
#  Modèle CNN
# ─────────────────────────────────────────────────────────────────────────────

def build_cnn(lr: float = 0.001):
    """
    CNN classique pour MNIST.

    Architecture :
      Conv2D(32, 3×3, relu) → Conv2D(64, 3×3, relu) → MaxPool(2×2)
      → Dropout(0.25) → Flatten
      → Dense(128, relu) → Dropout(0.4) → Dense(10, softmax)
    """
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Entraînement — generator
# ─────────────────────────────────────────────────────────────────────────────

def train_cnn_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    *,
    epochs:     int   = 10,
    batch_size: int   = 128,
    lr:         float = 0.001,
) -> Generator:
    """
    Generator qui entraîne le CNN époque par époque.

    Yields
    ------
    ("epoch", ep, total, loss, acc, val_loss, val_acc)
    ("done",  model, history_dict)
    """
    import tensorflow as tf

    model = build_cnn(lr)

    hist_loss, hist_acc, hist_val_loss, hist_val_acc = [], [], [], []

    class _YieldCallback(tf.keras.callbacks.Callback):
        def __init__(self, updates_list):
            super().__init__()
            self._updates = updates_list

        def on_epoch_end(self, epoch, logs=None):
            loss     = float(logs.get("loss",     0))
            acc      = float(logs.get("accuracy", 0))
            val_loss = float(logs.get("val_loss", 0))
            val_acc  = float(logs.get("val_accuracy", 0))
            hist_loss.append(loss);     hist_acc.append(acc)
            hist_val_loss.append(val_loss); hist_val_acc.append(val_acc)
            self._updates.append(
                ("epoch", epoch + 1, epochs, loss, acc, val_loss, val_acc)
            )

    updates: list = []
    cb = _YieldCallback(updates)

    for ep in range(epochs):
        model.fit(
            X_train, y_train,
            epochs=1,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=0,
            callbacks=[cb],
            shuffle=True,
        )
        if updates:
            yield updates.pop()

    history = {
        "loss":     hist_loss,
        "accuracy": hist_acc,
        "val_loss": hist_val_loss,
        "val_accuracy": hist_val_acc,
    }
    yield ("done", model, history)


# ─────────────────────────────────────────────────────────────────────────────
#  Évaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_cnn(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Évalue le modèle sur le jeu de test."""
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    probs  = model.predict(X_test, verbose=0)
    preds  = probs.argmax(axis=1)
    return {
        "test_loss": float(loss),
        "test_acc":  float(acc),
        "preds":     preds.tolist(),
        "actuals":   y_test.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Prédiction sur image externe
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_digit_image(img) -> np.ndarray:
    """
    Convertit une image PIL ou dict (Sketchpad) en tableau 28×28×1.

    - Gère le format dict renvoyé par gr.Sketchpad (clé "composite")
    - Force grayscale, fond noir, chiffre blanc (style MNIST)
    - Retourne float32 shape (1, 28, 28, 1) normalisé [0, 1]
    """
    if isinstance(img, dict):
        # gr.Sketchpad renvoie {"background": ..., "layers": [...], "composite": ...}
        src = img.get("composite") or img.get("layers", [None])[0] or img.get("background")
        if src is None:
            return None
        img = src

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype("uint8"))

    img = img.convert("L")                           # niveaux de gris
    arr = np.array(img, dtype="float32")

    # Si fond clair (dessin sur blanc), inverser
    if arr.mean() > 127:
        arr = 255.0 - arr

    # Normalise + redimensionne
    arr = arr / 255.0
    pil = Image.fromarray((arr * 255).astype("uint8"))
    pil = pil.resize((28, 28), Image.LANCZOS)
    arr = np.array(pil, dtype="float32") / 255.0
    return arr[np.newaxis, ..., np.newaxis]           # (1, 28, 28, 1)


def predict_digit(model, img) -> tuple[int, dict[str, float]] | None:
    """
    Retourne (classe prédite, dict {chiffre: probabilité}).

    img peut être une PIL.Image, un np.ndarray ou un dict Sketchpad.
    """
    arr = preprocess_digit_image(img)
    if arr is None:
        return None
    probs = model(arr, training=False).numpy()[0]
    pred  = int(probs.argmax())
    return pred, {str(i): float(probs[i]) for i in range(10)}


# ─────────────────────────────────────────────────────────────────────────────
#  Figures
# ─────────────────────────────────────────────────────────────────────────────

def make_sample_grid(X: np.ndarray, y: np.ndarray, n: int = 20) -> plt.Figure:
    """Grille de n images MNIST avec leur label."""
    n = min(n, len(X))
    cols = 10
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.9, rows * 1.0))
    fig.patch.set_facecolor(DARK_BG)

    for i, ax in enumerate(axes.ravel()):
        ax.set_facecolor(DARK_BG)
        ax.axis("off")
        if i < n:
            idx = i * (len(X) // n)
            ax.imshow(X[idx, ..., 0], cmap="gray", vmin=0, vmax=1)
            ax.set_title(str(y[idx]), color="white", fontsize=9, pad=2)

    fig.suptitle(f"Exemples MNIST ({len(X):,} images disponibles)",
                 color="white", fontsize=11, y=1.01)
    fig.tight_layout(pad=0.2)
    return fig


def make_training_curves(history: dict) -> plt.Figure:
    """Loss + Accuracy (train et val) en deux subplots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
    fig.patch.set_facecolor(DARK_BG)
    eps = range(1, len(history["loss"]) + 1)

    for ax, key_tr, key_val, title, ylabel in [
        (ax1, "loss",     "val_loss",     "Perte",     "Perte (cross-entropy)"),
        (ax2, "accuracy", "val_accuracy", "Précision", "Précision"),
    ]:
        ax.set_facecolor(PANEL_BG)
        ax.plot(eps, history[key_tr],  color=PALETTE[0], linewidth=2, label="Train")
        ax.plot(eps, history[key_val], color=PALETTE[1], linewidth=2, label="Test",
                linestyle="--")
        ax.set_title(title,  color="white")
        ax.set_xlabel("Époques", color="white")
        ax.set_ylabel(ylabel,    color="white")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")
        ax.legend(facecolor=PANEL_BG, labelcolor="white")

    fig.tight_layout()
    return fig


def make_confusion_10(preds, actuals) -> plt.Figure:
    """Matrice de confusion 10×10 pour les chiffres 0–9."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(actuals, preds, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    im = ax.imshow(cm, cmap="Purples")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(10)); ax.set_xticklabels(range(10), color="white")
    ax.set_yticks(range(10)); ax.set_yticklabels(range(10), color="white")
    ax.set_xlabel("Prédit",  color="white")
    ax.set_ylabel("Réel",    color="white")
    ax.set_title("Matrice de confusion (jeu test)", color="white")

    for i in range(10):
        for j in range(10):
            val = cm[i, j]
            color = "black" if val > cm.max() * 0.5 else "white"
            ax.text(j, i, str(val), ha="center", va="center",
                    color=color, fontsize=8)

    fig.tight_layout()
    return fig
