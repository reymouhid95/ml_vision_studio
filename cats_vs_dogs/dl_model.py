"""
Approche DL — Transfer learning MobileNetV2 sur Chats vs Chiens.

Deux phases d'entraînement :
  Phase 1 — Base gelée (10 époques, lr=1e-3)
    Seule la tête de classification est entraînée.
    Rapide et stable : la base ImageNet est déjà excellente.

  Phase 2 — Fine-tuning (lr=1e-5, époques configurables)
    Les 30 dernières couches de MobileNetV2 sont dégelées.
    Le très faible taux d'apprentissage préserve les poids pré-entraînés.

Le preprocessing (normalisation [0,1] → [-1,1]) est intégré dans le modèle
via mobilenet_v2.preprocess_input, donc predict_dl reçoit simplement float32 [0,1].
"""
from __future__ import annotations

from pathlib import Path
from typing import Generator

import numpy as np

MODEL_DIR   = Path(__file__).parent / "models"
CNN_PATH    = MODEL_DIR / "cnn_model.keras"
CLASS_NAMES = ["chat", "chien"]

PHASE1_EPOCHS = 10   # tête seule, base gelée
PATIENCE      = 5    # early stopping


# ── Architecture ─────────────────────────────────────────────────────────────

def build_transfer_model(img_size: int = 96):
    """
    MobileNetV2 (ImageNet) + tête de classification.
    Retourne (model, base_model) pour pouvoir dégeler la base en phase 2.
    Les images entrantes sont en float32 [0, 1] ; le preprocessing est intégré.
    """
    import tensorflow as tf

    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False   # Phase 1 : tout gelé

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    # [0,1] → [-1,1] (équivalent à mobilenet_v2.preprocess_input)
    # Rescaling est une couche native : pas de problème de sérialisation.
    x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base


# ── Entraînement ─────────────────────────────────────────────────────────────

def train_dl_model(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    *,
    finetune_epochs: int = 15,
    batch_size:      int = 32,
) -> Generator:
    """
    Générateur — deux phases de training, puis résultats finaux.

    Yields :
      ("phase", num, phase1_total, phase2_total)   — début de phase
      ("epoch", ep, total_epochs, tl, ta, vl, va)  — fin d'époque
      ("done",  results_dict)                       — résultats finaux
    """
    import tensorflow as tf
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    img_size      = X_train.shape[1]
    total_epochs  = PHASE1_EPOCHS + finetune_epochs
    model, base   = build_transfer_model(img_size)
    loss_hist     = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.1),
    ], name="augment")

    ds_train = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(len(X_train), seed=0)
        .batch(batch_size)
        .map(lambda x, y: (augment(x, training=True), y),
             num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    ds_val = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    def _run_phase(phase_epochs: int, ep_offset: int, lr: float):
        """Entraîne `phase_epochs` époques, sauvegarde le meilleur modèle."""
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        best_val = float("inf")
        patience_left = PATIENCE

        for ep in range(1, phase_epochs + 1):
            hist = model.fit(ds_train, validation_data=ds_val, epochs=1, verbose=0)
            tl = float(hist.history["loss"][0])
            ta = float(hist.history["accuracy"][0])
            vl = float(hist.history["val_loss"][0])
            va = float(hist.history["val_accuracy"][0])

            loss_hist["train_loss"].append(tl)
            loss_hist["train_acc"].append(ta)
            loss_hist["val_loss"].append(vl)
            loss_hist["val_acc"].append(va)

            yield ("epoch", ep_offset + ep, total_epochs, tl, ta, vl, va)

            if vl < best_val:
                best_val = vl
                patience_left = PATIENCE
                model.save(CNN_PATH)
            else:
                patience_left -= 1
                if patience_left == 0:
                    break

    # ── Phase 1 : tête seule ──────────────────────────────────────────────────
    yield ("phase", 1, PHASE1_EPOCHS, finetune_epochs)
    yield from _run_phase(PHASE1_EPOCHS, ep_offset=0, lr=1e-3)

    # ── Phase 2 : fine-tuning des 30 dernières couches ────────────────────────
    yield ("phase", 2, PHASE1_EPOCHS, finetune_epochs)
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    yield from _run_phase(finetune_epochs, ep_offset=PHASE1_EPOCHS, lr=1e-5)

    # ── Évaluation finale ─────────────────────────────────────────────────────
    model = tf.keras.models.load_model(CNN_PATH)
    y_proba = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_pred  = y_proba.argmax(axis=1)

    yield ("done", {
        "test_acc":  float(accuracy_score(y_test, y_pred)),
        "report":    classification_report(y_test, y_pred, target_names=CLASS_NAMES),
        "cm":        confusion_matrix(y_test, y_pred).tolist(),
        "loss_hist": loss_hist,
        "preds":     y_pred.tolist(),
        "actuals":   y_test.tolist(),
    })


# ── Prédiction ────────────────────────────────────────────────────────────────

def predict_dl(img: np.ndarray) -> dict[str, float]:
    """
    img : (H, W, 3) float32 [0, 1].
    Le preprocessing est géré en interne par le modèle.
    """
    import tensorflow as tf

    if not CNN_PATH.exists():
        raise FileNotFoundError(f"Modèle DL non trouvé : {CNN_PATH}. Entraînez d'abord.")

    model    = tf.keras.models.load_model(CNN_PATH)
    img_size = model.input_shape[1]
    x        = tf.image.resize(img, [img_size, img_size])
    x        = tf.cast(x, tf.float32)[tf.newaxis]
    proba    = model.predict(x, verbose=0)[0]
    return {CLASS_NAMES[i]: float(proba[i]) for i in range(len(CLASS_NAMES))}


def model_trained() -> bool:
    return CNN_PATH.exists()
