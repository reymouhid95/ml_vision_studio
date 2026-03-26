"""
Approche DL — CNN entraîné from scratch sur Chats vs Chiens.

Architecture :
  Conv2D(32)  → BN → MaxPool
  Conv2D(64)  → BN → MaxPool
  Conv2D(128) → BN → MaxPool
  GlobalAveragePooling2D
  Dense(128, relu) → Dropout(0.5)
  Dense(2, softmax)

Augmentation on-the-fly : flip horizontal, rotation ±10°, zoom ±10%.
Early stopping sur val_loss (patience=5), sauvegarde du meilleur modèle.
"""
from __future__ import annotations

from pathlib import Path
from typing import Generator

import numpy as np

MODEL_DIR   = Path(__file__).parent / "models"
CNN_PATH    = MODEL_DIR / "cnn_model.keras"
CLASS_NAMES = ["chat", "chien"]


# ── Architecture ─────────────────────────────────────────────────────────────

def build_cnn(img_size: int = 96) -> "tf.keras.Model":
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = inputs
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Entraînement ─────────────────────────────────────────────────────────────

def train_dl_model(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    *,
    epochs:     int = 20,
    batch_size: int = 32,
) -> Generator:
    """
    Générateur — yield epoch par epoch, puis résultats finaux.

    Yields :
      ("epoch", ep, total, train_loss, train_acc, val_loss, val_acc)
      ("done",  results_dict)
    """
    import tensorflow as tf
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    img_size = X_train.shape[1]
    model    = build_cnn(img_size)

    # Augmentation (appliquée uniquement en train)
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ], name="augment")

    ds_train = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(2000, seed=0)
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

    loss_hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val_loss = float("inf")
    patience_left = 5

    for ep in range(1, epochs + 1):
        hist = model.fit(ds_train, validation_data=ds_val, epochs=1, verbose=0)
        tl = float(hist.history["loss"][0])
        ta = float(hist.history["accuracy"][0])
        vl = float(hist.history["val_loss"][0])
        va = float(hist.history["val_accuracy"][0])

        loss_hist["train_loss"].append(tl)
        loss_hist["train_acc"].append(ta)
        loss_hist["val_loss"].append(vl)
        loss_hist["val_acc"].append(va)

        yield ("epoch", ep, epochs, tl, ta, vl, va)

        # Sauvegarde du meilleur modèle + early stopping
        if vl < best_val_loss:
            best_val_loss = vl
            patience_left = 5
            model.save(CNN_PATH)
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    # Reload best weights
    model = tf.keras.models.load_model(CNN_PATH)

    # Évaluation finale sur test
    y_proba = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_pred  = y_proba.argmax(axis=1)

    test_acc = float(accuracy_score(y_test, y_pred))
    report   = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    cm       = confusion_matrix(y_test, y_pred).tolist()

    yield ("done", {
        "test_acc":  test_acc,
        "report":    report,
        "cm":        cm,
        "loss_hist": loss_hist,
        "preds":     y_pred.tolist(),
        "actuals":   y_test.tolist(),
    })


# ── Prédiction ────────────────────────────────────────────────────────────────

def predict_dl(img: np.ndarray) -> dict[str, float]:
    """
    img : (H, W, 3) float32 [0, 1].
    Retourne {class_name: probabilité}.
    """
    import tensorflow as tf

    if not CNN_PATH.exists():
        raise FileNotFoundError(f"Modèle DL non trouvé : {CNN_PATH}. Entraînez d'abord.")

    model    = tf.keras.models.load_model(CNN_PATH)
    img_size = model.input_shape[1]
    x        = tf.image.resize(img, [img_size, img_size])
    x        = tf.cast(x, tf.float32)[tf.newaxis]   # (1, H, W, 3)
    proba    = model.predict(x, verbose=0)[0]
    return {CLASS_NAMES[i]: float(proba[i]) for i in range(len(CLASS_NAMES))}


def model_trained() -> bool:
    return CNN_PATH.exists()
