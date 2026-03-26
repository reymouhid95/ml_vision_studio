from __future__ import annotations

from typing import Generator

import numpy as np

HP_LR_OPTS    = [0.0001, 0.0005, 0.001, 0.005, 0.01]
HP_BATCH_OPTS = [4, 8, 16, 32, 64]


def extract_mel_features(audio_path: str, sr: int = 22050) -> np.ndarray:
    """
    Load an audio clip and return a 40-dim Mel feature vector (float32).

    Steps (mirrors the JS computeMelFeatures logic):
      1. Load with librosa at 22 050 Hz mono, max 3 s
      2. Compute mel spectrogram  (n_mels=40, fmin=80, fmax=8000)
      3. Log-compress: log(S + 1e-8)
      4. Average over time axis → shape (40,)
      5. Min-max normalize to [0, 1]
    """
    import librosa

    y, _ = librosa.load(audio_path, sr=sr, mono=True, duration=3.0)
    if len(y) == 0:
        return np.zeros(40, dtype=np.float32)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmin=80, fmax=8000)
    log_S = np.log(S + 1e-8)                   # log compress
    feat = log_S.mean(axis=1).astype(np.float32)  # average over time

    mn, mx = feat.min(), feat.max()
    feat = (feat - mn) / (mx - mn + 1e-8)
    return feat


def train_audio_model(
    classes: list[dict],
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    hidden_units: int,
) -> Generator:
    """
    Generator that trains a Dense Mel-feature classifier.

    Each iteration yields one of:
      ("epoch", ep, total, loss, acc, val_acc)
      ("done",  model, loss_hist, class_names, preds, actuals)

    classes: [{name: str, samples: [np.ndarray shape (40,)]}]
    """
    import tensorflow as tf

    n_classes = len(classes)
    class_names = [c["name"] for c in classes]
    h = max(32, hidden_units)
    h2 = max(16, h // 2)

    # Build dataset
    xs, ys = [], []
    for ci, cls in enumerate(classes):
        for feat in cls["samples"]:
            xs.append(feat)
            ys.append(ci)

    xs = np.array(xs, dtype=np.float32)
    ys_oh = tf.keras.utils.to_categorical(ys, n_classes)

    # Architecture matching JS: Dense(H,relu) → BN → Dropout(0.3) → Dense(H/2,relu) → Dropout(0.2) → Dense(N,softmax)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(40,)),
        tf.keras.layers.Dense(h, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(h2, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    loss_hist: list[float] = []
    use_val = len(xs) >= 8 and n_classes >= 2

    class _YieldCallback(tf.keras.callbacks.Callback):
        def __init__(self, buf):
            super().__init__()
            self._buf = buf

        def on_epoch_end(self, epoch, logs=None):
            l  = float(logs.get("loss", 0))
            a  = float(logs.get("accuracy", 0))
            va = logs.get("val_accuracy")
            va = float(va) if va is not None else None
            loss_hist.append(l)
            self._buf.append(("epoch", epoch + 1, epochs, l, a, va))

    updates: list = []
    cb = _YieldCallback(updates)

    for ep in range(epochs):
        model.fit(
            xs, ys_oh,
            epochs=1,
            batch_size=batch_size,
            verbose=0,
            validation_split=0.15 if use_val else 0.0,
            shuffle=True,
            callbacks=[cb],
        )
        if updates:
            yield updates.pop()

    raw = model.predict(xs, verbose=0)
    preds   = raw.argmax(axis=1).tolist()
    actuals = [int(y) for y in ys]

    yield ("done", model, loss_hist, class_names, preds, actuals)


def predict_audio(
    model,
    audio_path: str,
    class_names: list[str],
) -> dict[str, float]:
    """Return {class_name: probability} for gr.Label."""
    feat = extract_mel_features(audio_path)
    arr = feat[np.newaxis]                          # (1, 40)
    probs = model.predict(arr, verbose=0)[0]
    return {name: float(p) for name, p in zip(class_names, probs)}
