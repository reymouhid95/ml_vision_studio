from __future__ import annotations

from typing import Generator

import numpy as np
from PIL import Image

HP_LR_OPTS    = [0.0001, 0.0005, 0.001, 0.005, 0.01]
HP_BATCH_OPTS = [4, 8, 16, 32, 64]


def _pil_to_array(img: Image.Image) -> np.ndarray:
    """Resize to 224×224, normalize to [0, 1], return shape (224, 224, 3)."""
    img = img.resize((224, 224)).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def train_image_model(
    classes: list[dict],
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    dense_units: int,
) -> Generator:
    """
    Generator that trains a MobileNetV2 transfer-learning model.

    Each iteration yields one of:
      ("epoch", ep, total, loss, acc)
      ("done",  model, loss_hist, class_names, preds, actuals)

    classes: [{name: str, samples: [PIL.Image]}]
    """
    import tensorflow as tf
    from utils.augmentation import augment_image

    n_classes = len(classes)
    class_names = [c["name"] for c in classes]

    # Build dataset with augmentation (original + flip + brighten = 3×)
    xs, ys = [], []
    for ci, cls in enumerate(classes):
        for img in cls["samples"]:
            img_224 = img.resize((224, 224)).convert("RGB")
            all_imgs = [img_224] + augment_image(img_224)
            for aug in all_imgs:
                xs.append(_pil_to_array(aug))
                ys.append(ci)

    xs = np.array(xs, dtype=np.float32)
    ys = tf.keras.utils.to_categorical(ys, n_classes)

    # Build model
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(base.input, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    loss_hist: list[float] = []

    class _YieldCallback(tf.keras.callbacks.Callback):
        def __init__(self, gen_list):
            super().__init__()
            self._updates = gen_list

        def on_epoch_end(self, epoch, logs=None):
            l = float(logs.get("loss", 0))
            a = float(logs.get("accuracy", 0))
            loss_hist.append(l)
            self._updates.append(("epoch", epoch + 1, epochs, l, a))

    updates: list = []
    cb = _YieldCallback(updates)

    # Run training epoch by epoch to yield updates
    for ep in range(epochs):
        model.fit(
            xs, ys,
            epochs=1,
            batch_size=batch_size,
            verbose=0,
            callbacks=[cb],
            shuffle=True,
        )
        if updates:
            yield updates.pop()

    # Evaluate on training set for confusion matrix
    raw = model.predict(xs, verbose=0)
    preds   = raw.argmax(axis=1).tolist()
    actuals = ys.argmax(axis=1).tolist()

    yield ("done", model, loss_hist, class_names, preds, actuals)


def predict_image(
    model,
    image: Image.Image,
    class_names: list[str],
) -> dict[str, float]:
    """Return {class_name: probability} dict suitable for gr.Label."""
    arr = _pil_to_array(image)[np.newaxis]          # (1, 224, 224, 3)
    probs = model.predict(arr, verbose=0)[0]
    return {name: float(p) for name, p in zip(class_names, probs)}
