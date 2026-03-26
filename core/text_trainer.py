from __future__ import annotations

import re
from typing import Generator

import numpy as np

_USE_MODEL = None   # singleton: loaded once, shared across sessions


def ensure_use():
    """Lazy-load the Universal Sentence Encoder (512-dim) exactly once."""
    global _USE_MODEL
    if _USE_MODEL is None:
        import tensorflow_hub as hub
        _USE_MODEL = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder/4"
        )
    return _USE_MODEL


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of strings → shape (N, 512) float32."""
    model = ensure_use()
    return model(texts).numpy().astype(np.float32)


def embed_single(text: str) -> np.ndarray:
    """Embed one string → shape (512,) float32."""
    return embed_texts([text])[0]


def split_text_into_chunks(raw: str) -> list[str]:
    """
    Split raw text into clean chunks suitable as training samples.
    Mirrors JS addTextSample: split on blank lines / newlines,
    filter len > 15, truncate each chunk to 800 chars.
    """
    parts = re.split(r"\n{2,}|\n", raw)
    chunks = []
    for p in parts:
        p = p.strip()
        if len(p) > 15:
            chunks.append(p[:800])
    return chunks


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, numerically stable."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def build_knn_index(
    text_classes: list[dict],
) -> tuple[list[dict], list[str]]:
    """
    Embed all samples in all classes and return a flat KNN index.

    Returns:
      knn_entries: [{classIdx, classId, embedding, text}]
      class_names: [str]
    """
    class_names = [c["name"] for c in text_classes]
    knn_entries: list[dict] = []

    for ci, cls in enumerate(text_classes):
        texts = [s for s in cls["samples"] if s.strip()]
        if not texts:
            continue
        embeddings = embed_texts(texts)
        for text, emb in zip(texts, embeddings):
            knn_entries.append({
                "classIdx": ci,
                "embedding": emb,
                "text": text,
            })

    return knn_entries, class_names


def classify_knn(
    query_emb: np.ndarray,
    knn_entries: list[dict],
    class_names: list[str],
    k: int = 5,
) -> dict[str, float]:
    """
    Weighted cosine-similarity KNN classification.
    Returns {class_name: probability} normalised to sum=1.
    """
    sims = [
        (cosine_similarity(query_emb, e["embedding"]), e["classIdx"])
        for e in knn_entries
    ]
    sims.sort(reverse=True)
    top_k = sims[:k]

    votes = np.zeros(len(class_names), dtype=np.float32)
    for sim, ci in top_k:
        w = max(sim, 0.0)
        votes[ci] += w

    total = votes.sum() + 1e-8
    probs = votes / total
    return {name: float(p) for name, p in zip(class_names, probs)}


def knn_leave_one_out(
    knn_entries: list[dict],
    class_names: list[str],
) -> tuple[list[int], list[int]]:
    """
    Leave-one-out evaluation of the KNN index.
    Returns (preds, actuals) as integer class indices.
    """
    preds, actuals = [], []
    for i, item in enumerate(knn_entries):
        rest = knn_entries[:i] + knn_entries[i+1:]
        scores = classify_knn(item["embedding"], rest, class_names)
        pred_class = max(scores, key=scores.get)
        preds.append(class_names.index(pred_class))
        actuals.append(item["classIdx"])
    return preds, actuals


def train_text_nn_model(
    knn_entries: list[dict],
    class_names: list[str],
) -> Generator:
    """
    Generator that fine-tunes a Dense head on the pre-computed USE embeddings.

    Yields:
      ("epoch", ep, total, loss, acc, val_acc)
      ("done",  model, loss_hist, class_names, preds, actuals)

    Architecture mirrors JS startTextNNTraining:
      Dense(512→256, relu) → BN → Dropout(0.3) → Dense(256→128, relu) → Dropout(0.2) → Dense(N, softmax)
    """
    import tensorflow as tf

    n_classes = len(class_names)
    xs = np.array([e["embedding"] for e in knn_entries], dtype=np.float32)
    ys_raw = [e["classIdx"] for e in knn_entries]
    ys = tf.keras.utils.to_categorical(ys_raw, n_classes)

    n_samples = len(xs)
    epochs = min(100, max(20, n_samples * 2))
    use_val = n_samples >= 8

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(512,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    loss_hist: list[float] = []

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
            xs, ys,
            epochs=1,
            batch_size=16,
            verbose=0,
            validation_split=0.15 if use_val else 0.0,
            shuffle=True,
            callbacks=[cb],
        )
        if updates:
            yield updates.pop()

    raw = model.predict(xs, verbose=0)
    preds   = raw.argmax(axis=1).tolist()
    actuals = ys_raw

    yield ("done", model, loss_hist, class_names, preds, actuals)


def classify_with_nn(
    model,
    query_emb: np.ndarray,
    class_names: list[str],
) -> dict[str, float]:
    """Return {class_name: probability} using the trained NN head."""
    arr = query_emb[np.newaxis].astype(np.float32)  # (1, 512)
    probs = model.predict(arr, verbose=0)[0]
    return {name: float(p) for name, p in zip(class_names, probs)}
