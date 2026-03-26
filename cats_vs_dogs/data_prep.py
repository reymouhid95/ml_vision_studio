"""
Préparation des données Chats vs Chiens.

Téléchargement via tensorflow_datasets (premier lancement : ~786 MB).
Sauvegarde des splits numpy dans cats_vs_dogs/data/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DATA_DIR    = Path(__file__).parent / "data"
IMG_SIZE    = 96       # résolution commune ML + DL
CLASS_NAMES = ["chat", "chien"]   # label 0 = cat, 1 = dog dans cats_vs_dogs

# Taille des splits par défaut
N_TRAIN = 1400
N_VAL   = 300
N_TEST  = 300


def _npz(split: str) -> Path:
    return DATA_DIR / f"{split}.npz"


def is_prepared() -> bool:
    return all(_npz(s).exists() for s in ("train", "val", "test"))


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (images, labels) depuis le fichier npz sauvegardé."""
    d = np.load(_npz(split))
    return d["images"].astype(np.float32), d["labels"].astype(np.int32)


def split_counts() -> dict[str, int]:
    """Nombre d'exemples par split (sans charger les images)."""
    if not is_prepared():
        return {}
    counts = {}
    for s in ("train", "val", "test"):
        d = np.load(_npz(s))
        counts[s] = int(d["labels"].shape[0])
    return counts


def download_and_prepare(
    n_train: int = N_TRAIN,
    n_val:   int = N_VAL,
    n_test:  int = N_TEST,
) -> str:
    """
    Télécharge cats_vs_dogs via tensorflow_datasets et sauvegarde les splits.
    Premier lancement : télécharge ~786 MB dans ~/tensorflow_datasets/.
    Retourne un message de statut.
    """
    import tensorflow as tf
    import tensorflow_datasets as tfds

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    total = n_train + n_val + n_test

    ds = tfds.load(
        "cats_vs_dogs",
        split=f"train[:{total}]",
        as_supervised=True,
        shuffle_files=True,
    )

    imgs, lbls = [], []
    for img, lbl in ds:
        img_r = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        imgs.append(tf.cast(img_r, tf.float32).numpy() / 255.0)
        lbls.append(int(lbl.numpy()))

    imgs = np.array(imgs, dtype=np.float32)
    lbls = np.array(lbls, dtype=np.int32)

    # Mélange reproductible
    rng  = np.random.default_rng(42)
    perm = rng.permutation(len(imgs))
    imgs, lbls = imgs[perm], lbls[perm]

    splits = {
        "train": (imgs[:n_train],                 lbls[:n_train]),
        "val":   (imgs[n_train:n_train + n_val],  lbls[n_train:n_train + n_val]),
        "test":  (imgs[n_train + n_val:],          lbls[n_train + n_val:]),
    }
    for name, (X, y) in splits.items():
        np.savez_compressed(_npz(name), images=X, labels=y)

    cats = int((lbls == 0).sum())
    dogs = int((lbls == 1).sum())
    return (
        f"✓ Dataset prêt — {n_train} train / {n_val} val / {n_test} test\n"
        f"  {cats} chats · {dogs} chiens · images {IMG_SIZE}×{IMG_SIZE}\n"
        f"  Sauvegardé dans {DATA_DIR}"
    )
