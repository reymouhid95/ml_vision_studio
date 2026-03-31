"""
Préparation des données Chats vs Chiens.

Téléchargement via tensorflow_datasets (premier lancement : ~786 MB).
Sauvegarde des splits numpy dans cats_vs_dogs/data/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DATA_DIR    = Path(__file__).parent / "data"
IMG_SIZE    = 160      # résolution commune ML + DL (EfficientNetB0 natif 224, fonctionne à 160)
CLASS_NAMES = ["chat", "chien"]   # label 0 = cat, 1 = dog dans cats_vs_dogs

# Taille des splits par défaut
# Le dataset complet a ~23 000 images ; 7 000 offre un bon compromis vitesse/précision.
N_TRAIN = 8000
N_VAL   = 1000
N_TEST  = 1000


def _npz(split: str) -> Path:
    return DATA_DIR / f"{split}.npz"


def is_prepared() -> bool:
    for s in ("train", "val", "test"):
        p = _npz(s)
        if not p.exists():
            return False
        try:
            d = np.load(p)
            _ = d["labels"].shape  # probe integrity
        except Exception:
            p.unlink(missing_ok=True)  # delete corrupted file
            return False
    return True


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (images, labels) depuis le fichier npz sauvegardé."""
    try:
        d = np.load(_npz(split))
        return d["images"].astype(np.float32), d["labels"].astype(np.int32)
    except (EOFError, Exception) as e:
        _npz(split).unlink(missing_ok=True)
        raise RuntimeError(
            f"Fichier {split}.npz corrompu (supprimé). "
            "Cliquez sur 'Télécharger le dataset' pour le re-préparer."
        ) from e


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

    # Collecte équilibrée : exactement n_per_class images par classe
    n_per_class = (n_train + n_val + n_test) // 2

    ds = tfds.load(
        "cats_vs_dogs",
        split="train",          # itère tout le dataset ; on s'arrête dès qu'on a assez
        as_supervised=True,
        shuffle_files=True,
    )

    buckets: dict[int, list] = {0: [], 1: []}
    for img, lbl in ds:
        cls = int(lbl.numpy())
        if len(buckets[cls]) < n_per_class:
            img_r = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
            buckets[cls].append(tf.cast(img_r, tf.float32).numpy() / 255.0)
        if all(len(v) >= n_per_class for v in buckets.values()):
            break

    imgs = np.array(buckets[0] + buckets[1], dtype=np.float32)
    lbls = np.array([0] * len(buckets[0]) + [1] * len(buckets[1]), dtype=np.int32)

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
