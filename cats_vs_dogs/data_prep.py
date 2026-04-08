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
# Oxford IIIT Pet : ~2400 chats, ~4977 chiens → max ~4800 images équilibrées.
# On s'arrête à 1600/200/200 pour rester dans les limites de la classe minoritaire.
N_TRAIN = 1600
N_VAL   =  200
N_TEST  =  200


def _npz(split: str) -> Path:
    return DATA_DIR / f"{split}.npz"


def is_prepared() -> bool:
    for s in ("train", "val", "test"):
        p = _npz(s)
        if not p.exists():
            return False
        try:
            d = np.load(p)
            if d["labels"].shape[0] == 0:  # split vide → rerequiert téléchargement
                return False
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


_CAT_BREEDS = {
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue",
    "Siamese", "Sphynx",
}


def _label_from_filename(filename: str) -> int | None:
    """0 = chat, 1 = chien, None = inconnu.
    Convention Oxford IIIT Pet : breed Uppercase = chat, lowercase = chien."""
    stem = Path(filename).stem           # ex: "Abyssinian_34" ou "beagle_001"
    parts = stem.rsplit("_", 1)
    if not parts:
        return None
    breed = parts[0]
    if not breed:
        return None
    # Uppercase first letter → chat
    return 0 if breed[0].isupper() else 1


def download_and_prepare(
    n_train: int = N_TRAIN,
    n_val:   int = N_VAL,
    n_test:  int = N_TEST,
) -> str:
    """
    Télécharge Oxford IIIT Pet Dataset et sauvegarde les splits.
    Premier lancement : télécharge ~800 MB dans ~/.keras/datasets/.
    Retourne un message de statut.
    """
    import tensorflow as tf

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Oxford IIIT Pet Dataset — source stable et pérenne
    _URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"

    tar_path = tf.keras.utils.get_file(
        "oxford_iiit_pet_images.tar.gz",
        origin=_URL,
        extract=True,
    )
    images_dir = Path(tar_path).parent / "images"

    # Collecte équilibrée : exactement n_per_class images par classe
    n_per_class = (n_train + n_val + n_test) // 2
    buckets: dict[int, list] = {0: [], 1: []}

    files = sorted(images_dir.glob("*.jpg"))
    for f in files:
        label = _label_from_filename(f.name)
        if label is None:
            continue
        if len(buckets[label]) >= n_per_class:
            continue
        try:
            raw = tf.io.read_file(str(f))
            img = tf.image.decode_jpeg(raw, channels=3)
            img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
            buckets[label].append(tf.cast(img, tf.float32).numpy() / 255.0)
        except Exception:
            continue  # ignore les images corrompues

    if not buckets[0] or not buckets[1]:
        raise RuntimeError(
            "Aucune image trouvée dans Oxford IIIT Pet. "
            f"Vérifiez que {images_dir} contient les fichiers .jpg extraits."
        )

    # Équilibrage : garder le même nombre d'images par classe
    n_effective = min(len(buckets[0]), len(buckets[1]))
    cats_imgs = buckets[0][:n_effective]
    dogs_imgs = buckets[1][:n_effective]

    imgs = np.array(cats_imgs + dogs_imgs, dtype=np.float32)
    lbls = np.array([0] * len(cats_imgs) + [1] * len(dogs_imgs), dtype=np.int32)

    # Mélange reproductible
    rng  = np.random.default_rng(42)
    perm = rng.permutation(len(imgs))
    imgs, lbls = imgs[perm], lbls[perm]

    # Adapter les splits si moins d'images disponibles que demandé
    total_available = len(imgs)
    total_requested = n_train + n_val + n_test
    if total_available < total_requested:
        ratio = total_available / total_requested
        n_train = max(1, int(n_train * ratio))
        n_val   = max(1, int(n_val   * ratio))
        n_test  = max(1, total_available - n_train - n_val)

    splits = {
        "train": (imgs[:n_train],                 lbls[:n_train]),
        "val":   (imgs[n_train:n_train + n_val],  lbls[n_train:n_train + n_val]),
        "test":  (imgs[n_train + n_val:n_train + n_val + n_test],
                  lbls[n_train + n_val:n_train + n_val + n_test]),
    }
    for name, (X, y) in splits.items():
        np.savez_compressed(_npz(name), images=X, labels=y)

    cats_count = int((lbls == 0).sum())
    dogs_count = int((lbls == 1).sum())
    return (
        f"✓ Dataset prêt — {n_train} train / {n_val} val / {n_test} test\n"
        f"  {cats_count} chats · {dogs_count} chiens · images {IMG_SIZE}×{IMG_SIZE}\n"
        f"  Sauvegardé dans {DATA_DIR}"
    )
