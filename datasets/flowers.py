"""
tf_flowers — dataset de démo pour l'onglet Image.
5 classes : dandelion, daisy, tulips, sunflowers, roses.
Téléchargement via tensorflow_datasets (~210 MB premier lancement).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA_DIR   = Path(__file__).parent / "flowers_data"
_META_PATH = DATA_DIR / "meta.json"
N_PER_CLASS = 80   # 80/classe × 5 = 400 images ; augmentation ×3 → ~1200 ex.
IMG_SIZE    = 224  # identique à core/image_trainer.py


def _class_names() -> list[str]:
    if _META_PATH.exists():
        return json.loads(_META_PATH.read_text())["class_names"]
    return []


def is_prepared() -> bool:
    names = _class_names()
    if not names:
        return False
    return all((DATA_DIR / f"{n}.npz").exists() for n in names)


def load_all_as_image_classes() -> list[dict]:
    """Retourne list[{name, samples: [PIL.Image]}] compatible avec image_trainer."""
    from PIL import Image as PILImage

    classes = []
    for name in _class_names():
        d = np.load(DATA_DIR / f"{name}.npz")
        imgs_u8 = d["images"]   # (N, H, W, 3) uint8
        pil_imgs = [PILImage.fromarray(imgs_u8[i]) for i in range(len(imgs_u8))]
        classes.append({"name": name, "samples": pil_imgs})
    return classes


def sample_counts() -> dict[str, int]:
    if not is_prepared():
        return {}
    counts = {}
    for name in _class_names():
        d = np.load(DATA_DIR / f"{name}.npz")
        counts[name] = int(d["images"].shape[0])
    return counts


def download_and_prepare(n_per_class: int = N_PER_CLASS) -> str:
    """
    Télécharge tf_flowers et sauvegarde N_PER_CLASS images par classe en uint8 npz.
    Premier lancement : ~210 MB téléchargés dans ~/tensorflow_datasets/.
    """
    import tensorflow as tf
    import tensorflow_datasets as tfds

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ds, info = tfds.load(
        "tf_flowers",
        split="train",
        as_supervised=True,
        shuffle_files=True,
        with_info=True,
    )
    class_names: list[str] = info.features["label"].names
    # ex. : ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']

    buckets: dict[str, list[np.ndarray]] = {n: [] for n in class_names}

    for img_t, lbl_t in ds:
        name = class_names[int(lbl_t.numpy())]
        if len(buckets[name]) < n_per_class:
            arr = tf.cast(tf.image.resize(img_t, [IMG_SIZE, IMG_SIZE]), tf.uint8).numpy()
            buckets[name].append(arr)
        if all(len(v) >= n_per_class for v in buckets.values()):
            break

    for name, arrs in buckets.items():
        np.savez_compressed(
            DATA_DIR / f"{name}.npz",
            images=np.array(arrs, dtype=np.uint8),
        )

    _META_PATH.write_text(json.dumps({"class_names": class_names}))

    lines = [f"  {n}: {len(buckets[n])} images" for n in class_names]
    return (
        f"✓ tf_flowers prêt — {len(class_names)} classes\n"
        + "\n".join(lines)
        + f"\n  Résolution {IMG_SIZE}×{IMG_SIZE} · {DATA_DIR}"
    )
