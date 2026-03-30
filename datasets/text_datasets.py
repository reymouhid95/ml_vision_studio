"""
AG News Subset — dataset de démo pour l'onglet Texte.
4 classes : Monde, Sports, Business, Tech.
Téléchargement via tensorflow_datasets (~31 MB).
"""
from __future__ import annotations

import json
from pathlib import Path

DATA_DIR    = Path(__file__).parent / "text_data"
_DATA_PATH  = DATA_DIR / "ag_news.json"
CLASS_NAMES = ["Monde", "Sports", "Business", "Tech"]
N_PER_CLASS = 80   # 80 articles/classe × 4 = 320 exemples


def is_prepared() -> bool:
    return _DATA_PATH.exists()


def load_all_as_text_classes() -> list[dict]:
    """Retourne list[{name, samples: [str]}] compatible avec text_trainer."""
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_counts() -> dict[str, int]:
    if not is_prepared():
        return {}
    data = load_all_as_text_classes()
    return {item["name"]: len(item["samples"]) for item in data}


def download_and_prepare(n_per_class: int = N_PER_CLASS) -> str:
    """
    Télécharge ag_news_subset via tensorflow_datasets et sauvegarde en JSON.
    Premier lancement : ~31 MB.
    Labels TFDS : 0=World, 1=Sports, 2=Business, 3=Sci/Tech.
    """
    import tensorflow_datasets as tfds

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ds = tfds.load("ag_news_subset", split="train", as_supervised=True)

    buckets: dict[int, list[str]] = {i: [] for i in range(4)}

    for text_t, lbl_t in ds:
        ci = int(lbl_t.numpy())
        if len(buckets[ci]) >= n_per_class:
            continue
        raw = text_t.numpy().decode("utf-8").strip()
        if len(raw) > 20:
            # Tronquer à 800 chars pour cohérence avec split_text_into_chunks
            buckets[ci].append(raw[:800])
        if all(len(v) >= n_per_class for v in buckets.values()):
            break

    data = [
        {"name": CLASS_NAMES[i], "samples": buckets[i]}
        for i in range(4)
    ]
    with open(_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    lines = [f"  {CLASS_NAMES[i]}: {len(buckets[i])} articles" for i in range(4)]
    return (
        f"✓ AG News prêt — 4 classes\n"
        + "\n".join(lines)
        + f"\n  {DATA_DIR}"
    )
