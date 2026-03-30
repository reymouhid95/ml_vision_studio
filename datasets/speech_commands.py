"""
Google Speech Commands — dataset de démo pour l'onglet Audio.
10 mots clés : yes / no / up / down / left / right / on / off / stop / go.
Téléchargement via tensorflow_datasets (streaming partiel — ~100–150 MB).

Stratégie taille :  split="train[:6%]" → TFDS ne télécharge que les premiers shards,
ce qui représente environ 100–150 MB au lieu des ~1.4 GB du dataset complet.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA_DIR    = Path(__file__).parent / "speech_data"
_META_PATH  = DATA_DIR / "meta.json"
CLASS_NAMES = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
N_PER_CLASS = 50    # 50/classe × 10 = 500 clips total
SAMPLE_RATE = 16000


def is_prepared() -> bool:
    if not _META_PATH.exists():
        return False
    return all((DATA_DIR / f"{n}.npz").exists() for n in CLASS_NAMES)


def load_all_as_audio_classes() -> list[dict]:
    """Retourne list[{name, samples: [np.ndarray(40,)]}] compatible avec audio_trainer."""
    classes = []
    for name in CLASS_NAMES:
        path = DATA_DIR / f"{name}.npz"
        if path.exists():
            d = np.load(path)
            classes.append({"name": name, "samples": list(d["features"])})
    return classes


def _extract_mel(y: np.ndarray) -> np.ndarray:
    """40-dim Mel feature — identique à core/audio_trainer.extract_mel_features."""
    import librosa

    S = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=40, fmin=80, fmax=8000)
    log_S = np.log(S + 1e-8)
    feat = log_S.mean(axis=1).astype(np.float32)
    mn, mx = feat.min(), feat.max()
    return (feat - mn) / (mx - mn + 1e-8)


def download_and_prepare(n_per_class: int = N_PER_CLASS) -> str:
    """
    Télécharge un sous-ensemble de Speech Commands via tensorflow_datasets.
    Utilise split="train[:6%]" pour limiter le téléchargement à ~150 MB.
    """
    import tensorflow_datasets as tfds

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # [:6%] → ~5100 exemples sur 85511 ; couvre largement 50/classe × 10 classes
    ds, info = tfds.load(
        "speech_commands",
        split="train[:6%]",
        as_supervised=True,
        with_info=True,
        shuffle_files=False,  # lecture séquentielle → moins de shards téléchargés
    )
    label_names: list[str] = info.features["label"].names

    target_set = set(CLASS_NAMES)
    buckets: dict[str, list[np.ndarray]] = {n: [] for n in CLASS_NAMES}

    for audio_t, lbl_t in ds:
        word = label_names[int(lbl_t.numpy())]
        if word not in target_set:
            continue
        if len(buckets[word]) >= n_per_class:
            continue

        # audio_t : Tensor float32 (16000,) — valeurs dans [-1, 1]
        y = audio_t.numpy().astype(np.float32)
        if y.ndim > 1:
            y = y.mean(axis=-1)   # mono si besoin

        feat = _extract_mel(y)
        buckets[word].append(feat)

        if all(len(v) >= n_per_class for v in buckets.values()):
            break

    saved = []
    for name in CLASS_NAMES:
        feats = buckets[name]
        if feats:
            np.savez_compressed(
                DATA_DIR / f"{name}.npz",
                features=np.array(feats, dtype=np.float32),
            )
            saved.append(name)

    _META_PATH.write_text(json.dumps({"class_names": saved}))

    lines = [f"  {n}: {len(buckets[n])} clips" for n in CLASS_NAMES]
    return (
        f"✓ Speech Commands prêt — {len(saved)}/10 mots chargés\n"
        + "\n".join(lines)
        + f"\n  Features 40-dim Mel · {DATA_DIR}"
    )
