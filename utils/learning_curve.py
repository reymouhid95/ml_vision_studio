"""
Courbe d'apprentissage — diagnostic underfitting / overfitting.

Principe :
  Split fixe train (80%) / val (20%).
  Entraîne sur 10%, 20%, …, 100% du train.
  Trace train_acc et val_acc en fonction du nombre d'exemples.

Interprétation :
  - train >> val et val ↗ encore   → surapprentissage + plus de données aiderait
  - train ≈ val ≈ bas              → sous-apprentissage (modèle ou données à revoir)
  - train ≈ val et val plateau     → convergé, plus de données n'aiderait plus beaucoup

Modalités :
  image  → features MobileNetV2 (une extraction) + Logistic Regression (très rapide)
  audio  → ré-entraîne réseau Dense 40-dim sur chaque fraction (~5-10 s)
  texte  → KNN cosine sur fractions, LOO sur train + acc sur val (pas d'entraîn.)
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Visualisation ─────────────────────────────────────────────────────────────

def _lc_figure(
    sizes: list[int],
    train_accs: list[float],
    val_accs: list[float],
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#0f0f1a")

    ax.plot(sizes, train_accs, "o-", color="#a855f7", linewidth=2, markersize=5, label="Train")
    ax.plot(sizes, val_accs,   "o-", color="#f59e0b", linewidth=2, markersize=5, label="Validation")
    ax.fill_between(sizes, train_accs, val_accs, alpha=0.08, color="#a855f7")

    ax.set_xlabel("Exemples d'entraînement", color="white", fontsize=9)
    ax.set_ylabel("Précision", color="white", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_title(title, color="white", fontsize=11)
    ax.tick_params(colors="white")
    ax.grid(alpha=0.2, color="#444")
    ax.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")

    # Annotation diagnostic sur le graphe
    gap = train_accs[-1] - val_accs[-1]
    slope = val_accs[-1] - val_accs[len(val_accs) // 2] if len(val_accs) >= 2 else 0

    if gap > 0.20:
        msg, clr = f"Surapprentissage gap={gap:.0%}", "#f87171"
    elif val_accs[-1] < 0.55:
        msg, clr = "Sous-apprentissage", "#fbbf24"
    elif slope > 0.03:
        msg, clr = "Plus de données aideraient ↗", "#34d399"
    else:
        msg, clr = "Courbe convergée ✓", "#6ee7b7"

    ax.text(0.02, 0.04, msg, transform=ax.transAxes,
            fontsize=9, color=clr, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f0f1a", edgecolor=clr, alpha=0.7))

    fig.tight_layout(pad=1.5)
    return fig


def _diagnostic(train_accs: list[float], val_accs: list[float], n_classes: int) -> str:
    gap   = train_accs[-1] - val_accs[-1]
    slope = val_accs[-1] - val_accs[len(val_accs) // 2] if len(val_accs) >= 2 else 0
    chance = 1 / n_classes

    lines = [
        f"Précision finale  —  train : {train_accs[-1]:.1%}  |  val : {val_accs[-1]:.1%}"
    ]
    if gap > 0.20:
        lines += [
            f"⚠️  Surapprentissage important (gap={gap:.0%}).",
            "    → Ajoutez des données d'entraînement ou augmentez la régularisation (dropout).",
        ]
    elif val_accs[-1] < chance + 0.05:
        lines += [
            f"⚠️  Précision proche du hasard (aléatoire={chance:.0%}).",
            "    → Vérifiez la qualité et la diversité des données.",
        ]
    elif slope > 0.03:
        lines += [
            "💡 La précision de validation est encore en progression.",
            "    → Collecter plus de données améliorera probablement le modèle.",
        ]
    else:
        lines += [
            "✅ La courbe a convergé — plus de données n'apporteraient pas d'amélioration majeure.",
            "    → Le modèle est bien ajusté pour la quantité de données disponible.",
        ]
    return "\n".join(lines)


# ── Image ─────────────────────────────────────────────────────────────────────

def image_learning_curve(
    classes: list[dict],
    n_points: int = 8,
) -> tuple[plt.Figure | None, str]:
    """
    Courbe d'apprentissage Image.
    Extrait les features MobileNetV2 une seule fois, puis entraîne
    une Régression Logistique sur des fractions croissantes → très rapide.
    """
    import tensorflow as tf
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    IMG_SIZE = 224
    xs_raw, ys = [], []
    for ci, cls in enumerate(classes):
        for img in cls["samples"]:
            arr = np.array(img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB"), dtype=np.float32)
            xs_raw.append(arr / 127.5 - 1.0)   # normalisation MobileNetV2
            ys.append(ci)

    n_total = len(xs_raw)
    n_classes = len(classes)
    if n_total < n_classes * 2:
        return None, "Pas assez d'exemples (minimum 2 par classe)."

    # Extraction de features (passage unique dans MobileNetV2, par batches → pas d'OOM)
    xs_arr = np.array(xs_raw, dtype=np.float32)
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    pool = tf.keras.layers.GlobalAveragePooling2D()
    BATCH = 32
    feat_list = []
    for i in range(0, len(xs_arr), BATCH):
        batch = xs_arr[i:i + BATCH]
        feat_list.append(pool(base(batch, training=False)).numpy())
    feats = np.concatenate(feat_list, axis=0)   # (N, 1280)

    # Mélange reproductible
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_total)
    feats, ys = feats[perm], np.array(ys)[perm]

    n_val   = max(n_classes, int(n_total * 0.2))
    n_train = n_total - n_val
    if n_train < n_classes:
        return None, "Pas assez d'exemples pour séparer train / val."

    X_tr, y_tr = feats[:n_train], ys[:n_train]
    X_val, y_val = feats[n_train:], ys[n_train:]

    clf = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=500, C=10, random_state=0)),
    ])

    fracs = np.linspace(1 / n_points, 1.0, n_points)
    sizes, train_accs, val_accs = [], [], []

    for frac in fracs:
        n_use = max(n_classes, int(n_train * frac))
        X_sub, y_sub = X_tr[:n_use], y_tr[:n_use]
        if len(np.unique(y_sub)) < n_classes:
            continue

        clf.fit(X_sub, y_sub)
        sizes.append(n_use)
        train_accs.append(float((clf.predict(X_sub) == y_sub).mean()))
        val_accs.append(float((clf.predict(X_val) == y_val).mean()))

    if not sizes:
        return None, "Données insuffisantes après filtrage."

    fig  = _lc_figure(sizes, train_accs, val_accs, "Courbe d'apprentissage — Image")
    diag = _diagnostic(train_accs, val_accs, n_classes)
    return fig, diag


# ── Audio ─────────────────────────────────────────────────────────────────────

def audio_learning_curve(
    classes: list[dict],
    n_points: int = 8,
    epochs: int = 40,
) -> tuple[plt.Figure | None, str]:
    """
    Courbe d'apprentissage Audio.
    Ré-entraîne le réseau Dense 40-dim sur chaque fraction.
    Rapide (~5-15 s au total).
    """
    import tensorflow as tf

    xs, ys = [], []
    for ci, cls in enumerate(classes):
        for feat in cls["samples"]:
            xs.append(feat)
            ys.append(ci)

    n_total   = len(xs)
    n_classes = len(classes)
    if n_total < n_classes * 2:
        return None, "Pas assez d'échantillons (minimum 2 par classe)."

    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.int32)

    rng = np.random.default_rng(42)
    perm = rng.permutation(n_total)
    xs, ys = xs[perm], ys[perm]

    n_val   = max(n_classes, int(n_total * 0.2))
    n_train = n_total - n_val
    if n_train < n_classes:
        return None, "Pas assez d'échantillons pour séparer train / val."

    X_tr, y_tr = xs[:n_train], ys[:n_train]
    X_val, y_val = xs[n_train:], ys[n_train:]

    def _build():
        h = 128
        m = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(40,)),
            tf.keras.layers.Dense(h, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(h // 2, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ])
        m.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return m

    fracs = np.linspace(1 / n_points, 1.0, n_points)
    sizes, train_accs, val_accs = [], [], []

    for frac in fracs:
        n_use = max(n_classes, int(n_train * frac))
        X_sub, y_sub = X_tr[:n_use], y_tr[:n_use]
        if len(np.unique(y_sub)) < n_classes:
            continue

        y_oh = tf.keras.utils.to_categorical(y_sub, n_classes)
        model = _build()
        model.fit(X_sub, y_oh, epochs=epochs,
                  batch_size=max(4, n_use // 8),
                  verbose=0, shuffle=True)

        tr_preds  = model(X_sub,  training=False).numpy().argmax(axis=1)
        val_preds = model(X_val, training=False).numpy().argmax(axis=1)
        sizes.append(n_use)
        train_accs.append(float((tr_preds == y_sub).mean()))
        val_accs.append(float((val_preds == y_val).mean()))

    if not sizes:
        return None, "Données insuffisantes après filtrage."

    fig  = _lc_figure(sizes, train_accs, val_accs, "Courbe d'apprentissage — Audio")
    diag = _diagnostic(train_accs, val_accs, n_classes)
    return fig, diag


# ── Texte ─────────────────────────────────────────────────────────────────────

def text_learning_curve(
    knn_entries: list[dict],
    class_names: list[str],
    n_points: int = 8,
) -> tuple[plt.Figure | None, str]:
    """
    Courbe d'apprentissage Texte (KNN cosine).
    Pas d'entraînement : LOO sur le train + acc sur val à chaque fraction.
    """
    from core.text_trainer import classify_knn

    n_total   = len(knn_entries)
    n_classes = len(class_names)
    if n_total < n_classes * 2:
        return None, "Pas assez d'embeddings (minimum 2 par classe)."

    rng = np.random.default_rng(42)
    perm = rng.permutation(n_total).tolist()
    entries = [knn_entries[i] for i in perm]

    n_val   = max(n_classes, int(n_total * 0.2))
    n_train = n_total - n_val
    if n_train < n_classes:
        return None, "Pas assez d'embeddings pour séparer train / val."

    train_pool = entries[:n_train]
    val_pool   = entries[n_train:]

    fracs = np.linspace(1 / n_points, 1.0, n_points)
    sizes, train_accs, val_accs = [], [], []

    for frac in fracs:
        n_use  = max(n_classes, int(n_train * frac))
        subset = train_pool[:n_use]

        # Toutes les classes présentes ?
        if len({e["classIdx"] for e in subset}) < n_classes:
            continue

        # LOO accuracy sur le sous-ensemble train
        correct_tr = 0
        for i, item in enumerate(subset):
            rest = subset[:i] + subset[i + 1:]
            if not rest:
                continue
            scores = classify_knn(item["embedding"], rest, class_names)
            if max(scores, key=scores.get) == class_names[item["classIdx"]]:
                correct_tr += 1

        # Accuracy sur val
        correct_val = 0
        for item in val_pool:
            scores = classify_knn(item["embedding"], subset, class_names)
            if max(scores, key=scores.get) == class_names[item["classIdx"]]:
                correct_val += 1

        sizes.append(n_use)
        train_accs.append(correct_tr / max(n_use, 1))
        val_accs.append(correct_val / max(len(val_pool), 1))

    if not sizes:
        return None, "Données insuffisantes après filtrage."

    fig  = _lc_figure(sizes, train_accs, val_accs, "Courbe d'apprentissage — Texte (KNN)")
    diag = _diagnostic(train_accs, val_accs, n_classes)
    return fig, diag


# ── Chats & Chiens ─────────────────────────────────────────────────────────────

def cats_dogs_learning_curve(
    n_points: int = 8,
) -> tuple[plt.Figure | None, str]:
    """
    Courbe d'apprentissage Chats & Chiens.
    Extrait les features EfficientNetB0 UNE fois (batches 32 → pas d'OOM),
    puis entraîne une Régression Logistique sur des fractions croissantes.
    Charge les splits depuis cats_vs_dogs/data/train.npz et val.npz.
    """
    import tensorflow as tf
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from cats_vs_dogs.data_prep import is_prepared, load_split

    if not is_prepared():
        return None, "Dataset non préparé. Cliquez sur 'Télécharger le dataset'."

    try:
        X_train, y_train = load_split("train")
        X_val,   y_val   = load_split("val")
    except RuntimeError as e:
        return None, str(e)

    img_size = X_train.shape[1]
    base = tf.keras.applications.EfficientNetB0(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    pool = tf.keras.layers.GlobalAveragePooling2D()

    def _extract(X: np.ndarray) -> np.ndarray:
        feat_list = []
        for i in range(0, len(X), 32):
            batch = X[i:i + 32] * 255.0   # [0,1] → [0,255] pour EfficientNetB0
            feat_list.append(pool(base(batch, training=False)).numpy())
        return np.concatenate(feat_list, axis=0)

    feats_tr  = _extract(X_train)
    feats_val = _extract(X_val)

    n_train   = len(feats_tr)
    n_classes = 2

    clf = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=300, C=1.0, random_state=0)),
    ])

    fracs = np.linspace(1 / n_points, 1.0, n_points)
    sizes, train_accs, val_accs = [], [], []

    for frac in fracs:
        n_use = max(n_classes, int(n_train * frac))
        clf.fit(feats_tr[:n_use], y_train[:n_use])
        sizes.append(n_use)
        train_accs.append(float((clf.predict(feats_tr[:n_use]) == y_train[:n_use]).mean()))
        val_accs.append(float((clf.predict(feats_val) == y_val).mean()))

    fig  = _lc_figure(sizes, train_accs, val_accs,
                      "Courbe d'apprentissage — Chats & Chiens (EfficientNetB0)")
    diag = _diagnostic(train_accs, val_accs, n_classes)
    return fig, diag


# ── Helpers régression ────────────────────────────────────────────────────────

def _lc_figure_r2(
    sizes: list[int],
    train_r2: list[float],
    val_r2: list[float],
    title: str,
) -> plt.Figure:
    """Courbe d'apprentissage pour les modèles de régression (score R²)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#0f0f1a")

    ax.plot(sizes, train_r2, "o-", color="#a855f7", linewidth=2, markersize=5, label="Train R²")
    ax.plot(sizes, val_r2,   "o-", color="#f59e0b", linewidth=2, markersize=5, label="Val R²")
    ax.fill_between(sizes, train_r2, val_r2, alpha=0.08, color="#a855f7")

    ax.axhline(1.0, color="#4ade80", linewidth=0.8, linestyle=":", alpha=0.5, label="R²=1 (parfait)")
    ax.axhline(0.0, color="#f87171", linewidth=0.8, linestyle=":", alpha=0.5, label="R²=0 (moyenne)")

    ymin = min(0.0, min(val_r2) - 0.05)
    ax.set_ylim(ymin, 1.08)
    ax.set_xlabel("Exemples d'entraînement", color="white", fontsize=9)
    ax.set_ylabel("Score R²", color="white", fontsize=9)
    ax.set_title(title, color="white", fontsize=11)
    ax.tick_params(colors="white")
    ax.grid(alpha=0.2, color="#444")
    ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")

    # Annotation diagnostic
    gap   = train_r2[-1] - val_r2[-1]
    slope = val_r2[-1] - val_r2[len(val_r2) // 2] if len(val_r2) >= 2 else 0

    if gap > 0.10:
        msg, clr = f"Surapprentissage gap={gap:.2f}", "#f87171"
    elif val_r2[-1] < 0.50:
        msg, clr = "Sous-apprentissage (R² faible)", "#fbbf24"
    elif slope > 0.02:
        msg, clr = "Plus de données aideraient ↗", "#34d399"
    else:
        msg, clr = f"Convergé ✓  R²={val_r2[-1]:.3f}", "#6ee7b7"

    ax.text(0.02, 0.04, msg, transform=ax.transAxes,
            fontsize=9, color=clr, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f0f1a", edgecolor=clr, alpha=0.7))

    fig.tight_layout(pad=1.5)
    return fig


def _diagnostic_r2(train_r2: list[float], val_r2: list[float]) -> str:
    gap   = train_r2[-1] - val_r2[-1]
    slope = val_r2[-1] - val_r2[len(val_r2) // 2] if len(val_r2) >= 2 else 0

    lines = [
        f"Score R² final  —  train : {train_r2[-1]:.4f}  |  val : {val_r2[-1]:.4f}",
        f"(R²=1 = prédiction parfaite, R²=0 = prédire la moyenne)"
    ]
    if gap > 0.10:
        lines += [
            f"⚠️  Surapprentissage (gap={gap:.2f}) : le modèle mémorise l'entraînement.",
            "    → Augmentez le dataset ou réduisez la complexité du modèle.",
        ]
    elif val_r2[-1] < 0.50:
        lines += [
            "⚠️  R² faible : la régression n'explique pas bien la variance.",
            "    → Vérifiez la relation entre la feature et la cible.",
        ]
    elif slope > 0.02:
        lines += [
            "💡 Le R² de validation est encore en progression.",
            "    → Plus de données amélioreraient probablement le modèle.",
        ]
    else:
        lines += [
            "✅ Courbe convergée — le modèle est correctement ajusté.",
            f"    → R² val = {val_r2[-1]:.4f} : la régression explique {val_r2[-1]:.1%} de la variance.",
        ]
    return "\n".join(lines)


# ── Prix Maison ───────────────────────────────────────────────────────────────

def price_learning_curve(
    n_samples: int = 100,
    noise_level: float = 30.0,
    n_points: int = 8,
) -> tuple[plt.Figure | None, str]:
    """
    Courbe d'apprentissage Régression Linéaire (score R²).

    Génère le dataset complet une fois, puis entraîne la régression
    sur des fractions croissantes (10% → 100% du train).
    Split fixe 80 % / 20 %.  Rapide (~0.1 s).
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from core.price_predictor import generate_dataset

    X, y = generate_dataset(n_samples, noise_level)
    if len(X) < 4:
        return None, "Pas assez d'exemples (minimum 4)."

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    n_train = len(X_tr)

    fracs = np.linspace(1 / n_points, 1.0, n_points)
    sizes, train_r2, val_r2 = [], [], []

    for frac in fracs:
        n_use = max(2, int(n_train * frac))
        model = LinearRegression()
        model.fit(X_tr[:n_use], y_tr[:n_use])
        sizes.append(n_use)
        train_r2.append(float(model.score(X_tr[:n_use], y_tr[:n_use])))
        val_r2.append(float(model.score(X_val, y_val)))

    fig  = _lc_figure_r2(sizes, train_r2, val_r2,
                          "Courbe d'apprentissage — Prix Maison (R²)")
    diag = _diagnostic_r2(train_r2, val_r2)
    return fig, diag


# ── MNIST ─────────────────────────────────────────────────────────────────────

def mnist_learning_curve(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    n_points: int = 8,
) -> tuple[plt.Figure | None, str]:
    """
    Courbe d'apprentissage MNIST (Régression Logistique sklearn, rapide).

    Utilise au plus 10 000 exemples d'entraînement et 2 000 de test
    pour rester rapide (~20 secondes). Mesure train_acc et test_acc
    sur des fractions croissantes.

    Remarque : la Régression Logistique dépasse 90 % sur MNIST,
    ce qui illustre déjà très bien la dynamique train/val.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline     import Pipeline
    from sklearn.preprocessing import StandardScaler

    MAX_TRAIN = 10_000
    MAX_TEST  =  2_000

    # Aplatir (N, 28, 28, 1) → (N, 784)
    X_tr_flat = X_train[:MAX_TRAIN].reshape(MAX_TRAIN, -1)
    y_tr      = y_train[:MAX_TRAIN]
    X_te_flat = X_test[:MAX_TEST].reshape(MAX_TEST, -1)
    y_te      = y_test[:MAX_TEST]

    n_train = len(X_tr_flat)
    if n_train < 20:
        return None, "Pas assez d'exemples MNIST. Chargez d'abord le dataset."

    clf = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=300, C=0.1, random_state=0, n_jobs=-1)),
    ])

    fracs = np.linspace(1 / n_points, 1.0, n_points)
    sizes, train_accs, val_accs = [], [], []

    for frac in fracs:
        n_use = max(10, int(n_train * frac))
        clf.fit(X_tr_flat[:n_use], y_tr[:n_use])
        sizes.append(n_use)
        train_accs.append(float(clf.score(X_tr_flat[:n_use], y_tr[:n_use])))
        val_accs.append(float(clf.score(X_te_flat, y_te)))

    fig  = _lc_figure(sizes, train_accs, val_accs,
                      "Courbe d'apprentissage — MNIST (Régression Logistique)")
    diag = _diagnostic(train_accs, val_accs, n_classes=10)
    return fig, diag
