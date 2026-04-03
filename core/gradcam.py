"""
Grad-CAM (Gradient-weighted Class Activation Mapping).

Visualise quelles zones de l'image le réseau de neurones regarde
pour prendre sa décision.

Supporte :
  - EfficientNetB0 (modèle Chats & Chiens, chargé depuis disque)
  - CNN MNIST (modèle en mémoire)
"""
from __future__ import annotations

import matplotlib.cm as mcm
import matplotlib.pyplot as plt
import numpy as np

DARK_BG = "#1a1a2e"


# ─── Résolution de la couche cible ───────────────────────────────────────────

def _find_feature_layer_output(model):
    """
    Renvoie le tenseur de sortie de la meilleure couche pour Grad-CAM.

    Priorité :
      1. Sub-modèle avec > 50 couches internes (EfficientNetB0).
      2. Dernière couche Conv2D trouvée dans les couches directes.
    """
    import tensorflow as tf

    # Cas EfficientNetB0 (sous-modèle imbriqué)
    for layer in reversed(model.layers):
        sub_layers = getattr(layer, "layers", [])
        if len(sub_layers) > 50:
            return layer.output, layer.name

    # Cas CNN simple (MNIST) — dernière Conv2D
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.output, layer.name

    raise ValueError(
        "Impossible de trouver une couche convolutive pour Grad-CAM. "
        "Assurez-vous que le modèle est bien entraîné."
    )


# ─── Calcul Grad-CAM ─────────────────────────────────────────────────────────

def compute_gradcam(
    model,
    img_array: np.ndarray,
    class_idx: int | None = None,
) -> tuple[np.ndarray, int, float]:
    """
    Calcule la heatmap Grad-CAM pour une image.

    Parameters
    ----------
    model     : modèle Keras compilé
    img_array : float32, shape (1, H, W, C), normalisé [0, 1]
    class_idx : classe cible (None → classe prédite)

    Returns
    -------
    heatmap   : (H_feat, W_feat) float32 normalisé [0, 1]
    class_idx : int
    confidence: float (probabilité de la classe prédite)
    """
    import tensorflow as tf

    inputs = tf.cast(img_array, tf.float32)

    # Un modèle Sequential chargé depuis disque n'a pas encore model.input défini
    # tant qu'il n'a pas été appelé une fois — on force ce passage forward.
    try:
        _ = model.input
    except (AttributeError, RuntimeError):
        model(inputs, training=False)

    feat_output, _ = _find_feature_layer_output(model)
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[feat_output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_out, predictions = grad_model(inputs)
        if class_idx is None:
            class_idx = int(tf.argmax(predictions[0]).numpy())
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out_np = conv_out[0].numpy()
    pooled_grads_np = pooled_grads.numpy()

    # Pondère chaque filtre par son gradient moyen
    for i, w in enumerate(pooled_grads_np):
        conv_out_np[:, :, i] *= w

    heatmap = np.mean(conv_out_np, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_val = heatmap.max()
    if max_val > 0:
        heatmap = heatmap / max_val

    confidence = float(predictions[0][class_idx].numpy())
    return heatmap.astype(np.float32), class_idx, confidence


# ─── Superposition heatmap / image ───────────────────────────────────────────

def resize_heatmap(heatmap: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Redimensionne la heatmap à (target_h, target_w) via PIL."""
    from PIL import Image
    h_uint8 = (heatmap * 255).astype(np.uint8)
    h_resized = np.array(
        Image.fromarray(h_uint8).resize((target_w, target_h), Image.LANCZOS)
    ) / 255.0
    return h_resized.astype(np.float32)


def overlay_heatmap(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Superpose la heatmap colorée sur l'image originale.

    original_img : (H, W, 3) uint8 ou float32 [0,1]
    heatmap      : (H, W) float32 [0,1] — déjà redimensionnée
    Returns      : (H, W, 3) uint8
    """
    h, w = original_img.shape[:2]
    heatmap_r = resize_heatmap(heatmap, h, w)

    colormap = mcm.jet(heatmap_r)[:, :, :3]   # (H, W, 3) float [0,1]

    if original_img.dtype == np.uint8:
        orig_f = original_img.astype(np.float32) / 255.0
    else:
        orig_f = original_img.astype(np.float32)
        if orig_f.max() > 1.0:
            orig_f = orig_f / 255.0

    # Si image en niveaux de gris, convertir en RGB
    if orig_f.ndim == 2:
        orig_f = np.stack([orig_f] * 3, axis=-1)
    elif orig_f.shape[-1] == 1:
        orig_f = np.concatenate([orig_f] * 3, axis=-1)

    blended = orig_f * (1 - alpha) + colormap * alpha
    return np.clip(blended * 255, 0, 255).astype(np.uint8)


# ─── Figure complète nette ───────────────────────────────────────────────────

def make_gradcam_figure(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    class_name: str,
    confidence: float,
    title: str = "Grad-CAM — Zones d'attention du réseau",
) -> plt.Figure:
    """
    3 sous-figures côte à côte : Image | Heatmap | Superposition.
    """
    def _to_uint8(img):
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 1) if img.max() <= 1.0 else np.clip(img, 0, 255)
            img = (img * 255 if img.max() <= 1.0 else img).astype(np.uint8)
        return img

    h, w = original_img.shape[:2]
    heatmap_r = resize_heatmap(heatmap, h, w)
    heatmap_rgb = (mcm.jet(heatmap_r)[:, :, :3] * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor(DARK_BG)

    panels = [
        (_to_uint8(original_img), "Image originale"),
        (heatmap_rgb,             "Heatmap Grad-CAM"),
        (overlay,                 f"Prédiction : {class_name}\nConfiance : {confidence:.1%}"),
    ]
    for ax, (img, subtit) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(subtit, color="white", fontsize=10, fontweight="bold", pad=6)
        ax.axis("off")

    fig.suptitle(title, color="#a855f7", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig
