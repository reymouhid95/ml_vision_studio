from __future__ import annotations

import os
import site
import sys


# ── CUDA 11.8 via wheels pip (nvidia-*-cu11) pour Quadro M2000M Maxwell ──────
# TF 2.14 inclut des kernels pré-compilés pour sm_50/sm_52 (Maxwell CC 5.x).
# Le linker dynamique doit connaître les chemins .so avant le démarrage du
# processus → on re-exécute si LD_LIBRARY_PATH n'est pas encore configuré.
def _setup_cuda_and_reexec() -> None:
    sp = next((p for p in site.getsitepackages() if "site-packages" in p), None)
    if sp is None:
        return
    nvidia_root = os.path.join(sp, "nvidia")
    if not os.path.isdir(nvidia_root):
        return
    lib_dirs = [
        os.path.join(nvidia_root, pkg, "lib")
        for pkg in os.listdir(nvidia_root)
        if os.path.isdir(os.path.join(nvidia_root, pkg, "lib"))
    ]
    if not lib_dirs:
        return
    new_paths = ":".join(lib_dirs)
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if new_paths not in current:
        os.environ["LD_LIBRARY_PATH"] = new_paths + (":" + current if current else "")
        os.execv(sys.executable, [sys.executable] + sys.argv)

# Supprime les logs verbeux TensorFlow/CUDA AVANT le re-exec
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

_setup_cuda_and_reexec()

# ── libdevice.10.bc requis par XLA pour compiler les kernels GPU (Maxwell) ────
def _set_xla_libdevice() -> None:
    import site as _site
    sp = next((p for p in _site.getsitepackages() if "site-packages" in p), None)
    if sp is None:
        return
    libdevice = os.path.join(sp, "nvidia", "cuda_nvcc", "nvvm", "libdevice")
    if os.path.isdir(libdevice):
        xla = os.environ.get("XLA_FLAGS", "")
        flag = f"--xla_gpu_cuda_data_dir={libdevice}"
        if flag not in xla:
            os.environ["XLA_FLAGS"] = (xla + " " + flag).strip()

_set_xla_libdevice()

import copy
import json
import ssl
import tempfile

import certifi

ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

try:
    import importlib
    import pathlib

    import gradio_client.utils as _gc_utils
    _gc_path = pathlib.Path(_gc_utils.__file__)
    _src = _gc_path.read_text()
    _changed = False
    if 'if not isinstance(schema, dict): return "any"' not in _src:
        _src = _src.replace(
            "def get_type(schema: dict):",
            'def get_type(schema: dict):\n    if not isinstance(schema, dict): return "any"',
        )
        _changed = True
    if 'if not isinstance(schema, dict): return "Any"' not in _src:
        _src = _src.replace(
            "    if schema == {}:",
            '    if not isinstance(schema, dict): return "Any"\n    if schema == {}:',
        )
        _changed = True
    if _changed:
        _gc_path.write_text(_src)
        importlib.reload(_gc_utils)
except Exception:
    pass

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# Ensure local packages resolve when run from project root
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from cats_vs_dogs.data_prep import CLASS_NAMES as CD_CLASS_NAMES
from cats_vs_dogs.data_prep import IMG_SIZE as CD_IMG_SIZE
from cats_vs_dogs.data_prep import (download_and_prepare, is_prepared,
                                    load_split, split_counts)
from cats_vs_dogs.dl_model import model_trained as dl_model_trained
from cats_vs_dogs.dl_model import predict_dl, train_dl_model
from cats_vs_dogs.ml_model import models_trained as ml_models_trained
from cats_vs_dogs.ml_model import predict_ml, train_ml_models
from core.audio_trainer import HP_BATCH_OPTS as AUD_BATCH_OPTS
from core.audio_trainer import HP_LR_OPTS as AUD_LR_OPTS
from core.audio_trainer import (extract_mel_features, predict_audio,
                                train_audio_model)
from core.clustering import (flatten, make_elbow_figure, make_tsne_figure,
                             run_elbow, run_kmeans, run_tsne)
from core.gradcam import compute_gradcam, make_gradcam_figure, overlay_heatmap
from core.image_trainer import HP_BATCH_OPTS as IMG_BATCH_OPTS
from core.image_trainer import HP_LR_OPTS as IMG_LR_OPTS
from core.image_trainer import predict_image, train_image_model
from core.mnist_model import (evaluate_cnn, load_mnist, make_confusion_10,
                              make_sample_grid, make_training_curves,
                              predict_digit, train_cnn_model)
from core.multivariate_regression import (ALL_FEATURE_NAMES,
                                          FEATURE_DESCRIPTIONS,
                                          make_importance_figure,
                                          make_scatter_residuals_figure,
                                          predict_multivariate,
                                          train_multivariate_model)
from core.price_predictor import (generate_dataset, make_dataset_figure,
                                  make_regression_figure, predict_price,
                                  train_price_model)
from core.text_trainer import (build_knn_index, classify_knn, classify_with_nn,
                               embed_single, knn_leave_one_out,
                               split_text_into_chunks, train_text_nn_model)
# Datasets de démo
from datasets.flowers import download_and_prepare as flowers_download
from datasets.flowers import is_prepared as flowers_prepared
from datasets.flowers import load_all_as_image_classes as flowers_load
from datasets.flowers import sample_counts as flowers_counts
from datasets.speech_commands import CLASS_NAMES as SPEECH_CLASS_NAMES
from datasets.speech_commands import download_and_prepare as speech_download
from datasets.speech_commands import is_prepared as speech_prepared
from datasets.speech_commands import load_all_as_audio_classes as speech_load
from datasets.text_datasets import download_and_prepare as agnews_download
from datasets.text_datasets import is_prepared as agnews_prepared
from datasets.text_datasets import load_all_as_text_classes as agnews_load
from datasets.text_datasets import sample_counts as agnews_counts
from utils.augmentation import augment_image
from utils.confusion_matrix import make_confusion_figure
# Courbes d'apprentissage
from utils.learning_curve import (audio_learning_curve,
                                  cats_dogs_learning_curve,
                                  image_learning_curve, mnist_learning_curve,
                                  price_learning_curve, text_learning_curve)
from utils.pdf_import import extract_pdf_page_images, extract_pdf_text
# Suggestions automatiques
from utils.suggestions import (analyze_class_balance, analyze_training_results,
                               format_suggestions)
from utils.url_import import fetch_url_text

# ─────────────────────────────────────────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────────────────────────────────────────

def make_initial_state() -> dict:
    return {
        # Image
        "image_classes":     [],    # [{name, samples: [PIL.Image 224×224]}]
        "image_model":       None,
        "image_class_names": [],
        "image_trained":     False,
        # Audio
        "audio_classes":     [],    # [{name, samples: [np.ndarray (40,)]}]
        "audio_model":       None,
        "audio_class_names": [],
        "audio_trained":     False,
        # Text
        "text_classes":      [],    # [{name, samples: [str]}]
        "text_knn":          [],    # [{classIdx, embedding, text}]
        "text_model":        None,
        "text_class_names":  [],
        "text_mode":         "knn",
        "text_trained":      False,
        # Prix maison
        "price_model":       None,
        "price_metrics":     None,
        "price_X":           None,
        "price_y":           None,
        "price_X_test":      None,
        "price_y_test":      None,
        "price_y_pred":      None,
        # MNIST
        "mnist_X_train":     None,
        "mnist_y_train":     None,
        "mnist_X_test":      None,
        "mnist_y_test":      None,
        "mnist_model":       None,
        "mnist_trained":     False,
        # Clustering (K-Means)
        "clustering_k":      None,
        "clustering_sil":    None,
        "clustering_ari":    None,
        # Régression multivariée
        "multireg_pipe":     None,
        "multireg_metrics":  None,
        "multireg_features": None,
    }


def _s(state: dict) -> dict:
    """Shallow-copy state (keeps model references intact)."""
    return {**state}


def _image_summary(st: dict) -> str:
    if not st["image_classes"]:
        return "Aucune classe."
    lines = [f"{len(st['image_classes'])} classe(s) :"]
    for c in st["image_classes"]:
        lines.append(f"  • {c['name']} : {len(c['samples'])} image(s)")
    if st["image_trained"]:
        lines.append("✓ Modèle entraîné")
    return "\n".join(lines)


def _audio_summary(st: dict) -> str:
    if not st["audio_classes"]:
        return "Aucune classe."
    lines = [f"{len(st['audio_classes'])} classe(s) :"]
    for c in st["audio_classes"]:
        lines.append(f"  • {c['name']} : {len(c['samples'])} échantillon(s)")
    if st["audio_trained"]:
        lines.append("✓ Modèle entraîné")
    return "\n".join(lines)


def _text_summary(st: dict) -> str:
    if not st["text_classes"]:
        return "Aucune classe."
    lines = [f"{len(st['text_classes'])} classe(s) :"]
    for c in st["text_classes"]:
        lines.append(f"  • {c['name']} : {len(c['samples'])} texte(s)")
    if st["text_trained"]:
        lines.append("✓ Index KNN prêt" + (" + NN" if st["text_mode"] == "nn" else ""))
    return "\n".join(lines)


def _loss_fig(loss_hist: list[float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 2.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#0f0f1a")
    ax.plot(loss_hist, color="#a855f7", linewidth=2)
    ax.set_xlabel("Époque", color="white", fontsize=9)
    ax.set_ylabel("Perte",  color="white", fontsize=9)
    ax.set_title("Courbe de perte", color="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.grid(alpha=0.2, color="#444")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    fig.tight_layout(pad=1.0)
    return fig


def _cd_train_fig(loss_hist: dict) -> plt.Figure:
    """Dual chart: train/val loss + train/val accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="white")
        ax.grid(alpha=0.2, color="#444")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")
    epochs = range(1, len(loss_hist["train_loss"]) + 1)
    ax1.plot(epochs, loss_hist["train_loss"], color="#a855f7", linewidth=2, label="train")
    ax1.plot(epochs, loss_hist["val_loss"],   color="#f59e0b", linewidth=2, label="val")
    ax1.set_title("Perte",     color="white", fontsize=10)
    ax1.set_xlabel("Époque",   color="white", fontsize=9)
    ax1.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
    ax2.plot(epochs, loss_hist["train_acc"], color="#10b981", linewidth=2, label="train")
    ax2.plot(epochs, loss_hist["val_acc"],   color="#f59e0b", linewidth=2, label="val")
    ax2.set_title("Précision", color="white", fontsize=10)
    ax2.set_xlabel("Époque",   color="white", fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
    fig.tight_layout(pad=1.5)
    return fig


def _cls_choices(classes: list[dict]) -> list[str]:
    return [c["name"] for c in classes]


# ─────────────────────────────────────────────────────────────────────────────
#  IMAGE CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def img_add_class(class_name: str, state: dict):
    state = _s(state)
    state["image_classes"] = copy.deepcopy(state["image_classes"])
    name = class_name.strip() or f"Classe {len(state['image_classes']) + 1}"
    state["image_classes"].append({"name": name, "samples": []})
    choices = _cls_choices(state["image_classes"])
    return (
        state,
        gr.Dropdown(choices=choices, value=choices[-1]),
        _image_summary(state),
        "",  # clear name input
    )


def img_capture_sample(image, active_cls: str, state: dict):
    if image is None:
        return state, _image_summary(state), "Aucune image capturée."
    if not active_cls:
        return state, _image_summary(state), "Sélectionnez d'abord une classe."
    state = _s(state)
    state["image_classes"] = copy.deepcopy(state["image_classes"])
    cls = next((c for c in state["image_classes"] if c["name"] == active_cls), None)
    if cls is None:
        return state, _image_summary(state), "Classe introuvable."
    img_224 = image.resize((224, 224)).convert("RGB")
    cls["samples"].append(img_224)
    return state, _image_summary(state), f"Image #{len(cls['samples'])} ajoutée à « {active_cls} »."


def img_train(epochs, lr, batch_size, units, state, progress=gr.Progress()):
    classes = state["image_classes"]
    if len(classes) < 2:
        yield state, "Ajoutez au moins 2 classes.", None, None, ""
        return
    if any(len(c["samples"]) < 3 for c in classes):
        yield state, "Chaque classe doit avoir au moins 3 images.", None, None, ""
        return

    # Suggestions avant entraînement
    pre_sugg = analyze_class_balance(classes)

    log_lines: list[str] = []
    loss_hist: list[float] = []

    for update in train_image_model(
        classes, epochs=epochs, lr=lr, batch_size=batch_size, dense_units=units
    ):
        tag = update[0]
        if tag == "epoch":
            _, ep, total, loss, acc = update
            progress(ep / total, desc=f"Époque {ep}/{total}")
            log_lines.append(f"[{ep:3d}/{total}]  perte={loss:.4f}  acc={acc:.2%}")
            loss_hist.append(loss)
            yield state, "\n".join(log_lines[-20:]), _loss_fig(loss_hist), None, ""

        elif tag == "done":
            _, model, lh, cnames, preds, actuals = update
            state = _s(state)
            state["image_model"]       = model
            state["image_class_names"] = cnames
            state["image_trained"]     = True
            conf = make_confusion_figure(preds, actuals, cnames)

            train_acc = sum(p == a for p, a in zip(preds, actuals)) / max(len(preds), 1)
            post_sugg = analyze_training_results(
                lh, train_acc=train_acc, n_classes=len(cnames)
            )
            sugg_txt = format_suggestions(pre_sugg + post_sugg)
            log_lines.append("✓ Entraînement terminé !")
            yield state, "\n".join(log_lines[-20:]), _loss_fig(lh), conf, sugg_txt


def img_predict(image, state: dict):
    if image is None or state["image_model"] is None:
        return None
    return predict_image(state["image_model"], image, state["image_class_names"])


def img_save_model(state: dict):
    if state["image_model"] is None:
        gr.Warning("Aucun modèle image entraîné.")
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".keras", delete=False,
                                      prefix="ml_vision_image_")
    state["image_model"].save(tmp.name)
    return tmp.name


# ─────────────────────────────────────────────────────────────────────────────
#  AUDIO CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def aud_add_class(class_name: str, state: dict):
    state = _s(state)
    state["audio_classes"] = copy.deepcopy(state["audio_classes"])
    name = class_name.strip() or f"Classe {len(state['audio_classes']) + 1}"
    state["audio_classes"].append({"name": name, "samples": []})
    choices = _cls_choices(state["audio_classes"])
    return (
        state,
        gr.Dropdown(choices=choices, value=choices[-1]),
        _audio_summary(state),
        "",  # clear name input
    )


def aud_add_sample(audio_path, active_cls: str, state: dict):
    if audio_path is None:
        return state, "Aucun audio enregistré.", _audio_summary(state)
    if not active_cls:
        return state, "Sélectionnez d'abord une classe.", _audio_summary(state)
    try:
        feat = extract_mel_features(audio_path)
    except Exception as e:
        return state, f"Erreur extraction : {e}", _audio_summary(state)

    state = _s(state)
    state["audio_classes"] = copy.deepcopy(state["audio_classes"])
    cls = next((c for c in state["audio_classes"] if c["name"] == active_cls), None)
    if cls is None:
        return state, "Classe introuvable.", _audio_summary(state)
    cls["samples"].append(feat)
    n = len(cls["samples"])
    return state, f"Échantillon #{n} ajouté à « {active_cls} ».", _audio_summary(state)


def aud_train(epochs, lr, batch_size, units, state, progress=gr.Progress()):
    classes = state["audio_classes"]
    if len(classes) < 2:
        yield state, "Ajoutez au moins 2 classes.", None, None, ""
        return
    if any(len(c["samples"]) < 3 for c in classes):
        yield state, "Chaque classe doit avoir au moins 3 échantillons.", None, None, ""
        return

    pre_sugg = analyze_class_balance(classes)

    log_lines: list[str] = []
    loss_hist: list[float] = []
    last_val_acc: float | None = None

    for update in train_audio_model(
        classes, epochs=epochs, lr=lr, batch_size=batch_size, hidden_units=units
    ):
        tag = update[0]
        if tag == "epoch":
            _, ep, total, loss, acc, val_acc = update
            progress(ep / total, desc=f"Époque {ep}/{total}")
            line = f"[{ep:3d}/{total}]  perte={loss:.4f}  acc={acc:.2%}"
            if val_acc is not None:
                line += f"  val={val_acc:.2%}"
                last_val_acc = val_acc
            log_lines.append(line)
            loss_hist.append(loss)
            yield state, "\n".join(log_lines[-20:]), _loss_fig(loss_hist), None, ""

        elif tag == "done":
            _, model, lh, cnames, preds, actuals = update
            state = _s(state)
            state["audio_model"]       = model
            state["audio_class_names"] = cnames
            state["audio_trained"]     = True
            conf = make_confusion_figure(preds, actuals, cnames)

            train_acc = sum(p == a for p, a in zip(preds, actuals)) / max(len(preds), 1)
            post_sugg = analyze_training_results(
                lh, train_acc=train_acc, val_acc=last_val_acc, n_classes=len(cnames)
            )
            sugg_txt = format_suggestions(pre_sugg + post_sugg)
            log_lines.append("✓ Entraînement audio terminé !")
            yield state, "\n".join(log_lines[-20:]), _loss_fig(lh), conf, sugg_txt


def aud_predict(audio_path, state: dict):
    if audio_path is None or state["audio_model"] is None:
        return None
    return predict_audio(state["audio_model"], audio_path, state["audio_class_names"])


def aud_save_model(state: dict):
    if state["audio_model"] is None:
        gr.Warning("Aucun modèle audio entraîné.")
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".keras", delete=False,
                                      prefix="ml_vision_audio_")
    state["audio_model"].save(tmp.name)
    return tmp.name


# ─────────────────────────────────────────────────────────────────────────────
#  TEXT CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def txt_add_class(class_name: str, state: dict):
    state = _s(state)
    state["text_classes"] = copy.deepcopy(state["text_classes"])
    name = class_name.strip() or f"Classe {len(state['text_classes']) + 1}"
    state["text_classes"].append({"name": name, "samples": []})
    choices = _cls_choices(state["text_classes"])
    return (
        state,
        gr.Dropdown(choices=choices, value=choices[-1]),
        _text_summary(state),
        "",  # clear name input
    )


def _txt_add_chunks(chunks: list[str], active_cls: str, state: dict):
    if not active_cls:
        return state, "Sélectionnez d'abord une classe.", _text_summary(state)
    if not chunks:
        return state, "Aucun contenu à ajouter.", _text_summary(state)
    state = _s(state)
    state["text_classes"] = copy.deepcopy(state["text_classes"])
    cls = next((c for c in state["text_classes"] if c["name"] == active_cls), None)
    if cls is None:
        return state, "Classe introuvable.", _text_summary(state)
    cls["samples"].extend(chunks)
    return state, f"{len(chunks)} segment(s) ajouté(s) à « {active_cls} ».", _text_summary(state)


def txt_add_from_file(file_obj, active_cls: str, state: dict):
    if file_obj is None:
        return state, "Aucun fichier fourni.", _text_summary(state)
    path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    try:
        if path.lower().endswith(".pdf"):
            raw = extract_pdf_text(path)
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read()
        chunks = split_text_into_chunks(raw)
        return _txt_add_chunks(chunks, active_cls, state)
    except Exception as e:
        return state, f"Erreur lecture fichier : {e}", _text_summary(state)


def txt_add_from_url(url: str, active_cls: str, state: dict):
    if not url.strip():
        return state, "Entrez une URL.", _text_summary(state)
    try:
        raw = fetch_url_text(url.strip())
        chunks = split_text_into_chunks(raw)
        return _txt_add_chunks(chunks, active_cls, state)
    except Exception as e:
        return state, f"Erreur URL : {e}", _text_summary(state)


def txt_add_direct(text: str, active_cls: str, state: dict):
    if not text.strip():
        return state, "Entrez du texte.", _text_summary(state)
    chunks = split_text_into_chunks(text)
    st, msg, summ = _txt_add_chunks(chunks, active_cls, state)
    return st, msg, summ, ""  # also clear the textarea


def txt_index_knn(state: dict, progress=gr.Progress()):
    classes = state["text_classes"]
    if len(classes) < 2:
        return state, "Ajoutez au moins 2 classes.", None, ""
    if any(len(c["samples"]) < 1 for c in classes):
        return state, "Chaque classe doit avoir au moins 1 texte.", None, ""

    pre_sugg = analyze_class_balance(classes)

    progress(0, desc="Chargement USE…")
    try:
        knn_entries, class_names = build_knn_index(classes)
    except Exception as e:
        return state, f"Erreur indexation : {e}", None, ""

    progress(0.7, desc="Évaluation leave-one-out…")
    preds, actuals = knn_leave_one_out(knn_entries, class_names)
    conf = make_confusion_figure(preds, actuals, class_names)

    loo_acc = sum(p == a for p, a in zip(preds, actuals)) / max(len(preds), 1)
    post_sugg = analyze_training_results([], train_acc=loo_acc, n_classes=len(class_names))
    sugg_txt = format_suggestions(pre_sugg + post_sugg)

    state = _s(state)
    state["text_knn"]          = knn_entries
    state["text_class_names"]  = class_names
    state["text_trained"]      = True
    state["text_mode"]         = "knn"
    progress(1.0, desc="Terminé")
    return state, f"✓ {len(knn_entries)} embeddings indexés. LOO acc={loo_acc:.1%}", conf, sugg_txt


def txt_train_nn(state: dict, progress=gr.Progress()):
    if not state["text_trained"] or not state["text_knn"]:
        yield state, "Indexez d'abord les embeddings KNN.", None
        return

    log_lines: list[str] = []
    loss_hist: list[float] = []
    class_names = state["text_class_names"]

    for update in train_text_nn_model(state["text_knn"], class_names):
        tag = update[0]
        if tag == "epoch":
            _, ep, total, loss, acc, val_acc = update
            progress(ep / total, desc=f"Époque {ep}/{total}")
            line = f"[{ep:3d}/{total}]  perte={loss:.4f}  acc={acc:.2%}"
            if val_acc is not None:
                line += f"  val={val_acc:.2%}"
            log_lines.append(line)
            loss_hist.append(loss)
            yield state, "\n".join(log_lines[-20:]), None

        elif tag == "done":
            _, model, lh, cnames, preds, actuals = update
            state = _s(state)
            state["text_model"] = model
            state["text_mode"]  = "nn"
            conf = make_confusion_figure(preds, actuals, cnames)
            log_lines.append("✓ Réseau NN entraîné !")
            yield state, "\n".join(log_lines[-20:]), conf


def txt_export_json(state: dict):
    if not state["text_trained"]:
        gr.Warning("Indexez d'abord le modèle texte.")
        return None
    data = {
        "class_names": state["text_class_names"],
        "text_classes": [
            {"name": c["name"], "samples": c["samples"]}
            for c in state["text_classes"]
        ],
        "knn": [
            {
                "classIdx": e["classIdx"],
                "text":     e["text"],
                "embedding": e["embedding"].tolist(),
            }
            for e in state["text_knn"]
        ],
    }
    tmp = tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, prefix="ml_vision_text_", mode="w"
    )
    json.dump(data, tmp, ensure_ascii=False, indent=2)
    tmp.close()
    return tmp.name


def txt_load_json(file_obj, state: dict):
    if file_obj is None:
        return state, "Aucun fichier.", None, _text_summary(state)
    path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        text_classes = data.get("text_classes", [])
        class_names  = data.get("class_names", [c["name"] for c in text_classes])
        knn_entries  = [
            {
                "classIdx":  e["classIdx"],
                "text":      e["text"],
                "embedding": np.array(e["embedding"], dtype=np.float32),
            }
            for e in data.get("knn", [])
        ]
        state = _s(state)
        state["text_classes"]     = text_classes
        state["text_class_names"] = class_names
        state["text_knn"]         = knn_entries
        state["text_trained"]     = True
        state["text_mode"]        = "knn"
        choices = _cls_choices(text_classes)
        return (
            state,
            f"✓ Modèle texte chargé : {len(knn_entries)} embeddings, {len(class_names)} classes.",
            gr.Dropdown(choices=choices, value=choices[0] if choices else None),
            _text_summary(state),
        )
    except Exception as e:
        return state, f"Erreur chargement JSON : {e}", None, _text_summary(state)


# ── tf_flowers (Image) ────────────────────────────────────────────────────────

def flowers_status_str() -> str:
    if not flowers_prepared():
        return "Dataset non téléchargé."
    counts = flowers_counts()
    lines = [f"✓ tf_flowers prêt — {sum(counts.values())} images"]
    for name, n in counts.items():
        lines.append(f"  • {name}: {n}")
    return "\n".join(lines)


def flowers_prepare_cb():
    yield "Téléchargement tf_flowers (~210 MB la première fois)…"
    try:
        msg = flowers_download()
        yield msg
    except Exception as e:
        yield f"Erreur : {e}"


def flowers_to_image_cb(state: dict):
    if not flowers_prepared():
        return state, gr.Dropdown(), _image_summary(state), "Téléchargez d'abord tf_flowers."
    import copy
    classes = flowers_load()
    state = _s(state)
    state["image_classes"] = copy.deepcopy(classes)
    state["image_trained"] = False
    state["image_model"]   = None
    choices = _cls_choices(classes)
    n = sum(len(c["samples"]) for c in classes)
    return (
        state,
        gr.Dropdown(choices=choices, value=choices[0] if choices else None),
        _image_summary(state),
        f"✓ {n} images chargées ({len(classes)} classes) — prêt à entraîner !",
    )


# ── Speech Commands (Audio) ───────────────────────────────────────────────────

def speech_status_str() -> str:
    if not speech_prepared():
        return "Dataset non téléchargé."
    lines = ["✓ Speech Commands prêt"]
    for name in SPEECH_CLASS_NAMES:
        from pathlib import Path
        p = Path(__file__).parent / "datasets" / "speech_data" / f"{name}.npz"
        if p.exists():
            import numpy as np
            n = np.load(p)["features"].shape[0]
            lines.append(f"  • {name}: {n} clips")
    return "\n".join(lines)


def speech_prepare_cb():
    yield "Téléchargement Speech Commands (~150 MB, premiers shards uniquement)…"
    try:
        msg = speech_download()
        yield msg
    except Exception as e:
        yield f"Erreur : {e}"


def speech_to_audio_cb(state: dict):
    if not speech_prepared():
        return state, gr.Dropdown(), _audio_summary(state), "Téléchargez d'abord Speech Commands."
    import copy
    classes = speech_load()
    state = _s(state)
    state["audio_classes"] = copy.deepcopy(classes)
    state["audio_trained"] = False
    state["audio_model"]   = None
    choices = _cls_choices(classes)
    n = sum(len(c["samples"]) for c in classes)
    return (
        state,
        gr.Dropdown(choices=choices, value=choices[0] if choices else None),
        _audio_summary(state),
        f"✓ {n} clips chargés ({len(classes)} classes) — prêt à entraîner !",
    )


# ── AG News (Texte) ───────────────────────────────────────────────────────────

def agnews_status_str() -> str:
    if not agnews_prepared():
        return "Dataset non téléchargé."
    counts = agnews_counts()
    lines = [f"✓ AG News prêt — {sum(counts.values())} articles"]
    for name, n in counts.items():
        lines.append(f"  • {name}: {n}")
    return "\n".join(lines)


def agnews_prepare_cb():
    yield "Téléchargement AG News (~31 MB la première fois)…"
    try:
        msg = agnews_download()
        yield msg
    except Exception as e:
        yield f"Erreur : {e}"


def agnews_to_text_cb(state: dict):
    if not agnews_prepared():
        return state, gr.Dropdown(), _text_summary(state), "Téléchargez d'abord AG News."
    import copy
    classes = agnews_load()
    state = _s(state)
    state["text_classes"]  = copy.deepcopy(classes)
    state["text_trained"]  = False
    state["text_knn"]      = []
    state["text_model"]    = None
    choices = _cls_choices(classes)
    n = sum(len(c["samples"]) for c in classes)
    return (
        state,
        gr.Dropdown(choices=choices, value=choices[0] if choices else None),
        _text_summary(state),
        f"✓ {n} articles chargés ({len(classes)} classes) — prêt à indexer !",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  COURBES D'APPRENTISSAGE CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def img_lc_cb(state: dict):
    classes = state["image_classes"]
    if len(classes) < 2:
        return None, "Ajoutez au moins 2 classes avec des images."
    if any(len(c["samples"]) < 2 for c in classes):
        return None, "Chaque classe doit avoir au moins 2 images."
    try:
        fig, diag = image_learning_curve(classes)
        return fig, diag
    except Exception as e:
        return None, f"Erreur : {e}"


def aud_lc_cb(state: dict):
    classes = state["audio_classes"]
    if len(classes) < 2:
        return None, "Ajoutez au moins 2 classes avec des échantillons."
    if any(len(c["samples"]) < 2 for c in classes):
        return None, "Chaque classe doit avoir au moins 2 échantillons."
    try:
        fig, diag = audio_learning_curve(classes)
        return fig, diag
    except Exception as e:
        return None, f"Erreur : {e}"


def txt_lc_cb(state: dict):
    if not state["text_trained"] or not state["text_knn"]:
        return None, "Indexez d'abord les embeddings KNN."
    try:
        fig, diag = text_learning_curve(state["text_knn"], state["text_class_names"])
        return fig, diag
    except Exception as e:
        return None, f"Erreur : {e}"


def cd_lc_cb():
    try:
        fig, diag = cats_dogs_learning_curve()
        return fig, diag
    except Exception as e:
        return None, f"Erreur : {e}"


# ─────────────────────────────────────────────────────────────────────────────
#  PREDICTION CALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def pred_classify(image, audio_path, modality: str, state: dict):
    if modality == "Image":
        if image is None:
            gr.Warning("Fournissez une image.")
            return None
        if state["image_model"] is None:
            gr.Warning("Entraînez d'abord le modèle image.")
            return None
        return predict_image(state["image_model"], image, state["image_class_names"])
    else:
        if audio_path is None:
            gr.Warning("Enregistrez ou importez un audio.")
            return None
        if state["audio_model"] is None:
            gr.Warning("Entraînez d'abord le modèle audio.")
            return None
        return predict_audio(state["audio_model"], audio_path, state["audio_class_names"])


# ─────────────────────────────────────────────────────────────────────────────
#  CHAT CALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def _classify_text_query(query: str, state: dict) -> tuple[dict, str]:
    """Returns (scores_dict, excerpts_str)."""
    if not state["text_trained"]:
        return {}, "Modèle texte non entraîné."
    emb = embed_single(query)
    if state["text_mode"] == "nn" and state["text_model"] is not None:
        scores = classify_with_nn(state["text_model"], emb, state["text_class_names"])
    else:
        scores = classify_knn(emb, state["text_knn"], state["text_class_names"])

    # Top-3 similar excerpts from winning class
    winning = max(scores, key=scores.get)
    ci = state["text_class_names"].index(winning)
    sims = [
        (e["text"], float(np.dot(emb, e["embedding"]) /
                         (np.linalg.norm(emb)*np.linalg.norm(e["embedding"])+1e-8)))
        for e in state["text_knn"] if e["classIdx"] == ci
    ]
    sims.sort(key=lambda x: x[1], reverse=True)
    excerpts_lines = []
    for txt, sim in sims[:3]:
        short = txt[:120] + ("…" if len(txt) > 120 else "")
        excerpts_lines.append(f"• ({sim:.2f}) {short}")
    excerpts = "\n".join(excerpts_lines)

    return scores, excerpts


def chat_send(message: str, history: list, modality: str, state: dict):
    if not message.strip():
        return history, ""

    if modality in ("Image", "Audio"):
        reply = (
            f"Pour tester le modèle {modality.lower()}, "
            f"utilisez l'onglet **Prédiction** et sélectionnez la modalité {modality}."
        )
        history = history + [[message, reply]]
        return history, ""

    if not state["text_trained"]:
        reply = "Le modèle texte n'est pas encore entraîné. Allez dans l'onglet **Texte** pour l'entraîner."
        history = history + [[message, reply]]
        return history, ""

    try:
        scores, excerpts = _classify_text_query(message, state)
    except Exception as e:
        history = history + [[message, f"Erreur : {e}"]]
        return history, ""

    # Format response
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_class, top_prob = sorted_scores[0]
    lines = [f"**Classe détectée : {top_class}** ({top_prob:.1%})\n"]
    for cls, prob in sorted_scores:
        bar_filled = int(prob * 20)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        lines.append(f"`{bar}` {cls} — {prob:.1%}")
    if excerpts:
        lines.append("\n**Extraits similaires :**")
        lines.append(excerpts)
    reply = "\n".join(lines)
    history = history + [[message, reply]]
    return history, ""


# ─────────────────────────────────────────────────────────────────────────────
#  CHATS & CHIENS CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def cd_data_status() -> str:
    if not is_prepared():
        return "Dataset non téléchargé."
    counts = split_counts()
    return (
        f"✓ Dataset prêt — {counts.get('train',0)} train / "
        f"{counts.get('val',0)} val / {counts.get('test',0)} test"
    )


def cd_download():
    yield "Téléchargement en cours (~786 MB la première fois)…"
    try:
        msg = download_and_prepare()
        yield msg
    except Exception as e:
        yield f"Erreur : {e}"


def cd_train_ml():
    if not is_prepared():
        yield "Données non disponibles. Téléchargez d'abord le dataset.", None, None
        return

    X_tr, y_tr = load_split("train")
    X_v,  y_v  = load_split("val")
    X_te, y_te = load_split("test")

    log = ""
    results = None
    gen = train_ml_models(X_tr, y_tr, X_v, y_v, X_te, y_te)
    for item in gen:
        if isinstance(item, str):
            log += item + "\n"
            yield log, None, None
        elif isinstance(item, dict):
            results = item

    if results is None:
        yield log + "Erreur : pas de résultats.", None, None
        return

    svm = results["svm"]
    rf  = results["rf"]
    log += (
        f"\n──────────────────────────────\n"
        f"SVM   test accuracy : {svm['test_acc']:.3f}\n"
        f"RF    test accuracy : {rf['test_acc']:.3f}\n\n"
        f"=== SVM — Rapport de classification ===\n{svm['report']}\n"
        f"=== Random Forest — Rapport de classification ===\n{rf['report']}"
    )
    fig_svm = make_confusion_figure(svm["preds"], svm["actuals"], CD_CLASS_NAMES)
    fig_rf  = make_confusion_figure(rf["preds"],  rf["actuals"],  CD_CLASS_NAMES)
    yield log, fig_svm, fig_rf


def cd_train_dl(finetune_epochs: int, batch_size: int):
    if not is_prepared():
        yield "Données non disponibles. Téléchargez d'abord le dataset.", None, None
        return

    X_tr, y_tr = load_split("train")
    X_v,  y_v  = load_split("val")
    X_te, y_te = load_split("test")

    log     = ""
    results = None
    try:
        gen = train_dl_model(X_tr, y_tr, X_v, y_v, X_te, y_te,
                             finetune_epochs=finetune_epochs, batch_size=batch_size)
        for item in gen:
            if item[0] == "phase":
                _, num, p1, p2 = item
                label = ("🔒 Base gelée" if num == 1
                         else "🔓 Fine-tuning (top 30 couches)")
                log += f"\n── Phase {num} — {label} ──\n"
                yield log, None, None
            elif item[0] == "epoch":
                _, ep, total, tl, ta, vl, va = item
                log += (f"Époque {ep:>3}/{total} — "
                        f"loss {tl:.4f}  acc {ta:.3f}  "
                        f"val_loss {vl:.4f}  val_acc {va:.3f}\n")
                yield log, None, None
            elif item[0] == "done":
                results = item[1]
    except Exception as e:
        yield log + f"\n❌ Erreur : {e}", None, None
        return

    if results is None:
        yield log + "Erreur : pas de résultats.", None, None
        return

    log += (
        f"\n──────────────────────────────\n"
        f"CNN test accuracy : {results['test_acc']:.3f}\n\n"
        f"=== CNN — Rapport de classification ===\n{results['report']}"
    )
    fig_curves = _cd_train_fig(results["loss_hist"])
    fig_cm     = make_confusion_figure(results["preds"], results["actuals"], CD_CLASS_NAMES)
    yield log, fig_curves, fig_cm


def cd_predict(img):
    if img is None:
        return "Chargez une image.", None, None, None, None

    # Normalisation PIL → float32 numpy [0,1] à la résolution d'entraînement
    pil = Image.fromarray(img).convert("RGB").resize((CD_IMG_SIZE, CD_IMG_SIZE))
    arr = np.array(pil, dtype=np.float32) / 255.0

    ml_status = ml_models_trained()
    results   = {}

    if ml_status["svm"]:
        results["SVM"] = predict_ml(arr, "svm")
    if ml_status["rf"]:
        results["RF"]  = predict_ml(arr, "rf")
    if dl_model_trained():
        results["CNN"] = predict_dl(arr)

    if not results:
        return "Aucun modèle entraîné. Entraînez SVM/RF et/ou le CNN d'abord.", None, None, None, None

    # ── Ensemble pondéré ──────────────────────────────────────────────────────
    # CNN (transfer learning) reçoit plus de poids car bien plus précis
    WEIGHTS = {"CNN": 0.60, "SVM": 0.20, "RF": 0.20}
    ensemble: dict[str, float] = {cls: 0.0 for cls in CD_CLASS_NAMES}
    total_w = sum(WEIGHTS[k] for k in results)
    for name, probs in results.items():
        w = WEIGHTS[name] / total_w   # renormalise si certains modèles manquent
        for cls in CD_CLASS_NAMES:
            ensemble[cls] += w * probs.get(cls, 0.0)

    def _label(probs: dict) -> dict | None:
        return {f"{k} ({v:.1%})": v for k, v in probs.items()} if probs else None

    return (
        "Prédictions effectuées.",
        _label(results.get("SVM", {})),
        _label(results.get("RF",  {})),
        _label(results.get("CNN", {})),
        _label(ensemble),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PRIX MAISON — CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def price_generate_dataset_cb(n_samples: int, noise_level: float, state: dict):
    """Génère et affiche le dataset synthétique."""
    X, y = generate_dataset(int(n_samples), float(noise_level))
    st = _s(state)
    st["price_X"] = X
    st["price_y"] = y
    fig = make_dataset_figure(X, y)
    msg = (
        f"Dataset généré : {len(y)} maisons\n"
        f"Surface : {X.min():.0f} – {X.max():.0f} m²\n"
        f"Prix    : {y.min():.0f} – {y.max():.0f} k€\n"
        f"Relation réelle : prix ≈ 2.5 × surface + 50 (+ bruit)"
    )
    return st, fig, msg


def price_train_cb(n_samples: int, noise_level: float, state: dict):
    """Entraîne la régression linéaire et retourne métriques + figure."""
    model, metrics, X, y, X_test, y_test, y_pred = train_price_model(
        int(n_samples), float(noise_level)
    )
    st = _s(state)
    st["price_model"]   = model
    st["price_metrics"] = metrics
    st["price_X"]       = X
    st["price_y"]       = y
    st["price_X_test"]  = X_test
    st["price_y_test"]  = y_test
    st["price_y_pred"]  = y_pred

    fig = make_regression_figure(model, X, y, X_test, y_test, y_pred)

    summary = (
        "📊 **Résultats du modèle**\n\n"
        f"**Équation apprise** : Prix = {metrics['coef']:.2f} × Surface + {metrics['intercept']:.1f} k€\n"
        f"*(Relation réelle : 2.50 × Surface + 50.0)*\n\n"
        f"| Métrique | Train | Test |\n"
        f"|---------|-------|------|\n"
        f"| R²      | {metrics['R²_train']:.3f} | {metrics['R²_test']:.3f} |\n"
        f"| RMSE (k€)| {metrics['RMSE_train']:.1f} | {metrics['RMSE_test']:.1f} |\n"
        f"| MAE  (k€)| {metrics['MAE_train']:.1f} | {metrics['MAE_test']:.1f} |\n\n"
        f"Entraîné sur {metrics['n_train']} exemples, testé sur {metrics['n_test']}."
    )
    return st, fig, summary


def price_predict_cb(surface: float, state: dict):
    """Prédit le prix pour une surface donnée."""
    model = state.get("price_model")
    if model is None:
        return state, None, "⚠️ Entraîner le modèle d'abord."

    price = predict_price(model, float(surface))
    X = state["price_X"]
    y = state["price_y"]
    X_test = state["price_X_test"]
    y_test = state["price_y_test"]
    y_pred = state["price_y_pred"]

    fig = make_regression_figure(model, X, y, X_test, y_test, y_pred,
                                 highlight_x=float(surface), highlight_y=price)
    msg = f"🏠 Surface : {surface:.0f} m²  →  Prix estimé : **{price:.0f} k€** ({price*1000:.0f} €)"
    return state, fig, msg


def price_lc_cb(n_samples: float, noise_level: float, state: dict):
    """Courbe d'apprentissage — régression linéaire (R²)."""
    model = state.get("price_model")
    if model is None:
        return None, "⚠️ Entraînez d'abord le modèle pour générer la courbe."
    fig, diag = price_learning_curve(int(n_samples), float(noise_level))
    return fig, diag


# ─────────────────────────────────────────────────────────────────────────────
#  MNIST — CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def mnist_load_cb(state: dict):
    """Charge le dataset MNIST et affiche une grille d'exemples."""
    try:
        (X_train, y_train), (X_test, y_test) = load_mnist()
    except Exception as e:
        return state, None, f"Erreur lors du chargement : {e}"

    st = _s(state)
    st["mnist_X_train"] = X_train
    st["mnist_y_train"] = y_train
    st["mnist_X_test"]  = X_test
    st["mnist_y_test"]  = y_test

    fig = make_sample_grid(X_train, y_train, n=20)
    msg = (
        f"✓ MNIST chargé\n"
        f"  Train : {len(X_train):,} images\n"
        f"  Test  : {len(X_test):,} images\n"
        f"  Classes : 10 chiffres (0–9)\n"
        f"  Format  : 28×28 pixels, niveaux de gris"
    )
    return st, fig, msg


def mnist_train_cb(epochs, batch_size, lr, state: dict, progress=gr.Progress()):
    """Generator : entraîne le CNN et streame log + courbes en temps réel."""
    X_train = state.get("mnist_X_train")
    if X_train is None:
        yield state, "⚠️ Chargez d'abord le dataset.", None, None, ""
        return

    X_test  = state["mnist_X_test"]
    y_train = state["mnist_y_train"]
    y_test  = state["mnist_y_test"]

    log_lines: list[str] = []

    for update in train_cnn_model(
        X_train, y_train, X_test, y_test,
        epochs=epochs, batch_size=batch_size, lr=lr,
    ):
        tag = update[0]
        if tag == "epoch":
            _, ep, total, loss, acc, val_loss, val_acc = update
            progress(ep / total, desc=f"Époque {ep}/{total}")
            log_lines.append(
                f"[{ep:3d}/{total}]  perte={loss:.4f}  acc={acc:.2%}"
                f"  │  val_perte={val_loss:.4f}  val_acc={val_acc:.2%}"
            )
            yield state, "\n".join(log_lines[-20:]), None, None, ""

        elif tag == "done":
            _, model, history = update
            # Évaluation complète sur le jeu test
            eval_res = evaluate_cnn(model, X_test, y_test)
            cm_fig   = make_confusion_10(eval_res["preds"], eval_res["actuals"])
            curve_fig = make_training_curves(history)

            st = _s(state)
            st["mnist_model"]   = model
            st["mnist_trained"] = True

            test_acc = eval_res["test_acc"]
            log_lines.append(
                f"✓ Entraînement terminé — Précision test : {test_acc:.2%}"
            )
            summary = (
                f"## Résultats\n\n"
                f"| Métrique   | Valeur |\n"
                f"|------------|--------|\n"
                f"| Précision test | **{test_acc:.2%}** |\n"
                f"| Perte test     | {eval_res['test_loss']:.4f} |\n"
                f"| Images test    | {len(X_test):,} |\n\n"
                f"*Référence : un CNN standard atteint ~99% sur MNIST.*"
            )
            yield st, "\n".join(log_lines[-20:]), curve_fig, cm_fig, summary


def mnist_predict_cb(sketch, state: dict):
    """Prédit le chiffre dessiné ou uploaded."""
    model = state.get("mnist_model")
    if model is None:
        return None, "⚠️ Entraînez d'abord le modèle."
    if sketch is None:
        return None, "⚠️ Dessinez ou importez un chiffre."

    result = predict_digit(model, sketch)
    if result is None:
        return None, "⚠️ Image non reconnue. Vérifiez le canvas."

    pred, probs = result
    bar = {str(k): v for k, v in sorted(probs.items(), key=lambda x: -x[1])}
    msg = f"Chiffre détecté : **{pred}** (confiance : {probs[str(pred)]:.1%})"
    return bar, msg


def mnist_lc_cb(state: dict):
    """Courbe d'apprentissage MNIST (Régression Logistique sklearn, ~20s)."""
    X_train = state.get("mnist_X_train")
    if X_train is None:
        return None, "⚠️ Chargez d'abord le dataset MNIST."
    fig, diag = mnist_learning_curve(
        X_train, state["mnist_y_train"],
        state["mnist_X_test"], state["mnist_y_test"],
    )
    return fig, diag


# ─────────────────────────────────────────────────────────────────────────────
#  CLUSTERING — CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def clustering_elbow_cb(k_max: int, state: dict, progress=gr.Progress()):
    """Calcule la méthode du coude (inertie + silhouette) sur MNIST."""
    X_train = state.get("mnist_X_train")
    if X_train is None:
        return None, "⚠️ Chargez d'abord le dataset MNIST (onglet Chiffres MNIST)."
    progress(0.1, desc="Préparation des données…")
    X_flat = flatten(X_train)
    progress(0.2, desc="Calcul K-Means pour K = 2 … " + str(int(k_max)) + "…")
    ks, inertias, silhouettes, best_k = run_elbow(X_flat, k_max=int(k_max), n_elbow=3000)
    progress(1.0, desc="Terminé")
    fig = make_elbow_figure(ks, inertias, silhouettes, best_k=best_k)
    msg = (
        f"✓ Méthode du coude calculée ({len(ks)} valeurs de K)\n"
        f"K recommandé par silhouette maximale : **K = {best_k}**\n"
        f"Silhouette max : {max(silhouettes):.4f}"
    )
    return fig, msg, int(best_k)


def clustering_run_cb(k: int, n_tsne: int, state: dict, progress=gr.Progress()):
    """Lance K-Means + t-SNE sur MNIST."""
    X_train = state.get("mnist_X_train")
    y_train = state.get("mnist_y_train")
    if X_train is None:
        yield state, None, "⚠️ Chargez d'abord le dataset MNIST.", ""
        return

    progress(0.05, desc="Aplatissage des images…")
    X_flat = flatten(X_train)
    k = int(k)
    n_tsne = int(n_tsne)

    progress(0.15, desc=f"K-Means K={k} en cours…")
    result = run_kmeans(X_flat, k=k, true_labels=y_train, n_kmeans=5000)

    progress(0.55, desc=f"t-SNE (n={min(n_tsne, len(result['X_sub']))} points) …")
    X_2d, tsne_idx = run_tsne(result["X_sub"], n_tsne=n_tsne)

    labels_tsne = result["labels"][tsne_idx]
    true_tsne   = y_train[result["idx"]][tsne_idx] if y_train is not None else None

    progress(0.90, desc="Génération des figures…")
    tsne_fig = make_tsne_figure(X_2d, labels_tsne, true_tsne, k=k)

    st = _s(state)
    st["clustering_k"]   = k
    st["clustering_sil"] = result["silhouette"]
    st["clustering_ari"] = result["ari"]

    ari_str = f"  │  ARI = {result['ari']:.4f}" if result["ari"] is not None else ""
    msg = (
        f"✓ K-Means terminé (K={k})\n"
        f"Inertie     : {result['inertia']:.1f}\n"
        f"Silhouette  : {result['silhouette']:.4f}{ari_str}\n\n"
        f"t-SNE : {len(X_2d)} points visualisés\n\n"
        "**ARI (Adjusted Rand Index)** mesure l'accord entre les clusters trouvés\n"
        "et les vraies étiquettes (0 = aléatoire, 1 = parfait)."
    )
    progress(1.0, desc="Terminé !")
    yield st, tsne_fig, msg


# ─────────────────────────────────────────────────────────────────────────────
#  RÉGRESSION MULTIVARIÉE — CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def multireg_train_cb(
    feat_selection: list[str],
    model_type_radio: str,
    state: dict,
    progress=gr.Progress(),
):
    """Entraîne la régression multivariée sur California Housing."""
    if not feat_selection:
        return state, None, None, "⚠️ Sélectionnez au moins une feature."

    progress(0.1, desc="Chargement California Housing…")
    mtype = "ridge" if "Ridge" in model_type_radio else "linear"

    progress(0.3, desc="Entraînement du modèle…")
    try:
        pipe, metrics, X_all, y_all, X_test, y_test, y_pred = train_multivariate_model(
            feat_selection, model_type=mtype
        )
    except Exception as e:
        return state, None, None, f"❌ Erreur : {e}"

    progress(0.75, desc="Génération des figures…")
    fig_imp = make_importance_figure(metrics["coefs"], metrics["features"])
    fig_sc  = make_scatter_residuals_figure(y_test, y_pred, metrics["R²_test"])

    st = _s(state)
    st["multireg_pipe"]     = pipe
    st["multireg_metrics"]  = metrics
    st["multireg_features"] = metrics["features"]

    n_feat = len(metrics["features"])
    summary = (
        f"## Résultats — Régression {metrics['model_type'].capitalize()}\n\n"
        f"**Features utilisées** : {n_feat} / {len(ALL_FEATURE_NAMES)}\n\n"
        f"| Métrique  | Train  | Test  |\n"
        f"|-----------|--------|-------|\n"
        f"| **R²**    | {metrics['R²_train']:.3f} | **{metrics['R²_test']:.3f}** |\n"
        f"| RMSE (k$) | {metrics['RMSE_train']:.1f}  | {metrics['RMSE_test']:.1f}  |\n"
        f"| MAE  (k$) | {metrics['MAE_train']:.1f}  | {metrics['MAE_test']:.1f}  |\n\n"
        f"Entraîné sur **{metrics['n_train']:,}** exemples, testé sur **{metrics['n_test']:,}**.\n\n"
        f"*California Housing : 20 640 quartiers californiens (recensement 1990).*"
    )
    progress(1.0)
    return st, fig_imp, fig_sc, summary


# ─────────────────────────────────────────────────────────────────────────────
#  GRAD-CAM — CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def gradcam_cd_cb(img_input, progress=gr.Progress()):
    """Grad-CAM sur le modèle EfficientNetB0 Chats & Chiens."""
    import tensorflow as tf

    from cats_vs_dogs.dl_model import CNN_PATH

    if img_input is None:
        return None, "⚠️ Chargez une image."
    if not CNN_PATH.exists():
        return None, "⚠️ Entraînez d'abord le CNN Chats & Chiens."

    progress(0.2, desc="Chargement du modèle…")
    try:
        model = tf.keras.models.load_model(CNN_PATH)
    except Exception as e:
        return None, f"❌ Erreur chargement modèle : {e}"

    img_size = model.input_shape[1]
    from PIL import Image as PILImage
    pil = PILImage.fromarray(img_input).convert("RGB").resize((img_size, img_size))
    arr = np.array(pil, dtype=np.float32) / 255.0
    img_batch = arr[np.newaxis]

    progress(0.5, desc="Calcul Grad-CAM…")
    try:
        heatmap, pred_idx, confidence = compute_gradcam(model, img_batch)
    except Exception as e:
        return None, f"❌ Erreur Grad-CAM : {e}"

    from cats_vs_dogs.dl_model import CLASS_NAMES as CD_CLASS_NAMES
    class_name = CD_CLASS_NAMES[pred_idx] if pred_idx < len(CD_CLASS_NAMES) else str(pred_idx)

    progress(0.85, desc="Génération des figures…")
    overlay = overlay_heatmap(arr, heatmap)
    fig = make_gradcam_figure(
        arr, heatmap, overlay, class_name, confidence,
        title="Grad-CAM — EfficientNetB0 Chats & Chiens",
    )
    progress(1.0)
    msg = f"Prédiction : **{class_name}** (confiance : {confidence:.1%})"
    return fig, msg


def gradcam_mnist_cb(sketch, state: dict, progress=gr.Progress()):
    """Grad-CAM sur le CNN MNIST (modèle en mémoire)."""
    model = state.get("mnist_model")
    if model is None:
        return None, "⚠️ Entraînez d'abord le CNN MNIST."
    if sketch is None:
        return None, "⚠️ Dessinez un chiffre dans le canvas."

    progress(0.3, desc="Prétraitement…")
    result = predict_digit(model, sketch)
    if result is None:
        return None, "⚠️ Image non reconnue."

    pred, probs = result
    # Récupérer le tableau numpy 28×28
    from core.mnist_model import preprocess_digit_image
    img_batch = preprocess_digit_image(sketch)
    if img_batch is None:
        return None, "⚠️ Impossible de prétraiter l'image."

    progress(0.55, desc="Calcul Grad-CAM…")
    try:
        heatmap, pred_idx, confidence = compute_gradcam(model, img_batch, class_idx=int(pred))
    except Exception as e:
        return None, f"❌ Erreur Grad-CAM : {e}"

    # Convertir img_batch en RGB pour l'affichage
    img_2d = img_batch[0, :, :, 0]
    img_rgb = np.stack([img_2d] * 3, axis=-1)

    progress(0.85, desc="Génération des figures…")
    overlay = overlay_heatmap(img_rgb, heatmap)
    fig = make_gradcam_figure(
        img_rgb, heatmap, overlay, str(pred), confidence,
        title="Grad-CAM — CNN MNIST",
    )
    progress(1.0)
    return fig, f"Chiffre : **{pred}** — confiance : {confidence:.1%}"


# ─────────────────────────────────────────────────────────────────────────────
#  DASHBOARD — CALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def dashboard_refresh_cb(state: dict):
    """Construit le tableau comparatif et le graphe des métriques."""
    from cats_vs_dogs.dl_model import model_trained as dl_model_trained
    from cats_vs_dogs.ml_model import models_trained as ml_models_trained

    def _badge(ok: bool) -> str:
        return "✅ Entraîné" if ok else "⬜ Non entraîné"

    ml_status = ml_models_trained()

    rows = [
        {
            "Modèle": "🖼️ Image (MobileNetV2)",
            "Algorithme": "Transfer Learning",
            "Données": "Personnalisées",
            "Métrique": "Acc (train)",
            "Valeur": "—",
            "Statut": _badge(state.get("image_trained", False)),
        },
        {
            "Modèle": "🔊 Audio (Mel + Dense)",
            "Algorithme": "Réseau Dense",
            "Données": "Personnalisées",
            "Métrique": "Acc (train)",
            "Valeur": "—",
            "Statut": _badge(state.get("audio_trained", False)),
        },
        {
            "Modèle": "📝 Texte (USE + KNN)",
            "Algorithme": "K-NN cosinus",
            "Données": "Personnalisées",
            "Métrique": "LOO Acc",
            "Valeur": "—",
            "Statut": _badge(state.get("text_trained", False)),
        },
        {
            "Modèle": "🐱 SVM Chats & Chiens",
            "Algorithme": "SVM (HOG)",
            "Données": "TF-Dogs (~786 MB)",
            "Métrique": "Test Acc",
            "Valeur": "—",
            "Statut": _badge(ml_status.get("svm", False)),
        },
        {
            "Modèle": "🐶 RF Chats & Chiens",
            "Algorithme": "Random Forest",
            "Données": "TF-Dogs (~786 MB)",
            "Métrique": "Test Acc",
            "Valeur": "—",
            "Statut": _badge(ml_status.get("rf", False)),
        },
        {
            "Modèle": "🧠 CNN Chats & Chiens",
            "Algorithme": "EfficientNetB0",
            "Données": "TF-Dogs (~786 MB)",
            "Métrique": "Test Acc",
            "Valeur": "—",
            "Statut": _badge(dl_model_trained()),
        },
        {
            "Modèle": "🔢 CNN MNIST",
            "Algorithme": "CNN (2 Conv)",
            "Données": "MNIST (70k img)",
            "Métrique": "Test Acc",
            "Valeur": "—",
            "Statut": _badge(state.get("mnist_trained", False)),
        },
        {
            "Modèle": "🏠 Régression 1D",
            "Algorithme": "Linéaire (sklearn)",
            "Données": "Synthétique",
            "Métrique": "R² test",
            "Valeur": (
                f"{state['price_metrics']['R²_test']:.3f}"
                if state.get("price_metrics") else "—"
            ),
            "Statut": _badge(state.get("price_model") is not None),
        },
        {
            "Modèle": "🏘️ Régression Multi",
            "Algorithme": state.get("multireg_metrics", {}).get("model_type", "—").capitalize() if state.get("multireg_metrics") else "—",
            "Données": "California Housing",
            "Métrique": "R² test",
            "Valeur": (
                f"{state['multireg_metrics']['R²_test']:.3f}"
                if state.get("multireg_metrics") else "—"
            ),
            "Statut": _badge(state.get("multireg_pipe") is not None),
        },
        {
            "Modèle": "🔵 K-Means",
            "Algorithme": "K-Means",
            "Données": "MNIST (embed)",
            "Métrique": "Silhouette / ARI",
            "Valeur": (
                f"{state['clustering_sil']:.4f} / {state['clustering_ari']:.4f}"
                if state.get("clustering_sil") is not None and state.get("clustering_ari") is not None
                else (f"{state['clustering_sil']:.4f}" if state.get("clustering_sil") is not None else "—")
            ),
            "Statut": _badge(state.get("clustering_k") is not None),
        },
    ]

    # ── Tableau HTML ────────────────────────────────────────────
    header = ["Modèle", "Algorithme", "Données", "Métrique", "Valeur", "Statut"]
    tbl_rows = "\n".join(
        "<tr>" + "".join(f"<td>{row[h]}</td>" for h in header) + "</tr>"
        for row in rows
    )
    thead = "<tr>" + "".join(f"<th>{h}</th>" for h in header) + "</tr>"
    html = f"""
    <style>
    .dash-table {{
        width: 100%; border-collapse: collapse; font-size: 0.88rem;
        background: #111827; border-radius: 10px; overflow: hidden;
    }}
    .dash-table th {{
        background: #1f2937; color: #a855f7; padding: 8px 12px;
        text-align: left; font-weight: 600; border-bottom: 2px solid #374151;
    }}
    .dash-table td {{
        color: #d1d5db; padding: 7px 12px; border-bottom: 1px solid #1f2937;
    }}
    .dash-table tr:hover td {{ background: #1f2937; }}
    </style>
    <table class="dash-table">
    <thead>{thead}</thead>
    <tbody>{tbl_rows}</tbody>
    </table>
    """

    # ── Figure comparative (R² uniquement pour les modèles qui en ont) ──────
    metric_rows = [r for r in rows if r["Valeur"] != "—" and "/" not in r["Valeur"]]
    fig = None
    if metric_rows:
        fig, ax = plt.subplots(figsize=(8, max(3, len(metric_rows) * 0.6 + 1)))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")
        ax.grid(alpha=0.2, color="#444", axis="x")

        names   = [r["Modèle"].split(" ", 1)[-1] for r in metric_rows]
        values  = [float(r["Valeur"]) for r in metric_rows]
        colors  = ["#a855f7", "#22d3ee", "#4ade80", "#f97316"]
        bars = ax.barh(names, values,
                       color=[colors[i % len(colors)] for i in range(len(names))],
                       alpha=0.85, edgecolor="#333", linewidth=0.5)
        for bar, v in zip(bars, values):
            ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", color="white", fontsize=9)
        ax.set_xlabel("R² (test)", color="white", fontsize=9)
        ax.set_title("Comparaison R² — modèles de régression", color="white", fontsize=11, fontweight="bold")
        ax.set_yticklabels(names, color="white", fontsize=9)
        ax.set_xlim(0, 1.05)
        fig.tight_layout()

    n_trained = sum(1 for r in rows if "✅" in r["Statut"])
    summary = f"**{n_trained} / {len(rows)}** modèles entraînés dans cette session."
    return html, fig, summary


# ─────────────────────────────────────────────────────────────────────────────
#  GRADIO UI
# ─────────────────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        title="ML Vision Studio",
        theme=gr.themes.Default(
            primary_hue="purple",
            neutral_hue="slate",
        ),
        css="""
            /* ── Layout ─────────────────────────────────── */
            footer { display: none !important; }
            .gradio-container { max-width: 1440px !important; margin: 0 auto !important; }

            /* ── Header ─────────────────────────────────── */
            .hero-block {
                background: linear-gradient(135deg, #1e1b4b 0%, #2d1b69 50%, #1a1a2e 100%) !important;
                border: 1px solid #4c1d95 !important;
                border-radius: 14px !important;
                padding: 1.4rem 2rem !important;
                margin-bottom: 0.5rem !important;
                box-shadow: 0 4px 24px rgba(124, 58, 237, 0.25) !important;
            }
            .hero-block h1 { color: #e9d5ff !important; font-size: 1.9rem !important; margin: 0 !important; }
            .hero-block p  { color: #c4b5fd !important; margin: 0.25rem 0 0 !important; font-size: 0.95rem !important; }

            /* ── Cards / Groups ──────────────────────────── */
            .gr-group {
                background: #111827 !important;
                border: 1px solid #1f2937 !important;
                border-radius: 12px !important;
            }

            /* ── Accordions ──────────────────────────────── */
            .gr-accordion { border-radius: 10px !important; border: 1px solid #1f2937 !important; }

            /* ── Primary buttons ─────────────────────────── */
            .gr-button-primary,
            button[variant="primary"],
            button.primary {
                background: linear-gradient(135deg, #6d28d9, #a855f7) !important;
                border: none !important;
                transition: transform 0.15s ease, box-shadow 0.15s ease !important;
            }
            .gr-button-primary:hover,
            button[variant="primary"]:hover {
                transform: translateY(-1px) !important;
                box-shadow: 0 4px 16px rgba(168, 85, 247, 0.45) !important;
            }

            /* ── Secondary buttons ───────────────────────── */
            .gr-button-secondary,
            button[variant="secondary"] {
                border-color: #374151 !important;
                color: #d1d5db !important;
            }

            /* ── Inputs ──────────────────────────────────── */
            textarea, .gr-text-input input {
                background: #1f2937 !important;
                border-color: #374151 !important;
                color: #f9fafb !important;
            }

            /* ── Plots containers ────────────────────────── */
            .plot-container > div { border-radius: 8px !important; overflow: hidden !important; }

            /* ── Markdown inside tabs ────────────────────── */
            .tab-content .prose h2 { color: #c4b5fd !important; }
            .tab-content .prose h3 { color: #a5b4fc !important; }
            .tab-content .prose code { background: #1f2937 !important; color: #a855f7 !important; }

            /* ── Section labels ──────────────────────────── */
            .section-label {
                font-size: 0.75rem !important;
                font-weight: 600 !important;
                text-transform: uppercase !important;
                letter-spacing: 0.05em !important;
                color: #6b7280 !important;
                margin-bottom: 0.5rem !important;
            }

            /* ── Badge trained ───────────────────────────── */
            .badge-ok  { color: #4ade80 !important; font-weight: 700; }
            .badge-no  { color: #6b7280 !important; }
        """,
    ) as demo:
        state = gr.State(make_initial_state)

        with gr.Group(elem_classes="hero-block"):
            gr.Markdown(
                "# ML Vision Studio\n"
                "Plateforme pédagogique d'IA — entraînez, visualisez et comparez "
                "des modèles **Machine Learning** et **Deep Learning** sur image, audio, texte et données tabulaires."
            )

        with gr.Tabs():

            # ── TAB 1 : PRÉDICTION ────────────────────────────────────────────
            with gr.TabItem("Prédiction"):
                pred_modality = gr.Radio(
                    ["Image", "Audio"], value="Image", label="Modalité"
                )
                with gr.Row():
                    with gr.Column():
                        pred_image = gr.Image(
                            sources=["webcam", "upload"],
                            type="pil", label="Image / Webcam",
                        )
                        pred_audio = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath", label="Audio / Micro",
                            visible=False,
                        )
                        pred_btn = gr.Button("Classifier", variant="primary")
                    with gr.Column():
                        pred_label = gr.Label(num_top_classes=7, label="Résultats")

                pred_modality.change(
                    fn=lambda m: (
                        gr.Image(visible=(m == "Image")),
                        gr.Audio(visible=(m == "Audio")),
                    ),
                    inputs=pred_modality,
                    outputs=[pred_image, pred_audio],
                )
                pred_btn.click(
                    fn=pred_classify,
                    inputs=[pred_image, pred_audio, pred_modality, state],
                    outputs=pred_label,
                )

            # ── TAB 2 : IMAGE ─────────────────────────────────────────────────
            with gr.TabItem("Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### Données")
                            img_cam = gr.Image(
                                sources=["webcam", "upload"],
                                type="pil", label="Webcam / Import", height=260,
                            )
                            with gr.Row():
                                img_cls_name = gr.Textbox(
                                    placeholder="Nom de la classe…",
                                    label="", scale=3,
                                )
                                img_add_btn = gr.Button("+ Ajouter", scale=1)
                            img_cls_drop = gr.Dropdown(
                                choices=[], label="Classe active", interactive=True
                            )
                            img_cap_btn = gr.Button("Capturer cette image", variant="secondary")
                            img_cap_status = gr.Textbox(
                                label="Statut capture", interactive=False, lines=1
                            )
                        img_summary = gr.Textbox(
                            label="Résumé", interactive=False, lines=5
                        )

                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### Hyperparamètres")
                            img_epochs = gr.Slider(5, 200, value=50, step=5, label="Époques")
                            img_lr     = gr.Dropdown(
                                IMG_LR_OPTS, value=0.001, label="Taux d'apprentissage"
                            )
                            img_batch  = gr.Dropdown(
                                IMG_BATCH_OPTS, value=16, label="Taille de lot"
                            )
                            img_units  = gr.Slider(
                                16, 512, value=100, step=16, label="Neurones cachés"
                            )
                            img_train_btn = gr.Button("Entraîner", variant="primary")
                        img_log  = gr.Textbox(
                            label="Journal d'entraînement", lines=8, interactive=False
                        )
                        img_loss = gr.Plot(label="Courbe de perte")
                        img_conf = gr.Plot(label="Matrice de confusion")
                        img_suggestions = gr.Textbox(
                            label="Suggestions", interactive=False, lines=3
                        )
                        with gr.Row():
                            img_save_btn = gr.Button("Sauvegarder modèle")
                            img_save_file = gr.File(
                                label="Fichier modèle", visible=False, interactive=False
                            )
                        with gr.Accordion("📈 Courbe d'apprentissage", open=False):
                            gr.Markdown(
                                "Analyse si collecter plus de données améliorerait le modèle.  \n"
                                "Utilise les features MobileNetV2 + Régression Logistique (rapide)."
                            )
                            img_lc_btn  = gr.Button("Générer la courbe", variant="secondary")
                            img_lc_plot = gr.Plot(label="Courbe d'apprentissage")
                            img_lc_diag = gr.Textbox(
                                label="Diagnostic", interactive=False, lines=4
                            )

                with gr.Accordion("💡 Dataset de démo — tf_flowers (5 classes)", open=False):
                    gr.Markdown(
                        "**tf_flowers** contient ~3600 images de fleurs (daisy, dandelion, "
                        "roses, sunflowers, tulips). Téléchargement ~210 MB via tensorflow_datasets."
                    )
                    fl_status_txt = gr.Textbox(
                        value=flowers_status_str, label="Statut", interactive=False, lines=3
                    )
                    with gr.Row():
                        fl_dl_btn   = gr.Button("Télécharger tf_flowers")
                        fl_load_btn = gr.Button("Charger dans Image", variant="primary")
                    fl_dl_log = gr.Textbox(label="Progression", lines=2, interactive=False)

                img_add_btn.click(
                    fn=img_add_class,
                    inputs=[img_cls_name, state],
                    outputs=[state, img_cls_drop, img_summary, img_cls_name],
                )
                img_cap_btn.click(
                    fn=img_capture_sample,
                    inputs=[img_cam, img_cls_drop, state],
                    outputs=[state, img_summary, img_cap_status],
                )
                img_train_btn.click(
                    fn=img_train,
                    inputs=[img_epochs, img_lr, img_batch, img_units, state],
                    outputs=[state, img_log, img_loss, img_conf, img_suggestions],
                )
                img_save_btn.click(
                    fn=img_save_model,
                    inputs=state,
                    outputs=img_save_file,
                )
                img_lc_btn.click(
                    fn=img_lc_cb,
                    inputs=state,
                    outputs=[img_lc_plot, img_lc_diag],
                )
                fl_dl_btn.click(fn=flowers_prepare_cb, outputs=fl_dl_log)
                fl_load_btn.click(
                    fn=flowers_to_image_cb,
                    inputs=state,
                    outputs=[state, img_cls_drop, img_summary, fl_dl_log],
                )

            # ── TAB 3 : AUDIO ─────────────────────────────────────────────────
            with gr.TabItem("Audio"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### Enregistrement")
                            aud_recorder = gr.Audio(
                                sources=["microphone", "upload"],
                                type="filepath",
                                label="Enregistrer un son (~0.5–2s)",
                            )
                            gr.Markdown("_Clips courts (0.5–2 s) recommandés_")
                            with gr.Row():
                                aud_cls_name = gr.Textbox(
                                    placeholder="Nom de la classe…",
                                    label="", scale=3,
                                )
                                aud_add_btn = gr.Button("+ Ajouter", scale=1)
                            aud_cls_drop = gr.Dropdown(
                                choices=[], label="Classe active", interactive=True
                            )
                            aud_sample_btn = gr.Button(
                                "Ajouter cet échantillon", variant="secondary"
                            )
                            aud_status = gr.Textbox(
                                label="Statut", interactive=False, lines=1
                            )
                        aud_summary = gr.Textbox(
                            label="Résumé", interactive=False, lines=5
                        )

                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### Hyperparamètres")
                            aud_epochs = gr.Slider(5, 200, value=50, step=5, label="Époques")
                            aud_lr     = gr.Dropdown(
                                AUD_LR_OPTS, value=0.001, label="Taux d'apprentissage"
                            )
                            aud_batch  = gr.Dropdown(
                                AUD_BATCH_OPTS, value=16, label="Taille de lot"
                            )
                            aud_units  = gr.Slider(
                                16, 512, value=128, step=16, label="Neurones cachés"
                            )
                            aud_train_btn = gr.Button(
                                "Entraîner Audio", variant="primary"
                            )
                        aud_log  = gr.Textbox(
                            label="Journal d'entraînement", lines=8, interactive=False
                        )
                        aud_loss = gr.Plot(label="Courbe de perte")
                        aud_conf = gr.Plot(label="Matrice de confusion")
                        aud_suggestions = gr.Textbox(
                            label="Suggestions", interactive=False, lines=3
                        )
                        with gr.Row():
                            aud_save_btn  = gr.Button("Sauvegarder modèle")
                            aud_save_file = gr.File(
                                label="Fichier modèle", visible=False, interactive=False
                            )
                        with gr.Accordion("📈 Courbe d'apprentissage", open=False):
                            gr.Markdown(
                                "Ré-entraîne le réseau dense sur des fractions croissantes.  \n"
                                "Durée estimée : 5–15 secondes."
                            )
                            aud_lc_btn  = gr.Button("Générer la courbe", variant="secondary")
                            aud_lc_plot = gr.Plot(label="Courbe d'apprentissage")
                            aud_lc_diag = gr.Textbox(
                                label="Diagnostic", interactive=False, lines=4
                            )

                with gr.Accordion("💡 Dataset de démo — Speech Commands (10 mots)", open=False):
                    gr.Markdown(
                        "**Speech Commands** (Google) — 10 mots : yes/no/up/down/left/right/on/off/stop/go.  \n"
                        "Téléchargement partiel ~150 MB (premiers shards seulement)."
                    )
                    sp_status_txt = gr.Textbox(
                        value=speech_status_str, label="Statut", interactive=False, lines=3
                    )
                    with gr.Row():
                        sp_dl_btn   = gr.Button("Télécharger Speech Commands")
                        sp_load_btn = gr.Button("Charger dans Audio", variant="primary")
                    sp_dl_log = gr.Textbox(label="Progression", lines=2, interactive=False)

                aud_add_btn.click(
                    fn=aud_add_class,
                    inputs=[aud_cls_name, state],
                    outputs=[state, aud_cls_drop, aud_summary, aud_cls_name],
                )
                aud_sample_btn.click(
                    fn=aud_add_sample,
                    inputs=[aud_recorder, aud_cls_drop, state],
                    outputs=[state, aud_status, aud_summary],
                )
                aud_train_btn.click(
                    fn=aud_train,
                    inputs=[aud_epochs, aud_lr, aud_batch, aud_units, state],
                    outputs=[state, aud_log, aud_loss, aud_conf, aud_suggestions],
                )
                aud_save_btn.click(
                    fn=aud_save_model,
                    inputs=state,
                    outputs=aud_save_file,
                )
                aud_lc_btn.click(
                    fn=aud_lc_cb,
                    inputs=state,
                    outputs=[aud_lc_plot, aud_lc_diag],
                )
                sp_dl_btn.click(fn=speech_prepare_cb, outputs=sp_dl_log)
                sp_load_btn.click(
                    fn=speech_to_audio_cb,
                    inputs=state,
                    outputs=[state, aud_cls_drop, aud_summary, sp_dl_log],
                )

            # ── TAB 4 : TEXTE ─────────────────────────────────────────────────
            with gr.TabItem("Texte"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### Classes")
                            with gr.Row():
                                txt_cls_name = gr.Textbox(
                                    placeholder="Nom de la classe…",
                                    label="", scale=3,
                                )
                                txt_add_btn = gr.Button("+ Ajouter", scale=1)
                            txt_cls_drop = gr.Dropdown(
                                choices=[], label="Classe cible", interactive=True
                            )

                        with gr.Group():
                            gr.Markdown("### Importer du contenu")
                            txt_file = gr.File(
                                label="Fichier (.txt .md .pdf)",
                                file_types=[".txt", ".md", ".pdf"],
                            )
                            txt_import_file_btn = gr.Button("Importer fichier")
                            txt_url = gr.Textbox(
                                placeholder="https://…", label="URL"
                            )
                            txt_import_url_btn = gr.Button("Importer URL")
                            with gr.Row():
                                txt_direct = gr.Textbox(
                                    placeholder="Collez ou tapez du texte…",
                                    label="Saisie directe", lines=4, scale=4,
                                )
                                txt_direct_btn = gr.Button(
                                    "Ajouter", scale=1
                                )
                            txt_import_status = gr.Textbox(
                                label="Statut import", interactive=False, lines=1
                            )
                        txt_summary = gr.Textbox(
                            label="Résumé", interactive=False, lines=5
                        )

                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### Entraînement")
                            txt_mode = gr.Radio(
                                ["KNN", "NN"], value="KNN",
                                label="Mode de classification",
                            )
                            txt_index_btn    = gr.Button(
                                "Indexer embeddings (KNN)", variant="primary"
                            )
                            txt_nn_train_btn = gr.Button(
                                "Entraîner réseau NN (optionnel)"
                            )
                        txt_log  = gr.Textbox(
                            label="Journal", lines=6, interactive=False
                        )
                        txt_conf = gr.Plot(label="Matrice de confusion")
                        txt_suggestions = gr.Textbox(
                            label="Suggestions", interactive=False, lines=3
                        )
                        with gr.Row():
                            txt_export_btn  = gr.Button("Exporter JSON")
                            txt_export_file = gr.File(
                                label="", visible=False, interactive=False
                            )
                            txt_load_btn = gr.UploadButton(
                                "Importer JSON", file_types=[".json"]
                            )
                        with gr.Accordion("📈 Courbe d'apprentissage", open=False):
                            gr.Markdown(
                                "KNN LOO sur des fractions croissantes des embeddings.  \n"
                                "Nécessite d'indexer les embeddings d'abord."
                            )
                            txt_lc_btn  = gr.Button("Générer la courbe", variant="secondary")
                            txt_lc_plot = gr.Plot(label="Courbe d'apprentissage")
                            txt_lc_diag = gr.Textbox(
                                label="Diagnostic", interactive=False, lines=4
                            )

                with gr.Accordion("💡 Dataset de démo — AG News (4 classes)", open=False):
                    gr.Markdown(
                        "**AG News** — articles de presse en 4 catégories : Monde, Sports, Business, Tech.  \n"
                        "Téléchargement ~31 MB via tensorflow_datasets."
                    )
                    ag_status_txt = gr.Textbox(
                        value=agnews_status_str, label="Statut", interactive=False, lines=3
                    )
                    with gr.Row():
                        ag_dl_btn   = gr.Button("Télécharger AG News")
                        ag_load_btn = gr.Button("Charger dans Texte", variant="primary")
                    ag_dl_log = gr.Textbox(label="Progression", lines=2, interactive=False)

                txt_add_btn.click(
                    fn=txt_add_class,
                    inputs=[txt_cls_name, state],
                    outputs=[state, txt_cls_drop, txt_summary, txt_cls_name],
                )
                txt_import_file_btn.click(
                    fn=txt_add_from_file,
                    inputs=[txt_file, txt_cls_drop, state],
                    outputs=[state, txt_import_status, txt_summary],
                )
                txt_import_url_btn.click(
                    fn=txt_add_from_url,
                    inputs=[txt_url, txt_cls_drop, state],
                    outputs=[state, txt_import_status, txt_summary],
                )
                txt_direct_btn.click(
                    fn=txt_add_direct,
                    inputs=[txt_direct, txt_cls_drop, state],
                    outputs=[state, txt_import_status, txt_summary, txt_direct],
                )
                txt_index_btn.click(
                    fn=txt_index_knn,
                    inputs=state,
                    outputs=[state, txt_log, txt_conf, txt_suggestions],
                )
                txt_nn_train_btn.click(
                    fn=txt_train_nn,
                    inputs=state,
                    outputs=[state, txt_log, txt_conf],
                )
                txt_export_btn.click(
                    fn=txt_export_json,
                    inputs=state,
                    outputs=txt_export_file,
                )
                txt_load_btn.upload(
                    fn=txt_load_json,
                    inputs=[txt_load_btn, state],
                    outputs=[state, txt_log, txt_cls_drop, txt_summary],
                )
                ag_dl_btn.click(fn=agnews_prepare_cb, outputs=ag_dl_log)
                ag_load_btn.click(
                    fn=agnews_to_text_cb,
                    inputs=state,
                    outputs=[state, txt_cls_drop, txt_summary, ag_dl_log],
                )
                txt_lc_btn.click(
                    fn=txt_lc_cb,
                    inputs=state,
                    outputs=[txt_lc_plot, txt_lc_diag],
                )
                # Sync mode radio → state
                def _set_text_mode(mode_str: str, st: dict):
                    st = _s(st)
                    st["text_mode"] = "nn" if mode_str == "NN" else "knn"
                    return st
                txt_mode.change(
                    fn=_set_text_mode,
                    inputs=[txt_mode, state],
                    outputs=state,
                )

            # ── TAB 5 : CHATS & CHIENS ───────────────────────────────────────
            with gr.TabItem("🐱 Chats & Chiens"):
                gr.Markdown(
                    "## Classificateur Chats vs Chiens\n"
                    "Comparez une approche **ML classique** (HOG + SVM / Random Forest) "
                    "et un **réseau CNN** entraîné from scratch."
                )

                # ── Données ──────────────────────────────────────────────────
                with gr.Accordion("1. Données", open=True):
                    cd_status_txt = gr.Textbox(
                        label="Statut", value=cd_data_status, lines=2,
                        interactive=False,
                    )
                    cd_dl_btn  = gr.Button("Télécharger le dataset (~786 MB)")
                    cd_dl_log  = gr.Textbox(label="Progression", lines=3,
                                            interactive=False)
                    cd_dl_btn.click(fn=cd_download, outputs=cd_dl_log)

                # ── Approche ML ───────────────────────────────────────────────
                with gr.Accordion("2. Approche ML — SVM + Random Forest", open=False):
                    gr.Markdown(
                        "**Features** : HOG (forme/contours) + histogramme couleur (3×32 bins)  \n"
                        "**Classifieurs** : SVM (noyau RBF, C=10) et Random Forest (200 arbres)"
                    )
                    cd_ml_btn = gr.Button("Entraîner SVM + Random Forest",
                                          variant="primary")
                    cd_ml_log = gr.Textbox(label="Log", lines=12, interactive=False)
                    with gr.Row():
                        cd_ml_cm_svm = gr.Plot(label="Matrice de confusion — SVM")
                        cd_ml_cm_rf  = gr.Plot(label="Matrice de confusion — RF")
                    cd_ml_btn.click(
                        fn=cd_train_ml,
                        outputs=[cd_ml_log, cd_ml_cm_svm, cd_ml_cm_rf],
                    )

                # ── Approche DL ───────────────────────────────────────────────
                with gr.Accordion("3. Approche DL — Transfer Learning EfficientNetB0", open=False):
                    gr.Markdown(
                        "**Base** : EfficientNetB0 pré-entraîné ImageNet (gelé en phase 1)  \n"
                        "**Tête** : GlobalAvgPool → Dense(128) → Dropout(0.3) → Softmax  \n"
                        "**Phase 1** (10 époques, lr=1e-3) — tête seule  \n"
                        "**Phase 2** (époques configurables, lr=1e-5) — fine-tuning top 30 couches  \n"
                        "Augmentation : flip horizontal, rotation ±10°, zoom ±10°, luminosité ±10%"
                    )
                    with gr.Row():
                        cd_dl_epochs = gr.Slider(5, 30, value=15, step=1,
                                                 label="Époques fine-tuning (phase 2)")
                        cd_dl_batch  = gr.Slider(8, 64, value=32, step=8,
                                                 label="Batch size")
                    cd_cnn_btn    = gr.Button("Entraîner le CNN", variant="primary")
                    cd_cnn_log    = gr.Textbox(label="Log", lines=12, interactive=False)
                    cd_cnn_curves = gr.Plot(label="Courbes d'entraînement")
                    cd_cnn_cm     = gr.Plot(label="Matrice de confusion — CNN")
                    cd_cnn_btn.click(
                        fn=cd_train_dl,
                        inputs=[cd_dl_epochs, cd_dl_batch],
                        outputs=[cd_cnn_log, cd_cnn_curves, cd_cnn_cm],
                    )

                # ── Courbe d'apprentissage ─────────────────────────────────────
                with gr.Accordion("📈 Courbe d'apprentissage", open=False):
                    gr.Markdown(
                        "Features EfficientNetB0 extraites une seule fois (batches 32) "
                        "+ Régression Logistique sur fractions croissantes.  \n"
                        "Nécessite que le dataset soit téléchargé."
                    )
                    cd_lc_btn  = gr.Button("Générer la courbe", variant="secondary")
                    cd_lc_plot = gr.Plot(label="Courbe d'apprentissage")
                    cd_lc_diag = gr.Textbox(label="Diagnostic", interactive=False, lines=4)
                    cd_lc_btn.click(
                        fn=cd_lc_cb,
                        inputs=[],
                        outputs=[cd_lc_plot, cd_lc_diag],
                    )

                # ── Prédiction ────────────────────────────────────────────────
                with gr.Accordion("4. Prédiction", open=False):
                    gr.Markdown(
                        "Chargez une image — les modèles entraînés prédisent en parallèle.  \n"
                        "**Ensemble** : moyenne pondérée CNN×0.6 + SVM×0.2 + RF×0.2"
                    )
                    cd_pred_img = gr.Image(label="Image", type="numpy", height=250)
                    cd_pred_btn = gr.Button("Prédire", variant="primary")
                    cd_pred_msg = gr.Textbox(label="Statut", interactive=False)
                    with gr.Row():
                        cd_pred_svm      = gr.Label(label="SVM",      num_top_classes=2)
                        cd_pred_rf       = gr.Label(label="RF",        num_top_classes=2)
                        cd_pred_cnn      = gr.Label(label="CNN",       num_top_classes=2)
                        cd_pred_ensemble = gr.Label(label="🏆 Ensemble", num_top_classes=2)
                    cd_pred_btn.click(
                        fn=cd_predict,
                        inputs=cd_pred_img,
                        outputs=[cd_pred_msg, cd_pred_svm, cd_pred_rf,
                                 cd_pred_cnn, cd_pred_ensemble],
                    )

                # ── Grad-CAM ──────────────────────────────────────────────────
                with gr.Accordion("🔥 Grad-CAM — Zones d'attention du CNN", open=False):
                    gr.Markdown(
                        "**Grad-CAM** visualise les zones de l'image sur lesquelles "
                        "le réseau se concentre pour prédire « chat » ou « chien ».  \n"
                        "Nécessite que le **CNN EfficientNetB0** soit entraîné."
                    )
                    cd_gcam_img = gr.Image(label="Image (chat ou chien)", type="numpy", height=250)
                    cd_gcam_btn = gr.Button("Analyser avec Grad-CAM", variant="primary")
                    cd_gcam_msg = gr.Textbox(label="Prédiction", interactive=False, lines=1)
                    cd_gcam_fig = gr.Plot(label="Image | Heatmap | Superposition")
                    cd_gcam_btn.click(
                        fn=gradcam_cd_cb,
                        inputs=cd_gcam_img,
                        outputs=[cd_gcam_fig, cd_gcam_msg],
                    )

            # ── TAB 7 : PRIX MAISON ───────────────────────────────────────────
            with gr.TabItem("🏠 Prix Maison"):
                gr.Markdown(
                    "## Régression Linéaire — Prédiction de Prix Immobilier\n"
                    "Prédit le **prix (k€)** d'une maison à partir de sa **surface (m²)** "
                    "à l'aide d'une régression linéaire simple.\n\n"
                    "**Relation réelle** : Prix = 2.5 × Surface + 50 + bruit gaussien"
                )

                with gr.Row():
                    # ── Colonne gauche : Dataset ──────────────────────────────
                    with gr.Column():
                        gr.Markdown("### 1. Créer le dataset")
                        price_n_slider    = gr.Slider(20, 500, value=100, step=10,
                                                      label="Nombre d'exemples")
                        price_noise_slider = gr.Slider(5, 100, value=30, step=5,
                                                       label="Niveau de bruit (k€)")
                        price_gen_btn     = gr.Button("Générer le dataset", variant="secondary")
                        price_gen_status  = gr.Textbox(label="Statut", interactive=False, lines=5)
                        price_dataset_plot = gr.Plot(label="Nuage de points")

                    # ── Colonne droite : Entraînement ─────────────────────────
                    with gr.Column():
                        gr.Markdown("### 2. Entraîner le modèle")
                        price_train_btn   = gr.Button("Entraîner la régression linéaire",
                                                      variant="primary")
                        price_metrics_md  = gr.Markdown("*Aucun modèle entraîné.*")
                        price_reg_plot    = gr.Plot(label="Droite de régression + résidus")

                gr.Markdown("### 3. Prédire le prix")
                gr.Markdown(
                    "Bougez le curseur — la prédiction se met à jour **en temps réel** "
                    "dès que le modèle est entraîné."
                )
                price_surface_slider = gr.Slider(10, 300, value=100, step=5,
                                                 label="Surface de la maison (m²)")
                price_pred_result = gr.Markdown("*Entraînez d'abord le modèle.*")

                # ── Courbe d'apprentissage ─────────────────────────────────────
                with gr.Accordion("📈 Courbe d'apprentissage (R²)", open=False):
                    gr.Markdown(
                        "Entraîne la régression sur des fractions croissantes du dataset "
                        "(10 % → 100 %).  \n"
                        "Mesure le **score R²** sur train et validation — rapide (~0.1 s).  \n"
                        "Nécessite que le modèle ait été entraîné au moins une fois."
                    )
                    price_lc_btn  = gr.Button("Générer la courbe", variant="secondary")
                    price_lc_plot = gr.Plot(label="Courbe d'apprentissage — R²")
                    price_lc_diag = gr.Textbox(label="Diagnostic", interactive=False, lines=5)

                # Événements
                price_gen_btn.click(
                    fn=price_generate_dataset_cb,
                    inputs=[price_n_slider, price_noise_slider, state],
                    outputs=[state, price_dataset_plot, price_gen_status],
                )
                price_train_btn.click(
                    fn=price_train_cb,
                    inputs=[price_n_slider, price_noise_slider, state],
                    outputs=[state, price_reg_plot, price_metrics_md],
                )
                # Prédiction en temps réel à chaque mouvement du slider
                price_surface_slider.change(
                    fn=price_predict_cb,
                    inputs=[price_surface_slider, state],
                    outputs=[state, price_reg_plot, price_pred_result],
                )
                price_lc_btn.click(
                    fn=price_lc_cb,
                    inputs=[price_n_slider, price_noise_slider, state],
                    outputs=[price_lc_plot, price_lc_diag],
                )

            # ── TAB 8 : CHIFFRES MNIST ────────────────────────────────────────
            with gr.TabItem("🔢 Chiffres MNIST"):
                gr.Markdown(
                    "## Reconnaissance de Chiffres Manuscrits — Deep Learning (CNN)\n"
                    "Un réseau de neurones convolutif classifie les chiffres 0–9 "
                    "du dataset **MNIST** (70 000 images 28×28, niveaux de gris)."
                )

                # ── Architecture (accordéon pédagogique) ──────────────────────
                with gr.Accordion("🏗️ Architecture du CNN", open=False):
                    gr.Markdown(
                        "```\n"
                        "Input (28×28×1)\n"
                        "  │\n"
                        "  ├─ Conv2D(32 filtres, 3×3, relu, padding=same)\n"
                        "  ├─ Conv2D(64 filtres, 3×3, relu, padding=same)\n"
                        "  ├─ MaxPooling2D(2×2)        ← réduit à 14×14\n"
                        "  ├─ Dropout(0.25)            ← régularisation\n"
                        "  ├─ Flatten                  ← 64×14×14 = 12 544 valeurs\n"
                        "  ├─ Dense(128, relu)\n"
                        "  ├─ Dropout(0.40)\n"
                        "  └─ Dense(10, softmax)       ← 10 probabilités (0–9)\n"
                        "```\n\n"
                        "**Optimiseur** : Adam  |  "
                        "**Perte** : sparse categorical cross-entropy  |  "
                        "**Précision attendue** : ~99 % en 5 époques"
                    )

                with gr.Row():
                    # ── Colonne gauche : Données + Entraînement ───────────────
                    with gr.Column(scale=1):
                        gr.Markdown("### 1. Charger les données")
                        with gr.Row():
                            mnist_load_btn     = gr.Button("Charger MNIST",
                                                           variant="secondary", scale=2)
                            mnist_load_train_btn = gr.Button("⚡ Charger et entraîner",
                                                             variant="primary", scale=3)
                        mnist_load_status  = gr.Textbox(label="Statut", interactive=False,
                                                        lines=6)
                        mnist_sample_plot  = gr.Plot(label="Exemples MNIST")

                        gr.Markdown("### 2. Entraîner le CNN")
                        with gr.Row():
                            mnist_epochs_sl = gr.Slider(1, 20, value=5, step=1,
                                                        label="Époques")
                            mnist_batch_sl  = gr.Slider(32, 256, value=128, step=32,
                                                        label="Batch size")
                        mnist_lr_radio = gr.Radio(
                            [0.0001, 0.0005, 0.001, 0.005],
                            value=0.001, label="Taux d'apprentissage",
                        )
                        mnist_train_btn  = gr.Button("Entraîner le CNN", variant="primary")
                        mnist_train_log  = gr.Textbox(label="Log d'entraînement",
                                                      interactive=False, lines=10)

                    # ── Colonne droite : Résultats ────────────────────────────
                    with gr.Column(scale=1):
                        gr.Markdown("### Résultats")
                        mnist_summary_md  = gr.Markdown(
                            "*Entraînez le modèle pour voir les résultats.*"
                        )
                        mnist_curves_plot = gr.Plot(label="Courbes train / test")
                        mnist_cm_plot     = gr.Plot(label="Matrice de confusion (10×10)")

                # ── Prédiction ────────────────────────────────────────────────
                gr.Markdown("### 3. Prédire un chiffre")
                gr.Markdown(
                    "Dessinez un chiffre (trait **blanc** sur fond **noir**, style MNIST) "
                    "ou importez une image. Cliquez sur **Reconnaître**."
                )
                with gr.Row():
                    mnist_sketch = gr.Sketchpad(
                        label="Dessinez un chiffre ici",
                        type="pil",
                        canvas_size=(280, 280),
                        brush=gr.Brush(colors=["#ffffff"], color_mode="fixed",
                                       default_size=24),
                        layers=False,
                        height=320,
                    )
                    with gr.Column():
                        mnist_pred_btn    = gr.Button("Reconnaître le chiffre",
                                                      variant="primary")
                        mnist_pred_result = gr.Markdown("*Dessinez puis cliquez.*")
                        mnist_pred_label  = gr.Label(label="Probabilités par classe",
                                                     num_top_classes=10)

                # ── Courbe d'apprentissage ─────────────────────────────────────
                with gr.Accordion("📈 Courbe d'apprentissage", open=False):
                    gr.Markdown(
                        "Entraîne une **Régression Logistique** sklearn sur des fractions "
                        "croissantes des données MNIST (pixels 28×28 aplatis).  \n"
                        "Utilise 10 000 exemples max pour rester rapide (~20 secondes).  \n"
                        "Illustre la dynamique train/val et le seuil de saturation.  \n"
                        "Nécessite que le dataset soit chargé."
                    )
                    mnist_lc_btn  = gr.Button("Générer la courbe (~20 s)",
                                              variant="secondary")
                    mnist_lc_plot = gr.Plot(label="Courbe d'apprentissage — MNIST")
                    mnist_lc_diag = gr.Textbox(label="Diagnostic", interactive=False,
                                               lines=5)

                # Événements
                mnist_load_btn.click(
                    fn=mnist_load_cb,
                    inputs=[state],
                    outputs=[state, mnist_sample_plot, mnist_load_status],
                )
                mnist_train_btn.click(
                    fn=mnist_train_cb,
                    inputs=[mnist_epochs_sl, mnist_batch_sl, mnist_lr_radio, state],
                    outputs=[state, mnist_train_log, mnist_curves_plot,
                             mnist_cm_plot, mnist_summary_md],
                )
                # Bouton "Charger et entraîner" — chaîne les deux opérations
                mnist_load_train_btn.click(
                    fn=mnist_load_cb,
                    inputs=[state],
                    outputs=[state, mnist_sample_plot, mnist_load_status],
                ).then(
                    fn=mnist_train_cb,
                    inputs=[mnist_epochs_sl, mnist_batch_sl, mnist_lr_radio, state],
                    outputs=[state, mnist_train_log, mnist_curves_plot,
                             mnist_cm_plot, mnist_summary_md],
                )
                mnist_pred_btn.click(
                    fn=mnist_predict_cb,
                    inputs=[mnist_sketch, state],
                    outputs=[mnist_pred_label, mnist_pred_result],
                )
                mnist_lc_btn.click(
                    fn=mnist_lc_cb,
                    inputs=[state],
                    outputs=[mnist_lc_plot, mnist_lc_diag],
                )

                # ── Grad-CAM MNIST ────────────────────────────────────────────
                with gr.Accordion("🔥 Grad-CAM — Ce que le CNN voit", open=False):
                    gr.Markdown(
                        "**Grad-CAM** met en évidence les pixels déterminants pour "
                        "la reconnaissance du chiffre.  \n"
                        "Dessinez un chiffre ci-dessous (ou réutilisez le canvas ci-dessus), "
                        "puis cliquez **Analyser**.  \n"
                        "Nécessite que le **CNN MNIST** soit entraîné."
                    )
                    with gr.Row():
                        mnist_gcam_sketch = gr.Sketchpad(
                            label="Dessinez un chiffre",
                            type="pil",
                            canvas_size=(280, 280),
                            brush=gr.Brush(colors=["#ffffff"], color_mode="fixed",
                                           default_size=24),
                            layers=False,
                            height=300,
                        )
                        with gr.Column():
                            mnist_gcam_btn = gr.Button(
                                "Analyser avec Grad-CAM", variant="primary"
                            )
                            mnist_gcam_msg = gr.Textbox(
                                label="Prédiction", interactive=False, lines=2
                            )
                    mnist_gcam_fig = gr.Plot(label="Image | Heatmap | Superposition")
                    mnist_gcam_btn.click(
                        fn=gradcam_mnist_cb,
                        inputs=[mnist_gcam_sketch, state],
                        outputs=[mnist_gcam_fig, mnist_gcam_msg],
                    )

            # ── TAB 6 : CHAT ──────────────────────────────────────────────────
            with gr.TabItem("💬 Chat"):
                chat_modality = gr.Radio(
                    ["Texte", "Image", "Audio"], value="Texte",
                    label="Modèle actif",
                )
                chatbot = gr.Chatbot(
                    label="Conversation", height=400, bubble_full_width=False
                )
                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Posez une question…",
                        label="", lines=1, scale=4,
                    )
                    chat_send_btn = gr.Button("Envoyer", variant="primary", scale=1)

                chat_send_btn.click(
                    fn=chat_send,
                    inputs=[chat_input, chatbot, chat_modality, state],
                    outputs=[chatbot, chat_input],
                )
                chat_input.submit(
                    fn=chat_send,
                    inputs=[chat_input, chatbot, chat_modality, state],
                    outputs=[chatbot, chat_input],
                )

            # ── TAB 9 : K-MEANS CLUSTERING ────────────────────────────────────
            with gr.TabItem("🔵 Clustering"):
                gr.Markdown(
                    "## K-Means Clustering — Apprentissage Non Supervisé\n"
                    "Regroupe les images MNIST en **K clusters sans étiquettes**,\n"
                    "puis compare visuellement la structure trouvée aux vraies classes."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### Étape 1 — Méthode du coude")
                            gr.Markdown(
                                "Calcule l'inertie et le score **silhouette** pour K = 2 … K_max.  \n"
                                "Le K optimal = là où la silhouette est maximale."
                            )
                            cl_kmax_sl  = gr.Slider(5, 15, value=12, step=1,
                                                     label="K maximum à tester")
                            cl_elbow_btn = gr.Button("Calculer la méthode du coude",
                                                      variant="primary")
                            cl_best_k_nb = gr.Number(label="K recommandé", interactive=False,
                                                      precision=0)
                        cl_elbow_msg = gr.Textbox(label="Résultat", interactive=False, lines=3)
                        cl_elbow_fig = gr.Plot(label="Inertie + Silhouette")

                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### Étape 2 — Clustering + Visualisation t-SNE")
                            gr.Markdown(
                                "**t-SNE** réduit les 784 dimensions MNIST à 2D pour visualiser  \n"
                                "comment les clusters K-Means correspondent aux vraies classes."
                            )
                            cl_k_sl     = gr.Slider(2, 15, value=10, step=1,
                                                     label="K (nombre de clusters)")
                            cl_tsne_sl  = gr.Slider(500, 3000, value=2000, step=500,
                                                     label="Points pour t-SNE")
                            cl_run_btn  = gr.Button("Lancer K-Means + t-SNE (~30 s)",
                                                     variant="primary")
                        cl_run_msg  = gr.Textbox(label="Métriques", interactive=False, lines=6)
                        cl_tsne_fig = gr.Plot(label="Projection t-SNE")

                with gr.Accordion("📖 Concepts clés", open=False):
                    gr.Markdown(
                        "#### K-Means\n"
                        "Assigne chaque point au centroïde le plus proche, puis recalcule les centroïdes.  \n"
                        "Minimise l'**inertie** = somme des distances² point→centroïde.\n\n"
                        "#### Méthode du coude\n"
                        "L'inertie décroît rapidement jusqu'au bon K, puis ralentit → forme un « coude ».\n\n"
                        "#### Score Silhouette\n"
                        "Mesure la cohésion (distance aux voisins du même cluster) vs séparation "
                        "(distance au cluster le plus proche). Va de -1 à +1 (1 = parfait).\n\n"
                        "#### ARI (Adjusted Rand Index)\n"
                        "Compare les clusters trouvés aux vraies étiquettes. "
                        "0 = aléatoire, 1 = identique. Attention : K-Means sans supervision "
                        "ne peut pas atteindre 1.0 sur MNIST (10 classes, mais les chiffres se ressemblent).\n\n"
                        "#### t-SNE\n"
                        "Réduction non linéaire : préserve les structures locales. "
                        "Les points proches en 784D restent proches en 2D."
                    )

                cl_elbow_btn.click(
                    fn=clustering_elbow_cb,
                    inputs=[cl_kmax_sl, state],
                    outputs=[cl_elbow_fig, cl_elbow_msg, cl_best_k_nb],
                )
                cl_run_btn.click(
                    fn=clustering_run_cb,
                    inputs=[cl_k_sl, cl_tsne_sl, state],
                    outputs=[state, cl_tsne_fig, cl_run_msg],
                )

            # ── TAB 10 : RÉGRESSION MULTIVARIÉE ──────────────────────────────
            with gr.TabItem("🏘️ Régression Multi"):
                gr.Markdown(
                    "## Régression Multivariée — California Housing\n"
                    "Prédit la **valeur médiane des maisons** (k$) à partir de "
                    "jusqu'à **8 variables** socio-économiques.  \n"
                    "Dataset : Census Bureau 1990 — 20 640 quartiers californiens."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### Features à inclure")
                            mr_feat_check = gr.CheckboxGroup(
                                choices=ALL_FEATURE_NAMES,
                                value=["MedInc", "HouseAge", "AveRooms",
                                       "AveOccup", "Latitude", "Longitude"],
                                label="Variables explicatives",
                            )
                            gr.Markdown(
                                "| Feature | Description |\n"
                                "|---------|-------------|\n"
                                "| `MedInc` | Revenu médian (×10k$) |\n"
                                "| `HouseAge` | Âge médian des logements |\n"
                                "| `AveRooms` | Nb pièces moyen |\n"
                                "| `AveBedrms` | Nb chambres moyen |\n"
                                "| `Population` | Population quartier |\n"
                                "| `AveOccup` | Occupants moyens |\n"
                                "| `Latitude` | Latitude géo |\n"
                                "| `Longitude` | Longitude géo |"
                            )

                        with gr.Group():
                            gr.Markdown("### Modèle")
                            mr_model_radio = gr.Radio(
                                ["Linéaire (OLS)", "Ridge (régularisation L2)"],
                                value="Linéaire (OLS)",
                                label="Type de régression",
                            )
                            mr_train_btn = gr.Button(
                                "Entraîner (~2 s)", variant="primary"
                            )
                        mr_summary_md = gr.Markdown("*Sélectionnez des features et entraînez.*")

                    with gr.Column(scale=2):
                        with gr.Row():
                            mr_imp_fig = gr.Plot(label="Importance des features")
                            mr_sc_fig  = gr.Plot(label="Prédit vs Réel + Résidus")

                with gr.Accordion("📖 Concepts clés", open=False):
                    gr.Markdown(
                        "#### Régression multivariée\n"
                        "Étend la régression linéaire à plusieurs variables :  \n"
                        "`Prix = a₁×MedInc + a₂×HouseAge + … + b`  \n"
                        "La solution est toujours calculée analytiquement (équations normales).\n\n"
                        "#### Ridge (régularisation L2)\n"
                        "Ajoute une pénalité `λ‖w‖²` aux coefficients : empêche l'overfitting "
                        "et stabilise les estimations quand les features sont corrélées.\n\n"
                        "#### R² (score de détermination)\n"
                        "Part de la variance expliquée par le modèle. 0.60 = 60% de la "
                        "variabilité des prix est capturée.  \n"
                        "California Housing plafonne autour de R²≈0.60–0.65 avec la régression "
                        "linéaire (relations non-linéaires importantes).\n\n"
                        "#### Graphe des résidus\n"
                        "Un nuage aléatoire autour de 0 → modèle bien spécifié.  \n"
                        "Un entonnoir → hétéroscédasticité (log-transformer y)."
                    )

                mr_train_btn.click(
                    fn=multireg_train_cb,
                    inputs=[mr_feat_check, mr_model_radio, state],
                    outputs=[state, mr_imp_fig, mr_sc_fig, mr_summary_md],
                )

            # ── TAB 11 : DASHBOARD ────────────────────────────────────────────
            with gr.TabItem("📊 Dashboard"):
                gr.Markdown(
                    "## Tableau de Bord — Vue d'Ensemble des Modèles\n"
                    "Résumé de tous les modèles entraînés dans la session courante."
                )
                dash_refresh_btn = gr.Button("🔄 Rafraîchir le tableau", variant="primary")
                dash_summary_md  = gr.Markdown("*Cliquez sur Rafraîchir.*")
                dash_table_html  = gr.HTML()
                dash_metric_fig  = gr.Plot(label="Comparaison R² — modèles de régression")

                with gr.Accordion("📖 Lecture du tableau", open=False):
                    gr.Markdown(
                        "**Modèle** : identifiant de l'algorithme  \n"
                        "**Métrique** : précision (classification) ou R² (régression), silhouette (clustering)  \n"
                        "**Statut** : ✅ = modèle entraîné cette session  \n\n"
                        "Les métriques de précision image/audio/texte/MNIST ne sont pas persistées "
                        "dans la session Gradio actuelle — relancez l'entraînement pour les voir."
                    )

                dash_refresh_btn.click(
                    fn=dashboard_refresh_cb,
                    inputs=state,
                    outputs=[dash_table_html, dash_metric_fig, dash_summary_md],
                )


    return demo


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=False,  
        share=True,
        debug=True,
        show_error=True,
    )
