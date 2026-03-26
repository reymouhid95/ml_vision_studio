"""
ML Vision Studio — Python + Gradio
Equivalent of index.html / app.js rewritten in Python.

Run:
  cd ml_vision_studio
  python app.py
"""
from __future__ import annotations

import copy
import json
import os
import ssl
import sys
import tempfile

# Fix SSL certificate verification on Homebrew Python (macOS)
# Homebrew Python doesn't install system CA certs; patch to use certifi's bundle.
# - ssl patch:      covers urllib (Keras weight downloads)
# - env vars:       covers requests + TensorFlow Hub downloads
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

# Support HEIC/HEIF (photos iPhone)
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
from core.audio_trainer import HP_BATCH_OPTS as AUD_BATCH_OPTS
from core.audio_trainer import HP_LR_OPTS as AUD_LR_OPTS
from core.audio_trainer import (extract_mel_features, predict_audio,
                                train_audio_model)
from core.image_trainer import HP_BATCH_OPTS as IMG_BATCH_OPTS
from core.image_trainer import HP_LR_OPTS as IMG_LR_OPTS
from core.image_trainer import predict_image, train_image_model
from core.text_trainer import (build_knn_index, classify_knn, classify_with_nn,
                               embed_single, knn_leave_one_out,
                               split_text_into_chunks, train_text_nn_model)
from PIL import Image
from utils.augmentation import augment_image
from utils.confusion_matrix import make_confusion_figure
from utils.pdf_import import extract_pdf_page_images, extract_pdf_text
from utils.url_import import fetch_url_text
from cats_vs_dogs.data_prep import (
    download_and_prepare, is_prepared, load_split, split_counts,
    CLASS_NAMES as CD_CLASS_NAMES,
)
from cats_vs_dogs.ml_model import (
    train_ml_models, predict_ml, models_trained as ml_models_trained,
)
from cats_vs_dogs.dl_model import (
    train_dl_model, predict_dl, model_trained as dl_model_trained,
)

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
        yield state, "Ajoutez au moins 2 classes.", None, None
        return
    if any(len(c["samples"]) < 3 for c in classes):
        yield state, "Chaque classe doit avoir au moins 3 images.", None, None
        return

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
            yield state, "\n".join(log_lines[-20:]), _loss_fig(loss_hist), None

        elif tag == "done":
            _, model, lh, cnames, preds, actuals = update
            state = _s(state)
            state["image_model"]       = model
            state["image_class_names"] = cnames
            state["image_trained"]     = True
            conf = make_confusion_figure(preds, actuals, cnames)
            log_lines.append("✓ Entraînement terminé !")
            yield state, "\n".join(log_lines[-20:]), _loss_fig(lh), conf


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
        yield state, "Ajoutez au moins 2 classes.", None, None
        return
    if any(len(c["samples"]) < 3 for c in classes):
        yield state, "Chaque classe doit avoir au moins 3 échantillons.", None, None
        return

    log_lines: list[str] = []
    loss_hist: list[float] = []

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
            log_lines.append(line)
            loss_hist.append(loss)
            yield state, "\n".join(log_lines[-20:]), _loss_fig(loss_hist), None

        elif tag == "done":
            _, model, lh, cnames, preds, actuals = update
            state = _s(state)
            state["audio_model"]       = model
            state["audio_class_names"] = cnames
            state["audio_trained"]     = True
            conf = make_confusion_figure(preds, actuals, cnames)
            log_lines.append("✓ Entraînement audio terminé !")
            yield state, "\n".join(log_lines[-20:]), _loss_fig(lh), conf


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
        return state, "Ajoutez au moins 2 classes.", None
    if any(len(c["samples"]) < 1 for c in classes):
        return state, "Chaque classe doit avoir au moins 1 texte.", None

    progress(0, desc="Chargement USE…")
    try:
        knn_entries, class_names = build_knn_index(classes)
    except Exception as e:
        return state, f"Erreur indexation : {e}", None

    progress(0.7, desc="Évaluation leave-one-out…")
    preds, actuals = knn_leave_one_out(knn_entries, class_names)
    conf = make_confusion_figure(preds, actuals, class_names)

    state = _s(state)
    state["text_knn"]          = knn_entries
    state["text_class_names"]  = class_names
    state["text_trained"]      = True
    state["text_mode"]         = "knn"
    progress(1.0, desc="Terminé")
    return state, f"✓ {len(knn_entries)} embeddings indexés.", conf


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


def cd_train_dl(epochs: int, batch_size: int):
    if not is_prepared():
        yield "Données non disponibles. Téléchargez d'abord le dataset.", None, None
        return

    X_tr, y_tr = load_split("train")
    X_v,  y_v  = load_split("val")
    X_te, y_te = load_split("test")

    log     = ""
    results = None
    gen = train_dl_model(X_tr, y_tr, X_v, y_v, X_te, y_te,
                         epochs=epochs, batch_size=batch_size)
    for item in gen:
        if item[0] == "epoch":
            _, ep, total, tl, ta, vl, va = item
            log += f"Époque {ep}/{total} — loss {tl:.4f}  acc {ta:.3f}  val_loss {vl:.4f}  val_acc {va:.3f}\n"
            yield log, None, None
        elif item[0] == "done":
            results = item[1]

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
        return "Chargez une image.", None, None, None

    # Normalize PIL → float32 numpy [0,1]
    pil = Image.fromarray(img).convert("RGB").resize((96, 96))
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
        return "Aucun modèle entraîné. Entraînez SVM/RF et/ou le CNN d'abord.", None, None, None

    def _label(probs: dict) -> dict:
        return {f"{k} ({v:.1%})": v for k, v in probs.items()}

    return (
        "Prédictions effectuées.",
        _label(results.get("SVM", {})) or None,
        _label(results.get("RF",  {})) or None,
        _label(results.get("CNN", {})) or None,
    )


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
            .gr-group { border-radius: 10px !important; }
            footer { display: none !important; }
        """,
    ) as demo:
        state = gr.State(make_initial_state)

        gr.Markdown("# ML Vision Studio\nEntraînez et testez des modèles image, audio et texte.")

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
                        with gr.Row():
                            img_save_btn = gr.Button("Sauvegarder modèle")
                            img_save_file = gr.File(
                                label="Fichier modèle", visible=False, interactive=False
                            )

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
                    outputs=[state, img_log, img_loss, img_conf],
                )
                img_save_btn.click(
                    fn=img_save_model,
                    inputs=state,
                    outputs=img_save_file,
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
                        with gr.Row():
                            aud_save_btn  = gr.Button("Sauvegarder modèle")
                            aud_save_file = gr.File(
                                label="Fichier modèle", visible=False, interactive=False
                            )

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
                    outputs=[state, aud_log, aud_loss, aud_conf],
                )
                aud_save_btn.click(
                    fn=aud_save_model,
                    inputs=state,
                    outputs=aud_save_file,
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
                        with gr.Row():
                            txt_export_btn  = gr.Button("Exporter JSON")
                            txt_export_file = gr.File(
                                label="", visible=False, interactive=False
                            )
                            txt_load_btn = gr.UploadButton(
                                "Importer JSON", file_types=[".json"]
                            )

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
                    outputs=[state, txt_log, txt_conf],
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
                with gr.Accordion("3. Approche DL — CNN", open=False):
                    gr.Markdown(
                        "**Architecture** : Conv2D(32→64→128) → BN → MaxPool → GlobalAvgPool "
                        "→ Dense(128) → Dropout(0.5) → Softmax  \n"
                        "Augmentation : flip horizontal, rotation ±10°, zoom ±10%"
                    )
                    with gr.Row():
                        cd_dl_epochs = gr.Slider(5, 40, value=20, step=1,
                                                 label="Époques max")
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

                # ── Prédiction ────────────────────────────────────────────────
                with gr.Accordion("4. Prédiction", open=False):
                    gr.Markdown("Chargez une image — les modèles entraînés prédisent en parallèle.")
                    cd_pred_img = gr.Image(label="Image", type="numpy", height=250)
                    cd_pred_btn = gr.Button("Prédire", variant="primary")
                    cd_pred_msg = gr.Textbox(label="Statut", interactive=False)
                    with gr.Row():
                        cd_pred_svm = gr.Label(label="SVM",  num_top_classes=2)
                        cd_pred_rf  = gr.Label(label="RF",   num_top_classes=2)
                        cd_pred_cnn = gr.Label(label="CNN",  num_top_classes=2)
                    cd_pred_btn.click(
                        fn=cd_predict,
                        inputs=cd_pred_img,
                        outputs=[cd_pred_msg, cd_pred_svm, cd_pred_rf, cd_pred_cnn],
                    )

            # ── TAB 6 : CHAT ──────────────────────────────────────────────────
            with gr.TabItem("Chat"):
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

    return demo


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=False,   # ouvre le navigateur manuellement sur http://127.0.0.1:7860
        share=False,
    )
