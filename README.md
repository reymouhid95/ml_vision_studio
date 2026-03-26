# ML Vision Studio — Python / Gradio

Interface web locale pour entraîner et tester des modèles de machine learning sur trois modalités (image, audio, texte) ainsi qu'un module de comparaison ML vs DL sur le dataset Chats & Chiens.

---

## Démarrage rapide

```bash
cd ml_vision_studio
venv/bin/python app.py
# → http://127.0.0.1:7860
```

---

## Structure du projet

```
ml_vision_studio/
├── app.py                    # Point d'entrée — UI Gradio + callbacks
├── requirements.txt
│
├── core/                     # Modules d'entraînement réutilisables
│   ├── image_trainer.py      # Transfer learning MobileNetV2
│   ├── audio_trainer.py      # Classificateur Mel spectrogram
│   └── text_trainer.py       # USE embeddings + KNN / NN
│
├── cats_vs_dogs/             # Module Chats & Chiens (ML vs DL)
│   ├── data_prep.py          # Téléchargement + splits numpy
│   ├── ml_model.py           # HOG + histogramme → SVM / Random Forest
│   └── dl_model.py           # CNN entraîné from scratch
│
└── utils/
    ├── augmentation.py       # Flip + luminosité (PIL)
    ├── confusion_matrix.py   # Figure matplotlib (heatmap + F1/P/R)
    ├── pdf_import.py         # Extraction texte/images depuis PDF
    └── url_import.py         # Scraping web (requests + BeautifulSoup)
```

---

## Installation

### Prérequis

- **Python 3.11** (TensorFlow n'est pas compatible avec Python 3.14)
- Homebrew Python 3.11 : `/usr/local/Cellar/python@3.11/.../bin/python3.11`

### Première installation

```bash
# 1. Créer le virtualenv avec Python 3.11
/usr/local/Cellar/python@3.11/3.11.15/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
    -m venv venv

# 2. Installer llvmlite/numba en premier (pas de build from source)
venv/bin/pip install llvmlite==0.43.0 numba==0.60.0

# 3. Installer les dépendances
venv/bin/pip install -r requirements.txt

# 4. Patcher gradio_client (bug schema bool)
#    Voir section "Corrections connues" ci-dessous
```

### Dépendances principales

| Package               | Version        | Rôle                                            |
| --------------------- | -------------- | ----------------------------------------------- |
| `gradio`              | `>=4.0,<5.0`   | Interface web                                   |
| `starlette`           | `>=0.27,<0.47` | Serveur ASGI (starlette 1.0 casse jinja2)       |
| `huggingface_hub`     | `==0.24.7`     | Requis par gradio 4.x (1.x a supprimé HfFolder) |
| `tensorflow`          | `>=2.13,<3.0`  | Moteur d'entraînement                           |
| `tensorflow-hub`      | `>=0.14`       | USE (Universal Sentence Encoder)                |
| `tensorflow-datasets` | `>=4.9`        | Dataset cats_vs_dogs                            |
| `librosa`             | `>=0.10`       | Mel spectrogram audio                           |
| `llvmlite`            | `==0.43.0`     | Dépendance de librosa (wheel pré-compilé)       |
| `numba`               | `==0.60.0`     | Dépendance de librosa (wheel pré-compilé)       |
| `scikit-learn`        | `>=1.3`        | SVM, Random Forest, métriques                   |
| `scikit-image`        | `>=0.21`       | HOG features                                    |

---

## Onglets de l'interface

### 1. Prédiction

Test rapide avec les modèles déjà entraînés.

- **Image** : webcam ou import → classification par le modèle image
- **Audio** : micro ou import → classification par le modèle audio

### 2. Image

Entraînement d'un classificateur d'images personnalisé.

**Pipeline :**

1. Créer des classes (ex. : « chat », « chien », « oiseau »)
2. Capturer ou importer des images par classe (min. 3 par classe)
3. Configurer les hyperparamètres et lancer l'entraînement
4. Visualiser la courbe de perte et la matrice de confusion
5. Sauvegarder le modèle (`.keras`)

**Architecture (`core/image_trainer.py`) :**

```
MobileNetV2 (ImageNet, gelé)
  → GlobalAveragePooling2D
  → Dense(units, relu)
  → Dense(N classes, softmax)
```

Augmentation : flip horizontal + luminosité ×1.3 (×3 le dataset).

### 3. Audio

Entraînement d'un classificateur de sons.

**Pipeline :**

1. Créer des classes (ex. : « musique », « parole », « silence »)
2. Enregistrer des clips courts (0,5–2 s recommandés) par classe
3. Lancer l'entraînement

**Extraction de features (`core/audio_trainer.py`) :**

```
librosa.load (22 050 Hz, mono, max 3 s)
  → melspectrogram (n_mels=40, fmin=80, fmax=8000)
  → log(S + 1e-8)
  → moyenne temporelle → vecteur (40,)
  → normalisation min-max [0, 1]
```

**Architecture :**

```
Dense(H, relu) → BatchNorm → Dropout(0.3)
  → Dense(H/2, relu) → Dropout(0.2)
  → Dense(N, softmax)
```

### 4. Texte

Classification de texte sur corpus personnalisé.

**Import :** fichiers `.txt` / `.md` / `.pdf`, URL, saisie directe.

**Deux modes :**

| Mode             | Description                                                   |
| ---------------- | ------------------------------------------------------------- |
| **KNN** (défaut) | Universal Sentence Encoder (512-dim) + similarité cosinus k=5 |
| **NN**           | Tête Dense entraînée sur les embeddings USE pré-calculés      |

**Architecture NN (`core/text_trainer.py`) :**

```
Input(512)
  → Dense(256, relu) → BatchNorm → Dropout(0.3)
  → Dense(128, relu) → Dropout(0.2)
  → Dense(N, softmax)
```

Évaluation KNN : leave-one-out sur l'index.
Export / import du modèle au format JSON (embeddings inclus).

### 5. Chats & Chiens

Comparaison pédagogique **ML classique vs Deep Learning** sur le dataset Cats vs Dogs.

#### Étape 1 — Données

- Téléchargement via `tensorflow_datasets` (~786 MB, mis en cache dans `~/tensorflow_datasets/`)
- Split : 1 400 train / 300 val / 300 test
- Images redimensionnées à 96×96, normalisées [0, 1]
- Sauvegarde numpy compressée dans `cats_vs_dogs/data/`

#### Étape 2 — Approche ML (`cats_vs_dogs/ml_model.py`)

**Extraction de features :**

- **HOG** (Histogram of Oriented Gradients) : capture la forme et les contours
  - `orientations=9`, `pixels_per_cell=(8,8)`, `cells_per_block=(2,2)`, `block_norm=L2-Hys`
- **Histogramme couleur** : 3 canaux × 32 bins = 96 features

**Classifieurs :**

| Modèle        | Paramètres                                     |
| ------------- | ---------------------------------------------- |
| SVM           | noyau RBF, C=10, gamma=scale, probability=True |
| Random Forest | 200 arbres, n_jobs=-1                          |

Les deux sont encapsulés dans un `sklearn.Pipeline` avec `StandardScaler`.

**Limites attendues :**

- Features manuelles ne capturent pas la structure globale
- Sensibles aux variations de pose, fond, luminosité
- Précision typique : 60–75 % sur ce dataset

#### Étape 3 — Approche DL (`cats_vs_dogs/dl_model.py`)

**Architecture CNN :**

```
Input(96, 96, 3)
  → Conv2D(32, 3×3, same) → BatchNorm → ReLU → MaxPool(2×2)
  → Conv2D(64, 3×3, same) → BatchNorm → ReLU → MaxPool(2×2)
  → Conv2D(128, 3×3, same) → BatchNorm → ReLU → MaxPool(2×2)
  → GlobalAveragePooling2D
  → Dense(128, relu) → Dropout(0.5)
  → Dense(2, softmax)
```

**Entraînement :**

- Optimiseur : Adam (lr=1e-3)
- Augmentation on-the-fly : flip horizontal, rotation ±10°, zoom ±10%
- Early stopping : patience=5 sur `val_loss`, sauvegarde du meilleur modèle
- Précision typique : 80–88 % sur ce dataset

#### Étape 4 — Prédiction

Upload d'une image → prédiction simultanée par SVM, RF et CNN avec affichage des probabilités.

### 6. Chat

Interface conversationnelle pour tester le modèle texte.

- Texte : embed la requête → KNN ou NN → classe + barres de confiance + extraits similaires
- Image / Audio : redirige vers l'onglet Prédiction avec un message explicatif

---

## Architecture technique

### Gestion d'état

L'état de session est un dictionnaire unique passé à chaque callback Gradio (`gr.State`).
**Règle de mutation :**

```python
state = {**state}                           # shallow-copy toujours
state["image_classes"] = copy.deepcopy(...)  # deep-copy uniquement si on mute la liste
```

### Pattern générateur

Les fonctions d'entraînement sont des générateurs Python pour le streaming des logs :

```python
# Côté core (ex : image_trainer.py)
yield ("epoch", ep, total, loss, acc)
yield ("done", model, loss_hist, class_names, preds, actuals)

# Côté app.py
for update in train_image_model(classes, ...):
    if update[0] == "epoch":
        yield state, log, fig, None      # → Gradio stream
    elif update[0] == "done":
        yield state, log, fig, conf_fig
```

### Matrice de confusion (`utils/confusion_matrix.py`)

Figure matplotlib réutilisée pour Image, Audio et Chats & Chiens :

```python
make_confusion_figure(preds, actuals, class_names) → plt.Figure
```

---

## Corrections connues

### Patch `gradio_client` (bug `additionalProperties: false`)

Le schéma JSON de Gradio peut contenir `"additionalProperties": false` (un booléen).
`gradio_client 1.3.0` appelle `"const" in schema` sur ce booléen → `TypeError`.

**Fichier :** `venv/lib/python3.11/site-packages/gradio_client/utils.py`

```python
# Ajouter en tête de get_type() :
def get_type(schema: dict):
    if not isinstance(schema, dict):
        return "any"
    ...

# Ajouter en tête de _json_schema_to_python_type() :
def _json_schema_to_python_type(schema, defs):
    if not isinstance(schema, dict):
        return "Any"
    ...
```

### SSL sur Homebrew Python (macOS)

Homebrew Python 3.11 ne configure pas les certificats CA système.
Corrigé dans `app.py` au démarrage :

```python
import certifi, ssl, os
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
os.environ.setdefault("SSL_CERT_FILE", certifi.where())      # TensorFlow Hub
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where()) # requests
```

---

## Déploiement

### Google Colab

```python
# Cellule 1 — Cloner le projet
!git clone https://github.com/USER/REPO.git
%cd REPO/ml_vision_studio

# Cellule 2 — Dépendances
!pip install llvmlite==0.43.0 numba==0.60.0
!pip install -r requirements.txt

# Cellule 3 — Lancer (URL publique 72h)
import subprocess
subprocess.Popen(["python", "app.py"])  # modifié avec share=True
```

Modifier dans `app.py` :

```python
demo.launch(share=True)   # génère une URL *.gradio.live valide 72h
```

### Hugging Face Spaces

1. Créer un Space de type **Gradio**
2. Ajouter en tête du `README.md` du Space :
   ```yaml
   ---
   sdk: gradio
   sdk_version: "4.x"
   ---
   ```
3. `git push` le contenu de `ml_vision_studio/`
4. Ajouter un `packages.txt` si des dépendances système manquent

**Limites du tier gratuit :** CPU uniquement, ~16 GB RAM, inactivité après 90 min.
