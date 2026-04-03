# ML Vision Studio — Python / Gradio

Interface web locale pour entraîner et tester des modèles de machine learning sur trois modalités (image, audio, texte), un module de comparaison ML vs DL sur le dataset Chats & Chiens, une **régression linéaire** de prix immobilier et la **reconnaissance de chiffres manuscrits** par CNN.

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
│   ├── text_trainer.py       # USE embeddings + KNN / NN
│   ├── price_predictor.py    # Régression Linéaire sklearn  ← nouveau
│   └── mnist_model.py        # CNN TensorFlow/Keras         ← nouveau
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

### 7. Prix Maison ⭐ nouveau

Démonstration de la **régression linéaire** : prédit le prix (k€) d'une maison
à partir de sa surface (m²).

#### Concept : régression linéaire

L'algorithme cherche la droite `Prix = a × Surface + b` qui **minimise
l'erreur quadratique** entre les prédictions et les prix réels.

La solution analytique (équations normales) :

```
a = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / Σ[(xᵢ - x̄)²]
b = ȳ - a·x̄
```

#### Dataset synthétique

Relation vraie codée dans `core/price_predictor.py` :

```python
prix = 2.5 × surface + 50 + bruit_gaussien(σ = noise_level)
```

Exemples concrets (`n=100`, `noise_level=30 k€`) :

| Surface | Formule exacte          | Fourchette réelle (±1σ) |
| ------- | ----------------------- | ----------------------- |
| 30 m²   | 2.5×30+50 = **125 k€**  | 95–155 k€               |
| 80 m²   | 2.5×80+50 = **250 k€**  | 220–280 k€              |
| 120 m²  | 2.5×120+50 = **350 k€** | 320–380 k€              |
| 200 m²  | 2.5×200+50 = **550 k€** | 520–580 k€              |

Le bruit (σ=30 k€ par défaut) simule les facteurs réels non capturés
(localisation, état du bien, étage, exposition).

#### Ce que le modèle apprend

Après entraînement sur 80 exemples (split 80/20) le modèle trouve :

```
Prix = 2.53 × Surface + 43.8 k€    ← appris par gradient
Prix = 2.50 × Surface + 50.0 k€    ← vraie relation
```

L'écart (2.53 vs 2.50, 43.8 vs 50) est dû au bruit aléatoire
et à la taille limitée du dataset.

#### Métriques d'évaluation

| Métrique | Formule               | Interprétation                                              | Valeur typique |
| -------- | --------------------- | ----------------------------------------------------------- | -------------- | -------------------------------------------- | ------ |
| **R²**   | `1 - SS_res / SS_tot` | Part de la variance expliquée. 1.0 = parfait, 0.0 = inutile | ~0.97          |
| **RMSE** | `√(Σ(ŷ−y)²/n)`        | Erreur en k€, pénalise les gros écarts                      | ~30 k€         |
| **MAE**  | `Σ                    | ŷ−y                                                         | /n`            | Erreur absolue moyenne, robuste aux outliers | ~24 k€ |

Lecture concrète d'un résultat R²=0.97, RMSE=30 k€, MAE=24 k€ :

- 97% des variations de prix sont expliquées par la surface
- En moyenne, l'erreur de prédiction est de ±30 000 €
- La moitié des prédictions sont à moins de 24 000 € du vrai prix

#### Graphe des résidus

Le sous-graphe "Résidus vs Prédiction" vérifie les hypothèses du modèle :

- **Nuage aléatoire autour de 0** → modèle bien spécifié ✓
- **Entonnoir** (résidus croissent avec ŷ) → hétéroscédasticité, il faudrait transformer y
- **Courbe** → relation non-linéaire non capturée → essayer regression polynomiale

#### Module `core/price_predictor.py`

```python
generate_dataset(n_samples=100, noise_level=30) → (X, y)
train_price_model(n_samples, noise_level)       → (model, metrics, X, y, X_test, y_test, y_pred)
predict_price(model, surface_m2)                → float   # prix en k€
make_dataset_figure(X, y)                       → plt.Figure  # nuage de points
make_regression_figure(model, X, y, ...)        → plt.Figure  # droite + résidus
```

#### Utilisation

```
1. Sliders : nb d'exemples (20–500) + niveau de bruit (5–100 k€)
2. « Générer le dataset » → nuage de points
3. « Entraîner » → équation apprise + tableau R²/RMSE/MAE + droite + résidus
4. Slider surface (10–300 m²) → prix prédit surligné sur la courbe
```

---

### 8. Chiffres MNIST ⭐ nouveau

Démonstration du **Deep Learning** sur un problème classique de vision :
reconnaître les chiffres manuscrits 0–9 du dataset MNIST.

#### Dataset MNIST

| Propriété         | Valeur                            |
| ----------------- | --------------------------------- |
| Source            | Yann LeCun, AT&T Bell Labs (1998) |
| Images train      | 60 000                            |
| Images test       | 10 000                            |
| Résolution        | 28×28 pixels                      |
| Couleurs          | Niveaux de gris (1 canal)         |
| Classes           | 10 (chiffres 0–9)                 |
| Précision humaine | ~98.5%                            |

Chaque pixel va de 0 (noir) à 255 (blanc), divisé par 255 avant de
passer dans le réseau → valeurs dans [0, 1].

#### Architecture CNN (`core/mnist_model.py`)

```
Input (28×28×1)
     │
Conv2D(32 filtres, 3×3, relu, padding='same')  → (28×28×32)
     │   Détecte : contours, angles, petites courbes locales
Conv2D(64 filtres, 3×3, relu, padding='same')  → (28×28×64)
     │   Combine les motifs simples (ex : "arc + trait" = partie d'un "3")
MaxPooling2D(2×2)                              → (14×14×64)
     │   Réduit la taille ×2, conserve les features dominantes
Dropout(0.25)
     │   Désactive 25% des neurones aléatoirement → évite le surapprentissage
Flatten()                                      → (12 544,)
     │
Dense(128, relu)                               → (128,)
     │   Combine globalement tous les patterns détectés
Dropout(0.4)
     │
Dense(10, softmax)                             → (10,)
         [p(0), p(1), ..., p(9)]   ← somme = 1.0
```

**Total : 1 625 866 paramètres**

#### Pourquoi des couches de convolution ?

Un réseau Dense classique traite chaque pixel indépendamment :
le "7" en haut à gauche est différent du "7" décalé de 5 pixels.
La convolution résout ça : **le même filtre glisse sur toute l'image**,
rendant le réseau invariant aux translations.

Exemple de ce qu'apprend le filtre n°3 de Conv1 :

```
Avant entraînement : valeurs aléatoires
Après entraînement :
  [ -0.2  0.8  0.0 ]    ← répond fort à un contour vertical
  [ -0.1  0.9  0.1 ]
  [ -0.2  0.8 -0.1 ]
```

#### Performances par nombre d'époques

| Époques | Précision test | Temps (CPU) |
| ------- | -------------- | ----------- |
| 1       | ~97.5%         | ~90 s       |
| 3       | ~98.5%         | ~4 min      |
| 5       | ~99.0%         | ~7 min      |
| 10      | ~99.2%         | ~14 min     |

Sur 10 000 chiffres test, 99% de précision = **9 900 chiffres reconnus**,
seuls 100 sont des erreurs.

#### Exemple concret d'une prédiction — chiffre "3"

```
Image 28×28 : fond noir (0.0), traits blancs (0.8–1.0)

→ Conv1 (32 filtres) détecte localement :
   • courbe en haut à droite du "3"
   • trait horizontal central du "3"
   • courbe en bas à droite du "3"

→ Conv2 (64 filtres) combine :
   "deux arcs horizontaux superposés" = motif caractéristique du "3"
   (distinct du "2" qui a un arc + ligne droite)

→ Dense(128) résume globalement : "pattern = chiffre arrondi à droite"

→ Softmax(10) :
   {"0":0.00, "1":0.00, "2":0.01, "3":0.97, "4":0.00,
    "5":0.01, "6":0.00, "7":0.00, "8":0.01, "9":0.00}
              ↑
   Prédit "3" avec 97% de confiance
```

#### Confusions typiques (matrice de confusion 10×10)

Certaines paires de chiffres se ressemblent visuellement :

| Chiffre réel | Parfois prédit | Raison                                |
| ------------ | -------------- | ------------------------------------- |
| 4            | 9              | Même boucle fermée en haut            |
| 3            | 8              | Arcs similaires (le 8 est un 3 fermé) |
| 1            | 7              | Trait vertical prédominant            |
| 5            | 6              | Courbe inférieure similaire           |

La matrice 10×10 affichée après l'entraînement montre exactement
combien d'exemples de chaque classe tombent dans chaque prédiction.

#### Prétraitement du dessin (`preprocess_digit_image`)

```python
# 1. Extraire le dessin du format Sketchpad Gradio
composite = sketch_dict["composite"]   # image RGBA

# 2. Convertir en niveaux de gris
img_gray = img.convert("L")

# 3. Si fond clair (dessin sur blanc) → inverser  (MNIST = blanc sur noir)
if arr.mean() > 127:
    arr = 255 - arr

# 4. Normaliser + redimensionner à 28×28
arr = arr / 255.0
arr = img.resize((28, 28), LANCZOS)

# 5. Format batch Keras : (1, 28, 28, 1)
return arr[np.newaxis, ..., np.newaxis]
```

Exemple : vous dessinez un "5" (trait blanc, fond noir 280×280) →

```
Redimensionné à 28×28 → pixels de valeur 0.0 à 1.0
model([arr]) → [0.00, 0.00, 0.01, 0.00, 0.00, 0.96, 0.01, 0.00, 0.01, 0.01]
                                                      ↑
Prédit "5" avec 96% de confiance
```

#### Utilisation

```
1. « Charger MNIST »         → télécharge ~11 MB, affiche 20 exemples
2. Sliders : époques / batch size / taux d'apprentissage
3. « Entraîner le CNN »      → log streaming + courbes loss/accuracy + matrice 10×10
4. Dessinez un chiffre blanc sur fond noir dans le canvas (280×280)
5. « Reconnaître le chiffre » → classe prédite + barres de probabilités 0–9
```

---

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
# Côté core (ex : image_trainer.py, mnist_model.py)
yield ("epoch", ep, total, loss, acc)
yield ("epoch", ep, total, loss, acc, val_loss, val_acc)   # MNIST : inclut val
yield ("done", model, loss_hist, class_names, preds, actuals)
yield ("done", model, history_dict)                        # MNIST : history complet

# Côté app.py
for update in train_cnn_model(X_train, y_train, X_test, y_test, ...):
    if update[0] == "epoch":
        yield state, log, None, None, ""   # → Gradio stream
    elif update[0] == "done":
        yield state, log, curve_fig, cm_fig, summary
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
