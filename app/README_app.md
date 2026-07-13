# ADLIS - CHU de Dijon

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.0+-brightgreen)](https://streamlit.io/)
[![Made with](https://img.shields.io/badge/Made%20with-%E2%9D%A4-red)](#)

**ADLIS** (Automatisation par Deep Learning de l’Identification des Sidéroblastes en Couronne) est une application **Streamlit** développée pour assister le diagnostic en hématologie. Elle permet de détecter et de classifier automatiquement les **sidéroblastes en couronne** à partir d’images microscopiques, grâce à des modèles **Faster R-CNN** et **ResNet18 custom**.  
L'application est conçue pour être simple, élégante et intuitive, tout en offrant des résultats fiables et mesurables.

---

## Fonctionnalités

### Détection & Classification
- **Détection** des cellules à l’aide du modèle **Faster R-CNN**.
- **Classification** des régions détectées via un **classifieur ResNet18 personnalisé**.
- **Affichage côte à côte** des images originales et annotées (boîtes englobantes + labels).

### Évaluation des Métriques
- Chargement automatique des **annotations Ground Truth** au format XML.
- Calcul de :
  - **Matrice de confusion**
  - **Classification report**
  - **F1-score**, **précision** et **rappel** (moyenne pondérée).

### Interface Graphique Streamlit
- **Thème personnalisé bleu ciel**, avec :
  - Dégradés doux sur fond et sidebar.
  - Boutons stylisés avec effets de survol.
- **Organisation en onglets** :
  - `Détection & Classification`
  - `Métriques`
  - `À propos`

---

## Design de l'Interface

### Couleurs
- **Fond principal** : `#E6F0FA` (bleu ciel clair)
- **Sidebar** : `#D4EAF5`
- **Bannière** : `#A9D3EE`
- **Boutons** :
  - Fond : `#0099CC`
  - Hover : `#0077AA`
  - Texte : `#FFFFFF`

### Polices
- **Police principale** : `Roboto`, sans-serif
- **Titres** : `Montserrat`, sans-serif

---

## Prérequis

- Python 3.7 ou supérieur
- [Streamlit](https://streamlit.io)
- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Pillow](https://python-pillow.org/)
- [Matplotlib](https://matplotlib.org/)

### Installation

```bash
pip install streamlit torch torchvision scikit-learn pillow matplotlib
```

---

## Structure du Projet

```
ADLIS/
├── App_Streamlit/
│   ├── app.py                # Application principale Streamlit         
├── GUI_pipeline/
│   └── test/
│       └── annotations/      # Fichiers XML d'annotations
├── models/                   # (optionnel) modèles .pth si inclus
├── assets/                   # Images de démonstration ou logos
└── README.md                 # Ce fichier
```

---

## Lancer l'application

Placez-vous dans le dossier `App_Streamlit` :

```bash
streamlit run app.py
```

---

## Utilisation

### 1. **Chargement des modèles**
- Depuis la **sidebar**, importez :
  - Le fichier `.pth` du modèle de **détection** (Faster R-CNN)
  - Le fichier `.pth` du modèle de **classification** (ResNet18 custom)

### 2. **Chargement des images**
- Importez une ou plusieurs **images** à analyser.

### 3. **Lancer l’analyse**
- Cliquez sur le bouton **"Lancer l’analyse"**.
- Résultats :
  - Images annotées affichées côte à côte.
  - Résultats sauvegardés automatiquement.

---

## Métriques

- Dans l’onglet **Métriques**, l’application :
  - Recherche les fichiers XML dans `GUI_pipeline/test/annotations/`
  - Calcule :
    - Matrice de confusion
    - Rapport de classification
    - F1-score, précision, rappel

---

## Crédits 
Modèles entraînés sur un dataset annoté de sidéroblastes.


