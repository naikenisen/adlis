"""
Détection de cellules d'intérêt avec un modèle Faster R-CNN
============================================================

Script d'introduction (cours) : inférence d'un modèle de détection d'objets
sur une image de frottis sanguin, puis affichage du résultat avec matplotlib.

Tout tient dans ce seul fichier. Les seules choses dont il a besoin sont :
  - le fichier de poids du modèle (.pth)
  - une image à analyser (.jpg)

Pour l'exécuter comme un notebook : copiez chaque section dans une cellule,
ou lancez simplement le fichier avec  `python detection.py`.
"""

# ---------------------------------------------------------------------------
# 1) Importation des bibliothèques
# ---------------------------------------------------------------------------
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


# ---------------------------------------------------------------------------
# 2) Paramètres à régler
# ---------------------------------------------------------------------------

# Chemin vers le fichier de poids du modèle (entraîné au préalable)
CHEMIN_MODELE = "/home/naiken/coding/adlis/Streamlit_app/detection_model.pth"

# Chemin vers l'image à analyser
CHEMIN_IMAGE = "/home/naiken/coding/adlis/object_detection_dataset/dataset_split/test/images/Image_20240604_102333934.jpg"

# Seuil de confiance : on ne garde que les détections dont le score est
# supérieur à cette valeur (entre 0 et 1). Plus c'est haut, moins on a de boîtes.
SEUIL = 0.5

# On utilise le GPU s'il est disponible, sinon le processeur (CPU)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Calcul effectué sur : {DEVICE}")


# ---------------------------------------------------------------------------
# 3) Construction du modèle (Faster R-CNN ResNet50-FPN v2)
# ---------------------------------------------------------------------------
# Notre modèle distingue 2 classes : le fond (arrière-plan) et "Cellule".
NB_CLASSES = 2

# On part de l'architecture Faster R-CNN fournie par torchvision...
modele = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)

# ... puis on remplace la dernière couche pour qu'elle prédise nos 2 classes.
nb_features = modele.roi_heads.box_predictor.cls_score.in_features
modele.roi_heads.box_predictor = (
    torchvision.models.detection.faster_rcnn.FastRCNNPredictor(nb_features, NB_CLASSES)
)


# ---------------------------------------------------------------------------
# 4) Chargement des poids entraînés
# ---------------------------------------------------------------------------
# Le fichier .pth contient un dictionnaire avec, entre autres, les poids du
# modèle sous la clé 'model_state_dict'.
checkpoint = torch.load(CHEMIN_MODELE, map_location=DEVICE)
modele.load_state_dict(checkpoint["model_state_dict"])

# On envoie le modèle sur le bon appareil et on le passe en mode "évaluation"
# (indispensable pour l'inférence : désactive l'apprentissage).
modele.to(DEVICE)
modele.eval()
print("Modèle chargé et prêt.")


# ---------------------------------------------------------------------------
# 5) Préparation de l'image
# ---------------------------------------------------------------------------
# On ouvre l'image et on la convertit en RGB (3 canaux de couleur).
image_pil = Image.open(CHEMIN_IMAGE).convert("RGB")

# On transforme l'image en tableau de nombres, normalisés entre 0 et 1.
image_np = np.array(image_pil, dtype=np.float32) / 255.0

# Le modèle attend un tenseur de forme (canaux, hauteur, largeur).
image_tensor = torch.tensor(image_np).permute(2, 0, 1)


# ---------------------------------------------------------------------------
# 6) Inférence : on demande au modèle ses prédictions
# ---------------------------------------------------------------------------
# torch.no_grad() : on ne calcule pas les gradients (plus rapide, moins de mémoire).
with torch.no_grad():
    predictions = modele([image_tensor.to(DEVICE)])

# Le modèle renvoie une liste (une entrée par image). On prend la première.
prediction = predictions[0]

# On récupère les boîtes, les scores et les étiquettes, ramenés sur le CPU.
boites = prediction["boxes"].cpu().numpy()    # [xmin, ymin, xmax, ymax]
scores = prediction["scores"].cpu().numpy()   # confiance entre 0 et 1

# On ne conserve que les détections suffisamment sûres.
boites_gardees = boites[scores >= SEUIL]
scores_gardes = scores[scores >= SEUIL]
print(f"Nombre de cellules détectées : {len(boites_gardees)}")


# ---------------------------------------------------------------------------
# 7) Affichage du résultat avec matplotlib
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 8))
ax.imshow(image_pil)

# Pour chaque détection, on dessine un rectangle autour de la cellule.
for (xmin, ymin, xmax, ymax), score in zip(boites_gardees, scores_gardes):
    rectangle = patches.Rectangle(
        (xmin, ymin),            # coin supérieur gauche
        xmax - xmin,             # largeur
        ymax - ymin,             # hauteur
        linewidth=2,
        edgecolor="lime",
        facecolor="none",
    )
    ax.add_patch(rectangle)
    ax.text(
        xmin, ymin - 5,
        f"{score:.2f}",
        color="lime",
        fontsize=9,
        weight="bold",
    )

ax.set_title(f"Cellules détectées : {len(boites_gardees)} (seuil = {SEUIL})")
ax.axis("off")
plt.tight_layout()
plt.show()
