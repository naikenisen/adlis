import os
import streamlit as st
import torch
import torchvision
import re
import xml.etree.ElementTree as ET
import io
from PIL import Image, ImageDraw, ImageFont
from functools import partial
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import models
from torchvision.transforms import v2
from torchvision.ops import nms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
from collections import Counter

# --- Définition des chemins ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
ANNOTATIONS_DIR = os.path.join(ROOT_DIR, "GUI_pipeline", "test", "annotations")

st.set_page_config(
    page_title="ADLIS - CHU Dijon",
    layout="wide"
)

# --- CSS personnalisé avec background bleu ciel clair ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&family=Roboto:wght@400;500&display=swap');
    
    /* Forcer le fond du conteneur principal en dégradé bleu ciel clair */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Roboto', sans-serif;
        background: linear-gradient(90deg, #ADD8E6 30%, #FEE7EC 90%);
    }
    [data-testid="stHeader"] {
        background: linear-gradient(90deg, #ADD8E6 30%, #FEE7EC 90%);
    }
    
    /* Bannière principale */
    .main-title {
        text-align: center;
        background: linear-gradient(90deg, #6495ED 30%, #FFB6C1 90%) !important;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        color: #003366;
    }
    .main-title h1 {
        font-family: 'Montserrat', sans-serif;
        font-size: 3rem;
        margin: 0;
    }
    .main-title p {
        font-size: 1.2rem;
        margin: 8px 0 0;
    }
    
    /* Sidebar : identique à la bannière principale */
    [data-testid="stSidebar"] > div:first-child {
       background: linear-gradient(90deg, #8EC0FA 30%, #FFDDE0 90%) !important;
        padding: 20px;
        border-radius: 0 0 12px 0;
    }
    
    /* Bouton personnalisé */
    .stButton>button {
        background-color: #BCEBFF;
        color: #003366;
        border-radius: 8px;
        border: 2px solid #CCCCFF;
        padding: 10px 24px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: background-color 0.3s ease, transform 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #6495ED 30%, #FFB6C1 90%) !important;
        transform: scale(1.03);
    }
    
    /* Espacement des colonnes */
    div[data-testid="column"] {
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Bannière principale
st.markdown(
    """
    <div class="main-title">
        <h1>ADLIS - CHU de Dijon</h1>
        <p>Automatisation par Deep Learning de l’Identification des Sidéroblastes en Couronne</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Couleurs & Utilitaires ---
COLOR_SIDERO = "red"
COLOR_ERYTHRO = "blue"
colors_dict = {
    "Sideroblastes en couronne": COLOR_SIDERO,
    "Érythroblastes": COLOR_ERYTHRO
}

# --- Fonctions de chargement des modèles (Faster R-CNN & Classifieur) ---
@st.cache_resource
def create_fasterrcnn_model(num_classes=2):
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.eval()
    return model

@st.cache_resource
def load_fasterrcnn_model(model_path, device, num_classes=2):
    model = create_fasterrcnn_model(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    return model

@st.cache_resource
def load_classifier(model_path, device):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 512), torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(64, 2)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    return model

# --- Transformations pour la classification ---
inference_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])

# --- Fonctions de prédiction & classification ---
def predict_fasterrcnn(image, model, device, threshold=0.5, iou_threshold=0.4):
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(image_tensor)[0]
    boxes = preds["boxes"]
    scores = preds["scores"]
    mask = scores > threshold
    boxes = boxes[mask]
    scores = scores[mask]
    keep = nms(boxes, scores, iou_threshold=iou_threshold)
    boxes = boxes[keep]
    scores = scores[keep]
    return boxes.cpu().numpy(), scores.cpu().numpy()

def classify_crop(image, classifier, device):
    image_tensor = inference_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = classifier(image_tensor)
        _, predicted = torch.max(output, 1)
    return "Sideroblastes en couronne" if predicted.item() == 0 else "Érythroblastes"

def draw_boxes_and_labels(image, boxes, labels, scores=None):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for i, (box, label) in enumerate(zip(boxes, labels)):
        color = colors_dict.get(label, "green")
        draw.rectangle(box.tolist(), outline=color, width=3)
        txt = label
        if scores is not None:
            txt += f" ({scores[i]:.2f})"
        text_pos = (box[0], box[1] - 10)
        draw.text(text_pos, txt, fill=color, font=font)
    return image

# --- Extraction des annotations XML ---
def extract_annotations(xml_bytes):
    tree = ET.parse(io.BytesIO(xml_bytes))
    root = tree.getroot()
    gt_boxes = []
    gt_labels = []
    for obj in root.findall("object"):
        label = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        gt_boxes.append([xmin, ymin, xmax, ymax])
        gt_labels.append(label)
    return gt_boxes, gt_labels

# --- Calcul de l'IoU ---
def compute_iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

# --- Onglet : Détection & Classification ---
def detection_classification_tab():
    st.subheader("Détection & Classification")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    st.sidebar.subheader("FASTER RCNN & CLASSIFICATION")
    detection_model_path = st.sidebar.file_uploader("Fichier .pth (détection)", type="pth")
    classifier_model_path = st.sidebar.file_uploader("Fichier .pth (classification)", type="pth")
    confidence = st.sidebar.slider("Seuil de confiance (détection)", 0.1, 1.0, 0.5, 0.05)
    iou_threshold = 0.4

    uploaded_images = st.file_uploader(
        "Importer une ou plusieurs images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if st.button("Lancer l'analyse"):
        if not uploaded_images:
            st.warning("Veuillez importer au moins une image.")
            return
        if not detection_model_path:
            st.warning("Veuillez charger un modèle de détection.")
            return
        if not classifier_model_path:
            st.warning("Veuillez charger un modèle de classification.")
            return
        
        # Sauvegarde temporaire des poids
        with open("temp_det_model.pth", "wb") as f:
            f.write(detection_model_path.getbuffer())
        with open("temp_cls_model.pth", "wb") as f:
            f.write(classifier_model_path.getbuffer())
        
        st.info("Chargement des modèles...")
        detection_model = load_fasterrcnn_model("temp_det_model.pth", device, num_classes=2)
        classifier_model = load_classifier("temp_cls_model.pth", device)
        st.success("Modèles chargés avec succès.")
        
        if "results" not in st.session_state:
            st.session_state.results = []
        
        for img_file in uploaded_images:
            image = Image.open(img_file).convert("RGB")
            with st.spinner("Analyse en cours..."):
                boxes, scores = predict_fasterrcnn(image, detection_model, device,
                                                   threshold=confidence,
                                                   iou_threshold=iou_threshold)
                nb_detected = len(boxes)
                if nb_detected == 0:
                    st.info("Aucun objet détecté avec le seuil choisi.")
                else:
                    final_labels = []
                    for box in boxes:
                        crop = image.crop(box)
                        label = classify_crop(crop, classifier_model, device)
                        final_labels.append(label)
                    
                    annotated_img = draw_boxes_and_labels(image.copy(), boxes, final_labels, scores=scores)
                    
                    # Affichage côte à côte
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, use_container_width=True, caption="Image originale")
                    with col2:
                        st.image(annotated_img, use_container_width=True, caption="Détection & Classification")
                    
                    st.success(f"{nb_detected} objet(s) détecté(s).")
                    
                    counts = Counter(final_labels)
                    for k, v in counts.items():
                        st.write(f"- {k} : {v}")
                    
                    st.session_state.results.append({
                        "filename": img_file.name,
                        "pred_boxes": boxes,
                        "pred_scores": scores,
                        "pred_labels": final_labels
                    })
            st.markdown("---")

# --- Onglet : Métriques Automatisées ---
def metrics_tab():
    st.subheader("Métriques Automatisées")
    st.markdown("Les métriques sont calculées automatiquement à partir des résultats de la détection & classification et des annotations XML correspondantes dans le dossier d'annotations.")
    
    if "results" not in st.session_state or len(st.session_state.results) == 0:
        st.info("Aucun résultat disponible. Veuillez lancer l'analyse dans l'onglet Détection & Classification.")
        return

    aggregated_gt_labels = []
    aggregated_pred_labels = []
    iou_thresh = 0.5

    for res in st.session_state.results:
        filename = res["filename"]
        target = (filename + ".xml").lower()
        found_file = None
        for f in os.listdir(ANNOTATIONS_DIR):
            if f.lower() == target:
                found_file = f
                break
        if found_file is None:
            digits = re.findall(r'\d+', filename)
            substring = '_'.join(digits) if digits else filename.lower()
            for f in os.listdir(ANNOTATIONS_DIR):
                if substring in f.lower():
                    found_file = f
                    break
        if found_file is None:
            st.warning(f"Fichier d'annotation non trouvé pour {filename} dans {ANNOTATIONS_DIR}.")
            continue

        xml_path = os.path.join(ANNOTATIONS_DIR, found_file)
        with open(xml_path, "rb") as f:
            xml_bytes = f.read()
        gt_boxes, gt_labels = extract_annotations(xml_bytes)
        if not gt_boxes:
            st.warning(f"Aucune annotation trouvée dans {xml_path}.")
            continue

        matched_gt_labels = []
        matched_pred_labels = []
        used_gt = set()
        for i, p_box in enumerate(res["pred_boxes"]):
            best_iou = 0
            best_gt_index = -1
            for j, gt_box in enumerate(gt_boxes):
                if j in used_gt:
                    continue
                iou = compute_iou(p_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = j
            if best_iou >= iou_thresh and best_gt_index != -1:
                used_gt.add(best_gt_index)
                matched_gt_labels.append(gt_labels[best_gt_index])
                matched_pred_labels.append(res["pred_labels"][i])
        
        if not matched_gt_labels:
            st.warning(f"Aucun appariement trouvé pour {filename}.")
        else:
            aggregated_gt_labels.extend(matched_gt_labels)
            aggregated_pred_labels.extend(matched_pred_labels)
            st.write(f"Pour {filename} : {len(matched_gt_labels)} objets appariés.")

    if len(aggregated_gt_labels) == 0:
        st.error("Aucun appariement global n'a été trouvé. Vérifiez les annotations et la correspondance avec les prédictions.")
        return

    labels_order = list(set(aggregated_gt_labels + aggregated_pred_labels))
    cm = confusion_matrix(aggregated_gt_labels, aggregated_pred_labels, labels=labels_order)
    report = classification_report(aggregated_gt_labels, aggregated_pred_labels, target_names=labels_order)
    f1 = f1_score(aggregated_gt_labels, aggregated_pred_labels, average='weighted')
    precision = precision_score(aggregated_gt_labels, aggregated_pred_labels, average='weighted')
    recall = recall_score(aggregated_gt_labels, aggregated_pred_labels, average='weighted')
    
    st.markdown("#### Matrice de Confusion")
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=range(cm.shape[1]), yticks=range(cm.shape[0]),
           xticklabels=labels_order, yticklabels=labels_order,
           ylabel='Ground Truth', xlabel='Prédictions')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center", color="black")
    st.pyplot(fig)
    
    st.markdown("#### Rapport de Classification")
    st.text(report)
    
    st.markdown("#### Autres Métriques")
    st.write(f"**F1-score (moyenne pondérée) :** {f1:.2f}")
    st.write(f"**Précision (moyenne pondérée) :** {precision:.2f}")
    st.write(f"**Rappel (moyenne pondérée) :** {recall:.2f}")

# --- Onglet : À propos ---
def about_tab():
    st.subheader("À propos du projet ADLIS - CHU de Dijon")
    st.markdown("""
    **ADLIS** : Automatisation par Deep Learning de l’Identification des Sidéroblastes en Couronne  
    **Objectif** : Faciliter le diagnostic en hématologie via la détection et la classification automatiques des sidéroblastes en couronne.  
    **Équipe** :  
    - 
    **Contact** : contact@chu-dijon.fr
    """)

# --- Application principale ---
def main():
    tabs = st.tabs(["Détection & Classification", "Métriques", "À propos"])
    
    with tabs[0]:
        detection_classification_tab()
    with tabs[1]:
        metrics_tab()
    with tabs[2]:
        about_tab()

if __name__ == "__main__":
    main()
