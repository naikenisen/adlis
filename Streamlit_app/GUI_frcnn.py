import streamlit as st
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN_ResNet50_FPN_V2_Weights, FastRCNNPredictor
)
from functools import partial
from PIL import Image, ImageDraw
import torchvision.transforms as T
from collections import Counter
from torchvision.ops import nms
from torchvision import models
from torchvision.transforms import v2

# Configuration de la page Streamlit
st.set_page_config(page_title="Pipeline Détection et Classification", layout="wide")

# Couleurs d’annotation
colors = {"Sideroblastes en couronne": "red", "Érythroblastes": "blue"}

@st.cache_resource
def create_model(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.eval()
    return model

@st.cache_resource
def load_detection_model(model_path, device):
    model = create_model(num_classes=2)

    checkpoint = torch.load(model_path, map_location=device)

    # Vérifie si c’est un dict avec une clé "model_state_dict"
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint  # dans le cas d’un simple state_dict

    model.load_state_dict(state_dict)
    model.to(device)
    return model


@st.cache_resource
def load_classifier(path, device):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 512), torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(64, 2)
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval().to(device)
    return model

# Transformations pour la classification
inference_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_fasterrcnn(image, model, device, threshold=0.5, iou_threshold=0.4):
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(image_tensor)[0]

    boxes = predictions['boxes']
    scores = predictions['scores']

    mask = scores > threshold
    boxes = boxes[mask]
    scores = scores[mask]

    keep = nms(boxes, scores, iou_threshold=iou_threshold)
    boxes = boxes[keep]

    return boxes.cpu().numpy()

def classify_crop(image, model, device):
    image_tensor = inference_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return "Sideroblastes en couronne" if predicted.item() == 0 else "Érythroblastes"

def draw_boxes(image, boxes, labels):
    draw = ImageDraw.Draw(image)
    for box, label in zip(boxes, labels):
        draw.rectangle(box.tolist(), outline=colors[label], width=3)
    return image

def main():
    st.title("Pipeline Détection et Classification")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    uploaded_files = st.file_uploader("Importer des images à analyser", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    detection_model_path = st.sidebar.file_uploader("Modèle Faster R-CNN (.pth)", type="pth")
    classifier_model_path = st.sidebar.file_uploader("Modèle de classification (.pth)", type="pth")

    confidence = st.slider("Seuil de confiance détection", 0.1, 1.0, 0.5, 0.05)

    if uploaded_files and detection_model_path and classifier_model_path:
        with open("detection_model.pth", "wb") as f:
            f.write(detection_model_path.getbuffer())
        with open("classifier_model.pth", "wb") as f:
            f.write(classifier_model_path.getbuffer())

        detection_model = load_detection_model("detection_model.pth", device)
        classifier_model = load_classifier("classifier_model.pth", device)

        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            st.markdown(f"## Résultat pour : `{uploaded_file.name}`")

            with st.spinner("Traitement en cours..."):
                boxes = predict_fasterrcnn(image, detection_model, device, threshold=confidence)

                if len(boxes) > 0:
                    labels = []
                    for box in boxes:
                        cropped_img = image.crop(box)
                        label = classify_crop(cropped_img, classifier_model, device)
                        labels.append(label)

                    class_counts = Counter(labels)

                    image_annotated = draw_boxes(image.copy(), boxes, labels)
                    st.image(image_annotated, caption="Résultat de l'analyse", use_container_width=True)

                    st.markdown("### Légende")
                    for label, color in colors.items():
                        st.markdown(
                            f"<span style='display:inline-block;width:20px;height:20px;background-color:{color};margin-right:10px;'></span>{label}",
                            unsafe_allow_html=True
                        )

                    st.success(f"{len(boxes)} objets détectés et classifiés.")
                    st.markdown("### Nombre de cellules par classe")
                    for label, count in class_counts.items():
                        st.markdown(f"- **{label}** : {count}")
                else:
                    st.info("Aucun objet détecté avec le seuil choisi.")

if __name__ == '__main__':
    main()
