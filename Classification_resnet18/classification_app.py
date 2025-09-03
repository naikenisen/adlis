import streamlit as st
from PIL import Image
import torch
import os
from torchvision import models
from torchvision.transforms import v2
from streamlit_tree_select import tree_select
import base64
from io import BytesIO

# Chargement du modèle
@st.cache_resource
def load_model(path):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 512), torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(64, 2)
    )
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

# Transformations pour inférence
inference_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.uint8, scale=True),
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

st.set_page_config(page_title="Diagnostic assisté par IA", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #FFFFFF;
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    h1, h2, h3 { color: #004080; }
    .img-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 10px;
    }
    .img-grid img {
        width: 100%;
        border-radius: 6px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Diagnostic assisté par Intelligence Artificielle")

# Sélection visuelle du répertoire avec tree_select
def build_tree(path):
    tree = {"label": os.path.basename(path), "value": path, "children": []}
    try:
        entries = os.listdir(path)
        for entry in entries:
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                tree["children"].append(build_tree(full_path))
    except PermissionError:
        pass
    return tree

with st.sidebar:
    st.header("Configuration")

    model_file = st.file_uploader("Charger le modèle IA (.pth)", type=["pth"])

    root_directory = st.text_input("Dossier racine pour exploration", value=".")
    if os.path.isdir(root_directory):
        tree_data = build_tree(root_directory)
        selected_folder = tree_select([tree_data], expand_on_click=True)

        folder_path = selected_folder['checked'][0] if selected_folder.get('checked') else None
    else:
        st.warning("Entrez un dossier racine valide.")
        folder_path = None

if model_file and folder_path:
    model_path = "uploaded_model.pth"
    with open(model_path, "wb") as f:
        f.write(model_file.getbuffer())

    model = load_model(model_path)

    class_images = {"Sideroblastes en couronne": [], "Érythroblastes": []}

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path).convert('RGB')

        image_tensor = inference_transforms(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)

        class_name = "Sideroblastes en couronne" if predicted.item() == 0 else "Érythroblastes"

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode()
        class_images[class_name].append(encoded_image)

    # Calcul du ratio
    nb_sideroblastes = len(class_images["Sideroblastes en couronne"])
    nb_erythroblastes = len(class_images["Érythroblastes"])
    total = nb_sideroblastes + nb_erythroblastes
    percent = (nb_sideroblastes / total * 100) if total else 0

    st.markdown(f"### Ratio sidéroblastes en couronnes / érythroblastes : {nb_sideroblastes}/{total} ({percent:.1f}%)")

    # Affichage mosaïque
    col1, col2 = st.columns(2)

    for idx, (class_name, images_encoded) in enumerate(class_images.items()):
        with [col1, col2][idx]:
            st.subheader(class_name)
            html_code = '<div class="img-grid">'
            for img_encoded in images_encoded:
                html_code += f'<img src="data:image/jpeg;base64,{img_encoded}">'
            html_code += '</div>'
            st.markdown(html_code, unsafe_allow_html=True)
