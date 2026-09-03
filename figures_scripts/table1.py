import os
import sys
import pandas as pd
import torch
import torchvision.transforms as T
from torchvision import models
from torchvision.transforms import v2
from torchvision.ops import nms
from PIL import Image
from tqdm import tqdm

# Configuration des chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.append(ROOT_DIR)

# Import du modèle de détection existant
from detection.model import create_model as create_fasterrcnn_model

IMAGES_DIR = os.path.join(ROOT_DIR, "dataset", "images")
METADATA_PATH = os.path.join(ROOT_DIR, "dataset", "metadata.csv")
SPLIT_PATH = os.path.join(ROOT_DIR, "dataset", "split.csv")
DETECTION_WEIGHTS = os.path.join(ROOT_DIR, "weights", "detection.pth")
CLASSIFICATION_WEIGHTS = os.path.join(ROOT_DIR, "weights", "classification.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_fasterrcnn_model(model_path, device, num_classes=2):
    model = create_fasterrcnn_model(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def load_classifier(model_path, device):
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 512), torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(64, 2)
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval().to(device)
    return model

def parse_sidero_count(val):
    if pd.isna(val):
        return None
    val_str = str(val).replace('%', '').replace(' ', '').strip()
    try:
        num = int(val_str)
        if num < 5:
            return '<5%'
        elif 5 <= num <= 14:
            return 'entre 5 et 14 %'
        else:
            return '>15%'
    except ValueError:
        return None

def main():
    print(f"Utilisation du device : {device}")
    
    # Chargement des métadonnées
    df_meta = pd.read_csv(METADATA_PATH)
    df_split = pd.read_csv(SPLIT_PATH)
    
    # Fusionner pour avoir split et category pour chaque image
    df = pd.merge(df_split, df_meta[['filename', 'sidero_count']], on='filename', how='left')
    df['category'] = df['sidero_count'].apply(parse_sidero_count)
    df = df.dropna(subset=['category'])
    
    # Filtrer les images qui existent physiquement
    valid_images = []
    for _, row in df.iterrows():
        img_path = os.path.join(IMAGES_DIR, row['filename'])
        if os.path.exists(img_path):
            valid_images.append(row)
    df = pd.DataFrame(valid_images)
    
    print(f"Nombre total d'images à traiter : {len(df)}")
    
    # Initialiser les compteurs
    # Structure: dict[split][category] = {'SC': 0, 'SN': 0}
    splits = ['train', 'valid', 'test']
    categories = ['<5%', 'entre 5 et 14 %', '>15%']
    results = {s: {c: {'SC': 0, 'SN': 0} for c in categories} for s in splits}
    
    # Chargement des modèles
    print("Chargement des modèles...")
    det_model = load_fasterrcnn_model(DETECTION_WEIGHTS, device)
    cls_model = load_classifier(CLASSIFICATION_WEIGHTS, device)
    
    import torchvision.transforms.v2.functional as TF
    class PadToSquare:
        def __init__(self, padding_mode='constant'):
            self.padding_mode = padding_mode

        def __call__(self, img):
            h, w = TF.get_size(img)
            max_size = max(h, w)
            pad_left = (max_size - w) // 2
            pad_right = max_size - w - pad_left
            pad_top = (max_size - h) // 2
            pad_bottom = max_size - h - pad_top
            
            fill_val = 0
            if self.padding_mode == 'constant':
                if isinstance(img, torch.Tensor):
                    top = img[:, 0:1, :]
                    bottom = img[:, -1:, :]
                    left = img[:, :, 0:1]
                    right = img[:, :, -1:]
                    
                    borders = torch.cat([
                        top.reshape(img.shape[0], -1),
                        bottom.reshape(img.shape[0], -1),
                        left.reshape(img.shape[0], -1),
                        right.reshape(img.shape[0], -1)
                    ], dim=1)
                    fill_val = borders.median(dim=1).values.tolist()

            return TF.pad(img, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=fill_val, padding_mode=self.padding_mode)

    det_transform = T.Compose([T.ToTensor()])
    cls_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        PadToSquare(padding_mode='constant'),
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    print("Début du traitement des images...")
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Inférence"):
            split = row['split']
            cat = row['category']
            
            # Au cas où le split aurait une casse inattendue ou ne serait pas dans nos clés prévues
            if split not in splits:
                if split not in results:
                    results[split] = {c: {'SC': 0, 'SN': 0} for c in categories}
            
            img_path = os.path.join(IMAGES_DIR, row['filename'])
            image = Image.open(img_path).convert("RGB")
            
            # Détection
            img_tensor = det_transform(image).unsqueeze(0).to(device)
            preds = det_model(img_tensor)[0]
            
            boxes = preds["boxes"]
            scores = preds["scores"]
            
            # Seuil de détection (par défaut 0.5 dans app.py)
            mask = scores > 0.5
            boxes = boxes[mask]
            scores = scores[mask]
            
            # NMS
            keep = nms(boxes, scores, iou_threshold=0.4)
            boxes = boxes[keep]
            
            if len(boxes) == 0:
                continue
            
            # Extraire les crops pour la classification
            crops = []
            for box in boxes:
                box_np = box.cpu().numpy()
                crop = image.crop(box_np)
                crops.append(cls_transform(crop))
            
            # Batcher la classification
            crops_tensor = torch.stack(crops).to(device)
            
            # Prédiction par batch (pour éviter OOM si trop de bounding boxes, on peut subdiviser, 
            # mais généralement < 100 par image donc ça passe)
            cls_outputs = cls_model(crops_tensor)
            _, predicted = torch.max(cls_outputs, 1)
            
            # 0 -> SC, 1 -> SN
            sc_count = (predicted == 0).sum().item()
            sn_count = (predicted == 1).sum().item()
            
            results[split][cat]['SC'] += sc_count
            results[split][cat]['SN'] += sn_count

    # Génération du tableau récapitulatif
    table_data = []
    for split in splits:
        if split not in results:
            continue
        for cat in categories:
            sc = results[split][cat]['SC']
            sn = results[split][cat]['SN']
            if sn == 0:
                ratio = "N/A"
            else:
                ratio = f"{(sc / sn) * 100:.2f} %"
            
            table_data.append([split, cat, sc, sn, ratio])
            
    print("\n" + "="*50)
    print("RÉSULTATS DE CLASSIFICATION SC / SN")
    print("="*50)
    
    # Utilisation de Pandas pour l'affichage au lieu de tabulate
    df_results = pd.DataFrame(table_data, columns=["Split", "Catégorie", "Somme SC", "Somme SN", "Ratio (SC / SN * 100)"])
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    main()
