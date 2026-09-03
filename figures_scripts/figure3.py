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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import xml.etree.ElementTree as ET

# Configuration des chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.append(ROOT_DIR)

# Import du modèle de détection existant
from detection.model import create_model as create_fasterrcnn_model

IMAGES_DIR = os.path.join(ROOT_DIR, "dataset", "images")
ANNOTATIONS_DIR = os.path.join(ROOT_DIR, "dataset", "annotations")
METADATA_PATH = os.path.join(ROOT_DIR, "dataset", "metadata.csv")
SPLIT_PATH = os.path.join(ROOT_DIR, "dataset", "split.csv")
DETECTION_WEIGHTS = os.path.join(ROOT_DIR, "weights", "detection.pth")
CLASSIFICATION_WEIGHTS = os.path.join(ROOT_DIR, "weights", "classification.pth")
OUTPUT_FIGURE_PATH = os.path.join(ROOT_DIR, "figures_export", "figure3.png")

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

def ratio_to_category(sc, sn):
    if sn == 0:
        ratio = 100.0 if sc > 0 else 0.0
    else:
        ratio = (sc / sn) * 100.0
        
    if ratio < 5:
        return '<5%'
    elif 5 <= ratio <= 14:
        return 'entre 5 et 14 %'
    else:
        return '>15%'

def get_xml_counts(xml_path):
    if not os.path.exists(xml_path):
        return 0, 0
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        sc_count = 0
        sn_count = 0
        for obj in root.findall('object'):
            name_node = obj.find('name')
            if name_node is not None:
                name = name_node.text
                if name == 'SC':
                    sc_count += 1
                elif name == 'SN':
                    sn_count += 1
        return sc_count, sn_count
    except Exception as e:
        print(f"Error reading {xml_path}: {e}")
        return 0, 0

def row_normalize(cm: np.ndarray) -> np.ndarray:
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    return cm / row_sum

def make_annotation(cm: np.ndarray) -> np.ndarray:
    pct = row_normalize(cm)
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{int(cm[i, j])}\n({pct[i, j]*100:.0f}%)"
    return annot

def main():
    print(f"Using device: {device}")
    
    # Chargement des métadonnées
    df_meta = pd.read_csv(METADATA_PATH)
    df_split = pd.read_csv(SPLIT_PATH)
    
    # Fusionner pour avoir split et patient (directory_name) pour chaque image
    df = pd.merge(df_split, df_meta[['filename', 'directory_name']], on='filename', how='left')
    df = df.dropna(subset=['directory_name'])
    
    # Filtrer les images qui existent physiquement
    valid_images = []
    for _, row in df.iterrows():
        img_path = os.path.join(IMAGES_DIR, row['filename'])
        if os.path.exists(img_path):
            valid_images.append(row)
    df = pd.DataFrame(valid_images)
    
    print(f"Total number of images to process: {len(df)}")
    
    # Calculer le ground truth (annoté) par patient
    print("Computing ground truth annotations per patient...")
    patient_true_counts = {}
    for _, row in df.iterrows():
        patient = row['directory_name']
        filename = row['filename']
        base_name = os.path.splitext(filename)[0]
        xml_path = os.path.join(ANNOTATIONS_DIR, f"{base_name}.xml")
        
        if patient not in patient_true_counts:
            patient_true_counts[patient] = {'true_SC': 0, 'true_SN': 0}
            
        sc, sn = get_xml_counts(xml_path)
        patient_true_counts[patient]['true_SC'] += sc
        patient_true_counts[patient]['true_SN'] += sn
    
    # Dictionnaire pour stocker les prédictions par patient
    patient_results = {}
    
    # Chargement des modèles
    print("Loading models...")
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
    
    print("Starting image processing...")
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
            split = row['split']
            patient = row['directory_name']
            
            if split not in patient_results:
                patient_results[split] = {}
            if patient not in patient_results[split]:
                true_sc = patient_true_counts[patient]['true_SC']
                true_sn = patient_true_counts[patient]['true_SN']
                true_cat = ratio_to_category(true_sc, true_sn)
                patient_results[split][patient] = {'SC': 0, 'SN': 0, 'true_cat': true_cat}
                
            img_path = os.path.join(IMAGES_DIR, row['filename'])
            image = Image.open(img_path).convert("RGB")
            
            img_tensor = det_transform(image).unsqueeze(0).to(device)
            preds = det_model(img_tensor)[0]
            
            boxes = preds["boxes"]
            scores = preds["scores"]
            
            mask = scores > 0.5
            boxes = boxes[mask]
            scores = scores[mask]
            
            keep = nms(boxes, scores, iou_threshold=0.4)
            boxes = boxes[keep]
            
            if len(boxes) == 0:
                continue
            
            crops = []
            for box in boxes:
                box_np = box.cpu().numpy()
                crop = image.crop(box_np)
                crops.append(cls_transform(crop))
            
            crops_tensor = torch.stack(crops).to(device)
            
            cls_outputs = cls_model(crops_tensor)
            _, predicted = torch.max(cls_outputs, 1)
            
            sc_count = (predicted == 0).sum().item()
            sn_count = (predicted == 1).sum().item()
            
            patient_results[split][patient]['SC'] += sc_count
            patient_results[split][patient]['SN'] += sn_count

    print("Generating figure...")
    display_categories = ['<5%', '[5;14]', '≥15%']
    internal_categories = ['<5%', 'entre 5 et 14 %', '>15%']
    splits = ['train', 'valid', 'test']
    display_names = {"train": "Train", "valid": "Valid", "test": "Test"}
    
    y_trues = {s: [] for s in splits}
    y_preds = {s: [] for s in splits}
    
    true_ratios = {s: [] for s in splits}
    pred_ratios = {s: [] for s in splits}
    
    def calc_ratio(sc, sn):
        if sn == 0:
            return 100.0 if sc > 0 else 0.0
        return (sc / sn) * 100.0
    
    for split in splits:
        if split in patient_results:
            for patient, data in patient_results[split].items():
                pred_cat = ratio_to_category(data['SC'], data['SN'])
                y_trues[split].append(data['true_cat'])
                y_preds[split].append(pred_cat)
                
                # Ratios quantitatifs
                true_sc = patient_true_counts[patient]['true_SC']
                true_sn = patient_true_counts[patient]['true_SN']
                true_ratios[split].append(calc_ratio(true_sc, true_sn))
                pred_ratios[split].append(calc_ratio(data['SC'], data['SN']))

    # Style pour matplotlib
    rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.6,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'figure.dpi': 300,
        'savefig.dpi': 300
    })
    sns.set_theme(context='paper', style='ticks')

    mm = 1/25.4
    fig_w, fig_h = 200*mm, 200*mm  
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(3, 3, wspace=0.5, hspace=0.6, left=0.08, right=0.90, top=0.95, bottom=0.08)

    panel_letters = [['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']]
    mappable_for_cbar = None
    axes_top = []

    for col, name in enumerate(splits):
        # 1. Matrice de confusion
        ax_cm = fig.add_subplot(gs[0, col])
        axes_top.append(ax_cm)
        
        if len(y_trues[name]) == 0:
            ax_cm.set_title(f"{display_names[name]} Set (No Data)", pad=6, fontweight='bold', fontsize=9)
            ax_cm.axis('off')
            continue

        cm = confusion_matrix(y_trues[name], y_preds[name], labels=internal_categories)
        pct = row_normalize(cm)
        annot = make_annotation(cm)
        
        hm = sns.heatmap(
            pct,
            annot=annot,
            fmt='',
            cmap='Blues',
            vmin=0, vmax=1,
            ax=ax_cm,
            cbar=False,
            square=True,
            annot_kws={"size": 5.5}
        )
        
        ax_cm.set_title(f"{display_names[name]} Set", pad=15, fontweight='bold', fontsize=9)
        ax_cm.set_xlabel('Predictions (%)', fontsize=8)
        ax_cm.set_ylabel('Ground truth (%)', fontsize=8)
        ax_cm.set_xticklabels(display_categories, fontsize=6, rotation=45, ha='right')
        ax_cm.set_yticklabels(display_categories, rotation=0, fontsize=6, va='center')
        
        acc = accuracy_score(y_trues[name], y_preds[name])
        metrics_text = f"Accuracy {acc:.2f} ({len(y_trues[name])} patients)"
        ax_cm.text(0.5, 1.05, metrics_text, transform=ax_cm.transAxes, 
                ha='center', va='bottom', fontsize=7)
        
        ax_cm.text(-0.35, 1.15, panel_letters[0][col], transform=ax_cm.transAxes, 
                fontsize=10, fontweight='bold', va='bottom')
                
        if col == 2 or mappable_for_cbar is None:
            mappable_for_cbar = ax_cm.collections[0]

        # 2. Droite de régression (Scatter plot % pred vs % GT)
        ax_reg = fig.add_subplot(gs[1, col])
        tr = np.array(true_ratios[name])
        pr = np.array(pred_ratios[name])
        
        if len(tr) > 0:
            ax_reg.scatter(pr, tr, alpha=0.7, s=15, color='#1f77b4')
            
            # Droite de régression
            if len(pr) > 1:
                m, b = np.polyfit(pr, tr, 1)
                x_range = np.linspace(min(pr), max(pr), 100)
                ax_reg.plot(x_range, m*x_range + b, color='red', linestyle='--', linewidth=1, label=f'Fit: y={m:.2f}x+{b:.2f}')
            
            # Ligne y=x
            max_val = max(np.max(tr), np.max(pr))
            ax_reg.plot([0, max_val], [0, max_val], color='gray', linestyle=':', linewidth=1, label='y=x')
            
            ax_reg.set_xlabel('Predictions (%)', fontsize=8)
            ax_reg.set_ylabel('Ground truth (%)', fontsize=8)
        
        ax_reg.text(-0.35, 1.05, panel_letters[1][col], transform=ax_reg.transAxes, 
                fontsize=10, fontweight='bold', va='bottom')

        # 3. Bland-Altman Plot
        ax_ba = fig.add_subplot(gs[2, col])
        if len(tr) > 0:
            mean_vals = (tr + pr) / 2
            diff_vals = pr - tr
            md = np.mean(diff_vals)
            sd = np.std(diff_vals, axis=0) if len(diff_vals) > 1 else 0
            
            ax_ba.scatter(mean_vals, diff_vals, alpha=0.7, s=15, color='#2ca02c')
            ax_ba.axhline(md, color='red', linestyle='-', linewidth=1, label=f'Mean: {md:.2f}')
            if sd > 0:
                ax_ba.axhline(md + 1.96*sd, color='gray', linestyle='--', linewidth=1, label=f'+1.96 SD: {md+1.96*sd:.2f}')
                ax_ba.axhline(md - 1.96*sd, color='gray', linestyle='--', linewidth=1, label=f'-1.96 SD: {md-1.96*sd:.2f}')
            
            ax_ba.set_xlabel('Mean (Pred, GT) (%)', fontsize=8)
            ax_ba.set_ylabel('Diff (Pred - GT) (%)', fontsize=8)
            
        ax_ba.text(-0.35, 1.05, panel_letters[2][col], transform=ax_ba.transAxes, 
                fontsize=10, fontweight='bold', va='bottom')

    if mappable_for_cbar is not None:
        cbar = fig.colorbar(
            mappable_for_cbar, ax=axes_top, location='right', fraction=0.03, pad=0.02
        )

    plt.savefig(OUTPUT_FIGURE_PATH, bbox_inches='tight', dpi=300)
    print(f"\nFigure successfully saved: {OUTPUT_FIGURE_PATH}")

if __name__ == "__main__":
    main()
