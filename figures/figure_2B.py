""" 
A) Prédiction du pourcentage de SC par patients : le scripte reprendre la pipeline utilisée par
app/app.py, pour la détection suivie de la classification des SC et SN, sur chaques images des sous-dossiers de
dataset/test_externe. Pour chaque sous-dossiers de dataset/test_externe il détermine un pourcentrage
de sidéroblastes en couronnes avec la formule suivant ((SC/SC+SN)*100). 
Le scripte crée "dataset/inference-test-externe.csv" avec deux colonnes : "id" qui correspond au sous-dossiers
et "prediction" qui correspond au pourcentrage de sidéroblastes en couronnes.

B) Création du Bland-Altman plot et régression linéaire : le scripte reprendre 
"dataset/test-externe.csv" et "dataset/inference-test-externe.csv" pour produire 
un Bland-Altman plot et une regression linéaire pour comparer les "prediction" de 
"dataset/inference-test-externe.csv" aux "valeur" de "dataset/test-externe.csv". 
La correspondance se fera entre "id" de "dataset/test-externe.csv" et "id" de 
"dataset/inference-test-externe.csv".

"""
import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from torchvision.transforms import v2
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import models
from torchvision.ops import nms
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from tqdm import tqdm

# Set absolute paths from project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

test_externe_dir = os.path.join(project_root, "dataset/test_externe")
inference_csv = os.path.join(project_root, "dataset/inference-test-externe.csv")
ground_truth_csv = os.path.join(project_root, "dataset/test-externe.csv")
output_figure = os.path.join(project_root, "figures/figure_2C.png")

# Use best models. Change these paths if your models are elsewhere
det_model_path = os.path.join(project_root, "weights/detection.pth")
cls_model_path = os.path.join(project_root, "weights/classification.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_fasterrcnn_model(model_path, device, num_classes=2):
    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model

def load_classifier(model_path, device):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 512), torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(64, 2)
    )
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model

inference_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])

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
    return boxes.cpu().numpy()

def classify_crop(image, classifier, device):
    image_tensor = inference_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = classifier(image_tensor)
        _, predicted = torch.max(output, 1)
    # class 0 corresponds to SC, class 1 corresponds to SN
    return "SC" if predicted.item() == 0 else "SN"

def part_A():
    print("Loading models...")
    if not os.path.exists(det_model_path) or not os.path.exists(cls_model_path):
        print(f"Error: Could not find models at:\n- {det_model_path}\n- {cls_model_path}")
        return

    det_model = load_fasterrcnn_model(det_model_path, device)
    cls_model = load_classifier(cls_model_path, device)

    results = []
    
    if not os.path.exists(test_externe_dir):
        print(f"Error: Directory {test_externe_dir} not found.")
        return

    print(f"Scanning {test_externe_dir}...")
    subfolders = [f.path for f in os.scandir(test_externe_dir) if f.is_dir()]
    
    for folder in tqdm(subfolders, desc="Processing Patients"):
        patient_id = os.path.basename(folder)
        image_files = glob.glob(os.path.join(folder, "*.*"))
        image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        sc_count = 0
        sn_count = 0
        
        for img_path in tqdm(image_files, desc=f"Patient {patient_id}", leave=False):
            try:
                image = Image.open(img_path).convert("RGB")
                boxes = predict_fasterrcnn(image, det_model, device)
                
                for box in boxes:
                    crop = image.crop(box)
                    label = classify_crop(crop, cls_model, device)
                    if label == "SC":
                        sc_count += 1
                    else:
                        sn_count += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        total = sc_count + sn_count
        percentage = (sc_count / total * 100.0) if total > 0 else 0.0
        
        print(f"Patient {patient_id}: SC={sc_count}, SN={sn_count}, %SC={percentage:.2f}%")
        results.append({"id": patient_id, "prediction": percentage})
        
    df_results = pd.DataFrame(results)
    df_results.to_csv(inference_csv, index=False)
    print(f"Predictions saved to {inference_csv}")

def part_B():
    print("Generating plots...")
    if not os.path.exists(inference_csv) or not os.path.exists(ground_truth_csv):
        print(f"Missing CSV files for plotting:\n- {inference_csv}\n- {ground_truth_csv}")
        return
        
    df_pred = pd.read_csv(inference_csv)
    df_gt = pd.read_csv(ground_truth_csv)
    
    # Safe string conversion to merge patient IDs correctly
    df_pred['id'] = df_pred['id'].astype(str).str.strip()
    df_gt['id'] = df_gt['id'].astype(str).str.strip()
    
    df = pd.merge(df_gt, df_pred, on='id', how='inner')
    
    if df.empty:
        print("Merged dataframe is empty! Check the IDs in both CSVs.")
        return
        
    valeur = df['valeur'].values
    prediction = df['prediction'].values
    
    # Bland-Altman
    mean = np.mean([valeur, prediction], axis=0)
    diff = prediction - valeur
    md = np.mean(diff)
    sd = np.std(diff, axis=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Linear Regression
    ax1.scatter(valeur, prediction, alpha=0.7, color='#1F77B4')
    
    reg = LinearRegression().fit(valeur.reshape(-1, 1), prediction)
    line_x = np.array([min(valeur), max(valeur)])
    line_y = reg.predict(line_x.reshape(-1, 1))
    ax1.plot(line_x, line_y, color='red', label='Regression Line')
    
    ideal_x = np.array([0, max(valeur)])
    ax1.plot(ideal_x, ideal_x, color='gray', linestyle='--', label='y = x')
    
    r, p = pearsonr(valeur, prediction)
    ax1.text(0.05, 0.95, f'r = {r:.2f}\np = {p:.2e}', transform=ax1.transAxes, 
             fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
             
    ax1.set_xlabel('Ground Truth (%)')
    ax1.set_ylabel('Prediction (%)')
    ax1.set_title('Linear Regression')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Plot 2: Bland-Altman
    ax2.scatter(mean, diff, alpha=0.7, color='#1F77B4')
    ax2.axhline(md, color='red', linestyle='-', label=f'Mean Diff = {md:.2f}')
    ax2.axhline(md + 1.96*sd, color='gray', linestyle='--', label=f'+1.96 SD = {md + 1.96*sd:.2f}')
    ax2.axhline(md - 1.96*sd, color='gray', linestyle='--', label=f'-1.96 SD = {md - 1.96*sd:.2f}')
    
    ax2.set_xlabel('Mean of GT and Prediction (%)')
    ax2.set_ylabel('Difference (Prediction - GT) (%)')
    ax2.set_title('Bland-Altman Plot')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_figure, dpi=300)
    plt.close()
    print(f"Figure saved to {output_figure}")

if __name__ == "__main__":
    if not os.path.exists(inference_csv):
        part_A()
    else:
        print(f"Inference file {inference_csv} already exists.")
        print("Skipping part A (delete the CSV to rerun inference on images).")
    
    part_B()
