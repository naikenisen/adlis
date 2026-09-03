import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

import os
import sys

# Set absolute paths from project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Paths
MODEL_PATH = os.path.join(project_root, "weights/classification.pth")
TRAINSET_PATH = os.path.join(project_root, "dataset/classification_set/train")
VALIDSET_PATH = os.path.join(project_root, "dataset/classification_set/valid")
TESTSET_PATH = os.path.join(project_root, "dataset/classification_set/test")
OUTPUT_FIGURE_PATH = os.path.join(project_root, "figures/figure_2A.png")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Styling: Grandes revues
rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,             # 7–9 pt classique
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
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none'
})

# Transforms (same as validation)
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset classes
train_dataset = datasets.ImageFolder(root=TRAINSET_PATH, transform=test_transforms)
valid_dataset = datasets.ImageFolder(root=VALIDSET_PATH, transform=test_transforms)
test_dataset = datasets.ImageFolder(root=TESTSET_PATH, transform=test_transforms)

num_classes = len(train_dataset.classes)
model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(64, num_classes)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
model = model.to(device)
model.eval()

def evaluate_split(dataset: datasets.ImageFolder, batch_size: int = 64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_labels, all_probs = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return (
        np.array(all_labels),
        np.array(all_probs)
    )

splits = {
    "Train": train_dataset,
    "Valid": valid_dataset,
    "Test": test_dataset,
}

# --- Calculate Optimal Threshold on Validation Set ---
y_true_val, y_score_val = evaluate_split(valid_dataset)
prob_sc_val = y_score_val[:, 0]
label_sc_val = 1 - y_true_val  # SC=1, SN=0

thresholds = np.linspace(0.01, 0.99, 99)
f1s = []
for t in thresholds:
    preds = np.where(prob_sc_val >= t, 1, 0)
    f1s.append(f1_score(label_sc_val, preds))
best_thresh = thresholds[np.argmax(f1s)]
print(f"Optimal SC Classification Threshold (Max F1): {best_thresh:.4f}")

# --- Generate predictions for all sets ---
results = {}
for name, ds in splits.items():
    if name == "Valid":
        y_true, y_score = y_true_val, y_score_val
    else:
        y_true, y_score = evaluate_split(ds)
        
    prob_sc = y_score[:, 0]
    # Apply optimal threshold: if prob_sc >= best_thresh -> 0 (SC), else 1 (SN)
    y_pred = np.where(prob_sc >= best_thresh, 0, 1)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    
    # Class 0 (SC) as positive for Sens/Spec
    TP, FN = cm[0, 0], cm[0, 1]
    FP, TN = cm[1, 0], cm[1, 1]
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    results[name] = {
        "y_true": y_true,
        "y_score": y_score,
        "y_pred": y_pred,
        "cm": cm,
        "acc": acc,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }

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

sns.set_theme(context='paper', style='ticks')

mm = 1/25.4
# We use 180mm width. Height 190mm for 3 rows.
fig_w, fig_h = 180*mm, 190*mm  
fig = plt.figure(figsize=(fig_w, fig_h))
# Constrained layout is better, but gridspec allows custom spacing
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.3, 
                       left=0.08, right=0.90, top=0.90, bottom=0.08)

# Add title showing the optimal threshold
fig.suptitle(f"Optimal SC Threshold: {best_thresh:.3f} (Max F1 Score)", 
             fontsize=10, fontweight='bold', color='#D62728', y=0.98)

class_labels = list(train_dataset.classes)
label_mapping = {
    'SC': 'SC',
    'SN': 'SN'
}
class_labels = [label_mapping.get(label, label) for label in class_labels]

split_colors = {
    "Train": '#1b9e77',
    "Valid": '#d95f02',
    "Test":  '#7570b3',
}

panel_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
split_names = ["Train", "Valid", "Test"]

# Top row: Confusion matrices
mappable_for_cbar = None
axes_top = []
for col, name in enumerate(split_names):
    ax = fig.add_subplot(gs[0, col])
    axes_top.append(ax)
    cm = results[name]["cm"]
    pct = row_normalize(cm)
    annot = make_annotation(cm)
    
    hm = sns.heatmap(
        pct,
        annot=annot,
        fmt='',
        cmap='Blues',
        vmin=0, vmax=1,
        ax=ax,
        cbar=False,
        square=True,
        annot_kws={"size": 7}
    )
    
    ax.set_title(f"{name} Set", pad=6, fontweight='bold', fontsize=9)
    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_ylabel('True', fontsize=8)
    ax.set_xticklabels(class_labels, fontsize=7)
    ax.set_yticklabels(class_labels, rotation=0, fontsize=7, va='center')
    
    metrics_text = (f"Acc {results[name]['acc']:.2f} • Sens {results[name]['sensitivity']:.2f}\n"
                   f"Spec {results[name]['specificity']:.2f} • F1 {results[name]['f1']:.2f}")
    ax.text(0.5, -0.3, metrics_text, transform=ax.transAxes, 
            ha='center', va='top', fontsize=7)
    
    ax.text(-0.25, 1.08, panel_letters[col], transform=ax.transAxes, 
            fontsize=10, fontweight='bold', va='bottom')
            
    if col == 2:
        mappable_for_cbar = ax.collections[0]

# Add colorbar
if mappable_for_cbar is not None:
    cbar = fig.colorbar(
        mappable_for_cbar, ax=axes_top, location='right', fraction=0.03, pad=0.02
    )
    cbar.set_label('Row-normalized proportion')

# Middle row: ROC curves
for col, name in enumerate(split_names):
    ax = fig.add_subplot(gs[1, col])
    y_true = results[name]["y_true"]
    y_score = results[name]["y_score"]
    
    # We invert labels so SC (0) is positive
    label_sc = 1 - y_true
    prob_sc = y_score[:, 0]
    
    fpr, tpr, _ = roc_curve(label_sc, prob_sc)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=split_colors[name], lw=1.5, 
            label=f'AUC = {roc_auc:.3f}')
    
    ax.plot([0, 1], [0, 1], color='gray', lw=1.0, linestyle='--', alpha=0.7)
    
    ax.set_xlabel('False Positive Rate (for SC)', fontsize=8)
    ax.set_ylabel('True Positive Rate (for SC)', fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', frameon=False, fontsize=7)
    ax.tick_params(labelsize=7)
    
    ax.text(-0.25, 1.08, panel_letters[col + 3], transform=ax.transAxes, 
            fontsize=10, fontweight='bold', va='bottom')

# Bottom row: PR curves
for col, name in enumerate(split_names):
    ax = fig.add_subplot(gs[2, col])
    y_true = results[name]["y_true"]
    y_score = results[name]["y_score"]
    
    # We invert labels so SC (0) is positive
    label_sc = 1 - y_true
    prob_sc = y_score[:, 0]
    
    precision, recall, _ = precision_recall_curve(label_sc, prob_sc)
    pr_auc = average_precision_score(label_sc, prob_sc)
    
    ax.plot(recall, precision, color=split_colors[name], lw=1.5, 
            label=f'AUC-PR = {pr_auc:.3f}')
    
    ax.set_xlabel('Recall (for SC)', fontsize=8)
    ax.set_ylabel('Precision (for SC)', fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.legend(loc='lower left', frameon=False, fontsize=7)
    ax.tick_params(labelsize=7)
    
    ax.text(-0.25, 1.08, panel_letters[col + 6], transform=ax.transAxes, 
            fontsize=10, fontweight='bold', va='bottom')

plt.savefig(OUTPUT_FIGURE_PATH, bbox_inches='tight', dpi=300)
root, _ = os.path.splitext(OUTPUT_FIGURE_PATH)
plt.savefig(f"{root}.pdf", bbox_inches='tight', dpi=300)
print(f"\nFigure saved: {OUTPUT_FIGURE_PATH}")
