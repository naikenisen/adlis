#create a script to evaluate my model "/home/naiken/coding/ADLIS/Streamlit_app/classifier_model.pth"
#using the testset : "Classification_resnet50/classification_dataset/test"
#by generating the F1-score and the Accuracy and a confusion matrix using matplotlib

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize

# Paths
MODEL_PATH = "/home/naiken/coding/.adlis/Streamlit_app/classifier_model.pth"
TRAINSET_PATH = "/home/naiken/coding/.adlis/classification_dataset/train"
VALIDSET_PATH = "/home/naiken/coding/.adlis/classification_dataset/valid"
TESTSET_PATH = "/home/naiken/coding/.adlis/classification_dataset/test"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Styling: Nature-like figure aesthetics
def set_nature_style():
    """Configure matplotlib for Nature-style figures"""
    mpl.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 8,
        'axes.linewidth': 0.6,
        'lines.linewidth': 1.5,
        'axes.grid': False,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
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

# Dataset classes (use train set as source of class order)
train_dataset = datasets.ImageFolder(root=TRAINSET_PATH, transform=test_transforms)
valid_dataset = datasets.ImageFolder(root=VALIDSET_PATH, transform=test_transforms)
test_dataset = datasets.ImageFolder(root=TESTSET_PATH, transform=test_transforms)

# Model (same architecture as training)
num_classes = len(train_dataset.classes)
model = models.resnet18(weights=None)
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
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

def evaluate_split(dataset: datasets.ImageFolder, batch_size: int = 64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return (
        np.array(all_labels),
        np.array(all_probs),
        np.array(all_preds),
    )

"""Evaluate Train, Valid, Test sets"""
splits = {
    "Train": train_dataset,
    "Valid": valid_dataset,
    "Test": test_dataset,
}

results = {}
for name, ds in splits.items():
    y_true, y_score, y_pred = evaluate_split(ds)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate sensitivity and specificity
    # For binary: cm[0,0]=TP, cm[0,1]=FN, cm[1,0]=FP, cm[1,1]=TN
    # Assuming class 0 (SC) is positive
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

for name in ["Train", "Valid", "Test"]:
    print(f"{name} Accuracy: {results[name]['acc']:.4f}")
    print(f"{name} F1-score: {results[name]['f1']:.4f}")
    print(f"{name} Sensitivity: {results[name]['sensitivity']:.4f}")
    print(f"{name} Specificity: {results[name]['specificity']:.4f}")

# Helper functions for confusion matrix visualization
def row_normalize(cm: np.ndarray) -> np.ndarray:
    """Normalize confusion matrix by row (proportions)"""
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    return cm / row_sum

def make_annotation(cm: np.ndarray) -> np.ndarray:
    """Generate annotations with counts and percentages"""
    pct = row_normalize(cm)
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{int(cm[i, j])}\n({pct[i, j]*100:.0f}%)"
    return annot

# Apply Nature styling
set_nature_style()
sns.set_theme(context='paper', style='ticks')

# Create figure with 2 rows and 3 columns
mm = 1/25.4
fig_w, fig_h = 180*mm, 130*mm  # Double-column width, taller for 2 rows
fig = plt.figure(figsize=(fig_w, fig_h))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.3, 
                       left=0.08, right=0.95, top=0.95, bottom=0.08)

# Class labels
class_labels = list(train_dataset.classes)

# Map technical labels to descriptive names
label_mapping = {
    'SC': 'ring sideroblasts',
    'SN': 'erythroblasts'
}
class_labels = [label_mapping.get(label, label) for label in class_labels]

# Color scheme
split_colors = {
    "Train": '#1b9e77',
    "Valid": '#d95f02',
    "Test":  '#7570b3',
}

panel_letters = ['A', 'B', 'C', 'D', 'E', 'F']
split_names = ["Train", "Valid", "Test"]

# Plot confusion matrices (top row)
for col, name in enumerate(split_names):
    ax = fig.add_subplot(gs[0, col])
    cm = results[name]["cm"]
    pct = row_normalize(cm)
    annot = make_annotation(cm)
    
    # Heatmap
    sns.heatmap(
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
    
    # Labels and title
    ax.set_title(f"{name} Set", pad=6, fontweight='bold', fontsize=9)
    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_ylabel('True', fontsize=8)
    ax.set_xticklabels(class_labels, fontsize=7)
    ax.set_yticklabels(class_labels, rotation=90, fontsize=7, va='center')
    
    # Metrics text below matrix
    metrics_text = (f"Acc {results[name]['acc']:.2f} • "
                   f"Sens {results[name]['sensitivity']:.2f} • "
                   f"Spec {results[name]['specificity']:.2f} • "
                   f"F1 {results[name]['f1']:.2f}")
    ax.text(0.5, -0.28, metrics_text, transform=ax.transAxes, 
            ha='center', va='top', fontsize=7)
    
    # Panel letter
    ax.text(-0.25, 1.08, panel_letters[col], transform=ax.transAxes, 
            fontsize=10, fontweight='bold', va='bottom')

# Plot ROC curves (bottom row)
for col, name in enumerate(split_names):
    ax = fig.add_subplot(gs[1, col])
    y_true = results[name]["y_true"]
    y_score = results[name]["y_score"]
    
    if num_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=split_colors[name], lw=1.5, 
                label=f'AUC = {roc_auc:.3f}')
    else:
        # Multi-class: plot macro-average ROC
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        fpr_dict = {}
        tpr_dict = {}
        for i in range(num_classes):
            fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        
        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_classes):
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
        mean_tpr /= num_classes
        
        roc_auc = auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, color=split_colors[name], lw=1.5,
                label=f'Macro AUC = {roc_auc:.3f}')
    
    # Chance line
    ax.plot([0, 1], [0, 1], color='gray', lw=1.0, linestyle='--', alpha=0.7)
    
    # Styling
    ax.set_xlabel('False Positive Rate', fontsize=8)
    ax.set_ylabel('True Positive Rate', fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', frameon=False, fontsize=7)
    ax.tick_params(labelsize=7)
    
    # Panel letter
    ax.text(-0.25, 1.08, panel_letters[col + 3], transform=ax.transAxes, 
            fontsize=10, fontweight='bold', va='bottom')

# Save figure
plt.savefig('classification_results_comprehensive.png', bbox_inches='tight')
plt.savefig('classification_results_comprehensive.pdf', bbox_inches='tight')
plt.savefig('classification_results_comprehensive.svg', bbox_inches='tight')
print("\nFigure saved: classification_results_comprehensive.[png/pdf/svg]")