import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix

import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Paths
MODEL_PATH = os.path.join(project_root, "weights/classification.pth")
TESTSET_PATH = "/home/naiken/coding/adlis/dataset/classification_set/test"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Styling: Nature-like figure aesthetics
def set_nature_style():
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

test_dataset = datasets.ImageFolder(root=TESTSET_PATH, transform=test_transforms)
num_classes = len(test_dataset.classes)

# Model (same architecture as training)
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

# Get class names and mapping
class_to_idx = test_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Map technical labels to descriptive names
label_mapping = {
    'SC': 'ring sideroblasts',
    'SN': 'erythroblasts'
}

def get_image_paths(dataset):
    # Returns a list of image file paths in the same order as dataset samples
    return [s[0] for s in dataset.samples]

def evaluate_testset(dataset, batch_size=20):
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

y_true, y_score, y_pred = evaluate_testset(test_dataset)
image_paths = get_image_paths(test_dataset)

# Build DataFrame with results
df = pd.DataFrame({
    'image_path': image_paths,
    'true_idx': y_true,
    'pred_idx': y_pred,
})
df['true_label'] = df['true_idx'].map(idx_to_class)
df['pred_label'] = df['pred_idx'].map(idx_to_class)
df['true_label_desc'] = df['true_label'].map(label_mapping)
df['pred_label_desc'] = df['pred_label'].map(label_mapping)

# Identify FP and FN
# For binary: class 0 (SC) is positive, class 1 (SN) is negative
FP = df[(df['true_idx'] == 1) & (df['pred_idx'] == 0)]
FN = df[(df['true_idx'] == 0) & (df['pred_idx'] == 1)]

# Helper: get image thumbnails for plotting
def plot_examples(df, title, out_path, n=24):
    set_nature_style()
    sns.set_theme(context='paper', style='ticks')
    n = min(n, len(df))
    if n == 0:
        print(f"No examples for {title}")
        return
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2.2, nrows*2.2))
    axes = axes.flatten()
    for i, (idx, row) in enumerate(df.head(n).iterrows()):
        img = plt.imread(row['image_path'])
        axes[i].imshow(img)
        axes[i].axis('off')
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    fig.suptitle(title, fontsize=10, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved: {out_path}")

# Plot and save FP and FN boards
plot_examples(FP, "Faux positifs (FP)", "false_positives_board.png", n=20)
plot_examples(FN, "Faux négatifs (FN)", "false_negatives_board.png", n=20)

print("\nCSV and PNG boards for FP/FN saved.")
