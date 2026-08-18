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
from tqdm import tqdm

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
        for inputs, labels in tqdm(loader, desc="Evaluating test set"):
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

# Identify TP, TN, FP, FN
# For binary: class 0 (SC) is positive, class 1 (SN) is negative
TP = df[(df['true_idx'] == 0) & (df['pred_idx'] == 0)]
TN = df[(df['true_idx'] == 1) & (df['pred_idx'] == 1)]
FP = df[(df['true_idx'] == 1) & (df['pred_idx'] == 0)]
FN = df[(df['true_idx'] == 0) & (df['pred_idx'] == 1)]

def plot_all_boards(tp_df, tn_df, fp_df, fn_df, out_path, n=16):
    set_nature_style()
    sns.set_theme(context='paper', style='ticks')
    
    fig = plt.figure(figsize=(12, 12))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, figure=fig, wspace=0.1, hspace=0.3)
    
    categories = [
        ("Vrais Positifs (VP)", tp_df, gs[0, 0]),
        ("Vrais Négatifs (VN)", tn_df, gs[0, 1]),
        ("Faux Positifs (FP)", fp_df, gs[1, 0]),
        ("Faux Négatifs (FN)", fn_df, gs[1, 1])
    ]
    
    for title, df_subset, cell in categories:
        inner_gs = cell.subgridspec(4, 4, wspace=0.05, hspace=0.05)
        ax_title = fig.add_subplot(cell)
        ax_title.axis('off')
        ax_title.set_title(title, fontweight='bold', fontsize=12, pad=10)
        
        subset = df_subset.head(n)
        for i in range(16):
            r = i // 4
            c = i % 4
            ax = fig.add_subplot(inner_gs[r, c])
            ax.axis('off')
            if i < len(subset):
                img_path = subset.iloc[i]['image_path']
                try:
                    img = plt.imread(img_path)
                    ax.imshow(img)
                except Exception:
                    pass
                    
    output_path = os.path.join(project_root, "figures", out_path)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")

plot_all_boards(TP, TN, FP, FN, "figure_S_2.png", n=16)

print("\nBoard for VP/VN/FP/FN saved.")
