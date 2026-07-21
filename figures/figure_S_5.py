#!/usr/bin/env python3
"""
Generate F1-score vs Classification Threshold curve on the classification test set.
Finds the optimal probability threshold for class SC to reduce false positives.
"""
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(project_root, "weights/classification.pth")
test_dir = os.path.join(project_root, "dataset/classification_set/test")
output_path = os.path.join(project_root, "figures/figure_S_5.png")

def load_classifier(model_path, device, num_classes=2):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 512), torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Dropout(0.5),
        torch.nn.Linear(64, num_classes)
    )
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model

def apply_nature_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "axes.facecolor": "#FCFCFD",
        "figure.facecolor": "white",
        "grid.color": "#D9DEE7",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.9,
        "savefig.dpi": 600,
        "figure.dpi": 150,
    })

def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#2E3440")
    ax.spines["bottom"].set_color("#2E3440")
    ax.tick_params(axis="both", width=0.8, length=3)

def main():
    print(f"Loading classifier on {device}...")
    model = load_classifier(model_path, device)
    
    valid_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_dataset = datasets.ImageFolder(root=test_dir, transform=valid_transforms)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"Found {len(test_dataset)} images in classification test set.")
    
    all_probs_sc = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Running Inference"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            
            # class 0 is SC, class 1 is SN
            # We want the probability of being SC
            prob_sc = probs[:, 0]
            
            all_probs_sc.extend(prob_sc.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    all_probs_sc = np.array(all_probs_sc)
    all_labels = np.array(all_labels)
    
    thresholds = np.linspace(0.01, 0.99, 99)
    precisions = []
    recalls = []
    f1s = []
    
    for t in thresholds:
        # Predict SC (0) if prob_sc >= t, else SN (1)
        preds = np.where(all_probs_sc >= t, 0, 1)
        
        # Calculate for SC (class 0)
        # True Positive: predicted SC and actual SC (label 0)
        tp = np.sum((preds == 0) & (all_labels == 0))
        # False Positive: predicted SC but actual SN (label 1)
        fp = np.sum((preds == 0) & (all_labels == 1))
        # False Negative: predicted SN but actual SC (label 0)
        fn = np.sum((preds == 1) & (all_labels == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
    f1s = np.array(f1s)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    
    best_idx = np.argmax(f1s)
    best_thresh = thresholds[best_idx]
    best_f1 = f1s[best_idx]
    best_prec = precisions[best_idx]
    best_rec = recalls[best_idx]
    
    print("\n" + "="*50)
    print(f"Optimal SC Classification Threshold: {best_thresh:.4f}")
    print(f"Max F1-Score (for SC): {best_f1:.4f}")
    print(f"Precision at optimal: {best_prec:.4f}")
    print(f"Recall at optimal: {best_rec:.4f}")
    print("="*50)

    # Plot
    apply_nature_style()
    fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
    
    ax.plot(thresholds, f1s, color="#1F77B4", label="F1-Score")
    ax.plot(thresholds, precisions, color="#2CA02C", linestyle="--", alpha=0.7, label="Precision")
    ax.plot(thresholds, recalls, color="#D62728", linestyle="--", alpha=0.7, label="Recall")
    
    ax.axvline(x=best_thresh, color="gray", linestyle=":", alpha=0.8)
    ax.scatter([best_thresh], [best_f1], color="black", zorder=5)
    ax.text(best_thresh + 0.02, best_f1, f"  Conf: {best_thresh:.2f}\n  F1: {best_f1:.2f}\n  Prec: {best_prec:.2f}", 
            va="center", ha="left", fontsize=8, fontweight="bold",
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
            
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("SC Probability Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_title("Classification Optimization (Class: SC)", pad=10, fontweight="bold")
    ax.grid(alpha=0.35)
    ax.legend(loc="lower center", frameon=True, ncol=3, bbox_to_anchor=(0.5, -0.3))
    
    style_axis(ax)
    
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    root, _ = os.path.splitext(output_path)
    fig.savefig(f"{root}.pdf", bbox_inches="tight")
    print(f"Saved figure to {output_path}")

if __name__ == "__main__":
    main()
