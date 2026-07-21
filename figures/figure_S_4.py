#!/usr/bin/env python3
"""
Generate F1-score vs Confidence curve on the test split.
Finds the optimal confidence threshold that maximizes the F1-score.
"""
import os
import sys
import glob
import xml.etree.ElementTree as ET
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from tqdm.auto import tqdm

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from detection import model as od_model

images_dir = os.path.join(project_root, "dataset/images")
annot_dir = os.path.join(project_root, "dataset/annotations")
split_csv = os.path.join(project_root, "dataset/split.csv")
model_path = os.path.join(project_root, "weights/detection.pth")
output_path = os.path.join(project_root, "figures/figure_S_4.png")

iou_threshold = 0.5
min_score = 0.05  # lower bound for confidence to save computation

def compute_iou(box_a, box_b):
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter_w = max(0.0, x_b - x_a)
    inter_h = max(0.0, y_b - y_a)
    inter_area = inter_w * inter_h
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom

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

def find_annotation_file(annot_dir, image_path):
    image_name = os.path.basename(image_path)
    stem, _ = os.path.splitext(image_name)
    candidates = [
        os.path.join(annot_dir, f"{stem}.xml"),
        os.path.join(annot_dir, f"{image_name}.xml"),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    glob_candidates = sorted(glob.glob(os.path.join(annot_dir, f"{stem}*.xml")))
    if glob_candidates:
        return glob_candidates[0]
    return None

def parse_voc_boxes(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    labels = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        try:
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
        except (ValueError, AttributeError):
            continue
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(1)
    return boxes, labels

def load_model(model_path, device, num_classes=2):
    model = od_model.create_model(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test split images
    test_images = []
    with open(split_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['split'] == 'test':
                test_images.append(row['filename'])
                
    if not test_images:
        print("No test images found in split.csv")
        return

    model = load_model(model_path, device)
    
    image_paths = [os.path.join(images_dir, fname) for fname in test_images if os.path.exists(os.path.join(images_dir, fname))]
    print(f"Found {len(image_paths)} test images.")

    detections = []
    total_gt = 0

    progress = tqdm(image_paths, desc="Running Inference on Test Set")
    for image_path in progress:
        annotation_file = find_annotation_file(annot_dir, image_path)
        if not annotation_file:
            continue
            
        gt_boxes, _ = parse_voc_boxes(annotation_file)
        total_gt += len(gt_boxes)

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(np.transpose(image_rgb, (2, 0, 1))).float().unsqueeze(0)

        with torch.no_grad():
            output = model(image_tensor.to(device))[0]

        boxes = output["boxes"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy()

        keep = (scores >= min_score) & (labels > 0)
        boxes = boxes[keep]
        scores = scores[keep]

        # Sort by confidence
        order = np.argsort(-scores)
        boxes = boxes[order]
        scores = scores[order]

        used_gt = set()
        for pred_box, score in zip(boxes, scores):
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in used_gt:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            is_tp = 0
            is_fp = 1
            if best_iou >= iou_threshold and best_gt_idx != -1:
                is_tp = 1
                is_fp = 0
                used_gt.add(best_gt_idx)

            detections.append((float(score), is_tp, is_fp))

    if not detections:
        print("No detections made.")
        return

    # Sort all detections globally by confidence score descending
    detections.sort(key=lambda x: x[0], reverse=True)
    
    scores = np.array([d[0] for d in detections], dtype=np.float32)
    tp = np.array([d[1] for d in detections], dtype=np.float32)
    fp = np.array([d[2] for d in detections], dtype=np.float32)
    
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    recall = cum_tp / max(total_gt, 1)
    
    # Calculate F1 Score
    f1 = 2 * (precision * recall) / np.maximum(precision + recall, 1e-9)
    
    # Find optimal threshold
    best_idx = np.argmax(f1)
    best_conf = scores[best_idx]
    best_f1 = f1[best_idx]
    best_prec = precision[best_idx]
    best_rec = recall[best_idx]
    
    print("\n" + "="*50)
    print(f"Optimal Confidence Threshold: {best_conf:.4f}")
    print(f"Max F1-Score: {best_f1:.4f}")
    print(f"Precision at optimal: {best_prec:.4f}")
    print(f"Recall at optimal: {best_rec:.4f}")
    print("="*50)

    # Make Plot
    apply_nature_style()
    fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
    
    ax.plot(scores, f1, color="#1F77B4", label="F1-Score")
    ax.plot(scores, precision, color="#2CA02C", linestyle="--", alpha=0.7, label="Precision")
    ax.plot(scores, recall, color="#D62728", linestyle="--", alpha=0.7, label="Recall")
    
    # Mark the optimal point
    ax.axvline(x=best_conf, color="gray", linestyle=":", alpha=0.8)
    ax.scatter([best_conf], [best_f1], color="black", zorder=5)
    ax.text(best_conf + 0.02, best_f1, f"  Conf: {best_conf:.2f}\n  F1: {best_f1:.2f}", 
            va="center", ha="left", fontsize=8, fontweight="bold",
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
    
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_title("F1-Score vs Confidence Threshold", pad=10, fontweight="bold")
    ax.grid(alpha=0.35)
    ax.legend(loc="lower center", frameon=True, ncol=3, bbox_to_anchor=(0.5, -0.25))
    
    style_axis(ax)
    
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    root, _ = os.path.splitext(output_path)
    fig.savefig(f"{root}.pdf", bbox_inches="tight")
    print(f"Saved figure to {output_path}")

if __name__ == "__main__":
    main()
