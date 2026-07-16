#!/usr/bin/env python3
"""
Evaluate Faster R-CNN object detection on train/valid/test splits and generate a
single matplotlib figure with:
- mAP@IoU (single-class AP for each split)
- Precision-Recall curves
- IoU CDF curves (for matched true positives)

Expected dataset layout:
dataset/
  images/
  annotations/
  split.csv
"""

import argparse
import glob
import importlib
import importlib.util
import os
import sys
import xml.etree.ElementTree as ET
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    tqdm = importlib.import_module("tqdm.auto").tqdm
except ImportError as exc:
    raise ImportError(
        "tqdm is required for progress reporting. Install with: pip install tqdm"
    ) from exc

try:
    cv2 = importlib.import_module("cv2")
except ImportError as exc:
    raise ImportError(
        "OpenCV is required for classifier_eval.py. Install with: pip install opencv-python"
    ) from exc


def compute_iou(box_a, box_b):
    """Compute IoU between two boxes [xmin, ymin, xmax, ymax]."""
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
    """Apply a modern publication style inspired by top-tier journal figures."""
    plt.rcParams.update(
        {
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
            "lines.markersize": 3,
            "savefig.dpi": 600,
            "figure.dpi": 150,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax):
    """Reduce clutter and keep print-friendly axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#2E3440")
    ax.spines["bottom"].set_color("#2E3440")
    ax.tick_params(axis="both", width=0.8, length=3)


def compute_ap(recall, precision):
    """Compute AP from precision-recall curve using integral method."""
    if len(recall) == 0 or len(precision) == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    change_idxs = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[change_idxs + 1] - mrec[change_idxs]) * mpre[change_idxs + 1])
    return float(ap)


def find_annotation_file(annot_dir, image_path):
    """Find matching XML annotation file for an image."""
    image_name = os.path.basename(image_path)
    stem, _ = os.path.splitext(image_name)

    candidates = [
        os.path.join(annot_dir, f"{stem}.xml"),
        os.path.join(annot_dir, f"{image_name}.xml"),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand

    # Fallback for files such as Image_xxx.jpg.xml.
    glob_candidates = sorted(glob.glob(os.path.join(annot_dir, f"{stem}*.xml")))
    if glob_candidates:
        return glob_candidates[0]
    return None


def parse_voc_boxes(xml_path):
    """Parse Pascal VOC boxes and map all known classes to label=1."""
    if xml_path is None or not os.path.exists(xml_path):
        return [], []

    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    labels = []
    for obj in root.findall("object"):
        name_elem = obj.find("name")
        bbox = obj.find("bndbox")
        if name_elem is None or bbox is None:
            continue

        class_name = (name_elem.text or "").strip()
        # The detector is trained as single foreground class.
        if class_name not in {"SN", "SC", "Cellule"}:
            class_name = "Cellule"

        xmin_elem = bbox.find("xmin")
        ymin_elem = bbox.find("ymin")
        xmax_elem = bbox.find("xmax")
        ymax_elem = bbox.find("ymax")
        if (
            xmin_elem is None
            or ymin_elem is None
            or xmax_elem is None
            or ymax_elem is None
        ):
            continue

        xmin_text = xmin_elem.text
        ymin_text = ymin_elem.text
        xmax_text = xmax_elem.text
        ymax_text = ymax_elem.text
        if (
            xmin_text is None
            or ymin_text is None
            or xmax_text is None
            or ymax_text is None
        ):
            continue

        try:
            xmin = float(xmin_text)
            ymin = float(ymin_text)
            xmax = float(xmax_text)
            ymax = float(ymax_text)
        except ValueError:
            continue

        if xmax <= xmin or ymax <= ymin:
            continue

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(1)

    return boxes, labels


def list_images(images_dir):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(images_dir, ext)))
    return sorted(paths)


def load_model(model_path, device, num_classes=2):
    project_root = os.path.dirname(os.path.abspath(__file__))
    detector_dir = os.path.join(project_root, "Object_detection_fast_rcnn")
    if detector_dir not in sys.path:
        sys.path.insert(0, detector_dir)

    model_file = os.path.join(detector_dir, "model.py")
    spec = importlib.util.spec_from_file_location("od_model", model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load model module from: {model_file}")

    od_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(od_model)

    model = od_model.create_model(num_classes=num_classes)
    # PyTorch >=2.6 defaults to weights_only=True, which can fail for
    # checkpoints saved as full pickled objects/dicts in legacy projects.
    # This project's checkpoint is trusted and local.
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        # Compatibility with older PyTorch versions that do not expose
        # the weights_only argument.
        checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate_split(model, images_dir, annot_dir, image_filenames, device, iou_threshold=0.5, min_score=0.001, split_name="test"):
    image_paths = [os.path.join(images_dir, fname) for fname in image_filenames if os.path.exists(os.path.join(images_dir, fname))]
    
    if not image_paths:
        return {
            "ap": 0.0,
            "mAP": 0.0,
            "precision": np.array([]),
            "recall": np.array([]),
            "ious": [],
            "num_images": 0,
            "num_gt": 0,
            "num_preds": 0,
        }

    detections = []
    matched_ious = []
    total_gt = 0

    progress = tqdm(
        image_paths,
        total=len(image_paths),
        desc=f"Evaluating {split_name}",
        unit="img",
        dynamic_ncols=True,
        leave=True,
    )

    for image_path in progress:
        annotation_file = find_annotation_file(annot_dir, image_path)
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
                matched_ious.append(best_iou)

            detections.append((float(score), is_tp, is_fp))

        progress.set_postfix(
            {
                "GT": total_gt,
                "Pred": len(detections),
                "Matched": len(matched_ious),
            }
        )

    if detections:
        detections.sort(key=lambda x: x[0], reverse=True)
        tp = np.array([d[1] for d in detections], dtype=np.float32)
        fp = np.array([d[2] for d in detections], dtype=np.float32)
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        recall = cum_tp / max(total_gt, 1)
        ap = compute_ap(recall, precision)
    else:
        precision = np.array([])
        recall = np.array([])
        ap = 0.0

    return {
        "ap": ap,
        "mAP": ap,  # single foreground class
        "precision": precision,
        "recall": recall,
        "ious": matched_ious,
        "num_images": len(image_paths),
        "num_gt": total_gt,
        "num_preds": len(detections),
    }


def make_figure(results_by_split, split_order, output_path, iou_threshold):
    apply_nature_style()
    # Nature double-column width is ~183 mm (~7.2 inches).
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True)
    palette = {
        "eval": "#1F77B4",
        "test": "#D62728",
        "other": "#2CA02C",
    }

    def color_for_split(split_name):
        return palette.get(split_name, palette["other"])

    # 1) mAP bar chart
    map_values = [results_by_split[s]["mAP"] for s in split_order]
    bar_colors = [color_for_split(s) for s in split_order]
    axes[0].bar(split_order, map_values, color=bar_colors, width=0.52, edgecolor="#2E3440", linewidth=0.6)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title(f"mAP@{iou_threshold:.2f}", pad=6, fontweight="bold")
    axes[0].set_ylabel("mAP")
    for idx, value in enumerate(map_values):
        axes[0].text(idx, min(value + 0.025, 0.985), f"{value:.3f}", ha="center", fontsize=7, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.45)
    style_axis(axes[0])

    # 2) Precision-Recall curves
    has_pr = False
    for split in split_order:
        rec = results_by_split[split]["recall"]
        pre = results_by_split[split]["precision"]
        ap = results_by_split[split]["ap"]
        if len(rec) == 0:
            continue
        has_pr = True
        axes[1].plot(rec, pre, color=color_for_split(split), label=f"{split} (AP={ap:.3f})")

    axes[1].set_xlim(0, 1.0)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall", pad=6, fontweight="bold")
    axes[1].grid(alpha=0.35)
    if has_pr:
        axes[1].legend(loc="lower left", frameon=False)
    style_axis(axes[1])

    # 3) IoU CDF curves (matched TP only)
    has_iou = False
    for split in split_order:
        ious = np.array(results_by_split[split]["ious"], dtype=np.float32)
        if ious.size == 0:
            continue
        has_iou = True
        ious = np.sort(ious)
        cdf = np.arange(1, len(ious) + 1) / len(ious)
        axes[2].plot(
            ious,
            cdf,
            color=color_for_split(split),
            label=f"{split} (mean={ious.mean():.3f})",
        )

    axes[2].set_xlim(0, 1.0)
    axes[2].set_ylim(0, 1.0)
    axes[2].set_xlabel("IoU")
    axes[2].set_ylabel("Cumulative proportion")
    axes[2].set_title("IoU CDF", pad=6, fontweight="bold")
    axes[2].grid(alpha=0.35)
    if has_iou:
        axes[2].legend(loc="lower right", frameon=False)
    else:
        axes[2].text(0.5, 0.5, "No matched detections", ha="center", va="center")
    style_axis(axes[2])

    # Panel labels (A, B, C) for journal-style multi-panel figures.
    axes[0].text(-0.22, 1.08, "A", transform=axes[0].transAxes, fontsize=10.5, fontweight="bold")
    axes[1].text(-0.22, 1.08, "B", transform=axes[1].transAxes, fontsize=10.5, fontweight="bold")
    axes[2].text(-0.22, 1.08, "C", transform=axes[2].transAxes, fontsize=10.5, fontweight="bold")

    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    root, _ = os.path.splitext(output_path)
    fig.savefig(f"{root}.pdf", bbox_inches="tight")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Evaluate detector and plot mAP/PR/IoU.")
    parser.add_argument(
        "--images-dir",
        default="dataset/images",
        help="Directory containing images.",
    )
    parser.add_argument(
        "--annot-dir",
        default="dataset/annotations",
        help="Directory containing annotations.",
    )
    parser.add_argument(
        "--split-csv",
        default="dataset/split.csv",
        help="CSV file mapping images to splits.",
    )
    parser.add_argument(
        "--model-path",
        default="Streamlit_app/detection_model.pth",
        help="Path to Faster R-CNN checkpoint.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for TP.")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.001,
        help="Minimum score kept before PR/AP calculation.",
    )
    parser.add_argument(
        "--output",
        default="classifier_eval.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--splits",
        default="eval,test",
        help="Comma-separated splits to evaluate (e.g. eval,test).",
    )
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    parser.add_argument("--no-show", action="store_true", help="Do not display figure window.")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    requested_splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not requested_splits:
        raise ValueError("No split provided. Use --splits eval,test")

    split_mapping = {}
    for split in requested_splits:
        if split == "eval":
            split_mapping["valid"] = split
        else:
            split_mapping[split] = split
            
    images_by_split = {split: [] for split in requested_splits}
    if not os.path.exists(args.split_csv):
        raise FileNotFoundError(f"Split CSV not found: {args.split_csv}")
        
    with open(args.split_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_split = row['split']
            if csv_split in split_mapping:
                mapped_split = split_mapping[csv_split]
                images_by_split[mapped_split].append(row['filename'])

    model = load_model(args.model_path, device)

    results_by_split = {}
    for split in requested_splits:
        print(f"Evaluating split: {split} (found {len(images_by_split[split])} images in CSV)")
        results = evaluate_split(
            model,
            args.images_dir,
            args.annot_dir,
            images_by_split[split],
            device=device,
            iou_threshold=args.iou_threshold,
            min_score=args.min_score,
            split_name=split,
        )
        results_by_split[split] = results

        mean_iou = float(np.mean(results["ious"])) if results["ious"] else 0.0
        print(
            f"  images={results['num_images']} | gt_boxes={results['num_gt']} | "
            f"preds={results['num_preds']} | AP={results['ap']:.4f} | mean IoU={mean_iou:.4f}"
        )

    fig = make_figure(results_by_split, requested_splits, args.output, args.iou_threshold)
    print(f"Figure saved to: {args.output}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
