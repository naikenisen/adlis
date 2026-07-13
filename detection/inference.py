import numpy as np
import cv2
import torch
import glob
import os
import time
import argparse

from model import create_model
from config import NUM_CLASSES, DEVICE, CLASSES

np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='/home/naiken/coding/ADLIS/Object_detection_fast_rcnn/dataset_split/test/images',
                    help='Path to input image directory')
parser.add_argument('--imgsz', default=None, type=int,
                    help='Image resize shape (height=width=imgsz)')
parser.add_argument('--threshold', default=0.25, type=float,
                    help='Detection threshold (score >= this value)')
args = vars(parser.parse_args())

os.makedirs('inference_outputs/images', exist_ok=True)

# Palette de couleurs (pour BG, Cellule)
# => COLORS[0], COLORS[1]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load('/home/naiken/coding/ADLIS/Streamlit_app/detection_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

DIR_TEST = args['input']
test_images = glob.glob(f"{DIR_TEST}/*.jpg")
print(f"Test instances: {len(test_images)}")

frame_count = 0
total_fps = 0

def compute_iou(boxA, boxB):
    # box format: [xmin, ymin, xmax, ymax]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Pour stocker les stats globales
TP, FP, FN = 0, 0, 0
ious = []

for i, img_path in enumerate(test_images):
    image_name = os.path.splitext(os.path.basename(img_path))[0]
    image = cv2.imread(img_path)
    orig_image = image.copy()

    if args['imgsz'] is not None:
        image = cv2.resize(image, (args['imgsz'], args['imgsz']))
    print(f"Processing image shape: {image.shape}")

    # BGR -> RGB, [0,1]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0

    # (C, H, W) + batch dimension
    image_input = np.transpose(image, (2, 0, 1))
    image_input = torch.tensor([image_input], dtype=torch.float)

    start_time = time.time()
    with torch.no_grad():
        outputs = model(image_input.to(DEVICE))
    end_time = time.time()

    fps = 1 / (end_time - start_time)
    total_fps += fps
    frame_count += 1

    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    boxes = outputs[0]['boxes'].numpy()
    scores = outputs[0]['scores'].numpy()
    labels = outputs[0]['labels'].numpy()

    # Filtrage par threshold
    mask = scores >= args['threshold']
    boxes = boxes[mask].astype(np.int32)
    labels = labels[mask]

    # Dessiner
    for idx, box in enumerate(boxes):
        class_idx = labels[idx]  # 1 => Cellule
        class_name = CLASSES[class_idx]  
        color = COLORS[class_idx]

        if args['imgsz'] is not None:
            # Re-scale si l'image a été redimensionnée
            (h_orig, w_orig) = orig_image.shape[:2]
            (h_new, w_new) = image.shape[:2]
            xmin = int((box[0] / w_new) * w_orig)
            ymin = int((box[1] / h_new) * h_orig)
            xmax = int((box[2] / w_new) * w_orig)
            ymax = int((box[3] / h_new) * h_orig)
        else:
            (xmin, ymin, xmax, ymax) = box

        cv2.rectangle(orig_image, (xmin, ymin), (xmax, ymax), color[::-1], 2)
        cv2.putText(orig_image,
                    class_name,
                    (xmin, max(ymin - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color[::-1],
                    2,
                    lineType=cv2.LINE_AA)

    cv2.imshow('Prediction', orig_image)
    cv2.waitKey(1)
    out_path = f"inference_outputs/images/{image_name}.jpg"
    cv2.imwrite(out_path, orig_image)
    print(f"Image {i+1} done, saved to {out_path}")
    print("-" * 50)

    # Charger les annotations ground truth (exemple: fichier .txt ou .json par image)
    # Format attendu: [[xmin, ymin, xmax, ymax, class_idx], ...]
    gt_path = img_path.replace('.jpg', '.txt')  # à adapter selon ton format
    gt_boxes = []
    gt_labels = []
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                gt_boxes.append([int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])])
                gt_labels.append(int(parts[4]))

    matched_gt = set()
    matched_pred = set()
    # Pour chaque prédiction, chercher la meilleure GT
    for pred_idx, pred_box in enumerate(boxes):
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_iou >= 0.5 and labels[pred_idx] == gt_labels[best_gt_idx] and best_gt_idx not in matched_gt:
            TP += 1
            ious.append(best_iou)
            matched_gt.add(best_gt_idx)
            matched_pred.add(pred_idx)
        else:
            FP += 1

    FN += len(gt_boxes) - len(matched_gt)

# Calculs finaux
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
mean_iou = np.mean(ious) if ious else 0

print(f"Mean IoU (Jaccard): {mean_iou:.4f}")
print(f"F1-score: {f1_score:.4f}")
print(f"Accuracy: {accuracy:.4f}")

cv2.destroyAllWindows()
if frame_count > 0:
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.2f}")
print('TEST PREDICTIONS COMPLETE')
