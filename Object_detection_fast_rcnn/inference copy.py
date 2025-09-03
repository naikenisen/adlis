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
    print(f"\nProcessing image: {image_name}, shape: {image.shape}")

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

    print(f"Predicted boxes: {boxes}")
    print(f"Predicted labels: {labels}")
    print(f"Predicted scores: {scores[mask]}")

    # Charger les annotations ground truth (exemple: fichier .xml par image)
    gt_path = os.path.join(
        '/home/naiken/coding/ADLIS/Object_detection_fast_rcnn/dataset_split/test/annotations',
        image_name + '.jpg.xml'
    )
    gt_boxes = []
    gt_labels = []
    if os.path.exists(gt_path):
        import xml.etree.ElementTree as ET
        tree = ET.parse(gt_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            class_name = obj.find('name').text
            # Mapping SN/SC vers Cellule
            if class_name in ['SN', 'SC']:
                class_name = 'Cellule'
            try:
                class_idx = CLASSES.index(class_name)
            except ValueError:
                print(f"Classe inconnue dans l'annotation: {class_name}")
                continue
            gt_boxes.append([xmin, ymin, xmax, ymax])
            gt_labels.append(class_idx)
    print(f"GT boxes: {gt_boxes}")
    print(f"GT labels: {gt_labels}")

    matched_gt = set()
    matched_pred = set()
    for pred_idx, pred_box in enumerate(boxes):
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = compute_iou(pred_box, gt_box)
            print(f"  Pred box {pred_idx} vs GT box {gt_idx}: IoU={iou:.2f}, pred_label={labels[pred_idx]}, gt_label={gt_labels[gt_idx]}")
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_iou >= 0.5 and best_gt_idx != -1 and labels[pred_idx] == gt_labels[best_gt_idx] and best_gt_idx not in matched_gt:
            TP += 1
            ious.append(best_iou)
            matched_gt.add(best_gt_idx)
            matched_pred.add(pred_idx)
            print(f"  --> TP (IoU={best_iou:.2f})")
        else:
            FP += 1
            print(f"  --> FP")

    FN += len(gt_boxes) - len(matched_gt)
    print(f"TP: {TP}, FP: {FP}, FN: {FN}")

# Calculs finaux
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
mean_iou = np.mean(ious) if ious else 0

print(f"Mean IoU (Jaccard): {mean_iou:.4f}")
print(f"F1-score: {f1_score:.4f}")
print(f"Accuracy: {accuracy:.4f}")

if frame_count > 0:
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.2f}")
print('TEST PREDICTIONS COMPLETE')
