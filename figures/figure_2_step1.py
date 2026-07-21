import os
import sys
import glob
import pandas as pd
import torch
import torchvision.transforms as T
from torchvision.transforms import v2
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import models
from torchvision.ops import nms
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

test_externe_dir = os.path.join(project_root, "dataset/test_externe")
inference_csv = os.path.join(project_root, "dataset/inference-test-externe.csv")

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
    return "SC" if predicted.item() == 0 else "SN"

def main():
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

if __name__ == "__main__":
    main()
