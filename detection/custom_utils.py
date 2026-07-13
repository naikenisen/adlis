import albumentations as A
import torch
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from torchvision.ops import box_iou

plt.style.use('ggplot')

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value: float):
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self) -> float:
        if self.iterations == 0:
            return 0.0
        return self.current_total / self.iterations
    
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

def collate_fn(batch):
    return tuple(zip(*batch))

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45),
        A.Blur(blur_limit=3, p=0.1),
        A.MotionBlur(blur_limit=3, p=0.1),
        A.MedianBlur(blur_limit=3, p=0.1),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc', 
        'label_fields': ['labels']
    })

def save_loss_plot(OUT_DIR, train_loss_list, x_label='iterations',
                   y_label='train loss', save_name='train_loss'):
    """
    Sauvegarde la courbe de la perte d'entraînement
    """
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss_list, color='tab:blue', label='Train Loss')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(f"{OUT_DIR}/{save_name}.png")
    plt.close()
    print('SAVING PLOTS COMPLETE...')

class SaveBestModel:
    """
    Sauvegarde du meilleur modèle selon une métrique à maximiser (ex: F1).
    """
    def __init__(self, best_metric=0.0):
        self.best_metric = best_metric
        
    def __call__(self, model, current_metric, epoch, OUT_DIR):
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            print(f"\nBEST VALIDATION METRIC: {self.best_metric:.4f}")
            print(f"SAVING BEST MODEL FOR EPOCH: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict()
            }, f"{OUT_DIR}/best_model.pth")

### Métriques de détection

def compute_batch_detection_metrics(preds, targets, iou_threshold=0.5, score_threshold=0.5):
    """
    Calcule TP, FP, FN pour un batch d'images.
    """
    total_tp, total_fp, total_fn = 0, 0, 0
    for i in range(len(preds)):
        pred_boxes = preds[i]['boxes']
        pred_scores = preds[i]['scores']
        pred_labels = preds[i]['labels']
        gt_boxes = targets[i]['boxes']
        gt_labels = targets[i]['labels']
        
        # Filtrer par score
        keep = pred_scores >= score_threshold
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        
        if len(pred_boxes) == 0:
            total_fn += len(gt_boxes)
            continue
        
        ious = box_iou(pred_boxes, gt_boxes)  # (pred_count, gt_count)
        
        matched_gt = set()
        tp = 0
        fp = 0
        for p_idx in range(len(pred_boxes)):
            iou_vals = ious[p_idx]
            max_iou_val, max_iou_idx = torch.max(iou_vals, dim=0)
            
            pred_cls = pred_labels[p_idx].item()
            gt_cls = gt_labels[max_iou_idx].item()
            
            if max_iou_val.item() >= iou_threshold and (pred_cls == gt_cls):
                if max_iou_idx.item() not in matched_gt:
                    tp += 1
                    matched_gt.add(max_iou_idx.item())
                else:
                    fp += 1
            else:
                fp += 1
        
        fn = len(gt_boxes) - len(matched_gt)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn

    return total_tp, total_fp, total_fn

def precision_recall_f1(tp, fp, fn, eps=1e-6):
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * (precision * recall) / (precision + recall + eps)
    return precision, recall, f1

@torch.no_grad()
def evaluate_detection_model(model, data_loader, device, iou_threshold=0.5, score_threshold=0.5):
    """
    Passe sur tout le data_loader pour calculer P, R, F1
    """
    model.eval()
    total_tp, total_fp, total_fn = 0, 0, 0
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        preds = model(images)
        
        tp, fp, fn = compute_batch_detection_metrics(
            preds, targets, 
            iou_threshold=iou_threshold, 
            score_threshold=score_threshold
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
    precision, recall, f1 = precision_recall_f1(total_tp, total_fp, total_fn)
    return precision, recall, f1
