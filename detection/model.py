import torchvision
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN_ResNet50_FPN_V2_Weights, FastRCNNPredictor
)

def create_model(num_classes=2, freeze_backbone=False, trainable_layers=3):
    """
    Crée un Faster R-CNN ResNet50-FPN pré-entraîné, 
    adapté à 'num_classes' (ex: 2 pour [BG, Cellule]).
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if freeze_backbone:
        # Figer tout
        for param in model.backbone.body.parameters():
            param.requires_grad = False
        # Défiger layer3 + layer4 si on veut
        if trainable_layers >= 1:
            for param in model.backbone.body.layer4.parameters():
                param.requires_grad = True
        if trainable_layers >= 2:
            for param in model.backbone.body.layer3.parameters():
                param.requires_grad = True
    
    return model
