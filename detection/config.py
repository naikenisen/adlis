import torch

BATCH_SIZE = 12
RESIZE_TOw = 1920
RESIZE_TOh = 1080
NUM_EPOCHS = 300
NUM_WORKERS = 4

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TRAIN_DIR = 'dataset/train/images'
TRAIN_ANNOT_DIR = 'dataset/train/annotations'
VALID_DIR = 'dataset/valid/images'
VALID_ANNOT_DIR = 'dataset/valid/annotations'
TEST_DIR = 'dataset/test/images'
TEST_ANNOT_DIR = 'dataset/test/annotations'

# Une seule classe "Cellule" + background
CLASSES = [
    '__background__',  
    'Cellule'
]
NUM_CLASSES = len(CLASSES)

OUT_DIR = 'outputs'
