import torch

BATCH_SIZE = 12
RESIZE_TOw = 1920
RESIZE_TOh = 1080
NUM_EPOCHS = 300
NUM_WORKERS = 4

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TRAIN_DIR = '/home/gcp-datathon/Documents/ADLIS/DATASET/train/images'
TRAIN_ANNOT_DIR = '/home/gcp-datathon/Documents/ADLIS/DATASET/train/annotations'
VALID_DIR = '/home/gcp-datathon/Documents/ADLIS/DATASET/test/images'
VALID_ANNOT_DIR = '/home/gcp-datathon/Documents/ADLIS/DATASET/test/annotations'

# Une seule classe "Cellule" + background
CLASSES = [
    '__background__',  
    'Cellule'
]
NUM_CLASSES = len(CLASSES)

OUT_DIR = 'outputs'
