import torch
import cv2
import numpy as np
import os
import glob

from xml.etree import ElementTree as et
import csv
from config import (CLASSES, RESIZE_TOw, RESIZE_TOh, BATCH_SIZE, IMAGES_DIR, ANNOT_DIR, SPLIT_CSV)
from torch.utils.data import Dataset, DataLoader
from custom_utils import collate_fn, get_train_transform, get_valid_transform

class CustomDataset(Dataset):
    def __init__(self, dir_path, annot_path, width, height, classes, split='train', split_csv='dataset/split.csv', transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.annot_path = annot_path
        self.height = height
        self.width = width
        self.classes = classes  # ['__background__', 'Cellule']
        
        self.all_images = []
        with open(split_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['split'] == split:
                    self.all_images.append(row['filename'])
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)

        # Lire l'image (BGR => RGB => float32 => [0,1] après /255.0)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found or is empty: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0
        
        # Fichier .xml correspondant
        annot_filename = os.path.splitext(image_name)[0] + '.xml'
        annot_file_path = os.path.join(self.annot_path, annot_filename)
        
        boxes = []
        labels = []
        
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        
        # Dimensions d'origine (avant resize)
        image_width = image.shape[1]
        image_height = image.shape[0]
        
        # Récupérer les bounding boxes
        for member in root.findall('object'):
            name = member.find('name').text
            # Conversion : toute annotation SN ou SC => "Cellule" (index = 1)
            if name in ['SN', 'SC']:
                label_idx = 1
            else:
                label_idx = 1  # ou alors tout passe en "Cellule"
            
            # Lire xmin/xmax/ymin/ymax
            xmin = int(member.find('bndbox/xmin').text)
            xmax = int(member.find('bndbox/xmax').text)
            ymin = int(member.find('bndbox/ymin').text)
            ymax = int(member.find('bndbox/ymax').text)
            
            # Resize en proportion
            xmin_final = (xmin / image_width) * self.width
            xmax_final = (xmax / image_width) * self.width
            ymin_final = (ymin / image_height) * self.height
            ymax_final = (ymax / image_height) * self.height

            # Eviter des boxes nulles
            if xmax_final == xmin_final:
                xmax_final += 1
            if ymax_final == ymin_final:
                ymax_final += 1

            # Clipper si besoin
            if xmax_final > self.width:
                xmax_final = self.width
            if ymax_final > self.height:
                ymax_final = self.height
            
            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
            labels.append(label_idx)
        
        # Convertir en tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) \
            if len(boxes) > 0 else torch.as_tensor([], dtype=torch.float32)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor([idx])

        # Appliquer les éventuelles augmentations Albumentations
        if self.transforms:
            sample = self.transforms(
                image=image_resized,
                bboxes=target['boxes'],
                labels=[int(l) for l in labels]
            )
            image_resized = sample['image']
            # Reconvertir en tensor
            target['boxes'] = torch.as_tensor(sample['bboxes'], dtype=torch.float32)
        
        # Si la box est NaN ou vide
        if len(target['boxes']) == 0:
            target['boxes'] = torch.zeros((0,4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
        
        return image_resized, target

    def __len__(self):
        return len(self.all_images)

# Création dataset + dataloader
def create_train_dataset(DIR):
    return CustomDataset(
        DIR, ANNOT_DIR, RESIZE_TOw, RESIZE_TOh, CLASSES, split='train', split_csv=SPLIT_CSV, transforms=get_train_transform()
    )

def create_valid_dataset(DIR):
    return CustomDataset(
        DIR, ANNOT_DIR, RESIZE_TOw, RESIZE_TOh, CLASSES, split='valid', split_csv=SPLIT_CSV, transforms=get_valid_transform()
    )

def create_train_loader(train_dataset, num_workers=0):
    return DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )

def create_valid_loader(valid_dataset, num_workers=0):
    return DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
