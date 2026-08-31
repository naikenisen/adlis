import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,  models
from torchvision.transforms import v2
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
import torch.nn.functional as F
import seaborn as sns

import torchvision.transforms.v2.functional as TF

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 100
num_classes = 2

class PadToSquare:
    def __init__(self, padding_mode='constant'):
        self.padding_mode = padding_mode

    def __call__(self, img):
        h, w = TF.get_size(img)
        max_size = max(h, w)
        pad_left = (max_size - w) // 2
        pad_right = max_size - w - pad_left
        pad_top = (max_size - h) // 2
        pad_bottom = max_size - h - pad_top
        
        fill_val = 0
        if self.padding_mode == 'constant':
            if isinstance(img, torch.Tensor):
                # Extraire les bords (haut, bas, gauche, droite)
                top = img[:, 0:1, :]
                bottom = img[:, -1:, :]
                left = img[:, :, 0:1]
                right = img[:, :, -1:]
                
                # Concaténer tous les pixels des bords
                borders = torch.cat([
                    top.reshape(img.shape[0], -1),
                    bottom.reshape(img.shape[0], -1),
                    left.reshape(img.shape[0], -1),
                    right.reshape(img.shape[0], -1)
                ], dim=1)
                
                # Utiliser la médiane pour être robuste aux artefacts ou bruits sur les bords
                fill_val = borders.median(dim=1).values.tolist()

        return TF.pad(img, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=fill_val, padding_mode=self.padding_mode)

# Transformations from train.py
train_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.uint8, scale=True), 

    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomApply([
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
    ], p=0.8),
    v2.ToDtype(torch.float32, scale=True),
    PadToSquare(padding_mode='constant'),
    v2.Resize((224, 224)),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

valid_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the datasets
train_dataset = datasets.ImageFolder(root='dataset/classification_set/train', transform=train_transforms)
valid_dataset = datasets.ImageFolder(root='dataset/classification_set/valid', transform=valid_transforms)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)


# Compute class weights based on the training dataset
class_counts = np.bincount(train_dataset.targets)  # Count number of samples for each class
class_weights = 1.0 / class_counts  # Inverse of class frequency
class_weights = class_weights / class_weights.sum()  # Normalize weights
class_weights_tensor = torch.FloatTensor(class_weights).to(device)  # Move weights to the GPU if available


# Use a pre-trained model (ResNet50)
model = models.resnet50(weights='IMAGENET1K_V1')
# Modify the final layer to classify two classes
# Customize the fully connected (FC) layer
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(64, num_classes)  # num_classes = 2 for binary classification
)
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True)

best_f1 = 0.0

history_train_loss = []
history_valid_loss = []
history_valid_f1 = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

    for inputs, labels in train_loader_tqdm:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_loader_tqdm.set_postfix(loss=running_loss/len(train_loader))


    epoch_train_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_train_loss:.4f}')

    # Validation
    model.eval()
    valid_running_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        valid_loader_tqdm = tqdm(valid_loader, desc="Validating", unit="batch")

        for inputs, labels in valid_loader_tqdm:

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            
            probs = F.softmax(outputs, dim=1)[:, 0]  # Store probs of class 0 (SC)

            all_labels.extend(labels.cpu().numpy())    # Store actual labels
            all_probs.extend(probs.cpu().numpy())      # Store probabilities for AUC

    epoch_valid_loss = valid_running_loss / len(valid_loader)

    # Invert labels for metrics so SC (class 0) becomes the positive class
    labels_sc = 1 - np.array(all_labels)
    probs_sc = np.array(all_probs)

    # Compute F1 score targeting SC
    preds_sc = (probs_sc >= 0.5).astype(int)
    f1 = f1_score(labels_sc, preds_sc)
    print(f'Validation Loss: {epoch_valid_loss:.4f} | F1 Score (SC): {f1:.4f}')
    
    history_train_loss.append(epoch_train_loss)
    history_valid_loss.append(epoch_valid_loss)
    history_valid_f1.append(f1)

    # Save best model based on F1 Score
    if f1 > best_f1:
        best_f1 = f1
        import os
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weights_dir = os.path.join(ROOT_DIR, 'weights')
        os.makedirs(weights_dir, exist_ok=True)
        
        torch.save(model.state_dict(), os.path.join(weights_dir, 'classification.pth'))
        print(f'New best model saved based on F1 Score: {f1:.4f}')

    # Step the scheduler based on the validation Loss
    scheduler.step(epoch_valid_loss)

print("Training complete.")
