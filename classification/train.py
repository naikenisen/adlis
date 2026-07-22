import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,  models
from torchvision.transforms import v2
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, average_precision_score
import torch.nn.functional as F
import seaborn as sns

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 100
num_classes = 2

# Define transforms for training and validation
train_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point

    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomZoomOut(),
    v2.RandomRotation(90),
    v2.GaussianBlur(3),
    v2.RandomAdjustSharpness(0),
    v2.ToDtype(torch.float32, scale=True),
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

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Loss function and optimizer
criterion = FocalLoss(weight=class_weights_tensor, gamma=2.0)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_pr_auc = 0.0

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


    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    # Validation
    model.eval()
    correct = 0
    total = 0
    TP, TN, FP, FN = 0, 0, 0, 0
    all_preds = []
    all_labels = []
    all_probs = []


    with torch.no_grad():
        valid_loader_tqdm = tqdm(valid_loader, desc="Validating", unit="batch")

        for inputs, labels in valid_loader_tqdm:

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1)[:, 0]  # Store probs of class 0 (SC)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())  # Store predictions
            all_labels.extend(labels.cpu().numpy())    # Store actual labels
            all_probs.extend(probs.cpu().numpy())      # Store probabilities for AUC

            # Compute TP, TN, FP, FN for sensitivity and specificity (Tracking Class 0: SC)
            for i in range(len(labels)):
                if labels[i] == 0 and predicted[i] == 0:
                    TP += 1  # True Positive: It's SC and predicted SC
                elif labels[i] == 1 and predicted[i] == 1:
                    TN += 1  # True Negative: It's SN and predicted SN
                elif labels[i] == 1 and predicted[i] == 0:
                    FP += 1  # False Positive: It's SN but predicted SC
                elif labels[i] == 0 and predicted[i] == 1:
                    FN += 1  # False Negative: It's SC but predicted SN
            #valid_loader_tqdm.set_postfix(loss=running_loss/len(va))



    # Compute validation accuracy
    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')

    # Compute F1 Score
    f1 = f1_score(all_labels, all_preds, average='weighted')  # Weighted F1 score for imbalanced classes
    print(f'F1 Score (weighted): {f1:.4f}')

    # Invert labels for metrics so SC (class 0) becomes the positive class
    labels_sc = 1 - np.array(all_labels)
    probs_sc = np.array(all_probs)

    # Compute AUC-ROC and AUC-PR targeting SC
    auc = roc_auc_score(labels_sc, probs_sc)
    pr_auc = average_precision_score(labels_sc, probs_sc)
    print(f'AUC-ROC (SC): {auc:.4f}')
    print(f'AUC-PR (SC): {pr_auc:.4f}')

    # Save best model based on AUC-PR
    if pr_auc > best_pr_auc:
        best_pr_auc = pr_auc
        import os
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weights_dir = os.path.join(ROOT_DIR, 'weights')
        os.makedirs(weights_dir, exist_ok=True)
        
        torch.save(model.state_dict(), os.path.join(weights_dir, 'classification.pth'))
        print(f'New best model saved based on AUC-PR: {pr_auc:.4f}')
        print('Confusion matrix')
        # Confusion Matrix Plotting
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=valid_dataset.classes, yticklabels=valid_dataset.classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.savefig('cm.png')
        plt.close()


    accuracy = 100 * correct / total
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    print(f'Validation Accuracy: {accuracy:.2f}%')
    print(f'Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}')

print("Training complete.")
