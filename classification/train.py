import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,  models
from torchvision.transforms import v2
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
import seaborn as sns

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 200
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

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

best_pr_auc = 0.0

history_train_loss = []
history_valid_loss = []
history_valid_pr_auc = []

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

    # Compute AUC-PR targeting SC
    pr_auc = average_precision_score(labels_sc, probs_sc)
    print(f'Validation Loss: {epoch_valid_loss:.4f} | AUC-PR (SC): {pr_auc:.4f}')
    
    history_train_loss.append(epoch_train_loss)
    history_valid_loss.append(epoch_valid_loss)
    history_valid_pr_auc.append(pr_auc)

    # Save best model based on AUC-PR
    if pr_auc > best_pr_auc:
        best_pr_auc = pr_auc
        import os
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weights_dir = os.path.join(ROOT_DIR, 'weights')
        os.makedirs(weights_dir, exist_ok=True)
        
        torch.save(model.state_dict(), os.path.join(weights_dir, 'classification.pth'))
        print(f'New best model saved based on AUC-PR: {pr_auc:.4f}')

    # Step the scheduler based on the validation AUC-PR
    scheduler.step(pr_auc)

print("Training complete.")

# Generate Figure S_6
import os
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fig_path = os.path.join(ROOT_DIR, 'figures', 'figure_S_6.png')
os.makedirs(os.path.dirname(fig_path), exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

epochs_range = range(1, len(history_train_loss) + 1)
ax1.plot(epochs_range, history_train_loss, label='Train Loss', color='#1F77B4', lw=2)
ax1.plot(epochs_range, history_valid_loss, label='Validation Loss', color='#FF7F0E', lw=2)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss (Weighted CrossEntropy)')
ax1.set_title('a) Training and Validation Loss')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

ax2.plot(epochs_range, history_valid_pr_auc, label='Validation AUC-PR (SC)', color='#2CA02C', lw=2)
ax2.set_xlabel('Epochs')
ax2.set_ylabel('AUC-PR')
ax2.set_title('b) Validation AUC-PR over Epochs')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"Saved learning curves to {fig_path}")
