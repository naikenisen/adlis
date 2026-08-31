import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision import datasets
from torchvision.transforms import v2

import torchvision.transforms.v2.functional as TF

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
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ], p=0.8),
    v2.ToDtype(torch.float32, scale=True),
    PadToSquare(padding_mode='constant'),
    v2.Resize((224, 224)),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Inverse normalization function to display images properly
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def unnormalize(tensor):
    img = tensor.permute(1, 2, 0).numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def main():
    # Load dataset without transforms to get the raw PIL Images
    dataset_dir = '/home/naiken/coding/adlis/dataset/classification_set/test'
    dataset = datasets.ImageFolder(root=dataset_dir)

    # Filtrer pour ne garder que la classe 'SC'
    if 'SC' in dataset.class_to_idx:
        sc_class_idx = dataset.class_to_idx['SC']
        sc_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == sc_class_idx]
    else:
        raise ValueError("La classe 'SC' n'a pas été trouvée dans le dataset.")

    # Pick 10 random images parmis la classe SC
    num_images = min(10, len(sc_indices))
    indices = random.sample(sc_indices, num_images)

    # Setup matplotlib figure
    fig, axes = plt.subplots(num_images, 2, figsize=(8, 3 * num_images))
    fig.suptitle('Augmentation Test: Original vs Augmented', fontsize=16)

    for i, idx in enumerate(indices):
        pil_img, label = dataset[idx]

        # Apply transformations
        aug_tensor = train_transforms(pil_img)
        aug_img = unnormalize(aug_tensor)

        # Show Original
        axes[i, 0].imshow(pil_img)
        axes[i, 0].axis('off')

        # Show Augmented
        axes[i, 1].imshow(aug_img)
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.97)
    
    # Save the figure just in case the environment doesn't support display
    plt.savefig('augmentation_preview.png')
    print("Figure saved to augmentation_preview.png")
    
    # Display the figure
    plt.show()

if __name__ == "__main__":
    main()
