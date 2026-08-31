import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision import datasets
from torchvision.transforms import v2

import torchvision.transforms.v2.functional as TF

class PadToSquare:
    def __init__(self, fill=0, padding_mode='constant'):
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        h, w = TF.get_size(img)
        max_size = max(h, w)
        pad_left = (max_size - w) // 2
        pad_right = max_size - w - pad_left
        pad_top = (max_size - h) // 2
        pad_bottom = max_size - h - pad_top
        return TF.pad(img, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=self.fill, padding_mode=self.padding_mode)

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
    PadToSquare(fill=0, padding_mode='constant'),
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

    # Pick 10 random images
    num_images = 10
    indices = random.sample(range(len(dataset)), num_images)

    # Setup matplotlib figure
    fig, axes = plt.subplots(num_images, 2, figsize=(8, 3 * num_images))
    fig.suptitle('Augmentation Test: Original vs Augmented', fontsize=16)

    for i, idx in enumerate(indices):
        pil_img, label = dataset[idx]
        class_name = dataset.classes[label]

        # Apply transformations
        aug_tensor = train_transforms(pil_img)
        aug_img = unnormalize(aug_tensor)

        # Show Original
        axes[i, 0].imshow(pil_img)
        axes[i, 0].set_title(f'Original - {class_name}')
        axes[i, 0].axis('off')

        # Show Augmented
        axes[i, 1].imshow(aug_img)
        axes[i, 1].set_title(f'Augmented - {class_name}')
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
