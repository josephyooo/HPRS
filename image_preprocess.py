import os
import cv2
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import random
import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F

# Define the directory containing the augmented images
cwd = os.path.dirname(__file__)
data_dir = os.path.join(cwd, "Image_Data/DermNet")

# Define the transformations to be applied to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Function to load and augment images from a directory
def augment_images(directory, num_augments):
    augmented_images = []
    for filename in os.listdir(directory):
        # Load the original image
        image_path = os.path.join(directory, filename)
        original_image = Image.open(image_path)
        
        # Apply transformations to the original image
        for _ in range(num_augments):
            # Randomly select augmentation techniques
            chosen_transforms = random.sample(transform_options, k=random.randint(1, max_transforms))
            composed_transform = transforms.Compose(chosen_transforms)
            augmented_image = composed_transform(original_image)
            augmented_images.append(transform(augmented_image))
    return augmented_images

# Define the subfolders containing the images
subfolders = ['Alopecia-pictures', 'seborrheic-dermatitis-pictures', 'Psoriasis-pictures']

# Create a dictionary to store augmented images for each subfolder
augmented_images_by_subfolder = {}

# Define augmentation strategy
augmentation_strategy = {
    'Alopecia-pictures': {
        'less_than_50': 6,
        'between_50_and_100': 4,
        'more_than_100': 2
    },
    'seborrheic-dermatitis-pictures': {
        'less_than_50': 6,
        'between_50_and_100': 4,
        'more_than_100': 2
    },
    'Psoriasis-pictures': {
        'less_than_50': 6,
        'between_50_and_100': 4,
        'more_than_100': 2
    }
}

# Augmentation techniques
transform_options = [
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5)
]
max_transforms = len(transform_options)

# Augment images for each subfolder based on the strategy
for subfolder in subfolders:
    subfolder_path = os.path.join(data_dir, subfolder)
    num_images = len(os.listdir(subfolder_path))
    num_augments = None
    if num_images < 50:
        num_augments = augmentation_strategy[subfolder]['less_than_50']
    elif 50 <= num_images <= 100:
        num_augments = augmentation_strategy[subfolder]['between_50_and_100']
    else:
        num_augments = augmentation_strategy[subfolder]['more_than_100']
        
    augmented_images = augment_images(subfolder_path, num_augments=num_augments)
    augmented_images_by_subfolder[subfolder] = augmented_images

# Convert the augmented images to torch tensors
augmented_images_tensors_by_subfolder = {}
for subfolder, images in augmented_images_by_subfolder.items():
    augmented_images_tensors_by_subfolder[subfolder] = torch.stack(images)

# Display the size of tensors for each subfolder
for subfolder, tensors in augmented_images_tensors_by_subfolder.items():
    print(f"Subfolder: {subfolder}, Tensor size: {tensors.size()}")

# Display 20 pictures from each subfolder
for subfolder, images in augmented_images_by_subfolder.items():
    print(f"\nShowing images from subfolder: {subfolder}")
    plt.figure(figsize=(20, 5))
    for i in range(20):  # Displaying 20 images from each subfolder
        plt.subplot(2, 10, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))
        plt.axis('off')
    plt.show()

# Combine all augmented images and create labels
all_images = torch.cat([torch.stack(images) for images in augmented_images_by_subfolder.values()], dim=0)
num_classes = len(subfolders)
all_labels = torch.tensor(sum([[i] * len(images) for i, images in enumerate(augmented_images_by_subfolder.values())], []))

# Split data into training and validation sets
dataset = list(zip(all_images, all_labels))
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Define DataLoader for training and validation sets
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Check the sizes of training and validation sets
print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
