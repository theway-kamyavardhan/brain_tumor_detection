# ===========================
# IMPORTS
# ===========================
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# ===========================
# CONFIG
# ===========================
IMG_SIZE = 224
BATCH_SIZE = 32
VAL_SPLIT = 0.15


# ===========================
# DATA PIPELINE
# ===========================
def get_dataloaders(train_path, test_path):

    # -------- TRAIN AUGMENTATION --------
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),
        transforms.ToTensor(),  # auto normalize (0-1)
    ])

    # -------- TEST / VALIDATION --------
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # Load full training dataset
    full_dataset = datasets.ImageFolder(train_path, transform=train_transform)

    # Split into train & validation
    total_size = len(full_dataset)
    val_size = int(VAL_SPLIT * total_size)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Validation should NOT have augmentation
    val_dataset.dataset.transform = test_transform

    # Test dataset
    test_dataset = datasets.ImageFolder(test_path, transform=test_transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, full_dataset.classes


# ===========================
# OPTIONAL: ANALYSIS + VISUALIZATION
# ===========================
if __name__ == "__main__":

    TRAIN_PATH = "../dataset/Training"
    TEST_PATH = "../dataset/Testing"

    train_loader, val_loader, test_loader, class_names = get_dataloaders(TRAIN_PATH, TEST_PATH)

    print("\nClass Names:", class_names)
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Testing batches:", len(test_loader))

    # -------- CLASS DISTRIBUTION --------
    labels = []
    for _, y in train_loader:
        labels.extend(y.numpy())

    class_counts = Counter(labels)

    print("\nClass Distribution (Training):")
    for cls, count in class_counts.items():
        print(f"{class_names[cls]}: {count}")

    # -------- SHOW SAMPLE IMAGES --------
    images, labels = next(iter(train_loader))

    plt.figure(figsize=(10,10))

    for i in range(9):
        img = images[i].permute(1, 2, 0).numpy()

        plt.subplot(3,3,i+1)
        plt.imshow(img)
        plt.title(class_names[labels[i]])
        plt.axis('off')

    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig("outputs/plots/sample_images.png")
    plt.show()

    print("\nSample images saved at: outputs/plots/sample_images.png")