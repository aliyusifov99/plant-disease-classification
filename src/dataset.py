import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np

from src.config import (
    DATA_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED
)


def get_class_names():
    """Get sorted list of class names from directory structure."""
    class_dirs = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    return class_dirs


def get_class_mapping():
    """Create mapping from class name to index and vice versa."""
    class_names = get_class_names()
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for idx, name in enumerate(class_names)}
    return class_to_idx, idx_to_class


class PlantDiseaseDataset(Dataset):
    """Custom Dataset for PlantVillage images."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(mode="train"):
    """
    Get image transforms for training or validation/test.
    
    Args:
        mode: "train" for training transforms (with augmentation)
              "val" or "test" for validation/test transforms (no augmentation)
    """
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]    # ImageNet stds
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def load_all_data():
    """Load all image paths and labels from the dataset directory."""
    class_to_idx, _ = get_class_mapping()
    
    image_paths = []
    labels = []
    
    for class_name, class_idx in class_to_idx.items():
        class_dir = DATA_DIR / class_name
        
        # Get all jpg images (case insensitive)
        for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]:
            for img_path in class_dir.glob(ext):
                image_paths.append(img_path)
                labels.append(class_idx)
    
    return image_paths, labels


def create_data_splits():
    """
    Split data into train, validation, and test sets.
    
    Returns:
        Dictionary with train, val, test image paths and labels
    """
    image_paths, labels = load_all_data()
    
    # Convert to numpy for easier shuffling
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    # Shuffle data
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(len(image_paths))
    image_paths = image_paths[indices]
    labels = labels[indices]
    
    # Calculate split sizes
    total_size = len(image_paths)
    train_size = int(total_size * TRAIN_SPLIT)
    val_size = int(total_size * VAL_SPLIT)
    
    # Split data
    splits = {
        "train": {
            "paths": image_paths[:train_size],
            "labels": labels[:train_size]
        },
        "val": {
            "paths": image_paths[train_size:train_size + val_size],
            "labels": labels[train_size:train_size + val_size]
        },
        "test": {
            "paths": image_paths[train_size + val_size:],
            "labels": labels[train_size + val_size:]
        }
    }
    
    print(f"Dataset splits:")
    print(f"  Train: {len(splits['train']['paths'])} images")
    print(f"  Val:   {len(splits['val']['paths'])} images")
    print(f"  Test:  {len(splits['test']['paths'])} images")
    
    return splits


def get_data_loaders():
    """
    Create DataLoaders for train, validation, and test sets.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    splits = create_data_splits()
    
    # Create datasets
    train_dataset = PlantDiseaseDataset(
        splits["train"]["paths"],
        splits["train"]["labels"],
        transform=get_transforms("train")
    )
    
    val_dataset = PlantDiseaseDataset(
        splits["val"]["paths"],
        splits["val"]["labels"],
        transform=get_transforms("val")
    )
    
    test_dataset = PlantDiseaseDataset(
        splits["test"]["paths"],
        splits["test"]["labels"],
        transform=get_transforms("test")
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# Quick test
if __name__ == "__main__":
    print("Testing dataset module...\n")
    
    # Test class mapping
    class_to_idx, idx_to_class = get_class_mapping()
    print(f"Number of classes: {len(class_to_idx)}")
    print(f"First 5 classes: {list(class_to_idx.keys())[:5]}\n")
    
    # Test data loaders
    train_loader, val_loader, test_loader = get_data_loaders()
    
    # Test one batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch test:")
    print(f"  Image batch shape: {images.shape}")
    print(f"  Labels batch shape: {labels.shape}")
    print(f"  Sample labels: {labels[:5].tolist()}")
    print(f"  Sample class names: {[idx_to_class[l.item()] for l in labels[:5]]}")