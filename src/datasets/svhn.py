"""Balanced SVHN subsets from the official training partition only."""

from pathlib import Path

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import SVHN

from .cifar10 import IMAGENET_MEAN, IMAGENET_STD, balanced_split_indices


def get_svhn_datasets(data_dir="data/raw", train_size=10000, val_size=2000, seed=42):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    root = str(Path(data_dir))
    train_data = SVHN(root=root, split="train", download=True, transform=transform)
    val_data = SVHN(root=root, split="train", download=True, transform=transform)
    train_indices, val_indices = balanced_split_indices(
        train_data.labels, train_size=train_size, val_size=val_size, seed=seed
    )
    return Subset(train_data, train_indices), Subset(val_data, val_indices)


def get_svhn_loaders(
    data_dir="data/raw", train_size=10000, val_size=2000, batch_size=16, seed=42
):
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    train_data, val_data = get_svhn_datasets(data_dir, train_size, val_size, seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=0)
    return train_loader, val_loader
