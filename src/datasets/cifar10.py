from collections import defaultdict
from pathlib import Path
from random import Random

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def balanced_split_indices(targets, train_size=2000, val_size=5000, seed=42):
    by_class = defaultdict(list)
    for index, label in enumerate(targets):
        by_class[int(label)].append(index)

    class_count = len(by_class)
    if class_count == 0 or train_size <= 0 or val_size <= 0:
        raise ValueError("Targets and split sizes must be nonempty")
    if train_size % class_count or val_size % class_count:
        raise ValueError("Split sizes must be divisible by the number of classes")

    train_per_class = train_size // class_count
    val_per_class = val_size // class_count
    rng = Random(seed)
    train_indices = []
    val_indices = []

    for label in sorted(by_class):
        indices = by_class[label]
        if len(indices) < train_per_class + val_per_class:
            raise ValueError(f"Not enough images in class {label} for the requested splits")
        rng.shuffle(indices)
        train_indices.extend(indices[:train_per_class])
        val_indices.extend(indices[train_per_class:train_per_class + val_per_class])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def get_cifar10_datasets(data_dir="data/raw", train_size=2000, val_size=5000, seed=42):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    root = str(Path(data_dir))
    train_data = CIFAR10(root=root, train=True, download=True, transform=train_transform)
    val_data = CIFAR10(root=root, train=True, download=True, transform=eval_transform)
    test_data = CIFAR10(root=root, train=False, download=True, transform=eval_transform)

    train_indices, val_indices = balanced_split_indices(
        train_data.targets, train_size=train_size, val_size=val_size, seed=seed
    )
    return Subset(train_data, train_indices), Subset(val_data, val_indices), test_data


def get_cifar10_loaders(
    data_dir="data/raw", train_size=2000, val_size=5000, batch_size=16, seed=42
):
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")

    train_data, val_data, test_data = get_cifar10_datasets(
        data_dir=data_dir, train_size=train_size, val_size=val_size, seed=seed
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=0)
    return train_loader, val_loader, test_loader
