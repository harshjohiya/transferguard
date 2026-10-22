import argparse
from collections import Counter
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.utils import make_grid

from .cifar10 import IMAGENET_MEAN, IMAGENET_STD, get_cifar10_loaders


def class_counts(dataset):
    if hasattr(dataset, "indices"):
        return Counter(dataset.dataset.targets[index] for index in dataset.indices)
    return Counter(dataset.targets)


def check_batch(name, loader):
    images, labels = next(iter(loader))
    assert images.ndim == 4 and images.shape[1:] == (3, 128, 128)
    assert labels.ndim == 1 and len(labels) == len(images)
    assert torch.all((labels >= 0) & (labels < 10))
    counts = class_counts(loader.dataset)
    assert set(counts) == set(range(10))
    assert len(set(counts.values())) == 1
    print(f"{name}: {len(loader.dataset)} images, batch {tuple(images.shape)}")
    print(f"  classes: {dict(sorted(counts.items()))}")
    return images


def main():
    parser = argparse.ArgumentParser(description="Verify the CIFAR-10 data pipeline")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/figures/cifar10_samples.png"))
    args = parser.parse_args()

    train, val, test = get_cifar10_loaders(
        data_dir=args.data_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    assert len(test.dataset) == 10000
    assert not set(train.dataset.indices).intersection(val.dataset.indices)

    images = check_batch("Train", train)
    check_batch("Validation", val)
    check_batch("Test", test)
    print("Training and validation indices: no overlap")

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    samples = (images[:16].cpu() * std + mean).clamp(0, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    transforms.ToPILImage()(make_grid(samples, nrow=4)).save(args.output)
    print(f"Sample grid: {args.output}")

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
