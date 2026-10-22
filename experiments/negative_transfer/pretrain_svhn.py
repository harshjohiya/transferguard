"""Pretrain MobileNetV3-Small on the SVHN training partition."""

import argparse

import torch

from experiments.baselines.train import seed_everything
from experiments.negative_transfer.common import train_and_save
from src.datasets.cifar10 import IMAGENET_MEAN, IMAGENET_STD
from src.datasets.svhn import get_svhn_loaders
from src.models.mobilenet import get_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.train_size, args.val_size, args.lr) <= 0:
        parser.error("epochs, batch size, split sizes, and learning rate must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
    seed_everything(args.seed)
    train_loader, val_loader = get_svhn_loaders(
        train_size=args.train_size, val_size=args.val_size,
        batch_size=args.batch_size, seed=args.seed,
    )
    config = {
        "mode": "source_svhn", "architecture": "MobileNetV3-Small", "num_classes": 10,
        "initialization": "random", "dataset": "SVHN official training partition",
        "split": "balanced_split_indices", "train_size": args.train_size,
        "val_size": args.val_size, "seed": args.seed, "batch_size": args.batch_size,
        "epochs": args.epochs, "device": args.device, "lr": args.lr, "num_workers": 0,
        "image_size": [128, 128],
        "eval_transform": "Resize(128,128), ToTensor, ImageNet Normalize",
        "normalization_mean": IMAGENET_MEAN, "normalization_std": IMAGENET_STD,
        "loss": "CrossEntropyLoss", "selection_metric": "val_accuracy",
    }
    return train_and_save(
        get_model("scratch"), "source_svhn", train_loader, val_loader,
        config, torch.device(args.device),
    )


if __name__ == "__main__":
    main()
