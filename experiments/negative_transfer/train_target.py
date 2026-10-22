"""Fine-tune SVHN-pretrained MobileNetV3-Small on CIFAR-10."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from experiments.baselines.train import seed_everything
from experiments.negative_transfer.common import train_and_save
from src.datasets.cifar10 import IMAGENET_MEAN, IMAGENET_STD, get_cifar10_loaders
from src.models.mobilenet import get_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--lr", type=float, default=0.0001)
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.train_size, args.val_size, args.lr) <= 0:
        parser.error("epochs, batch size, split sizes, and learning rate must be positive")
    return args


def load_svhn_model(checkpoint_path):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=True)
    config_path = checkpoint_path.with_name("config.json")
    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    if (source_config.get("mode") != "source_svhn"
            or source_config.get("architecture") != "MobileNetV3-Small"
            or source_config.get("num_classes") != 10
            or source_config.get("dataset") != "SVHN official training partition"):
        raise ValueError("Source configuration is not a compatible SVHN MobileNetV3-Small run")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("mode") != "source_svhn":
        raise ValueError("Checkpoint is not a Phase 3 SVHN source checkpoint")
    model = get_model("scratch")
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (KeyError, RuntimeError, TypeError) as error:
        raise ValueError("Checkpoint weights do not match MobileNetV3-Small") from error
    old_layer = model.classifier[-1]
    model.classifier[-1] = nn.Linear(old_layer.in_features, 10)
    return model, checkpoint_path, source_config


def main(argv=None):
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
    seed_everything(args.seed)
    model, checkpoint_path, source_config = load_svhn_model(args.source_checkpoint)
    train_loader, val_loader, _ = get_cifar10_loaders(
        train_size=args.train_size, val_size=args.val_size,
        batch_size=args.batch_size, seed=args.seed,
    )
    config = {
        "mode": "svhn_finetune", "epochs": args.epochs, "batch_size": args.batch_size,
        "train_size": args.train_size, "val_size": args.val_size, "seed": args.seed,
        "device": args.device, "lr": args.lr, "num_workers": 0,
        "dataset": "CIFAR-10 official training partition",
        "split": "balanced_split_indices", "image_size": [128, 128],
        "eval_transform": "Resize(128,128), ToTensor, ImageNet Normalize",
        "normalization_mean": IMAGENET_MEAN, "normalization_std": IMAGENET_STD,
        "loss": "CrossEntropyLoss", "selection_metric": "val_accuracy",
        "initialization": "SVHN-pretrained features; new random classifier output",
        "source_checkpoint": str(checkpoint_path), "source_config": source_config,
    }
    return train_and_save(
        model, "svhn_finetune", train_loader, val_loader,
        config, torch.device(args.device),
    )


if __name__ == "__main__":
    main()
