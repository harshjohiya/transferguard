"""Train one CIFAR-10 MobileNetV3-Small baseline."""

import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import torch
from torch import nn

from src.datasets.cifar10 import IMAGENET_MEAN, IMAGENET_STD, get_cifar10_loaders
from src.models.mobilenet import MODES, get_model, set_train_mode


DEFAULT_LR = {"scratch": 0.001, "frozen": 0.001, "finetune": 0.0001}
RUNS_DIR = Path("runs")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="scratch")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--lr", type=float)
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0 or args.train_size <= 0 or args.val_size <= 0:
        parser.error("epochs, batch size, and split sizes must be positive")
    if args.lr is not None and args.lr <= 0:
        parser.error("learning rate must be positive")
    return args


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_run_dir(mode):
    parent = RUNS_DIR / mode
    parent.mkdir(parents=True, exist_ok=True)
    while True:
        name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
        run_dir = parent / name
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            continue


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    total_loss = correct = seen = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            count = labels.size(0)
            total_loss += loss.item() * count
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += count
    return total_loss / seen, correct / seen


def save_checkpoint(path, model, epoch, val_loss, val_accuracy, mode):
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "mode": mode,
    }, path)


def main(argv=None):
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
    seed_everything(args.seed)
    device = torch.device(args.device)
    lr = args.lr if args.lr is not None else DEFAULT_LR[args.mode]
    train_loader, val_loader, _ = get_cifar10_loaders(
        train_size=args.train_size, val_size=args.val_size,
        batch_size=args.batch_size, seed=args.seed,
    )
    model = get_model(args.mode).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=lr
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    run_dir = make_run_dir(args.mode)
    config = {
        "mode": args.mode, "epochs": args.epochs, "batch_size": args.batch_size,
        "train_size": args.train_size, "val_size": args.val_size, "seed": args.seed,
        "device": args.device, "lr": lr, "num_workers": 0,
        "dataset": "CIFAR-10 official training partition",
        "split": "balanced_split_indices", "image_size": [128, 128],
        "eval_transform": "Resize(128,128), ToTensor, ImageNet Normalize",
        "normalization_mean": IMAGENET_MEAN, "normalization_std": IMAGENET_STD,
        "loss": "CrossEntropyLoss", "selection_metric": "val_accuracy",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Run directory: {run_dir}")

    best_accuracy = -1.0
    best_loss = None
    best_epoch = None
    start = time.perf_counter()
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=(
            "epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "elapsed_seconds"
        ))
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            set_train_mode(model, args.mode)
            train_loss, train_accuracy = run_epoch(
                model, train_loader, criterion, device, optimizer, scaler
            )
            model.eval()
            val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
            writer.writerow({
                "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
                "val_loss": val_loss, "val_accuracy": val_accuracy,
                "elapsed_seconds": time.perf_counter() - start,
            })
            file.flush()
            if val_accuracy > best_accuracy:
                best_accuracy, best_loss, best_epoch = val_accuracy, val_loss, epoch
                save_checkpoint(run_dir / "best_model.pth", model, epoch, val_loss, val_accuracy, args.mode)
            print(f"Epoch {epoch}/{args.epochs}: train acc {train_accuracy:.4f}, "
                  f"val acc {val_accuracy:.4f}, val loss {val_loss:.4f}")

    summary = {
        "mode": args.mode, "best_val_accuracy": best_accuracy,
        "best_val_loss": best_loss, "best_epoch": best_epoch,
        "training_seconds": time.perf_counter() - start, "epochs": args.epochs,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Best validation accuracy: {best_accuracy:.4f} (epoch {best_epoch})")
    return run_dir


if __name__ == "__main__":
    main()
