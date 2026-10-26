"""Train TransferGuard candidates; select once on held-out validation images."""

import argparse
import csv
import hashlib
import json
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from experiments.baselines.train import run_epoch, save_checkpoint, seed_everything
from experiments.negative_transfer.train_target import load_svhn_model
from experiments.transferability.compare import load_run
from src.datasets.cifar10 import (
    IMAGENET_MEAN, IMAGENET_STD, balanced_split_indices,
    get_cifar10_eval_transform, get_cifar10_train_transform,
)
from src.guard.decision import paired_guard, split_validation_indices
from src.guard.policies import POLICIES, apply_policy, set_policy_train_mode
from src.models.mobilenet import get_model


RUNS_DIR = Path("runs/transferguard")
LOGME_DIR = Path("runs/logme")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("auto", "imagenet", "svhn"), default="auto")
    parser.add_argument("--source-checkpoint", type=Path, help="Phase 3 SVHN best_model.pth")
    parser.add_argument("--imagenet-logme-run", type=Path)
    parser.add_argument("--svhn-logme-run", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--scratch-lr", type=float, default=0.001)
    parser.add_argument("--frozen-lr", type=float, default=0.001)
    parser.add_argument("--late-lr", type=float, default=0.0001)
    parser.add_argument("--full-lr", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.scratch_lr, args.frozen_lr,
           args.late_lr, args.full_lr, args.bootstrap_samples) <= 0:
        parser.error("Epochs, batch size, learning rates, and bootstrap samples must be positive")
    if args.source == "imagenet" and args.source_checkpoint:
        parser.error("--source-checkpoint is only valid for SVHN")
    return args


def digest(values):
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def target_loaders(batch_size, seed):
    """Load only the official training partition; reserve guard images unseen."""
    train_data = CIFAR10("data/raw", train=True, download=True,
                         transform=get_cifar10_train_transform())
    eval_data = CIFAR10("data/raw", train=True, download=True,
                        transform=get_cifar10_eval_transform())
    train_indices, validation_indices = balanced_split_indices(train_data.targets, 2000, 5000, seed)
    selection_indices, guard_indices = split_validation_indices(validation_indices, train_data.targets, seed)
    if set(train_indices) & (set(selection_indices) | set(guard_indices)):
        raise RuntimeError("Training and validation indices overlap")
    train_loader = DataLoader(Subset(train_data, train_indices), batch_size=batch_size,
                              shuffle=True, num_workers=0)
    select_loader = DataLoader(Subset(eval_data, selection_indices), batch_size=batch_size,
                               shuffle=False, num_workers=0)
    guard_loader = DataLoader(Subset(eval_data, guard_indices), batch_size=batch_size,
                              shuffle=False, num_workers=0)
    identity = {
        "indices_sha256": digest(train_indices),
        "labels_sha256": digest([int(train_data.targets[i]) for i in train_indices]),
        "selection_indices_sha256": digest(selection_indices),
        "guard_indices_sha256": digest(guard_indices),
    }
    return train_loader, select_loader, guard_loader, identity


def load_logme_scores(args, identity):
    """Read saved final-stage scores and reject mismatched target preprocessing."""
    runs = {}
    rejected = []
    for source in ("imagenet", "svhn"):
        explicit = getattr(args, f"{source}_logme_run")
        paths = [explicit] if explicit else sorted((LOGME_DIR / source).glob("*/"), reverse=True)
        for path in paths:
            try:
                config, rows = load_run(path, source)
                expected = {
                    "architecture": "MobileNetV3-Small",
                    "dataset": "CIFAR-10 official training partition",
                    "split": "balanced_split_indices", "train_size": 2000, "val_size": 5000,
                    "seed": args.seed, "image_size": [128, 128],
                    "eval_transform": "Resize(128,128), ToTensor, ImageNet Normalize",
                    "normalization_mean": list(IMAGENET_MEAN),
                    "normalization_std": list(IMAGENET_STD),
                    "indices_sha256": identity["indices_sha256"],
                    "labels_sha256": identity["labels_sha256"],
                    "logme_variant": "official later fixed-point; direct SVD, float64",
                }
                for field, value in expected.items():
                    if config.get(field) != value:
                        raise ValueError(f"Incompatible {source} LogME run: {field}")
                row = next((row for row in rows if row["stage"] == "final"), None)
                if row is None or config["feature_stages"]["final"] != [12, 576]:
                    raise ValueError(f"{path} lacks the expected final-stage score")
                if source == "svhn" and args.source_checkpoint:
                    saved = config.get("source_checkpoint") or {}
                    actual = hashlib.sha256(args.source_checkpoint.read_bytes()).hexdigest()
                    if saved.get("sha256") != actual:
                        raise ValueError("SVHN LogME score belongs to a different source checkpoint")
                if source == "svhn" and not args.source_checkpoint:
                    saved = config.get("source_checkpoint") or {}
                    if not saved.get("path") or not saved.get("sha256"):
                        raise ValueError("SVHN LogME run lacks checkpoint identity")
                runs[source] = {"run": str(path), "score": float(row["logme_score"]),
                                "config": config}
                break
            except (OSError, KeyError, TypeError, ValueError) as error:
                if explicit:
                    raise ValueError(f"Invalid {source} LogME run {path}: {error}") from error
                rejected.append(f"{path}: {error}")
        # An absent score excludes that source from automatic ranking.
    if not runs:
        raise ValueError("No compatible saved ImageNet or SVHN final-stage LogME score. "
                         + "; ".join(rejected[:4]))
    variants = {run["config"]["logme_variant"] for run in runs.values()}
    if len(variants) != 1:
        raise ValueError("LogME runs use different scoring variants")
    return runs


def choose_source(args, scores):
    if args.source == "auto":
        source = max(scores, key=lambda name: scores[name]["score"])
    else:
        source = args.source
        if source not in scores:
            raise ValueError(f"No compatible saved LogME score for {source}")
    checkpoint = args.source_checkpoint
    if source == "svhn" and checkpoint is None:
        checkpoint = Path(scores[source]["config"]["source_checkpoint"]["path"])
    return source, checkpoint


def make_run_dir():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        path = RUNS_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue


def train_model(model, mode, train_loader, select_loader, device, epochs, lr, path, seed):
    path.mkdir()
    # Keep the minibatch order and augmentation stream reproducible per candidate.
    seed_everything(seed)
    model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    best_accuracy, best_epoch = -1.0, None
    with (path / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=(
            "epoch", "train_loss", "train_accuracy", "selection_loss",
            "selection_accuracy", "elapsed_seconds"))
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            if mode == "scratch":
                model.train()
            else:
                set_policy_train_mode(model, mode)
            train_loss, train_accuracy = run_epoch(
                model, train_loader, criterion, device, optimizer, scaler)
            model.eval()
            val_loss, val_accuracy = run_epoch(model, select_loader, criterion, device)
            writer.writerow({"epoch": epoch, "train_loss": train_loss,
                             "train_accuracy": train_accuracy, "selection_loss": val_loss,
                             "selection_accuracy": val_accuracy,
                             "elapsed_seconds": time.perf_counter() - start})
            file.flush()
            if val_accuracy > best_accuracy:
                best_accuracy, best_epoch = val_accuracy, epoch
                save_checkpoint(path / "best_model.pth", model, epoch, val_loss, val_accuracy, mode)
            print(f"{mode} {epoch}/{epochs}: selection accuracy {val_accuracy:.4f}")
    return {"selection_accuracy": best_accuracy, "best_epoch": best_epoch,
            "training_seconds": time.perf_counter() - start,
            "gpu_peak_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}


def predict_checkpoint(model, checkpoint, loader, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    labels, predictions = [], []
    with torch.inference_mode():
        for images, batch_labels in loader:
            predictions.extend(model(images.to(device)).argmax(dim=1).cpu().tolist())
            labels.extend(batch_labels.tolist())
    return labels, predictions


def main(argv=None):
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_loader, select_loader, guard_loader, identity = target_loaders(args.batch_size, args.seed)
    scores = load_logme_scores(args, identity)
    source, checkpoint = choose_source(args, scores)
    if source == "svhn":
        source_model, checkpoint, _ = load_svhn_model(checkpoint)
        saved_hash = scores[source]["config"]["source_checkpoint"]["sha256"]
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != saved_hash:
            raise ValueError("SVHN checkpoint differs from its LogME source")
    else:
        source_model = get_model("finetune")
    # One source model, including one newly initialized CIFAR-10 output layer,
    # is copied before any candidate is trained.
    run_dir = make_run_dir()
    config = {**vars(args), **identity, "requested_source": args.source,
              "source": source, "source_checkpoint": str(checkpoint) if checkpoint else None,
              "logme_scores": {name: {"score": item["score"], "run": item["run"]}
                               for name, item in scores.items()},
              "dataset": "CIFAR-10 official training partition", "train_size": 2000,
              "selection_size": 2500, "guard_size": 2500,
              "validation_use": "selection for checkpoints/policy; guard once for final decision"}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Run directory: {run_dir}; source: {source}; LogME: {scores[source]['score']:.6f}")
    results = {}
    scratch = get_model("scratch")
    results["scratch"] = train_model(scratch, "scratch", train_loader, select_loader,
                                      device, args.epochs, args.scratch_lr, run_dir / "scratch", args.seed)
    del scratch
    for policy in POLICIES:
        model = apply_policy(deepcopy(source_model), policy)
        results[policy] = train_model(model, policy, train_loader, select_loader,
                                      device, args.epochs, getattr(args, f"{policy}_lr"),
                                      run_dir / policy, args.seed)
        del model
    best_policy = max(POLICIES, key=lambda name: results[name]["selection_accuracy"])
    transfer_model = apply_policy(deepcopy(source_model), best_policy)
    transfer_labels, transfer_predictions = predict_checkpoint(
        transfer_model, run_dir / best_policy / "best_model.pth", guard_loader, device)
    del transfer_model, source_model
    scratch = get_model("scratch")
    scratch_labels, scratch_predictions = predict_checkpoint(
        scratch, run_dir / "scratch" / "best_model.pth", guard_loader, device)
    if transfer_labels != scratch_labels:
        raise RuntimeError("Guard predictions are not aligned on the same images")
    guard = paired_guard(transfer_labels, transfer_predictions, scratch_predictions,
                         seed=args.seed, bootstrap_samples=args.bootstrap_samples)
    summary = {"source": source, "logme_score": scores[source]["score"],
               "selected_policy": best_policy, "selection_accuracies":
               {name: result["selection_accuracy"] for name, result in results.items()},
               "guard": guard, "candidate_guard_accuracy": guard["transfer_accuracy"],
               "scratch_guard_accuracy": guard["scratch_accuracy"],
               "final_decision": guard["decision"], "training": results,
               "total_training_seconds": sum(result["training_seconds"] for result in results.values()),
               "gpu_peak_bytes": max((result["gpu_peak_bytes"] or 0) for result in results.values())
               if device.type == "cuda" else None}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Selected policy: {best_policy}; guard difference: {guard['accuracy_difference']:.4f}; "
          f"95% CI [{guard['ci_lower']:.4f}, {guard['ci_upper']:.4f}]; decision: {guard['decision']}")
    return run_dir


if __name__ == "__main__":
    main()
