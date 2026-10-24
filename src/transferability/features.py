"""Deterministic CIFAR-10 features from source MobileNetV3-Small backbones."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from experiments.negative_transfer.train_target import load_svhn_model
from src.datasets.cifar10 import balanced_split_indices, get_cifar10_eval_transform
from src.models.mobilenet import get_model


# TorchVision MobileNetV3-Small, default width_mult=1, reduced_tail=False.
STAGES = {"early": (1, 16), "middle": (8, 48), "final": (12, 576)}


def load_source_model(source, checkpoint=None):
    if source == "svhn":
        if checkpoint is None:
            raise ValueError("SVHN requires --source-checkpoint")
        model, path, config = load_svhn_model(checkpoint)
        information = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "source_config": config}
    elif source in ("imagenet", "random"):
        if checkpoint is not None:
            raise ValueError("--source-checkpoint is only valid for SVHN")
        model = get_model("finetune" if source == "imagenet" else "scratch")
        information = None
    else:
        raise ValueError(f"Unknown source: {source}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, information


def target_training_subset(data_dir="data/raw", train_size=2000, val_size=5000, seed=42):
    """Use Phase 1's training indices with its validation transform; never load test data."""
    dataset = CIFAR10(root=str(Path(data_dir)), train=True, download=True,
                      transform=get_cifar10_eval_transform())
    indices, validation_indices = balanced_split_indices(
        dataset.targets, train_size=train_size, val_size=val_size, seed=seed)
    if set(indices) & set(validation_indices):
        raise RuntimeError("Target training and validation indices overlap")
    labels = [int(dataset.targets[index]) for index in indices]
    digest = hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode()).hexdigest()
    label_digest = hashlib.sha256(json.dumps(labels, separators=(",", ":")).encode()).hexdigest()
    return Subset(dataset, indices), {"indices_sha256": digest, "labels_sha256": label_digest}


def extract_features(model, loader, device, stages=("final",)):
    """Return CPU feature matrices and labels in loader order, without gradients."""
    unknown = set(stages) - STAGES.keys()
    if unknown:
        raise ValueError(f"Unknown feature stages: {sorted(unknown)}")
    if not stages:
        raise ValueError("At least one feature stage is required")
    model.eval()
    outputs = {stage: [] for stage in stages}
    labels = []
    stage_at = {STAGES[name][0]: name for name in stages}
    with torch.inference_mode():
        for images, batch_labels in loader:
            x = images.to(device)
            for index, layer in enumerate(model.features):
                x = layer(x)
                if index in stage_at:
                    name = stage_at[index]
                    pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
                    if pooled.shape[1] != STAGES[name][1]:
                        raise RuntimeError(f"Unexpected {name} feature dimension: {pooled.shape[1]}")
                    outputs[name].append(pooled.cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
    if not labels:
        raise ValueError("Cannot extract features from an empty loader")
    return {name: np.concatenate(chunks) for name, chunks in outputs.items()}, np.concatenate(labels)


def target_loader(subset, batch_size):
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
