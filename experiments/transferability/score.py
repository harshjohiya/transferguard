"""Score frozen source representations on the Phase 1 CIFAR-10 training subset."""

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import torch

from experiments.baselines.train import seed_everything
from src.datasets.cifar10 import IMAGENET_MEAN, IMAGENET_STD
from src.transferability.features import (
    STAGES, extract_features, load_source_model, target_loader, target_training_subset,
)
from src.transferability.logme import logme_score


RUNS_DIR = Path("runs/logme")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("imagenet", "svhn", "random"), required=True)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--layers", choices=("final", "all"), default="final")
    args = parser.parse_args(argv)
    if min(args.train_size, args.val_size, args.batch_size) <= 0:
        parser.error("Train size, validation size, and batch size must be positive")
    if args.source == "svhn" and args.source_checkpoint is None:
        parser.error("SVHN requires --source-checkpoint")
    if args.source != "svhn" and args.source_checkpoint is not None:
        parser.error("--source-checkpoint is only valid for SVHN")
    return args


def make_run_dir(source):
    parent = RUNS_DIR / source
    parent.mkdir(parents=True, exist_ok=True)
    while True:
        path = parent / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue


def main(argv=None):
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
    seed_everything(args.seed)
    model, checkpoint = load_source_model(args.source, args.source_checkpoint)
    model = model.to(torch.device(args.device))
    subset, identity = target_training_subset(
        train_size=args.train_size, val_size=args.val_size, seed=args.seed)
    stages = ("final",) if args.layers == "final" else tuple(STAGES)
    start = time.perf_counter()
    features, labels = extract_features(
        model, target_loader(subset, args.batch_size), torch.device(args.device), stages)
    extraction_seconds = time.perf_counter() - start
    label_hash = hashlib.sha256(json.dumps(labels.tolist(), separators=(",", ":")).encode()).hexdigest()
    if label_hash != identity["labels_sha256"] or len(labels) != args.train_size:
        raise RuntimeError("Extracted labels do not match the fixed training subset")
    rows = []
    for stage in stages:
        start = time.perf_counter()
        score = logme_score(features[stage], labels)
        rows.append({"source": args.source, "stage": stage,
                     "feature_dim": features[stage].shape[1], "num_examples": len(labels),
                     "logme_score": score, "extraction_seconds": extraction_seconds,
                     "scoring_seconds": time.perf_counter() - start})
    config = {
        "source": args.source, "architecture": "MobileNetV3-Small", "dataset": "CIFAR-10 official training partition",
        "split": "balanced_split_indices", "train_size": args.train_size, "val_size": args.val_size,
        "seed": args.seed, "batch_size": args.batch_size, "device": args.device,
        "layers": args.layers, "feature_stages": {stage: list(STAGES[stage]) for stage in stages},
        "image_size": [128, 128], "eval_transform": "Resize(128,128), ToTensor, ImageNet Normalize",
        "normalization_mean": IMAGENET_MEAN, "normalization_std": IMAGENET_STD,
        "logme_variant": "official later fixed-point; direct SVD, float64", **identity,
        "source_checkpoint": checkpoint,
    }
    run_dir = make_run_dir(args.source)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "scores.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"source": args.source, "indices_sha256": identity["indices_sha256"],
               "labels_sha256": identity["labels_sha256"], "scores": rows,
               "source_checkpoint": checkpoint}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Run directory: {run_dir}")
    for row in rows:
        print(f"{row['stage']}: {row['logme_score']:.6f} ({row['feature_dim']} dimensions)")
    return run_dir


if __name__ == "__main__":
    main()
