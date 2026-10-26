"""Fixed validation split and paired, one-sided transfer decision."""

from collections import defaultdict
from random import Random

import numpy as np


def split_validation_indices(indices, targets, seed=42):
    """Split Phase 1's balanced 5,000 indices into two balanced 2,500 sets."""
    by_class = defaultdict(list)
    for index in indices:
        by_class[int(targets[index])].append(index)
    if len(indices) != 5000 or len(set(indices)) != 5000 or len(by_class) != 10:
        raise ValueError("Expected 5,000 distinct CIFAR-10 validation indices")
    if any(len(group) != 500 for group in by_class.values()):
        raise ValueError("Expected 500 validation images per class")
    selection, guard = [], []
    for label in sorted(by_class):
        selection.extend(by_class[label][:250])
        guard.extend(by_class[label][250:])
    rng = Random(seed)
    rng.shuffle(selection)
    rng.shuffle(guard)
    return selection, guard


def paired_guard(labels, transfer_predictions, scratch_predictions, seed=42, bootstrap_samples=10000):
    """Accept transfer only when its paired 95% bootstrap lower bound exceeds zero."""
    labels = np.asarray(labels)
    transfer = np.asarray(transfer_predictions)
    scratch = np.asarray(scratch_predictions)
    if labels.ndim != 1 or not len(labels) or transfer.shape != labels.shape or scratch.shape != labels.shape:
        raise ValueError("Labels and both prediction vectors must be aligned and nonempty")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    transfer_correct = transfer == labels
    scratch_correct = scratch == labels
    paired_difference = transfer_correct.astype(np.int8) - scratch_correct.astype(np.int8)
    rng = np.random.default_rng(seed)
    # Small chunks bound memory use for large bootstrap counts.
    means = []
    for start in range(0, bootstrap_samples, 1000):
        count = min(1000, bootstrap_samples - start)
        draws = rng.integers(0, len(labels), size=(count, len(labels)))
        means.append(paired_difference[draws].mean(axis=1))
    lower, upper = np.quantile(np.concatenate(means), [0.025, 0.975])
    return {
        "transfer_accuracy": float(transfer_correct.mean()),
        "scratch_accuracy": float(scratch_correct.mean()),
        "accuracy_difference": float(paired_difference.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "decision": "transfer" if lower > 0 else "scratch",
    }
