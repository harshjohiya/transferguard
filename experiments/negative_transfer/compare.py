"""Compare matching CIFAR-10 runs from scratch, ImageNet, and SVHN."""

import argparse
import json
from pathlib import Path

from experiments.baselines.train import RUNS_DIR


MODES = ("scratch", "finetune", "svhn_finetune")
LABELS = {"scratch": "Scratch", "finetune": "ImageNet", "svhn_finetune": "SVHN"}
MATCH_FIELDS = (
    "dataset", "split", "train_size", "val_size", "seed", "image_size",
    "batch_size", "eval_transform", "normalization_mean", "normalization_std",
    "loss", "selection_metric", "epochs",
)


def load_run(run_dir, expected_mode):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if config.get("mode") != expected_mode or summary.get("mode") != expected_mode:
        raise ValueError(f"{run_dir} is not a {expected_mode} run")
    missing = [field for field in MATCH_FIELDS if field not in config]
    if missing:
        raise ValueError(f"{run_dir} lacks comparison settings: {', '.join(missing)}")
    return run_dir, config, summary


def choose_run(mode, selected):
    if selected is not None:
        return load_run(selected, mode)
    parent = RUNS_DIR / mode
    candidates = sorted(path for path in parent.glob("*") if (path / "summary.json").is_file())
    if not candidates:
        raise ValueError(f"No completed {mode} runs found in {parent}")
    if len(candidates) > 1:
        raise ValueError(f"Multiple {mode} runs found in {parent}; specify the run directory")
    return load_run(candidates[0], mode)


def compare_runs(scratch, imagenet, svhn):
    reference = scratch[1]
    for run in (imagenet, svhn):
        mismatches = [field for field in MATCH_FIELDS if run[1][field] != reference[field]]
        if mismatches:
            raise ValueError(f"Incompatible run {run[0]}: differing {', '.join(mismatches)}")
    print("Preliminary CIFAR-10 validation comparison")
    print("Transfer gap = transferred accuracy - scratch accuracy (percentage points)")
    print("Mode       Initialization   Best val acc   Val loss   Gap (pp)   Time (s)")
    print("-" * 76)
    scratch_accuracy = scratch[2]["best_val_accuracy"]
    for run in (scratch, imagenet, svhn):
        mode = run[1]["mode"]
        summary = run[2]
        gap = (summary["best_val_accuracy"] - scratch_accuracy) * 100
        print(f"{mode:<10} {LABELS[mode]:<16} {summary['best_val_accuracy']:>12.4f} "
              f"{summary['best_val_loss']:>10.4f} {gap:>10.2f} "
              f"{summary['training_seconds']:>10.1f}")
    print("A negative validation gap is a candidate observation, not definitive proof of negative transfer.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-run", type=Path)
    parser.add_argument("--imagenet-run", type=Path)
    parser.add_argument("--svhn-run", type=Path)
    args = parser.parse_args(argv)
    try:
        runs = [
            choose_run("scratch", args.scratch_run),
            choose_run("finetune", args.imagenet_run),
            choose_run("svhn_finetune", args.svhn_run),
        ]
        compare_runs(*runs)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
