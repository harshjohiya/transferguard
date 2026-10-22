"""Print preliminary validation comparisons of baseline runs."""

import json

from experiments.baselines.train import RUNS_DIR
from src.models.mobilenet import MODES


MATCH_FIELDS = (
    "dataset", "split", "train_size", "val_size", "seed", "image_size",
    "batch_size", "eval_transform", "normalization_mean", "normalization_std",
    "loss", "selection_metric",
)


def main():
    runs = []
    for path in RUNS_DIR.glob("*/**/summary.json"):
        config_path = path.with_name("config.json")
        if not config_path.is_file():
            print(f"Skipping incomplete run: {path.parent}")
            continue
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            config = json.loads(config_path.read_text(encoding="utf-8"))
            key = tuple(json.dumps(config.get(field), sort_keys=True) for field in MATCH_FIELDS)
            runs.append((path.parent, summary, config, key))
        except (OSError, ValueError, TypeError) as error:
            print(f"Skipping unreadable run {path.parent}: {error}")

    latest = {}
    for run in sorted(runs, key=lambda item: item[0].name):
        mode = run[2].get("mode")
        if mode in MODES:
            latest[mode] = run
    if not latest:
        print("No completed baseline runs found in runs/.")
        return

    reference = latest.get("scratch")
    if reference is None:
        print("No scratch run found; accuracy differences cannot be calculated.")
        reference = next(iter(latest.values()))
    print("Preliminary validation comparisons (not evidence of definitive negative transfer)")
    print("Mode       Best val acc   Val loss   Time (s)   Epochs   Delta vs scratch")
    print("-" * 78)
    for mode in MODES:
        run = latest.get(mode)
        if run is None:
            continue
        path, summary, config, key = run
        if key != reference[3]:
            mismatches = [field for field in MATCH_FIELDS if config.get(field) != reference[2].get(field)]
            print(f"Incompatible {mode} run {path}: differing {', '.join(mismatches)}")
            continue
        delta = "n/a" if "scratch" not in latest else f"{summary['best_val_accuracy'] - latest['scratch'][1]['best_val_accuracy']:+.4f}"
        print(f"{mode:<10} {summary['best_val_accuracy']:>12.4f} "
              f"{summary['best_val_loss']:>10.4f} {summary['training_seconds']:>10.1f} "
              f"{summary['epochs']:>8} {delta:>18}")


if __name__ == "__main__":
    main()
