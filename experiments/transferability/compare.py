"""Compare explicit, compatible ImageNet, SVHN, and random LogME runs."""

import argparse
import csv
import json
import math
from pathlib import Path


MATCH_FIELDS = (
    "architecture", "dataset", "split", "train_size", "val_size", "seed", "layers",
    "feature_stages", "image_size", "eval_transform", "normalization_mean",
    "normalization_std", "indices_sha256", "labels_sha256", "logme_variant",
)


def load_run(path, source):
    path = Path(path)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("source") != source:
        raise ValueError(f"{path} is not a {source} run")
    with (path / "scores.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows or len({row["stage"] for row in rows}) != len(rows):
        raise ValueError(f"{path} has missing or duplicate feature stage scores")
    if set(row["stage"] for row in rows) != set(config["feature_stages"]):
        raise ValueError(f"{path} scores do not match its configured stages")
    for row in rows:
        if row["source"] != source or int(row["feature_dim"]) != config["feature_stages"][row["stage"]][1]:
            raise ValueError(f"{path} has inconsistent score metadata")
        if int(row["num_examples"]) != config["train_size"]:
            raise ValueError(f"{path} has an inconsistent example count")
        if not math.isfinite(float(row["logme_score"])):
            raise ValueError(f"{path} has a non-finite LogME score")
        for field in ("extraction_seconds", "scoring_seconds"):
            if not math.isfinite(float(row[field])) or float(row[field]) < 0:
                raise ValueError(f"{path} has invalid timing metadata: {field}")
    return config, rows


def compare_runs(runs):
    baseline = runs["imagenet"][0]
    for source, (config, _) in runs.items():
        for field in MATCH_FIELDS:
            if field not in config or field not in baseline or config[field] != baseline[field]:
                raise ValueError(f"Incompatible {source} run: {field}")
    stages = list(baseline["feature_stages"])
    result = []
    for stage in stages:
        stage_rows = [next(row for row in runs[source][1] if row["stage"] == stage)
                      for source in ("imagenet", "svhn", "random")]
        ordered = sorted(stage_rows, key=lambda row: float(row["logme_score"]), reverse=True)
        for row in stage_rows:
            result.append({**row, "rank": ordered.index(row) + 1})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for source in ("imagenet", "svhn", "random"):
        parser.add_argument(f"--{source}-run", type=Path, required=True)
    parser.add_argument("--phase3-observed", action="store_true",
                        help="Show the single-seed validation accuracies supplied in the Phase 4 brief")
    args = parser.parse_args(argv)
    runs = {source: load_run(getattr(args, f"{source}_run"), source)
            for source in ("imagenet", "svhn", "random")}
    rows = compare_runs(runs)
    print("Source     Stage   Dim  Examples  LogME       Rank  Extract(s)  Score(s)")
    for row in rows:
        print(f"{row['source']:<10} {row['stage']:<7} {int(row['feature_dim']):>3}  "
              f"{int(row['num_examples']):>8}  {float(row['logme_score']):>10.6f}  "
              f"{row['rank']:>4}  {float(row['extraction_seconds']):>10.2f}  "
              f"{float(row['scoring_seconds']):>8.2f}")
    if args.phase3_observed:
        print("\nExploratory Phase 3 single-seed CIFAR-10 validation accuracies from the brief:")
        print("scratch 57.14%; ImageNet fine-tuning 85.60%; SVHN fine-tuning 55.50%")
        print("These do not establish correlation or reliable negative-transfer detection.")
    return rows


if __name__ == "__main__":
    main()
