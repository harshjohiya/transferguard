"""Frozen TransferGuard decision protocol (Workstream B).

This file is deliberately small and has NO torch dependency. It holds
  1. the decision thresholds, copied unchanged from the REFINE notebook (Step 10),
  2. the scoring / regret arithmetic, and
  3. the summary across conditions and seeds.

Commit this file BEFORE looking at any independent results. Do not edit the
thresholds afterwards; `independent_validation.py` records this file's SHA-256
in every run so any later change is visible.

Usage (summarise finished runs, no GPU needed):
    python experiments/guard_protocol.py summarize results/validation
"""
import csv
import json
import math
import statistics
import sys
from pathlib import Path

PROTOCOL_VERSION = "guard-v1"

# --- Frozen decision rule (identical to notebook 3, Step 10) ----------------
CENTROID_MIN_PCT = 20.0      # feature probe must reach at least this accuracy
PILOT_MARGIN_POINTS = 2.0    # frozen pilot must beat scratch pilot by this much
PILOT_EPOCHS = 5             # decision uses only the first 5 target epochs

# --- Diagnostic only, NOT part of the decision -------------------------------
COLLAPSE_STD = 1e-6          # mean feature std below this = collapsed source

# Conditions that were examined while designing the thresholds. Results on
# these are development results, never independent evidence.
DEVELOPMENT_CONDITIONS = {(42, 40), (42, 80)}


def pilot_score(val_curve, k=PILOT_EPOCHS):
    """Validation accuracy at epoch k (same as the notebook's pilot_score)."""
    if len(val_curve) < k:
        raise ValueError(f"need at least {k} epochs, got {len(val_curve)}")
    return float(val_curve[k - 1])


def guard_choice(centroid_acc, frozen_pilot, scratch_pilot):
    """Combined rule: probe AND pilot must both favour frozen transfer."""
    use_frozen = (centroid_acc >= CENTROID_MIN_PCT
                  and frozen_pilot >= scratch_pilot + PILOT_MARGIN_POINTS)
    return "frozen" if use_frozen else "scratch"


def pilot_only_choice(frozen_pilot, scratch_pilot):
    """Trivial baseline: ignore the probe (the issue asks to compare to this)."""
    return "frozen" if frozen_pilot >= scratch_pilot + PILOT_MARGIN_POINTS else "scratch"


def probe_only_choice(centroid_acc):
    """Feature-only baseline: ignore the pilot."""
    return "frozen" if centroid_acc >= CENTROID_MIN_PCT else "scratch"


def evaluate_condition(seed, noise, centroid_acc, feature_std,
                       scratch_curve, frozen_curve):
    """Score every selector for one (seed, noise) condition.

    scratch_curve / frozen_curve: validation accuracy per epoch (full runs).
    Decisions use only the probe and the first PILOT_EPOCHS epochs; the full
    curves are used afterwards to find the truly better method.
    """
    scratch_pilot = pilot_score(scratch_curve)
    frozen_pilot = pilot_score(frozen_curve)
    scratch_best = float(max(scratch_curve))
    frozen_best = float(max(frozen_curve))
    oracle_best = max(scratch_best, frozen_best)
    better = "frozen" if frozen_best > scratch_best else "scratch"  # tie -> scratch

    def final_acc(choice):
        return frozen_best if choice == "frozen" else scratch_best

    choices = {
        "guard": guard_choice(centroid_acc, frozen_pilot, scratch_pilot),
        "pilot_only": pilot_only_choice(frozen_pilot, scratch_pilot),
        "probe_only": probe_only_choice(centroid_acc),
    }
    row = {
        "seed": seed,
        "noise": noise,
        "protocol_version": PROTOCOL_VERSION,
        "development_condition": (seed, noise) in DEVELOPMENT_CONDITIONS,
        "centroid_acc": centroid_acc,
        "mean_feature_std": feature_std,
        "collapsed": feature_std < COLLAPSE_STD,
        "scratch_pilot": scratch_pilot,
        "frozen_pilot": frozen_pilot,
        "scratch_best": scratch_best,
        "frozen_best": frozen_best,
        "negative_transfer_gap": scratch_best - frozen_best,  # >0 means transfer hurt
        "better_method": better,
    }
    for name, choice in choices.items():
        regret = oracle_best - final_acc(choice)
        row[f"{name}_choice"] = choice
        row[f"{name}_regret"] = regret
        row[f"{name}_correct"] = regret == 0
    return row


def _mean_std(values):
    values = list(values)
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else float("nan")
    return mean, std


def summarize(rows):
    """One summary row per (noise, development/independent) group."""
    groups = {}
    for r in rows:
        key = (int(r["noise"]), "development" if r["development_condition"] else "independent")
        groups.setdefault(key, []).append(r)

    out = []
    for (noise, kind), rs in sorted(groups.items()):
        item = {"noise": noise, "kind": kind, "n_runs": len(rs),
                "seeds": " ".join(str(r["seed"]) for r in rs)}
        for col in ("centroid_acc", "scratch_best", "frozen_best", "negative_transfer_gap"):
            m, s = _mean_std(r[col] for r in rs)
            item[f"{col}_mean"], item[f"{col}_std"] = m, s
        for sel in ("guard", "pilot_only", "probe_only"):
            m, s = _mean_std(r[f"{sel}_regret"] for r in rs)
            item[f"{sel}_regret_mean"], item[f"{sel}_regret_std"] = m, s
            item[f"{sel}_correct"] = f"{sum(bool(r[f'{sel}_correct']) for r in rs)}/{len(rs)}"
        item["collapsed_runs"] = f"{sum(bool(r['collapsed']) for r in rs)}/{len(rs)}"
        out.append(item)
    return out


def load_rows(results_dir):
    rows = []
    for path in sorted(Path(results_dir).glob("seed*_noise*/condition_result.json")):
        rows.append(json.loads(path.read_text()))
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _fmt(x):
    return "nan" if isinstance(x, float) and math.isnan(x) else (f"{x:.2f}" if isinstance(x, float) else str(x))


def summarize_directory(results_dir):
    rows = load_rows(results_dir)
    if not rows:
        print(f"No finished conditions found in {results_dir}")
        return
    write_csv(Path(results_dir) / "per_condition.csv", rows)
    summary = summarize(rows)
    write_csv(Path(results_dir) / "summary_by_noise.csv", summary)
    print(f"{len(rows)} condition(s); protocol {PROTOCOL_VERSION}")
    for s in summary:
        print(f"\nnoise {s['noise']}% [{s['kind']}] n={s['n_runs']} seeds: {s['seeds']}")
        print(f"  scratch best  {_fmt(s['scratch_best_mean'])} +- {_fmt(s['scratch_best_std'])}")
        print(f"  frozen best   {_fmt(s['frozen_best_mean'])} +- {_fmt(s['frozen_best_std'])}")
        print(f"  gap (scratch-frozen) {_fmt(s['negative_transfer_gap_mean'])} +- {_fmt(s['negative_transfer_gap_std'])}")
        print(f"  collapsed sources: {s['collapsed_runs']}")
        for sel in ("guard", "pilot_only", "probe_only"):
            print(f"  {sel:10s} correct {s[f'{sel}_correct']:>5s}  regret "
                  f"{_fmt(s[f'{sel}_regret_mean'])} +- {_fmt(s[f'{sel}_regret_std'])}")
    print("\nstd is nan when a group has a single run.")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "summarize":
        summarize_directory(sys.argv[2])
    else:
        print(__doc__)
