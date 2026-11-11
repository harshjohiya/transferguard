"""Exploratory guard-rule comparison on the existing seed-42 CIFAR-10 runs.

This reuses development conditions; it is not independent detector validation.
"""
import pandas as pd
import numpy as np
from pathlib import Path

root = Path(__file__).resolve().parent.parent
results_dir = root / "results"

# Load results
scratch = pd.read_csv(root / "transferguard_cifar10_results" / "scratch_small_history.csv")
frozen40 = pd.read_csv(root / "transferguard_noise40_results" / "frozen_noise40_reset_history.csv")
frozen80 = pd.read_csv(root / "transferguard_cifar10_results" / "frozen_noise80_history.csv")
probes = pd.read_csv(root / "transferguard_refine_results" / "feature_probe.csv")


# Extract pilot and final accuracies
def pilot_accuracy(history):
    early = history[history["epoch"] <= 5]
    return float(early["val_acc"].iloc[-1])


def best_accuracy(history):
    return float(history["val_acc"].max())


scratch_pilot = pilot_accuracy(scratch)
scratch_final = best_accuracy(scratch)

frozen40_pilot = pilot_accuracy(frozen40)
frozen80_pilot = pilot_accuracy(frozen80)

frozen40_final = best_accuracy(frozen40)
frozen80_final = best_accuracy(frozen80)


# Centroid accuracies
centroid40 = float(probes.loc[probes["source_noise"] == 40, "centroid_accuracy"].iloc[0])
centroid80 = float(probes.loc[probes["source_noise"] == 80, "centroid_accuracy"].iloc[0])


# Decision thresholds
CENTROID_THRESHOLD = 20
PILOT_THRESHOLD = 2

conditions = {
    40: (centroid40, frozen40_pilot, frozen40_final),
    80: (centroid80, frozen80_pilot, frozen80_final)
}

rows = []

for noise, (centroid, frozen_pilot, frozen_final) in conditions.items():

    feature_condition = centroid >= CENTROID_THRESHOLD
    pilot_condition = frozen_pilot >= scratch_pilot + PILOT_THRESHOLD
    combined_condition = feature_condition and pilot_condition

    strategies = {
        "feature_only": feature_condition,
        "pilot_only": pilot_condition,
        "combined": combined_condition
    }

    oracle_accuracy = max(scratch_final, frozen_final)

    for strategy, choose_frozen in strategies.items():

        selected_method = "frozen" if choose_frozen else "scratch"
        selected_accuracy = frozen_final if choose_frozen else scratch_final

        regret = oracle_accuracy - selected_accuracy
        correct = np.isclose(selected_accuracy, oracle_accuracy)

        rows.append({
            "source_noise": noise,
            "strategy": strategy,
            "centroid_accuracy": centroid,
            "scratch_pilot": scratch_pilot,
            "frozen_pilot": frozen_pilot,
            "pilot_difference": frozen_pilot - scratch_pilot,
            "selected_method": selected_method,
            "selected_accuracy": selected_accuracy,
            "oracle_accuracy": oracle_accuracy,
            "regret": regret,
            "correct_decision": correct
        })


# Results
results = pd.DataFrame(rows)

print("\nDetailed Results")
print(results.to_string(index=False))


# Summary
summary = results.groupby("strategy").agg(
    selection_accuracy=("correct_decision", "mean"),
    mean_regret=("regret", "mean"),
    total_conditions=("source_noise", "count")
).reset_index()

summary["selection_accuracy"] *= 100

print("\nSummary")
print(summary.to_string(index=False))


# Save results
results_dir.mkdir(exist_ok=True)
results.to_csv(results_dir / "primary_model_verification.csv", index=False)
summary.to_csv(results_dir / "primary_model_verification_summary.csv", index=False)

print("\nResults saved successfully.")
