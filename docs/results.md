# Exploratory CIFAR-10 results

These are **previously measured, exploratory single-run best validation accuracies** from the local CSV histories. They use seed `42`, the same fixed CIFAR-10 split, and 2,000 clean validation images. They are not official test results or repeated-seed estimates. The generated CSV files and checkpoints stay in ignored local results directories.

| Method | Source-label corruption | Best validation accuracy | Local evidence |
| --- | ---: | ---: | --- |
| Scratch SmallCNN | None | 57.15% | `transferguard_cifar10_results/control_validation_results.csv` |
| Scratch BigCNN | None | 57.35% | `transferguard_cifar10_results/control_validation_results.csv` |
| Frozen transfer | 40% | 67.65% | `transferguard_noise40_results/comparison_noise40.csv` |
| Frozen transfer | 80% | 10.65% | `transferguard_cifar10_results/control_validation_results.csv` |
| REFINE | 40% | 67.95% | `transferguard_refine_results/validation_comparison.csv` |
| REFINE | 80% | 56.40% | `transferguard_refine_results/validation_comparison.csv` |

The 40% frozen entry is the final-classifier-reset run. At 80%, both the original frozen and final-classifier-reset frozen histories peaked at 10.65%. A separate 40% final-layer-reset full fine-tuning run reached 69.40% (`transferguard_noise40_results/comparison_noise40.csv`). At 80%, final-layer-reset fine-tuning stayed at 10.65%, while resetting **both fully connected layers** before fine-tuning reached 57.70% (`transferguard_cifar10_results/finetune_noise80_repaired_v1_history.csv`), approximately the 57.35% scratch BigCNN result. These are additional observed runs, not a general claim that resetting layers always repairs transfer.

## Observed feature health

The 80% source checkpoint showed network activation collapse in the recorded source-health analysis: mean feature standard deviation was about `1.78e-9`, and nearly all measured feature entries were zero (`transferguard_refine_results/source_health.csv`). The 40% source had mean feature standard deviation about `0.0197`. The local feature probe reported centroid accuracies of 64.2% at 40% noise and 9.8% at 80% noise (`transferguard_refine_results/feature_probe.csv`). These observations are consistent with the 80% frozen-transfer failure; they do not establish a universal causal mechanism.

## Hypotheses and next experiments

- **Hypothesis:** the collapsed 80% source features leave too little usable information for a frozen classifier, whereas REFINE's trainable target branch can learn residual target information.
- **Hypothesis:** early feature quality combined with a short pilot comparison may help choose between scratch and frozen transfer.
- **Next:** repeat across seeds, independent source corruptions and target tasks; predefine detector thresholds before testing on unseen conditions; report uncertainty and official test accuracy only after selection is frozen.

The current TransferGuard rule was demonstrated on the same two conditions used to inspect it. The local `guard_demo.csv` reports zero regret for those two conditions, but this is **not independent detector validation**. REFINE is attributed to [Xu et al., *Residual Feature Integration is Sufficient to Prevent Negative Transfer*](https://arxiv.org/abs/2505.11771) and its [original code](https://github.com/Xyc-arch/ReFine-residual-feature-integration).
