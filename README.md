# TransferGuard

TransferGuard is an exploratory study of **negative transfer** in lightweight vision models. Negative transfer occurs when using a pretrained source model makes a target task perform worse than training a suitable model from scratch. We compare scratch training, conventional transfer, and the published REFINE method under corrupted source labels, then examine an early warning rule for choosing whether to transfer.

The current experiments use CIFAR-10 with a fixed seed (`42`), disjoint source (10,000 images), target training (4,000), validation (2,000), and reserved calibration (500) subsets from the official training split. The official test set is reserved for final evaluation. The source labels are dynamically changed to an incorrect class on 40% or 80% of accesses. The notebooks use a SmallCNN and BigCNN based on the architecture dimensions in the [REFINE authors' CIFAR-10 implementation](https://github.com/Xyc-arch/ReFine-residual-feature-integration/blob/main/model_def_test10/model_def.py).

## Experiments

| Notebook | Purpose |
| --- | --- |
| [`TransferGuard_CIFAR10_80pct_Negative_Transfer_LocalGPU.ipynb`](notebooks/TransferGuard_CIFAR10_80pct_Negative_Transfer_LocalGPU.ipynb) | Train the 80% noisy source and compare scratch SmallCNN with frozen transfer; optional matched BigCNN scratch and full fine-tuning controls. |
| [`TransferGuard_CIFAR10_Noise40_Clean.ipynb`](notebooks/TransferGuard_CIFAR10_Noise40_Clean.ipynb) | Train the 40% noisy source; compare matched BigCNN scratch, frozen transfer, final-layer-reset fine-tuning, and both-FC-layer-reset fine-tuning. |
| [`TransferGuard_REFINE_40_80_Local_GPU.ipynb`](notebooks/TransferGuard_REFINE_40_80_Local_GPU.ipynb) | Compare REFINE at both noise levels with saved baselines, inspect source features, and demonstrate an exploratory two-signal choice between scratch and frozen transfer. |

Frozen transfer keeps the source feature extractor fixed and trains a classifier. Fine-tuning updates source weights on the clean target set. REFINE concatenates frozen source features with features from a trainable target branch. **REFINE is the work of Xu, Nakada, Zhang, and Li**, described in [*Residual Feature Integration is Sufficient to Prevent Negative Transfer*](https://arxiv.org/abs/2505.11771) ([original implementation](https://github.com/Xyc-arch/ReFine-residual-feature-integration)); this repository adapts its CIFAR-10 architecture for a local research comparison.

TransferGuard's current demonstration uses a small feature-quality probe on the reserved calibration set and the first five validation epochs of scratch and frozen runs. Its fixed rule chooses frozen transfer only when centroid accuracy is at least 20% and frozen pilot accuracy exceeds scratch pilot accuracy by at least two points. These conditions were examined on the same 40% and 80% settings used to develop the demonstration. **The detector has not been validated on independent unseen conditions.**

See [exploratory single-run results](docs/results.md) for measured validation accuracies, evidence about source activation collapse, and limitations.

## Local setup

Use Python 3.11 on Windows, a compatible NVIDIA driver, and a CUDA-capable GPU. From the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Select **Windows / Pip / Python / your supported CUDA platform** on the [official PyTorch install page](https://pytorch.org/get-started/locally/) and run its command to install a matching `torch` and `torchvision` pair. Then install the remaining notebook dependencies:

```powershell
python -m pip install -r requirements.txt
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
python -m ipykernel install --user --name transferguard --display-name "Python (TransferGuard)"
jupyter lab
```

The final check must print `True` for CUDA. Choose the new kernel in Jupyter or VS Code. The selected notebooks deliberately stop if CUDA is unavailable. No local virtual environment is committed.

## Run order and local files

1. Run Steps 1–11 of the 80% notebook to generate `transferguard_cifar10_results/source_noise80_last.pt` and its baseline histories. Optional controls are controlled by the flags in its first code cell. Run Step 12's official test evaluation only after the comparison is finalized.
2. Run the 40% notebook to generate `transferguard_noise40_results/source_noise40_last.pt` and its comparison histories. It can reuse a matched BigCNN scratch checkpoint from step 1.
3. Run the REFINE notebook after both source checkpoints exist. Its later cells read the earlier CSV histories and optionally run the feature probe and TransferGuard demonstration. The official test evaluation flag is off by default; leave it off while selecting methods.

Open the notebooks from the repository root or `notebooks/`; their setup cells resolve datasets and outputs to the repository root in either case. CIFAR-10 downloads into `data/` on first use. Each notebook writes checkpoints, epoch histories, splits, plots, and analysis files into its own ignored results directory. A fresh clone contains **no datasets or checkpoints**, so reproducing REFINE requires running the earlier source training first. Training is GPU intensive; no training is required to read the code or documented results.

## Additional analysis scripts

The `experiments/` directory contains two follow-up analyses. Run `python experiments/primary_verification.py` after the three notebooks to compare feature-only, pilot-only, and combined decisions on the **same seed-42 development conditions**. It reads their local CSVs and writes derived tables to ignored `results/`; these are not independent detector validation.

`python experiments/test_guard_protocol.py` checks the frozen decision rule without a GPU. `python experiments/independent_validation.py --dry-run` prints the planned new-seed conditions. The full validation runner needs CUDA and writes its own checkpoints and results under ignored `results/validation/`. Its GPU training path has not yet been smoke-tested, and no independent results are included in this repository.

## Current limits and contributions

The reported numbers are single runs on one CIFAR-10 split and seed. They are best validation accuracies, not estimates from repeated seeds or held-out detector tests. The demonstration's pilot signals use the validation histories, and its thresholds require independent evaluation before claims of generalization. The early PathMNIST and OrganAMNIST notebooks remain local exploratory work and are outside this primary workflow.

To contribute, see [CONTRIBUTING.md](CONTRIBUTING.md). Start with a branch, keep datasets and checkpoints local, and include the exact split, seed, method, and evidence for each new result in a pull request.
