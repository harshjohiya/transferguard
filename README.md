# TransferGuard

TransferGuard studies how to detect and avoid negative transfer in lightweight vision models. Phase 1 prepares a reproducible CIFAR-10 pipeline. Phase 2 trains preliminary MobileNetV3-Small baselines.

## Environment

Use Python 3.11 or 3.12 on Windows. From the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')"
```

`requirements.txt` uses matching PyTorch and TorchVision CUDA 12.6 wheels from the official PyTorch index. The RTX 2050 has 4 GB VRAM, so the default batch size is 16.

## Dataset

The first run downloads CIFAR-10 to `data/raw/`. Training uses random crop and horizontal flip. All images are resized to 128 x 128, converted to tensors, and normalized with ImageNet mean and standard deviation. With seed 42, the official training partition supplies 2,000 class-balanced training images and 5,000 class-balanced validation images without shared indices. The official 10,000-image test partition stays separate.

```powershell
.\.venv\Scripts\python.exe -m src.datasets.check_cifar10
.\.venv\Scripts\python.exe -m src.datasets.check_cifar10 --train-size 2000 --val-size 5000 --batch-size 16 --seed 42
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The check prints dataset sizes, class counts, image batch shapes, split overlap, and CUDA status. It saves a sample grid to `results/figures/cifar10_samples.png`. Split sizes must be positive multiples of 10 and fit within the official training partition. The Python API is `src.datasets.get_cifar10_datasets()` or `src.datasets.get_cifar10_loaders()`.

## Baseline training

The `scratch` model starts with random weights and trains all parameters. The `frozen` model loads ImageNet weights, trains only its classifier, and keeps feature BatchNorm layers in evaluation mode. The `finetune` model loads the same ImageNet weights and trains all parameters at a smaller learning rate. All modes predict the 10 CIFAR-10 classes. ImageNet weights are downloaded on first use of `frozen` or `finetune`.

Run each mode from the repository root:

```powershell
.\.venv\Scripts\python.exe -m experiments.baselines.train --mode scratch
.\.venv\Scripts\python.exe -m experiments.baselines.train --mode frozen
.\.venv\Scripts\python.exe -m experiments.baselines.train --mode finetune
.\.venv\Scripts\python.exe -m experiments.baselines.compare
```

Defaults are 5 epochs, 2,000 training images, 5,000 validation images, batch size 16, seed 42, and CUDA. Default learning rates are 0.001 for `scratch` and `frozen`, and 0.0001 for `finetune`. Override them with `--epochs`, `--batch-size`, `--train-size`, `--val-size`, `--seed`, `--device`, or `--lr`. For example, use `--device cpu` if CUDA is unavailable. With 4 GB GPU memory, reduce `--batch-size` if CUDA reports out of memory. Mixed precision is enabled on CUDA.

Each invocation creates a unique folder under `runs/<mode>/` with `best_model.pth`, `metrics.csv`, `config.json`, and `summary.json`. The checkpoint is selected by validation accuracy; the official CIFAR-10 test partition is never used for training or model selection. The comparison command uses the latest completed run for each mode and reports incompatible data splits or evaluation settings. These five-epoch comparisons are preliminary validation results.

## Phase 3: cross-domain transfer

Phase 3 tests whether pretraining on SVHN digits helps or harms CIFAR-10 classification compared with scratch and ImageNet initialization. Install SciPy for TorchVision's SVHN `.mat` loader after installing the CUDA-enabled PyTorch requirements:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-svhn.txt
```

This separate requirements file only installs SciPy and leaves the existing CUDA PyTorch installation untouched. The SVHN loader downloads the official *training* partition to `data/raw/`, then uses reproducible, class-balanced 10,000/2,000 subsets. It resizes RGB digit images to 128 x 128 and applies ImageNet normalization without horizontal flips. The official SVHN test partition is not loaded for source model selection.

Run source pretraining, then pass its printed checkpoint path to target training:

```powershell
.\.venv\Scripts\python.exe -m experiments.negative_transfer.pretrain_svhn --epochs 12 --train-size 10000 --val-size 2000 --batch-size 16 --seed 42 --device cuda
.\.venv\Scripts\python.exe -m experiments.negative_transfer.train_target --source-checkpoint runs\source_svhn\RUN_ID\best_model.pth --epochs 30 --batch-size 16 --train-size 2000 --val-size 5000 --seed 42 --device cuda
```

For a comparison with the same target training budget, train the scratch and ImageNet baselines for 30 epochs too:

```powershell
.\.venv\Scripts\python.exe -m experiments.baselines.train --mode scratch --epochs 30 --batch-size 16 --train-size 2000 --val-size 5000 --seed 42 --device cuda
.\.venv\Scripts\python.exe -m experiments.baselines.train --mode finetune --epochs 30 --batch-size 16 --train-size 2000 --val-size 5000 --seed 42 --device cuda
.\.venv\Scripts\python.exe -m experiments.negative_transfer.compare --scratch-run runs\scratch\RUN_ID --imagenet-run runs\finetune\RUN_ID --svhn-run runs\svhn_finetune\RUN_ID
```

Replace each `RUN_ID` with its own run directory name. The comparison command can omit run arguments when exactly one completed run exists for each mode. It reports ambiguity or incompatible target splits, seeds, evaluation settings, and epoch budgets rather than choosing silently. Source results are saved under `runs/source_svhn/`, and target results under `runs/svhn_finetune/`; each run contains a checkpoint, metrics CSV, configuration JSON, and summary JSON. The target configuration also records the source checkpoint path and source configuration.

Transfer gap is transferred validation accuracy minus scratch validation accuracy, in percentage points. A negative gap is a candidate negative-transfer observation; it does not prove negative transfer. Longer runs and repeated seeds are needed before drawing a conclusion. The official CIFAR-10 test partition is never used during target training or model selection.
