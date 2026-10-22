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
