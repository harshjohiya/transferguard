# TransferGuard

TransferGuard studies how to detect and avoid negative transfer in lightweight vision models. Phase 1 prepares a reproducible CIFAR-10 pipeline; it does not train models.

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
