# TransferGuard

## Notebook-first research workflow

Start with [01_negative_transfer.ipynb](01_negative_transfer.ipynb) for the controlled scratch, ImageNet, SVHN, and optional PathMNIST comparison. Continue with [02_transferguard.ipynb](02_transferguard.ipynb) for frozen features, the verified LogME score, transfer-advantage curves, three adaptation policies, and the validation-based scratch fallback. Open the notebooks from the repository root so the existing Python modules are importable. Training, source pretraining, scoring, and final test evaluation are opt-in cells.

The research split is 2,000 balanced CIFAR-10 training images plus 2,500 selection and 2,500 guard images from the official training partition. The official test partition is opened only after the rule and policy are fixed. New notebook outputs live in `runs/notebooks/`; previous P1–P5 outputs retain their original paths. Set the seed, checkpoint paths, epoch budget, and learning rates in the configuration cells. The notebooks write configuration, metrics, and checkpoints for each new run.

Before interpreting a transfer gap, check scratch convergence, repeat prespecified seeds and learning rates, and report both source and target training cost. A falling transfer advantage is an exploratory warning, not a tuned decision rule. The guard selects transfer only when its paired bootstrap lower bound over scratch is positive. Do not revise that rule using final test results.

### Migration and verification

The notebook orchestration is new; the verified P1–P5 modules remain dependencies during parity verification. Notebook 01 reuses P1 balanced splits, P2 epoch/checkpoint code, and the P3 SVHN loader. Notebook 02 reuses P4 feature extraction and LogME plus P5 policy training and paired guard. No old module or historical result has been removed.

Run the small existing checks before new experiments:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -v
```

For checkpoint parity, use the same saved P3 source checkpoint in both paths and compare `load_svhn_model` state dictionaries; compare LogME on the same feature matrix and labels with `src.transferability.logme.logme_score`; compare notebook split index lists with P1/P5 split functions. Keep old modules until these checks and a notebook smoke execution pass. Full training is never started by opening a notebook.

## Legacy P1–P5 commands and documentation

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

## Phase 4: transferability estimation

Phase 4 computes class-wise LogME scores from frozen ImageNet, SVHN-source, or random MobileNetV3-Small features on the same 2,000 CIFAR-10 training images. It uses the deterministic validation preprocessing and never fits scores on validation or test images. Run `python -m experiments.transferability.score --help` for options. For all three feature stages:

```powershell
.\.venv\Scripts\python.exe -m experiments.transferability.score --source imagenet --layers all --device cuda
.\.venv\Scripts\python.exe -m experiments.transferability.score --source svhn --source-checkpoint runs\source_svhn\RUN_ID\best_model.pth --layers all --device cuda
.\.venv\Scripts\python.exe -m experiments.transferability.score --source random --layers all --device cuda
.\.venv\Scripts\python.exe -m experiments.transferability.compare --imagenet-run runs\logme\imagenet\RUN_ID --svhn-run runs\logme\svhn\RUN_ID --random-run runs\logme\random\RUN_ID
```

Results are saved in unique `runs/logme/<source>/` directories. See [Phase 4 methodology and interpretation](docs/p4_transferability.md) for the evidence formula, feature dimensions, comparability checks, and limitations. LogME is an estimate of frozen-feature suitability, not demonstrated negative transfer.

## Phase 5: selective transfer and validation guard

TransferGuard ranks saved, compatible final-stage LogME scores, then trains a scratch reference and three adaptations of the chosen source: frozen features, late feature blocks, and all parameters. All four models use the same 2,000 CIFAR-10 training images and epoch budget. A balanced 2,500-image validation partition selects checkpoints and the adaptation policy. A disjoint balanced 2,500-image guard partition compares that policy with scratch using a paired bootstrap. Transfer is selected only if the lower 95% confidence bound for its accuracy improvement is above zero. No official CIFAR-10 test images are loaded by this experiment.

The command only runs when invoked; it can require substantial GPU time. From the repository root:

```powershell
.\.venv\Scripts\python.exe -m experiments.transferguard.run --source auto --epochs 30 --batch-size 16 --seed 42 --device cuda
.\.venv\Scripts\python.exe -m experiments.transferguard.run --source svhn --source-checkpoint runs\source_svhn\20261023-113328-c9cdc993\best_model.pth --epochs 30 --batch-size 16 --seed 42 --device cuda
```

By default, the script uses the newest compatible saved LogME run per available source. Pass `--imagenet-logme-run` and/or `--svhn-logme-run` to pin runs. `--source imagenet` or `--source svhn` overrides LogME ranking for a controlled experiment, while still requiring a compatible score. The SVHN checkpoint must match the checkpoint hashed by its LogME run. Learning rates can be set with `--scratch-lr`, `--frozen-lr`, `--late-lr`, and `--full-lr`; `--bootstrap-samples` and `--device cpu` are also available. Each invocation writes only to a unique `runs/transferguard/` folder, with candidate checkpoints, epoch metrics, configuration, and a decision summary. See [Phase 5 method and limitations](docs/p5_transferguard.md).
