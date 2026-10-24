# Phase 4: LogME transferability estimation

Phase 4 scores frozen MobileNetV3-Small representations before CIFAR-10 fine-tuning. It uses the classification LogME method of [You et al., ICML 2021](https://proceedings.mlr.press/v139/you21b.html), with the later fixed-point computational formulation in the [authors' implementation](https://github.com/thuml/LogME/blob/main/LogME.py). The repository implementation uses direct float64 SVD rather than the official code's optional Gram-matrix SVD shortcut. It does not use the distinct `_fit_icml` update routine.

## Calculation

For each target class, labels are binary one-vs-rest values `y`. For frozen feature matrix `X` with `N` examples and `D` dimensions, LogME assumes `y = Xw + noise`, with spherical Gaussian weight precision `alpha` and noise precision `beta`. Let `A = alpha I + beta XᵀX` and `m = beta A⁻¹Xᵀy`. The per-example log evidence is

```text
[D log(alpha) + N log(beta) - log|A|
 - beta ||y-Xm||² - alpha ||m||² - N log(2π)] / (2N).
```

The fixed-point iterations update `gamma = sum_i beta*s_i²/(alpha+beta*s_i²)`, `alpha = gamma/||m||²`, and `beta = (N-gamma)/||y-Xm||²`, where `s_i` are singular values of `X`. The implementation computes these terms from a direct SVD, handles null singular directions in the determinant, and bounds positive precisions to avoid infinite values for degenerate data. It stops at a relative log-precision change below `1e-7` or 128 iterations. The reported score is the mean evidence across classes. Higher values suggest more suitable frozen features under this model. Scores use the same algorithm and settings for every initialization and stage.

## Features and samples

The installed TorchVision MobileNetV3-Small has `features[0]` as the stem, `features[1:12]` as inverted residual blocks, and `features[12]` as the final convolution. The predefined stages are:

| Stage | Last feature index | Pooled channels |
| --- | ---: | ---: |
| early | 1 | 16 |
| middle | 8 | 48 |
| final | 12 | 576 |

Each spatial map uses adaptive global average pooling to one value per channel. The final stage is the default. The classifier is never used. ImageNet uses TorchVision's source weights, SVHN uses a validated Phase 3 `source_svhn` checkpoint, and random uses a fresh untrained model. All parameters are frozen and models stay in evaluation mode.

The target data is the official CIFAR-10 **training** partition. Phase 1's `balanced_split_indices` selects the same class-balanced 2,000 training indices with seed 42 and a disjoint 5,000-index validation reservation. Phase 4 loads only the training partition and extracts only those 2,000 selected images. It applies the existing validation transform: resize to 128×128, tensor conversion, and ImageNet normalization. Random crops or flips would change the feature matrix between runs and undermine source comparisons. Index and ordered-label SHA-256 hashes are saved in every run. Validation and test examples never fit LogME.

## Run

From the repository root, with dependencies installed:

```powershell
.\.venv\Scripts\python.exe -m experiments.transferability.score --source imagenet --layers all --device cuda
.\.venv\Scripts\python.exe -m experiments.transferability.score --source svhn --source-checkpoint runs\source_svhn\RUN_ID\best_model.pth --layers all --device cuda
.\.venv\Scripts\python.exe -m experiments.transferability.score --source random --layers all --device cuda
.\.venv\Scripts\python.exe -m experiments.transferability.compare --imagenet-run runs\logme\imagenet\RUN_ID --svhn-run runs\logme\svhn\RUN_ID --random-run runs\logme\random\RUN_ID --phase3-observed
```

Replace each `RUN_ID` with the respective directory. Omit `--layers all` to score only the final stage. The scoring command also supports `--train-size`, `--val-size`, `--batch-size`, `--seed`, and `--device cpu`. Keep train size, validation size, seed, and layer choice identical across sources. Each invocation writes a unique directory under `runs/logme/<source>/` containing `config.json`, `scores.csv`, and `summary.json`. These record feature dimensions, counts, timings, data hashes, and SVHN checkpoint identity. Comparison requires explicit run directories and rejects mismatched data, preprocessing, labels, or stages.

LogME estimates compatibility of **frozen** features with target labels. It cannot guarantee end-to-end fine-tuning accuracy: optimization, training budget, regularization, and representation changes also matter. The Phase 3 validation accuracies (scratch 57.14%, ImageNet 85.60%, SVHN 55.50%) are single-seed observations. Comparing these with LogME is exploratory; two pretrained sources do not support a reliable correlation estimate or a claim that negative transfer has been detected. Phase 4 does not implement a transfer decision rule.
