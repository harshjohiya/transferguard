"""Workstream B: independent validation of the TransferGuard detector.

For every (seed, source-noise) condition this script
  1. builds that seed's own CIFAR-10 split (source / target / validation / calibration),
  2. pretrains a NEW BigCNN source on labels corrupted at the given rate
     (noise 0 = clean-source control),
  3. trains the scratch SmallCNN and the frozen-transfer BigCNN on the target set,
  4. runs the feature probe and the guard rule from guard_protocol.py,
  5. records choices, the truly better method, correctness and regret.

The decision thresholds live in guard_protocol.py and are NOT arguments here.
The official CIFAR-10 test set is never loaded.

Setup mirrors the three notebooks (same models, optimiser, epochs, batch size,
split sizes, dynamic label noise, final-layer-reset frozen transfer).

Example:
    python experiments/independent_validation.py --seeds 43 44 --noises 0 40 80
Resume: re-run the same command; finished conditions are skipped and the
source / target runs continue from their last saved epoch.
"""
import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

import guard_protocol as gp

ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------------
# Models (copied unchanged from the notebooks)
# ----------------------------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x).flatten(1))


class BigCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 80, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(80, 160, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(160, 320, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(320, 640, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(640, 640, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(640, 768, 3, padding=1), nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(768, 2560), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(2560, 10),
        )

    def get_features(self, x):
        """2,560-dim source representation (same as notebook 3)."""
        x = self.conv_layers(x).flatten(1)
        return self.fc_layers[1](self.fc_layers[0](x))

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x).flatten(1))


class NoisyLabels(Dataset):
    """With probability `rate`, return a random WRONG label, resampled on every access."""

    def __init__(self, clean_dataset, rate):
        self.data = clean_dataset
        self.rate = rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        if np.random.rand() < self.rate:
            wrong = np.random.randint(0, 9)
            if wrong >= label:
                wrong += 1
            label = int(wrong)
        return image, label


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def set_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_splits(seed):
    """Same slicing as the notebooks, but with this seed's permutation."""
    idx = np.random.RandomState(seed).permutation(50000)
    splits = dict(source=idx[:10000], target=idx[10000:14000],
                  validation=idx[14000:16000], calibration=idx[16000:16500])
    assert len(set(np.concatenate(list(splits.values())))) == 16500
    return splits


def make_loader(data, batch_size, shuffle, pin):
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=pin)


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += len(y)
    return 100 * correct / total


@torch.no_grad()
def collect_features(model, loader, device):
    model.eval()
    fs, ys = [], []
    for x, y in loader:
        fs.append(model.get_features(x.to(device)).cpu())
        ys.append(y)
    return torch.cat(fs), torch.cat(ys)


def train(model, loader, val_loader, name, epochs, run_dir, device, lr):
    """SGD(lr, momentum 0.9), one validation pass per epoch, resumable.

    Returns the history as [[epoch, train_loss, val_acc, seconds], ...].
    Note: a resumed run does not restore RNG state, so it is statistically
    equivalent to, not bit-identical with, an uninterrupted run.
    """
    model = model.to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    last_path = run_dir / f"{name}_last.pt"
    history, start = [], 0
    if last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        if saved["epochs"] != epochs:
            raise ValueError(f"{last_path}: epoch setting changed; use a new --out folder")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, start = saved["history"], saved["epoch"]
        print(f"  resuming {name} from epoch {start}")

    for epoch in range(start, epochs):
        t0 = time.time()
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        val_acc = accuracy(model, val_loader, device)
        history.append([epoch + 1, total / len(loader.dataset), val_acc, time.time() - t0])
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1, "epochs": epochs, "history": history}, last_path)
        print(f"  {name}: {epoch + 1:02}/{epochs} | loss {history[-1][1]:.3f} | val {val_acc:.2f}%")

    with open(run_dir / f"{name}_history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_acc", "seconds"])
        w.writerows(history)
    return history


# ----------------------------------------------------------------------------
# One condition
# ----------------------------------------------------------------------------
def run_condition(seed, noise, args, device, train_data, protocol_sha):
    run_dir = args.out / f"seed{seed}_noise{noise}"
    result_path = run_dir / "condition_result.json"
    if result_path.exists():
        print(f"seed {seed} noise {noise}%: already finished, skipping")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== seed {seed} | source noise {noise}% ===")

    pin = device.type == "cuda"
    bs, lr = args.batch_size, args.lr
    splits = make_splits(seed)
    np.savez(run_dir / "split_indices.npz", **splits)
    source_noisy = NoisyLabels(Subset(train_data, splits["source"]), noise / 100)
    target = Subset(train_data, splits["target"])
    val = Subset(train_data, splits["validation"])
    cal = Subset(train_data, splits["calibration"])
    source_loader = make_loader(source_noisy, bs, True, pin)
    target_loader = make_loader(target, bs, True, pin)
    val_loader = make_loader(val, bs, False, pin)
    cal_loader = make_loader(cal, bs, False, pin)

    # 1. Source pretraining on corrupted labels (the final epoch is what we reuse).
    set_seed(seed)
    source = BigCNN()
    train(source, source_loader, val_loader, "source", args.source_epochs, run_dir, device, lr)
    source_val = accuracy(source, val_loader, device)

    # 2. Cheap feature probe + health check (decision-time information).
    t0 = time.time()
    x_tr, y_tr = collect_features(source, make_loader(Subset(target, range(500)), bs, False, pin), device)
    x_cal, y_cal = collect_features(source, cal_loader, device)
    centroids = torch.stack([x_tr[y_tr == c].mean(0) for c in range(10)])
    centroid_acc = 100 * (torch.cdist(x_cal, centroids).argmin(1) == y_cal).float().mean().item()
    feature_std = x_cal.std(0, unbiased=False).mean().item()
    zero_pct = 100 * (x_cal == 0).float().mean().item()
    probe_seconds = time.time() - t0
    print(f"  probe: centroid {centroid_acc:.1f}% | feature std {feature_std:.3g} | zeros {zero_pct:.1f}%")
    del source

    # 3. Scratch SmallCNN baseline.
    set_seed(seed)
    scratch_hist = train(SmallCNN(), target_loader, val_loader, "scratch", args.target_epochs,
                         run_dir, device, lr)

    # 4. Frozen transfer: freeze all but a RESET final layer (as in the 40% notebook).
    set_seed(seed)
    frozen = BigCNN()
    state = torch.load(run_dir / "source_last.pt", map_location="cpu", weights_only=True)["model"]
    frozen.load_state_dict(state)
    for p in frozen.parameters():
        p.requires_grad = False
    frozen.fc_layers[-1].reset_parameters()
    for p in frozen.fc_layers[-1].parameters():
        p.requires_grad = True
    frozen_hist = train(frozen, target_loader, val_loader, "frozen", args.target_epochs,
                        run_dir, device, lr)

    # 5. Score every selector. Decisions use only probe + first pilot epochs.
    row = gp.evaluate_condition(seed, noise, centroid_acc, feature_std,
                                [h[2] for h in scratch_hist], [h[2] for h in frozen_hist])
    row.update({
        "zero_feature_pct": zero_pct,
        "source_clean_val_acc": source_val,
        "probe_seconds": probe_seconds,
        "pilot_seconds": sum(h[3] for h in scratch_hist[:gp.PILOT_EPOCHS])
                         + sum(h[3] for h in frozen_hist[:gp.PILOT_EPOCHS]),
        "source_epochs": args.source_epochs,
        "target_epochs": args.target_epochs,
        "protocol_sha256": protocol_sha,
        "source_checkpoint_sha256": sha256_file(run_dir / "source_last.pt"),
    })
    result_path.write_text(json.dumps(row, indent=2))
    print(f"  guard: {row['guard_choice']} | better: {row['better_method']} | "
          f"regret {row['guard_regret']:.2f} pts (pilot-only: {row['pilot_only_choice']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[43, 44])
    ap.add_argument("--noises", type=int, nargs="+", default=[0, 40, 80],
                    help="source-label corruption in percent; 0 = clean-source control")
    ap.add_argument("--source-epochs", type=int, default=60)
    ap.add_argument("--target-epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "validation")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--expect-protocol-sha", default=None,
                    help="abort unless guard_protocol.py has this SHA-256 (use to lock the thresholds)")
    ap.add_argument("--allow-cpu", action="store_true", help="smoke tests only; far too slow for real runs")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    if args.target_epochs < gp.PILOT_EPOCHS:
        ap.error(f"--target-epochs must be >= {gp.PILOT_EPOCHS} (the pilot length)")
    protocol_sha = sha256_file(Path(gp.__file__))
    if args.expect_protocol_sha and args.expect_protocol_sha != protocol_sha:
        raise SystemExit(f"guard_protocol.py changed!\n  expected {args.expect_protocol_sha}\n  actual   {protocol_sha}")

    conditions = [(s, n) for s in args.seeds for n in args.noises]
    print(f"protocol {gp.PROTOCOL_VERSION} sha256 {protocol_sha}")
    print(f"thresholds: centroid >= {gp.CENTROID_MIN_PCT}%, frozen pilot >= scratch pilot + "
          f"{gp.PILOT_MARGIN_POINTS} pts, pilot = first {gp.PILOT_EPOCHS} epochs")
    for s, n in conditions:
        tag = "DEVELOPMENT (used to design thresholds)" if (s, n) in gp.DEVELOPMENT_CONDITIONS else "independent"
        print(f"  seed {s} noise {n}%: {tag}")
    if args.dry_run:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise SystemExit("CUDA not available. Use a GPU machine (or --allow-cpu for a tiny smoke test).")

    args.out.mkdir(parents=True, exist_ok=True)
    config = {"source_epochs": args.source_epochs, "target_epochs": args.target_epochs,
              "batch_size": args.batch_size, "lr": args.lr, "protocol_sha256": protocol_sha}
    config_path = args.out / "run_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text())
        if old != config:
            raise SystemExit(f"{config_path} differs from this run's settings; use a new --out folder.\n"
                             f"  old: {old}\n  new: {config}")
    else:
        config_path.write_text(json.dumps(config, indent=2))

    args.data_dir.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = datasets.CIFAR10(root=str(args.data_dir), train=True, download=True, transform=transform)

    for seed, noise in conditions:
        run_condition(seed, noise, args, device, train_data, protocol_sha)

    print("\n=== summary ===")
    gp.summarize_directory(args.out)


if __name__ == "__main__":
    main()
