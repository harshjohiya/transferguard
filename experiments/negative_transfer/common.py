"""Training and output files shared by the Phase 3 experiments."""

import csv
import json
import time

import torch
from torch import nn

from experiments.baselines.train import make_run_dir, run_epoch, save_checkpoint
from src.models.mobilenet import set_train_mode


def train_and_save(model, mode, train_loader, val_loader, config, device):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    run_dir = make_run_dir(mode)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Run directory: {run_dir}")

    best_accuracy, best_loss, best_epoch = -1.0, None, None
    start = time.perf_counter()
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=(
            "epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "elapsed_seconds"
        ))
        writer.writeheader()
        for epoch in range(1, config["epochs"] + 1):
            set_train_mode(model, mode)
            train_loss, train_accuracy = run_epoch(
                model, train_loader, criterion, device, optimizer, scaler
            )
            model.eval()
            val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
            writer.writerow({
                "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
                "val_loss": val_loss, "val_accuracy": val_accuracy,
                "elapsed_seconds": time.perf_counter() - start,
            })
            file.flush()
            if val_accuracy > best_accuracy:
                best_accuracy, best_loss, best_epoch = val_accuracy, val_loss, epoch
                save_checkpoint(run_dir / "best_model.pth", model, epoch, val_loss, val_accuracy, mode)
            print(f"Epoch {epoch}/{config['epochs']}: train acc {train_accuracy:.4f}, "
                  f"val acc {val_accuracy:.4f}, val loss {val_loss:.4f}")

    summary = {
        "mode": mode, "best_val_accuracy": best_accuracy, "best_val_loss": best_loss,
        "best_epoch": best_epoch, "training_seconds": time.perf_counter() - start,
        "epochs": config["epochs"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Best validation accuracy: {best_accuracy:.4f} (epoch {best_epoch})")
    return run_dir
