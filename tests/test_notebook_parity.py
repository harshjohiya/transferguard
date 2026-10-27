"""Small notebook-to-legacy parity checks; no datasets or lengthy training."""
import ast
import csv
import json
import platform
import time
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from experiments.baselines.train import run_epoch, save_checkpoint, seed_everything
from experiments.negative_transfer.train_target import load_svhn_model
from src.datasets.cifar10 import balanced_split_indices, get_cifar10_eval_transform, get_cifar10_train_transform
from src.guard.decision import split_validation_indices
from src.models.mobilenet import get_model
from src.transferability.features import extract_features, load_source_model
from src.transferability.logme import logme_score


ROOT = Path(__file__).resolve().parents[1]


def notebook_cell(filename, prefix):
    notebook = json.loads((ROOT / filename).read_text(encoding="utf-8"))
    return next("".join(cell["source"]) for cell in notebook["cells"]
                if cell["cell_type"] == "code" and "".join(cell["source"]).startswith(prefix))


def notebook_function(filename, name, namespace):
    notebook = json.loads((ROOT / filename).read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        functions = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == name]
        if functions:
            module = ast.Module(body=functions, type_ignores=[])
            exec(compile(module, filename, "exec"), namespace)
            return namespace[name]
    raise AssertionError(f"{name} is missing from {filename}")


class NotebookParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_target_indices_match_p1_and_p5(self):
        class FakeCIFAR10:
            def __init__(self, root, train, download, transform):
                self.targets = [label for label in range(10) for _ in range(1000)]
                self.transform = transform

            def __len__(self):
                return len(self.targets)

            def __getitem__(self, index):
                return torch.zeros(3, 128, 128), self.targets[index]

        namespace = dict(CIFAR10=FakeCIFAR10, TRAIN_SIZE=2000, VALIDATION_SIZE=5000,
                         BATCH_SIZE=16, SEED=42, DataLoader=DataLoader, Subset=Subset,
                         balanced_split_indices=balanced_split_indices,
                         split_validation_indices=split_validation_indices,
                         get_cifar10_train_transform=get_cifar10_train_transform,
                         get_cifar10_eval_transform=get_cifar10_eval_transform)
        code = notebook_cell("01_negative_transfer.ipynb", "train_data = CIFAR10")
        exec(compile(code, "01_negative_transfer.ipynb", "exec"), namespace)
        expected_train, expected_val = balanced_split_indices(
            FakeCIFAR10(None, True, False, None).targets, 2000, 5000, 42)
        expected_select, expected_guard = split_validation_indices(
            expected_val, FakeCIFAR10(None, True, False, None).targets, 42)
        self.assertEqual(namespace["train_idx"], expected_train)
        self.assertEqual(namespace["selection_idx"], expected_select)
        self.assertEqual(namespace["guard_idx"], expected_guard)

    def test_one_epoch_checkpoint_and_metrics_match_p2(self):
        with tempfile.TemporaryDirectory() as directory:
            images = torch.randn(4, 3, 4, 4)
            labels = torch.tensor([0, 1, 0, 1])
            loader = DataLoader(TensorDataset(images, labels), batch_size=2)
            def small_model(source, checkpoint=None):
                return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 2))
            namespace = dict(SEED=42, EPOCHS=1, DEVICE=torch.device("cpu"),
                             ROOT=Path(directory), LR={"scratch": 1e-3},
                             TRAIN_SIZE=4, BATCH_SIZE=2, train_idx=[0, 1, 2, 3],
                             selection_idx=[4, 5, 6, 7], guard_idx=[8, 9],
                             train_loader=loader, selection_loader=loader,
                             seed_everything=seed_everything, make_target_model=small_model,
                             run_epoch=run_epoch, save_checkpoint=save_checkpoint,
                             torch=torch, nn=torch.nn, Path=Path, csv=csv, json=json,
                             time=time, platform=platform)
            train_target = notebook_function(
                "01_negative_transfer.ipynb", "train_target", namespace)
            run_dir = train_target("scratch")
            checkpoint = torch.load(run_dir/"best_model.pth", weights_only=True)
            with (run_dir/"metrics.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))

            seed_everything(42)
            model = small_model("scratch")
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            criterion = torch.nn.CrossEntropyLoss()
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            model.train()
            train_loss, train_acc = run_epoch(
                model, loader, criterion, torch.device("cpu"), optimizer, scaler)
            model.eval()
            val_loss, val_acc = run_epoch(model, loader, criterion, torch.device("cpu"))
            self.assertAlmostEqual(float(row["train_loss"]), train_loss)
            self.assertAlmostEqual(float(row["train_accuracy"]), train_acc)
            self.assertAlmostEqual(float(row["selection_loss"]), val_loss)
            self.assertAlmostEqual(float(row["selection_accuracy"]), val_acc)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(checkpoint["model_state_dict"][key], value)

    def test_svhn_checkpoint_initialization_matches_p3(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best_model.pth"
            seed_everything(7)
            source = get_model("scratch")
            save_checkpoint(path, source, 1, 1.0, 0.5, "source_svhn")
            path.with_name("config.json").write_text(json.dumps({
                "mode": "source_svhn", "architecture": "MobileNetV3-Small",
                "num_classes": 10, "dataset": "SVHN official training partition"
            }))
            namespace = dict(SEED=42, seed_everything=seed_everything,
                             get_model=get_model, load_svhn_model=load_svhn_model,
                             torch=torch, nn=torch.nn)
            make_target_model = notebook_function(
                "01_negative_transfer.ipynb", "make_target_model", namespace)
            actual = make_target_model("svhn", path)
            seed_everything(42)
            expected, _, _ = load_svhn_model(path)
            for key, value in expected.state_dict().items():
                torch.testing.assert_close(actual.state_dict()[key], value)

    def test_random_frozen_features_and_logme_match_p4(self):
        images = torch.randn(4, 3, 64, 64)
        labels = torch.tensor([0, 1, 0, 1])
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        namespace = dict(SEED=42, DEVICE=torch.device("cpu"), feature_loader=loader,
                         seed_everything=seed_everything, load_source_model=load_source_model,
                         extract_features=extract_features)
        source_features = notebook_function(
            "02_transferguard.ipynb", "source_features", namespace)
        actual, actual_labels, _ = source_features("random")
        seed_everything(42)
        model, _ = load_source_model("random")
        expected, expected_labels = extract_features(
            model, loader, torch.device("cpu"), ("final",))
        np.testing.assert_array_equal(actual, expected["final"])
        np.testing.assert_array_equal(actual_labels, expected_labels)
        self.assertEqual(logme_score(actual, actual_labels),
                         logme_score(expected["final"], expected_labels))


if __name__ == "__main__":
    unittest.main()
