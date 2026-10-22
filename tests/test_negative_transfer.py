import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from experiments.baselines.train import run_epoch, save_checkpoint
from experiments.negative_transfer import compare
from experiments.negative_transfer.common import train_and_save
from experiments.negative_transfer.train_target import load_svhn_model
from src.datasets.svhn import get_svhn_datasets, get_svhn_loaders
from src.models.mobilenet import get_model


class FakeSVHN:
    calls = []

    def __init__(self, root, split, download, transform):
        self.calls.append(split)
        self.labels = [label for label in range(10) for _ in range(200)]
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(Image.new("RGB", (32, 32), (128, 64, 32))), self.labels[index]

    def __len__(self):
        return len(self.labels)


@patch("src.datasets.svhn.SVHN", FakeSVHN)
class SVHNDatasetTests(unittest.TestCase):
    def test_balanced_disjoint_reproducible_and_preprocessed(self):
        FakeSVHN.calls.clear()
        train, val = get_svhn_datasets("unused", train_size=200, val_size=100, seed=42)
        other_train, other_val = get_svhn_datasets("unused", train_size=200, val_size=100, seed=42)
        self.assertEqual(train.indices, other_train.indices)
        self.assertEqual(val.indices, other_val.indices)
        self.assertFalse(set(train.indices) & set(val.indices))
        self.assertEqual(set(Counter(train.dataset.labels[i] for i in train.indices).values()), {20})
        self.assertEqual(set(Counter(val.dataset.labels[i] for i in val.indices).values()), {10})
        self.assertEqual(set(FakeSVHN.calls), {"train"})
        image, label = train[0]
        self.assertEqual(tuple(image.shape), (3, 128, 128))
        self.assertIn(label, range(10))
        self.assertFalse(any(isinstance(step, transforms.RandomHorizontalFlip)
                             for step in train.dataset.transform.transforms))
        train_loader, val_loader = get_svhn_loaders("unused", 200, 100, batch_size=16, seed=42)
        self.assertEqual((train_loader.num_workers, val_loader.num_workers), (0, 0))
        self.assertEqual(tuple(next(iter(train_loader))[0].shape), (16, 3, 128, 128))


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make_source(self, directory):
        source = get_model("scratch")
        with torch.no_grad():
            source.features[0][0].weight.fill_(0.123)
            source.classifier[-1].weight.fill_(1.0)
        path = Path(directory) / "best_model.pth"
        save_checkpoint(path, source, 1, 1.0, 0.5, "source_svhn")
        config = {
            "mode": "source_svhn", "architecture": "MobileNetV3-Small",
            "num_classes": 10, "dataset": "SVHN official training partition",
        }
        path.with_name("config.json").write_text(json.dumps(config), encoding="utf-8")
        return path, source

    def test_source_load_reinitialization_and_synthetic_step(self):
        with tempfile.TemporaryDirectory() as directory:
            path, source = self.make_source(directory)
            model, resolved, config = load_svhn_model(path)
            self.assertEqual(resolved, path.resolve())
            self.assertEqual(config["mode"], "source_svhn")
            torch.testing.assert_close(model.features[0][0].weight, source.features[0][0].weight)
            self.assertFalse(torch.equal(model.classifier[-1].weight, source.classifier[-1].weight))
            self.assertEqual(tuple(model(torch.randn(2, 3, 64, 64)).shape), (2, 10))
            self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
            model.train()
            loader = DataLoader(TensorDataset(torch.randn(2, 3, 64, 64), torch.tensor([0, 1])), batch_size=2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            loss, _ = run_epoch(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"), optimizer, scaler)
            self.assertGreater(loss, 0)
            self.assertIsNotNone(model.features[0][0].weight.grad)

    def test_rejects_incompatible_source_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.make_source(directory)
            checkpoint = torch.load(path, weights_only=True)
            checkpoint["model_state_dict"].pop("classifier.3.weight")
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "do not match"):
                load_svhn_model(path)

    def test_training_writes_checkpoint_metrics_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "source_run"
            run_dir.mkdir()
            model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 10))
            loader = DataLoader(
                TensorDataset(torch.randn(4, 3, 4, 4), torch.tensor([0, 1, 2, 3])),
                batch_size=2,
            )
            with patch("experiments.negative_transfer.common.make_run_dir", return_value=run_dir):
                train_and_save(model, "source_svhn", loader, loader,
                               {"epochs": 1, "lr": 0.001}, torch.device("cpu"))
            for filename in ("best_model.pth", "metrics.csv", "config.json", "summary.json"):
                self.assertTrue((run_dir / filename).is_file())
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["best_epoch"], 1)


class ComparisonTests(unittest.TestCase):
    def make_run(self, directory, mode, seed=42, epochs=30):
        path = Path(directory) / mode / "run"
        path.mkdir(parents=True)
        config = {field: "same" for field in compare.MATCH_FIELDS}
        config.update(mode=mode, seed=seed, epochs=epochs)
        summary = {
            "mode": mode, "best_val_accuracy": 0.5,
            "best_val_loss": 1.0, "training_seconds": 10.0,
        }
        (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return compare.load_run(path, mode)

    def test_rejects_incompatible_seed_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            scratch = self.make_run(directory, "scratch")
            imagenet = self.make_run(directory, "finetune", seed=43)
            svhn = self.make_run(directory, "svhn_finetune", epochs=29)
            with self.assertRaisesRegex(ValueError, "seed"):
                compare.compare_runs(scratch, imagenet, svhn)
            imagenet[1]["seed"] = 42
            with self.assertRaisesRegex(ValueError, "epochs"):
                compare.compare_runs(scratch, imagenet, svhn)

    def test_reports_ambiguous_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_run(directory, "scratch")
            second = Path(directory) / "scratch" / "another"
            second.mkdir()
            (second / "summary.json").write_text("{}", encoding="utf-8")
            with patch.object(compare, "RUNS_DIR", Path(directory)):
                with self.assertRaisesRegex(ValueError, "Multiple scratch runs"):
                    compare.choose_run("scratch", None)


if __name__ == "__main__":
    unittest.main()
