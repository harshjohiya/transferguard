import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import mobilenet_v3_small

from experiments.baselines import compare
from experiments.baselines.train import run_epoch, save_checkpoint
from src.datasets.cifar10 import get_cifar10_datasets
from src.models.mobilenet import get_model, set_train_mode


class FakeCIFAR10:
    def __init__(self, root, train, download, transform):
        self.targets = [label for label in range(10) for _ in range(1000 if train else 20)]


def offline_mobilenet(weights):
    return mobilenet_v3_small(weights=None)


@patch("src.models.mobilenet.mobilenet_v3_small", side_effect=offline_mobilenet)
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_output_shape_and_trainable_parameters(self, factory):
        for mode in ("scratch", "frozen", "finetune"):
            with self.subTest(mode=mode):
                model = get_model(mode)
                self.assertEqual(tuple(model(torch.randn(2, 3, 64, 64)).shape), (2, 10))
                self.assertTrue(all(parameter.requires_grad for parameter in model.classifier.parameters()))
                self.assertEqual(
                    all(parameter.requires_grad for parameter in model.features.parameters()),
                    mode != "frozen",
                )
                del model
        self.assertIsNone(factory.call_args_list[0].kwargs["weights"])
        self.assertIsNotNone(factory.call_args_list[1].kwargs["weights"])
        self.assertEqual(factory.call_args_list[1].kwargs["weights"], factory.call_args_list[2].kwargs["weights"])

    def test_frozen_batchnorm_stays_in_eval_during_training(self, factory):
        model = get_model("frozen")
        set_train_mode(model, "frozen")
        self.assertTrue(model.classifier.training)
        batchnorms = [module for module in model.features.modules() if isinstance(module, nn.BatchNorm2d)]
        self.assertTrue(batchnorms)
        self.assertTrue(all(not module.training for module in batchnorms))
        loader = DataLoader(TensorDataset(torch.randn(2, 3, 64, 64), torch.tensor([0, 1])), batch_size=2)
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.001)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        before = model.classifier[-1].weight.detach().clone()
        loss, accuracy = run_epoch(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"), optimizer, scaler)
        self.assertGreater(loss, 0)
        self.assertGreaterEqual(accuracy, 0)
        self.assertFalse(torch.equal(before, model.classifier[-1].weight))
        self.assertTrue(all(not module.training for module in batchnorms))

    def test_synthetic_backward_and_checkpoint(self, factory):
        model = get_model("scratch")
        set_train_mode(model, "scratch")
        loader = DataLoader(TensorDataset(torch.randn(2, 3, 64, 64), torch.tensor([0, 1])), batch_size=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        loss, _ = run_epoch(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"), optimizer, scaler)
        self.assertGreater(loss, 0)
        self.assertIsNotNone(model.features[0][0].weight.grad)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best_model.pth"
            save_checkpoint(path, model, 1, 1.2, 0.5, "scratch")
            saved = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(saved["epoch"], 1)
            self.assertEqual(saved["val_accuracy"], 0.5)
            restored = get_model("scratch")
            restored.load_state_dict(saved["model_state_dict"])
            torch.testing.assert_close(restored.classifier[-1].weight, model.classifier[-1].weight)


class SplitTests(unittest.TestCase):
    @patch("src.datasets.cifar10.CIFAR10", FakeCIFAR10)
    def test_training_modes_share_dataset_indices(self):
        splits = []
        for _mode in ("scratch", "frozen", "finetune"):
            train, val, _ = get_cifar10_datasets("unused", train_size=200, val_size=500, seed=42)
            splits.append((train.indices, val.indices))
        self.assertEqual(splits[0], splits[1])
        self.assertEqual(splits[1], splits[2])
        self.assertFalse(set(splits[0][0]) & set(splits[0][1]))


class ComparisonTests(unittest.TestCase):
    def test_only_matching_runs_are_compared(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode, accuracy, seed in (
                ("scratch", 0.4, 42), ("frozen", 0.5, 42), ("finetune", 0.6, 43)
            ):
                run_dir = root / mode / "20260101-000000-00000001"
                run_dir.mkdir(parents=True)
                config = {field: "same" for field in compare.MATCH_FIELDS}
                config.update(mode=mode, seed=seed)
                summary = {
                    "best_val_accuracy": accuracy, "best_val_loss": 1.0,
                    "training_seconds": 5.0, "epochs": 5,
                }
                (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
                (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            output = io.StringIO()
            with patch.object(compare, "RUNS_DIR", root), contextlib.redirect_stdout(output):
                compare.main()
            result = output.getvalue()
            self.assertIn("+0.1000", result)
            self.assertIn("Incompatible finetune run", result)
            self.assertIn("seed", result)


if __name__ == "__main__":
    unittest.main()
