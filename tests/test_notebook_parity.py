"""Small notebook-to-legacy parity checks; no datasets or lengthy training."""
import ast
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.baselines.train import save_checkpoint, seed_everything
from experiments.negative_transfer.train_target import load_svhn_model
from src.models.mobilenet import get_model
from src.transferability.features import extract_features, load_source_model
from src.transferability.logme import logme_score


ROOT = Path(__file__).resolve().parents[1]


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
