import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torchvision.models import mobilenet_v3_small

from experiments.baselines.train import save_checkpoint
from experiments.transferability.compare import compare_runs, load_run
from src.datasets.cifar10 import balanced_split_indices
from src.transferability.features import (
    STAGES, extract_features, load_source_model, target_loader, target_training_subset,
)
from src.transferability.logme import logme_score


class FakeCIFAR10:
    calls = []

    def __init__(self, root, train, download, transform):
        self.calls.append(train)
        self.train = train
        self.targets = [label for label in range(10) for _ in range(100)]
        self.transform = transform

    def __getitem__(self, index):
        image = Image.new("RGB", (32, 32), (index % 255, 64, 32))
        return self.transform(image), self.targets[index]

    def __len__(self):
        return len(self.targets)


def direct_reference(x, labels):
    """Small-case Gaussian evidence via a direct posterior solve and slogdet."""
    n, d = x.shape
    scores = []
    for label in np.unique(labels):
        y = (labels == label).astype(float)
        alpha = beta = 1.0
        for _ in range(256):
            a = alpha * np.eye(d) + beta * x.T @ x
            covariance = np.linalg.solve(a, np.eye(d))
            mean = beta * np.linalg.solve(a, x.T @ y)
            gamma = d - alpha * np.trace(covariance)
            alpha_next = np.clip(gamma / max(mean @ mean, 1e-12), 1e-12, 1e12)
            beta_next = np.clip((n - gamma) / max(np.sum((y - x @ mean) ** 2), 1e-12), 1e-12, 1e12)
            change = max(abs(np.log(alpha_next / alpha)), abs(np.log(beta_next / beta)))
            alpha, beta = alpha_next, beta_next
            if change < 1e-7:
                break
        a = alpha * np.eye(d) + beta * x.T @ x
        mean = beta * np.linalg.solve(a, x.T @ y)
        score = (d * np.log(alpha) + n * np.log(beta) - np.linalg.slogdet(a)[1]
                 - beta * np.sum((y - x @ mean) ** 2) - alpha * (mean @ mean)
                 - n * np.log(2 * np.pi)) / (2 * n)
        scores.append(score)
    return float(np.mean(scores))


class LogMETests(unittest.TestCase):
    def test_matches_direct_gaussian_reference(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(24, 4))
        labels = (x[:, 0] + 0.3 * x[:, 1] > 0).astype(int)
        self.assertAlmostEqual(logme_score(x, labels), direct_reference(x, labels), places=5)

    def test_rank_deficient_and_uninformative_features(self):
        rng = np.random.default_rng(11)
        labels = np.repeat([0, 1], 30)
        informative = np.column_stack((labels * 2 - 1, labels * 2 - 1))
        random = rng.normal(size=(60, 2))
        self.assertTrue(np.isfinite(logme_score(np.zeros((60, 3)), labels)))
        self.assertTrue(np.isfinite(logme_score(informative, labels)))
        self.assertAlmostEqual(logme_score(informative, labels),
                               direct_reference(informative, labels), places=4)
        self.assertGreater(logme_score(informative, labels), logme_score(random, labels))

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            logme_score(np.ones((2, 2)), np.array([0, 0]))
        with self.assertRaises(ValueError):
            logme_score(np.array([[np.nan], [1.0]]), np.array([0, 1]))


@patch("src.transferability.features.CIFAR10", FakeCIFAR10)
class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_fixed_subset_no_leakage_and_determinism(self):
        FakeCIFAR10.calls.clear()
        subset, identity = target_training_subset("unused", 20, 20, 42)
        second, second_identity = target_training_subset("unused", 20, 20, 42)
        self.assertEqual(identity, second_identity)
        self.assertEqual(subset.indices, second.indices)
        self.assertEqual(FakeCIFAR10.calls, [True, True])
        self.assertTrue(subset.dataset.train)
        self.assertEqual(len(subset), 20)
        _, validation_indices = balanced_split_indices(subset.dataset.targets, 20, 20, 42)
        self.assertFalse(set(subset.indices) & set(validation_indices))
        self.assertFalse(any("Random" in type(step).__name__
                             for step in subset.dataset.transform.transforms))
        first, _ = subset[0]
        torch.testing.assert_close(first, subset[0][0])
        with patch("src.transferability.features.get_model",
                   side_effect=lambda mode: mobilenet_v3_small(weights=None)):
            model, _ = load_source_model("random")
        before = {key: value.clone() for key, value in model.state_dict().items()}
        loader = target_loader(subset, 4)
        features, labels = extract_features(model, loader, torch.device("cpu"), tuple(STAGES))
        repeat, repeat_labels = extract_features(model, loader, torch.device("cpu"), tuple(STAGES))
        for stage, (_, dimension) in STAGES.items():
            self.assertEqual(features[stage].shape, (20, dimension))
            np.testing.assert_array_equal(features[stage], repeat[stage])
        np.testing.assert_array_equal(labels, repeat_labels)
        np.testing.assert_array_equal(labels, [subset.dataset.targets[i] for i in subset.indices])
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key])

    def test_source_checkpoint_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best_model.pth"
            model = mobilenet_v3_small(weights=None)
            model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 10)
            save_checkpoint(path, model, 1, 1.0, 0.5, "source_svhn")
            config = {"mode": "source_svhn", "architecture": "MobileNetV3-Small",
                      "num_classes": 10, "dataset": "SVHN official training partition"}
            path.with_name("config.json").write_text(json.dumps(config), encoding="utf-8")
            with patch("src.models.mobilenet.mobilenet_v3_small",
                       side_effect=lambda weights: mobilenet_v3_small(weights=None)):
                restored, details = load_source_model("svhn", path)
                self.assertEqual(details["path"], str(path.resolve()))
                torch.testing.assert_close(restored.features[0][0].weight, model.features[0][0].weight)
                self.assertTrue(all(not p.requires_grad for p in restored.parameters()))
                config["dataset"] = "CIFAR-10"
                path.with_name("config.json").write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_source_model("svhn", path)


class ComparisonTests(unittest.TestCase):
    def test_matching_runs_and_rejected_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = {}
            for source, score in (("imagenet", 0.3), ("svhn", 0.2), ("random", 0.1)):
                path = Path(directory) / source
                path.mkdir()
                config = {field: "same" for field in (
                    "architecture", "dataset", "split", "eval_transform", "indices_sha256",
                    "labels_sha256", "logme_variant")}
                config.update(source=source, train_size=20, val_size=20, seed=42, layers="final",
                              feature_stages={"final": [12, 576]}, image_size=[128, 128],
                              normalization_mean=[0.1] * 3, normalization_std=[0.2] * 3)
                (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
                with (path / "scores.csv").open("w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=("source", "stage", "feature_dim",
                        "num_examples", "logme_score", "extraction_seconds", "scoring_seconds"))
                    writer.writeheader()
                    writer.writerow(dict(source=source, stage="final", feature_dim=576,
                                         num_examples=20, logme_score=score,
                                         extraction_seconds=1, scoring_seconds=2))
                runs[source] = load_run(path, source)
            rows = compare_runs(runs)
            self.assertEqual([row["rank"] for row in rows], [1, 2, 3])
            runs["svhn"][0]["indices_sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "indices_sha256"):
                compare_runs(runs)


if __name__ == "__main__":
    unittest.main()
