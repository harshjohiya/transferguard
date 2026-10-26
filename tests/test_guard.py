import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import mobilenet_v3_small

from experiments.baselines.train import run_epoch
from experiments.transferguard import run
from src.datasets.cifar10 import balanced_split_indices
from src.guard.decision import paired_guard, split_validation_indices
from src.guard.policies import LATE_START, POLICIES, apply_policy, set_policy_train_mode


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        model = mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 10)
        return model

    def test_freezing_and_frozen_batchnorm_statistics(self):
        for policy in POLICIES:
            with self.subTest(policy=policy):
                model = apply_policy(self.model(), policy)
                for index, block in enumerate(model.features):
                    expected = policy == "full" or (policy == "late" and index >= LATE_START)
                    self.assertTrue(all(p.requires_grad == expected for p in block.parameters()))
                    if not expected:
                        self.assertFalse(block.training)
                self.assertTrue(all(p.requires_grad for p in model.classifier.parameters()))
                frozen_bns = [module for index, block in enumerate(model.features)
                              if policy == "frozen" or (policy == "late" and index < LATE_START)
                              for module in block.modules() if isinstance(module, nn.BatchNorm2d)]
                before = [bn.running_mean.clone() for bn in frozen_bns]
                set_policy_train_mode(model, policy)
                model(torch.randn(2, 3, 64, 64))
                self.assertTrue(all(not bn.training for bn in frozen_bns))
                for bn, expected in zip(frozen_bns, before):
                    torch.testing.assert_close(bn.running_mean, expected)

    def test_only_intended_parameters_update(self):
        loader = DataLoader(TensorDataset(torch.randn(2, 3, 64, 64), torch.tensor([0, 1])),
                            batch_size=2)
        for policy in POLICIES:
            with self.subTest(policy=policy):
                model = apply_policy(self.model(), policy)
                before = {name: value.detach().clone() for name, value in model.named_parameters()}
                optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)
                run_epoch(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"), optimizer,
                          torch.amp.GradScaler("cuda", enabled=False))
                for name, value in model.named_parameters():
                    if not value.requires_grad:
                        torch.testing.assert_close(value, before[name])
                self.assertFalse(torch.equal(model.classifier[-1].weight,
                                             before["classifier.3.weight"]))
                if policy == "late":
                    self.assertFalse(torch.equal(model.features[-1][0].weight,
                                                 before["features.12.0.weight"]))
                if policy == "full":
                    self.assertFalse(torch.equal(model.features[0][0].weight,
                                                 before["features.0.0.weight"]))


class DecisionTests(unittest.TestCase):
    def test_balanced_reproducible_disjoint_partitions(self):
        targets = [label for label in range(10) for _ in range(1000)]
        train, validation = balanced_split_indices(targets, 2000, 5000, 42)
        selection, guard = split_validation_indices(validation, targets, 42)
        self.assertEqual((selection, guard), split_validation_indices(validation, targets, 42))
        self.assertEqual((len(selection), len(guard)), (2500, 2500))
        self.assertFalse(set(selection) & set(guard))
        self.assertFalse(set(train) & (set(selection) | set(guard)))
        self.assertEqual(set(Counter(targets[i] for i in selection).values()), {250})
        self.assertEqual(set(Counter(targets[i] for i in guard).values()), {250})
        self.assertEqual(set(selection) | set(guard), set(validation))

    def test_paired_accuracy_and_clear_decisions(self):
        labels = [0] * 100
        transfer = [0] * 80 + [1] * 20
        scratch = [0] * 60 + [1] * 40
        result = paired_guard(labels, transfer, scratch, bootstrap_samples=1000)
        self.assertAlmostEqual(result["transfer_accuracy"], 0.8)
        self.assertAlmostEqual(result["scratch_accuracy"], 0.6)
        self.assertAlmostEqual(result["accuracy_difference"], 0.2)
        self.assertGreater(result["ci_lower"], 0)
        self.assertEqual(result["decision"], "transfer")
        harmful = paired_guard(labels, scratch, transfer, bootstrap_samples=1000)
        self.assertEqual(harmful["decision"], "scratch")
        uncertain = paired_guard(labels, [0] + [1] * 99, [1] * 100,
                                 bootstrap_samples=1000)
        self.assertEqual(uncertain["decision"], "scratch")
        self.assertLessEqual(uncertain["ci_lower"], 0)
        self.assertEqual(result, paired_guard(labels, transfer, scratch, bootstrap_samples=1000))


class LogMEIntegrationTests(unittest.TestCase):
    def test_score_ranking_and_dataset_validation(self):
        args = run.parse_args(["--device", "cpu"])
        identity = {"indices_sha256": "indices", "labels_sha256": "labels"}
        config = {
            "architecture": "MobileNetV3-Small", "dataset": "CIFAR-10 official training partition",
            "split": "balanced_split_indices", "train_size": 2000, "val_size": 5000,
            "seed": 42, "image_size": [128, 128],
            "eval_transform": "Resize(128,128), ToTensor, ImageNet Normalize",
            "normalization_mean": list(run.IMAGENET_MEAN),
            "normalization_std": list(run.IMAGENET_STD),
            "feature_stages": {"final": [12, 576]},
            "logme_variant": "official later fixed-point; direct SVD, float64", **identity,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("imagenet", "svhn"):
                (root / name / "one").mkdir(parents=True)
            def load(path, source):
                score = 0.3 if source == "imagenet" else -0.2
                return dict(config), [{"stage": "final", "logme_score": score}]
            with patch.object(run, "LOGME_DIR", root), patch.object(run, "load_run", side_effect=load):
                scores = run.load_logme_scores(args, identity)
                self.assertEqual(run.choose_source(args, scores)[0], "imagenet")
                with self.assertRaisesRegex(ValueError, "indices_sha256"):
                    run.load_logme_scores(args, {**identity, "indices_sha256": "wrong"})


if __name__ == "__main__":
    unittest.main()
