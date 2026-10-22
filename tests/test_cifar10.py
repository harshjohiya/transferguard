import unittest
from collections import Counter
from unittest.mock import patch

import torch
from PIL import Image
from torchvision import transforms

from src.datasets.cifar10 import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    balanced_split_indices,
    get_cifar10_datasets,
    get_cifar10_loaders,
)


class FakeCIFAR10:
    classes = [str(label) for label in range(10)]

    def __init__(self, root, train, download, transform):
        self.targets = [label for label in range(10) for _ in range(1000 if train else 20)]
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = Image.new("RGB", (32, 32), (128, 64, 32))
        return self.transform(image), self.targets[index]


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.targets = [label for label in range(10) for _ in range(5000)]

    def test_reproducible_balanced_disjoint_splits(self):
        train, val = balanced_split_indices(self.targets)
        self.assertEqual((train, val), balanced_split_indices(self.targets))
        self.assertNotEqual((train, val), balanced_split_indices(self.targets, seed=43))
        self.assertEqual(len(train), 2000)
        self.assertEqual(len(val), 5000)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(Counter(self.targets[i] for i in train).values()), {200})
        self.assertEqual(set(Counter(self.targets[i] for i in val).values()), {500})

    def test_invalid_sizes(self):
        for train_size, val_size in [(0, 500), (201, 500), (200, 50000)]:
            with self.subTest(train_size=train_size, val_size=val_size):
                with self.assertRaises(ValueError):
                    balanced_split_indices(self.targets, train_size, val_size)


@patch("src.datasets.cifar10.CIFAR10", FakeCIFAR10)
class DatasetTests(unittest.TestCase):
    def test_partitions_transforms_and_labels(self):
        train, val, test = get_cifar10_datasets("unused", train_size=200, val_size=500)
        self.assertEqual((len(train), len(val), len(test)), (200, 500, 200))
        self.assertFalse(set(train.indices) & set(val.indices))
        self.assertTrue(train.dataset.train)
        self.assertTrue(val.dataset.train)
        self.assertFalse(test.train)

        train_steps = train.dataset.transform.transforms
        val_steps = val.dataset.transform.transforms
        self.assertTrue(any(isinstance(step, transforms.RandomCrop) for step in train_steps))
        self.assertTrue(any(isinstance(step, transforms.RandomHorizontalFlip) for step in train_steps))
        self.assertFalse(any(isinstance(step, transforms.RandomCrop) for step in val_steps))
        self.assertEqual(test.transform.transforms, val_steps)

        for dataset in (train, val, test):
            image, label = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 128, 128))
            self.assertIn(label, range(10))

        image, _ = val[0]
        expected = torch.tensor([
            (value / 255 - mean) / std
            for value, mean, std in zip((128, 64, 32), IMAGENET_MEAN, IMAGENET_STD)
        ])
        torch.testing.assert_close(image[:, 0, 0], expected)

    def test_loaders(self):
        train, val, test = get_cifar10_loaders(
            "unused", train_size=200, val_size=500, batch_size=16
        )
        for loader in (train, val, test):
            images, labels = next(iter(loader))
            self.assertEqual(tuple(images.shape), (16, 3, 128, 128))
            self.assertEqual(tuple(labels.shape), (16,))
            self.assertEqual(loader.num_workers, 0)
        self.assertEqual(len(train), 13)
        self.assertEqual(len(val), 32)
        self.assertEqual(len(test), 13)


if __name__ == "__main__":
    unittest.main()
