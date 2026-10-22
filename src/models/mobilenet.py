"""MobileNetV3-Small baselines for CIFAR-10."""

import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


MODES = ("scratch", "frozen", "finetune")


def get_model(mode, num_classes=10):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    weights = None if mode == "scratch" else MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    if mode == "frozen":
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        model.features.eval()

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"{mode}: {total:,} total parameters; {trainable:,} trainable parameters")
    return model


def set_train_mode(model, mode):
    """Restore frozen feature evaluation after each call to model.train()."""
    model.train()
    if mode == "frozen":
        model.features.eval()
