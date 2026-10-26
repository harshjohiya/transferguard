"""Trainable parts of the existing TorchVision MobileNetV3-Small."""

POLICIES = ("frozen", "late", "full")
# Phase 4's middle representation ends at features[8].
LATE_START = 9


def apply_policy(model, policy):
    """Configure a source-initialized model for one CIFAR-10 adaptation policy."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    if len(model.features) != 13:
        raise ValueError("Expected TorchVision MobileNetV3-Small feature blocks")
    if model.classifier[-1].out_features != 10:
        raise ValueError("Expected a ten-class CIFAR-10 output layer")
    for index, block in enumerate(model.features):
        trainable = policy == "full" or (policy == "late" and index >= LATE_START)
        for parameter in block.parameters():
            parameter.requires_grad_(trainable)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)
    set_policy_train_mode(model, policy)
    return model


def set_policy_train_mode(model, policy):
    """Call at each epoch start so frozen BatchNorm statistics cannot drift."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    model.train()
    if policy == "frozen":
        model.features.eval()
    elif policy == "late":
        for block in model.features[:LATE_START]:
            block.eval()
