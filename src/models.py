"""Model registry: models expose features immediately before the classifier."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights, progress=False)
        self.embedding_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(images))


# Register additional constructors with the same interface here.
MODEL_REGISTRY = {"resnet18": ResNet18}


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    try:
        constructor = MODEL_REGISTRY[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODEL_REGISTRY)}") from exc
    return constructor(num_classes=num_classes, pretrained=pretrained)
