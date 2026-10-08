"""Model registry: models expose features immediately before the classifier."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from transformers import (BlipForConditionalGeneration, BlipVisionConfig,
                          BlipVisionModel, ViTConfig, ViTModel)


BLIP_CHECKPOINT = "Salesforce/blip-image-captioning-base"
DINO_CHECKPOINT = "facebook/dino-vitb16"


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


class FrozenVisionClassifier(nn.Module):
    """A fixed image encoder and a trainable linear probe."""

    def __init__(self, backbone: nn.Module, embedding_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.embedding_dim = embedding_dim
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(images))


class Blip(FrozenVisionClassifier):
    def __init__(self, num_classes: int, pretrained: bool = True):
        if pretrained:
            # Retain only the image encoder; the caption decoder is unnecessary.
            backbone = BlipForConditionalGeneration.from_pretrained(BLIP_CHECKPOINT).vision_model
        else:
            backbone = BlipVisionModel(BlipVisionConfig())
        super().__init__(backbone, backbone.config.hidden_size, num_classes)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.backbone(pixel_values=images,
                                 interpolate_pos_encoding=True).pooler_output


class Dino(FrozenVisionClassifier):
    def __init__(self, num_classes: int, pretrained: bool = True):
        if pretrained:
            backbone = ViTModel.from_pretrained(DINO_CHECKPOINT, add_pooling_layer=False)
        else:
            backbone = ViTModel(ViTConfig(), add_pooling_layer=False)
        super().__init__(backbone, backbone.config.hidden_size, num_classes)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.backbone(pixel_values=images,
                                 interpolate_pos_encoding=True).last_hidden_state[:, 0]


MODEL_REGISTRY = {"resnet18": ResNet18, "blip": Blip, "dino": Dino}


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    try:
        constructor = MODEL_REGISTRY[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODEL_REGISTRY)}") from exc
    return constructor(num_classes=num_classes, pretrained=pretrained)
