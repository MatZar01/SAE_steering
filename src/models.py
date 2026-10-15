"""Model registry: models expose features immediately before the classifier."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from transformers import (Blip2VisionConfig, Blip2VisionModel,
                          DINOv3ViTConfig, DINOv3ViTModel)


BLIP_CHECKPOINT = "Salesforce/blip2-opt-2.7b"
DINO_CHECKPOINT = "facebook/dinov3-vitb16-pretrain-lvd1689m"


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
            # Load the vision weights from the full BLIP-2 checkpoint without
            # instantiating its Q-Former or language model.
            vision_config = Blip2VisionConfig.from_pretrained(BLIP_CHECKPOINT)
            backbone = Blip2VisionModel.from_pretrained(
                BLIP_CHECKPOINT, config=vision_config,
                key_mapping={r"^vision_model\.": ""},
            )
        else:
            backbone = Blip2VisionModel(Blip2VisionConfig())
        super().__init__(backbone, backbone.config.hidden_size, num_classes)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.backbone(pixel_values=images,
                                 interpolate_pos_encoding=True).pooler_output


class Dino(FrozenVisionClassifier):
    def __init__(self, num_classes: int, pretrained: bool = True):
        if pretrained:
            backbone = DINOv3ViTModel.from_pretrained(DINO_CHECKPOINT)
        else:
            backbone = DINOv3ViTModel(DINOv3ViTConfig(
                hidden_size=768, intermediate_size=3072, num_hidden_layers=12,
                num_attention_heads=12, num_register_tokens=4,
            ))
        super().__init__(backbone, backbone.config.hidden_size, num_classes)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.backbone(pixel_values=images).pooler_output


MODEL_REGISTRY = {"resnet18": ResNet18, "blip": Blip, "dino": Dino}


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    try:
        constructor = MODEL_REGISTRY[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODEL_REGISTRY)}") from exc
    return constructor(num_classes=num_classes, pretrained=pretrained)
