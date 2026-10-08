"""Contract checks for the pretrained vision encoders without weight downloads."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from cfgs.cfg import ExperimentConfig
from src.augments import BLIP_MEAN, BLIP_STD, build_transforms
from src.models import build_model
from src.trainer import Classifier


class FakeVisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.projection = nn.Linear(3, 4)
        self.dropout = nn.Dropout(0.9)

    def forward(self, pixel_values, interpolate_pos_encoding):
        assert interpolate_pos_encoding
        features = self.dropout(self.projection(pixel_values.mean(dim=(2, 3))))
        return SimpleNamespace(pooler_output=features, last_hidden_state=features[:, None])


class FrozenModelsTest(unittest.TestCase):
    def test_pretrained_encoders_train_only_new_classifier(self):
        for name, patched in (("blip", "src.models.BlipForConditionalGeneration.from_pretrained"),
                              ("dino", "src.models.ViTModel.from_pretrained")):
            with self.subTest(name=name), patch(patched) as load:
                encoder = FakeVisionEncoder()
                load.return_value = (SimpleNamespace(vision_model=encoder)
                                     if name == "blip" else encoder)
                config = ExperimentConfig(model_name=name, pretrained=True, num_workers=0)
                module = Classifier(config, num_classes=3)
                module.train()
                self.assertFalse(module.model.backbone.training)
                self.assertTrue(module.model.classifier.training)
                self.assertEqual(module.model.embedding_dim, 4)
                self.assertEqual(module.model.classifier.out_features, 3)
                self.assertFalse(any(p.requires_grad for p in encoder.parameters()))

                images = torch.randn(2, 3, 32, 32)
                features = module.model.forward_features(images)
                self.assertEqual(features.shape, (2, 4))
                self.assertFalse(features.requires_grad)
                loss = module(images).sum()
                loss.backward()
                self.assertIsNotNone(module.model.classifier.weight.grad)
                self.assertTrue(all(p.grad is None for p in encoder.parameters()))
                optimizer = module.configure_optimizers()["optimizer"]
                self.assertEqual({id(p) for group in optimizer.param_groups for p in group["params"]},
                                 {id(p) for p in module.model.classifier.parameters()})

    def test_blip_normalization_uses_checkpoint_values(self):
        from cfgs.cfg import AugmentConfig

        transform = build_transforms(AugmentConfig(image_size=384), model_name="blip")
        self.assertEqual(transform.transforms[0].size, (384, 384))
        normalize = transform.transforms[-1]
        self.assertEqual(normalize.mean, BLIP_MEAN)
        self.assertEqual(normalize.std, BLIP_STD)


if __name__ == "__main__":
    unittest.main()
