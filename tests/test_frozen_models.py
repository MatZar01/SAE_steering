"""Contract checks for the pretrained vision encoders without weight downloads."""

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from torch import nn
from transformers import Blip2Config, Blip2VisionConfig, Blip2VisionModel

from cfgs.cfg import ExperimentConfig
from src.augments import BLIP_MEAN, BLIP_STD, build_transforms
from src.models import BLIP_CHECKPOINT, DINO_CHECKPOINT, build_model
from src.trainer import Classifier


class FakeVisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.projection = nn.Linear(3, 4)
        self.dropout = nn.Dropout(0.9)

    def forward(self, pixel_values, interpolate_pos_encoding=False):
        features = self.dropout(self.projection(pixel_values.mean(dim=(2, 3))))
        return SimpleNamespace(pooler_output=features, last_hidden_state=features[:, None])


class FrozenModelsTest(unittest.TestCase):
    def test_blip2_loads_vision_weights_from_full_checkpoint(self):
        vision_config = Blip2VisionConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, image_size=32, patch_size=16,
        )
        source = Blip2VisionModel(vision_config)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            Blip2Config(vision_config=vision_config.to_dict()).save_pretrained(checkpoint)
            save_file({f"vision_model.{name}": value.contiguous()
                       for name, value in source.state_dict().items()},
                      checkpoint / "model.safetensors")
            with patch("src.models.BLIP_CHECKPOINT", str(checkpoint)):
                model = build_model("blip", num_classes=3)
            self.assertEqual(model.embedding_dim, 32)
            for name, value in source.state_dict().items():
                torch.testing.assert_close(model.backbone.state_dict()[name], value)

    def test_pretrained_encoders_train_only_new_classifier(self):
        for name, patched in (("blip", "src.models.Blip2VisionModel.from_pretrained"),
                              ("dino", "src.models.DINOv3ViTModel.from_pretrained")):
            with self.subTest(name=name), patch(patched) as load, \
                    patch("src.models.Blip2VisionConfig.from_pretrained"):
                encoder = FakeVisionEncoder()
                load.return_value = encoder
                config = ExperimentConfig(model_name=name, pretrained=True, num_workers=0)
                module = Classifier(config, num_classes=3)
                self.assertEqual(load.call_args.args[0],
                                 BLIP_CHECKPOINT if name == "blip" else DINO_CHECKPOINT)
                if name == "blip":
                    self.assertEqual(load.call_args.kwargs["key_mapping"],
                                     {r"^vision_model\.": ""})
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
