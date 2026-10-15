"""Frozen encoders are extracted once, then linear probes train on cached features."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn

from cfgs.cfg import AugmentConfig, ExperimentConfig
from src.models import FrozenVisionClassifier, MODEL_REGISTRY
from src.models_sae import build_sae
from src.trainer import run_experiment
from test_sae import iter_hidden_embeddings


class TinyFrozenModel(FrozenVisionClassifier):
    feature_calls = 0

    def __init__(self, num_classes: int, pretrained: bool = True):
        backbone = nn.Sequential(nn.Conv2d(3, 4, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        super().__init__(backbone, 4, num_classes)

    def forward_features(self, images):
        type(self).feature_calls += 1
        with torch.no_grad():
            return self.backbone(images)


class FrozenPipelineTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for split in ("train", "test"):
            for label in ("ant", "zebra"):
                folder = self.root / "DS" / split / label
                folder.mkdir(parents=True)
                for index in range(6 if split == "train" else 1):
                    Image.new("RGB", (32, 32), (index * 20, 50, 170)).save(
                        folder / f"{index:02}.png")

    def test_cached_features_train_full_splits_and_reload_for_sae(self):
        for model_name, sample_percentage in (("blip", 40), ("dino", None)):
            with self.subTest(model_name=model_name), patch.dict(
                    MODEL_REGISTRY, {model_name: TinyFrozenModel}):
                TinyFrozenModel.feature_calls = 0
                config = ExperimentConfig(
                    root_dir=self.root, dataset_path="DS", log_dir=f"runs_{model_name}",
                    model_name=model_name, pretrained=False, epochs=2, batch_size=4,
                    export_batch_size=4, num_workers=0, accelerator="cpu",
                    sae_smples=sample_percentage, augments=AugmentConfig(image_size=32),
                )
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    run = run_experiment(config)
                self.assertEqual(TinyFrozenModel.feature_calls, 4)
                self.assertIn("Extracting train features", output.getvalue())
                self.assertIn("Extracting test features", output.getvalue())
                out = run / "out"
                train = torch.load(out / "features_train.pt", weights_only=True)
                test = torch.load(out / "features_test.pt", weights_only=True)
                self.assertEqual((train.shape, test.shape), ((12, 4), (2, 4)))
                self.assertEqual(torch.load(out / "feature_labels_train.pt", weights_only=True).shape,
                                 (12,))
                exported = torch.load(out / "embeddings_train.pt", weights_only=True)
                self.assertEqual(exported.shape, (5 if sample_percentage else 12, 4))
                with (out / "feature_paths_train.csv").open(newline="") as handle:
                    full_paths = [row["image_path"] for row in csv.DictReader(handle)]
                with (out / "image_paths_train.csv").open(newline="") as handle:
                    sampled_paths = [row["image_path"] for row in csv.DictReader(handle)]
                indices = [full_paths.index(path) for path in sampled_paths]
                torch.testing.assert_close(exported, train[indices])
                if sample_percentage is None:
                    self.assertEqual((out / "features_train.pt").stat().st_ino,
                                     (out / "embeddings_train.pt").stat().st_ino)
                checkpoint = torch.load(run / "checkpoints/last.ckpt", weights_only=True)
                self.assertEqual(set(checkpoint["state_dict"]), {"model.weight", "model.bias"})
                saved_encoder = torch.load(run / "frozen_encoder.pt", weights_only=True)
                self.assertEqual(saved_encoder["embedding_dim"], 4)
                self.assertTrue(json.loads((run / "config.json").read_text())
                                ["frozen_feature_training"])
                with (run / "metrics.csv").open(newline="") as handle:
                    metrics = list(csv.DictReader(handle))
                self.assertEqual(sum(bool(row["train_loss"]) for row in metrics), 2)
                self.assertEqual(sum(bool(row["eval_loss"]) for row in metrics), 2)

                sae_run = run / "SAE" / "fixture"
                sae_run.mkdir(parents=True)
                architecture = dict(name="basic", input_dim=4, nb_concepts=8,
                                    top_k=2, threshold_momentum=0.9)
                sae = build_sae(**architecture).eval()
                torch.save({"format_version": 1, "architecture": architecture,
                            "source_model_relative_path": "../..",
                            "source_model_log_dir": str(run),
                            "state_dict": sae.state_dict()}, sae_run / "model.pt")
                rows = list(iter_hidden_embeddings(sae_run, self.root / "DS/test",
                                                   batch_size=2, device="cpu"))
                self.assertEqual(len(rows), 2)
                with torch.inference_mode():
                    expected = sae.encode(test)[1]
                torch.testing.assert_close(torch.stack([code for _, code in rows]), expected)


if __name__ == "__main__":
    unittest.main()
