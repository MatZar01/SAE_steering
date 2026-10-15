"""Rank and render train/test examples for individual SAE latents and groups."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from cfgs.cfg_sae import SAEInferenceConfig
from src.models_sae import build_sae
from src.sae_inference import run_sae_inference


class SAEInferenceTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "image_run"
        out = self.source / "out"
        out.mkdir(parents=True)
        (self.source / "config.json").write_text(json.dumps({"root_dir": str(self.root)}))
        self.sae_run = self.source / "SAE" / "sae_run"
        self.sae_run.mkdir(parents=True)
        self.steering_run = self.sae_run / "steering" / "steering_run"
        self.steering_run.mkdir(parents=True)

        architecture = dict(name="basic", input_dim=2, nb_concepts=3, top_k=2,
                            threshold_momentum=0.9)
        sae = build_sae(**architecture)
        with torch.no_grad():
            sae.encoder.final_block[0].weight.copy_(torch.tensor([
                [1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
            ]))
            sae.encoder.final_block[0].bias.copy_(torch.tensor([0.0, 0.0, -1.0]))
        torch.save({"format_version": 1, "architecture": architecture,
                    "source_model_relative_path": "../..",
                    "source_model_log_dir": str(self.source),
                    "state_dict": sae.state_dict()}, self.sae_run / "model.pt")
        for split, features in (
            ("train", torch.tensor([[1.0, 0.0], [3.0, 1.0],
                                    [0.0, 4.0], [0.0, 0.0]])),
            ("test", torch.tensor([[2.0, 2.0], [5.0, 0.0]])),
        ):
            torch.save(features, out / f"features_{split}.pt")
            torch.save(features[:1], out / f"embeddings_{split}.pt")
            paths = []
            for index in range(len(features)):
                path = Path("images") / split / f"{index}.png"
                absolute = self.root / path
                absolute.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 24), (index * 40, 80, 150)).save(absolute)
                paths.append(str(path))
            for filename, values in ((f"feature_paths_{split}.csv", paths),
                                     (f"image_paths_{split}.csv", paths[:1])):
                with (out / filename).open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["image_path"])
                    writer.writerows((value,) for value in values)
        with (self.steering_run / "neighborhoods.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["anchor", "size", "latents"])
            writer.writeheader()
            writer.writerows([
                {"anchor": 0, "size": 1, "latents": "0"},
                {"anchor": 0, "size": 2, "latents": "0|1"},
                {"anchor": 1, "size": 1, "latents": "1"},
                {"anchor": 1, "size": 2, "latents": "1|0"},
            ])

    def test_full_splits_ranked_and_rendered_separately(self):
        config = SAEInferenceConfig(sae_log_dir=str(self.sae_run),
                                    steering_log_dir=str(self.steering_run),
                                    top_active_images=2, batch_size=2, device="cpu")
        output = run_sae_inference(config)
        self.assertEqual(output, self.sae_run / "active_images")
        for split in ("train", "test"):
            directory = output / split
            self.assertEqual(len(list(directory.glob("*.txt"))), 5)
            self.assertEqual(len(list(directory.glob("*.png"))), 5)
            with Image.open(directory / "group_anchor_0000_size_02.png") as image:
                self.assertEqual(image.width, 384)
                self.assertGreaterEqual(image.height, 448)
            dead = (directory / "latent_0002.txt").read_text()
            self.assertEqual(dead.splitlines()[-1], "rank\tactivation\timage_path")

        train_single = (output / "train/latent_0000.txt").read_text().splitlines()
        self.assertIn("Latent 0", train_single[0])
        self.assertTrue(train_single[3].endswith("images/train/1.png"))
        self.assertTrue(train_single[4].endswith("images/train/0.png"))
        self.assertEqual([float(line.split("\t")[1]) for line in train_single[3:]], [3.0, 1.0])
        train_group = (output / "train/group_anchor_0000_size_02.txt").read_text().splitlines()
        self.assertIn("0|1", train_group[0])
        self.assertIn("Sum of SAE code activations", train_group[1])
        self.assertTrue(train_group[3].endswith("images/train/1.png"))
        self.assertTrue(train_group[4].endswith("images/train/2.png"))
        test_single = (output / "test/latent_0000.txt").read_text().splitlines()
        self.assertTrue(test_single[3].endswith("images/test/1.png"))
        self.assertEqual(float(test_single[3].split("\t")[1]), 5.0)

    def test_grid_side_must_be_integer(self):
        for value in (0, -1, True, 2.5, "3"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "top_active_images"):
                SAEInferenceConfig(sae_log_dir=str(self.sae_run),
                                   top_active_images=value).validate()

    def test_exported_embeddings_used_without_full_feature_cache(self):
        for split in ("train", "test"):
            (self.source / "out" / f"features_{split}.pt").unlink()
        output = run_sae_inference(SAEInferenceConfig(
            sae_log_dir=str(self.sae_run), top_active_images=1,
            batch_size=2, device="cpu"))
        for split in ("train", "test"):
            rows = (output / split / "latent_0000.txt").read_text().splitlines()
            self.assertEqual(len(rows), 4)
            self.assertTrue(rows[3].endswith(f"images/{split}/0.png"))


if __name__ == "__main__":
    unittest.main()
