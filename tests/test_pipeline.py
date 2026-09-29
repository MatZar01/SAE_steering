"""Small CPU integration tests; no pretrained download or real training run."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from cfgs.cfg import AugmentConfig, ExperimentConfig, SchedulerConfig
from src.data import ImageDataModule
from src.trainer import Classifier, run_experiment


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        dataset = self.root / "DS"
        for split in ("train", "test"):
            for label in ("zebra", "ant"):
                (dataset / split / label).mkdir(parents=True)
        # Create files out of order to check deterministic export alignment.
        for label in ("zebra", "ant"):
            for i in reversed(range(6)):
                Image.new("RGB", (32, 32), (i * 20, 255 - i * 20, 60)).save(
                    dataset / "train" / label / f"{i:02}.png")
            Image.new("RGB", (32, 32)).save(dataset / "test" / label / "sample.jpg")
        self.config = ExperimentConfig(
            root_dir=self.root, dataset_path="DS", pretrained=False,
            epochs=5, batch_size=4, num_workers=0, accelerator="cpu",
            augments=AugmentConfig(image_size=32),
        )
        torch.set_num_threads(2)

    def test_class_folders_and_test_evaluation(self):
        data = ImageDataModule(self.config)
        data.setup()
        self.assertEqual(len(data.train_records), 12)
        self.assertEqual(len(data.eval_records), 2)
        self.assertFalse(set(data.train_records) & set(data.eval_records))
        self.assertEqual(len(data.export_datasets["train"]), 12)
        self.assertEqual(len(data.export_datasets["test"]), 2)
        self.assertEqual({r.label for r in data.eval_records}, {"ant", "zebra"})
        unknown = self.root / "DS" / "test" / "unknown"
        unknown.mkdir()
        Image.new("RGB", (32, 32)).save(unknown / "sample.png")
        with self.assertRaisesRegex(ValueError, "Unknown test classes"):
            ImageDataModule(self.config).setup()

    def test_training_metrics_and_aligned_final_embeddings(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            run = run_experiment(self.config)
        lines = output.getvalue().strip().splitlines()
        self.assertEqual(len(lines), 1, lines)
        self.assertTrue(lines[0].startswith("Epoch 005 | train loss="))
        self.assertIn("test loss=", lines[0])
        self.assertRegex(run.name, r"^\d{6}-\d{2}_\d{2}_\d{2}$")
        self.assertEqual({path.name for path in (run / "out").iterdir()},
                         {f"{kind}_{split}.{extension}"
                          for split in ("train", "test")
                          for kind, extension in (("embeddings", "pt"), ("labels", "pt"),
                                                  ("image_paths", "csv"))})
        for split, count, expected_labels, expected_paths in (
            ("train", 12, [0] * 6 + [1] * 6,
             [f"DS/train/{label}/{i:02}.png"
              for label in ("ant", "zebra") for i in range(6)]),
            ("test", 2, [0, 1],
             [f"DS/test/{label}/sample.jpg" for label in ("ant", "zebra")]),
        ):
            with self.subTest(split=split):
                embeddings = torch.load(run / f"out/embeddings_{split}.pt", weights_only=True)
                labels = torch.load(run / f"out/labels_{split}.pt", weights_only=True)
                with (run / f"out/image_paths_{split}.csv").open(newline="") as handle:
                    paths = [r["image_path"] for r in csv.DictReader(handle)]
                self.assertEqual(embeddings.shape, (count, 512))
                self.assertEqual(labels.tolist(), expected_labels)
                self.assertEqual(paths, expected_paths)
                self.assertTrue(torch.isfinite(embeddings).all())
        self.assertEqual(json.loads((run / "class_to_idx.json").read_text()), {"ant": 0, "zebra": 1})
        metadata = json.loads((run / "config.json").read_text())
        self.assertEqual((metadata["num_train_images"], metadata["num_evaluation_images"]), (12, 2))
        self.assertEqual((metadata["num_export_train_images"], metadata["num_export_test_images"]),
                         (12, 2))
        with (run / "metrics.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len([r for r in rows if r.get("train_loss")]), 5)
        self.assertEqual(len([r for r in rows if r.get("eval_loss")]), 5)
        self.assertTrue((run / "checkpoints/last.ckpt").is_file())
        # Recompute from the final checkpoint to detect order or checkpoint mismatches.
        checkpoint = torch.load(run / "checkpoints/last.ckpt", weights_only=False)
        module = Classifier(self.config, 2)
        module.load_state_dict(checkpoint["state_dict"])
        module.eval()
        data = ImageDataModule(self.config)
        data.setup()
        for split in ("train", "test"):
            with self.subTest(split=split), torch.inference_mode():
                expected = torch.cat([module.model.forward_features(images)
                                      for images, _, _ in data.export_dataloader(split)])
                embeddings = torch.load(run / f"out/embeddings_{split}.pt", weights_only=True)
                torch.testing.assert_close(embeddings, expected)

    def test_plateau_scheduler_reduces_learning_rate(self):
        from src.scheduler import build_scheduler

        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        settings = build_scheduler(optimizer, SchedulerConfig(patience=0, factor=0.5))
        self.assertEqual(settings["monitor"], "eval_loss")
        settings["scheduler"].step(1.0)
        settings["scheduler"].step(1.0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.05)


if __name__ == "__main__":
    unittest.main()
