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
        train = self.root / "DS" / "train"
        train.mkdir(parents=True)
        # Deliberately nonalphabetic CSV order tests alignment independent of sorting.
        with (train.parent / "trainLabels.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "label"])
            for i in range(12):
                Image.new("RGB", (32, 32), (i * 20, 255 - i * 20, 60)).save(train / f"{i}.png")
                writer.writerow([i, "zebra" if i % 2 == 0 else "ant"])
        self.config = ExperimentConfig(
            root_dir=self.root, dataset_path="DS", pretrained=False,
            epochs=5, batch_size=4, num_workers=0, accelerator="cpu",
            augments=AugmentConfig(image_size=32),
        )
        torch.set_num_threads(2)

    def test_split_and_labeled_test(self):
        data = ImageDataModule(self.config)
        data.setup()
        other = ImageDataModule(self.config)
        other.setup()
        self.assertEqual(data.eval_records, other.eval_records)
        self.assertFalse(set(data.train_records) & set(data.eval_records))
        self.assertEqual(len(data.export_dataset), 12)
        self.assertEqual({r.label for r in data.eval_records}, {"ant", "zebra"})
        test = self.root / "DS" / "test"
        test.mkdir()
        Image.new("RGB", (32, 32)).save(test / "1.png")
        (test.parent / "testLabels.csv").write_text("id,label\n1,ant\n")
        self.config.test_labels_csv = "testLabels.csv"
        labeled = ImageDataModule(self.config)
        labeled.setup()
        self.assertEqual(len(labeled.train_records), 12)
        self.assertEqual(labeled.evaluation_source, "labeled_test")
        (test.parent / "testLabels.csv").write_text("id,label\n1,unknown\n")
        with self.assertRaisesRegex(ValueError, "Unknown test labels"):
            ImageDataModule(self.config).setup()

    def test_training_metrics_and_aligned_final_embeddings(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            run = run_experiment(self.config)
        lines = output.getvalue().strip().splitlines()
        self.assertEqual(len(lines), 1, lines)
        self.assertTrue(lines[0].startswith("Epoch 005 | train loss="))
        self.assertIn("val loss=", lines[0])
        self.assertRegex(run.name, r"^\d{6}-\d{2}_\d{2}_\d{2}$")
        embeddings = torch.load(run / "out/embeddings.pt", weights_only=True)
        labels = torch.load(run / "out/labels.pt", weights_only=True)
        with (run / "out/image_paths.csv").open(newline="") as handle:
            paths = [r["image_path"] for r in csv.DictReader(handle)]
        self.assertEqual(embeddings.shape, (12, 512))
        self.assertEqual(labels.tolist(), [1, 0] * 6)
        self.assertEqual(paths, [f"DS/train/{i}.png" for i in range(12)])
        self.assertTrue(torch.isfinite(embeddings).all())
        self.assertEqual(json.loads((run / "class_to_idx.json").read_text()), {"ant": 0, "zebra": 1})
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
        with torch.inference_mode():
            expected = torch.cat([module.model.forward_features(images)
                                  for images, _, _ in data.export_dataloader()])
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
