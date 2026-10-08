"""Small end-to-end checks for PDF point 1 using saved SAE artifacts."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from cfgs.cfg_sae import SteeringConfig
from src.models_sae import build_sae, load_sae
from src.steering import run_steering


class SteeringTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "image_run"
        (self.source / "out").mkdir(parents=True)
        self.sae_run = self.source / "SAE" / "test_run"
        self.sae_run.mkdir(parents=True)
        (self.source / "class_to_idx.json").write_text(json.dumps({"left": 0, "right": 1}))
        train = torch.tensor([
            [2.0, 0.0], [2.2, 0.1], [1.8, 0.0], [2.1, 0.2],
            [0.0, 2.0], [0.1, 2.2], [0.0, 1.8], [0.2, 2.1],
        ])
        test = torch.tensor([[2.05, 0.05], [1.9, 0.1], [0.05, 2.0], [0.1, 1.9]])
        torch.save(train, self.source / "out/embeddings_train.pt")
        torch.save(torch.tensor([0] * 4 + [1] * 4), self.source / "out/labels_train.pt")
        torch.save(test, self.source / "out/embeddings_test.pt")
        torch.save(torch.tensor([0, 0, 1, 1]), self.source / "out/labels_test.pt")

        architecture = dict(name="basic", input_dim=2, nb_concepts=4, top_k=2,
                            threshold_momentum=0.9)
        sae = build_sae(**architecture)
        with torch.no_grad():
            sae.encoder.final_block[0].weight.copy_(torch.tensor([
                [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0],
            ]))
            sae.encoder.final_block[0].bias.copy_(torch.tensor([0.0, 0.0, 0.0, -1.0]))
            sae.dictionary._weights.copy_(torch.tensor([
                [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0],
            ]))
        torch.save({"format_version": 1, "architecture": architecture,
                    "source_model_relative_path": "../..",
                    "source_model_log_dir": str(self.source),
                    "state_dict": sae.state_dict()}, self.sae_run / "model.pt")
        self.base_state = sae.state_dict()

    def test_nested_neighborhoods_fixed_targets_and_population_outputs(self):
        config = SteeringConfig(sae_log_dir=str(self.sae_run), max_anchors=None,
                                max_neighborhood_size=3, classifier_epochs=80,
                                classifier_learning_rate=0.05,
                                classifier_batch_size=4, evaluation_batch_size=3,
                                device="cpu")
        output = run_steering(config)
        self.assertEqual(output.parent, self.sae_run / "steering")
        with (output / "neighborhoods.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        with (output / "summary.csv").open(newline="") as handle:
            summaries = list(csv.DictReader(handle))
        metadata = json.loads((output / "metadata.json").read_text())
        stats = torch.load(output / "latent_statistics.pt", weights_only=True)
        classifier = torch.load(output / "classifier.pt", weights_only=True)

        self.assertEqual(metadata["num_non_dead_latents"], 3)
        self.assertEqual(metadata["num_sampled_anchors"], 3)
        self.assertEqual(len(rows), 9)
        self.assertEqual(len(summaries), 6)
        self.assertEqual(classifier["state_dict"]["weight"].shape, (2, 2))
        self.assertGreater(metadata["classifier_original_test_accuracy"], 0.9)
        self.assertEqual(stats["support_count"][3].item(), 0)
        self.assertEqual(stats["support_count"][0].item(), 4)
        self.assertEqual(stats["support_count"][1].item(), 4)
        sae, _ = load_sae(self.sae_run)
        head = nn.Linear(2, 2)
        head.load_state_dict(classifier["state_dict"])
        test = torch.load(self.source / "out/embeddings_test.pt", weights_only=True)
        labels = torch.load(self.source / "out/labels_test.pt", weights_only=True)
        with torch.inference_mode():
            codes = sae.encode(test)[1]
            baseline = head(sae.decode(codes)).argmax(dim=1)
            for row in rows:
                modified = codes.clone()
                modified[:, [int(latent) for latent in row["latents"].split("|")]] = 0
                ablated = head(sae.decode(modified)).argmax(dim=1)
                expected_drop = ((baseline == labels).float().mean()
                                 - (ablated == labels).float().mean()).item()
                self.assertAlmostEqual(expected_drop, float(row["overall_accuracy_drop"]))
        anchor_ids = stats["anchor_ids"].tolist()
        similarity = stats["similarity"]
        for column, anchor in enumerate(anchor_ids):
            self.assertAlmostEqual(similarity[anchor, column].item(), 1.0)
            for other_column, other in enumerate(anchor_ids):
                self.assertAlmostEqual(similarity[other, column].item(),
                                       similarity[anchor, other_column].item())
            group = [row for row in rows if int(row["anchor"]) == anchor]
            self.assertEqual([int(row["size"]) for row in group], [1, 2, 3])
            self.assertEqual(len({row["target_class"] for row in group}), 1)
            self.assertEqual([row["latents"].split("|")[0] for row in group], [str(anchor)] * 3)
            self.assertTrue(group[1]["latents"].startswith(group[0]["latents"] + "|"))
            self.assertTrue(group[2]["latents"].startswith(group[1]["latents"] + "|"))
            self.assertEqual([int(row["support_size"]) for row in group],
                             sorted(int(row["support_size"]) for row in group))
            for row in group:
                self.assertAlmostEqual(float(row["total_variance"]),
                                       float(row["mean_squared_distance_to_centroid"]))
                self.assertAlmostEqual(float(row["mean_per_dimension_variance"]) * 2,
                                       float(row["total_variance"]))
        for summary in summaries:
            if summary["population"] == "nonzero_size_1":
                self.assertEqual(int(summary["anchors"]), metadata["nonzero_size_1_anchors"])

    def test_invalid_configuration_fails_before_loading(self):
        with self.assertRaisesRegex(ValueError, "tversky_penalty"):
            run_steering(SteeringConfig(sae_log_dir=str(self.sae_run), tversky_penalty=0))

    def test_topk_and_batch_topk_checkpoints(self):
        train = torch.load(self.source / "out/embeddings_train.pt", weights_only=True)
        for name in ("topk", "batch_topk"):
            with self.subTest(name=name):
                architecture = dict(name=name, input_dim=2, nb_concepts=4, top_k=2,
                                    threshold_momentum=0.9)
                sae = build_sae(**architecture)
                sae.load_state_dict(self.base_state, strict=False)
                if name == "batch_topk":
                    sae.train()
                    sae.encode(train)
                torch.save({"format_version": 1, "architecture": architecture,
                            "source_model_relative_path": "../..",
                            "source_model_log_dir": str(self.source),
                            "state_dict": sae.state_dict()}, self.sae_run / "model.pt")
                output = run_steering(SteeringConfig(
                    sae_log_dir=str(self.sae_run), max_anchors=2,
                    max_neighborhood_size=2, classifier_epochs=2,
                    classifier_batch_size=4, evaluation_batch_size=3, device="cpu"))
                self.assertTrue((output / "neighborhoods.csv").is_file())


if __name__ == "__main__":
    unittest.main()
