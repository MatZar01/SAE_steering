"""Small end-to-end checks for PDF point 1 using saved SAE artifacts."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from cfgs.cfg_sae import SteeringConfig
from src.models_sae import build_sae, load_sae
from src.steering import (_active_state_metrics, _coherence_similarity_to_anchors,
                          _decoder_cosine_to_anchors, _random_similarity_to_anchors,
                          run_steering)


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

    def test_active_state_metrics_singleton_and_orthogonal_codes(self):
        codes = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
        support = codes > 0
        columns = {0: 0, 1: 1}
        variance, cosine = _active_state_metrics(codes, support, [0], columns)
        self.assertEqual((variance, cosine), (0.0, None))
        variance, cosine = _active_state_metrics(codes, support, [0, 1], columns)
        self.assertAlmostEqual(variance, 0.625)
        self.assertAlmostEqual(cosine, 0.0)

    def test_coherence_uses_full_image_activity_and_excludes_identical_pairs(self):
        support = torch.eye(3, dtype=torch.bool)
        anchors = torch.tensor([0, 1])
        similarity = _coherence_similarity_to_anchors(
            support, [0b011, 0b011, 0b100], anchors, 0.5, 8, 42)
        self.assertEqual(similarity[1, 0].item(), 1.0)
        self.assertEqual(similarity[0, 1].item(), 1.0)
        self.assertEqual(similarity[2, 0].item(), 0.0)

        partial = _coherence_similarity_to_anchors(
            torch.eye(2, dtype=torch.bool), [0b011, 0b101],
            torch.tensor([0]), 0.5, 8, 42)
        self.assertEqual(partial[1, 0].item(), 0.5)

        overlapping = _coherence_similarity_to_anchors(
            torch.ones((2, 2), dtype=torch.bool), [0b01, 0b10],
            torch.tensor([0]), 0.5, 8, 42)
        self.assertEqual(overlapping[1, 0].item(), 0.0)

        shared_only = torch.tensor([[True, True], [False, False]])
        similarity = _coherence_similarity_to_anchors(
            shared_only, [0b11, 0], torch.tensor([0]), 0.5, 8, 42)
        self.assertEqual(similarity[1, 0].item(), 0.0)

    def test_coherence_mode_ranks_and_evaluates_neighborhoods(self):
        config = SteeringConfig(sae_log_dir=str(self.sae_run), max_anchors=None,
                                max_neighborhood_size=3, tversky_mode="coherence",
                                coherence_pairs=12, classifier_epochs=2,
                                classifier_batch_size=4, evaluation_batch_size=3,
                                device="cpu")
        output = run_steering(config)
        with (output / "neighborhoods.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        metadata = json.loads((output / "metadata.json").read_text())
        stats = torch.load(output / "latent_statistics.pt", weights_only=True)
        self.assertEqual(len(rows), 9)
        self.assertEqual(stats["tversky_mode"], "coherence")
        self.assertEqual(metadata["config"]["coherence_pairs"], 12)
        self.assertIn("distinct training-image pairs", metadata["similarity"])
        for column, anchor in enumerate(stats["anchor_ids"].tolist()):
            expected = [anchor] + sorted(
                (latent for latent in range(4) if latent != anchor and stats["support_count"][latent] > 0),
                key=lambda latent: (-stats["similarity"][latent, column].item(), latent))
            actual = [int(latent) for latent in next(
                row for row in rows if int(row["anchor"]) == anchor and int(row["size"]) == 3
            )["latents"].split("|")]
            self.assertEqual(actual, expected)

    def test_decoder_cosine_uses_decoder_rows(self):
        dictionary = torch.tensor([[1.0, 0.0], [0.0, 2.0],
                                   [-3.0, 0.0], [0.0, 0.0]])
        similarity = _decoder_cosine_to_anchors(dictionary, torch.tensor([0, 2]))
        torch.testing.assert_close(similarity, torch.tensor([
            [1.0, -1.0], [0.0, 0.0], [-1.0, 1.0], [0.0, 0.0],
        ]))

    def test_random_similarity_is_seeded_and_symmetric(self):
        anchors = torch.tensor([0, 2, 3])
        similarity = _random_similarity_to_anchors(5, anchors, 42)
        torch.testing.assert_close(similarity,
                                   _random_similarity_to_anchors(5, anchors, 42))
        self.assertFalse(torch.equal(similarity,
                                     _random_similarity_to_anchors(5, anchors, 43)))
        self.assertTrue(((similarity >= 0) & (similarity <= 1)).all())
        for column, anchor in enumerate(anchors.tolist()):
            self.assertEqual(similarity[anchor, column].item(), 1.0)
            for other_column, other in enumerate(anchors.tolist()):
                self.assertEqual(similarity[other, column].item(),
                                 similarity[anchor, other_column].item())

    def test_cosine_and_random_modes_run_complete_steering(self):
        for mode in ("decoder_cosine", "random"):
            with self.subTest(mode=mode):
                config = SteeringConfig(
                    sae_log_dir=str(self.sae_run), max_anchors=None,
                    max_neighborhood_size=3, tversky_mode=mode,
                    classifier_epochs=2, classifier_batch_size=4,
                    evaluation_batch_size=3, device="cpu")
                output = run_steering(config)
                with (output / "neighborhoods.csv").open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                stats = torch.load(output / "latent_statistics.pt", weights_only=True)
                metadata = json.loads((output / "metadata.json").read_text())
                self.assertEqual(len(rows), 9)
                self.assertEqual(stats["tversky_mode"], mode)
                self.assertEqual(metadata["config"]["tversky_mode"], mode)
                if mode == "decoder_cosine":
                    sae, _ = load_sae(self.sae_run)
                    directions = F.normalize(sae.get_dictionary(), dim=1)
                    expected = directions @ directions[stats["anchor_ids"]].T
                    torch.testing.assert_close(stats["similarity"], expected)
                    self.assertIn("decoder weight vectors", metadata["similarity"])
                else:
                    expected = _random_similarity_to_anchors(
                        4, stats["anchor_ids"], config.seed)
                    torch.testing.assert_close(stats["similarity"], expected)
                    self.assertIn("uniform random", metadata["similarity"])
                for column, anchor in enumerate(stats["anchor_ids"].tolist()):
                    ranked = sorted(
                        (latent for latent in range(4) if latent != anchor
                         and stats["support_count"][latent] > 0),
                        key=lambda latent: (-stats["similarity"][latent, column].item(), latent))
                    actual = [int(latent) for latent in next(
                        row for row in rows if int(row["anchor"]) == anchor
                        and int(row["size"]) == 3
                    )["latents"].split("|")]
                    self.assertEqual(actual, [anchor] + ranked)

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
        train = torch.load(self.source / "out/embeddings_train.pt", weights_only=True)
        with torch.inference_mode():
            train_codes = sae.encode(train)[1]
        mean_code = train_codes.double().mean(dim=0)
        for row in rows:
            latents = [int(latent) for latent in row["latents"].split("|")]
            active = (train_codes[:, latents] > mean_code[latents]).any(dim=1)
            states = train_codes[active][:, latents].double()
            expected_variance = states.var(dim=0, unbiased=False).mean().item()
            self.assertAlmostEqual(float(row["active_state_variance"]), expected_variance)
            if len(states) >= 2:
                unit = F.normalize(states, dim=1)
                pairwise = unit @ unit.T
                expected_cosine = pairwise.triu(diagonal=1).sum().item() * 2 / (len(states) * (len(states) - 1))
                self.assertAlmostEqual(float(row["active_state_cosine_similarity"]), expected_cosine)
            else:
                self.assertEqual(row["active_state_cosine_similarity"], "")
        head = nn.Linear(2, 2)
        head.load_state_dict(classifier["state_dict"])
        test = torch.load(self.source / "out/embeddings_test.pt", weights_only=True)
        labels = torch.load(self.source / "out/labels_test.pt", weights_only=True)
        with torch.inference_mode():
            codes = sae.encode(test)[1]
            baseline_logits = head(sae.decode(codes))
            baseline = baseline_logits.argmax(dim=1)
            baseline_true_probability = baseline_logits.softmax(dim=1).gather(
                1, labels[:, None]).squeeze(1)
            for row in rows:
                modified = codes.clone()
                modified[:, [int(latent) for latent in row["latents"].split("|")]] = 0
                ablated_logits = head(sae.decode(modified))
                ablated = ablated_logits.argmax(dim=1)
                expected_drop = ((baseline == labels).float().mean()
                                 - (ablated == labels).float().mean()).item()
                self.assertAlmostEqual(expected_drop, float(row["overall_accuracy_drop"]))
                ablated_true_probability = ablated_logits.softmax(dim=1).gather(
                    1, labels[:, None]).squeeze(1)
                probability_change = baseline_true_probability - ablated_true_probability
                self.assertAlmostEqual(probability_change.mean().item(),
                                       float(row["overall_probability_drop"]), places=6)
                target = int(row["target_class_index"])
                self.assertAlmostEqual(probability_change[labels == target].mean().item(),
                                       float(row["target_probability_drop"]), places=6)
                self.assertAlmostEqual(probability_change[labels != target].mean().item(),
                                       float(row["non_target_mean_probability_drop"]), places=6)
                class_drops = json.loads(row["class_probability_drops"])
                for label_name, index in (("left", 0), ("right", 1)):
                    self.assertAlmostEqual(probability_change[labels == index].mean().item(),
                                           class_drops[label_name], places=6)
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
            if summary["population"] == "all":
                group = [row for row in rows if int(row["size"]) == int(summary["size"])]
                expected = sum(float(row["active_state_variance"]) for row in group) / len(group)
                self.assertAlmostEqual(float(summary["mean_active_state_variance"]), expected)
                expected_probability = sum(float(row["overall_probability_drop"])
                                           for row in group) / len(group)
                self.assertAlmostEqual(float(summary["mean_overall_probability_drop"]),
                                       expected_probability)
        self.assertIn("SAE code vector", metadata["active_state_metrics"])
        self.assertIn("softmax probability", metadata["probability_drop_definition"])

    def test_invalid_configuration_fails_before_loading(self):
        with self.assertRaisesRegex(ValueError, "tversky_penalty"):
            run_steering(SteeringConfig(sae_log_dir=str(self.sae_run), tversky_penalty=0))
        with self.assertRaisesRegex(ValueError, "tversky_mode"):
            SteeringConfig(sae_log_dir=str(self.sae_run), tversky_mode="unknown").validate()
        with self.assertRaisesRegex(ValueError, "coherence_pairs"):
            SteeringConfig(sae_log_dir=str(self.sae_run), coherence_pairs=0).validate()

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
