"""SAE behavior, persistence, training, and image-to-code integration tests."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from cfgs.cfg import ExperimentConfig, AugmentConfig
from cfgs.cfg_sae import SAEConfig
from src.augments import build_transforms
from src.models import build_model
from src.models_sae import build_sae, load_sae
from src.trainer_sae import SAETrainer, run_sae_experiment
from test_sae import iter_hidden_embeddings, resolve_sae_run


class SAEPipelineTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "out").mkdir()

    def test_sparsity_and_threshold_roundtrip(self):
        x = torch.randn(7, 8)
        for name in ("basic", "topk", "batch_topk"):
            with self.subTest(name=name):
                model = build_sae(name, 8, 16, top_k=3)
                _, codes, reconstructed = model(x)
                self.assertEqual(codes.shape, (7, 16))
                self.assertEqual(reconstructed.shape, x.shape)
                self.assertTrue((codes >= 0).all())
                if name == "topk":
                    self.assertTrue(((codes != 0).sum(dim=1) <= 3).all())
                if name == "batch_topk":
                    self.assertEqual((codes != 0).sum().item(), 7 * 3)
                    _, short_codes = model.encode(x[:1])
                    self.assertEqual((short_codes != 0).sum().item(), 3)
                    self.assertIn("running_threshold", model.state_dict())
                model.eval()
                restored = build_sae(name, 8, 16, top_k=3)
                restored.load_state_dict(model.state_dict())
                restored.eval()
                torch.testing.assert_close(model.encode(x)[1], restored.encode(x)[1])
                together = restored.encode(x)[1]
                separate = torch.cat([restored.encode(row[None])[1] for row in x])
                torch.testing.assert_close(together, separate)

    def test_concept_expansion_uses_feature_width(self):
        cfg = SAEConfig(model_log_dir=str(self.root), concept_expansion_factor=2,
                        model_name="topk", top_k=3)
        self.assertEqual(cfg.concept_count(512), 1024)
        self.assertEqual(cfg.concept_count(8), 16)
        for factor in (0, -1, 2.5, True, "2"):
            with self.subTest(factor=factor), self.assertRaisesRegex(
                    ValueError, "concept_expansion_factor"):
                SAEConfig(model_log_dir=str(self.root),
                          concept_expansion_factor=factor).validate()
        cfg.top_k = 17
        with self.assertRaisesRegex(ValueError, "top_k"):
            cfg.concept_count(8)

    def test_epoch_r2_and_dead_neurons_use_all_samples(self):
        module = SAETrainer(SAEConfig(model_log_dir=str(self.root),
                                      concept_expansion_factor=2), input_dim=2)
        module.train_totals[4] = 2
        module.train_totals[5] = 2
        module.train_input_sum[:] = torch.tensor([2, 4])
        module.train_input_squared[:] = torch.tensor([4, 16])
        module.train_active_counts[:] = torch.tensor([1, 0, 2, 0])
        r2, dead = module._quality_metrics("train", module.train_totals)
        self.assertAlmostEqual(r2.item(), 0.8)
        self.assertEqual(dead.item(), 2)

    def test_all_variants_train_save_load(self):
        torch.save(torch.randn(13, 8), self.root / "out/embeddings_train.pt")
        torch.save(torch.randn(3, 7), self.root / "out/embeddings_test.pt")
        for name in ("basic", "topk", "batch_topk"):
            with self.subTest(name=name):
                cfg = SAEConfig(model_log_dir=str(self.root), model_name=name,
                                concept_expansion_factor=2, top_k=3, epochs=5, batch_size=5,
                                accelerator="cpu")
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    run = run_sae_experiment(cfg)
                self.assertEqual(len(output.getvalue().strip().splitlines()), 1)
                self.assertIn("Epoch 005", output.getvalue())
                self.assertIn("r2=", output.getvalue())
                self.assertIn("dead=", output.getvalue())
                self.assertEqual(run.parent, self.root / "SAE")
                loaded, artifact = load_sae(run)
                self.assertEqual(artifact["architecture"]["name"], name)
                self.assertEqual(artifact["architecture"]["input_dim"], 8)
                self.assertEqual(artifact["architecture"]["nb_concepts"], 16)
                metadata = json.loads((run / "config.json").read_text())
                self.assertEqual(metadata["concept_expansion_factor"], 2)
                self.assertEqual(metadata["nb_concepts"], 16)
                self.assertTrue((run / "metrics.csv").is_file())
                with (run / "metrics.csv").open(newline="") as handle:
                    metrics = list(csv.DictReader(handle))
                for key in ("train_r2", "eval_r2", "train_dead_neurons", "eval_dead_neurons"):
                    self.assertEqual(sum(bool(row[key]) for row in metrics), 5)
                split = torch.load(run / "split_indices.pt", weights_only=True)
                self.assertEqual(sorted(torch.cat(list(split.values())).tolist()), list(range(13)))
                checkpoint = torch.load(run / "last.ckpt", weights_only=True)
                for key, tensor in loaded.state_dict().items():
                    torch.testing.assert_close(tensor, checkpoint["state_dict"]["model." + key])
                _, _, reconstructed = loaded(torch.randn(2, 8))
                self.assertTrue(torch.isfinite(reconstructed).all())

    def test_images_use_source_encoder_and_preprocessing(self):
        cfg = ExperimentConfig(root_dir=self.root, model_name="resnet18", pretrained=False,
                               augments=AugmentConfig(image_size=32))
        (self.root / "config.json").write_text(json.dumps(cfg.to_dict()))
        (self.root / "checkpoints").mkdir()
        encoder = build_model("resnet18", 2, pretrained=False).eval()
        torch.save({"state_dict": {"model." + k: v for k, v in encoder.state_dict().items()},
                    "hyper_parameters": {"num_classes": 2}}, self.root / "checkpoints/last.ckpt")
        images_dir = self.root / "images"
        images_dir.mkdir()
        image = Image.new("RGB", (40, 50), (20, 100, 200))
        image.save(images_dir / "sample.png")
        torch.save(torch.randn(5, 512), self.root / "out/embeddings_train.pt")
        run = run_sae_experiment(SAEConfig(model_log_dir=str(self.root), model_name="batch_topk",
                                           concept_expansion_factor=2, top_k=3, epochs=1, batch_size=3,
                                           accelerator="cpu"))
        sae, _ = load_sae(run)
        self.assertEqual(sae.nb_concepts, 1024)
        with torch.inference_mode():
            features = encoder.forward_features(build_transforms(cfg.augments)(image)[None])
            expected = sae.encode(features)[1][0]
        rows = list(iter_hidden_embeddings(self.root, images_dir, batch_size=1, device="cpu"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], str(images_dir / "sample.png"))
        torch.testing.assert_close(rows[0][1], expected)
        self.assertEqual(resolve_sae_run(self.root), run)
        self.assertEqual(resolve_sae_run(self.root / "SAE"), run)

    def test_invalid_embeddings(self):
        torch.save(torch.tensor([[float("nan"), 0.0], [1.0, 0.0]]),
                   self.root / "out/embeddings_train.pt")
        with self.assertRaisesRegex(ValueError, "NaN"):
            run_sae_experiment(SAEConfig(model_log_dir=str(self.root)))


if __name__ == "__main__":
    unittest.main()
