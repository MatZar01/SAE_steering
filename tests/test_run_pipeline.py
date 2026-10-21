"""Verify config dictionaries and exact run handoffs between stages."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_pipeline as pipeline


class RunPipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_overrides_and_run_paths_flow_through_all_stages(self):
        original_image_size = pipeline.IMAGE_CONFIG.augments.image_size
        original_sae_source = pipeline.SAE_CONFIG.model_log_dir
        original_steering_source = pipeline.STEERING_CONFIG.sae_log_dir
        original_inference_source = pipeline.INFERENCE_CONFIG.sae_log_dir
        image_run = self.root / "experiments" / "image_run"
        sae_run = image_run / "SAE" / "sae_run"
        steering_run = sae_run / "steering" / "steering_run"
        active_images = sae_run / "active_images"
        events = []

        def image_stage(config, sae_model_name):
            events.append("image")
            self.assertEqual(config.model_name, "dino")
            self.assertEqual(sae_model_name, "topk")
            self.assertEqual(config.augments.image_size, 384)
            self.assertEqual(config.scheduler.patience, 2)
            self.assertEqual(config.log_dir, "experiments")
            return image_run

        def sae_stage(config):
            events.append("sae")
            self.assertEqual(config.model_name, "topk")
            self.assertEqual(config.concept_expansion_factor, 2)
            self.assertEqual(config.top_k, 4)
            self.assertEqual(config.model_log_dir, str(image_run))
            return sae_run

        def steering_stage(config):
            events.append("steering")
            self.assertEqual(config.max_anchors, 5)
            self.assertEqual(config.max_neighborhood_size, 3)
            self.assertEqual(config.sae_log_dir, str(sae_run))
            self.assertEqual(config.model_log_dir, str(image_run))
            return steering_run

        def inference_stage(config):
            events.append("inference")
            self.assertEqual(config.top_active_images, 2)
            self.assertFalse(config.stitch_images)
            self.assertEqual(config.sae_log_dir, str(sae_run))
            self.assertEqual(config.steering_log_dir, str(steering_run))
            self.assertEqual(config.model_log_dir, str(image_run))
            return active_images

        overrides = {
            "image": {"root_dir": self.root, "log_dir": "experiments",
                      "model_name": "dino", "augments": {"image_size": 384},
                      "scheduler": {"patience": 2}},
            "sae": {"model_name": "topk", "concept_expansion_factor": 2, "top_k": 4},
            "steering": {"max_anchors": 5, "max_neighborhood_size": 3},
            "inference": {"top_active_images": 2, "stitch_images": False},
        }
        with (patch.object(pipeline, "run_experiment", side_effect=image_stage),
              patch.object(pipeline, "run_sae_experiment", side_effect=sae_stage),
              patch.object(pipeline, "run_steering", side_effect=steering_stage),
              patch.object(pipeline, "run_sae_inference", side_effect=inference_stage)):
            runs = pipeline.run_pipeline(overrides)
        self.assertEqual(events, ["image", "sae", "steering", "inference"])
        self.assertEqual(runs, {"image": image_run, "sae": sae_run,
                                "steering": steering_run, "inference": active_images})
        self.assertEqual(pipeline.IMAGE_CONFIG.augments.image_size, original_image_size)
        self.assertEqual(pipeline.SAE_CONFIG.model_log_dir, original_sae_source)
        self.assertEqual(pipeline.STEERING_CONFIG.sae_log_dir, original_steering_source)
        self.assertEqual(pipeline.INFERENCE_CONFIG.sae_log_dir, original_inference_source)

    def test_invalid_override_fails_before_training(self):
        with patch.object(pipeline, "run_experiment") as image_stage:
            with self.assertRaisesRegex(ValueError, "Unknown image.augments config field"):
                pipeline.run_pipeline({"image": {"augments": {"imag_size": 32}}})
            with self.assertRaisesRegex(ValueError, "set automatically"):
                pipeline.run_pipeline({"sae": {"model_log_dir": "another_run"}})
            with self.assertRaisesRegex(ValueError, "set automatically"):
                pipeline.run_pipeline({"steering": {"sae_log_dir": "another_run"}})
            with self.assertRaisesRegex(ValueError, "set automatically"):
                pipeline.run_pipeline({"inference": {"steering_log_dir": "another_run"}})
            with self.assertRaisesRegex(ValueError, "top_active_images"):
                pipeline.run_pipeline({"inference": {"top_active_images": 0}})
            image_stage.assert_not_called()


if __name__ == "__main__":
    unittest.main()
