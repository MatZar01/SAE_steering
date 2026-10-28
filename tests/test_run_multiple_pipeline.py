"""YAML experiment discovery and sequential pipeline handoff."""

import tempfile
import unittest
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import run_multiple_pipeline as multiple
import run_pipeline as pipeline


class RunMultiplePipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def test_default_yaml_matches_current_config_defaults(self):
        path, overrides = next((path, overrides) for path, overrides in
                               multiple.load_experiments(multiple.EXPERIMENTS_DIR)
                               if path.name == "default.yaml")
        self.assertEqual(path.name, "default.yaml")
        for stage, template in (("image", pipeline.IMAGE_CONFIG),
                                ("sae", pipeline.SAE_CONFIG),
                                ("steering", pipeline.STEERING_CONFIG),
                                ("inference", pipeline.INFERENCE_CONFIG)):
            configured = deepcopy(template)
            pipeline._apply_overrides(configured, overrides[stage], stage)
            self.assertEqual(asdict(configured), asdict(template), stage)

    def test_runs_yaml_files_in_sorted_order_and_passes_overrides(self):
        (self.directory / "b.yml").write_text("sae:\n  model_name: topk\n")
        (self.directory / "a.yaml").write_text("image:\n  epochs: 2\n")
        (self.directory / "ignored.txt").write_text("image:\n  epochs: 3\n")
        calls = []

        def fake_pipeline(overrides):
            calls.append(overrides)
            return {"image": self.directory / f"run_{len(calls)}"}

        with patch.object(multiple, "run_pipeline", side_effect=fake_pipeline):
            result = multiple.run_multiple_pipeline(self.directory)
        self.assertEqual(calls, [{"image": {"epochs": 2}},
                                 {"sae": {"model_name": "topk"}}])
        self.assertEqual(list(result), ["a.yaml", "b.yml"])
        self.assertEqual(result["b.yml"]["image"], self.directory / "run_2")

    def test_malformed_later_yaml_fails_before_any_pipeline_starts(self):
        (self.directory / "a.yaml").write_text("image: {}\n")
        (self.directory / "b.yaml").write_text("image: [\n")
        with patch.object(multiple, "run_pipeline") as run:
            with self.assertRaisesRegex(ValueError, "Invalid YAML in .*b.yaml"):
                multiple.run_multiple_pipeline(self.directory)
            run.assert_not_called()

    def test_invalid_later_settings_fail_before_any_pipeline_starts(self):
        (self.directory / "a.yaml").write_text("image: {}\n")
        (self.directory / "b.yaml").write_text("sae:\n  reanimation_top_k: 0\n")
        with patch.object(multiple, "run_pipeline") as run:
            with self.assertRaisesRegex(ValueError, "Invalid experiment config .*b.yaml.*reanimation_top_k"):
                multiple.run_multiple_pipeline(self.directory)
            run.assert_not_called()

    def test_requires_mapping_and_at_least_one_config(self):
        with self.assertRaisesRegex(ValueError, "No .yaml or .yml"):
            multiple.load_experiments(self.directory)
        (self.directory / "bad.yaml").write_text("- one\n- two\n")
        with self.assertRaisesRegex(ValueError, "must be a YAML mapping"):
            multiple.load_experiments(self.directory)


if __name__ == "__main__":
    unittest.main()
