"""Run image training, SAE training, steering, and active-image inference.

Edit PIPELINE_OVERRIDES to change any stage's dataclass settings. Nested settings
such as image augmentations and schedulers are dictionaries too.
"""

from copy import deepcopy
from dataclasses import fields, is_dataclass
from pathlib import Path

from cfgs.cfg import CONFIG as IMAGE_CONFIG
from cfgs.cfg_sae import CONFIG as SAE_CONFIG
from cfgs.cfg_sae import INFERENCE_CONFIG, STEERING_CONFIG
from src.sae_inference import run_sae_inference
from src.steering import run_steering
from src.trainer import run_experiment
from src.trainer_sae import run_sae_experiment


PIPELINE_OVERRIDES = {
    "image": {
        # "model_name": "dino",
        # "epochs": 20,
        # "augments": {"image_size": 224},
        # "scheduler": {"patience": 2},
    },
    "sae": {
        # "model_name": "topk",
        # "concept_expansion_factor": 4,
        # "top_k": 32,
        # "epochs": 30,
    },
    "steering": {
        # "max_anchors": 128,
        # "max_neighborhood_size": 16,
    },
    "inference": {
        # "top_active_images": 3,
        # "batch_size": 512,
    },
}

_LINKED_PATHS = {
    "sae": {"model_log_dir"},
    "steering": {"sae_log_dir", "model_log_dir"},
    "inference": {"sae_log_dir", "steering_log_dir", "model_log_dir"},
}


def _apply_overrides(config, overrides: dict, stage: str) -> None:
    if not isinstance(overrides, dict):
        raise TypeError(f"{stage} overrides must be a dictionary.")
    available = {field.name for field in fields(config)}
    for name, value in overrides.items():
        if name not in available:
            raise ValueError(f"Unknown {stage} config field {name!r}.")
        if name in _LINKED_PATHS.get(stage, set()):
            raise ValueError(f"{stage}.{name} is set automatically by the pipeline.")
        current = getattr(config, name)
        if is_dataclass(current):
            _apply_overrides(current, value, f"{stage}.{name}")
        elif isinstance(value, dict):
            raise TypeError(f"{stage}.{name} is not a nested config.")
        else:
            setattr(config, name, value)


def run_pipeline(overrides: dict | None = None) -> dict[str, Path]:
    """Return the exact run directories created by the four stages."""
    overrides = PIPELINE_OVERRIDES if overrides is None else overrides
    if not isinstance(overrides, dict):
        raise TypeError("Pipeline overrides must be a dictionary.")
    unknown = set(overrides) - {"image", "sae", "steering", "inference"}
    if unknown:
        raise ValueError(f"Unknown pipeline stage(s): {sorted(unknown)}.")

    image = deepcopy(IMAGE_CONFIG)
    sae = deepcopy(SAE_CONFIG)
    steering = deepcopy(STEERING_CONFIG)
    inference = deepcopy(INFERENCE_CONFIG)
    for stage, config in (("image", image), ("sae", sae), ("steering", steering),
                          ("inference", inference)):
        _apply_overrides(config, overrides.get(stage, {}), stage)

    # Validate all settings before the first potentially long training run.
    image.validate()
    sae.model_log_dir = str(image.resolve(image.log_dir))
    sae.validate()
    steering.sae_log_dir = str(image.resolve(image.log_dir))
    steering.validate()
    inference.sae_log_dir = str(image.resolve(image.log_dir))
    inference.validate()

    image_run = run_experiment(image, sae_model_name=sae.model_name)
    print(f"Image run: {image_run}", flush=True)

    sae.model_log_dir = str(image_run)
    sae_run = run_sae_experiment(sae)
    print(f"SAE run: {sae_run}", flush=True)

    steering.sae_log_dir = str(sae_run)
    steering.model_log_dir = str(image_run)
    steering_run = run_steering(steering)
    print(f"Steering run: {steering_run}", flush=True)

    inference.sae_log_dir = str(sae_run)
    inference.steering_log_dir = str(steering_run)
    inference.model_log_dir = str(image_run)
    active_images = run_sae_inference(inference)
    print(f"Active images: {active_images}", flush=True)
    return {"image": image_run, "sae": sae_run, "steering": steering_run,
            "inference": active_images}


if __name__ == "__main__":
    run_pipeline()
