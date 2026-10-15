"""Edit CONFIG below to configure the default experiment."""

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass
class AugmentConfig:
    image_size: int = 224
    crop_scale: tuple[float, float] = (0.8, 1.0)
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.0
    color_jitter: float = 0.2
    rotation_degrees: float = 10.0
    gaussian_blur_probability: float = 0.2
    gaussian_blur_kernel: int = 3
    gaussian_blur_sigma: tuple[float, float] = (0.1, 2.0)
    masking_probability: float = 0.25
    masking_scale: tuple[float, float] = (0.02, 0.15)


@dataclass
class SchedulerConfig:
    factor: float = 0.5
    patience: int = 3
    threshold: float = 1e-4
    cooldown: int = 0
    min_lr: float = 1e-7


@dataclass
class ExperimentConfig:
    root_dir: Path = ROOT_DIR
    dataset_path: str = "DS/cifar10"
    log_dir: str = "log_dir"
    model_name: str = "resnet18"  # resnet18, blip, dino
    pretrained: bool = True
    epochs: int = 30
    batch_size: int = 64
    export_batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    accelerator: str = "auto"
    precision: str = "32-true"
    deterministic: bool = True
    train_batches: int = 1024
    val_batches: int = 1024
    sae_smples: float | None = None  # Percent per split (e.g. 25.0); None embeds all images.
    print_every_n_epochs: int = 5
    augments: AugmentConfig = field(default_factory=AugmentConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    def resolve(self, path: str) -> Path:
        return (self.root_dir / path).resolve()

    def to_dict(self) -> dict:
        result = asdict(self)
        result["root_dir"] = str(self.root_dir.resolve())
        return result

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 2 or self.export_batch_size < 1 or self.num_workers < 0:
            raise ValueError("epochs >= 1, batch_size >= 2, export_batch_size >= 1 and num_workers >= 0 are required.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative.")
        if self.print_every_n_epochs < 1 or self.augments.image_size < 32:
            raise ValueError("Print interval must be positive; image_size must be at least 32.")
        if self.sae_smples is not None and (
            type(self.sae_smples) not in (int, float)
            or not math.isfinite(self.sae_smples)
            or not 0 < self.sae_smples <= 100
        ):
            raise ValueError("sae_smples must be a percentage in (0, 100] or None.")
        if not 0 < self.scheduler.factor < 1:
            raise ValueError("Scheduler factor must be between 0 and 1.")


CONFIG = ExperimentConfig()
