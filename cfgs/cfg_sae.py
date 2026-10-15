"""Configuration for SAE training and neighborhood steering evaluation."""

from dataclasses import asdict, dataclass, field
from pathlib import Path

from cfgs.cfg import ROOT_DIR, SchedulerConfig


@dataclass
class SAEConfig:
    root_dir: Path = ROOT_DIR
    # Set this to the image experiment containing out/embeddings_train.pt.
    model_log_dir: str = ""
    model_name: str = "basic"  # basic, topk, batch_topk
    concept_expansion_factor: int = 4
    top_k: int = 32  # Per sample; batch_topk uses batch_size * top_k globally.
    l1_coefficient: float = 1e-3  # Applied only to basic SAE.
    threshold_momentum: float = 0.9
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    validation_fraction: float = 0.1
    num_workers: int = 0
    seed: int = 42
    accelerator: str = "auto"
    precision: str = "32-true"
    deterministic: bool = True
    print_every_n_epochs: int = 5
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    def source_dir(self) -> Path:
        if not self.model_log_dir:
            raise ValueError("Set model_log_dir in cfgs/cfg_sae.py or pass --model-log-dir.")
        return (self.root_dir / self.model_log_dir).resolve()

    def to_dict(self) -> dict:
        result = asdict(self)
        result["root_dir"] = str(self.root_dir.resolve())
        result["model_log_dir"] = str(self.source_dir())
        return result

    def concept_count(self, input_dim: int) -> int:
        """Resolve the SAE width from the exported feature dimension."""
        if type(input_dim) is not int or input_dim < 1:
            raise ValueError("input_dim must be a positive integer.")
        if type(self.concept_expansion_factor) is not int or self.concept_expansion_factor < 1:
            raise ValueError("concept_expansion_factor must be a positive integer.")
        nb_concepts = self.concept_expansion_factor * input_dim
        if self.model_name != "basic" and self.top_k > nb_concepts:
            raise ValueError("top_k must not exceed concept_expansion_factor * input_dim.")
        return nb_concepts

    def validate(self):
        self.source_dir()
        if self.model_name not in {"basic", "topk", "batch_topk"}:
            raise ValueError("model_name must be basic, topk, or batch_topk.")
        if type(self.concept_expansion_factor) is not int or self.concept_expansion_factor < 1:
            raise ValueError("concept_expansion_factor must be a positive integer.")
        if self.model_name != "basic" and (type(self.top_k) is not int or self.top_k < 1):
            raise ValueError("top_k must be a positive integer.")
        if self.epochs < 1 or self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("Require positive epochs/batch_size and nonnegative num_workers.")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1.")
        if not 0 <= self.threshold_momentum < 1 or self.l1_coefficient < 0:
            raise ValueError("Require momentum in [0, 1) and nonnegative L1 coefficient.")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.print_every_n_epochs < 1:
            raise ValueError("Invalid learning rate, weight decay, or print interval.")


CONFIG = SAEConfig()


@dataclass
class SteeringConfig:
    """Point 1 settings for a saved SAE and its source image experiment."""

    root_dir: Path = ROOT_DIR
    # Exact SAE run, its SAE directory, or the source image experiment.
    sae_log_dir: str = ""
    # Optional override when the source image experiment has been moved.
    model_log_dir: str = ""
    max_anchors: int | None = 128  # None evaluates every latent with nonempty support.
    max_neighborhood_size: int = 8  # Every size from 1 through this value is evaluated.
    tversky_penalty: float = 0.5  # Equal penalty for each one-sided support difference.
    classifier_epochs: int = 100
    classifier_batch_size: int = 1024
    classifier_learning_rate: float = 1e-3
    classifier_weight_decay: float = 1e-4
    evaluation_batch_size: int = 512
    seed: int = 42
    device: str = "auto"

    def resolve(self, path: str) -> Path:
        return (self.root_dir / path).resolve()

    def validate(self) -> None:
        if not self.sae_log_dir:
            raise ValueError("Set sae_log_dir in cfgs/cfg_sae.py or pass --sae-log-dir.")
        if self.max_anchors is not None and self.max_anchors < 1:
            raise ValueError("max_anchors must be positive or None.")
        if self.max_neighborhood_size < 1 or self.evaluation_batch_size < 1:
            raise ValueError("Neighborhood size and evaluation batch size must be positive.")
        if not 0 < self.tversky_penalty <= 1:
            raise ValueError("tversky_penalty must be in (0, 1].")
        if self.classifier_epochs < 1 or self.classifier_batch_size < 1:
            raise ValueError("Classifier epochs and batch size must be positive.")
        if self.classifier_learning_rate <= 0 or self.classifier_weight_decay < 0:
            raise ValueError("Invalid classifier learning rate or weight decay.")

    def to_dict(self) -> dict:
        result = asdict(self)
        result["root_dir"] = str(self.root_dir.resolve())
        result["sae_log_dir"] = str(self.resolve(self.sae_log_dir))
        if self.model_log_dir:
            result["model_log_dir"] = str(self.resolve(self.model_log_dir))
        return result


STEERING_CONFIG = SteeringConfig()


@dataclass
class SAEInferenceConfig:
    """Rank source images by SAE latent and discovered-group activation."""

    root_dir: Path = ROOT_DIR
    sae_log_dir: str = ""
    steering_log_dir: str = ""  # Exact steering run; latest run is used when empty.
    model_log_dir: str = ""  # Optional override for a moved image experiment.
    top_active_images: int = 3  # Grid side length; 3 selects up to 9 images.
    batch_size: int = 512
    device: str = "auto"

    def resolve(self, path: str) -> Path:
        return (self.root_dir / path).resolve()

    def validate(self) -> None:
        if not self.sae_log_dir:
            raise ValueError("Set sae_log_dir in cfgs/cfg_sae.py or pass --sae-log-dir.")
        if type(self.top_active_images) is not int or self.top_active_images < 1:
            raise ValueError("top_active_images must be a positive integer.")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")


INFERENCE_CONFIG = SAEInferenceConfig()
