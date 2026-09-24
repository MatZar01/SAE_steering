"""Configuration for training an SAE on an image experiment's embeddings."""

from dataclasses import asdict, dataclass, field
from pathlib import Path

from cfgs.cfg import ROOT_DIR, SchedulerConfig


@dataclass
class SAEConfig:
    root_dir: Path = ROOT_DIR
    # Set this to the image experiment containing out/embeddings.pt.
    model_log_dir: str = ""
    model_name: str = "basic"  # basic, topk, batch_topk
    nb_concepts: int = 2048
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

    def validate(self):
        self.source_dir()
        if self.model_name not in {"basic", "topk", "batch_topk"}:
            raise ValueError("model_name must be basic, topk, or batch_topk.")
        if self.nb_concepts < 1 or (self.model_name != "basic" and not 1 <= self.top_k <= self.nb_concepts):
            raise ValueError("Require nb_concepts >= 1 and 1 <= top_k <= nb_concepts.")
        if self.epochs < 1 or self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("Require positive epochs/batch_size and nonnegative num_workers.")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1.")
        if not 0 <= self.threshold_momentum < 1 or self.l1_coefficient < 0:
            raise ValueError("Require momentum in [0, 1) and nonnegative L1 coefficient.")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.print_every_n_epochs < 1:
            raise ValueError("Invalid learning rate, weight decay, or print interval.")


CONFIG = SAEConfig()
