"""Learning-rate scheduling on evaluation loss."""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from cfgs.cfg import SchedulerConfig


def build_scheduler(optimizer: Optimizer, config: SchedulerConfig) -> dict:
    return {
        "scheduler": ReduceLROnPlateau(
            optimizer, mode="min", factor=config.factor, patience=config.patience,
            threshold=config.threshold, cooldown=config.cooldown, min_lr=config.min_lr,
        ),
        "monitor": "eval_loss",
        "interval": "epoch",
        "frequency": 1,
        "strict": True,
    }
