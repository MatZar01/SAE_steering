"""Lightning training on precomputed embeddings with reproducible validation."""

import json
import logging
from importlib.metadata import version
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset

from cfgs.cfg_sae import SAEConfig
from src.models_sae import build_sae
from src.scheduler import build_scheduler
from src.trainer import create_run_dir


class SAETrainer(pl.LightningModule):
    def __init__(self, config: SAEConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.architecture = dict(name=config.model_name, input_dim=input_dim,
                                 nb_concepts=config.nb_concepts, top_k=config.top_k,
                                 threshold_momentum=config.threshold_momentum)
        self.model = build_sae(**self.architecture)
        self.save_hyperparameters({"config": config.to_dict(), "architecture": self.architecture})
        self.register_buffer("train_totals", torch.zeros(5, dtype=torch.float64), persistent=False)
        self.register_buffer("eval_totals", torch.zeros(5, dtype=torch.float64), persistent=False)

    def forward(self, embeddings):
        return self.model(embeddings)

    def _step(self, batch, totals):
        x = batch[0]
        _, codes, reconstructed = self(x)
        mse = F.mse_loss(reconstructed, x)
        l1 = codes.abs().sum(dim=1).mean()
        loss = mse + (self.config.l1_coefficient * l1 if self.config.model_name == "basic" else 0)
        l0 = (codes.detach() != 0).float().sum(dim=1).mean()
        totals[:4] += torch.stack([loss.detach(), mse.detach(), l1.detach(), l0]).double() * len(x)
        totals[4] += len(x)
        return loss

    def on_train_epoch_start(self):
        self.train_totals.zero_()

    def training_step(self, batch, batch_idx):
        return self._step(batch, self.train_totals)

    def on_validation_epoch_start(self):
        self.eval_totals.zero_()

    def validation_step(self, batch, batch_idx):
        self._step(batch, self.eval_totals)

    def _log_totals(self, name, totals):
        for metric, value in zip(("loss", "mse", "l1", "l0"), totals[:4] / totals[4]):
            self.log(f"{name}_{metric}", value)

    def on_validation_epoch_end(self):
        self._log_totals("eval", self.eval_totals)

    def on_train_epoch_end(self):
        self._log_totals("train", self.train_totals)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"])
        if (self.current_epoch + 1) % self.config.print_every_n_epochs == 0:
            train = (self.train_totals[:4] / self.train_totals[4]).tolist()
            val = (self.eval_totals[:4] / self.eval_totals[4]).tolist()
            print(f"Epoch {self.current_epoch + 1:03d} | train loss={train[0]:.6f} "
                  f"mse={train[1]:.6f} l0={train[3]:.2f} | val loss={val[0]:.6f} "
                  f"mse={val[1]:.6f} l0={val[3]:.2f}", flush=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate,
                                      weight_decay=self.config.weight_decay)
        return {"optimizer": optimizer, "lr_scheduler": build_scheduler(optimizer, self.config.scheduler)}


def run_sae_experiment(config: SAEConfig) -> Path:
    config.validate()
    source = config.source_dir()
    embedding_path = source / "out" / "embeddings_train.pt"
    if not embedding_path.is_file():
        raise FileNotFoundError(f"Missing {embedding_path}. Finish the image experiment's export first.")
    embeddings = torch.load(embedding_path, map_location="cpu", weights_only=True)
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError("embeddings_train.pt must contain a tensor of shape [N, D].")
    if min(embeddings.shape) < 1 or len(embeddings) < 2 or not embeddings.is_floating_point():
        raise ValueError("Require at least two floating-point embeddings with nonempty features.")
    embeddings = embeddings.detach().float().contiguous()
    if not torch.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinity.")
    for name in ("pytorch_lightning", "lightning_fabric"):
        logging.getLogger(name).setLevel(logging.WARNING)
    pl.seed_everything(config.seed, workers=True, verbose=False)
    indices = torch.randperm(len(embeddings), generator=torch.Generator().manual_seed(config.seed))
    n_val = min(len(embeddings) - 1, max(1, round(len(embeddings) * config.validation_fraction)))
    split = {"train": indices[n_val:], "val": indices[:n_val]}
    loaders = {}
    for name, subset in split.items():
        loaders[name] = DataLoader(
            TensorDataset(embeddings[subset]), batch_size=config.batch_size,
            shuffle=name == "train", num_workers=config.num_workers,
            persistent_workers=config.num_workers > 0,
            generator=torch.Generator().manual_seed(config.seed),
        )
    module = SAETrainer(config, embeddings.shape[1])
    run_dir = create_run_dir(source / "SAE")
    logger = CSVLogger(str(run_dir), name="", version="")
    _ = logger.experiment
    metadata = config.to_dict()
    metadata.update(input_dim=embeddings.shape[1], num_embeddings=len(embeddings),
                    num_train=len(split["train"]), num_validation=len(split["val"]),
                    overcomplete_version=version("overcomplete"),
                    torch_version=str(torch.__version__), lightning_version=pl.__version__)
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    torch.save(split, run_dir / "split_indices.pt")
    trainer = pl.Trainer(
        default_root_dir=run_dir, max_epochs=config.epochs, devices=1,
        accelerator=config.accelerator, precision=config.precision,
        deterministic=config.deterministic, logger=logger, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
        log_every_n_steps=1,
    )
    trainer.fit(module, train_dataloaders=loaders["train"], val_dataloaders=loaders["val"])
    trainer.save_checkpoint(run_dir / "last.ckpt")
    artifact = {
        "format_version": 1, "architecture": module.architecture, "config": metadata,
        "source_model_log_dir": str(source),
        "source_model_relative_path": "../..",
        "state_dict": {key: value.detach().cpu() for key, value in module.model.state_dict().items()},
    }
    torch.save(artifact, run_dir / "model.pt.tmp")
    (run_dir / "model.pt.tmp").replace(run_dir / "model.pt")
    return run_dir
