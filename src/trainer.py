"""Lightning training, experiment logging, and aligned final-model embeddings."""

import csv
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from cfgs.cfg import ExperimentConfig
from src.data import ImageDataModule
from src.models import build_model
from src.scheduler import build_scheduler


class Classifier(pl.LightningModule):
    def __init__(self, config: ExperimentConfig, num_classes: int,
                 evaluation_source: str = "stratified_train_holdout"):
        super().__init__()
        self.config = config
        self.evaluation_source = evaluation_source
        self.save_hyperparameters({"config": config.to_dict(), "num_classes": num_classes,
                                   "evaluation_source": evaluation_source})
        self.model = build_model(config.model_name, num_classes, config.pretrained)
        self.register_buffer("train_totals", torch.zeros(3, dtype=torch.float64), persistent=False)
        self.register_buffer("eval_totals", torch.zeros(3, dtype=torch.float64), persistent=False)

    def forward(self, images):
        return self.model(images)

    def _step(self, batch, totals):
        images, labels, _ = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        totals[0] += loss.detach().double() * labels.numel()
        totals[1] += (logits.detach().argmax(dim=1) == labels).sum()
        totals[2] += labels.numel()
        return loss

    def on_train_epoch_start(self):
        self.train_totals.zero_()

    def training_step(self, batch, batch_idx):
        return self._step(batch, self.train_totals)

    def on_validation_epoch_start(self):
        self.eval_totals.zero_()

    def validation_step(self, batch, batch_idx):
        self._step(batch, self.eval_totals)

    def on_validation_epoch_end(self):
        self.log("eval_loss", self.eval_totals[0] / self.eval_totals[2])
        self.log("eval_accuracy", self.eval_totals[1] / self.eval_totals[2])

    def on_train_epoch_end(self):
        train_loss, train_accuracy = self.train_totals[:2] / self.train_totals[2]
        self.log("train_loss", train_loss)
        self.log("train_accuracy", train_accuracy)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"])
        epoch = self.current_epoch + 1
        if epoch % self.config.print_every_n_epochs == 0:
            eval_loss, eval_accuracy = self.eval_totals[:2] / self.eval_totals[2]
            name = "test" if self.evaluation_source == "labeled_test" else "val"
            print(f"Epoch {epoch:03d} | train loss={train_loss.item():.4f} "
                  f"accuracy={train_accuracy.item():.4f} | "
                  f"{name} loss={eval_loss.item():.4f} accuracy={eval_accuracy.item():.4f}",
                  flush=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate,
                                      weight_decay=self.config.weight_decay)
        return {"optimizer": optimizer,
                "lr_scheduler": build_scheduler(optimizer, self.config.scheduler)}


def create_run_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().replace(microsecond=0)
    while True:
        run_dir = base / timestamp.strftime("%y%m%d-%H_%M_%S")
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            timestamp += timedelta(seconds=1)


def save_metadata(run_dir: Path, config: ExperimentConfig, data: ImageDataModule):
    metadata = config.to_dict()
    metadata.update(evaluation_source=data.evaluation_source,
                    num_train_images=len(data.train_records),
                    num_evaluation_images=len(data.eval_records),
                    num_export_images=len(data.all_records),
                    embedding_checkpoint="final_epoch",
                    embedding_order="train_labels_csv row order",
                    torch_version=torch.__version__, lightning_version=pl.__version__)
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (run_dir / "class_to_idx.json").write_text(json.dumps(data.class_to_idx, indent=2) + "\n")
    with (run_dir / "split.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "label", "split"])
        for split, records in (("train", data.train_records), ("eval", data.eval_records)):
            for record in records:
                relative = Path(os.path.relpath(record.path, config.root_dir)).as_posix()
                writer.writerow([relative, data.class_to_idx[record.label], split])


@torch.inference_mode()
def export_embeddings(module: Classifier, data: ImageDataModule, out_dir: Path):
    """Export every labeled training image, including holdout rows, in CSV order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    was_training = module.training
    module.eval()
    count = len(data.export_dataset)
    embeddings = torch.empty(count, module.model.embedding_dim, dtype=torch.float32)
    labels = torch.empty(count, dtype=torch.long)
    offset = 0
    try:
        with (out_dir / "image_paths.csv.tmp").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path"])
            for images, batch_labels, paths in data.export_dataloader():
                features = module.model.forward_features(images.to(module.device))
                end = offset + len(batch_labels)
                embeddings[offset:end] = features.cpu().float()
                labels[offset:end] = batch_labels
                writer.writerows((path,) for path in paths)
                offset = end
        if offset != count:
            raise RuntimeError(f"Exported {offset} images; expected {count}.")
        torch.save(embeddings, out_dir / "embeddings.pt.tmp")
        torch.save(labels, out_dir / "labels.pt.tmp")
        for filename in ("embeddings.pt", "labels.pt", "image_paths.csv"):
            (out_dir / (filename + ".tmp")).replace(out_dir / filename)
    finally:
        module.train(was_training)


def run_experiment(config: ExperimentConfig) -> Path:
    config.validate()
    # Keep normal console output to the requested epoch summaries; errors/warnings remain visible.
    for name in ("pytorch_lightning", "lightning_fabric"):
        logging.getLogger(name).setLevel(logging.WARNING)
    pl.seed_everything(config.seed, workers=True, verbose=False)
    data = ImageDataModule(config)
    data.setup()
    module = Classifier(config, len(data.class_to_idx), data.evaluation_source)
    run_dir = create_run_dir(config.resolve(config.log_dir))
    logger = CSVLogger(save_dir=str(run_dir), name="", version="")
    # Initialize in the empty directory before writing our own experiment metadata.
    _ = logger.experiment
    (run_dir / "out").mkdir()
    save_metadata(run_dir, config, data)
    checkpoint = ModelCheckpoint(dirpath=run_dir / "checkpoints", filename="best-{epoch:03d}",
                                 monitor="eval_loss", mode="min", save_top_k=1, save_last=True)
    trainer = pl.Trainer(
        default_root_dir=run_dir, max_epochs=config.epochs,
        accelerator=config.accelerator, devices=1, precision=config.precision,
        deterministic=config.deterministic, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0,
        check_val_every_n_epoch=1, log_every_n_steps=1,
        logger=logger,
        callbacks=[checkpoint],
    )
    trainer.fit(module, datamodule=data)
    # Explicitly save the final state: save_last behavior varies across Lightning versions.
    trainer.save_checkpoint(run_dir / "checkpoints" / "last.ckpt")
    # fit leaves the final epoch's parameters in module, independently of best checkpoint.
    export_embeddings(module, data, run_dir / "out")
    return run_dir
