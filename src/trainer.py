"""Lightning training, experiment logging, and aligned final-model embeddings."""

import csv
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from time import monotonic

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from cfgs.cfg import ExperimentConfig
from src.data import ImageDataModule
from src.models import build_model
from src.scheduler import build_scheduler


class Classifier(pl.LightningModule):
    def __init__(self, config: ExperimentConfig, num_classes: int,
                 feature_dim: int | None = None):
        super().__init__()
        self.config = config
        self.save_hyperparameters({"config": config.to_dict(), "num_classes": num_classes,
                                   "feature_dim": feature_dim})
        self.model = (torch.nn.Linear(feature_dim, num_classes) if feature_dim is not None
                      else build_model(config.model_name, num_classes, config.pretrained))
        self.register_buffer("train_totals", torch.zeros(3, dtype=torch.float64), persistent=False)
        self.register_buffer("eval_totals", torch.zeros(3, dtype=torch.float64), persistent=False)

    def forward(self, images):
        return self.model(images)

    def _step(self, batch, totals):
        inputs, labels = batch[:2]
        logits = self(inputs)
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
            print(f"Epoch {epoch:03d} | train loss={train_loss.item():.4f} "
                  f"accuracy={train_accuracy.item():.4f} | "
                  f"test loss={eval_loss.item():.4f} accuracy={eval_accuracy.item():.4f}",
                  flush=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW((p for p in self.parameters() if p.requires_grad),
                                      lr=self.config.learning_rate,
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
    frozen_features = config.model_name.lower() in {"blip", "dino"}
    metadata.update(num_train_images=len(data.train_records),
                    num_evaluation_images=len(data.eval_records),
                    num_export_train_images=len(data.export_datasets["train"]),
                    num_export_test_images=len(data.export_datasets["test"]),
                    embedding_checkpoint="frozen_encoder" if frozen_features else "final_epoch",
                    frozen_feature_training=frozen_features,
                    embedding_order="sorted class/filename within each split",
                    torch_version=torch.__version__, lightning_version=pl.__version__)
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (run_dir / "class_to_idx.json").write_text(json.dumps(data.class_to_idx, indent=2) + "\n")


@torch.inference_mode()
def export_embeddings(module: Classifier, data: ImageDataModule, out_dir: Path):
    """Export both splits with deterministic preprocessing and path order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    was_training = module.training
    module.eval()
    try:
        filenames = []
        for split in ("train", "test"):
            count = len(data.export_datasets[split])
            loader = data.export_dataloader(split)
            print(f"Exporting {split} embeddings: 0/{count}", flush=True)
            last_report = monotonic()
            embeddings = torch.empty(count, module.model.embedding_dim, dtype=torch.float32)
            labels = torch.empty(count, dtype=torch.long)
            offset = 0
            paths_name = f"image_paths_{split}.csv"
            with (out_dir / (paths_name + ".tmp")).open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["image_path"])
                for images, batch_labels, paths in loader:
                    features = module.model.forward_features(images.to(module.device))
                    end = offset + len(batch_labels)
                    embeddings[offset:end] = features.cpu().float()
                    labels[offset:end] = batch_labels
                    writer.writerows((path,) for path in paths)
                    offset = end
                    if monotonic() - last_report >= 30 or offset == count:
                        print(f"Exporting {split} embeddings: {offset}/{count}", flush=True)
                        last_report = monotonic()
            if offset != count:
                raise RuntimeError(f"Exported {offset} {split} images; expected {count}.")
            embeddings_name = f"embeddings_{split}.pt"
            labels_name = f"labels_{split}.pt"
            torch.save(embeddings, out_dir / (embeddings_name + ".tmp"))
            torch.save(labels, out_dir / (labels_name + ".tmp"))
            filenames.extend((embeddings_name, labels_name, paths_name))
        for filename in filenames:
            (out_dir / (filename + ".tmp")).replace(out_dir / filename)
    finally:
        module.train(was_training)


def _feature_device(accelerator: str) -> torch.device:
    if accelerator == "cpu":
        return torch.device("cpu")
    if accelerator in {"auto", "gpu"} and torch.cuda.is_available():
        return torch.device("cuda")
    if accelerator in {"auto", "mps"} and torch.backends.mps.is_available():
        return torch.device("mps")
    if accelerator == "auto":
        return torch.device("cpu")
    raise RuntimeError(f"Requested accelerator {accelerator!r} is unavailable.")


@torch.inference_mode()
def cache_frozen_features(model: torch.nn.Module, data: ImageDataModule,
                          out_dir: Path, device: torch.device):
    """Extract each full split once and derive the SAE subset without another encoder pass."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device).eval()
    full = {}
    for split in ("train", "test"):
        count = len(data.feature_datasets[split])
        features = torch.empty(count, model.embedding_dim, dtype=torch.float32)
        labels = torch.empty(count, dtype=torch.long)
        paths = []
        offset = 0
        with tqdm(total=count, desc=f"Extracting {split} features", unit="image",
                  file=sys.stdout) as progress:
            for images, batch_labels, batch_paths in data.feature_dataloader(split):
                batch_features = model.forward_features(images.to(device)).cpu().float()
                end = offset + len(batch_labels)
                features[offset:end] = batch_features
                labels[offset:end] = batch_labels
                paths.extend(batch_paths)
                progress.update(len(batch_labels))
                offset = end
        if offset != count:
            raise RuntimeError(f"Extracted {offset} {split} images; expected {count}.")
        full[split] = (features, labels)
        for name, tensor in ((f"features_{split}.pt", features),
                             (f"feature_labels_{split}.pt", labels)):
            torch.save(tensor, out_dir / (name + ".tmp"))
            (out_dir / (name + ".tmp")).replace(out_dir / name)
        paths_name = f"feature_paths_{split}.csv"
        with (out_dir / (paths_name + ".tmp")).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path"])
            writer.writerows((path,) for path in paths)
        (out_dir / (paths_name + ".tmp")).replace(out_dir / paths_name)

        positions = {record.path: index for index, record in
                     enumerate(data.feature_datasets[split].records)}
        indices = [positions[record.path] for record in data.export_datasets[split].records]
        selected = len(indices) == len(features) and all(i == index for i, index in enumerate(indices))
        names = ((f"embeddings_{split}.pt", f"features_{split}.pt", features),
                 (f"labels_{split}.pt", f"feature_labels_{split}.pt", labels),
                 (f"image_paths_{split}.csv", paths_name, paths))
        for output_name, source_name, values in names:
            destination = out_dir / output_name
            if selected:
                destination.hardlink_to(out_dir / source_name)
            elif output_name.endswith(".pt"):
                torch.save(values[indices], out_dir / (output_name + ".tmp"))
                (out_dir / (output_name + ".tmp")).replace(destination)
            else:
                with (out_dir / (output_name + ".tmp")).open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["image_path"])
                    writer.writerows((paths[index],) for index in indices)
                (out_dir / (output_name + ".tmp")).replace(destination)
    return full


def run_experiment(config: ExperimentConfig) -> Path:
    config.validate()
    # Keep normal console output to the requested epoch summaries; errors/warnings remain visible.
    for name in ("pytorch_lightning", "lightning_fabric"):
        logging.getLogger(name).setLevel(logging.WARNING)
    pl.seed_everything(config.seed, workers=True, verbose=False)
    data = ImageDataModule(config)
    data.setup()
    frozen_features = config.model_name.lower() in {"blip", "dino"}
    image_model = (build_model(config.model_name, len(data.class_to_idx), config.pretrained)
                   if frozen_features else None)
    module = None if frozen_features else Classifier(config, len(data.class_to_idx))
    run_dir = create_run_dir(config.resolve(config.log_dir))
    logger = CSVLogger(save_dir=str(run_dir), name="", version="")
    # Initialize in the empty directory before writing our own experiment metadata.
    _ = logger.experiment
    (run_dir / "out").mkdir()
    save_metadata(run_dir, config, data)
    if frozen_features:
        device = _feature_device(config.accelerator)
        features = cache_frozen_features(image_model, data, run_dir / "out", device)
        image_model.backbone.cpu()
        encoder_path = run_dir / "frozen_encoder.pt"
        encoder_tmp = encoder_path.with_suffix(".pt.tmp")
        torch.save({"state_dict": image_model.backbone.state_dict(),
                    "embedding_dim": image_model.embedding_dim},
                   encoder_tmp)
        encoder_tmp.replace(encoder_path)
        feature_dim = image_model.embedding_dim
        del image_model
        module = Classifier(config, len(data.class_to_idx), feature_dim=feature_dim)
        train_features, train_labels = features["train"]
        test_features, test_labels = features["test"]
        train_loader = DataLoader(
            TensorDataset(train_features, train_labels), batch_size=config.batch_size,
            shuffle=True, generator=torch.Generator().manual_seed(config.seed))
        val_loader = DataLoader(
            TensorDataset(test_features, test_labels), batch_size=config.batch_size)
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
        limit_train_batches=config.train_batches,
        limit_val_batches=config.val_batches,
    )
    if frozen_features:
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        trainer.fit(module, datamodule=data)
    # Explicitly save the final state: save_last behavior varies across Lightning versions.
    trainer.save_checkpoint(run_dir / "checkpoints" / "last.ckpt")
    # Frozen features were exported before fitting; ResNet features follow its final epoch.
    if not frozen_features:
        export_embeddings(module, data, run_dir / "out")
    return run_dir
