"""Class-folder image datasets for training, evaluation, and ordered export."""

import heapq
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from cfgs.cfg import ExperimentConfig
from src.augments import build_transforms


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: str


def read_records(split_dir: Path) -> list[ImageRecord]:
    """Read images directly below each class folder in deterministic path order."""
    if not split_dir.is_dir():
        raise NotADirectoryError(split_dir)
    classes = sorted(path for path in split_dir.iterdir() if path.is_dir())
    if not classes:
        raise ValueError(f"No class directories found in {split_dir}.")
    records = []
    for class_dir in classes:
        images = sorted(path for path in class_dir.iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f"No supported images found in class directory {class_dir}.")
        records.extend(ImageRecord(path, class_dir.name) for path in images)
    return records


def select_export_records(records: list[ImageRecord], percentage: float | None,
                          seed: int) -> list[ImageRecord]:
    """Balance exports across classes, drawing unique Gaussian-weighted images per class."""
    if percentage is None:
        return records
    if (type(percentage) not in (int, float) or not math.isfinite(percentage)
            or not 0 < percentage <= 100):
        raise ValueError("sae_smples must be a percentage in (0, 100] or None.")
    if percentage == 100:
        return records
    count = max(1, math.floor(len(records) * percentage / 100 + 0.5))

    by_label: dict[str, list[ImageRecord]] = {}
    for record in records:
        by_label.setdefault(record.label, []).append(record)
    labels = sorted(by_label)
    quotas = dict.fromkeys(labels, 0)
    remaining = count
    while remaining:
        for label in labels:
            if quotas[label] < len(by_label[label]):
                quotas[label] += 1
                remaining -= 1
                if remaining == 0:
                    break

    rng = random.Random(seed)
    selected = set()
    for label in labels:
        group = by_label[label]
        quota = quotas[label]
        if quota == len(group):
            selected.update(group)
            continue

        def draw_key(index: int) -> float:
            z = 6 * (index / (len(group) - 1) - 0.5)
            weight = math.exp(-0.5 * z * z)
            return -math.log1p(-rng.random()) / weight

        selected.update(group[index] for index in
                        heapq.nsmallest(quota, range(len(group)), key=draw_key))
    return [record for record in records if record in selected]


class ImageDataset(Dataset):
    def __init__(self, records, class_to_idx, transform, root_dir):
        self.records = records
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record.path) as image:
            tensor = self.transform(image.convert("RGB"))
        path = Path(os.path.relpath(record.path, self.root_dir)).as_posix()
        return tensor, self.class_to_idx[record.label], path


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self._ready = False

    def setup(self, stage=None):
        if self._ready:
            return
        cfg = self.config
        dataset = cfg.resolve(cfg.dataset_path)
        self.train_records = read_records(dataset / "train")
        self.eval_records = read_records(dataset / "test")
        self.class_to_idx = {label: i for i, label in
                             enumerate(sorted({r.label for r in self.train_records}))}
        if len(self.class_to_idx) < 2:
            raise ValueError("Classification requires at least two training classes.")
        unknown = {r.label for r in self.eval_records} - self.class_to_idx.keys()
        if unknown:
            raise ValueError(f"Unknown test classes: {sorted(unknown)}")
        train_transform = build_transforms(cfg.augments, training=True, model_name=cfg.model_name)
        eval_transform = build_transforms(cfg.augments, model_name=cfg.model_name)
        root_dir = cfg.root_dir.resolve()
        self.train_dataset = ImageDataset(self.train_records, self.class_to_idx,
                                          train_transform, root_dir)
        self.eval_dataset = ImageDataset(self.eval_records, self.class_to_idx,
                                         eval_transform, root_dir)
        export_records = {
            "train": select_export_records(self.train_records, cfg.sae_smples, cfg.seed),
            "test": select_export_records(self.eval_records, cfg.sae_smples, cfg.seed),
        }
        self.export_datasets = {
            split: ImageDataset(selected, self.class_to_idx, eval_transform, root_dir)
            for split, selected in export_records.items()
        }
        self._ready = True

    def _loader(self, dataset, shuffle=False):
        return DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=shuffle,
            num_workers=self.config.num_workers, pin_memory=torch.cuda.is_available(),
            persistent_workers=self.config.num_workers > 0,
            generator=torch.Generator().manual_seed(self.config.seed),
            # BatchNorm needs >1 sample for 32x32 inputs in its final feature map.
            drop_last=shuffle and len(dataset) % self.config.batch_size == 1,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.eval_dataset)

    def export_dataloader(self, split: str):
        return self._loader(self.export_datasets[split])
