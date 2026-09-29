"""Class-folder image datasets for training, evaluation, and ordered export."""

import os
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
        train_transform = build_transforms(cfg.augments, training=True)
        eval_transform = build_transforms(cfg.augments)
        root_dir = cfg.root_dir.resolve()
        self.train_dataset = ImageDataset(self.train_records, self.class_to_idx,
                                          train_transform, root_dir)
        self.eval_dataset = ImageDataset(self.eval_records, self.class_to_idx,
                                         eval_transform, root_dir)
        self.export_datasets = {
            "train": ImageDataset(self.train_records, self.class_to_idx, eval_transform, root_dir),
            "test": ImageDataset(self.eval_records, self.class_to_idx, eval_transform, root_dir),
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
