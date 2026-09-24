"""CSV-labeled images, reproducible stratified splits, and ordered export data."""

import csv
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


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: str


def read_records(csv_path: Path, image_dir: Path, extension: str) -> list[ImageRecord]:
    records, seen = [], set()
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"id", "label"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{csv_path} must contain id,label columns.")
        for row in reader:
            image_id, label = (row.get("id") or "").strip(), (row.get("label") or "").strip()
            if not image_id or not label:
                raise ValueError(f"Empty id or label in {csv_path}.")
            filename = image_id if Path(image_id).suffix else image_id + extension
            path = (image_dir / filename).resolve()
            if not path.is_relative_to(image_dir.resolve()):
                raise ValueError(f"Image id escapes the image directory: {image_id}")
            if path in seen:
                raise ValueError(f"Duplicate image in {csv_path}: {image_id}")
            if not path.is_file():
                raise FileNotFoundError(path)
            seen.add(path)
            records.append(ImageRecord(path, label))
    if not records:
        raise ValueError(f"No labeled images in {csv_path}.")
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
        self.all_records = read_records(dataset / cfg.train_labels_csv,
                                        dataset / cfg.train_dir, cfg.image_extension)
        self.class_to_idx = {label: i for i, label in
                             enumerate(sorted({r.label for r in self.all_records}))}
        if len(self.class_to_idx) < 2:
            raise ValueError("Classification requires at least two classes.")
        if cfg.test_labels_csv:
            self.train_records = self.all_records
            self.eval_records = read_records(dataset / cfg.test_labels_csv,
                                             dataset / cfg.test_dir, cfg.image_extension)
            unknown = {r.label for r in self.eval_records} - self.class_to_idx.keys()
            if unknown:
                raise ValueError(f"Unknown test labels: {sorted(unknown)}")
            if {r.path for r in self.train_records} & {r.path for r in self.eval_records}:
                raise ValueError("Training and evaluation images must not overlap.")
            self.evaluation_source = "labeled_test"
        else:
            rng = random.Random(cfg.seed)
            held_out = set()
            for label in self.class_to_idx:
                indices = [i for i, r in enumerate(self.all_records) if r.label == label]
                if len(indices) < 2:
                    raise ValueError(f"Class {label!r} needs at least two images for a holdout.")
                rng.shuffle(indices)
                count = min(len(indices) - 1, max(1, round(len(indices) * cfg.holdout_fraction)))
                held_out.update(indices[:count])
            self.train_records = [r for i, r in enumerate(self.all_records) if i not in held_out]
            self.eval_records = [r for i, r in enumerate(self.all_records) if i in held_out]
            self.evaluation_source = "stratified_train_holdout"
        train_transform = build_transforms(cfg.augments, training=True)
        eval_transform = build_transforms(cfg.augments)
        self.train_dataset = ImageDataset(self.train_records, self.class_to_idx,
                                           train_transform, cfg.root_dir)
        self.eval_dataset = ImageDataset(self.eval_records, self.class_to_idx,
                                          eval_transform, cfg.root_dir)
        self.export_dataset = ImageDataset(self.all_records, self.class_to_idx,
                                            eval_transform, cfg.root_dir)
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

    def export_dataloader(self):
        return self._loader(self.export_dataset)
