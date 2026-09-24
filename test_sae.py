"""Print each image's SAE hidden codes using the original trained image encoder."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from cfgs.cfg import AugmentConfig
from src.augments import build_transforms
from src.models import build_model
from src.models_sae import load_sae


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def resolve_sae_run(log_dir: str | Path) -> Path:
    """Accept an exact SAE run, an SAE directory, or its source image experiment."""
    path = Path(log_dir).resolve()
    if (path / "model.pt").is_file():
        return path
    base = path / "SAE" if (path / "SAE").is_dir() else path
    runs = sorted(p.parent for p in base.glob("*/model.pt"))
    if not runs:
        raise FileNotFoundError(f"No completed SAE run found in {path}.")
    return runs[-1]


class InferenceImages(Dataset):
    def __init__(self, directory: Path, transform):
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        self.paths = sorted(p for p in directory.rglob("*")
                            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.paths:
            raise ValueError(f"No supported images found in {directory}.")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, str(path)


@torch.inference_mode()
def iter_hidden_embeddings(log_dir, images_dir, batch_size=64, device="auto",
                           model_log_dir=None):
    """Yield (image path, CPU hidden-code tensor), in sorted recursive path order."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    run = resolve_sae_run(log_dir)
    sae, artifact = load_sae(run, device)
    # A relative reference permits moving the entire image experiment directory.
    source = (Path(model_log_dir).resolve() if model_log_dir else
              (run / artifact["source_model_relative_path"]).resolve())
    if not (source / "config.json").is_file() and model_log_dir is None:
        source = Path(artifact["source_model_log_dir"])
    source_config = json.loads((source / "config.json").read_text())
    checkpoint = torch.load(source / "checkpoints" / "last.ckpt",
                            map_location="cpu", weights_only=True)
    encoder = build_model(source_config["model_name"],
                          checkpoint["hyper_parameters"]["num_classes"], pretrained=False)
    state = {key.removeprefix("model."): value for key, value in checkpoint["state_dict"].items()
             if key.startswith("model.")}
    encoder.load_state_dict(state, strict=True)
    if encoder.embedding_dim != artifact["architecture"]["input_dim"]:
        raise ValueError("Image encoder output dimension does not match the SAE input dimension.")
    encoder.to(device).eval()
    transform = build_transforms(AugmentConfig(**source_config["augments"]))
    dataset = InferenceImages(Path(images_dir).resolve(), transform)
    for images, paths in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        features = encoder.forward_features(images.to(device))
        _, codes = sae.encode(features)
        for path, code in zip(paths, codes.cpu()):
            yield path, code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True, help="SAE run or source image log dir (latest completed SAE)")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--model-log-dir", help="Override the original image-model log directory")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    args = parser.parse_args()
    for path, code in iter_hidden_embeddings(args.log_dir, args.images_dir, args.batch_size,
                                            args.device, args.model_log_dir):
        # JSON lists print every hidden coordinate, without tensor display truncation.
        print(json.dumps({"image_path": path, "embedding": code.tolist()}), flush=True)


if __name__ == "__main__":
    main()
