"""Rank source images by saved SAE codes and render top-activation grids."""

import csv
import json
import textwrap
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from cfgs.cfg_sae import SAEInferenceConfig
from src.models_sae import load_sae, resolve_sae_run


def _device(name: str) -> torch.device:
    if name == "auto":
        name = ("cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def _steering_run(sae_run: Path, configured: str, config: SAEInferenceConfig) -> Path:
    if configured:
        run = config.resolve(configured)
        if not (run / "neighborhoods.csv").is_file():
            raise FileNotFoundError(f"Missing {run / 'neighborhoods.csv'}.")
        return run
    runs = sorted((sae_run / "steering").glob("*/neighborhoods.csv"))
    if not runs:
        raise FileNotFoundError(f"No steering neighborhoods found under {sae_run / 'steering'}.")
    return runs[-1].parent


def _groups(path: Path, num_latents: int) -> list[tuple[str, str, list[int]]]:
    groups = []
    seen = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            latents = [int(value) for value in row["latents"].split("|")]
            size = int(row["size"])
            anchor = int(row["anchor"])
            if (not latents or len(latents) != size or latents[0] != anchor
                    or len(set(latents)) != size
                    or any(latent < 0 or latent >= num_latents for latent in latents)):
                raise ValueError(f"Invalid latent group in {path}: {row!r}")
            if size == 1:
                continue  # Every individual latent is rendered separately below.
            identity = (anchor, size)
            if identity in seen:
                raise ValueError(f"Duplicate group anchor/size in {path}: {identity}")
            seen.add(identity)
            name = f"group_anchor_{anchor:04d}_size_{size:02d}"
            title = f"Group anchor {anchor}, size {size}: {'|'.join(map(str, latents))}"
            groups.append((name, title, latents))
    return groups


def _split_inputs(source: Path, split: str, image_root: Path, input_dim: int,
                  cache_dir: Path | None = None):
    out = source / "out"
    full_dir = cache_dir if cache_dir is not None else out
    full = full_dir / f"features_{split}.pt"
    if full.is_file():
        features_path = full
        paths_path = full_dir / f"feature_paths_{split}.csv"
    else:
        features_path = out / f"embeddings_{split}.pt"
        paths_path = out / f"image_paths_{split}.csv"
    features = torch.load(features_path, map_location="cpu", weights_only=True)
    if (not isinstance(features, torch.Tensor) or features.ndim != 2
            or features.shape[1] != input_dim or not features.is_floating_point()
            or not torch.isfinite(features).all()):
        raise ValueError(f"{features_path} must contain finite float features [N, {input_dim}].")
    with paths_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["image_path"]:
            raise ValueError(f"{paths_path} must have an image_path column.")
        image_paths = [image_root / row["image_path"] for row in reader]
    if len(image_paths) != len(features):
        raise ValueError(f"{paths_path} has {len(image_paths)} paths for {len(features)} features.")
    return features.float(), image_paths


@torch.inference_mode()
def _codes(sae, features: torch.Tensor, batch_size: int, device: torch.device):
    codes = torch.empty((len(features), sae.nb_concepts), dtype=torch.float32)
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        batch = sae.encode(features[start:end].to(device))[1]
        if not torch.isfinite(batch).all():
            raise ValueError("SAE produced non-finite codes.")
        codes[start:end] = batch.cpu().float()
    return codes


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _render_grid(path: Path, title: str, ranked: list[tuple[int, float]],
                 image_paths: list[Path], side: int) -> None:
    cell, image_size, label_height = 192, 158, 26
    title_lines = textwrap.wrap(title, width=max(12, side * cell // 11))
    title_height = max(64, 16 + 24 * len(title_lines))
    grid = Image.new("RGB", (cell * side, title_height + cell * side), "white")
    draw = ImageDraw.Draw(grid)
    draw.multiline_text((12, 12), "\n".join(title_lines), fill="black",
                        font=_font(18), spacing=4)
    for position, (index, activation) in enumerate(ranked):
        x = (position % side) * cell
        y = title_height + (position // side) * cell
        with Image.open(image_paths[index]) as original:
            thumbnail = ImageOps.contain(original.convert("RGB"), (image_size, image_size))
        grid.paste(thumbnail, (x + (cell - thumbnail.width) // 2,
                               y + (image_size - thumbnail.height) // 2))
        draw.text((x + 8, y + cell - label_height), f"activation: {activation:.6g}",
                  fill="black", font=_font(14))
    if not ranked:
        draw.text((12, title_height + 12), "No positive activations in this split.",
                  fill="black", font=_font(16))
    grid.save(path)


def _write_result(directory: Path, name: str, title: str, scores: torch.Tensor,
                  image_paths: list[Path], side: int, score_description: str) -> None:
    count = min(side * side, len(scores))
    active = torch.nonzero(scores > 0).flatten()
    if len(active) > count:
        cutoff = torch.topk(scores[active], count).values[-1]
        active = active[scores[active] >= cutoff]
    ranked = sorted(((int(index), float(scores[index])) for index in active),
                    key=lambda item: (-item[1], item[0]))[:count]
    lines = [title, f"Score: {score_description}", "rank\tactivation\timage_path"]
    lines += [f"{rank}\t{score:.8g}\t{image_paths[index]}"
              for rank, (index, score) in enumerate(ranked, start=1)]
    (directory / f"{name}.txt").write_text("\n".join(lines) + "\n")
    _render_grid(directory / f"{name}.png", title, ranked, image_paths, side)


def run_sae_inference(config: SAEInferenceConfig) -> Path:
    """Write per-latent and per-neighborhood train/test rankings into active_images."""
    config.validate()
    sae_run = resolve_sae_run(config.resolve(config.sae_log_dir))
    steering_run = _steering_run(sae_run, config.steering_log_dir, config)
    device = _device(config.device)
    sae, artifact = load_sae(sae_run, device)
    groups = _groups(steering_run / "neighborhoods.csv", sae.nb_concepts)
    source = (config.resolve(config.model_log_dir) if config.model_log_dir else
              (sae_run / artifact["source_model_relative_path"]).resolve())
    if not (source / "config.json").is_file() and not config.model_log_dir:
        source = Path(artifact["source_model_log_dir"])
    source_config = json.loads((source / "config.json").read_text())
    image_root = Path(source_config["root_dir"])
    cache_dir = (Path(source_config["feature_cache_dir"])
                 if source_config.get("feature_cache_dir") else None)
    output = sae_run / "active_images"
    for split in ("train", "test"):
        features, image_paths = _split_inputs(
            source, split, image_root, artifact["architecture"]["input_dim"], cache_dir)
        codes = _codes(sae, features, config.batch_size, device)
        directory = output / split
        directory.mkdir(parents=True, exist_ok=True)
        for latent in range(sae.nb_concepts):
            _write_result(directory, f"latent_{latent:04d}", f"Latent {latent}",
                          codes[:, latent], image_paths, config.top_active_images,
                          "SAE code activation")
        for name, title, latents in groups:
            _write_result(directory, name, title, codes[:, latents].sum(dim=1),
                          image_paths, config.top_active_images,
                          "Sum of SAE code activations for group latents")
        print(f"Saved {sae.nb_concepts} latent and {len(groups)} group grids for "
              f"{split} in {directory}", flush=True)
    return output
