"""Overcomplete SAE variants and portable inference checkpoints."""

from pathlib import Path

import torch
from torch import nn

# Overcomplete 0.2.2 imports its plotting helpers even when only SAEs are used.
# Recent Matplotlib releases removed this alias; supply its modern equivalent.
from matplotlib import cm, colormaps

if not hasattr(cm, "get_cmap"):
    cm.get_cmap = colormaps.get_cmap

from overcomplete.sae import SAE, TopKSAE, BatchTopKSAE
from overcomplete.sae.modules import MLPEncoder


class BasicSAE(SAE):
    """ReLU SAE; its training objective adds an L1 penalty to reconstruction MSE."""


class BatchTopKAutoencoder(BatchTopKSAE):
    """Overcomplete BatchTopK with a per-sample budget and persistent threshold."""

    def __init__(self, input_shape, nb_concepts, top_k, **kwargs):
        super().__init__(input_shape, nb_concepts, top_k=top_k, **kwargs)
        self.k_per_sample = top_k
        # Upstream 0.2.2 uses an ordinary attribute, which state_dict omits.
        del self.running_threshold
        self.register_buffer("running_threshold", None)

    def encode(self, x):
        if self.training:
            self.top_k = self.k_per_sample * x.shape[0]
        return super().encode(x)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + "running_threshold"
        if key in state_dict and self.running_threshold is None:
            self.running_threshold = torch.empty_like(
                state_dict[key], device=next(self.parameters()).device)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


SAE_REGISTRY = {"basic": BasicSAE, "topk": TopKSAE, "batch_topk": BatchTopKAutoencoder}


def build_sae(name: str, input_dim: int, nb_concepts: int, top_k: int = 32,
              threshold_momentum: float = 0.9) -> SAE:
    if name not in SAE_REGISTRY:
        raise ValueError(f"Unknown SAE {name!r}; choose from {sorted(SAE_REGISTRY)}.")
    if input_dim < 1 or nb_concepts < 1:
        raise ValueError("Input and hidden dimensions must be positive.")
    if name != "basic" and not 1 <= top_k <= nb_concepts:
        raise ValueError("top_k must be between 1 and nb_concepts.")
    # A single affine projection + ReLU, without batch-dependent normalization.
    encoder = MLPEncoder(input_dim, nb_concepts, nb_blocks=1, norm_layer=nn.Identity)
    options = {"encoder_module": encoder, "dictionary_params": {"normalization": "l2"}}
    if name != "basic":
        options["top_k"] = top_k
    if name == "batch_topk":
        options["threshold_momentum"] = threshold_momentum
    return SAE_REGISTRY[name](input_dim, nb_concepts, **options)


def load_sae(run_dir: str | Path, device: str | torch.device = "cpu"):
    """Load a final SAE model from a specific SAE run directory."""
    run_dir = Path(run_dir)
    artifact = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    if artifact.get("format_version") != 1:
        raise ValueError("Unsupported SAE checkpoint format.")
    model = build_sae(**artifact["architecture"])
    model.load_state_dict(artifact["state_dict"], strict=True)
    return model.to(device).eval(), artifact


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
