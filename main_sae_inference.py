"""Render the most activated train/test images for every SAE latent and steering group."""

import argparse
from copy import deepcopy

from cfgs.cfg_sae import INFERENCE_CONFIG
from src.sae_inference import run_sae_inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sae-log-dir", help="SAE run, SAE directory, or source image run")
    parser.add_argument("--steering-log-dir", help="Exact steering run; defaults to latest for SAE")
    parser.add_argument("--model-log-dir", help="Override a moved source image run")
    parser.add_argument("--top-active-images", type=int, help="Grid side length (3 means 3x3)")
    parser.add_argument("--stitch-images", action=argparse.BooleanOptionalAction, default=None,
                        help="Write PNG grids as well as text rankings (default: enabled)")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", help="auto, cpu, cuda, cuda:0, mps, etc.")
    config = deepcopy(INFERENCE_CONFIG)
    for name, value in vars(parser.parse_args()).items():
        if value is not None:
            setattr(config, name, value)
    print(f"Active images: {run_sae_inference(config)}", flush=True)


if __name__ == "__main__":
    main()
