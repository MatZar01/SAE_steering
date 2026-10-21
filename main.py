"""Run a configured image classification experiment."""

import argparse
from copy import deepcopy

from cfgs.cfg import CONFIG
from cfgs.cfg_sae import CONFIG as SAE_CONFIG
from src.trainer import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", help="Dataset path, relative to project root or absolute")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--export-batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--model-name")
    parser.add_argument("--sae-model-name", choices=["basic", "topk", "batch_topk"],
                        help="SAE model name to include in the image run directory")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--sae-smples", type=float, help="Percentage of each split to embed (0 < value <= 100)")
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu", "mps"])
    parser.add_argument("--no-pretrained", action="store_true", help="Disable weight download for offline runs")
    args = parser.parse_args()
    config = deepcopy(CONFIG)
    for key, value in vars(args).items():
        if key not in {"no_pretrained", "sae_model_name"} and value is not None:
            setattr(config, key, value)
    if args.no_pretrained:
        config.pretrained = False
    run_experiment(config, sae_model_name=args.sae_model_name or SAE_CONFIG.model_name)


if __name__ == "__main__":
    main()
