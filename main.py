"""Run a configured image classification experiment."""

import argparse
from copy import deepcopy

from cfgs.cfg import CONFIG
from src.trainer import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", help="Dataset path, relative to project root or absolute")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--model-name")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu", "mps"])
    parser.add_argument("--test-labels-csv", help="Labeled test CSV, relative to dataset directory")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable weight download for offline runs")
    args = parser.parse_args()
    config = deepcopy(CONFIG)
    for key, value in vars(args).items():
        if key != "no_pretrained" and value is not None:
            setattr(config, key, value)
    if args.no_pretrained:
        config.pretrained = False
    run_experiment(config)


if __name__ == "__main__":
    main()
