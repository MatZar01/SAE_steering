"""Run point 1: neighborhood-based SAE steering evaluation."""

import argparse
from copy import deepcopy

from cfgs.cfg_sae import STEERING_CONFIG
from src.steering import run_steering


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sae-log-dir", help="SAE run, SAE directory, or source image run")
    parser.add_argument("--model-log-dir", help="Override a moved source image run")
    parser.add_argument("--max-anchors", type=int)
    parser.add_argument("--all-anchors", action="store_true")
    parser.add_argument("--max-neighborhood-size", type=int)
    parser.add_argument("--tversky-penalty", type=float)
    parser.add_argument("--tversky-mode", choices=["support", "coherence"])
    parser.add_argument("--coherence-pairs", type=int)
    parser.add_argument("--classifier-epochs", type=int)
    parser.add_argument("--classifier-batch-size", type=int)
    parser.add_argument("--classifier-learning-rate", type=float)
    parser.add_argument("--classifier-weight-decay", type=float)
    parser.add_argument("--evaluation-batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", help="auto, cpu, cuda, cuda:0, mps, etc.")
    args = parser.parse_args()
    config = deepcopy(STEERING_CONFIG)
    for name, value in vars(args).items():
        if name != "all_anchors" and value is not None:
            setattr(config, name, value)
    if args.all_anchors:
        config.max_anchors = None
    run_dir = run_steering(config)
    print(f"Steering results: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
