"""Train an Overcomplete sparse autoencoder on a saved image experiment."""

import argparse
from copy import deepcopy

from cfgs.cfg_sae import CONFIG
from src.trainer_sae import run_sae_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-log-dir")
    parser.add_argument("--model-name", choices=["basic", "topk", "batch_topk"])
    parser.add_argument("--concept-expansion-factor", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--l1-coefficient", type=float)
    parser.add_argument("--dead-neuron-reanimation", action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--reanimation-loss-coefficient", type=float)
    parser.add_argument("--reanimation-top-k", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu", "mps"])
    config = deepcopy(CONFIG)
    for name, value in vars(parser.parse_args()).items():
        if value is not None:
            setattr(config, name, value)
    run_sae_experiment(config)


if __name__ == "__main__":
    main()
