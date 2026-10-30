import argparse
import copy
import itertools
from pathlib import Path

import yaml


def generate_configs(config_path):
    config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    # Parameters to vary
    expansion_factors = [2, 4, 8]
    top_k_values = [10, 16, 32]
    sae_models = ["topk", "batch_topk"]
    image_models = ["dino", "blip"]
    tversky_modes = ["support", "coherence", "decoder_cosine", "random"]


    output_dir = config_path.parent

    combinations = itertools.product(
    expansion_factors,
    top_k_values,
    sae_models,
    image_models,
    tversky_modes,
    )

    count = 0

    for expansion, top_k, sae_model, image_model, tversky_mode in combinations:

        config = copy.deepcopy(base_config)

        config["sae"]["concept_expansion_factor"] = expansion
        config["sae"]["top_k"] = top_k
        config["sae"]["model_name"] = sae_model
        config["image"]["model_name"] = image_model
        config["steering"]["tversky_mode"] = tversky_mode

        # Generate descriptive filename
        filename = (
        f"{image_model}_{sae_model}"
        f"_exp{expansion}"
        f"_k{top_k}"
        f"_{tversky_mode}.yaml"
        )

        output_path = output_dir / filename

        # Save configuration
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                config,
                f,
                sort_keys=False,
                default_flow_style=False,
            )

        print(f"Created: {output_path.name}")
        count += 1

    print(f"\nGenerated {count} configuration files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SAE experiment configurations."
    )

    parser.add_argument(
        "config",
        type=str,
        help="Path to the original YAML configuration file.",
    )

    args = parser.parse_args()

    generate_configs(args.config)