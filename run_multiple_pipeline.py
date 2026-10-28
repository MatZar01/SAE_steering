"""Run the complete pipeline once for each YAML file in experiments/."""

import argparse
from pathlib import Path

import yaml

from run_pipeline import prepare_pipeline_configs, run_pipeline


EXPERIMENTS_DIR = Path(__file__).resolve().parent / "experiments"


def load_experiments(directory: Path) -> list[tuple[Path, dict]]:
    """Read every YAML config before any pipeline starts."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Experiments directory does not exist: {directory}")
    paths = sorted(path for path in directory.iterdir()
                   if path.is_file() and path.suffix.lower() in {".yaml", ".yml"})
    if not paths:
        raise ValueError(f"No .yaml or .yml experiment configs found in {directory}")
    experiments = []
    for path in paths:
        try:
            with path.open(encoding="utf-8") as handle:
                overrides = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
        if not isinstance(overrides, dict):
            raise ValueError(f"Experiment config {path} must be a YAML mapping of pipeline stages.")
        try:
            prepare_pipeline_configs(overrides)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid experiment config {path}: {exc}") from exc
        experiments.append((path, overrides))
    return experiments


def run_multiple_pipeline(directory: Path = EXPERIMENTS_DIR) -> dict[str, dict[str, Path]]:
    """Run each experiment sequentially and return its four output directories."""
    experiments = load_experiments(directory)
    results = {}
    for number, (path, overrides) in enumerate(experiments, start=1):
        print(f"Experiment {number}/{len(experiments)}: {path.name}", flush=True)
        results[path.name] = run_pipeline(overrides)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", type=Path, default=EXPERIMENTS_DIR,
                        help="Directory of .yaml/.yml configs (default: project experiments/)")
    args = parser.parse_args()
    run_multiple_pipeline(args.experiments_dir)


if __name__ == "__main__":
    main()
