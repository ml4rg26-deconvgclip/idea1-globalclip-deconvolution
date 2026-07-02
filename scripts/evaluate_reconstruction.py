#!/usr/bin/env python3
"""Evaluate future reconstructed globalCLIP profiles against observed targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TODO = """
TODO:
  1. Load the trained reconstruction head from paths.reconstruction_checkpoint.
  2. Load held-out Parnet predictions and observed globalCLIP target profiles.
  3. Compute configured metrics such as Pearson, Spearman, and MSE.
  4. Save evaluation tables and plots under paths.evaluation_dir.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Placeholder entry point for reconstruction-head evaluation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vm.yaml"),
        help="Path to the VM YAML config.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split override for future evaluation runs.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Optional metric override for future evaluation runs.",
    )
    parser.add_argument(
        "--require-checkpoint",
        action="store_true",
        help="Fail if the configured reconstruction checkpoint does not exist.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    config_text = config_path.read_text(encoding="utf-8")

    try:
        import yaml
    except ImportError:
        try:
            config = json.loads(config_text)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                "PyYAML is required for non-JSON YAML configs. Install it with `pip install pyyaml`."
            ) from exc
    else:
        config = yaml.safe_load(config_text) or {}

    if not isinstance(config, dict):
        raise SystemExit(f"Expected a mapping in config file: {config_path}")

    return config


def get_section(config: dict[str, Any], section: str) -> dict[str, Any]:
    values = config.get(section, {})
    if not isinstance(values, dict):
        raise SystemExit(f"Expected config section to be a mapping: {section}")
    return values


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    paths = get_section(config, "paths")
    evaluation = get_section(config, "evaluation")

    checkpoint_path = Path(paths.get("reconstruction_checkpoint", ""))
    evaluation_dir = Path(paths.get("evaluation_dir", ""))
    split = args.split or evaluation.get("split")
    metrics = args.metrics or evaluation.get("metrics", [])

    if args.require_checkpoint and not checkpoint_path.exists():
        raise SystemExit(f"Configured reconstruction checkpoint does not exist: {checkpoint_path}")

    print(TODO.strip())
    print()
    print("Reconstruction evaluation configuration:")
    print(f"  checkpoint:        {checkpoint_path}")
    print(f"  checkpoint exists: {checkpoint_path.exists()}")
    print(f"  output directory:  {evaluation_dir}")
    print(f"  split:             {split}")
    print(f"  metrics:           {metrics}")
    print()
    print("No evaluation is run yet.")


if __name__ == "__main__":
    main()
