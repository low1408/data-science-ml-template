from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from src.project_config import load_project_config, run_project_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a tabular ML experiment from a TOML config."
    )
    parser.add_argument("config", type=Path, help="Path to a project TOML config.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = load_project_config(args.config)
    result = run_project_config(config)
    print(result.comparison.to_string())
    if result.artifact_paths:
        print("\nArtifacts:")
        for name, path in sorted(result.artifact_paths.items()):
            print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
