from __future__ import annotations

import argparse
from pathlib import Path

from src.data import CSVDataLoader
from src.preprocessing import infer_feature_columns


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print a first-draft column-role assignment for a CSV file."
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--target", required=True, help="Target column to exclude.")
    parser.add_argument(
        "--categorical-max-unique",
        type=int,
        default=None,
        help="Treat numeric columns with at most this many unique values as categorical.",
    )
    args = parser.parse_args()

    dataframe = CSVDataLoader(args.csv_path, base_path=Path.cwd()).load()
    columns = infer_feature_columns(
        dataframe,
        target_column=args.target,
        categorical_max_unique=args.categorical_max_unique,
    )

    print("[columns]")
    print(f"numeric = {list(columns.numeric)!r}")
    print(f"categorical = {list(columns.categorical)!r}")
    print(f"boolean = {list(columns.boolean)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
