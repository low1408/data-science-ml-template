from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pandas.api.types import is_dtype_equal


class DataValidationError(ValueError):
    pass


@dataclass(frozen=True)
class DataSchema:
    required_columns: list[str] = field(default_factory=list)
    dtypes: dict[str, str | type] = field(default_factory=dict)
    null_limits: dict[str, float] = field(default_factory=dict)
    unique_columns: list[str] = field(default_factory=list)
    ranges: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    errors: list[str]


def validate_dataframe(
    dataframe: pd.DataFrame,
    schema: DataSchema,
    *,
    raise_on_error: bool = True,
) -> ValidationResult:
    errors: list[str] = []

    missing_columns = [
        column for column in schema.required_columns if column not in dataframe.columns
    ]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")

    for column, expected_dtype in schema.dtypes.items():
        if column not in dataframe.columns:
            errors.append(f"Column '{column}' is missing for dtype check.")
            continue

        if not is_dtype_equal(dataframe[column].dtype, expected_dtype):
            errors.append(
                f"Column '{column}' has dtype {dataframe[column].dtype}, "
                f"expected {expected_dtype}."
            )

    for column, max_null_fraction in schema.null_limits.items():
        if column not in dataframe.columns:
            errors.append(f"Column '{column}' is missing for null check.")
            continue
        if not 0 <= max_null_fraction <= 1:
            errors.append(f"Null limit for '{column}' must be between 0 and 1.")
            continue

        null_fraction = float(dataframe[column].isna().mean())
        if null_fraction > max_null_fraction:
            errors.append(
                f"Column '{column}' has null fraction {null_fraction:.3f}, "
                f"above limit {max_null_fraction:.3f}."
            )

    for column in schema.unique_columns:
        if column not in dataframe.columns:
            errors.append(f"Column '{column}' is missing for uniqueness check.")
            continue
        if dataframe[column].duplicated().any():
            errors.append(f"Column '{column}' contains duplicate values.")

    for column, (minimum, maximum) in schema.ranges.items():
        if column not in dataframe.columns:
            errors.append(f"Column '{column}' is missing for range check.")
            continue

        series = dataframe[column].dropna()
        if minimum is not None and (series < minimum).any():
            errors.append(f"Column '{column}' contains values below {minimum}.")
        if maximum is not None and (series > maximum).any():
            errors.append(f"Column '{column}' contains values above {maximum}.")

    result = ValidationResult(is_valid=not errors, errors=errors)
    if raise_on_error and errors:
        raise DataValidationError("\n".join(errors))

    return result


def dataset_summary(dataframe: pd.DataFrame) -> dict[str, Any]:
    return {
        "shape": dataframe.shape,
        "columns": list(dataframe.columns),
        "dtypes": dataframe.dtypes.astype(str).to_dict(),
        "missing_values": dataframe.isna().sum().to_dict(),
        "missing_fraction": dataframe.isna().mean().to_dict(),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "cardinality": dataframe.nunique(dropna=True).to_dict(),
        "numeric_stats": _describe_or_empty(dataframe, include="number"),
        "categorical_stats": _describe_or_empty(
            dataframe,
            include=["object", "category", "bool", "str"],
        ),
    }


def _describe_or_empty(dataframe: pd.DataFrame, *, include: str | list[str]) -> pd.DataFrame:
    try:
        return dataframe.describe(include=include).transpose()
    except ValueError:
        return pd.DataFrame()
