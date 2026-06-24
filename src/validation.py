from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pandas.api.types import is_dtype_equal


from types import MappingProxyType


class ConfigurationError(ValueError):
    pass


class DataValidationError(ValueError):
    def __init__(self, errors: list[ValidationIssue]) -> None:
        self.errors = errors
        message = "\n".join(str(err) for err in errors)
        super().__init__(message)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    column: str | None
    expected: Any
    actual: Any

    def __str__(self) -> str:
        if self.code == "missing_column":
            return f"Missing required columns: ['{self.column}']"
        elif self.code == "missing_for_dtype":
            return f"Column '{self.column}' is missing for dtype check."
        elif self.code == "invalid_dtype":
            return f"Column '{self.column}' has dtype {self.actual}, expected {self.expected}."
        elif self.code == "missing_for_null_check":
            return f"Column '{self.column}' is missing for null check."
        elif self.code == "null_limit_exceeded":
            return f"Column '{self.column}' has null fraction {self.actual:.3f}, above limit {self.expected:.3f}."
        elif self.code == "missing_for_uniqueness":
            return f"Column '{self.column}' is missing for uniqueness check."
        elif self.code == "duplicate_values":
            return f"Column '{self.column}' contains duplicate values."
        elif self.code == "missing_for_range_check":
            return f"Column '{self.column}' is missing for range check."
        elif self.code == "value_below_min":
            return f"Column '{self.column}' contains values below {self.expected}."
        elif self.code == "value_above_max":
            return f"Column '{self.column}' contains values above {self.expected}."
        elif self.code == "missing_for_categorical_hygiene":
            return f"Column '{self.column}' is missing for categorical hygiene check."
        elif self.code == "category_blank_string":
            return f"Column '{self.column}' contains blank string categories."
        elif self.code == "category_whitespace":
            return f"Column '{self.column}' contains categories with leading/trailing whitespace."
        elif self.code == "category_normalization_collision":
            return f"Column '{self.column}' contains distinct categories that normalize to the same value."
        elif self.code == "missing_for_allowed_categories":
            return f"Column '{self.column}' is missing for allowed category check."
        elif self.code == "unexpected_category":
            return f"Column '{self.column}' contains categories outside the allowed set."
        return f"Validation error on column '{self.column}' (code={self.code}): expected {self.expected}, actual {self.actual}"


@dataclass(frozen=True)
class DataSchema:
    required_columns: list[str] | tuple[str, ...] = field(default_factory=list)
    dtypes: dict[str, str | type] | MappingProxyType[str, str | type] = field(default_factory=dict)
    null_limits: dict[str, float] | MappingProxyType[str, float] = field(default_factory=dict)
    unique_columns: list[str] | tuple[str, ...] = field(default_factory=list)
    ranges: dict[str, tuple[float | None, float | None]] | MappingProxyType[str, tuple[float | None, float | None]] = field(default_factory=dict)
    categorical_hygiene_columns: list[str] | tuple[str, ...] = field(default_factory=list)
    allowed_categories: dict[str, list[Any] | tuple[Any, ...] | set[Any]] | MappingProxyType[str, tuple[Any, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Check null_limits
        for column, max_null_fraction in self.null_limits.items():
            if not 0.0 <= max_null_fraction <= 1.0:
                raise ConfigurationError(
                    f"Null limit for '{column}' must be between 0 and 1, got {max_null_fraction}."
                )

        # Check ranges
        for column, r in self.ranges.items():
            if r is not None:
                if len(r) != 2:
                    raise ConfigurationError(
                        f"Range for '{column}' must be a tuple/list of length 2, got {r}."
                    )
                lo, hi = r
                if lo is not None and hi is not None and lo > hi:
                    raise ConfigurationError(
                        f"Range for '{column}' has lower bound {lo} greater than upper bound {hi}."
                    )

        object.__setattr__(self, "required_columns", tuple(self.required_columns))
        object.__setattr__(self, "dtypes", MappingProxyType(dict(self.dtypes)))
        object.__setattr__(self, "null_limits", MappingProxyType(dict(self.null_limits)))
        object.__setattr__(self, "unique_columns", tuple(self.unique_columns))
        object.__setattr__(self, "ranges", MappingProxyType(dict(self.ranges)))
        object.__setattr__(
            self,
            "categorical_hygiene_columns",
            tuple(self.categorical_hygiene_columns),
        )
        object.__setattr__(
            self,
            "allowed_categories",
            MappingProxyType(
                {column: tuple(values) for column, values in self.allowed_categories.items()}
            ),
        )


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    errors: list[ValidationIssue]


def validate_dataframe(
    dataframe: pd.DataFrame,
    schema: DataSchema,
    *,
    raise_on_error: bool = True,
) -> ValidationResult:
    errors: list[ValidationIssue] = []

    missing_columns = [
        column for column in schema.required_columns if column not in dataframe.columns
    ]
    for column in missing_columns:
        errors.append(
            ValidationIssue(
                code="missing_column",
                column=column,
                expected="present",
                actual="missing",
            )
        )

    for column, expected_dtype in schema.dtypes.items():
        if column not in dataframe.columns:
            errors.append(
                ValidationIssue(
                    code="missing_for_dtype",
                    column=column,
                    expected="present",
                    actual="missing",
                )
            )
            continue

        if not is_dtype_equal(dataframe[column].dtype, expected_dtype):
            errors.append(
                ValidationIssue(
                    code="invalid_dtype",
                    column=column,
                    expected=expected_dtype,
                    actual=dataframe[column].dtype,
                )
            )

    for column, max_null_fraction in schema.null_limits.items():
        if column not in dataframe.columns:
            errors.append(
                ValidationIssue(
                    code="missing_for_null_check",
                    column=column,
                    expected="present",
                    actual="missing",
                )
            )
            continue

        null_fraction = float(dataframe[column].isna().mean())
        if null_fraction > max_null_fraction:
            errors.append(
                ValidationIssue(
                    code="null_limit_exceeded",
                    column=column,
                    expected=max_null_fraction,
                    actual=null_fraction,
                )
            )

    for column in schema.unique_columns:
        if column not in dataframe.columns:
            errors.append(
                ValidationIssue(
                    code="missing_for_uniqueness",
                    column=column,
                    expected="present",
                    actual="missing",
                )
            )
            continue
        if dataframe[column].duplicated().any():
            errors.append(
                ValidationIssue(
                    code="duplicate_values",
                    column=column,
                    expected="unique",
                    actual="duplicated",
                )
            )

    for column, (minimum, maximum) in schema.ranges.items():
        if column not in dataframe.columns:
            errors.append(
                ValidationIssue(
                    code="missing_for_range_check",
                    column=column,
                    expected="present",
                    actual="missing",
                )
            )
            continue

        series = dataframe[column].dropna()
        if minimum is not None and (series < minimum).any():
            errors.append(
                ValidationIssue(
                    code="value_below_min",
                    column=column,
                    expected=minimum,
                    actual=series.min(),
                )
            )
        if maximum is not None and (series > maximum).any():
            errors.append(
                ValidationIssue(
                    code="value_above_max",
                    column=column,
                    expected=maximum,
                    actual=series.max(),
                )
            )

    for column in schema.categorical_hygiene_columns:
        if column not in dataframe.columns:
            errors.append(
                ValidationIssue(
                    code="missing_for_categorical_hygiene",
                    column=column,
                    expected="present",
                    actual="missing",
                )
            )
            continue

        string_values = dataframe[column].dropna().loc[
            lambda values: values.map(lambda value: isinstance(value, str))
        ]
        blank_counts = _value_counts(
            string_values[string_values.map(lambda value: value.strip() == "")]
        )
        if blank_counts:
            errors.append(
                ValidationIssue(
                    code="category_blank_string",
                    column=column,
                    expected="non-blank strings",
                    actual=blank_counts,
                )
            )

        whitespace_counts = _value_counts(
            string_values[
                string_values.map(
                    lambda value: value != "" and value.strip() != value
                )
            ]
        )
        if whitespace_counts:
            errors.append(
                ValidationIssue(
                    code="category_whitespace",
                    column=column,
                    expected="no leading/trailing whitespace",
                    actual=whitespace_counts,
                )
            )

        collisions = _normalization_collisions(string_values)
        if collisions:
            errors.append(
                ValidationIssue(
                    code="category_normalization_collision",
                    column=column,
                    expected="unique labels after strip/casefold normalization",
                    actual=collisions,
                )
            )

    for column, allowed_values in schema.allowed_categories.items():
        if column not in dataframe.columns:
            errors.append(
                ValidationIssue(
                    code="missing_for_allowed_categories",
                    column=column,
                    expected="present",
                    actual="missing",
                )
            )
            continue

        allowed = set(allowed_values)
        unexpected = dataframe[column].dropna().loc[
            lambda values: ~values.isin(allowed)
        ]
        unexpected_counts = _value_counts(unexpected)
        if unexpected_counts:
            errors.append(
                ValidationIssue(
                    code="unexpected_category",
                    column=column,
                    expected=tuple(allowed_values),
                    actual=unexpected_counts,
                )
            )

    result = ValidationResult(is_valid=not errors, errors=errors)
    if raise_on_error and errors:
        raise DataValidationError(errors)

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


def _normalize_category_label(value: str) -> str:
    return value.strip().casefold()


def _normalization_collisions(values: pd.Series) -> dict[str, list[Any]]:
    labels_by_normalized_value: dict[str, list[Any]] = {}
    for value in values.drop_duplicates():
        normalized = _normalize_category_label(value)
        labels_by_normalized_value.setdefault(normalized, []).append(value)

    return {
        normalized: labels
        for normalized, labels in labels_by_normalized_value.items()
        if len(labels) > 1
    }


def _value_counts(values: pd.Series) -> dict[Any, int]:
    return values.value_counts(dropna=False).to_dict()
