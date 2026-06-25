from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import warnings

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
    _warn_on_duplicate_rows(dataframe)
    _warn_on_constant_columns(dataframe)

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


def _warn_on_duplicate_rows(dataframe: pd.DataFrame) -> None:
    duplicate_count = int(dataframe.duplicated().sum())
    if duplicate_count:
        warnings.warn(
            f"Dataframe contains {duplicate_count} duplicate row(s).",
            UserWarning,
            stacklevel=3,
        )


def _warn_on_constant_columns(dataframe: pd.DataFrame) -> None:
    if dataframe.empty:
        return

    constant_columns = [
        column
        for column in dataframe.columns
        if dataframe[column].nunique(dropna=False) <= 1
    ]
    if constant_columns:
        warnings.warn(
            "Dataframe contains constant-value column(s): "
            f"{constant_columns}.",
            UserWarning,
            stacklevel=3,
        )


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


def validate_pipeline_and_search_configs(
    validation: Any,
    search: Any,
    task: str,
    target_column: str,
    feature_columns: Any,
) -> None:
    # 1. Validation method checks
    VALID_VALIDATION_METHODS = {"holdout", "kfold", "stratified_kfold", "group_kfold", "time_series_split", "expanding_window"}
    if validation.method not in VALID_VALIDATION_METHODS:
        raise ConfigurationError(
            f"validation.method must be one of {sorted(VALID_VALIDATION_METHODS)}, got '{validation.method}'."
        )

    # 2. Split numeric boundaries
    if validation.n_splits < 2:
        raise ConfigurationError(f"validation.n_splits must be at least 2, got {validation.n_splits}.")
    if not (0.0 < validation.test_size < 1.0):
        raise ConfigurationError(f"validation.test_size must be between 0.0 and 1.0 (exclusive), got {validation.test_size}.")

    # 3. Missing column checks
    if validation.method == "group_kfold" and validation.groups_column is None:
        raise ConfigurationError("validation.groups_column is required when validation.method is 'group_kfold'.")
    if validation.method == "time_series_split" and validation.time_column is None:
        raise ConfigurationError("validation.time_column is required when validation.method is 'time_series_split'.")
    if validation.method == "expanding_window" and validation.time_column is None:
        raise ConfigurationError("validation.time_column is required when validation.method is 'expanding_window'.")

    # 4. Stratified kfold + regression prevention
    if validation.method == "stratified_kfold" and task == "regression":
        raise ConfigurationError("validation.method 'stratified_kfold' is only supported for classification tasks.")

    # 5. Prevent leakage: validation columns cannot be feature or target columns
    all_features = set()
    if feature_columns is not None:
        all_features = (
            set(feature_columns.numeric)
            | set(feature_columns.categorical)
            | set(feature_columns.boolean)
            | set(feature_columns.datetime)
            | set(feature_columns.text)
        )
    if validation.groups_column is not None:
        if validation.groups_column in all_features:
            raise ConfigurationError(
                f"Validation groups column '{validation.groups_column}' cannot be included in the model feature columns."
            )
        if validation.groups_column == target_column:
            raise ConfigurationError(
                f"Validation groups column '{validation.groups_column}' cannot be the target column."
            )
    if validation.time_column is not None:
        if validation.time_column in all_features:
            raise ConfigurationError(
                f"Validation time column '{validation.time_column}' cannot be included in the model feature columns."
            )
        if validation.time_column == target_column:
            raise ConfigurationError(
                f"Validation time column '{validation.time_column}' cannot be the target column."
            )

    # 6. Search checks
    VALID_SEARCH_METHODS = {"none", "randomized", "grid"}
    if search.method not in VALID_SEARCH_METHODS:
        raise ConfigurationError(
            f"search.method must be one of {sorted(VALID_SEARCH_METHODS)}, got '{search.method}'."
        )

    if search.method != "none":
        if search.n_iter < 1:
            raise ConfigurationError(f"search.n_iter must be at least 1, got {search.n_iter}.")
        if search.refit is False:
            raise ConfigurationError("search.refit=False is not supported because the pipeline requires a fitted model to evaluate on the holdout split.")
        if validation.method == "holdout":
            warnings.warn(
                f"validation.method='holdout' is combined with an active hyperparameter search "
                f"(search.method='{search.method}'). "
                f"In this configuration, validation.n_splits={validation.n_splits} controls the "
                f"number of cross-validation folds used *inside* the search object "
                f"(GridSearchCV/RandomizedSearchCV), not the outer holdout split. "
                f"The outer train/test boundary is determined solely by validation.test_size="
                f"{validation.test_size}. Set validation.method to 'kfold' or 'stratified_kfold' "
                f"to run a consistent cross-validation for both baseline evaluation and search.",
                UserWarning,
                stacklevel=2,
            )

        if isinstance(search.scoring, dict):
            if isinstance(search.refit, bool):
                raise ConfigurationError(
                    "search.refit cannot be a boolean when search.scoring is a dictionary of multiple metrics. "
                    "Please specify a string metric name from scoring."
                )
            if search.refit not in search.scoring:
                raise ConfigurationError(
                    f"search.refit metric '{search.refit}' must be one of the search.scoring keys: {list(search.scoring.keys())}."
                )
        else:
            if isinstance(search.refit, str):
                raise ConfigurationError(
                    "search.refit can only be a string metric name when search.scoring is a dictionary of multiple metrics."
                )

