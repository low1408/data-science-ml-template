from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE


@dataclass(frozen=True)
class FeatureColumns:
    numeric: list[str]
    categorical: list[str]
    boolean: list[str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "boolean", self.boolean or [])


def _cast_to_object(values: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return values.astype(object)


def _cast_to_int(values: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return values.astype(int)


def build_preprocessor(
    dataframe: pd.DataFrame,
    *,
    scale_numeric: bool = False,
    feature_columns: FeatureColumns | None = None,
    numeric_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    boolean_columns: list[str] | None = None,
) -> ColumnTransformer:
    feature_columns = _resolve_feature_columns(
        dataframe,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        boolean_columns=boolean_columns,
    )

    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    boolean_pipeline = Pipeline(
        steps=[
            ("to_object", FunctionTransformer(_cast_to_object)),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("to_int", FunctionTransformer(_cast_to_int)),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if feature_columns.numeric:
        transformers.append(("numeric", Pipeline(numeric_steps), feature_columns.numeric))
    if feature_columns.categorical:
        transformers.append(("categorical", categorical_pipeline, feature_columns.categorical))
    if feature_columns.boolean:
        transformers.append(("boolean", boolean_pipeline, feature_columns.boolean))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_model_pipeline(
    estimator: object,
    dataframe: pd.DataFrame,
    *,
    scale_numeric: bool = False,
    feature_columns: FeatureColumns,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    dataframe,
                    scale_numeric=scale_numeric,
                    feature_columns=feature_columns,
                ),
            ),
            ("model", estimator),
        ]
    )


def _resolve_feature_columns(
    dataframe: pd.DataFrame,
    *,
    feature_columns: FeatureColumns | None,
    numeric_columns: list[str] | None,
    categorical_columns: list[str] | None,
    boolean_columns: list[str] | None,
) -> FeatureColumns:
    if feature_columns is not None:
        columns = feature_columns
    elif numeric_columns is not None or categorical_columns is not None or boolean_columns is not None:
        columns = FeatureColumns(
            numeric=numeric_columns or [],
            categorical=categorical_columns or [],
            boolean=boolean_columns or [],
        )
    else:
        raise ValueError(
            "Feature columns must be configured explicitly. Pass a FeatureColumns instance via "
            "feature_columns=, or supply numeric_columns/categorical_columns/boolean_columns."
        )

    _validate_feature_columns(dataframe, columns)
    return columns


def _validate_feature_columns(dataframe: pd.DataFrame, columns: FeatureColumns) -> None:
    configured_columns = columns.numeric + columns.categorical + (columns.boolean or [])
    missing_columns = [column for column in configured_columns if column not in dataframe.columns]
    if missing_columns:
        raise KeyError(f"Configured feature columns are missing from dataframe: {missing_columns}")

    # F-07: O(n) duplicate detection via Counter instead of O(n²) list.count()
    counts = Counter(configured_columns)
    duplicate_columns = sorted(col for col, n in counts.items() if n > 1)
    if duplicate_columns:
        raise ValueError(
            "Feature columns can only belong to one role. "
            f"Duplicates: {duplicate_columns}"
        )


# ── Backwards-compatible re-exports (F-08) ──────────────────────────────
# split_features_target and train_test_split_dataframe have moved to src.data.
# These re-exports preserve import compatibility but emit a DeprecationWarning.


def split_features_target(
    dataframe: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Deprecated: use ``from src.data import split_features_target`` instead."""
    warnings.warn(
        "Importing split_features_target from src.preprocessing is deprecated. "
        "Import from src.data instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from src.data import split_features_target as _split

    return _split(dataframe, target_column)


def train_test_split_dataframe(
    dataframe: pd.DataFrame,
    target_column: str,
    *,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    stratify: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Deprecated: use ``from src.data import train_test_split_dataframe`` instead."""
    warnings.warn(
        "Importing train_test_split_dataframe from src.preprocessing is deprecated. "
        "Import from src.data instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from src.data import train_test_split_dataframe as _split

    return _split(
        dataframe,
        target_column,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
