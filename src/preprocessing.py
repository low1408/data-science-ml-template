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


@dataclass(frozen=True)
class PreprocessingConfig:
    feature_columns: FeatureColumns
    scale_numeric: bool = False
    numeric_imputer_strategy: str = "median"
    categorical_imputer_strategy: str = "most_frequent"
    boolean_imputer_strategy: str = "most_frequent"
    remainder: str = "drop"


def _cast_to_object(values: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return values.astype(object)


def _cast_to_int(values: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return values.astype(int)


def build_preprocessor(
    dataframe: pd.DataFrame,
    *,
    config: PreprocessingConfig,
) -> ColumnTransformer:
    _validate_feature_columns(dataframe, config.feature_columns)

    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy=config.numeric_imputer_strategy)),
    ]
    if config.scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=config.categorical_imputer_strategy)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    boolean_pipeline = Pipeline(
        steps=[
            ("to_object", FunctionTransformer(_cast_to_object)),
            ("imputer", SimpleImputer(strategy=config.boolean_imputer_strategy)),
            ("to_int", FunctionTransformer(_cast_to_int)),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if config.feature_columns.numeric:
        transformers.append(("numeric", Pipeline(numeric_steps), config.feature_columns.numeric))
    if config.feature_columns.categorical:
        transformers.append(("categorical", categorical_pipeline, config.feature_columns.categorical))
    if config.feature_columns.boolean:
        transformers.append(("boolean", boolean_pipeline, config.feature_columns.boolean))

    return ColumnTransformer(transformers=transformers, remainder=config.remainder)


def build_model_pipeline(
    estimator: object,
    dataframe: pd.DataFrame,
    *,
    config: PreprocessingConfig,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    dataframe,
                    config=config,
                ),
            ),
            ("model", estimator),
        ]
    )



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
