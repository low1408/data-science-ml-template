from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features import FeaturePipeline, SklearnFeaturePipeline


@dataclass(frozen=True)
class FeatureColumns:
    numeric: list[str] | tuple[str, ...]
    categorical: list[str] | tuple[str, ...]
    boolean: list[str] | tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "numeric", tuple(self.numeric))
        object.__setattr__(self, "categorical", tuple(self.categorical))
        object.__setattr__(self, "boolean", tuple(self.boolean or ()))


from typing import Any
from sklearn.base import BaseEstimator, TransformerMixin


DEFAULT_BOOLEAN_MAPPING = {
    True: 1,
    False: 0,
    1: 1,
    0: 0,
    1.0: 1,
    0.0: 0,
    "true": 1,
    "false": 0,
    "True": 1,
    "False": 0,
    "TRUE": 1,
    "FALSE": 0,
    "yes": 1,
    "no": 0,
    "Yes": 1,
    "No": 0,
    "YES": 1,
    "NO": 0,
    "y": 1,
    "n": 0,
    "Y": 1,
    "N": 0,
    "1": 1,
    "0": 0,
}


class BooleanMappingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, mapping: dict[Any, int] | None = None) -> None:
        self.mapping = mapping

    def fit(self, X: pd.DataFrame, y: Any = None) -> BooleanMappingTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = pd.DataFrame(X).copy()

        current_mapping = dict(DEFAULT_BOOLEAN_MAPPING)
        if self.mapping is not None:
            current_mapping.update(self.mapping)

        def map_value(val: Any) -> Any:
            if pd.isna(val):
                return val

            # Try exact match first
            if val in current_mapping:
                return current_mapping[val]

            # If string, try normalized match
            if isinstance(val, str):
                normalized = val.strip().lower()
                if normalized in current_mapping:
                    return current_mapping[normalized]

            raise ValueError(
                f"Value {repr(val)} (type {type(val)}) could not be mapped to a boolean integer. "
                f"Supported values (case-insensitive for strings) are: {list(current_mapping.keys())}"
            )

        if hasattr(X_df, "map"):
            return X_df.map(map_value)
        else:
            return X_df.applymap(map_value)


@dataclass(frozen=True)
class PreprocessingConfig:
    feature_columns: FeatureColumns
    scale_numeric: bool = False
    numeric_imputer_strategy: str = "median"
    categorical_imputer_strategy: str = "most_frequent"
    boolean_imputer_strategy: str = "most_frequent"
    remainder: str = "drop"
    boolean_mapping: dict[Any, int] | None = None


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
            ("encoder", BooleanMappingTransformer(mapping=config.boolean_mapping)),
            ("to_object", FunctionTransformer(_cast_to_object)),
            ("imputer", SimpleImputer(strategy=config.boolean_imputer_strategy)),
            ("to_int", FunctionTransformer(_cast_to_int)),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str] | tuple[str, ...]]] = []
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
    feature_pipeline: FeaturePipeline | None = None,
) -> Pipeline:
    steps: list[tuple[str, Any]] = []
    if feature_pipeline is not None:
        steps.append(("feature_engineering", SklearnFeaturePipeline(feature_pipeline)))
        dataframe = feature_pipeline.transform(dataframe)

    steps.extend([
        (
            "preprocessor",
            build_preprocessor(
                dataframe,
                config=config,
            ),
        ),
        ("model", estimator),
    ])
    return Pipeline(steps=steps)



def _validate_feature_columns(dataframe: pd.DataFrame, columns: FeatureColumns) -> None:
    configured_columns = list(columns.numeric) + list(columns.categorical) + list(columns.boolean or ())
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



