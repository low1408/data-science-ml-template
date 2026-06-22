from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
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


def get_default_feature_columns() -> FeatureColumns | None:
    from src import config

    config_numeric = getattr(config, "NUMERIC_COLUMNS", None)
    config_categorical = getattr(config, "CATEGORICAL_COLUMNS", None)
    config_boolean = getattr(config, "BOOLEAN_COLUMNS", None)

    if (
        config_numeric is not None
        or config_categorical is not None
        or config_boolean is not None
    ):
        return FeatureColumns(
            numeric=config_numeric or [],
            categorical=config_categorical or [],
            boolean=config_boolean or [],
        )

    return None


def split_features_target(
    dataframe: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in dataframe.columns:
        raise KeyError(f"Target column not found: {target_column}")

    return dataframe.drop(columns=[target_column]), dataframe[target_column]


def train_test_split_dataframe(
    dataframe: pd.DataFrame,
    target_column: str,
    *,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    stratify: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    features, target = split_features_target(dataframe, target_column)
    stratify_values = target if stratify else None

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values,
    )


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
    feature_columns: FeatureColumns | None = None,
    numeric_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    boolean_columns: list[str] | None = None,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    dataframe,
                    scale_numeric=scale_numeric,
                    feature_columns=feature_columns,
                    numeric_columns=numeric_columns,
                    categorical_columns=categorical_columns,
                    boolean_columns=boolean_columns,
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
    elif (default_cols := get_default_feature_columns()) is not None:
        columns = default_cols
    else:
        raise ValueError(
            "Feature columns must be configured explicitly. Pass feature_columns, "
            "pass numeric_columns/categorical_columns/boolean_columns, or configure "
            "config.NUMERIC_COLUMNS / config.CATEGORICAL_COLUMNS / config.BOOLEAN_COLUMNS."
        )

    _validate_feature_columns(dataframe, columns)
    return columns


def _validate_feature_columns(dataframe: pd.DataFrame, columns: FeatureColumns) -> None:
    configured_columns = columns.numeric + columns.categorical + (columns.boolean or [])
    missing_columns = [column for column in configured_columns if column not in dataframe.columns]
    if missing_columns:
        raise KeyError(f"Configured feature columns are missing from dataframe: {missing_columns}")

    duplicate_columns = sorted(
        {column for column in configured_columns if configured_columns.count(column) > 1}
    )
    if duplicate_columns:
        raise ValueError(
            "Feature columns can only belong to one role. "
            f"Duplicates: {duplicate_columns}"
        )
