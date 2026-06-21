from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_string_dtype,
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE


@dataclass(frozen=True)
class ColumnTypes:
    numeric: list[str]
    categorical: list[str]
    datetime: list[str]
    boolean: list[str]
    text: list[str]


def detect_column_types(dataframe: pd.DataFrame) -> ColumnTypes:
    numeric: list[str] = []
    categorical: list[str] = []
    datetime: list[str] = []
    boolean: list[str] = []
    text: list[str] = []

    for column in dataframe.columns:
        series = dataframe[column]
        if is_bool_dtype(series):
            boolean.append(column)
        elif is_datetime64_any_dtype(series):
            datetime.append(column)
        elif is_numeric_dtype(series):
            numeric.append(column)
        elif is_string_dtype(series):
            text.append(column)
            categorical.append(column)
        else:
            categorical.append(column)

    return ColumnTypes(
        numeric=numeric,
        categorical=categorical,
        datetime=datetime,
        boolean=boolean,
        text=text,
    )


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


def build_preprocessor(
    dataframe: pd.DataFrame,
    *,
    scale_numeric: bool = False,
    numeric_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
) -> ColumnTransformer:
    column_types = detect_column_types(dataframe)
    numeric_columns = numeric_columns if numeric_columns is not None else column_types.numeric
    categorical_columns = (
        categorical_columns if categorical_columns is not None else column_types.categorical
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

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_model_pipeline(
    estimator: object,
    dataframe: pd.DataFrame,
    *,
    scale_numeric: bool = False,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(dataframe, scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )
