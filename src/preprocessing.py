from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
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


def infer_feature_columns(
    dataframe: pd.DataFrame,
    *,
    target_column: str | None = None,
    categorical_max_unique: int | None = None,
) -> FeatureColumns:
    """Infer initial tabular column roles from pandas dtypes.

    This is intended as a starting point for a portable template. Review the
    output before using it for a serious experiment.
    """
    numeric: list[str] = []
    categorical: list[str] = []
    boolean: list[str] = []

    for column in dataframe.columns:
        if column == target_column:
            continue
        series = dataframe[column]
        if is_bool_dtype(series):
            boolean.append(column)
        elif is_numeric_dtype(series):
            if _looks_boolean(series):
                boolean.append(column)
            elif (
                categorical_max_unique is not None
                and series.nunique(dropna=True) <= categorical_max_unique
            ):
                categorical.append(column)
            else:
                numeric.append(column)
        else:
            categorical.append(column)

    return FeatureColumns(
        numeric=numeric,
        categorical=categorical,
        boolean=boolean,
    )


def _looks_boolean(series: pd.Series) -> bool:
    non_null_values = set(series.dropna().unique().tolist())
    return bool(non_null_values) and non_null_values <= {0, 1, 0.0, 1.0}


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


class StratifiedHybridImputer(BaseEstimator, TransformerMixin):
    """Impute missing values from cohort, fallback-group, then global statistics."""

    def __init__(
        self,
        *,
        categorical_cols: tuple[str, ...] = (),
        numeric_cols: tuple[str, ...] = (),
        categorical_group_cols: tuple[str, ...] = ("branch", "client_id"),
        numeric_group_cols: tuple[str, ...] = ("client_id", "parcel_category"),
        fallback_group_col: str = "branch",
        min_samples: int = 1,
        add_missing_indicators: bool = True,
    ) -> None:
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.categorical_group_cols = categorical_group_cols
        self.numeric_group_cols = numeric_group_cols
        self.fallback_group_col = fallback_group_col
        self.min_samples = min_samples
        self.add_missing_indicators = add_missing_indicators

    def fit(self, X: pd.DataFrame, y: Any = None) -> StratifiedHybridImputer:
        X_df = pd.DataFrame(X).copy()
        self.cohort_stats_: dict[str, pd.Series] = {}
        self.fallback_stats_: dict[str, pd.Series] = {}
        self.global_stats_: dict[str, Any] = {}

        self._validate_group_columns(X_df)

        for column in self.categorical_cols:
            if column not in X_df.columns:
                continue
            self.cohort_stats_[column] = X_df.groupby(list(self.categorical_group_cols))[
                column
            ].apply(self._mode)
            self.fallback_stats_[column] = X_df.groupby(self.fallback_group_col)[
                column
            ].apply(self._mode)
            mode = X_df[column].mode(dropna=True)
            self.global_stats_[column] = mode.iloc[0] if not mode.empty else pd.NA

        for column in self.numeric_cols:
            if column not in X_df.columns:
                continue
            self.cohort_stats_[column] = X_df.groupby(list(self.numeric_group_cols))[
                column
            ].apply(self._median)
            self.fallback_stats_[column] = X_df.groupby(self.fallback_group_col)[
                column
            ].apply(self._median)
            self.global_stats_[column] = X_df[column].median()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = pd.DataFrame(X).copy()

        for column in self._target_columns:
            if column in X_df.columns and self.add_missing_indicators:
                X_df[f"is_{column}_missing"] = X_df[column].isna().astype(int)

        for column in self._target_columns:
            if column not in X_df.columns or column not in self.global_stats_:
                continue
            missing_mask = X_df[column].isna()
            if not missing_mask.any():
                continue

            group_cols = (
                self.categorical_group_cols
                if column in self.categorical_cols
                else self.numeric_group_cols
            )
            lookup_cols = tuple(dict.fromkeys((*group_cols, self.fallback_group_col)))
            temp = X_df.loc[missing_mask, list(lookup_cols)].copy()

            cohort_values = temp.merge(
                self.cohort_stats_[column].rename("cohort_value"),
                on=list(group_cols),
                how="left",
            )
            cohort_values.index = temp.index
            fallback_values = temp.merge(
                self.fallback_stats_[column].rename("fallback_value"),
                on=self.fallback_group_col,
                how="left",
            )
            fallback_values.index = temp.index
            imputed_values = (
                cohort_values["cohort_value"]
                .fillna(fallback_values["fallback_value"])
                .fillna(self.global_stats_[column])
            )

            if isinstance(X_df[column].dtype, pd.CategoricalDtype):
                for value in imputed_values.dropna().unique():
                    if value not in X_df[column].cat.categories:
                        X_df[column] = X_df[column].cat.add_categories(value)
            if str(X_df[column].dtype) == "Int64":
                X_df[column] = X_df[column].astype("float64")

            X_df.loc[missing_mask, column] = imputed_values

        return X_df

    @property
    def _target_columns(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.categorical_cols, *self.numeric_cols)))

    def _validate_group_columns(self, dataframe: pd.DataFrame) -> None:
        required = {
            *self.categorical_group_cols,
            *self.numeric_group_cols,
            self.fallback_group_col,
        }
        missing = sorted(column for column in required if column not in dataframe.columns)
        if missing:
            raise KeyError(
                "Stratified hybrid imputation requires grouping columns: "
                f"{missing}"
            )

    def _mode(self, values: pd.Series) -> Any:
        valid = values.dropna()
        if len(valid) < self.min_samples:
            return pd.NA
        mode = valid.mode(dropna=True)
        return mode.iloc[0] if not mode.empty else pd.NA

    def _median(self, values: pd.Series) -> Any:
        valid = values.dropna()
        if len(valid) < self.min_samples:
            return pd.NA
        return valid.median()


@dataclass(frozen=True)
class PreprocessingConfig:
    feature_columns: FeatureColumns
    imputer: str = "simple"
    scale_numeric: bool = False
    numeric_imputer_strategy: str = "median"
    categorical_imputer_strategy: str = "most_frequent"
    boolean_imputer_strategy: str = "most_frequent"
    remainder: str = "drop"
    boolean_mapping: dict[Any, int] | None = None
    stratified_categorical_columns: tuple[str, ...] | None = None
    stratified_numeric_columns: tuple[str, ...] | None = None
    stratified_categorical_group_cols: tuple[str, ...] = ("branch", "client_id")
    stratified_numeric_group_cols: tuple[str, ...] = ("client_id", "parcel_category")
    stratified_fallback_group_col: str = "branch"
    stratified_min_samples: int = 1
    add_missing_indicators: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "imputer", self.imputer.lower())
        if self.imputer not in {"simple", "stratified_hybrid"}:
            raise ValueError("imputer must be either 'simple' or 'stratified_hybrid'.")
        for field_name in (
            "stratified_categorical_columns",
            "stratified_numeric_columns",
            "stratified_categorical_group_cols",
            "stratified_numeric_group_cols",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, tuple(value))


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

    use_simple_imputer = config.imputer == "simple"
    numeric_steps: list[tuple[str, object]] = []
    if use_simple_imputer:
        numeric_steps.append(
            ("imputer", SimpleImputer(strategy=config.numeric_imputer_strategy))
        )
    if config.scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    if not numeric_steps:
        numeric_steps.append(("identity", FunctionTransformer()))

    categorical_steps: list[tuple[str, object]] = []
    if use_simple_imputer:
        categorical_steps.append(
            ("imputer", SimpleImputer(strategy=config.categorical_imputer_strategy))
        )
    categorical_steps.append(("onehot", OneHotEncoder(handle_unknown="ignore")))

    categorical_pipeline = Pipeline(steps=categorical_steps)
    boolean_steps: list[tuple[str, object]] = [
        ("encoder", BooleanMappingTransformer(mapping=config.boolean_mapping)),
        ("to_object", FunctionTransformer(_cast_to_object)),
        ("imputer", SimpleImputer(strategy=config.boolean_imputer_strategy)),
        ("to_int", FunctionTransformer(_cast_to_int)),
    ]
    boolean_pipeline = Pipeline(steps=boolean_steps)

    transformers: list[tuple[str, Pipeline, list[str] | tuple[str, ...]]] = []
    if config.feature_columns.numeric:
        transformers.append(
            ("numeric", Pipeline(numeric_steps), config.feature_columns.numeric)
        )
    if config.feature_columns.categorical:
        transformers.append(
            ("categorical", categorical_pipeline, config.feature_columns.categorical)
        )
    if config.feature_columns.boolean:
        transformers.append(
            ("boolean", boolean_pipeline, config.feature_columns.boolean)
        )
    indicator_columns = _stratified_indicator_columns(config)
    if indicator_columns:
        transformers.append(
            (
                "missing_indicators",
                Pipeline([("identity", FunctionTransformer())]),
                indicator_columns,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder=config.remainder)


def build_imputer(config: PreprocessingConfig) -> StratifiedHybridImputer | None:
    if config.imputer == "simple":
        return None
    return StratifiedHybridImputer(
        categorical_cols=(
            config.stratified_categorical_columns
            if config.stratified_categorical_columns is not None
            else config.feature_columns.categorical
        ),
        numeric_cols=(
            config.stratified_numeric_columns
            if config.stratified_numeric_columns is not None
            else config.feature_columns.numeric
        ),
        categorical_group_cols=config.stratified_categorical_group_cols,
        numeric_group_cols=config.stratified_numeric_group_cols,
        fallback_group_col=config.stratified_fallback_group_col,
        min_samples=config.stratified_min_samples,
        add_missing_indicators=config.add_missing_indicators,
    )


def _stratified_indicator_columns(config: PreprocessingConfig) -> tuple[str, ...]:
    if config.imputer != "stratified_hybrid" or not config.add_missing_indicators:
        return ()
    categorical_columns = (
        config.stratified_categorical_columns
        if config.stratified_categorical_columns is not None
        else config.feature_columns.categorical
    )
    numeric_columns = (
        config.stratified_numeric_columns
        if config.stratified_numeric_columns is not None
        else config.feature_columns.numeric
    )
    target_columns = tuple(dict.fromkeys((*categorical_columns, *numeric_columns)))
    return tuple(f"is_{column}_missing" for column in target_columns)


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
    imputer = build_imputer(config)
    if imputer is not None:
        steps.append(("imputer", imputer))

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
    configured_columns = (
        list(columns.numeric) + list(columns.categorical) + list(columns.boolean or ())
    )
    missing_columns = [
        column for column in configured_columns if column not in dataframe.columns
    ]
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
