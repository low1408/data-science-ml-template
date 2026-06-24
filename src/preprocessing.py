from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)

from src.features import FeaturePipeline, FeatureRole, SklearnFeaturePipeline


@dataclass(frozen=True)
class FeatureColumns:
    numeric: list[str] | tuple[str, ...]
    categorical: list[str] | tuple[str, ...]
    boolean: list[str] | tuple[str, ...] | None = None
    datetime: list[str] | tuple[str, ...] | None = None
    text: list[str] | tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "numeric", tuple(self.numeric))
        object.__setattr__(self, "categorical", tuple(self.categorical))
        object.__setattr__(self, "boolean", tuple(self.boolean or ()))
        object.__setattr__(self, "datetime", tuple(self.datetime or ()))
        object.__setattr__(self, "text", tuple(self.text or ()))


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
    datetime: list[str] = []

    for column in dataframe.columns:
        if column == target_column:
            continue
        series = dataframe[column]
        if is_bool_dtype(series):
            boolean.append(column)
        elif is_datetime64_any_dtype(series):
            datetime.append(column)
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
        datetime=datetime,
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


def _as_dataframe(
    X: pd.DataFrame | np.ndarray,
    columns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    return pd.DataFrame(X, columns=columns)


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


class QuantileCapper(BaseEstimator, TransformerMixin):
    """Clip numeric columns to quantiles learned from the training data."""

    def __init__(
        self,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> QuantileCapper:
        self._validate_quantiles()
        X_df = _as_dataframe(X)
        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        numeric = X_df.apply(pd.to_numeric, errors="coerce")
        self.lower_bounds_ = numeric.quantile(self.lower_quantile)
        self.upper_bounds_ = numeric.quantile(self.upper_quantile)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_df = _as_dataframe(X, tuple(self.feature_names_in_))
        numeric = X_df.apply(pd.to_numeric, errors="coerce")
        clipped = numeric.clip(
            lower=self.lower_bounds_,
            upper=self.upper_bounds_,
            axis="columns",
        )
        return clipped.to_numpy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return self.feature_names_in_

    def _validate_quantiles(self) -> None:
        if not 0.0 <= self.lower_quantile < self.upper_quantile <= 1.0:
            raise ValueError(
                "QuantileCapper requires 0.0 <= lower_quantile < upper_quantile <= 1.0."
            )


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encode categories by their training-set relative frequency."""

    def __init__(
        self,
        *,
        unknown_value: float = 0.0,
        missing_label: str = "__MISSING__",
    ) -> None:
        self.unknown_value = unknown_value
        self.missing_label = missing_label

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> FrequencyEncoder:
        X_df = _as_dataframe(X)
        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        denominator = len(X_df)
        self.frequencies_: dict[str, pd.Series] = {}
        for column in X_df.columns:
            key = self._working_key(X_df[column])
            self.frequencies_[column] = key.value_counts(dropna=False) / denominator
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_df = _as_dataframe(X, tuple(self.feature_names_in_))
        encoded = pd.DataFrame(index=X_df.index)
        for column in X_df.columns:
            key = self._working_key(X_df[column])
            encoded[column] = key.map(self.frequencies_[column]).fillna(
                self.unknown_value
            )
        return encoded.astype("float64").to_numpy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        feature_names = (
            input_features if input_features is not None else self.feature_names_in_
        )
        return np.asarray(feature_names, dtype=object)

    def _working_key(self, series: pd.Series) -> pd.Series:
        return series.astype("object").where(series.notna(), self.missing_label)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Group low-frequency categories under a stable rare-category label."""

    def __init__(
        self,
        *,
        min_frequency: float = 0.01,
        rare_label: str = "__RARE__",
        missing_label: str = "__MISSING__",
    ) -> None:
        self.min_frequency = min_frequency
        self.rare_label = rare_label
        self.missing_label = missing_label

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> RareCategoryGrouper:
        if not 0.0 <= self.min_frequency <= 1.0:
            raise ValueError("min_frequency must be between 0.0 and 1.0.")
        X_df = _as_dataframe(X)
        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        self.frequent_categories_: dict[str, set[Any]] = {}
        for column in X_df.columns:
            key = self._working_key(X_df[column])
            frequencies = key.value_counts(dropna=False) / len(key)
            self.frequent_categories_[column] = set(
                frequencies[frequencies >= self.min_frequency].index
            )
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        X_df = _as_dataframe(X, tuple(self.feature_names_in_))
        grouped = pd.DataFrame(index=X_df.index)
        for column in X_df.columns:
            key = self._working_key(X_df[column])
            grouped[column] = key.where(
                key.isin(self.frequent_categories_[column]),
                self.rare_label,
            )
        return grouped

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        feature_names = (
            input_features if input_features is not None else self.feature_names_in_
        )
        return np.asarray(feature_names, dtype=object)

    def _working_key(self, series: pd.Series) -> pd.Series:
        return series.astype("object").where(series.notna(), self.missing_label)


class PositiveBoxCoxTransformer(BaseEstimator, TransformerMixin):
    """Box-Cox power transform with explicit positive-value validation."""

    def __init__(self, *, standardize: bool = False) -> None:
        self.standardize = standardize

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Any = None,
    ) -> PositiveBoxCoxTransformer:
        X_df = _as_dataframe(X)
        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        values = X_df.apply(pd.to_numeric, errors="coerce").to_numpy()
        self._validate_positive(values)
        self.transformer_ = PowerTransformer(
            method="box-cox",
            standardize=self.standardize,
        )
        self.transformer_.fit(values)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_df = _as_dataframe(X, tuple(self.feature_names_in_))
        values = X_df.apply(pd.to_numeric, errors="coerce").to_numpy()
        self._validate_positive(values)
        return self.transformer_.transform(values)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return self.feature_names_in_

    def _validate_positive(self, values: np.ndarray) -> None:
        valid_values = values[~np.isnan(values)]
        if valid_values.size and np.any(valid_values <= 0):
            raise ValueError(
                "Box-Cox power transform requires strictly positive numeric values."
            )


class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract stable calendar parts from datetime-like columns."""

    output_parts: tuple[str, ...] = ("year", "month", "day", "dayofweek")

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Any = None,
    ) -> DateTimeFeatureExtractor:
        X_df = _as_dataframe(X)
        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_df = _as_dataframe(X, tuple(self.feature_names_in_))
        output = pd.DataFrame(index=X_df.index)
        for column in X_df.columns:
            values = pd.to_datetime(X_df[column], errors="coerce")
            output[f"{column}_year"] = values.dt.year
            output[f"{column}_month"] = values.dt.month
            output[f"{column}_day"] = values.dt.day
            output[f"{column}_dayofweek"] = values.dt.dayofweek
        return output.astype("float64").to_numpy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        feature_names = (
            input_features if input_features is not None else self.feature_names_in_
        )
        return np.asarray(
            [
                f"{column}_{part}"
                for column in feature_names
                for part in self.output_parts
            ],
            dtype=object,
        )


class MultiColumnTfidfVectorizer(BaseEstimator, TransformerMixin):
    """Apply one TfidfVectorizer per text column and concatenate the results."""

    def __init__(self, *, max_features: int = 1000) -> None:
        self.max_features = max_features

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Any = None,
    ) -> MultiColumnTfidfVectorizer:
        X_df = _as_dataframe(X)
        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        self.vectorizers_: dict[str, TfidfVectorizer] = {}
        for column in X_df.columns:
            vectorizer = TfidfVectorizer(max_features=self.max_features)
            vectorizer.fit(self._documents(X_df[column]))
            self.vectorizers_[column] = vectorizer
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> Any:
        from scipy import sparse

        X_df = _as_dataframe(X, tuple(self.feature_names_in_))
        matrices = [
            self.vectorizers_[column].transform(self._documents(X_df[column]))
            for column in X_df.columns
        ]
        if not matrices:
            return sparse.csr_matrix((len(X_df), 0))
        return sparse.hstack(matrices, format="csr")

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        columns = (
            input_features if input_features is not None else self.feature_names_in_
        )
        names: list[str] = []
        for column in columns:
            terms = self.vectorizers_[column].get_feature_names_out()
            names.extend(f"{column}__tfidf_{term}" for term in terms)
        return np.asarray(names, dtype=object)

    def _documents(self, series: pd.Series) -> pd.Series:
        documents = series.fillna("").astype(str)
        return documents.mask(documents.str.strip() == "", "__empty__")


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
    numeric_scaler: str = "none"
    cap_numeric_quantiles: bool = False
    quantile_cap_lower: float = 0.01
    quantile_cap_upper: float = 0.99
    numeric_power_transform: str = "none"
    numeric_distribution_transform: str = "none"
    quantile_transform_n_quantiles: int = 1000
    numeric_binning: str = "none"
    numeric_bin_count: int = 10
    categorical_encoding: str = "onehot"
    group_rare_categories: bool = False
    rare_category_min_frequency: float = 0.01
    frequency_unknown_value: float = 0.0
    add_simple_missing_indicators: bool = False
    numeric_imputer_strategy: str = "median"
    categorical_imputer_strategy: str = "most_frequent"
    boolean_imputer_strategy: str = "most_frequent"
    remainder: str = "drop"
    boolean_mapping: dict[Any, int] | None = None
    text_max_features: int = 1000
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
        object.__setattr__(self, "numeric_scaler", self.numeric_scaler.lower())
        if self.numeric_scaler not in {"none", "standard", "robust"}:
            raise ValueError(
                "numeric_scaler must be one of: 'none', 'standard', 'robust'."
            )
        object.__setattr__(
            self,
            "numeric_power_transform",
            self.numeric_power_transform.lower(),
        )
        if self.numeric_power_transform not in {"none", "yeo_johnson", "box_cox"}:
            raise ValueError(
                "numeric_power_transform must be one of: "
                "'none', 'yeo_johnson', 'box_cox'."
            )
        object.__setattr__(
            self,
            "numeric_distribution_transform",
            self.numeric_distribution_transform.lower(),
        )
        if self.numeric_distribution_transform not in {
            "none",
            "quantile_uniform",
            "quantile_normal",
        }:
            raise ValueError(
                "numeric_distribution_transform must be one of: "
                "'none', 'quantile_uniform', 'quantile_normal'."
            )
        if (
            self.numeric_power_transform != "none"
            and self.numeric_distribution_transform != "none"
        ):
            raise ValueError(
                "numeric_power_transform and numeric_distribution_transform "
                "are mutually exclusive."
            )
        object.__setattr__(self, "numeric_binning", self.numeric_binning.lower())
        if self.numeric_binning not in {"none", "uniform", "quantile", "kmeans"}:
            raise ValueError(
                "numeric_binning must be one of: "
                "'none', 'uniform', 'quantile', 'kmeans'."
            )
        if self.numeric_bin_count < 2:
            raise ValueError("numeric_bin_count must be at least 2.")
        object.__setattr__(
            self,
            "categorical_encoding",
            self.categorical_encoding.lower(),
        )
        if self.categorical_encoding not in {
            "onehot",
            "frequency",
            "ordinal",
            "target",
        }:
            raise ValueError(
                "categorical_encoding must be one of: "
                "'onehot', 'frequency', 'ordinal', 'target'."
            )
        if not 0.0 <= self.quantile_cap_lower < self.quantile_cap_upper <= 1.0:
            raise ValueError(
                "Quantile caps require "
                "0.0 <= quantile_cap_lower < quantile_cap_upper <= 1.0."
            )
        if not 0.0 <= self.rare_category_min_frequency <= 1.0:
            raise ValueError(
                "rare_category_min_frequency must be between 0.0 and 1.0."
            )
        if self.quantile_transform_n_quantiles < 1:
            raise ValueError("quantile_transform_n_quantiles must be at least 1.")
        if self.text_max_features < 1:
            raise ValueError("text_max_features must be at least 1.")
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
    if config.cap_numeric_quantiles:
        numeric_steps.append(
            (
                "quantile_capper",
                QuantileCapper(
                    lower_quantile=config.quantile_cap_lower,
                    upper_quantile=config.quantile_cap_upper,
                ),
            )
        )
    if config.numeric_power_transform == "yeo_johnson":
        numeric_steps.append(
            ("power", PowerTransformer(method="yeo-johnson", standardize=False))
        )
    elif config.numeric_power_transform == "box_cox":
        numeric_steps.append(("power", PositiveBoxCoxTransformer(standardize=False)))
    if config.numeric_distribution_transform != "none":
        distribution = (
            "normal"
            if config.numeric_distribution_transform == "quantile_normal"
            else "uniform"
        )
        n_quantiles = min(config.quantile_transform_n_quantiles, len(dataframe))
        numeric_steps.append(
            (
                "distribution",
                QuantileTransformer(
                    n_quantiles=n_quantiles,
                    output_distribution=distribution,
                    random_state=0,
                ),
            )
        )
    if config.numeric_binning != "none":
        numeric_steps.append(
            (
                "binning",
                KBinsDiscretizer(
                    n_bins=config.numeric_bin_count,
                    encode="onehot-dense",
                    strategy=config.numeric_binning,
                ),
            )
        )
    scaler = _numeric_scaler(config)
    if scaler is not None:
        numeric_steps.append(("scaler", scaler))
    if not numeric_steps:
        numeric_steps.append(("identity", FunctionTransformer()))

    categorical_steps: list[tuple[str, object]] = []
    if use_simple_imputer:
        categorical_steps.append(
            ("imputer", SimpleImputer(strategy=config.categorical_imputer_strategy))
        )
    if config.group_rare_categories:
        categorical_steps.append(
            (
                "rare_categories",
                RareCategoryGrouper(
                    min_frequency=config.rare_category_min_frequency
                ),
            )
        )
    if config.categorical_encoding == "onehot":
        categorical_steps.append(("onehot", OneHotEncoder(handle_unknown="ignore")))
    elif config.categorical_encoding == "frequency":
        categorical_steps.append(
            (
                "frequency",
                FrequencyEncoder(unknown_value=config.frequency_unknown_value),
            )
        )
    elif config.categorical_encoding == "ordinal":
        categorical_steps.append(
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-2,
                ),
            )
        )
    else:
        categorical_steps.append(("target", TargetEncoder()))

    categorical_pipeline = Pipeline(steps=categorical_steps)
    boolean_steps: list[tuple[str, object]] = [
        ("encoder", BooleanMappingTransformer(mapping=config.boolean_mapping)),
        ("to_object", FunctionTransformer(_cast_to_object)),
        ("imputer", SimpleImputer(strategy=config.boolean_imputer_strategy)),
        ("to_int", FunctionTransformer(_cast_to_int)),
    ]
    boolean_pipeline = Pipeline(steps=boolean_steps)
    datetime_pipeline = Pipeline(
        steps=[
            ("extractor", DateTimeFeatureExtractor()),
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    text_pipeline = Pipeline(
        steps=[
            ("tfidf", MultiColumnTfidfVectorizer(max_features=config.text_max_features)),
        ]
    )

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
    if config.feature_columns.datetime:
        transformers.append(
            ("datetime", datetime_pipeline, config.feature_columns.datetime)
        )
    if config.feature_columns.text:
        transformers.append(("text", text_pipeline, config.feature_columns.text))
    indicator_columns = _stratified_indicator_columns(config)
    if indicator_columns:
        transformers.append(
            (
                "missing_indicators",
                Pipeline([("identity", FunctionTransformer())]),
                indicator_columns,
            )
        )
    simple_indicator_columns = _simple_indicator_columns(config)
    if simple_indicator_columns:
        transformers.append(
            (
                "simple_missing_indicators",
                Pipeline([("indicator", MissingIndicator(features="all"))]),
                simple_indicator_columns,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder=config.remainder)


def _numeric_scaler(config: PreprocessingConfig) -> object | None:
    if config.numeric_scaler == "standard" or (
        config.scale_numeric and config.numeric_scaler == "none"
    ):
        return StandardScaler()
    if config.numeric_scaler == "robust":
        return RobustScaler()
    return None


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


def _simple_indicator_columns(config: PreprocessingConfig) -> tuple[str, ...]:
    if config.imputer != "simple" or not config.add_simple_missing_indicators:
        return ()
    return tuple(
        dict.fromkeys(
            (
                *config.feature_columns.numeric,
                *config.feature_columns.categorical,
                *config.feature_columns.boolean,
                *config.feature_columns.datetime,
            )
        )
    )


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
        config = with_feature_pipeline_columns(config, feature_pipeline)
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


def with_feature_pipeline_columns(
    config: PreprocessingConfig,
    feature_pipeline: FeaturePipeline,
) -> PreprocessingConfig:
    """Return config with generated feature columns added to their declared roles."""
    numeric = list(config.feature_columns.numeric)
    categorical = list(config.feature_columns.categorical)
    boolean = list(config.feature_columns.boolean)
    datetime = list(config.feature_columns.datetime)
    text = list(config.feature_columns.text)
    role_columns: dict[FeatureRole, list[str]] = {
        "numeric": numeric,
        "categorical": categorical,
        "boolean": boolean,
        "datetime": datetime,
        "text": text,
    }

    configured_roles: dict[str, FeatureRole] = {}
    for role, columns in role_columns.items():
        for column in columns:
            existing_role = configured_roles.get(column)
            if existing_role is not None and existing_role != role:
                raise ValueError(
                    "Feature columns can only belong to one role. "
                    f"Column {column!r} is configured as both "
                    f"{existing_role!r} and {role!r}."
                )
            configured_roles[column] = role

    for feature in feature_pipeline.features:
        configured_role = configured_roles.get(feature.name)
        if configured_role is not None:
            if configured_role != feature.role:
                raise ValueError(
                    f"Feature {feature.name!r} declares role {feature.role!r} "
                    f"but is configured as {configured_role!r}."
                )
            continue
        role_columns[feature.role].append(feature.name)
        configured_roles[feature.name] = feature.role

    return replace(
        config,
        feature_columns=FeatureColumns(
            numeric=numeric,
            categorical=categorical,
            boolean=boolean,
            datetime=datetime,
            text=text,
        ),
    )


def _validate_feature_columns(dataframe: pd.DataFrame, columns: FeatureColumns) -> None:
    configured_columns = (
        list(columns.numeric)
        + list(columns.categorical)
        + list(columns.boolean or ())
        + list(columns.datetime or ())
        + list(columns.text or ())
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
