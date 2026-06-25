from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import (
    SelectorMixin,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer, IterativeImputer, MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

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


class HashingEncoder(BaseEstimator, TransformerMixin):
    """Encode categories using feature hashing (the hashing trick).

    All categorical columns are hashed into a single fixed-size dense
    output vector of dimension ``n_features``, making it suitable for
    high-cardinality columns where one-hot encoding would produce too
    many output columns.

    Parameters
    ----------
    n_features : int, default=128
        Size of the output feature space shared across all columns.

    Notes
    -----
    Output column names follow the pattern ``hash_0``, ``hash_1``, …,
    ``hash_{n_features-1}``.
    """

    def __init__(self, *, n_features: int = 128) -> None:
        self.n_features = n_features

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> HashingEncoder:
        X_df = _as_dataframe(X)
        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        from sklearn.feature_extraction import FeatureHasher

        X_df = _as_dataframe(X, tuple(self.feature_names_in_))
        records = [
            {
                f"{col}={'' if pd.isna(val) else val}": 1
                for col, val in zip(X_df.columns, row)
            }
            for row in X_df.itertuples(index=False, name=None)
        ]
        hasher = FeatureHasher(
            n_features=self.n_features,
            input_type="dict",
            alternate_sign=False,
        )
        return hasher.transform(records).toarray()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(
            [f"hash_{i}" for i in range(self.n_features)], dtype=object
        )


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
        categorical_group_cols: tuple[str, ...] = (),
        numeric_group_cols: tuple[str, ...] = (),
        fallback_group_col: str | None = None,
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
            if self.categorical_group_cols:
                self.cohort_stats_[column] = X_df.groupby(list(self.categorical_group_cols))[
                    column
                ].apply(self._mode)
            if self.fallback_group_col is not None:
                self.fallback_stats_[column] = X_df.groupby(self.fallback_group_col)[
                    column
                ].apply(self._mode)
            mode = X_df[column].mode(dropna=True)
            self.global_stats_[column] = mode.iloc[0] if not mode.empty else pd.NA

        for column in self.numeric_cols:
            if column not in X_df.columns:
                continue
            if self.numeric_group_cols:
                self.cohort_stats_[column] = X_df.groupby(list(self.numeric_group_cols))[
                    column
                ].apply(self._median)
            if self.fallback_group_col is not None:
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
            # When no group columns are configured, skip cohort lookup and use global stats
            if not group_cols:
                imputed_values = pd.Series(
                    self.global_stats_[column], index=X_df.index[missing_mask]
                )
            else:
                lookup_cols = tuple(dict.fromkeys((*group_cols, self.fallback_group_col)))
                temp = X_df.loc[missing_mask, [c for c in lookup_cols if c is not None]].copy()

                cohort_values = temp.merge(
                    self.cohort_stats_[column].rename("cohort_value"),
                    on=list(group_cols),
                    how="left",
                )
                cohort_values.index = temp.index

                if self.fallback_group_col is not None and self.fallback_group_col in self.fallback_stats_:
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
                else:
                    imputed_values = (
                        cohort_values["cohort_value"]
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
        }
        if self.fallback_group_col is not None:
            required.add(self.fallback_group_col)
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
class FeatureSelectionConfig:
    enabled: bool = False
    mutual_information: bool = False
    vif: bool = False
    mi_strategy: str = "percentile"
    mi_k: int = 20
    mi_percentile: float = 50.0
    mi_threshold: float = 0.0
    mi_min_features: int = 1
    mi_random_state: int = 0
    vif_threshold: float = 10.0
    vif_min_features: int = 1
    vif_max_iter: int | None = None
    vif_max_features: int = 200

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "enabled",
            bool(self.enabled or self.mutual_information or self.vif),
        )
        strategy = self.mi_strategy.lower()
        if strategy not in {"k_best", "percentile", "threshold"}:
            raise ValueError(
                "feature_selection.mi_strategy must be one of: "
                "['k_best', 'percentile', 'threshold']."
            )
        object.__setattr__(self, "mi_strategy", strategy)

        if self.enabled and not (self.mutual_information or self.vif):
            raise ValueError(
                "Feature selection is enabled, but no selector is active. "
                "Set mutual_information=true and/or vif=true."
            )
        if self.mi_k < 1:
            raise ValueError("feature_selection.mi_k must be at least 1.")
        if not 0.0 < self.mi_percentile <= 100.0:
            raise ValueError(
                "feature_selection.mi_percentile must satisfy 0.0 < percentile <= 100.0."
            )
        if self.mi_min_features < 1:
            raise ValueError("feature_selection.mi_min_features must be at least 1.")
        if self.vif_threshold < 1.0:
            raise ValueError("feature_selection.vif_threshold must be at least 1.0.")
        if self.vif_min_features < 1:
            raise ValueError("feature_selection.vif_min_features must be at least 1.")
        if self.vif_max_iter is not None and self.vif_max_iter < 1:
            raise ValueError("feature_selection.vif_max_iter must be at least 1.")
        if self.vif_max_features < 1:
            raise ValueError("feature_selection.vif_max_features must be at least 1.")


class MISelector(BaseEstimator, SelectorMixin):
    """Select post-preprocessing features by univariate mutual information."""

    def __init__(
        self,
        *,
        task: str,
        strategy: str = "percentile",
        k: int = 20,
        percentile: float = 50.0,
        threshold: float = 0.0,
        min_features: int = 1,
        random_state: int = 0,
        discrete_features: str | bool | list[bool] = "auto",
    ) -> None:
        self.task = task
        self.strategy = strategy
        self.k = k
        self.percentile = percentile
        self.threshold = threshold
        self.min_features = min_features
        self.random_state = random_state
        self.discrete_features = discrete_features

    def fit(self, X: Any, y: Any = None) -> MISelector:
        self._validate_parameters()
        if y is None:
            raise ValueError("MISelector requires y during fit.")
        X_checked, y_checked = check_X_y(
            X,
            y,
            accept_sparse=("csr", "csc"),
            dtype=float,
        )
        self.n_features_in_ = X_checked.shape[1]

        if self.task == "classification":
            scores = mutual_info_classif(
                X_checked,
                y_checked,
                discrete_features=self.discrete_features,
                random_state=self.random_state,
            )
        elif self.task == "regression":
            scores = mutual_info_regression(
                X_checked,
                y_checked,
                discrete_features=self.discrete_features,
                random_state=self.random_state,
            )
        else:
            raise ValueError("MISelector task must be 'classification' or 'regression'.")

        self.scores_ = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0)
        self.support_ = self._build_support_mask(self.scores_)
        return self

    def _get_support_mask(self) -> np.ndarray:
        check_is_fitted(self, "support_")
        return self.support_

    def _validate_parameters(self) -> None:
        strategy = self.strategy.lower()
        if strategy not in {"k_best", "percentile", "threshold"}:
            raise ValueError(
                "MISelector strategy must be one of: "
                "['k_best', 'percentile', 'threshold']."
            )
        self.strategy_ = strategy
        if self.k < 1:
            raise ValueError("MISelector k must be at least 1.")
        if not 0.0 < self.percentile <= 100.0:
            raise ValueError("MISelector percentile must satisfy 0.0 < percentile <= 100.0.")
        if self.min_features < 1:
            raise ValueError("MISelector min_features must be at least 1.")

    def _build_support_mask(self, scores: np.ndarray) -> np.ndarray:
        n_features = len(scores)
        min_features = min(self.min_features, n_features)
        if self.strategy_ == "k_best":
            selection_count = min(self.k, n_features)
            self.selection_threshold_ = None
        elif self.strategy_ == "percentile":
            selection_count = int(np.ceil(n_features * (self.percentile / 100.0)))
            selection_count = min(max(selection_count, min_features), n_features)
            self.selection_threshold_ = None
        else:
            mask = scores >= self.threshold
            if int(mask.sum()) < min_features:
                mask = self._top_k_mask(scores, min_features)
            self.selection_threshold_ = float(self.threshold)
            self.selection_count_ = int(mask.sum())
            return mask

        selection_count = max(selection_count, min_features)
        self.selection_count_ = int(selection_count)
        return self._top_k_mask(scores, selection_count)

    def _top_k_mask(self, scores: np.ndarray, k: int) -> np.ndarray:
        order = np.argsort(-scores, kind="mergesort")
        mask = np.zeros(len(scores), dtype=bool)
        mask[order[:k]] = True
        return mask


class IterativeVIFSelector(BaseEstimator, SelectorMixin):
    """Iteratively drop features whose VIF exceeds a configured threshold."""

    def __init__(
        self,
        *,
        threshold: float = 10.0,
        min_features: int = 1,
        max_iter: int | None = None,
        max_features: int = 200,
    ) -> None:
        self.threshold = threshold
        self.min_features = min_features
        self.max_iter = max_iter
        self.max_features = max_features

    def fit(self, X: Any, y: Any = None) -> IterativeVIFSelector:
        self._validate_parameters()
        X_checked = check_array(X, accept_sparse=("csr", "csc"), dtype=float)
        self.n_features_in_ = X_checked.shape[1]
        if self.n_features_in_ > self.max_features:
            raise ValueError(
                "IterativeVIFSelector received "
                f"{self.n_features_in_} features, which exceeds max_features="
                f"{self.max_features}. Reduce dimensionality first or raise "
                "feature_selection.vif_max_features."
            )

        values = self._as_dense_array(X_checked)
        support = np.ones(self.n_features_in_, dtype=bool)
        self.vif_scores_ = np.full(self.n_features_in_, np.nan, dtype=float)
        self.vif_history_: list[dict[str, Any]] = []

        iteration = 0
        while int(support.sum()) > self.min_features:
            if self.max_iter is not None and iteration >= self.max_iter:
                break
            selected_indices = np.flatnonzero(support)
            current_vifs = self._calculate_vifs(values[:, selected_indices])
            self.vif_scores_[selected_indices] = current_vifs

            max_position = int(np.nanargmax(current_vifs))
            max_vif = float(current_vifs[max_position])
            drop_index = int(selected_indices[max_position])
            should_drop = max_vif > self.threshold
            self.vif_history_.append(
                {
                    "iteration": iteration,
                    "feature_index": drop_index,
                    "vif": max_vif,
                    "dropped": bool(should_drop),
                }
            )
            if not should_drop:
                break
            support[drop_index] = False
            iteration += 1

        if int(support.sum()) > 1:
            selected_indices = np.flatnonzero(support)
            self.vif_scores_[selected_indices] = self._calculate_vifs(
                values[:, selected_indices]
            )
        self.support_ = support
        self.selection_threshold_ = float(self.threshold)
        return self

    def _get_support_mask(self) -> np.ndarray:
        check_is_fitted(self, "support_")
        return self.support_

    def _validate_parameters(self) -> None:
        if self.threshold < 1.0:
            raise ValueError("IterativeVIFSelector threshold must be at least 1.0.")
        if self.min_features < 1:
            raise ValueError("IterativeVIFSelector min_features must be at least 1.")
        if self.max_iter is not None and self.max_iter < 1:
            raise ValueError("IterativeVIFSelector max_iter must be at least 1.")
        if self.max_features < 1:
            raise ValueError("IterativeVIFSelector max_features must be at least 1.")

    def _as_dense_array(self, X: Any) -> np.ndarray:
        from scipy import sparse

        if sparse.issparse(X):
            return X.toarray()
        return np.asarray(X, dtype=float)

    def _calculate_vifs(self, values: np.ndarray) -> np.ndarray:
        n_features = values.shape[1]
        if n_features == 1:
            return np.ones(1, dtype=float)

        vifs = np.empty(n_features, dtype=float)
        for idx in range(n_features):
            target = values[:, idx]
            predictors = np.delete(values, idx, axis=1)
            predictors = np.column_stack([np.ones(len(predictors)), predictors])
            try:
                coefficients, *_ = np.linalg.lstsq(predictors, target, rcond=None)
                fitted = predictors @ coefficients
                ss_res = float(np.sum((target - fitted) ** 2))
                ss_tot = float(np.sum((target - np.mean(target)) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
                vifs[idx] = float("inf") if r2 >= 1.0 else float(1.0 / (1.0 - r2))
            except np.linalg.LinAlgError:
                vifs[idx] = float("inf")
        return vifs


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
    categorical_onehot_max_cardinality: int = 10
    hashing_n_features: int = 128
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
    stratified_categorical_group_cols: tuple[str, ...] = ()
    stratified_numeric_group_cols: tuple[str, ...] = ()
    stratified_fallback_group_col: str | None = None
    stratified_min_samples: int = 1
    add_missing_indicators: bool = True
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)

    def __post_init__(self) -> None:
        def _validate_enum(
            field_name: str,
            value: str,
            allowed: set[str],
            *,
            error_msg: str | None = None,
        ) -> None:
            normed = value.lower()
            if normed not in allowed:
                label = error_msg or f"{field_name} must be one of: {sorted(allowed)}."
                raise ValueError(label)
            object.__setattr__(self, field_name, normed)

        if isinstance(self.feature_selection, dict):
            object.__setattr__(
                self,
                "feature_selection",
                FeatureSelectionConfig(**self.feature_selection),
            )
        _validate_enum(
            "imputer",
            self.imputer,
            {"simple", "stratified_hybrid", "knn", "iterative"},
        )
        _validate_enum(
            "numeric_scaler",
            self.numeric_scaler,
            {"none", "standard", "robust", "minmax"},
        )
        _validate_enum(
            "numeric_power_transform",
            self.numeric_power_transform,
            {"none", "yeo_johnson", "box_cox"},
        )
        _validate_enum(
            "numeric_distribution_transform",
            self.numeric_distribution_transform,
            {"none", "quantile_uniform", "quantile_normal"},
        )
        if (
            self.numeric_power_transform != "none"
            and self.numeric_distribution_transform != "none"
        ):
            raise ValueError(
                "numeric_power_transform and numeric_distribution_transform "
                "are mutually exclusive."
            )
        _validate_enum(
            "numeric_binning",
            self.numeric_binning,
            {"none", "uniform", "quantile", "kmeans"},
        )
        if self.numeric_bin_count < 2:
            raise ValueError("numeric_bin_count must be at least 2.")
        _validate_enum(
            "categorical_encoding",
            self.categorical_encoding,
            {"onehot", "frequency", "ordinal", "target", "hashing"},
        )
        if self.categorical_onehot_max_cardinality < 0:
            raise ValueError(
                "categorical_onehot_max_cardinality must be >= 0 (0 disables the warning)."
            )
        if self.hashing_n_features < 1:
            raise ValueError("hashing_n_features must be at least 1.")
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

    # ── Cardinality warning for one-hot encoding ───────────────────────────
    if (
        config.categorical_encoding == "onehot"
        and config.categorical_onehot_max_cardinality > 0
        and config.feature_columns.categorical
    ):
        import warnings as _warnings

        for col in config.feature_columns.categorical:
            if col in dataframe.columns:
                n_unique = int(dataframe[col].nunique(dropna=True))
                if n_unique > config.categorical_onehot_max_cardinality:
                    _warnings.warn(
                        f"Column '{col}' has {n_unique} unique values and will be "
                        f"one-hot encoded, producing {n_unique} output columns. "
                        f"Consider 'hashing', 'frequency', or 'target' encoding for "
                        f"high-cardinality columns "
                        f"(categorical_onehot_max_cardinality="
                        f"{config.categorical_onehot_max_cardinality}).",
                        UserWarning,
                        stacklevel=2,
                    )

    use_simple_imputer = config.imputer in {"simple", "knn", "iterative"}

    transformers: list[tuple[str, Pipeline, list[str] | tuple[str, ...]]] = []
    if config.feature_columns.numeric:
        transformers.append(
            ("numeric", Pipeline(_build_numeric_steps(config, len(dataframe), use_simple_imputer)), config.feature_columns.numeric)
        )
    if config.feature_columns.categorical:
        transformers.append(
            ("categorical", Pipeline(_build_categorical_steps(config, use_simple_imputer)), config.feature_columns.categorical)
        )
    if config.feature_columns.boolean:
        transformers.append(
            ("boolean", Pipeline(_build_boolean_steps(config)), config.feature_columns.boolean)
        )
    if config.feature_columns.datetime:
        transformers.append(
            (
                "datetime",
                Pipeline([
                    ("extractor", DateTimeFeatureExtractor()),
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                config.feature_columns.datetime,
            )
        )
    if config.feature_columns.text:
        transformers.append(
            (
                "text",
                Pipeline([("tfidf", MultiColumnTfidfVectorizer(max_features=config.text_max_features))]),
                config.feature_columns.text,
            )
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


def _build_numeric_steps(
    config: PreprocessingConfig,
    n_rows: int,
    use_simple_imputer: bool,
) -> list[tuple[str, object]]:
    """Build the ordered transformer steps for the numeric pipeline."""
    steps: list[tuple[str, object]] = []
    if use_simple_imputer:
        if config.imputer == "knn" or config.numeric_imputer_strategy == "knn":
            steps.append(("imputer", KNNImputer()))
        elif config.imputer == "iterative" or config.numeric_imputer_strategy in {"iterative", "mice"}:
            steps.append(("imputer", IterativeImputer(random_state=0)))
        else:
            steps.append(("imputer", SimpleImputer(strategy=config.numeric_imputer_strategy)))
    if config.cap_numeric_quantiles:
        steps.append((
            "quantile_capper",
            QuantileCapper(
                lower_quantile=config.quantile_cap_lower,
                upper_quantile=config.quantile_cap_upper,
            ),
        ))
    if config.numeric_power_transform == "yeo_johnson":
        steps.append(("power", PowerTransformer(method="yeo-johnson", standardize=False)))
    elif config.numeric_power_transform == "box_cox":
        steps.append(("power", PositiveBoxCoxTransformer(standardize=False)))
    if config.numeric_distribution_transform != "none":
        distribution = (
            "normal"
            if config.numeric_distribution_transform == "quantile_normal"
            else "uniform"
        )
        steps.append((
            "distribution",
            QuantileTransformer(
                n_quantiles=min(config.quantile_transform_n_quantiles, n_rows),
                output_distribution=distribution,
                random_state=0,
            ),
        ))
    if config.numeric_binning != "none":
        steps.append((
            "binning",
            KBinsDiscretizer(
                n_bins=config.numeric_bin_count,
                encode="onehot-dense",
                strategy=config.numeric_binning,
            ),
        ))
    scaler = _numeric_scaler(config)
    if scaler is not None:
        steps.append(("scaler", scaler))
    if not steps:
        steps.append(("identity", FunctionTransformer()))
    return steps


def _build_categorical_steps(
    config: PreprocessingConfig,
    use_simple_imputer: bool,
) -> list[tuple[str, object]]:
    """Build the ordered transformer steps for the categorical pipeline."""
    steps: list[tuple[str, object]] = []
    if use_simple_imputer:
        steps.append(("imputer", SimpleImputer(strategy=config.categorical_imputer_strategy)))
    if config.group_rare_categories:
        steps.append((
            "rare_categories",
            RareCategoryGrouper(min_frequency=config.rare_category_min_frequency),
        ))
    if config.categorical_encoding == "onehot":
        steps.append(("onehot", OneHotEncoder(handle_unknown="ignore")))
    elif config.categorical_encoding == "frequency":
        steps.append((
            "frequency",
            FrequencyEncoder(unknown_value=config.frequency_unknown_value),
        ))
    elif config.categorical_encoding == "ordinal":
        steps.append((
            "ordinal",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-2,
            ),
        ))
    elif config.categorical_encoding == "hashing":
        steps.append(("hashing", HashingEncoder(n_features=config.hashing_n_features)))
    else:
        steps.append(("target", TargetEncoder()))
    return steps


def _build_boolean_steps(
    config: PreprocessingConfig,
) -> list[tuple[str, object]]:
    """Build the ordered transformer steps for the boolean pipeline."""
    return [
        ("encoder", BooleanMappingTransformer(mapping=config.boolean_mapping)),
        ("to_object", FunctionTransformer(_cast_to_object)),
        ("imputer", SimpleImputer(strategy=config.boolean_imputer_strategy)),
        ("to_int", FunctionTransformer(_cast_to_int)),
    ]


def _numeric_scaler(config: PreprocessingConfig) -> object | None:
    if config.numeric_scaler == "standard" or (
        config.scale_numeric and config.numeric_scaler == "none"
    ):
        return StandardScaler()
    if config.numeric_scaler == "robust":
        return RobustScaler()
    if config.numeric_scaler == "minmax":
        return MinMaxScaler()
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
    task: str | None = None,
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
    ])
    steps.extend(build_feature_selection_steps(config=config, task=task))
    steps.append(("model", estimator))
    return Pipeline(steps=steps)


def build_feature_selection_steps(
    *,
    config: PreprocessingConfig,
    task: str | None = None,
) -> list[tuple[str, Any]]:
    selection = config.feature_selection
    if not selection.enabled:
        return []

    steps: list[tuple[str, Any]] = []
    if selection.mutual_information:
        if task not in {"classification", "regression"}:
            raise ValueError(
                "Mutual-information feature selection requires task to be "
                "'classification' or 'regression'."
            )
        steps.append(
            (
                "feature_selection_mi",
                MISelector(
                    task=task,
                    strategy=selection.mi_strategy,
                    k=selection.mi_k,
                    percentile=selection.mi_percentile,
                    threshold=selection.mi_threshold,
                    min_features=selection.mi_min_features,
                    random_state=selection.mi_random_state,
                ),
            )
        )
    if selection.vif:
        steps.append(
            (
                "feature_selection_vif",
                IterativeVIFSelector(
                    threshold=selection.vif_threshold,
                    min_features=selection.vif_min_features,
                    max_iter=selection.vif_max_iter,
                    max_features=selection.vif_max_features,
                ),
            )
        )
    return steps


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
