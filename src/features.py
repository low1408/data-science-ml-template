from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

FeatureRole = Literal["numeric", "categorical", "boolean", "datetime", "text"]
VALID_FEATURE_ROLES: tuple[FeatureRole, ...] = (
    "numeric",
    "categorical",
    "boolean",
    "datetime",
    "text",
)


def validate_columns(frame: pd.DataFrame, columns: Sequence[str], frame_name: str) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns for {frame_name}: {missing}")


class Feature(ABC):
    """One-column deterministic feature.

    Expected columns:
        ``requires`` must exist before ``transform`` runs. Dependencies can be
        raw inputs or earlier feature outputs.
    Returned columns:
        ``transform`` returns a single ``pd.Series`` named ``name``.
    Invalid-value behavior:
        Delegated to the concrete feature.
    Missing-value behavior:
        Delegated to the concrete feature.
    Row-count behavior:
        Row count and index must be preserved.
    Learns from data:
        No. Feature classes in this module are deterministic transforms.
    """

    name: str
    requires: tuple[str, ...]
    role: FeatureRole

    def __init__(
        self,
        *,
        name: str,
        requires: Sequence[str],
        role: FeatureRole = "numeric",
    ) -> None:
        if role not in VALID_FEATURE_ROLES:
            raise ValueError(
                "Feature role must be one of "
                f"{list(VALID_FEATURE_ROLES)}, got {role!r}."
            )
        self.name = name
        self.requires = tuple(requires)
        self.role = role

    def transform(self, frame: pd.DataFrame) -> pd.Series:
        validate_columns(frame, self.requires, frame_name=f"feature {self.name!r}")
        output = self._transform(frame)
        if not isinstance(output, pd.Series):
            raise TypeError(f"Feature {self.name!r} must return a pandas Series")
        if not output.index.equals(frame.index):
            raise ValueError(f"Feature {self.name!r} changed the row index")
        if len(output) != len(frame):
            raise ValueError(f"Feature {self.name!r} changed the row count")
        if output.name != self.name:
            raise ValueError(
                f"Feature {self.name!r} returned Series named {output.name!r}"
            )
        return output

    @abstractmethod
    def _transform(self, frame: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    @classmethod
    def from_fn(
        cls,
        name: str,
        requires: Sequence[str],
        fn: Callable[[pd.DataFrame], pd.Series],
        role: FeatureRole = "numeric",
    ) -> Feature:
        """Create a deterministic one-column feature from a callable."""
        return _CallableFeature(name=name, requires=requires, fn=fn, role=role)


class _CallableFeature(Feature):
    def __init__(
        self,
        *,
        name: str,
        requires: Sequence[str],
        fn: Callable[[pd.DataFrame], pd.Series],
        role: FeatureRole = "numeric",
    ) -> None:
        super().__init__(name=name, requires=requires, role=role)
        self.fn = fn

    def _transform(self, frame: pd.DataFrame) -> pd.Series:
        return self.fn(frame).rename(self.name)


@dataclass(frozen=True)
class EDAOnlyFeature:
    name: str
    requires: tuple[str, ...] = ()


class FeaturePipeline:
    """Ordered one-feature pipeline with dependency and output validation.

    Expected columns:
        ``input_columns`` derived from feature requirements must be present
        before transformation. Later features may require earlier outputs.
    Returned columns:
        All input columns followed by generated feature columns in registry
        order.
    Invalid-value behavior:
        Delegates scalar handling to each feature.
    Missing-value behavior:
        Delegates missing-value handling to each feature.
    Row-count behavior:
        Row count and row index are preserved.
    Learns from data:
        No.
    """

    features: tuple[Feature, ...]

    def __init__(self, features: Sequence[Feature]) -> None:
        self.features = tuple(features)
        self._validate_definition()

    @property
    def output_columns(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.features)

    @property
    def input_columns(self) -> tuple[str, ...]:
        produced: set[str] = set()
        required: list[str] = []
        for feature in self.features:
            for column in feature.requires:
                if column not in produced and column not in required:
                    required.append(column)
            produced.add(feature.name)
        return tuple(required)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        original_index = frame.index.copy()
        original_length = len(frame)
        original_dtypes = frame.dtypes.copy()

        for feature in self.features:
            if feature.name in result.columns:
                raise ValueError(
                    f"Feature {feature.name!r} attempted to overwrite an existing column"
                )
            output = feature.transform(result)
            if output.name in result.columns:
                raise ValueError(
                    f"Feature {feature.name!r} attempted to overwrite an existing column"
                )
            result[output.name] = output
            if len(result) != original_length or not result.index.equals(original_index):
                raise ValueError(f"Feature {feature.name!r} changed rows")
            if not result.loc[:, frame.columns].dtypes.equals(original_dtypes):
                raise ValueError(f"Feature {feature.name!r} changed input dtypes")

        if not frame.dtypes.equals(original_dtypes):
            raise ValueError("Feature pipeline mutated input dtypes")
        return result

    def _validate_definition(self) -> None:
        output_names = [feature.name for feature in self.features]
        generated_names = set(output_names)
        produced_by: dict[str, str] = {}
        available: set[str] = set()
        for feature in self.features:
            if not feature.name:
                raise ValueError("Feature names must be non-empty")
            previous = produced_by.get(feature.name)
            if previous is not None:
                raise ValueError(
                    f"Column {feature.name!r} is produced by both "
                    f"{previous!r} and {feature.__class__.__name__!r}"
                )
            for dependency in feature.requires:
                if dependency in generated_names and dependency not in available:
                    raise ValueError(
                        f"Feature {feature.name!r} depends on future feature "
                        f"{dependency!r}"
                    )
            produced_by[feature.name] = feature.__class__.__name__
            available.add(feature.name)


class SklearnFeaturePipeline(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible wrapper for FeaturePipeline."""

    def __init__(self, feature_pipeline: FeaturePipeline) -> None:
        self.feature_pipeline = feature_pipeline

    def fit(self, X: pd.DataFrame, y: Any = None) -> SklearnFeaturePipeline:
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        else:
            self.feature_names_in_ = np.asarray(
                [f"x{i}" for i in range(X.shape[1])],
                dtype=object,
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.feature_pipeline.transform(X)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is None:
            input_features = self.feature_names_in_
        return np.asarray(
            [*input_features, *self.feature_pipeline.output_columns],
            dtype=object,
        )


# ---------------------------------------------------------------------------
# Time-series-aware feature classes
# ---------------------------------------------------------------------------


class LagFeature(Feature):
    """Shift a single source column by ``lag`` rows.

    The frame **must** be sorted by time before this feature is applied.
    Produces NaN in the first ``lag`` rows.

    Parameters
    ----------
    name : str
        Output column name.
    source : str
        Name of the source column to lag.
    lag : int
        Number of rows to shift (must be >= 1).
    role : FeatureRole, default "numeric"
        Role assigned to the output column in the feature registry.

    Raises
    ------
    ValueError
        If ``lag < 1`` at construction time, or if the frame has fewer than
        ``lag + 1`` rows at transform time (all outputs would be NaN, which
        almost certainly indicates a pipeline misconfiguration such as requesting
        a 30-day lag from a 10-row dataset).
    """

    def __init__(
        self,
        *,
        name: str,
        source: str,
        lag: int,
        role: FeatureRole = "numeric",
    ) -> None:
        super().__init__(name=name, requires=[source], role=role)
        if lag < 1:
            raise ValueError(f"LagFeature lag must be >= 1, got {lag}.")
        self.source = source
        self.lag = lag

    def _transform(self, frame: pd.DataFrame) -> pd.Series:
        if len(frame) <= self.lag:
            raise ValueError(
                f"LagFeature '{self.name}': frame has {len(frame)} row(s) but "
                f"lag={self.lag} requires at least {self.lag + 1} rows to produce "
                f"any non-NaN output. This usually indicates a pipeline "
                f"misconfiguration (e.g. too large a lag for the dataset size)."
            )
        return frame[self.source].shift(self.lag).rename(self.name)


class RollingMeanFeature(Feature):
    """Rolling window mean of a single source column.

    The frame **must** be sorted by time before this feature is applied.
    Produces NaN in the first ``window - 1`` rows.

    Parameters
    ----------
    name : str
        Output column name.
    source : str
        Name of the source column.
    window : int
        Rolling window size (must be >= 2).
    role : FeatureRole, default "numeric"

    Raises
    ------
    ValueError
        If ``window < 2`` at construction time, or if the frame has fewer
        rows than ``window`` at transform time.
    """

    def __init__(
        self,
        *,
        name: str,
        source: str,
        window: int,
        role: FeatureRole = "numeric",
    ) -> None:
        super().__init__(name=name, requires=[source], role=role)
        if window < 2:
            raise ValueError(f"RollingMeanFeature window must be >= 2, got {window}.")
        self.source = source
        self.window = window

    def _transform(self, frame: pd.DataFrame) -> pd.Series:
        if len(frame) < self.window:
            raise ValueError(
                f"RollingMeanFeature '{self.name}': frame has {len(frame)} row(s) "
                f"but window={self.window} requires at least {self.window} rows."
            )
        return frame[self.source].rolling(self.window).mean().rename(self.name)


class RollingStdFeature(Feature):
    """Rolling window standard deviation of a single source column.

    The frame **must** be sorted by time before this feature is applied.
    Produces NaN in the first ``window - 1`` rows.

    Parameters
    ----------
    name : str
        Output column name.
    source : str
        Name of the source column.
    window : int
        Rolling window size (must be >= 2).
    role : FeatureRole, default "numeric"

    Raises
    ------
    ValueError
        If ``window < 2`` at construction time, or if the frame has fewer
        rows than ``window`` at transform time.
    """

    def __init__(
        self,
        *,
        name: str,
        source: str,
        window: int,
        role: FeatureRole = "numeric",
    ) -> None:
        super().__init__(name=name, requires=[source], role=role)
        if window < 2:
            raise ValueError(f"RollingStdFeature window must be >= 2, got {window}.")
        self.source = source
        self.window = window

    def _transform(self, frame: pd.DataFrame) -> pd.Series:
        if len(frame) < self.window:
            raise ValueError(
                f"RollingStdFeature '{self.name}': frame has {len(frame)} row(s) "
                f"but window={self.window} requires at least {self.window} rows."
            )
        return frame[self.source].rolling(self.window).std().rename(self.name)


class ExpandingMeanFeature(Feature):
    """Expanding (cumulative) mean of a single source column.

    The frame **must** be sorted by time before this feature is applied.
    The first row will be equal to the source column's own value (expanding
    mean of one element).

    Parameters
    ----------
    name : str
        Output column name.
    source : str
        Name of the source column.
    role : FeatureRole, default "numeric"
    """

    def __init__(
        self,
        *,
        name: str,
        source: str,
        role: FeatureRole = "numeric",
    ) -> None:
        super().__init__(name=name, requires=[source], role=role)
        self.source = source

    def _transform(self, frame: pd.DataFrame) -> pd.Series:
        return frame[self.source].expanding().mean().rename(self.name)
