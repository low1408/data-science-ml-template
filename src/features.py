from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

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
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.feature_pipeline.transform(X)
