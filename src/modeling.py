from __future__ import annotations

from typing import Callable, Mapping

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.config import RANDOM_STATE
from src.evaluation import TaskType
from src.features import FeaturePipeline
from src.preprocessing import PreprocessingConfig, build_model_pipeline


def _densify_if_sparse(values: object) -> object:
    if hasattr(values, "toarray"):
        return values.toarray()
    return values


def _hist_gradient_boosting_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("to_dense", FunctionTransformer(_densify_if_sparse, accept_sparse=True)),
            (
                "estimator",
                HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            ),
        ]
    )


def _hist_gradient_boosting_regressor() -> Pipeline:
    return Pipeline(
        steps=[
            ("to_dense", FunctionTransformer(_densify_if_sparse, accept_sparse=True)),
            (
                "estimator",
                HistGradientBoostingRegressor(random_state=RANDOM_STATE),
            ),
        ]
    )


def _optional_classification_estimators() -> dict[str, BaseEstimator]:
    estimators: dict[str, BaseEstimator] = {}
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        pass
    else:
        estimators["lightgbm"] = LGBMClassifier(random_state=RANDOM_STATE)

    try:
        from xgboost import XGBClassifier
    except ImportError:
        pass
    else:
        estimators["xgboost"] = XGBClassifier(random_state=RANDOM_STATE)

    try:
        from catboost import CatBoostClassifier
    except ImportError:
        pass
    else:
        estimators["catboost"] = CatBoostClassifier(
            random_seed=RANDOM_STATE,
            verbose=False,
        )

    return estimators


def _optional_regression_estimators() -> dict[str, BaseEstimator]:
    estimators: dict[str, BaseEstimator] = {}
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        pass
    else:
        estimators["lightgbm"] = LGBMRegressor(random_state=RANDOM_STATE)

    try:
        from xgboost import XGBRegressor
    except ImportError:
        pass
    else:
        estimators["xgboost"] = XGBRegressor(random_state=RANDOM_STATE)

    try:
        from catboost import CatBoostRegressor
    except ImportError:
        pass
    else:
        estimators["catboost"] = CatBoostRegressor(
            random_seed=RANDOM_STATE,
            verbose=False,
        )

    return estimators


# Extensible registry for task baseline estimators (F-5)
ESTIMATOR_REGISTRY: dict[TaskType, Callable[[], dict[str, BaseEstimator]]] = {
    "classification": lambda: {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "hist_gradient_boosting": _hist_gradient_boosting_classifier(),
        **_optional_classification_estimators(),
    },
    "regression": lambda: {
        "dummy": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(random_state=RANDOM_STATE),
        "hist_gradient_boosting": _hist_gradient_boosting_regressor(),
        **_optional_regression_estimators(),
    },
}


def baseline_estimators(task: TaskType) -> dict[str, BaseEstimator]:
    if task not in ESTIMATOR_REGISTRY:
        raise ValueError(
            f"task must be one of {list(ESTIMATOR_REGISTRY.keys())}, got {task!r}."
        )
    return ESTIMATOR_REGISTRY[task]()


def train_baseline_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    task: TaskType,
    config: PreprocessingConfig,
    estimators: Mapping[str, BaseEstimator] | None = None,
    feature_pipeline: FeaturePipeline | None = None,
) -> dict[str, BaseEstimator]:
    models: dict[str, BaseEstimator] = {}
    estimators_to_train = estimators if estimators is not None else baseline_estimators(task)

    for name, estimator in estimators_to_train.items():
        # Clone estimator to prevent mutations on caller-provided instances (F-6)
        cloned_estimator = clone(estimator)
        pipeline = build_model_pipeline(
            cloned_estimator,
            x_train,
            config=config,
            feature_pipeline=feature_pipeline,
        )
        pipeline.fit(x_train, y_train)
        models[name] = pipeline

    return models
