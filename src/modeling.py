from __future__ import annotations

from typing import Callable, Mapping

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.config import RANDOM_STATE
from src.evaluation import TaskType
from src.preprocessing import PreprocessingConfig, build_model_pipeline


# Extensible registry for task baseline estimators (F-5)
ESTIMATOR_REGISTRY: dict[TaskType, Callable[[], dict[str, BaseEstimator]]] = {
    "classification": lambda: {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(random_state=RANDOM_STATE),
    },
    "regression": lambda: {
        "dummy": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(random_state=RANDOM_STATE),
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
        )
        pipeline.fit(x_train, y_train)
        models[name] = pipeline

    return models

