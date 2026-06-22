from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.config import MODELS_DIR, RANDOM_STATE
from src.data import resolve_path
from src.evaluation import TaskType, evaluate_model, model_comparison_table
from src.preprocessing import FeatureColumns, build_model_pipeline


def baseline_estimators(task: TaskType) -> dict[str, BaseEstimator]:
    if task == "classification":
        return {
            "dummy": DummyClassifier(strategy="most_frequent"),
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(random_state=RANDOM_STATE),
        }
    if task == "regression":
        return {
            "dummy": DummyRegressor(strategy="mean"),
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(random_state=RANDOM_STATE),
        }

    raise ValueError("task must be either 'classification' or 'regression'.")


def train_baseline_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    task: TaskType,
    scale_numeric: bool = False,
    feature_columns: FeatureColumns,
    estimators: Mapping[str, BaseEstimator] | None = None,
) -> dict[str, BaseEstimator]:
    models: dict[str, BaseEstimator] = {}

    estimators_to_train = estimators if estimators is not None else baseline_estimators(task)

    for name, estimator in estimators_to_train.items():
        pipeline = build_model_pipeline(
            estimator,
            x_train,
            scale_numeric=scale_numeric,
            feature_columns=feature_columns,
        )
        pipeline.fit(x_train, y_train)
        models[name] = pipeline

    return models


def compare_models(
    models: dict[str, BaseEstimator],
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    task: TaskType,
) -> pd.DataFrame:
    results = {
        name: evaluate_model(model, x_test, y_test, task=task)
        for name, model in models.items()
    }
    return model_comparison_table(results)


def save_model(
    model: Any,
    file_path: str | Path,
    *,
    base_path: str | Path = MODELS_DIR,
) -> Path:
    path = resolve_path(file_path, base_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(file_path: str | Path, *, base_path: str | Path = MODELS_DIR) -> Any:
    path = resolve_path(file_path, base_path)
    return joblib.load(path)
