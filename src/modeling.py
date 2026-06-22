from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.config import MODELS_DIR, RANDOM_STATE
from src.evaluation import evaluate_model, model_comparison_table
from src.preprocessing import FeatureColumns, build_model_pipeline


def baseline_estimators(task: str) -> dict[str, BaseEstimator]:
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
    task: str,
    scale_numeric: bool = False,
    feature_columns: FeatureColumns | None = None,
    numeric_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    boolean_columns: list[str] | None = None,
) -> dict[str, BaseEstimator]:
    models: dict[str, BaseEstimator] = {}

    for name, estimator in baseline_estimators(task).items():
        pipeline = build_model_pipeline(
            estimator,
            x_train,
            scale_numeric=scale_numeric,
            feature_columns=feature_columns,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            boolean_columns=boolean_columns,
        )
        pipeline.fit(x_train, y_train)
        models[name] = pipeline

    return models


def compare_models(
    models: dict[str, BaseEstimator],
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    task: str,
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
    path = _resolve_output_path(file_path, base_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(file_path: str | Path, *, base_path: str | Path = MODELS_DIR) -> Any:
    path = _resolve_output_path(file_path, base_path)
    return joblib.load(path)


def _resolve_output_path(file_path: str | Path, base_path: str | Path) -> Path:
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = Path(base_path).expanduser() / path
    return path.resolve()
