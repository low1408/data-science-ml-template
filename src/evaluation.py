from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from math import sqrt
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

TaskType = Literal["classification", "regression"]


def classification_metrics(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    y_score: np.ndarray | Iterable[float] | None = None,
) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    if y_score is not None:
        try:
            y_score_array = np.asarray(y_score)
            if y_score_array.ndim == 2 and y_score_array.shape[1] > 2:
                # Multi-class: use one-vs-rest AUC
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_score_array, multi_class="ovr", average="weighted"
                )
            else:
                metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        except ValueError as exc:
            warnings.warn(
                f"ROC-AUC could not be computed and was omitted: {exc}",
                stacklevel=2,
            )
            logger.warning("ROC-AUC skipped: %s", exc)

    return {name: float(value) for name, value in metrics.items()}


def regression_metrics(
    y_true: Iterable[float],
    y_pred: Iterable[float],
) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_model(
    model: object,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    task: TaskType,
) -> dict[str, float]:
    y_pred = model.predict(x_test)

    if task == "classification":
        y_score = _predict_scores(model, x_test)
        return classification_metrics(y_test, y_pred, y_score)
    if task == "regression":
        return regression_metrics(y_test, y_pred)

    raise ValueError("task must be either 'classification' or 'regression'.")


def model_comparison_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(results, orient="index").rename_axis("model")


def _predict_scores(model: object, x_test: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)
        if probabilities.shape[1] == 2:
            return probabilities[:, 1]
        # Multi-class: return full probability matrix for OvR AUC
        return probabilities

    if hasattr(model, "decision_function"):
        return model.decision_function(x_test)

    return None
