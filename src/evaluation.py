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
    pos_label: Any = None,
) -> dict[str, float]:
    """Calculate classification evaluation metrics.

    Parameters
    ----------
    y_true : Iterable[Any]
        True labels.
    y_pred : Iterable[Any]
        Predicted labels.
    y_score : np.ndarray or Iterable[float] or None, default=None
        Target scores, can either be probability estimates of the positive class or
        confidence values. For binary classification, this should be a 1D array of
        probabilities/scores for the positive class.
    pos_label : Any, default=None
        The class label to treat as the positive class for binary ROC-AUC.
        If None, standard scikit-learn label ordering applies.

    Returns
    -------
    dict[str, float]
        Dictionary of metrics.
    """
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
                # Binary classification: binarize target based on pos_label if specified
                if pos_label is not None:
                    y_true_arr = np.asarray(y_true)
                    y_true_bin = np.where(y_true_arr == pos_label, 1, 0)
                    metrics["roc_auc"] = roc_auc_score(y_true_bin, y_score)
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
    pos_label: Any = None,
    positive_label: Any = None,
) -> dict[str, float]:
    """Evaluate model performance on test dataset.

    Parameters
    ----------
    model : object
        Fitted estimator/pipeline.
    x_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target labels.
    task : TaskType
        Either "classification" or "regression".
    pos_label : Any, default=None
        The class label to treat as the positive class for binary classification.
        If None, the default class ordering (model.classes_[1] if available) is used.
    positive_label : Any, default=None
        Alias for pos_label. If specified, pos_label must be None.

    Returns
    -------
    dict[str, float]
        Dictionary of evaluation metrics.
    """
    if positive_label is not None:
        if pos_label is not None:
            raise ValueError("Cannot specify both pos_label and positive_label.")
        pos_label = positive_label

    y_pred = model.predict(x_test)

    if task == "classification":
        y_score = _predict_scores(model, x_test, pos_label=pos_label)
        # Determine the positive label to use for ROC-AUC alignment
        resolved_pos_label = pos_label
        if resolved_pos_label is None and hasattr(model, "classes_") and len(model.classes_) == 2:
            resolved_pos_label = model.classes_[1]
        return classification_metrics(y_test, y_pred, y_score, pos_label=resolved_pos_label)
    if task == "regression":
        if pos_label is not None:
            warnings.warn("pos_label/positive_label is ignored for regression tasks.", UserWarning)
        return regression_metrics(y_test, y_pred)

    raise ValueError("task must be either 'classification' or 'regression'.")


def model_comparison_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(results, orient="index").rename_axis("model")


def _predict_scores(model: object, x_test: pd.DataFrame, pos_label: Any = None) -> np.ndarray | None:
    """Predict scores/probabilities for the positive class in classification.

    For binary classification, defaults to model.classes_[1] if pos_label is None.

    Parameters
    ----------
    model : object
        Fitted estimator/pipeline.
    x_test : pd.DataFrame
        Test features.
    pos_label : Any, default=None
        The class label to treat as the positive class.

    Returns
    -------
    np.ndarray or None
        Predicted scores/probabilities, or None if not supported.
    """
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)
        if probabilities.shape[1] == 2:
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                if pos_label is not None:
                    if pos_label in classes:
                        idx = classes.index(pos_label)
                        return probabilities[:, idx]
                    else:
                        raise ValueError(
                            f"pos_label {pos_label!r} not found in model.classes_ {classes!r}"
                        )
                else:
                    return probabilities[:, 1]
            return probabilities[:, 1]
        # Multi-class: return full probability matrix for OvR AUC
        return probabilities

    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_test)
        if hasattr(model, "classes_") and len(model.classes_) == 2 and pos_label is not None:
            classes = list(model.classes_)
            if pos_label in classes:
                idx = classes.index(pos_label)
                # Negate decision function scores if pos_label is the negative class (index 0)
                if idx == 0:
                    return -scores
                return scores
            else:
                raise ValueError(
                    f"pos_label {pos_label!r} not found in model.classes_ {classes!r}"
                )
        return scores

    return None
