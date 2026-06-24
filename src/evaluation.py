from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from math import sqrt
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    matthews_corrcoef,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import unique_labels

from src.artifacts import save_dataframe, save_json

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
        "mcc": _mcc_score(y_true, y_pred),
    }

    if y_score is not None:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UndefinedMetricWarning,
                )
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
            logger.warning("ROC-AUC skipped: %s", exc)
            metrics["roc_auc"] = float("nan")

    return {name: float(value) for name, value in metrics.items()}


def _mcc_score(y_true: Iterable[Any], y_pred: Iterable[Any]) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="A single label was found",
            category=UserWarning,
        )
        return float(matthews_corrcoef(y_true, y_pred))


def regression_metrics(
    y_true: Iterable[float],
    y_pred: Iterable[float],
) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "rmse": float(sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_model(
    model: BaseEstimator,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    task: TaskType,
    pos_label: Any = None,
    positive_label: Any = None,
    feature_importances_path: str | Path | None = None,
    confusion_matrix_dir: str | Path | None = None,
    artifacts_base_path: str | Path | None = None,
) -> dict[str, float]:
    """Evaluate model performance on test dataset.

    Parameters
    ----------
    model : BaseEstimator
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
    feature_importances_path : str or Path or None, default=None
        Optional CSV path for feature importances or coefficients. When provided,
        models exposing ``feature_importances_`` or ``coef_`` are saved there.
    confusion_matrix_dir : str or Path or None, default=None
        Optional directory where classification confusion matrix CSV and JSON
        artifacts are saved.
    artifacts_base_path : str or Path or None, default=None
        Optional base directory used with ``feature_importances_path``.

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
    _save_feature_importances_if_available(
        model,
        x_test,
        feature_importances_path=feature_importances_path,
        artifacts_base_path=artifacts_base_path,
    )

    if task == "classification":
        _save_confusion_matrix_if_requested(
            y_test,
            y_pred,
            confusion_matrix_dir=confusion_matrix_dir,
            artifacts_base_path=artifacts_base_path,
        )
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


def _predict_scores(model: BaseEstimator, x_test: pd.DataFrame, pos_label: Any = None) -> np.ndarray | None:
    """Predict scores/probabilities for the positive class in classification.

    For binary classification, defaults to model.classes_[1] if pos_label is None.

    Parameters
    ----------
    model : BaseEstimator
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
                        return probabilities[:, idx]  # type: ignore[no-any-return]
                    else:
                        raise ValueError(
                            f"pos_label {pos_label!r} not found in model.classes_ {classes!r}"
                        )
                else:
                    return probabilities[:, 1]  # type: ignore[no-any-return]
            return probabilities[:, 1]  # type: ignore[no-any-return]
        # Multi-class: return full probability matrix for OvR AUC
        return probabilities  # type: ignore[no-any-return]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_test)
        if hasattr(model, "classes_") and len(model.classes_) == 2 and pos_label is not None:
            classes = list(model.classes_)
            if pos_label in classes:
                idx = classes.index(pos_label)
                # Negate decision function scores if pos_label is the negative class (index 0)
                if idx == 0:
                    return -scores  # type: ignore[no-any-return]
                return scores  # type: ignore[no-any-return]
            else:
                raise ValueError(
                    f"pos_label {pos_label!r} not found in model.classes_ {classes!r}"
                )
        return scores  # type: ignore[no-any-return]

    return None


def confusion_matrix_dataframe(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
) -> pd.DataFrame:
    """Return a labelled confusion matrix as a dataframe."""
    labels = list(unique_labels(y_true, y_pred))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="A single label was found",
            category=UserWarning,
        )
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
    label_names = [str(label) for label in labels]
    return pd.DataFrame(
        matrix,
        index=pd.Index(label_names, name="actual"),
        columns=pd.Index(label_names, name="predicted"),
    )


def _save_confusion_matrix_if_requested(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    *,
    confusion_matrix_dir: str | Path | None,
    artifacts_base_path: str | Path | None,
) -> None:
    if confusion_matrix_dir is None:
        return

    matrix = confusion_matrix_dataframe(y_true, y_pred)
    csv_path = Path(confusion_matrix_dir) / "confusion_matrix.csv"
    json_path = Path(confusion_matrix_dir) / "confusion_matrix.json"
    payload = _confusion_matrix_payload(matrix)
    if artifacts_base_path is None:
        save_dataframe(matrix, csv_path)
        save_json(payload, json_path)
    else:
        save_dataframe(matrix, csv_path, base_path=artifacts_base_path)
        save_json(payload, json_path, base_path=artifacts_base_path)


def _confusion_matrix_payload(matrix: pd.DataFrame) -> dict[str, Any]:
    return {
        "labels": [str(label) for label in matrix.index.tolist()],
        "matrix": matrix.astype(int).values.tolist(),
    }


def _save_feature_importances_if_available(
    model: BaseEstimator,
    x_test: pd.DataFrame,
    *,
    feature_importances_path: str | Path | None,
    artifacts_base_path: str | Path | None,
) -> None:
    if feature_importances_path is None:
        return

    feature_importances = extract_feature_importances(model, x_test)
    if feature_importances is None:
        logger.info(
            "Feature importances skipped: model exposes neither feature_importances_ nor coef_."
        )
        return

    if artifacts_base_path is None:
        save_dataframe(feature_importances, feature_importances_path)
    else:
        save_dataframe(
            feature_importances,
            feature_importances_path,
            base_path=artifacts_base_path,
        )


def extract_feature_importances(
    model: BaseEstimator,
    x_test: pd.DataFrame,
) -> pd.DataFrame | None:
    """Return model feature importances or coefficients as a dataframe."""
    estimator = _final_estimator(model)
    source = "feature_importances_"
    values = getattr(estimator, "feature_importances_", None)

    if values is None:
        source = "coef_"
        values = getattr(estimator, "coef_", None)

    if values is None:
        return None

    values_array = np.asarray(values)
    importance = _coerce_importance_values(values_array, source)
    feature_names = _feature_names_for_model(model, x_test)
    if len(feature_names) != len(importance):
        logger.info(
            "Feature importance names did not match values; using generic feature names."
        )
        feature_names = [f"feature_{idx}" for idx in range(len(importance))]

    dataframe = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
            "source": source,
        }
    )
    return dataframe.sort_values(
        "importance",
        key=lambda series: series.abs(),
        ascending=False,
    ).reset_index(drop=True)


def _final_estimator(model: BaseEstimator) -> BaseEstimator:
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


def _feature_names_for_model(model: BaseEstimator, x_test: pd.DataFrame) -> list[str]:
    if isinstance(model, Pipeline) and len(model.steps) > 1:
        transformer = model[:-1]
        if hasattr(transformer, "get_feature_names_out"):
            try:
                return [
                    str(name)
                    for name in transformer.get_feature_names_out(x_test.columns)
                ]
            except TypeError:
                return [str(name) for name in transformer.get_feature_names_out()]
            except AttributeError:
                pass
        try:
            transformed = transformer.transform(x_test)
            return [f"feature_{idx}" for idx in range(transformed.shape[1])]
        except (AttributeError, ValueError):
            pass

    if hasattr(model, "feature_names_in_"):
        return [str(name) for name in model.feature_names_in_]

    estimator = _final_estimator(model)
    if hasattr(estimator, "feature_names_in_"):
        return [str(name) for name in estimator.feature_names_in_]

    return [str(name) for name in x_test.columns]


def _coerce_importance_values(
    values: np.ndarray,
    source: str,
) -> np.ndarray:
    if values.ndim == 1:
        importance = values
    elif source == "coef_":
        importance = np.mean(np.abs(values), axis=0)
    else:
        importance = values.reshape(-1)

    return importance.astype(float)


def compare_models(
    models: dict[str, BaseEstimator],
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    task: TaskType,
    pos_label: Any = None,
    positive_label: Any = None,
    feature_importances_dir: str | Path | None = None,
    confusion_matrices_dir: str | Path | None = None,
    artifacts_base_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compare multiple models on test dataset.

    Parameters
    ----------
    models : dict[str, BaseEstimator]
        Dictionary of fitted estimators/pipelines.
    x_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target labels.
    task : TaskType
        Either "classification" or "regression".
    pos_label : Any, default=None
        The class label to treat as the positive class for binary classification.
    positive_label : Any, default=None
        Alias for pos_label. If specified, pos_label must be None.
    feature_importances_dir : str or Path or None, default=None
        Optional directory for one feature-importance CSV per model.
    confusion_matrices_dir : str or Path or None, default=None
        Optional directory for one confusion-matrix CSV/JSON pair per model.
    artifacts_base_path : str or Path or None, default=None
        Optional base directory used with ``feature_importances_dir``.

    Returns
    -------
    pd.DataFrame
        Table comparing metrics for all models.
    """
    if positive_label is not None:
        if pos_label is not None:
            raise ValueError("Cannot specify both pos_label and positive_label.")
        pos_label = positive_label

    results = {}
    for name, model in models.items():
        feature_importances_path = None
        if feature_importances_dir is not None:
            feature_importances_path = Path(feature_importances_dir) / f"{name}.csv"
        confusion_matrix_dir = None
        if confusion_matrices_dir is not None:
            confusion_matrix_dir = Path(confusion_matrices_dir) / name
        results[name] = evaluate_model(
            model,
            x_test,
            y_test,
            task=task,
            pos_label=pos_label,
            feature_importances_path=feature_importances_path,
            confusion_matrix_dir=confusion_matrix_dir,
            artifacts_base_path=artifacts_base_path,
        )
    return model_comparison_table(results)
