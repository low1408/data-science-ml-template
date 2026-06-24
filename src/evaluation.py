from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from math import sqrt
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.inspection import permutation_importance
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
from src.config import RANDOM_STATE
from src.preprocessing import PreprocessingConfig, build_imputer

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
    permutation_importances_path: str | Path | None = None,
    confusion_matrix_dir: str | Path | None = None,
    artifacts_base_path: str | Path | None = None,
    permutation_scoring: str | None = None,
    permutation_n_repeats: int = 10,
    random_state: int = RANDOM_STATE,
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
    _save_permutation_importances_if_requested(
        model,
        x_test,
        y_test,
        task=task,
        permutation_importances_path=permutation_importances_path,
        artifacts_base_path=artifacts_base_path,
        scoring=permutation_scoring,
        n_repeats=permutation_n_repeats,
        random_state=random_state,
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


def _save_permutation_importances_if_requested(
    model: BaseEstimator,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    task: TaskType,
    permutation_importances_path: str | Path | None,
    artifacts_base_path: str | Path | None,
    scoring: str | None,
    n_repeats: int,
    random_state: int,
) -> None:
    if permutation_importances_path is None:
        return

    importances = permutation_feature_importance(
        model,
        x_test,
        y_test,
        task=task,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    if artifacts_base_path is None:
        save_dataframe(importances, permutation_importances_path)
    else:
        save_dataframe(
            importances,
            permutation_importances_path,
            base_path=artifacts_base_path,
        )


def permutation_feature_importance(
    model: BaseEstimator,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    task: TaskType,
    scoring: str | None = None,
    n_repeats: int = 10,
    random_state: int = RANDOM_STATE,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Return model-agnostic permutation feature importance on raw input columns."""
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")
    if scoring is None:
        scoring = (
            "f1_weighted"
            if task == "classification"
            else "neg_root_mean_squared_error"
        )

    result = permutation_importance(
        model,
        x_test,
        y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return pd.DataFrame(
        {
            "feature": [str(column) for column in x_test.columns],
            "importance_mean": result.importances_mean.astype(float),
            "importance_std": result.importances_std.astype(float),
            "scoring": scoring,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)


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
    permutation_importances_dir: str | Path | None = None,
    confusion_matrices_dir: str | Path | None = None,
    group_metrics_dir: str | Path | None = None,
    group_values: pd.DataFrame | pd.Series | Iterable[Any] | None = None,
    artifacts_base_path: str | Path | None = None,
    permutation_scoring: str | None = None,
    permutation_n_repeats: int = 10,
    random_state: int = RANDOM_STATE,
    permutation_importances: dict[str, pd.DataFrame] | None = None,
    group_breakdowns: dict[str, dict[str, pd.DataFrame]] | None = None,
    fairness_metrics_dict: dict[str, dict[str, dict[str, float]]] | None = None,
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
        
        permutation_importances_path = None
        if permutation_importances_dir is not None:
            permutation_importances_path = (
                Path(permutation_importances_dir) / f"{name}.csv"
            )

        if permutation_importances is not None or permutation_importances_path is not None:
            p_imp = permutation_feature_importance(
                model,
                x_test,
                y_test,
                task=task,
                scoring=permutation_scoring,
                n_repeats=permutation_n_repeats,
                random_state=random_state,
            )
            if permutation_importances is not None:
                permutation_importances[name] = p_imp
            if permutation_importances_path is not None:
                if artifacts_base_path is None:
                    save_dataframe(p_imp, permutation_importances_path)
                else:
                    save_dataframe(
                        p_imp,
                        permutation_importances_path,
                        base_path=artifacts_base_path,
                    )
            # prevent evaluate_model from calculating it again
            permutation_importances_path = None

        confusion_matrix_dir = None
        if confusion_matrices_dir is not None:
            confusion_matrix_dir = Path(confusion_matrices_dir) / name

        y_pred = model.predict(x_test)
        if group_values is not None:
            if isinstance(group_values, pd.DataFrame):
                # Multiple cohorts
                for col in group_values.columns:
                    col_groups = group_values[col]
                    breakdown = group_metric_breakdown(
                        y_test,
                        y_pred,
                        col_groups,
                        task=task,
                        pos_label=pos_label,
                    )
                    fairness = (
                        fairness_metrics(y_test, y_pred, col_groups, pos_label=pos_label)
                        if task == "classification"
                        else {}
                    )

                    if group_breakdowns is not None:
                        if name not in group_breakdowns:
                            group_breakdowns[name] = {}
                        group_breakdowns[name][col] = breakdown

                    if fairness_metrics_dict is not None and task == "classification":
                        if name not in fairness_metrics_dict:
                            fairness_metrics_dict[name] = {}
                        fairness_metrics_dict[name][col] = fairness

                    if group_metrics_dir is not None:
                        dest_dir = Path(group_metrics_dir) / name / col
                        if artifacts_base_path is None:
                            save_dataframe(breakdown, dest_dir / "breakdown.csv")
                            if fairness:
                                save_json(fairness, dest_dir / "fairness.json")
                        else:
                            save_dataframe(
                                breakdown,
                                dest_dir / "breakdown.csv",
                                base_path=artifacts_base_path,
                            )
                            if fairness:
                                save_json(
                                    fairness,
                                    dest_dir / "fairness.json",
                                    base_path=artifacts_base_path,
                                )
            else:
                # Single cohort (backward-compatible)
                col_name = getattr(group_values, "name", "group")
                if col_name is None:
                    col_name = "group"
                breakdown = group_metric_breakdown(
                    y_test,
                    y_pred,
                    group_values,
                    task=task,
                    pos_label=pos_label,
                )
                fairness = (
                    fairness_metrics(y_test, y_pred, group_values, pos_label=pos_label)
                    if task == "classification"
                    else {}
                )

                if group_breakdowns is not None:
                    if name not in group_breakdowns:
                        group_breakdowns[name] = {}
                    group_breakdowns[name][str(col_name)] = breakdown

                if fairness_metrics_dict is not None and task == "classification":
                    if name not in fairness_metrics_dict:
                        fairness_metrics_dict[name] = {}
                    fairness_metrics_dict[name][str(col_name)] = fairness

                if group_metrics_dir is not None:
                    dest_dir = Path(group_metrics_dir) / name
                    if artifacts_base_path is None:
                        save_dataframe(breakdown, dest_dir / "breakdown.csv")
                        if fairness:
                            save_json(fairness, dest_dir / "fairness.json")
                    else:
                        save_dataframe(
                            breakdown,
                            dest_dir / "breakdown.csv",
                            base_path=artifacts_base_path,
                        )
                        if fairness:
                            save_json(
                                fairness,
                                dest_dir / "fairness.json",
                                base_path=artifacts_base_path,
                            )

        results[name] = evaluate_model(
            model,
            x_test,
            y_test,
            task=task,
            pos_label=pos_label,
            feature_importances_path=feature_importances_path,
            permutation_importances_path=permutation_importances_path,
            confusion_matrix_dir=confusion_matrix_dir,
            artifacts_base_path=artifacts_base_path,
            permutation_scoring=permutation_scoring,
            permutation_n_repeats=permutation_n_repeats,
            random_state=random_state,
        )
    return model_comparison_table(results)


def group_metric_breakdown(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    groups: Iterable[Any],
    *,
    task: TaskType,
    pos_label: Any = None,
) -> pd.DataFrame:
    """Calculate per-cohort model metrics."""
    frame = pd.DataFrame(
        {"y_true": list(y_true), "y_pred": list(y_pred), "group": list(groups)}
    )
    rows: list[dict[str, Any]] = []
    for group, group_frame in frame.groupby("group", dropna=False):
        row: dict[str, Any] = {"group": group, "n": int(len(group_frame))}
        if task == "classification":
            row.update(classification_metrics(group_frame["y_true"], group_frame["y_pred"]))
            if pos_label is None:
                labels = list(unique_labels(frame["y_true"], frame["y_pred"]))
                resolved_pos_label = labels[-1] if labels else 1
            else:
                resolved_pos_label = pos_label
            row["positive_prediction_rate"] = float(
                (group_frame["y_pred"] == resolved_pos_label).mean()
            )
            row["true_positive_rate"] = _group_rate(
                group_frame,
                resolved_pos_label,
                predicted=True,
                actual=True,
            )
            row["false_positive_rate"] = _group_rate(
                group_frame,
                resolved_pos_label,
                predicted=True,
                actual=False,
            )
        elif task == "regression":
            row.update(regression_metrics(group_frame["y_true"], group_frame["y_pred"]))
            row["prediction_mean"] = float(
                pd.to_numeric(group_frame["y_pred"], errors="coerce").mean()
            )
            row["target_mean"] = float(
                pd.to_numeric(group_frame["y_true"], errors="coerce").mean()
            )
        else:
            raise ValueError("task must be either 'classification' or 'regression'.")
        rows.append(row)
    result = pd.DataFrame(rows)
    if "group" in result.columns:
        result = result.sort_values("group", key=lambda series: series.astype(str))
    return result.reset_index(drop=True)


def fairness_metrics(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    groups: Iterable[Any],
    *,
    pos_label: Any = None,
) -> dict[str, float]:
    """Calculate binary classification cohort fairness diagnostics."""
    breakdown = group_metric_breakdown(
        y_true,
        y_pred,
        groups,
        task="classification",
        pos_label=pos_label,
    )
    positive_rates = breakdown["positive_prediction_rate"].dropna()
    tpr = breakdown["true_positive_rate"].dropna()
    fpr = breakdown["false_positive_rate"].dropna()
    if positive_rates.empty:
        return {
            "demographic_parity_difference": float("nan"),
            "disparate_impact_ratio": float("nan"),
            "equalized_odds_difference": float("nan"),
        }
    max_rate = float(positive_rates.max())
    min_rate = float(positive_rates.min())
    equalized_odds_diffs = [
        float(tpr.max() - tpr.min()) if not tpr.empty else float("nan"),
        float(fpr.max() - fpr.min()) if not fpr.empty else float("nan"),
    ]
    finite_equalized_odds_diffs = [
        value for value in equalized_odds_diffs if not np.isnan(value)
    ]
    return {
        "demographic_parity_difference": max_rate - min_rate,
        "disparate_impact_ratio": (
            float(min_rate / max_rate) if max_rate > 0 else float("nan")
        ),
        "equalized_odds_difference": max(finite_equalized_odds_diffs)
        if finite_equalized_odds_diffs
        else float("nan"),
    }


def _save_group_diagnostics_if_requested(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    groups: Iterable[Any],
    *,
    task: TaskType,
    pos_label: Any,
    group_metrics_dir: Path,
    artifacts_base_path: str | Path | None,
) -> None:
    breakdown = group_metric_breakdown(
        y_true,
        y_pred,
        groups,
        task=task,
        pos_label=pos_label,
    )
    fairness = (
        fairness_metrics(y_true, y_pred, groups, pos_label=pos_label)
        if task == "classification"
        else {}
    )
    if artifacts_base_path is None:
        save_dataframe(breakdown, group_metrics_dir / "breakdown.csv")
        if fairness:
            save_json(fairness, group_metrics_dir / "fairness.json")
    else:
        save_dataframe(
            breakdown,
            group_metrics_dir / "breakdown.csv",
            base_path=artifacts_base_path,
        )
        if fairness:
            save_json(
                fairness,
                group_metrics_dir / "fairness.json",
                base_path=artifacts_base_path,
            )


def _group_rate(
    frame: pd.DataFrame,
    pos_label: Any,
    *,
    predicted: bool,
    actual: bool,
) -> float:
    actual_mask = (
        frame["y_true"] == pos_label if actual else frame["y_true"] != pos_label
    )
    denominator = int(actual_mask.sum())
    if denominator == 0:
        return float("nan")
    predicted_mask = (
        frame["y_pred"] == pos_label if predicted else frame["y_pred"] != pos_label
    )
    return float((predicted_mask & actual_mask).sum() / denominator)


def variance_inflation_factors(
    dataframe: pd.DataFrame,
    *,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Calculate VIF for numeric columns without requiring statsmodels."""
    selected = (
        list(columns)
        if columns is not None
        else list(dataframe.select_dtypes(include=[np.number]).columns)
    )
    numeric = dataframe[selected].apply(pd.to_numeric, errors="coerce").dropna(axis=0)
    if len(selected) < 2:
        raise ValueError("At least two numeric columns are required to calculate VIF.")
    if numeric.empty:
        raise ValueError("No complete numeric rows are available to calculate VIF.")
    rows: list[dict[str, Any]] = []
    values = numeric.to_numpy(dtype=float)
    for idx, column in enumerate(selected):
        y = values[:, idx]
        x = np.delete(values, idx, axis=1)
        x = np.column_stack([np.ones(len(x)), x])
        try:
            coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
            y_hat = x @ coefficients
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
            vif = float("inf") if r2 >= 1.0 else float(1.0 / (1.0 - r2))
        except np.linalg.LinAlgError:
            r2 = float("nan")
            vif = float("inf")
        rows.append({"feature": column, "vif": vif, "r2_with_other_features": r2})
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def mutual_information_scores(
    dataframe: pd.DataFrame,
    target: pd.Series,
    *,
    task: TaskType,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Calculate univariate mutual information scores after simple encoding."""
    encoded = pd.get_dummies(dataframe, dummy_na=True)
    encoded = encoded.replace([np.inf, -np.inf], np.nan)
    encoded = encoded.fillna(encoded.median(numeric_only=True)).fillna(0)
    discrete_features = [
        _encoded_column_is_discrete(column, dataframe) for column in encoded.columns
    ]
    if task == "classification":
        scores = mutual_info_classif(
            encoded,
            target,
            discrete_features=discrete_features,
            random_state=random_state,
        )
    elif task == "regression":
        scores = mutual_info_regression(
            encoded,
            target,
            discrete_features=discrete_features,
            random_state=random_state,
        )
    else:
        raise ValueError("task must be either 'classification' or 'regression'.")
    return pd.DataFrame(
        {
            "feature": encoded.columns.astype(str),
            "mutual_information": scores.astype(float),
        }
    ).sort_values("mutual_information", ascending=False).reset_index(drop=True)


def _encoded_column_is_discrete(encoded_column: str, dataframe: pd.DataFrame) -> bool:
    if encoded_column in dataframe.columns:
        return not pd.api.types.is_numeric_dtype(dataframe[encoded_column])
    for original_column in dataframe.columns:
        if encoded_column.startswith(f"{original_column}_"):
            return True
    return False


def imputation_reconstruction_error(
    dataframe: pd.DataFrame,
    config: PreprocessingConfig,
    *,
    mask_fraction: float = 0.1,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Artificially mask known numeric values and report imputation MSE/R2."""
    if not 0.0 < mask_fraction < 1.0:
        raise ValueError("mask_fraction must satisfy 0.0 < mask_fraction < 1.0.")
    numeric_columns = [
        col for col in config.feature_columns.numeric if col in dataframe.columns
    ]
    if not numeric_columns:
        raise ValueError("No configured numeric feature columns are present in dataframe.")

    rng = np.random.default_rng(random_state)
    masked = dataframe.copy()
    masks: dict[str, pd.Series] = {}
    for column in numeric_columns:
        known = masked[column].notna()
        selected = known & (rng.random(len(masked)) < mask_fraction)
        masks[column] = selected
        masked.loc[selected, column] = np.nan

    if config.imputer == "stratified_hybrid":
        imputer = build_imputer(config)
        if imputer is None:
            raise ValueError("Unable to build stratified imputer from config.")
        imputed = clone(imputer).fit_transform(masked)
    else:
        imputed = masked.copy()
        imputer = _numeric_imputer_for_config(config)
        imputed_values = imputer.fit_transform(masked[numeric_columns])
        imputed.loc[:, numeric_columns] = imputed_values

    rows: list[dict[str, Any]] = []
    all_actual: list[float] = []
    all_predicted: list[float] = []
    for column in numeric_columns:
        mask = masks[column]
        if not mask.any():
            continue
        actual = pd.to_numeric(dataframe.loc[mask, column], errors="coerce")
        predicted = pd.to_numeric(imputed.loc[mask, column], errors="coerce")
        valid = actual.notna() & predicted.notna()
        if not valid.any():
            continue
        actual_values = actual[valid].to_numpy(dtype=float)
        predicted_values = predicted[valid].to_numpy(dtype=float)
        mse = float(mean_squared_error(actual_values, predicted_values))
        rows.append(
            {
                "feature": column,
                "masked_count": int(valid.sum()),
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "r2": (
                    float(r2_score(actual_values, predicted_values))
                    if len(actual_values) > 1
                    else float("nan")
                ),
            }
        )
        all_actual.extend(actual_values.tolist())
        all_predicted.extend(predicted_values.tolist())
    if all_actual:
        overall_mse = float(mean_squared_error(all_actual, all_predicted))
        rows.append(
            {
                "feature": "__overall__",
                "masked_count": len(all_actual),
                "mse": overall_mse,
                "rmse": float(np.sqrt(overall_mse)),
                "r2": (
                    float(r2_score(all_actual, all_predicted))
                    if len(all_actual) > 1
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _numeric_imputer_for_config(config: PreprocessingConfig) -> BaseEstimator:
    if config.imputer == "knn" or config.numeric_imputer_strategy == "knn":
        return KNNImputer()
    if config.imputer == "iterative" or config.numeric_imputer_strategy in {
        "iterative",
        "mice",
    }:
        return IterativeImputer(random_state=RANDOM_STATE)
    return SimpleImputer(strategy=config.numeric_imputer_strategy)
