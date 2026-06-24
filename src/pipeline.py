"""Pipeline orchestration module (F-10).

Provides a lightweight ``run_pipeline`` function that enforces the correct
execution order:  Load → Validate → Split → Train → Evaluate → (Save).

This is deliberately minimal — callers can compose the same steps manually
if they need more control.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Mapping
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    matthews_corrcoef,
    precision_score,
    recall_score,
    make_scorer,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate

from src.config import RANDOM_STATE
from src.data import split_features_target, train_test_split_dataframe
from src.artifacts import save_dataframe, save_json, save_model
from src.evaluation import TaskType, compare_models
from src.modeling import baseline_estimators, train_baseline_models
from src.features import FeaturePipeline
from src.preprocessing import PreprocessingConfig, build_model_pipeline
from src.validation import DataSchema, validate_dataframe

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    """Dataclass holding the results of a pipeline execution (F-9)."""

    models: dict[str, BaseEstimator]
    comparison: pd.DataFrame
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    artifact_paths: dict[str, Path] = field(default_factory=dict)
    run_metadata: dict[str, Any] = field(default_factory=dict)
    cv_results: pd.DataFrame | None = None



def run_pipeline(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    task: TaskType,
    config: PreprocessingConfig,
    schema: DataSchema | None = None,
    test_size: float = 0.2,
    stratify: bool = False,
    random_state: int = RANDOM_STATE,
    cv_folds: int = 0,
    save_dir: str | Path | None = None,
    estimators: Mapping[str, BaseEstimator] | None = None,
    pos_label: Any = None,
    positive_label: Any = None,
    feature_pipeline: FeaturePipeline | None = None,
    run_config: Mapping[str, Any] | None = None,
) -> PipelineResult:

    """Execute a full train-evaluate pipeline in the correct order.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Raw input data containing both features and target.
    target_column : str
        Name of the column to predict.
    task : TaskType
        ``"classification"`` or ``"regression"``.
    config : PreprocessingConfig
        Configuration for preprocessing.
    schema : DataSchema | None
        Optional validation schema.  When provided, the dataframe is
        validated **before** splitting.
    test_size : float
        Fraction of data held out for evaluation (default 0.2).
    stratify : bool
        Whether to stratify the train/test split by the target (default False).
    random_state : int
        Random seed used for the train/test split.
    cv_folds : int
        Number of cross-validation folds to run before the holdout split.
        ``0`` disables cross-validation.
    save_dir : str | Path | None
        If provided, fitted models, metrics, and run metadata are saved to this
        directory under an isolated ``runs/run_<timestamp>`` subdirectory.
    estimators : Mapping[str, BaseEstimator] | None
        Optional estimators dictionary.
    pos_label : Any, default=None
        The class label to treat as the positive class for binary classification.
    positive_label : Any, default=None
        Alias for pos_label. If specified, pos_label must be None.
    feature_pipeline : FeaturePipeline | None
        Optional FeaturePipeline to be executed during model fitting and inference.
    run_config : Mapping[str, Any] | None
        Optional serializable configuration snapshot to persist with artifacts.

    Returns
    -------
    PipelineResult
        Fitted models, holdout metrics, train/test split data, optional
        cross-validation metrics, artifact paths, and run metadata.
    """
    if positive_label is not None:
        if pos_label is not None:
            raise ValueError("Cannot specify both pos_label and positive_label.")
        pos_label = positive_label
    if cv_folds < 0:
        raise ValueError("cv_folds must be greater than or equal to 0.")
    if cv_folds == 1:
        raise ValueError("cv_folds must be 0 or at least 2.")

    artifact_base_path = _make_run_artifact_dir(save_dir) if save_dir is not None else None

    run_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "target_column": target_column,
        "test_size": test_size,
        "stratify": stratify,
        "random_state": random_state,
        "cv_folds": cv_folds,
        "input_shape": dataframe.shape,
        "feature_columns": _to_jsonable(config.feature_columns),
        "schema": _to_jsonable(schema) if schema is not None else None,
        "feature_pipeline_outputs": (
            list(feature_pipeline.output_columns) if feature_pipeline is not None else []
        ),
        "estimator_names": list(estimators.keys()) if estimators is not None else None,
        "artifact_base_path": str(artifact_base_path) if artifact_base_path is not None else None,
    }

    # 1. Validate ─────────────────────────────────────────────────────
    if schema is not None:
        logger.info("Validating dataframe against schema…")
        validate_dataframe(dataframe, schema)

    # 2. Cross-validate (optional) ───────────────────────────────────
    cv_results = None
    if cv_folds:
        logger.info("Cross-validating baseline models (%d folds)…", cv_folds)
        cv_results = cross_validate_baseline_models(
            dataframe,
            target_column=target_column,
            task=task,
            config=config,
            cv_folds=cv_folds,
            random_state=random_state,
            estimators=estimators,
            feature_pipeline=feature_pipeline,
        )
        logger.info("Cross-validation results:\n%s", cv_results)

    # 3. Split ────────────────────────────────────────────────────────
    logger.info("Splitting data (test_size=%.2f, stratify=%s)…", test_size, stratify)
    x_train, x_test, y_train, y_test = train_test_split_dataframe(
        dataframe,
        target_column,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # 4. Train ────────────────────────────────────────────────────────
    logger.info("Training baseline models (task=%s)…", task)
    models = train_baseline_models(
        x_train,
        y_train,
        task=task,
        config=config,
        estimators=estimators,
        feature_pipeline=feature_pipeline,
    )

    # 5. Evaluate ─────────────────────────────────────────────────────
    logger.info("Evaluating %d models…", len(models))
    comparison = compare_models(
        models,
        x_test,
        y_test,
        task=task,
        pos_label=pos_label,
        feature_importances_dir=(
            "metrics/feature_importances" if artifact_base_path is not None else None
        ),
        confusion_matrices_dir=(
            "metrics/confusion_matrices"
            if artifact_base_path is not None and task == "classification"
            else None
        ),
        artifacts_base_path=artifact_base_path,
    )
    logger.info("Results:\n%s", comparison)

    # 6. Save (optional) ──────────────────────────────────────────────
    artifact_paths: dict[str, Path] = {}
    if artifact_base_path is not None:
        for name, model in models.items():
            path = save_model(model, f"models/{name}.joblib", base_path=artifact_base_path)
            artifact_paths[f"model:{name}"] = path
            logger.info("Saved %s → %s", name, path)
        artifact_paths["metrics"] = save_dataframe(
            comparison,
            "metrics/model_comparison.csv",
            base_path=artifact_base_path,
        )
        if cv_results is not None:
            artifact_paths["cv_metrics"] = save_dataframe(
                cv_results,
                "metrics/cross_validation.csv",
                base_path=artifact_base_path,
            )
        artifact_paths["metadata"] = save_json(
            run_metadata,
            "metadata/run_metadata.json",
            base_path=artifact_base_path,
        )
        if run_config is not None:
            artifact_paths["config"] = save_json(
                dict(run_config),
                "metadata/run_config.json",
                base_path=artifact_base_path,
            )

    return PipelineResult(
        models=models,
        comparison=comparison,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        artifact_paths=artifact_paths,
        run_metadata=run_metadata,
        cv_results=cv_results,
    )


def cross_validate_baseline_models(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    task: TaskType,
    config: PreprocessingConfig,
    cv_folds: int,
    random_state: int = RANDOM_STATE,
    estimators: Mapping[str, BaseEstimator] | None = None,
    feature_pipeline: FeaturePipeline | None = None,
) -> pd.DataFrame:
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")

    features, target = split_features_target(dataframe, target_column)
    estimators_to_evaluate = (
        estimators if estimators is not None else baseline_estimators(task)
    )
    splitter = _cv_splitter(task, target, cv_folds, random_state)
    scoring = _cv_scoring(task)

    rows: list[dict[str, float | str]] = []
    for name, estimator in estimators_to_evaluate.items():
        model = build_model_pipeline(
            clone(estimator),
            features,
            config=config,
            feature_pipeline=feature_pipeline,
        )
        scores = cross_validate(
            model,
            features,
            target,
            cv=splitter,
            scoring=scoring,
            error_score="raise",
        )
        row: dict[str, float | str] = {"model": name}
        for score_name in scoring:
            values = np.asarray(scores[f"test_{score_name}"], dtype=float)
            if score_name in _NEGATIVE_CV_SCORERS:
                values = -values
            row[f"{score_name}_mean"] = float(np.mean(values))
            row[f"{score_name}_std"] = float(np.std(values, ddof=0))
        rows.append(row)

    return pd.DataFrame(rows).set_index("model")


_NEGATIVE_CV_SCORERS = frozenset({"mae", "mape", "rmse"})


def _cv_scoring(task: TaskType) -> dict[str, Any]:
    if task == "classification":
        return {
            "accuracy": "accuracy",
            "precision": make_scorer(
                precision_score,
                average="weighted",
                zero_division=0,
            ),
            "recall": make_scorer(
                recall_score,
                average="weighted",
                zero_division=0,
            ),
            "f1": make_scorer(f1_score, average="weighted", zero_division=0),
            "mcc": make_scorer(matthews_corrcoef),
        }
    if task == "regression":
        return {
            "mae": make_scorer(mean_absolute_error, greater_is_better=False),
            "mape": make_scorer(
                mean_absolute_percentage_error,
                greater_is_better=False,
            ),
            "rmse": make_scorer(
                _root_mean_squared_error,
                greater_is_better=False,
            ),
            "r2": "r2",
        }
    raise ValueError("task must be either 'classification' or 'regression'.")


def _root_mean_squared_error(y_true: Any, y_pred: Any) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _cv_splitter(
    task: TaskType,
    target: pd.Series,
    cv_folds: int,
    random_state: int,
) -> KFold | StratifiedKFold:
    if task == "classification":
        class_counts = target.value_counts(dropna=False)
        if not class_counts.empty and int(class_counts.min()) >= cv_folds:
            return StratifiedKFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=random_state,
            )
        warnings.warn(
            "Stratified cross-validation requested by task type, but at least "
            "one class has fewer samples than cv_folds. Falling back to KFold.",
            UserWarning,
            stacklevel=2,
        )
    return KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)


def _make_run_artifact_dir(save_dir: str | Path) -> Path:
    root = Path(save_dir).expanduser()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = root / "runs" / f"run_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = root / "runs" / f"run_{timestamp}_{suffix}"
        suffix += 1
    return run_dir


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_to_jsonable(item) for item in value]
    return str(value)
