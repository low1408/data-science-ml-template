"""Pipeline orchestration module (F-10).

Provides a lightweight ``run_pipeline`` function that enforces the correct
execution order:  Load → Validate → Split → Train → Evaluate → (Save).

This is deliberately minimal — callers can compose the same steps manually
if they need more control.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from sklearn.base import BaseEstimator

from src.data import split_features_target, train_test_split_dataframe
from src.artifacts import save_model
from src.evaluation import TaskType, compare_models
from src.modeling import (
    baseline_estimators,
    train_baseline_models,
)
from src.features import FeaturePipeline
from src.preprocessing import FeatureColumns, PreprocessingConfig, build_model_pipeline
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



def run_pipeline(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    task: TaskType,
    config: PreprocessingConfig,
    schema: DataSchema | None = None,
    test_size: float = 0.2,
    stratify: bool = False,
    save_dir: str | Path | None = None,
    estimators: Mapping[str, BaseEstimator] | None = None,
    pos_label: Any = None,
    positive_label: Any = None,
    feature_pipeline: FeaturePipeline | None = None,
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
    save_dir : str | Path | None
        If provided, fitted models are saved to this directory.
    estimators : Mapping[str, BaseEstimator] | None
        Optional estimators dictionary.
    pos_label : Any, default=None
        The class label to treat as the positive class for binary classification.
    positive_label : Any, default=None
        Alias for pos_label. If specified, pos_label must be None.
    feature_pipeline : FeaturePipeline | None
        Optional FeaturePipeline to be executed during model fitting and inference.

    Returns
    -------
    dict
        ``{"models": ..., "comparison": ..., "x_train": ..., "x_test": ...,
        "y_train": ..., "y_test": ...}``
    """
    if positive_label is not None:
        if pos_label is not None:
            raise ValueError("Cannot specify both pos_label and positive_label.")
        pos_label = positive_label

    # 1. Validate ─────────────────────────────────────────────────────
    if schema is not None:
        logger.info("Validating dataframe against schema…")
        validate_dataframe(dataframe, schema)

    # 2. Split ────────────────────────────────────────────────────────
    logger.info("Splitting data (test_size=%.2f, stratify=%s)…", test_size, stratify)
    x_train, x_test, y_train, y_test = train_test_split_dataframe(
        dataframe,
        target_column,
        test_size=test_size,
        stratify=stratify,
    )

    # 3. Train ────────────────────────────────────────────────────────
    logger.info("Training baseline models (task=%s)…", task)
    models = train_baseline_models(
        x_train,
        y_train,
        task=task,
        config=config,
        estimators=estimators,
        feature_pipeline=feature_pipeline,
    )

    # 4. Evaluate ─────────────────────────────────────────────────────
    logger.info("Evaluating %d models…", len(models))
    comparison = compare_models(models, x_test, y_test, task=task, pos_label=pos_label)
    logger.info("Results:\n%s", comparison)

    # 5. Save (optional) ──────────────────────────────────────────────
    if save_dir is not None:
        for name, model in models.items():
            path = save_model(model, f"{name}.joblib", base_path=save_dir)
            logger.info("Saved %s → %s", name, path)

    return PipelineResult(
        models=models,
        comparison=comparison,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )

