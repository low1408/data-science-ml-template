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
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GroupKFold,
    TimeSeriesSplit,
    cross_validate,
    GridSearchCV,
    RandomizedSearchCV,
)

from src.config import RANDOM_STATE, ValidationConfig, SearchConfig
from src.data import split_features_target, train_test_split_dataframe
from src.artifacts import save_dataframe, save_json, save_model
from src.evaluation import (
    TaskType,
    compare_models,
    variance_inflation_factors,
    mutual_information_scores,
    imputation_reconstruction_error,
)
from src.modeling import baseline_estimators, train_baseline_models
from src.features import FeaturePipeline
from src.preprocessing import PreprocessingConfig, build_model_pipeline
from src.validation import DataSchema, validate_dataframe, ConfigurationError, validate_pipeline_and_search_configs

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
    cv_folds_results: pd.DataFrame | None = None
    permutation_importances: dict[str, pd.DataFrame] | None = None
    imputation_reconstruction_error: pd.DataFrame | None = None
    vif: pd.DataFrame | None = None
    mutual_information: pd.DataFrame | None = None
    group_breakdowns: dict[str, dict[str, pd.DataFrame]] | None = None
    fairness_metrics: dict[str, dict[str, dict[str, float]]] | None = None


def _build_cv_splitter(
    validation: ValidationConfig,
    task: TaskType,
    target: pd.Series,
    random_state: int,
) -> Any:
    method = validation.method
    n_splits = validation.n_splits
    if method == "holdout":
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    elif method == "kfold":
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    elif method == "stratified_kfold":
        if task == "classification":
            class_counts = target.value_counts(dropna=False)
            if not class_counts.empty and int(class_counts.min()) >= n_splits:
                return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            warnings.warn(
                "Stratified cross-validation requested, but at least "
                "one class has fewer samples than n_splits. Falling back to KFold.",
                UserWarning,
                stacklevel=2,
            )
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    elif method == "group_kfold":
        return GroupKFold(n_splits=n_splits)
    elif method == "time_series_split":
        return TimeSeriesSplit(n_splits=n_splits)
    raise ValueError(f"Unknown validation method: {method}")


def train_and_search_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    task: TaskType,
    config: PreprocessingConfig,
    validation: ValidationConfig,
    search_config: SearchConfig,
    random_state: int = RANDOM_STATE,
    estimators: Mapping[str, BaseEstimator] | None = None,
    feature_pipeline: FeaturePipeline | None = None,
    groups: pd.Series | None = None,
) -> tuple[dict[str, BaseEstimator], dict[str, Any], dict[str, pd.DataFrame]]:
    models: dict[str, BaseEstimator] = {}
    search_metadata: dict[str, Any] = {}
    search_cv_results: dict[str, pd.DataFrame] = {}

    estimators_to_evaluate = (
        estimators if estimators is not None else baseline_estimators(task)
    )

    for name, estimator in estimators_to_evaluate.items():
        # First, build model pipeline
        model_pipeline = build_model_pipeline(
            clone(estimator),
            x_train,
            config=config,
            feature_pipeline=feature_pipeline,
        )

        # Check if parameter search parameters are specified for this estimator
        raw_params = search_config.estimators.get(name, {}) if search_config.estimators else {}
        
        if search_config.method != "none" and raw_params:
            # Resolve parameter keys
            param_grid = {}
            valid_pipeline_params = set(model_pipeline.get_params().keys())
            
            for p_key, p_val in raw_params.items():
                if "__" in p_key:
                    resolved_key = p_key
                else:
                    nested_key = f"model__estimator__{p_key}"
                    simple_key = f"model__{p_key}"
                    if nested_key in valid_pipeline_params:
                        resolved_key = nested_key
                    elif simple_key in valid_pipeline_params:
                        resolved_key = simple_key
                    else:
                        raise ConfigurationError(
                            f"Parameter '{p_key}' for estimator '{name}' is invalid. "
                            f"Valid parameter keys for this estimator pipeline are: {sorted(valid_pipeline_params)}"
                        )
                # Validate that the resolved_key actually exists in pipeline parameters
                if resolved_key not in valid_pipeline_params:
                    raise ConfigurationError(
                        f"Parameter '{p_key}' (resolved as '{resolved_key}') for estimator '{name}' is invalid. "
                        f"Valid parameter keys for this estimator pipeline are: {sorted(valid_pipeline_params)}"
                    )
                param_grid[resolved_key] = p_val

            cv = _build_cv_splitter(validation, task, y_train, random_state)

            if search_config.method == "grid":
                search_obj = GridSearchCV(
                    model_pipeline,
                    param_grid=param_grid,
                    scoring=search_config.scoring,
                    refit=search_config.refit,
                    cv=cv,
                    n_jobs=search_config.n_jobs,
                    error_score="raise",
                )
            elif search_config.method == "randomized":
                search_obj = RandomizedSearchCV(
                    model_pipeline,
                    param_distributions=param_grid,
                    n_iter=search_config.n_iter,
                    scoring=search_config.scoring,
                    refit=search_config.refit,
                    cv=cv,
                    n_jobs=search_config.n_jobs,
                    random_state=random_state,
                    error_score="raise",
                )
            else:
                raise ValueError(f"Unknown search method: {search_config.method}")

            logger.info("Running hyperparameter search (%s) for model %s…", search_config.method, name)
            search_obj.fit(x_train, y_train, groups=groups)

            models[name] = search_obj.best_estimator_

            # Convert numpy/custom types to standard JSON-serializable types in best_params
            serializable_best_params = {}
            for pk, pv in search_obj.best_params_.items():
                clean_name = pk
                if pk.startswith("model__estimator__"):
                    clean_name = pk[len("model__estimator__"):]
                elif pk.startswith("model__"):
                    clean_name = pk[len("model__"):]
                serializable_best_params[clean_name] = _to_jsonable(pv)

            search_metadata[name] = {
                "best_params": serializable_best_params,
                "best_score": float(search_obj.best_score_) if search_obj.best_score_ is not None else None,
                "best_params_raw": _to_jsonable(search_obj.best_params_),
            }
            search_cv_results[name] = pd.DataFrame(search_obj.cv_results_)
        else:
            logger.info("Fitting model %s directly (no hyperparameter search)…", name)
            model_pipeline.fit(x_train, y_train)
            models[name] = model_pipeline

    return models, search_metadata, search_cv_results


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
    validation: ValidationConfig | None = None,
    search: SearchConfig | None = None,
    save_dir: str | Path | None = None,
    estimators: Mapping[str, BaseEstimator] | None = None,
    pos_label: Any = None,
    positive_label: Any = None,
    feature_pipeline: FeaturePipeline | None = None,
    run_config: Mapping[str, Any] | None = None,
    permutation_n_repeats: int = 10,
    permutation_scoring: str | None = None,
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

    if validation is None:
        if cv_folds > 0:
            method = "stratified_kfold" if stratify else "kfold"
            n_splits = cv_folds
        else:
            method = "holdout"
            n_splits = 5
        validation = ValidationConfig(
            method=method,
            n_splits=n_splits,
            test_size=test_size,
        )

    if search is None:
        search = SearchConfig(method="none")

    # 1. Enforce validation configs early (for both configuration and API-driven calls)
    validate_pipeline_and_search_configs(
        validation=validation,
        search=search,
        task=task,
        target_column=target_column,
        feature_columns=config.feature_columns,
    )

    # DataFrame presence and null checks for validation columns
    if validation.groups_column is not None:
        if validation.groups_column not in dataframe.columns:
            raise KeyError(f"Validation groups column '{validation.groups_column}' not found in dataframe.")
        if dataframe[validation.groups_column].isna().any():
            raise ConfigurationError(f"Validation groups column '{validation.groups_column}' contains missing values.")

    if validation.time_column is not None:
        if validation.time_column not in dataframe.columns:
            raise KeyError(f"Validation time column '{validation.time_column}' not found in dataframe.")
        if dataframe[validation.time_column].isna().any():
            raise ConfigurationError(f"Validation time column '{validation.time_column}' contains missing values.")
        # Attempt conversion to datetime to ensure correct chronological sorting
        try:
            dataframe = dataframe.copy()  # avoid modifying in-place
            dataframe[validation.time_column] = pd.to_datetime(dataframe[validation.time_column], errors="raise")
        except Exception as e:
            if not pd.api.types.is_numeric_dtype(dataframe[validation.time_column]):
                raise ConfigurationError(f"Validation time column '{validation.time_column}' must be a datetime or numeric type. Error: {e}")

    # Time-series temporal sorting
    if validation.time_column is not None:
        dataframe = dataframe.sort_values(by=validation.time_column).copy()

    artifact_base_path = _make_run_artifact_dir(save_dir) if save_dir is not None else None

    run_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "target_column": target_column,
        "test_size": validation.test_size,
        "stratify": stratify,
        "random_state": random_state,
        "cv_folds": cv_folds,
        "validation": _to_jsonable(validation),
        "search": _to_jsonable(search),
        "input_shape": dataframe.shape,
        "feature_columns": _to_jsonable(config.feature_columns),
        "schema": _to_jsonable(schema) if schema is not None else None,
        "feature_pipeline_outputs": (
            list(feature_pipeline.output_columns) if feature_pipeline is not None else []
        ),
        "estimator_names": list(estimators.keys()) if estimators is not None else list(baseline_estimators(task).keys()),
        "artifact_base_path": str(artifact_base_path) if artifact_base_path is not None else None,
    }

    # 1. Validate ─────────────────────────────────────────────────────
    if schema is not None:
        logger.info("Validating dataframe against schema…")
        validate_dataframe(dataframe, schema)

    # 2. Split ────────────────────────────────────────────────────────
    # Split first to prevent selection bias and run baseline CV only on train_df
    if validation.method == "time_series_split":
        logger.info("Performing temporal train/test split (test_size=%.2f)…", validation.test_size)
        split_idx = int(len(dataframe) * (1.0 - validation.test_size))
        if split_idx <= 0 or split_idx >= len(dataframe):
            train_df = dataframe.iloc[:split_idx]
            test_df = dataframe.iloc[split_idx:]
        else:
            cutoff_time = dataframe[validation.time_column].iloc[split_idx]
            train_mask = dataframe[validation.time_column] < cutoff_time
            train_df = dataframe[train_mask]
            test_df = dataframe[~train_mask]
            if train_df.empty or test_df.empty:
                train_df = dataframe.iloc[:split_idx]
                test_df = dataframe.iloc[split_idx:]
        x_train, y_train = split_features_target(train_df, target_column)
        x_test, y_test = split_features_target(test_df, target_column)
    elif validation.method == "group_kfold":
        logger.info("Performing group-aware train/test split (test_size=%.2f, group_column=%s)…", validation.test_size, validation.groups_column)
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=validation.test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(dataframe, groups=dataframe[validation.groups_column]))
        train_df = dataframe.iloc[train_idx]
        test_df = dataframe.iloc[test_idx]
        x_train, y_train = split_features_target(train_df, target_column)
        x_test, y_test = split_features_target(test_df, target_column)
    else:
        logger.info("Splitting data (test_size=%.2f, stratify=%s)…", validation.test_size, stratify)
        x_train, x_test, y_train, y_test = train_test_split_dataframe(
            dataframe,
            target_column,
            test_size=validation.test_size,
            random_state=random_state,
            stratify=stratify if validation.method in {"stratified_kfold", "holdout"} else False,
        )
        train_df = dataframe.loc[x_train.index]
        test_df = dataframe.loc[x_test.index]

    # 3. Cross-validate (optional) on train_df strictly ────────────────
    cv_summary = None
    cv_folds_df = None
    if validation.method != "holdout":
        logger.info("Cross-validating baseline models (%s) on training set…", validation.method)
        # Check group cardinality before running baseline CV on groups
        if validation.method == "group_kfold":
            train_groups = train_df[validation.groups_column]
            if train_groups.nunique() < validation.n_splits:
                raise ConfigurationError(
                    f"Number of unique groups in training set ({train_groups.nunique()}) "
                    f"is less than validation.n_splits ({validation.n_splits})."
                )
        
        cv_summary, cv_folds_df = cross_validate_baseline_models(
            train_df,
            target_column=target_column,
            task=task,
            config=config,
            validation=validation,
            random_state=random_state,
            estimators=estimators,
            feature_pipeline=feature_pipeline,
        )
        logger.info("Cross-validation results:\n%s", cv_summary)

    # Capture groups column if present
    groups_train = None
    groups_test = None
    if validation.groups_column is not None:
        if validation.groups_column in x_train.columns:
            groups_train = x_train[validation.groups_column]
        if validation.groups_column in x_test.columns:
            groups_test = x_test[validation.groups_column]
        if groups_train is not None:
            if search.method != "none":
                # Check unique groups cardinality for hyperparameter search
                if groups_train.nunique() < validation.n_splits:
                    raise ConfigurationError(
                        f"Number of unique training groups ({groups_train.nunique()}) "
                        f"is less than validation.n_splits ({validation.n_splits}) for hyperparameter search."
                    )

    # Exclude validation columns from training/test splits
    cols_to_drop = [col for col in [validation.groups_column, validation.time_column] if col is not None]
    if cols_to_drop:
        x_train = x_train.drop(columns=[col for col in cols_to_drop if col in x_train.columns])
        x_test = x_test.drop(columns=[col for col in cols_to_drop if col in x_test.columns])

    # 3.5 Calculate Diagnostics before modeling ──────────────────────
    vif_df = None
    numeric_cols_for_vif = list(x_train.select_dtypes(include=[np.number]).columns)
    if len(numeric_cols_for_vif) >= 2:
        try:
            vif_df = variance_inflation_factors(x_train, columns=numeric_cols_for_vif)
        except Exception as e:
            logger.warning("Could not calculate Variance Inflation Factors: %s", e)
    else:
        logger.warning("Fewer than 2 numeric columns in training set; skipping VIF calculation.")

    mi_df = None
    try:
        mi_df = mutual_information_scores(x_train, y_train, task=task, random_state=random_state)
    except Exception as e:
        logger.warning("Could not calculate Mutual Information scores: %s", e)

    imputation_reconstruction_df = None
    if config.feature_columns.numeric:
        try:
            imputation_reconstruction_df = imputation_reconstruction_error(
                train_df,
                config,
                random_state=random_state,
            )
        except Exception as e:
            logger.warning("Could not calculate imputation reconstruction error: %s", e)

    # Find all cohort columns present in the input dataframe
    potential_cohort_cols = []
    if validation.groups_column is not None:
        potential_cohort_cols.append(validation.groups_column)
    if hasattr(config, "stratified_categorical_group_cols") and config.stratified_categorical_group_cols:
        potential_cohort_cols.extend(config.stratified_categorical_group_cols)
    if hasattr(config, "stratified_numeric_group_cols") and config.stratified_numeric_group_cols:
        potential_cohort_cols.extend(config.stratified_numeric_group_cols)
    if hasattr(config, "stratified_fallback_group_col") and config.stratified_fallback_group_col:
        potential_cohort_cols.append(config.stratified_fallback_group_col)

    seen = set()
    cohort_cols = []
    for col in potential_cohort_cols:
        if col in dataframe.columns and col not in seen:
            seen.add(col)
            cohort_cols.append(col)

    groups_test_val = None
    if cohort_cols:
        if len(cohort_cols) == 1:
            groups_test_val = test_df[cohort_cols[0]]
        else:
            groups_test_val = test_df[cohort_cols]

    # 4. Train & Tune ──────────────────────────────────────────────────
    logger.info("Training estimators (task=%s, search_method=%s)…", task, search.method)
    models, search_metadata, search_cv_results = train_and_search_models(
        x_train,
        y_train,
        task=task,
        config=config,
        validation=validation,
        search_config=search,
        random_state=random_state,
        estimators=estimators,
        feature_pipeline=feature_pipeline,
        groups=groups_train,
    )
    if search_metadata:
        run_metadata["search_results"] = search_metadata

    # 5. Evaluate ─────────────────────────────────────────────────────
    logger.info("Evaluating %d models…", len(models))
    permutation_importances = {}
    group_breakdowns = {}
    fairness_metrics_dict = {}

    comparison = compare_models(
        models,
        x_test,
        y_test,
        task=task,
        pos_label=pos_label,
        feature_importances_dir=(
            "metrics/feature_importances" if artifact_base_path is not None else None
        ),
        permutation_importances_dir=(
            "metrics/permutation_importances" if artifact_base_path is not None else None
        ),
        confusion_matrices_dir=(
            "metrics/confusion_matrices"
            if artifact_base_path is not None and task == "classification"
            else None
        ),
        group_metrics_dir=(
            "metrics/group_metrics"
            if artifact_base_path is not None and groups_test_val is not None
            else None
        ),
        group_values=groups_test_val,
        artifacts_base_path=artifact_base_path,
        permutation_scoring=permutation_scoring,
        permutation_n_repeats=permutation_n_repeats,
        random_state=random_state,
        permutation_importances=permutation_importances,
        group_breakdowns=group_breakdowns,
        fairness_metrics_dict=fairness_metrics_dict,
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
        if vif_df is not None:
            artifact_paths["vif"] = save_dataframe(
                vif_df,
                "metrics/vif.csv",
                base_path=artifact_base_path,
            )
        if mi_df is not None:
            artifact_paths["mutual_information"] = save_dataframe(
                mi_df,
                "metrics/mutual_information.csv",
                base_path=artifact_base_path,
            )
        if imputation_reconstruction_df is not None:
            artifact_paths["imputation_reconstruction_error"] = save_dataframe(
                imputation_reconstruction_df,
                "metrics/imputation_reconstruction_error.csv",
                base_path=artifact_base_path,
            )
        for name in models:
            if permutation_importances and name in permutation_importances:
                artifact_paths[f"permutation_importances:{name}"] = (
                    artifact_base_path / "metrics" / "permutation_importances" / f"{name}.csv"
                )
            if group_breakdowns and name in group_breakdowns:
                for cohort in group_breakdowns[name]:
                    if isinstance(groups_test_val, pd.DataFrame):
                        artifact_paths[f"group_breakdown:{name}:{cohort}"] = (
                            artifact_base_path / "metrics" / "group_metrics" / name / cohort / "breakdown.csv"
                        )
                        if task == "classification":
                            artifact_paths[f"fairness_metrics:{name}:{cohort}"] = (
                                artifact_base_path / "metrics" / "group_metrics" / name / cohort / "fairness.json"
                            )
                    else:
                        artifact_paths[f"group_breakdown:{name}:{cohort}"] = (
                            artifact_base_path / "metrics" / "group_metrics" / name / "breakdown.csv"
                        )
                        if task == "classification":
                            artifact_paths[f"fairness_metrics:{name}:{cohort}"] = (
                                artifact_base_path / "metrics" / "group_metrics" / name / "fairness.json"
                            )

        if cv_summary is not None:
            # Backward compatibility key cv_metrics -> metrics/cross_validation.csv
            artifact_paths["cv_metrics"] = save_dataframe(
                cv_summary,
                "metrics/cross_validation.csv",
                base_path=artifact_base_path,
            )
            # New fold-level metrics
            if cv_folds_df is not None:
                artifact_paths["cv_folds_metrics"] = save_dataframe(
                    cv_folds_df,
                    "metrics/cross_validation_folds.csv",
                    base_path=artifact_base_path,
                )
        # Save search results if search was run
        for name, search_df in search_cv_results.items():
            artifact_paths[f"search_cv_results:{name}"] = save_dataframe(
                search_df,
                f"metrics/search/{name}_cv_results.csv",
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
        cv_results=cv_summary,
        cv_folds_results=cv_folds_df,
        permutation_importances=permutation_importances or None,
        imputation_reconstruction_error=imputation_reconstruction_df,
        vif=vif_df,
        mutual_information=mi_df,
        group_breakdowns=group_breakdowns or None,
        fairness_metrics=fairness_metrics_dict or None,
    )


def cross_validate_baseline_models(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    task: TaskType,
    config: PreprocessingConfig,
    validation: ValidationConfig,
    random_state: int = RANDOM_STATE,
    estimators: Mapping[str, BaseEstimator] | None = None,
    feature_pipeline: FeaturePipeline | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Time-series temporal sorting
    if validation.time_column is not None:
        if not dataframe[validation.time_column].is_monotonic_increasing:
            dataframe = dataframe.sort_values(by=validation.time_column).copy()

    features, target = split_features_target(dataframe, target_column)
    
    # Extract groups if applicable
    groups = None
    if validation.groups_column is not None:
        if validation.groups_column in features.columns:
            groups = features[validation.groups_column]

    # Exclude validation columns from features passed to the models
    cols_to_drop = [col for col in [validation.groups_column, validation.time_column] if col is not None]
    if cols_to_drop:
        features = features.drop(columns=[col for col in cols_to_drop if col in features.columns])

    estimators_to_evaluate = (
        estimators if estimators is not None else baseline_estimators(task)
    )
    splitter = _build_cv_splitter(validation, task, target, random_state)
    scoring = _cv_scoring(task)

    rows_summary: list[dict[str, float | str]] = []
    rows_folds: list[dict[str, Any]] = []

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
            groups=groups,
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
        rows_summary.append(row)

        # Record fold-level metrics
        n_splits = len(next(iter(scores.values())))
        for fold_idx in range(n_splits):
            fold_row = {"model": name, "fold": fold_idx + 1}
            for score_name in scoring:
                val = scores[f"test_{score_name}"][fold_idx]
                if score_name in _NEGATIVE_CV_SCORERS:
                    val = -val
                fold_row[score_name] = float(val)
            rows_folds.append(fold_row)

    return pd.DataFrame(rows_summary).set_index("model"), pd.DataFrame(rows_folds)


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
    if hasattr(value, "item") and callable(value.item):
        try:
            if not isinstance(value, (pd.DataFrame, pd.Series)) and (not isinstance(value, np.ndarray) or value.size == 1):
                value = value.item()
        except (ValueError, TypeError, AttributeError):
            pass

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
