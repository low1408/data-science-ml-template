from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import tomllib

import pandas as pd

from src.config import ValidationConfig, SearchConfig
from src.data import CSVDataLoader, ParquetDataLoader, SQLiteQueryLoader, SQLiteTableLoader
from src.evaluation import TaskType
from src.modeling import baseline_estimators
from src.pipeline import PipelineResult, run_pipeline
from src.preprocessing import FeatureColumns, FeatureSelectionConfig, PreprocessingConfig
from src.validation import DataSchema, ConfigurationError, validate_pipeline_and_search_configs



@dataclass(frozen=True)
class DataSourceConfig:
    kind: str
    path: str
    table_name: str | None = None
    query: str | None = None
    params: list[Any] | dict[str, Any] | None = None
    columns: list[str] | None = None
    predicates: dict[str, Any] | None = None
    limit: int | None = None
    read_options: dict[str, Any] = field(default_factory=dict)



@dataclass(frozen=True)
class PipelineConfig:
    target_column: str
    task: TaskType
    test_size: float = 0.2
    stratify: bool = False
    random_state: int = 128
    cv_folds: int = 0
    save_dir: str | None = None
    pos_label: Any = None
    estimator_names: tuple[str, ...] | None = None
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    permutation_n_repeats: int = 10
    permutation_scoring: str | None = None


@dataclass(frozen=True)
class ProjectConfig:
    data: DataSourceConfig
    pipeline: PipelineConfig
    preprocessing: PreprocessingConfig
    search: SearchConfig
    schema: DataSchema | None = None
    base_path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def load_project_config(path: str | Path) -> ProjectConfig:
    """Load a TOML project config."""
    config_path = Path(path).expanduser().resolve()
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return project_config_from_dict(raw, base_path=config_path.parent)


def project_config_from_dict(
    raw: dict[str, Any],
    *,
    base_path: str | Path | None = None,
) -> ProjectConfig:
    data_raw = _required_table(raw, "data")
    pipeline_raw = _required_table(raw, "pipeline")
    columns_raw = _required_table(raw, "columns")
    preprocessing_raw = raw.get("preprocessing", {})
    feature_selection_raw = dict(preprocessing_raw.get("feature_selection", {}))
    schema_raw = raw.get("schema")

    data = DataSourceConfig(
        kind=str(_required_value(data_raw, "kind")),
        path=str(_required_value(data_raw, "path")),
        table_name=data_raw.get("table_name"),
        query=data_raw.get("query"),
        params=data_raw.get("params"),
        columns=data_raw.get("columns"),
        predicates=data_raw.get("predicates"),
        limit=data_raw.get("limit"),
        read_options=dict(data_raw.get("read_options", {})),
    )
    task = str(_required_value(pipeline_raw, "task"))
    if task not in {"classification", "regression"}:
        raise ValueError("pipeline.task must be 'classification' or 'regression'.")

    # Parse validation settings (with backward compatibility)
    validation_raw = pipeline_raw.get("validation", {})
    legacy_cv_folds = int(pipeline_raw.get("cv_folds", 0))
    legacy_test_size = float(pipeline_raw.get("test_size", 0.2))
    legacy_stratify = bool(pipeline_raw.get("stratify", False))

    if validation_raw:
        method = str(validation_raw.get("method", "holdout")).lower()
        n_splits = int(validation_raw.get("n_splits", 5))
        test_size = float(validation_raw.get("test_size", 0.2))
        groups_column = validation_raw.get("groups_column")
        time_column = validation_raw.get("time_column")
        if groups_column is not None:
            groups_column = str(groups_column)
        if time_column is not None:
            time_column = str(time_column)
    else:
        test_size = legacy_test_size
        groups_column = None
        time_column = None
        if legacy_cv_folds > 0:
            n_splits = legacy_cv_folds
            method = "stratified_kfold" if legacy_stratify else "kfold"
        else:
            n_splits = 5
            method = "holdout"

    validation = ValidationConfig(
        method=method,
        n_splits=n_splits,
        test_size=test_size,
        groups_column=groups_column,
        time_column=time_column,
    )

    pipeline = PipelineConfig(
        target_column=str(_required_value(pipeline_raw, "target_column")),
        task=task,  # type: ignore[arg-type]
        test_size=legacy_test_size,
        stratify=legacy_stratify,
        random_state=int(pipeline_raw.get("random_state", 128)),
        cv_folds=legacy_cv_folds,
        save_dir=_resolve_optional_path(pipeline_raw.get("save_dir"), base_path),
        pos_label=pipeline_raw.get("pos_label"),
        estimator_names=_optional_tuple(pipeline_raw.get("estimator_names")),
        validation=validation,
        permutation_n_repeats=int(pipeline_raw.get("permutation_n_repeats", 10)),
        permutation_scoring=pipeline_raw.get("permutation_scoring"),
    )

    feature_columns = FeatureColumns(
        numeric=columns_raw.get("numeric", []),
        categorical=columns_raw.get("categorical", []),
        boolean=columns_raw.get("boolean", []),
        datetime=columns_raw.get("datetime", []),
        text=columns_raw.get("text", []),
    )

    # Parse search config
    search_raw = raw.get("search", {})
    search_method = str(search_raw.get("method", "none")).lower()
    search_n_iter = int(search_raw.get("n_iter", 10))
    search_n_jobs = int(search_raw.get("n_jobs", -1))
    search_scoring = search_raw.get("scoring")
    search_refit = search_raw.get("refit", True)
    search_estimators = dict(search_raw.get("estimators", {}))

    search = SearchConfig(
        method=search_method,
        n_iter=search_n_iter,
        n_jobs=search_n_jobs,
        scoring=search_scoring,
        refit=search_refit,
        estimators=search_estimators,
    )

    # Run unified configuration checks
    validate_pipeline_and_search_configs(
        validation=validation,
        search=search,
        task=task,
        target_column=pipeline.target_column,
        feature_columns=feature_columns,
    )

    # Validate search estimators match active estimators to prevent silent typos
    active_names = pipeline.estimator_names if pipeline.estimator_names is not None else baseline_estimators(task).keys()
    for est_name in search.estimators:
        if est_name not in active_names:
            raise ConfigurationError(
                f"Search estimator '{est_name}' is not in the active estimator list: {list(active_names)}."
            )

    preprocessing = PreprocessingConfig(
        feature_columns=feature_columns,
        feature_selection=FeatureSelectionConfig(
            enabled=bool(feature_selection_raw.get("enabled", False)),
            mutual_information=bool(
                feature_selection_raw.get(
                    "mutual_information",
                    feature_selection_raw.get("mi", False),
                )
            ),
            vif=bool(feature_selection_raw.get("vif", False)),
            mi_strategy=str(feature_selection_raw.get("mi_strategy", "percentile")),
            mi_k=int(feature_selection_raw.get("mi_k", 20)),
            mi_percentile=float(feature_selection_raw.get("mi_percentile", 50.0)),
            mi_threshold=float(feature_selection_raw.get("mi_threshold", 0.0)),
            mi_min_features=int(feature_selection_raw.get("mi_min_features", 1)),
            mi_random_state=int(feature_selection_raw.get("mi_random_state", 0)),
            vif_threshold=float(feature_selection_raw.get("vif_threshold", 10.0)),
            vif_min_features=int(feature_selection_raw.get("vif_min_features", 1)),
            vif_max_iter=(
                int(feature_selection_raw["vif_max_iter"])
                if feature_selection_raw.get("vif_max_iter") is not None
                else None
            ),
            vif_max_features=int(feature_selection_raw.get("vif_max_features", 200)),
        ),
        imputer=preprocessing_raw.get("imputer", "simple"),
        scale_numeric=bool(preprocessing_raw.get("scale_numeric", False)),
        numeric_scaler=preprocessing_raw.get("numeric_scaler", "none"),
        cap_numeric_quantiles=bool(
            preprocessing_raw.get("cap_numeric_quantiles", False)
        ),
        quantile_cap_lower=float(preprocessing_raw.get("quantile_cap_lower", 0.01)),
        quantile_cap_upper=float(preprocessing_raw.get("quantile_cap_upper", 0.99)),
        numeric_power_transform=preprocessing_raw.get("numeric_power_transform", "none"),
        numeric_distribution_transform=preprocessing_raw.get(
            "numeric_distribution_transform",
            "none",
        ),
        quantile_transform_n_quantiles=int(
            preprocessing_raw.get("quantile_transform_n_quantiles", 1000)
        ),
        numeric_binning=preprocessing_raw.get("numeric_binning", "none"),
        numeric_bin_count=int(preprocessing_raw.get("numeric_bin_count", 10)),
        categorical_encoding=preprocessing_raw.get("categorical_encoding", "onehot"),
        group_rare_categories=bool(
            preprocessing_raw.get("group_rare_categories", False)
        ),
        rare_category_min_frequency=float(
            preprocessing_raw.get("rare_category_min_frequency", 0.01)
        ),
        frequency_unknown_value=float(
            preprocessing_raw.get("frequency_unknown_value", 0.0)
        ),
        add_simple_missing_indicators=bool(
            preprocessing_raw.get("add_simple_missing_indicators", False)
        ),
        numeric_imputer_strategy=preprocessing_raw.get(
            "numeric_imputer_strategy",
            "median",
        ),
        categorical_imputer_strategy=preprocessing_raw.get(
            "categorical_imputer_strategy",
            "most_frequent",
        ),
        boolean_imputer_strategy=preprocessing_raw.get(
            "boolean_imputer_strategy",
            "most_frequent",
        ),
        remainder=preprocessing_raw.get("remainder", "drop"),
        boolean_mapping=preprocessing_raw.get("boolean_mapping"),
        text_max_features=int(preprocessing_raw.get("text_max_features", 1000)),
        stratified_categorical_columns=_optional_tuple(
            preprocessing_raw.get("stratified_categorical_columns")
        ),
        stratified_numeric_columns=_optional_tuple(
            preprocessing_raw.get("stratified_numeric_columns")
        ),
        stratified_categorical_group_cols=tuple(
            preprocessing_raw.get(
                "stratified_categorical_group_cols",
                ("branch", "client_id"),
            )
        ),
        stratified_numeric_group_cols=tuple(
            preprocessing_raw.get(
                "stratified_numeric_group_cols",
                ("client_id", "parcel_category"),
            )
        ),
        stratified_fallback_group_col=preprocessing_raw.get(
            "stratified_fallback_group_col",
            "branch",
        ),
        stratified_min_samples=int(preprocessing_raw.get("stratified_min_samples", 1)),
        add_missing_indicators=bool(
            preprocessing_raw.get("add_missing_indicators", True)
        ),
        categorical_onehot_max_cardinality=int(
            preprocessing_raw.get("categorical_onehot_max_cardinality", 10)
        ),
        hashing_n_features=int(preprocessing_raw.get("hashing_n_features", 128)),
    )

    schema = _build_schema(schema_raw) if schema_raw is not None else None
    return ProjectConfig(
        data=data,
        pipeline=pipeline,
        preprocessing=preprocessing,
        search=search,
        schema=schema,
        base_path=Path(base_path).resolve() if base_path is not None else None,
        raw=raw,
    )


def load_dataframe_from_config(
    config: ProjectConfig,
    *,
    base_path: str | Path | None = None,
) -> pd.DataFrame:
    source = config.data
    kind = source.kind.lower()
    loader_base = base_path if base_path is not None else config.base_path or Path.cwd()

    if kind == "csv":
        return CSVDataLoader(
            source.path,
            base_path=loader_base,
            **source.read_options,
        ).load()
    if kind == "parquet":
        return ParquetDataLoader(
            source.path,
            base_path=loader_base,
            **source.read_options,
        ).load()
    if kind == "sqlite_table":
        if source.table_name is None:
            raise ValueError("data.table_name is required for sqlite_table sources.")
        return SQLiteTableLoader(
            source.path,
            source.table_name,
            base_path=loader_base,
            columns=source.columns,
            predicates=source.predicates,
            limit=source.limit,
        ).load()
    if kind == "sqlite_query":
        if source.query is None:
            raise ValueError("data.query is required for sqlite_query sources.")
        params = source.params
        if isinstance(params, list):
            params = tuple(params)
        return SQLiteQueryLoader(
            source.path,
            source.query,
            params=params,
            base_path=loader_base,
        ).load()

    raise ValueError(
        "data.kind must be one of: csv, parquet, sqlite_table, sqlite_query."
    )


def run_project_config(
    config: ProjectConfig,
    *,
    base_path: str | Path | None = None,
) -> PipelineResult:
    dataframe = load_dataframe_from_config(config, base_path=base_path)
    estimators = None
    if config.pipeline.estimator_names is not None:
        registry = baseline_estimators(config.pipeline.task)
        missing = [
            name for name in config.pipeline.estimator_names if name not in registry
        ]
        if missing:
            raise ValueError(f"Unknown estimator names for task: {missing}")
        estimators = {name: registry[name] for name in config.pipeline.estimator_names}

    return run_pipeline(
        dataframe,
        target_column=config.pipeline.target_column,
        task=config.pipeline.task,
        config=config.preprocessing,
        schema=config.schema,
        test_size=config.pipeline.test_size,
        stratify=config.pipeline.stratify,
        random_state=config.pipeline.random_state,
        cv_folds=config.pipeline.cv_folds,
        validation=config.pipeline.validation,
        search=config.search,
        save_dir=config.pipeline.save_dir,
        estimators=estimators,
        pos_label=config.pipeline.pos_label,
        run_config=config.raw,
        permutation_n_repeats=config.pipeline.permutation_n_repeats,
        permutation_scoring=config.pipeline.permutation_scoring,
    )


def _build_schema(raw: dict[str, Any]) -> DataSchema:
    return DataSchema(
        required_columns=raw.get("required_columns", []),
        dtypes=raw.get("dtypes", {}),
        null_limits=raw.get("null_limits", {}),
        unique_columns=raw.get("unique_columns", []),
        ranges={
            column: tuple(bounds)
            for column, bounds in raw.get("ranges", {}).items()
        },
        categorical_hygiene_columns=raw.get("categorical_hygiene_columns", []),
        allowed_categories=raw.get("allowed_categories", {}),
    )


def _required_table(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing required [{key}] config table.")
    return value


def _required_value(raw: dict[str, Any], key: str) -> Any:
    if key not in raw:
        raise ValueError(f"Missing required config value: {key}")
    return raw[key]


def _optional_tuple(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(str(item) for item in value)


def _resolve_optional_path(
    value: Any,
    base_path: str | Path | None,
) -> str | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute() or base_path is None:
        return str(path)
    return str((Path(base_path) / path).resolve())
