from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import tomllib

import pandas as pd

from src.data import CSVDataLoader, ParquetDataLoader, SQLiteQueryLoader, SQLiteTableLoader
from src.evaluation import TaskType
from src.modeling import baseline_estimators
from src.pipeline import PipelineResult, run_pipeline
from src.preprocessing import FeatureColumns, PreprocessingConfig
from src.validation import DataSchema


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
    save_dir: str | None = None
    pos_label: Any = None
    estimator_names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ProjectConfig:
    data: DataSourceConfig
    pipeline: PipelineConfig
    preprocessing: PreprocessingConfig
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

    pipeline = PipelineConfig(
        target_column=str(_required_value(pipeline_raw, "target_column")),
        task=task,  # type: ignore[arg-type]
        test_size=float(pipeline_raw.get("test_size", 0.2)),
        stratify=bool(pipeline_raw.get("stratify", False)),
        random_state=int(pipeline_raw.get("random_state", 128)),
        save_dir=_resolve_optional_path(pipeline_raw.get("save_dir"), base_path),
        pos_label=pipeline_raw.get("pos_label"),
        estimator_names=_optional_tuple(pipeline_raw.get("estimator_names")),
    )

    feature_columns = FeatureColumns(
        numeric=columns_raw.get("numeric", []),
        categorical=columns_raw.get("categorical", []),
        boolean=columns_raw.get("boolean", []),
    )
    preprocessing = PreprocessingConfig(
        feature_columns=feature_columns,
        imputer=preprocessing_raw.get("imputer", "simple"),
        scale_numeric=bool(preprocessing_raw.get("scale_numeric", False)),
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
    )

    schema = _build_schema(schema_raw) if schema_raw is not None else None
    return ProjectConfig(
        data=data,
        pipeline=pipeline,
        preprocessing=preprocessing,
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
        save_dir=config.pipeline.save_dir,
        estimators=estimators,
        pos_label=config.pipeline.pos_label,
        run_config=config.raw,
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
