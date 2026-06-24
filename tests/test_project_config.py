from __future__ import annotations

import pandas as pd

from src.project_config import (
    load_project_config,
    project_config_from_dict,
    run_project_config,
)


def test_project_config_from_dict_builds_runtime_objects():
    config = project_config_from_dict(
        {
            "data": {"kind": "csv", "path": "data.csv"},
            "pipeline": {
                "target_column": "target",
                "task": "classification",
                "estimator_names": ["dummy"],
            },
            "columns": {
                "numeric": ["age"],
                "categorical": ["region"],
                "boolean": ["active"],
            },
            "preprocessing": {"scale_numeric": True},
            "schema": {
                "required_columns": ["age", "region", "active", "target"],
                "null_limits": {"age": 0.5},
            },
        }
    )

    assert config.data.kind == "csv"
    assert config.pipeline.estimator_names == ("dummy",)
    assert config.preprocessing.scale_numeric is True
    assert config.schema is not None
    assert config.schema.required_columns == ("age", "region", "active", "target")


def test_project_config_loads_stratified_hybrid_imputer_settings():
    config = project_config_from_dict(
        {
            "data": {"kind": "csv", "path": "data.csv"},
            "pipeline": {
                "target_column": "target",
                "task": "classification",
            },
            "columns": {
                "numeric": ["parcel_weight_kg"],
                "categorical": ["payment_method"],
            },
            "preprocessing": {
                "imputer": "stratified_hybrid",
                "stratified_categorical_group_cols": ["branch", "client_id"],
                "stratified_numeric_group_cols": ["client_id", "parcel_category"],
                "stratified_fallback_group_col": "branch",
                "stratified_min_samples": 2,
                "add_missing_indicators": False,
            },
        }
    )

    assert config.preprocessing.imputer == "stratified_hybrid"
    assert config.preprocessing.stratified_min_samples == 2
    assert config.preprocessing.add_missing_indicators is False
    assert config.preprocessing.stratified_numeric_group_cols == (
        "client_id",
        "parcel_category",
    )


def test_run_project_config_loads_csv_and_saves_reproducible_artifacts(tmp_path):
    dataframe = pd.DataFrame(
        {
            "age": [20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
            "region": ["a", "b", "a", "b", "a", "b"],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    dataframe.to_csv(tmp_path / "sample.csv", index=False)
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[data]
kind = "csv"
path = "sample.csv"

[pipeline]
target_column = "target"
task = "classification"
test_size = 0.33
stratify = true
random_state = 7
save_dir = "runs/example"
estimator_names = ["dummy"]

[columns]
numeric = ["age"]
categorical = ["region"]
boolean = []
""",
        encoding="utf-8",
    )

    config = load_project_config(config_path)
    result = run_project_config(config)

    assert "dummy" in result.models
    assert result.run_metadata["random_state"] == 7
    assert result.artifact_paths["metrics"].exists()
    assert result.artifact_paths["metadata"].exists()
    assert result.artifact_paths["config"].exists()
    assert (tmp_path / "runs/example/models/dummy.joblib").exists()
