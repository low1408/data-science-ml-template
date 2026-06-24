Data Science ML Template
========================

A lightweight skeleton for tabular machine learning projects. The intended
workflow is:

1. Point a TOML config at a CSV, Parquet file, or SQLite table/query.
2. Declare the target, task, split settings, column roles, and baseline models.
3. Run the pipeline.
4. Inspect saved models, metrics, metadata, and the exact config snapshot.

Quick Start
-----------

Use the checked-in virtual environment for local commands:

```bash
virtual_env/bin/python -m pytest
virtual_env/bin/python -m src.cli src/configs/online_shopping_example.toml
```

If the package is installed, the console script is also available:

```bash
tabular-ml-run src/configs/online_shopping_example.toml
```

Adapting To A New Dataset
-------------------------

Copy `src/configs/online_shopping_example.toml` and edit these sections:

- `[data]`: choose `csv`, `parquet`, `sqlite_table`, or `sqlite_query`.
- `[pipeline]`: set `target_column`, `task`, split options, `save_dir`, and optional `estimator_names`.
- `[columns]`: list numeric, categorical, and boolean feature columns.
- `[preprocessing]`: tune imputation, scaling, and remainder behavior.
- `[schema]`: add lightweight validation checks before splitting.

For a first draft of column roles from a CSV:

```bash
virtual_env/bin/python examples/infer_columns.py data/my_data.csv --target target
```

Review the generated roles before using them. Inference is intentionally
conservative and cannot know domain semantics.

Config Example
--------------

```toml
[data]
kind = "csv"
path = "../data/my_data.csv"

[pipeline]
target_column = "target"
task = "classification"
test_size = 0.2
stratify = true
random_state = 128
save_dir = "../runs/my_experiment"
estimator_names = ["dummy", "logistic_regression", "random_forest"]

[columns]
numeric = ["age", "income"]
categorical = ["region"]
boolean = ["is_active"]

[preprocessing]
scale_numeric = true
```

Optional numeric and categorical transforms are disabled by default. Enable them
only when they fit the dataset and model family:

```toml
[preprocessing]
numeric_imputer_strategy = "median"
scale_numeric = true
numeric_scaler = "robust"              # "none", "standard", or "robust"

cap_numeric_quantiles = true
quantile_cap_lower = 0.01
quantile_cap_upper = 0.99

numeric_power_transform = "yeo_johnson"  # "none", "yeo_johnson", or "box_cox"
numeric_distribution_transform = "none"  # "none", "quantile_uniform", or "quantile_normal"
numeric_binning = "none"                 # "none", "uniform", "quantile", or "kmeans"
numeric_bin_count = 10

categorical_encoding = "frequency"       # "onehot", "frequency", or "ordinal"
group_rare_categories = true
rare_category_min_frequency = 0.01
frequency_unknown_value = 0.0

add_simple_missing_indicators = true
```

Use `numeric_power_transform = "box_cox"` only for strictly positive numeric
features. The Box-Cox transformer raises a clear error for zero or negative
values instead of silently shifting the data. Power transforms and quantile
distribution transforms are mutually exclusive because both reshape numeric
distributions.

To use cohort-aware stratified imputation instead of the default sklearn
`SimpleImputer` steps, set `imputer = "stratified_hybrid"` and provide the
grouping columns used for the lookups. The imputer fills configured categorical
columns from categorical cohorts, configured numeric columns from numeric
cohorts, then falls back to the fallback group and finally the global
mode/median.

```toml
[preprocessing]
imputer = "stratified_hybrid"
scale_numeric = true
stratified_categorical_group_cols = ["branch", "client_id"]
stratified_numeric_group_cols = ["client_id", "parcel_category"]
stratified_fallback_group_col = "branch"
stratified_min_samples = 1
add_missing_indicators = true
```

Reproducible Run Outputs
------------------------

When `save_dir` is set, each run writes:

- `models/*.joblib`: fitted sklearn pipelines.
- `metrics/model_comparison.csv`: evaluation metrics by model.
- `metadata/run_metadata.json`: task, target, split seed, input shape, schema, and feature outputs.
- `metadata/run_config.json`: the TOML config snapshot used for the run.

Because feature engineering and preprocessing are inside the saved sklearn
pipelines, inference should use the same raw input columns as training.

Custom Feature Engineering
--------------------------

Use `Feature.from_fn()` for simple deterministic one-column features, then pass
a `FeaturePipeline` to `run_pipeline()`. The feature role is registered on the
feature itself, so generated columns are added to preprocessing automatically.

```python
feature_pipeline = FeaturePipeline([
    Feature.from_fn(
        name="exit_bounce_ratio",
        requires=["ExitRate", "BounceRate"],
        fn=lambda df: df["ExitRate"] / (df["BounceRate"] + 1e-5),
        role="numeric",
    )
])
```

When saving fitted pipelines with joblib, pass a module-level function instead
of a lambda so the callable can be pickled.

Subclassing `src.features.Feature` is still supported for complex or stateful
transforms. The feature layer validates dependency order, duplicate names, row
count, index preservation, name matching, and accidental input dtype mutation.

Project Layout
--------------

- `src/data.py`: tabular loaders and train/test splitting.
- `src/preprocessing.py`: column roles, role inference, and sklearn preprocessing.
- `src/features.py`: deterministic feature engineering contracts.
- `src/modeling.py`: baseline estimator registry and training.
- `src/evaluation.py`: classification/regression metrics.
- `src/pipeline.py`: end-to-end orchestration and artifact saving.
- `src/project_config.py`: TOML config parsing and config-driven execution.
- `src/cli.py`: command-line entry point.
