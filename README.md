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
- `[columns]`: list numeric, categorical, boolean, datetime, and text feature columns.
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
datetime = ["signup_date"]
text = ["support_notes"]

[preprocessing]
scale_numeric = true
text_max_features = 1000
```

Configuration Options
---------------------

### Data Sources

The `[data]` table selects the input driver with `kind`:

- `csv`: loads an in-memory table with `pandas.read_csv`. Put pandas CSV keyword
  arguments under `[data.read_options]`, for example `sep`, `decimal`, `encoding`,
  or `parse_dates`.
- `parquet`: loads a column-oriented Parquet file with `pandas.read_parquet`.
  Put pandas/pyarrow read keyword arguments under `[data.read_options]`.
- `sqlite_table`: loads rows from a SQLite table. Set `table_name`; optionally
  set `columns`, structured equality/`IN`/`IS NULL` `predicates`, and `limit`.
- `sqlite_query`: executes a raw SQLite query. Use `query` for the SELECT
  statement and `params` for parameter values.

```toml
[data]
kind = "csv"
path = "../data/training.csv"

[data.read_options]
sep = ","
decimal = "."
```

```toml
[data]
kind = "sqlite_table"
path = "../data/app.db"
table_name = "events"
columns = ["target", "age", "region", "event_time"]
predicates = { split = "train", region = ["north", "south"] }
limit = 10000
```

```toml
[data]
kind = "sqlite_query"
path = "../data/app.db"
query = "SELECT target, age, region FROM events WHERE split = ?"
params = ["train"]
```

### Column Roles

The `[columns]` table controls which preprocessing branch receives each feature:

- `numeric`: continuous or count-like numeric columns.
- `categorical`: string/category columns to encode.
- `boolean`: boolean-like columns mapped to 0/1.
- `datetime`: timestamp columns expanded into calendar components.
- `text`: free-text columns vectorized with TF-IDF.

### Numerical Preprocessing

Numeric options live in `[preprocessing]`:

- `numeric_imputer_strategy`: `median`, `mean`, `most_frequent`, `constant`, `knn`, `iterative`, or `mice`.
- `add_simple_missing_indicators`: when `true`, adds sklearn missingness
  indicators for configured simple-imputed columns.
- `numeric_scaler`: `none`, `standard`, `robust`, or `minmax`. `scale_numeric = true`
  is a backward-compatible shortcut for standard scaling when
  `numeric_scaler = "none"`.
- `cap_numeric_quantiles`: when `true`, clips numeric values between
  `quantile_cap_lower` and `quantile_cap_upper`.
- `numeric_power_transform`: `none`, `yeo_johnson`, or `box_cox`. Box-Cox
  requires strictly positive input values.
- `numeric_distribution_transform`: `none`, `quantile_uniform`, or
  `quantile_normal`.
- `quantile_transform_n_quantiles`: maximum number of quantiles used by the
  quantile distribution transformer.
- `numeric_binning`: `none`, `uniform`, `quantile`, or `kmeans`.
- `numeric_bin_count`: number of bins when numeric binning is enabled.

Power transforms and quantile distribution transforms are mutually exclusive.
The currently implemented numeric scalers are standard, robust, and min-max.

### Categorical Preprocessing

Categorical options live in `[preprocessing]`:

- `categorical_imputer_strategy`: usually `most_frequent` or `constant`.
- `categorical_encoding`: `onehot`, `frequency`, `ordinal`, or `target`.
- `group_rare_categories`: when `true`, folds categories below
  `rare_category_min_frequency` into a shared rare-category value before
  encoding.
- `rare_category_min_frequency`: minimum category frequency as a fraction,
  for example `0.01`.
- `frequency_unknown_value`: value used by frequency encoding for unseen
  categories at prediction time.

`onehot` uses `OneHotEncoder(handle_unknown="ignore")`. `ordinal` maps unknown
categories to `-1` and missing values to `-2`. `target` replaces categories
with target-derived encodings and should be used only when the validation split
reflects how the model will see categories in production.

### Text And Datetime Features

Datetime columns are parsed and expanded into numeric calendar features:
`year`, `month`, `day`, and `dayofweek`.

Text columns are vectorized with per-column TF-IDF. Set
`text_max_features` to cap the vocabulary size per source text column:

```toml
[preprocessing]
text_max_features = 1000
```

### Baseline Estimators

Set `[pipeline].task` to choose the estimator registry:

- Classification: `dummy`, `logistic_regression`, `random_forest`,
  `hist_gradient_boosting`.
- Regression: `dummy`, `linear_regression`, `random_forest`,
  `hist_gradient_boosting`.
- Optional for both tasks when installed: `lightgbm`, `xgboost`, `catboost`.

Use `estimator_names` to run a subset:

```toml
[pipeline]
task = "classification"
estimator_names = ["dummy", "logistic_regression", "random_forest"]
```

### Validation And Search Example

This complete example shows the current `[pipeline.validation]` and `[search]`
configuration style:

```toml
[data]
kind = "csv"
path = "../data/my_data.csv"

[pipeline]
target_column = "target"
task = "classification"                 # "classification" or "regression"
random_state = 128
save_dir = "../runs/my_experiment"
estimator_names = ["dummy", "logistic_regression", "random_forest"]

[pipeline.validation]
method = "stratified_kfold"              # holdout, kfold, stratified_kfold, group_kfold, time_series_split
n_splits = 5                             # CV folds for baseline CV and search CV
test_size = 0.2                          # final holdout fraction
# groups_column = "customer_id"          # required for group_kfold; do not list as a feature
# time_column = "event_time"             # required for time_series_split; do not list as a feature

[columns]
numeric = ["age", "income"]
categorical = ["region"]
boolean = ["is_active"]
datetime = ["signup_date"]
text = ["support_notes"]

[preprocessing]
numeric_imputer_strategy = "median"
numeric_scaler = "robust"
categorical_encoding = "onehot"
text_max_features = 1000

[search]
method = "randomized"                    # none, grid, randomized
n_iter = 20                              # used only by randomized search
n_jobs = -1                              # sklearn parallel jobs
scoring = "accuracy"                     # single sklearn metric
refit = true                             # keep best fitted estimator for holdout evaluation

[search.estimators.random_forest]
n_estimators = [100, 300, 500]           # bare estimator param names are auto-resolved
max_depth = [5, 10, 20]
```

For multi-metric search, use a TOML inline table and set `refit` to one of its
metric names:

```toml
[search]
method = "grid"
scoring = { accuracy = "accuracy", f1_macro = "f1_macro" }
refit = "f1_macro"
```

### Pipeline Validation

The optional `[pipeline.validation]` table controls the holdout split, optional
baseline cross-validation, and the CV splitter used by hyperparameter search.
If the table is omitted, legacy `[pipeline]` settings are mapped automatically:
`cv_folds > 0` becomes `kfold`, or `stratified_kfold` when `stratify = true`;
otherwise the method is `holdout`. Legacy `test_size` is still used as the
holdout fraction.

| Parameter | Type | Default | Restrictions | Description |
| --- | --- | --- | --- | --- |
| `method` | string | `"holdout"` | One of `holdout`, `kfold`, `stratified_kfold`, `group_kfold`, `time_series_split`. `stratified_kfold` is classification-only. | Selects the validation strategy. `holdout` performs only the final train/test split. The other methods also run baseline CV on the training split. Search uses this method's CV splitter when parameter grids are configured. |
| `n_splits` | integer | `5` | Must be `>= 2`. For `group_kfold`, the training data must contain at least this many unique groups. | Number of folds for baseline CV when `method != "holdout"` and for hyperparameter search folds when search is run. For `holdout`, it still controls search CV folds. |
| `test_size` | float | `0.2` | Must satisfy `0.0 < test_size < 1.0`. | Fraction held out for final evaluation before optional baseline CV and search are run on the training portion. |
| `groups_column` | string or null | `null` | Required when `method = "group_kfold"`. Must exist, contain no missing values, and cannot be the target or any feature column. | Supplies group labels for group-aware train/test splitting, baseline CV, and search CV so related rows stay together and group leakage is reduced. The column is dropped before model fitting. |
| `time_column` | string or null | `null` | Required when `method = "time_series_split"`. Must exist, contain no missing values, be datetime-convertible or numeric, and cannot be the target or any feature column. | Sorts rows chronologically before splitting and CV. The final holdout split uses the first timestamp at the split boundary as a cutoff: rows before it train, rows at or after it test. If that would create an empty side, the pipeline falls back to positional splitting. The column is dropped before model fitting. |

Validation methods:

| Method | Supported tasks | Final holdout split | Baseline CV/search splitter | Notes |
| --- | --- | --- | --- | --- |
| `holdout` | Classification and regression | Random train/test split using `test_size`; stratified only when legacy `stratify = true` is passed through the API/config. | `KFold` for search CV. No baseline CV summary is produced. | Default when no validation table and no legacy `cv_folds` are provided. |
| `kfold` | Classification and regression | Random train/test split using `test_size`. | Shuffled `KFold`. | Produces baseline cross-validation metrics on the training split. |
| `stratified_kfold` | Classification only | Random train/test split using `test_size` and target stratification. | `StratifiedKFold` when every class has at least `n_splits` samples; otherwise warns and falls back to `KFold`. | Use for classification datasets where class balance should be preserved across folds. |
| `group_kfold` | Classification and regression | `GroupShuffleSplit` using `groups_column` and `test_size`. | `GroupKFold`. | Keeps each group entirely within one fold/split where possible. |
| `time_series_split` | Classification and regression | Chronological split using `time_column` and `test_size`. | `TimeSeriesSplit`. | Sorts by time before splitting; equal timestamps at the cutoff are assigned to the test side unless fallback positional splitting is needed. |

### Hyperparameter Search

The optional top-level `[search]` table controls estimator tuning. Search runs
only for estimators that have a corresponding parameter sub-table under
`[search.estimators]`; estimators without configured parameters are fitted
directly.

| Parameter | Type | Default | Restrictions | Description |
| --- | --- | --- | --- | --- |
| `method` | string | `"none"` | One of `none`, `grid`, `randomized`. | `none` fits estimators directly. `grid` uses `GridSearchCV`. `randomized` uses `RandomizedSearchCV`. |
| `n_iter` | integer | `10` | Must be `>= 1` when `method != "none"`. Used only by `randomized`. | Number of parameter combinations sampled by randomized search. |
| `n_jobs` | integer | `-1` | Passed through to sklearn. | Number of parallel worker jobs for search; `-1` uses all available cores. |
| `scoring` | string, table/dict, or null | `null` | Must be a valid sklearn scoring string, or a multi-metric dictionary/table accepted by sklearn. | Optimization metric(s). `null` uses the estimator default score. |
| `refit` | bool or string | `true` | `false` is not supported when search runs because the pipeline needs a fitted best model for holdout evaluation. If `scoring` is multi-metric, `refit` must be a metric-name string from the scoring keys. If `scoring` is single-metric or null, `refit` cannot be a string. | Controls which searched model is refit and returned as the final estimator. |
| `estimators` | table of estimator sub-tables | `{}` | Each estimator key must be active in `[pipeline].estimator_names` or in the task's default estimator registry. Parameter names must resolve to valid sklearn pipeline parameters. | Defines parameter grids/distributions, for example `[search.estimators.random_forest]`. |

Parameter resolution for estimator sub-tables:

| Parameter key style | Example | Resolution behavior |
| --- | --- | --- |
| Bare estimator parameter | `n_estimators = [100, 300]` | The pipeline tries `model__estimator__n_estimators`, then `model__n_estimators`, and uses the first valid key. |
| Fully qualified pipeline parameter | `model__estimator__max_depth = [5, 10]` | Used as written when it exists in `model_pipeline.get_params()`. |
| Invalid parameter | `does_not_exist = [1, 2]` | Raises `ConfigurationError` and reports valid pipeline parameter keys. |

Optional numeric and categorical transforms are disabled by default. Enable them
only when they fit the dataset and model family:

```toml
[preprocessing]
numeric_imputer_strategy = "median"
scale_numeric = true
numeric_scaler = "robust"              # "none", "standard", "robust", or "minmax"

cap_numeric_quantiles = true
quantile_cap_lower = 0.01
quantile_cap_upper = 0.99

numeric_power_transform = "yeo_johnson"  # "none", "yeo_johnson", or "box_cox"
numeric_distribution_transform = "none"  # "none", "quantile_uniform", or "quantile_normal"
numeric_binning = "none"                 # "none", "uniform", "quantile", or "kmeans"
numeric_bin_count = 10

categorical_encoding = "frequency"       # "onehot", "frequency", "ordinal", or "target"
group_rare_categories = true
rare_category_min_frequency = 0.01
frequency_unknown_value = 0.0

add_simple_missing_indicators = true
text_max_features = 1000
```

Datetime columns are converted into year, month, day, and dayofweek numeric
features. Text columns are vectorized with per-column TF-IDF using
`text_max_features` as the maximum vocabulary size per source column.
Use `categorical_encoding = "target"` for high-cardinality categoricals when
the category values are known at prediction time and the validation split
reflects the production setting.

Use `numeric_power_transform = "box_cox"` only for strictly positive numeric
features. The Box-Cox transformer raises a clear error for zero or negative
values instead of silently shifting the data. Power transforms and quantile
distribution transforms are mutually exclusive because both reshape numeric
distributions.

Set `imputer` to select the imputation method:
- `simple` (default): uses sklearn `SimpleImputer` on all feature columns.
- `stratified_hybrid`: uses cohort-aware imputation fallback logic (filling configured categorical/numeric columns from cohort/group medians or modes).
- `knn`: uses sklearn `KNNImputer` for numeric columns, with `SimpleImputer` fallback for other types.
- `iterative`: uses sklearn `IterativeImputer` (MICE) for numeric columns, with `SimpleImputer` fallback for other types.

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
- `metrics/feature_importances/*.csv`: native model importances or coefficients when available.
- `metrics/permutation_importances/*.csv`: model-agnostic permutation feature importance on the holdout split.
- `metrics/group_metrics/<model>/breakdown.csv`: per-cohort metrics when `[pipeline.validation].groups_column` is configured.
- `metrics/group_metrics/<model>/fairness.json`: classification cohort fairness diagnostics when groups are configured.
- `metadata/run_metadata.json`: task, target, split seed, input shape, schema, and feature outputs.
- `metadata/run_config.json`: the TOML config snapshot used for the run.

Because feature engineering and preprocessing are inside the saved sklearn
pipelines, inference should use the same raw input columns as training.

### Diagnostic Helpers

The evaluation module also exposes standalone diagnostics for ablation and data
quality analysis:

- `permutation_feature_importance(model, x_test, y_test, task=...)`: measures
  the metric drop from shuffling each raw feature column.
- `imputation_reconstruction_error(dataframe, config)`: masks known numeric
  values, imputes them, and reports reconstruction MSE, RMSE, and R2.
- `variance_inflation_factors(dataframe)`: reports VIF for numeric features to
  flag multicollinearity.
- `mutual_information_scores(x, y, task=...)`: reports univariate non-linear
  feature-target association scores after simple encoding.
- `group_metric_breakdown(...)` and `fairness_metrics(...)`: calculate cohort
  performance and classification fairness diagnostics such as disparate impact,
  demographic parity difference, and equalized odds difference.

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
