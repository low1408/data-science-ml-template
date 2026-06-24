from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 128


@dataclass(frozen=True)
class ValidationConfig:
    method: str = "holdout"  # "holdout", "kfold", "stratified_kfold", "group_kfold", "time_series_split"
    n_splits: int = 5
    test_size: float = 0.2
    groups_column: str | None = None
    time_column: str | None = None


@dataclass(frozen=True)
class SearchConfig:
    method: str = "none"  # "none", "randomized", "grid"
    n_iter: int = 10
    n_jobs: int = -1
    scoring: str | dict[str, Any] | None = None
    refit: str | bool = True
    estimators: dict[str, dict[str, Any]] = field(default_factory=dict)



