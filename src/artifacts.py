from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib

from src.config import MODELS_DIR
from src.data import resolve_path

from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


def save_model(
    model: BaseEstimator,
    file_path: str | Path,
    *,
    base_path: str | Path = MODELS_DIR,
) -> Path:
    """Save model to disk using joblib.

    Parameters
    ----------
    model : BaseEstimator
        The fitted estimator/pipeline to save.
    file_path : str or Path
        Target filename or path.
    base_path : str or Path
        Base directory, defaults to MODELS_DIR.

    Returns
    -------
    Path
        The absolute path to the saved file.
    """
    path = resolve_path(file_path, base_path, confine_to_base=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(
    file_path: str | Path,
    *,
    base_path: str | Path = MODELS_DIR,
) -> BaseEstimator:
    """Load model from disk using joblib.

    Parameters
    ----------
    file_path : str or Path
        Path to the saved model file.
    base_path : str or Path
        Base directory, defaults to MODELS_DIR.

    Returns
    -------
    BaseEstimator
        The loaded estimator/pipeline.
    """
    path = resolve_path(file_path, base_path, confine_to_base=True)
    return joblib.load(path)

