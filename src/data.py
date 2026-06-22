from __future__ import annotations

import logging
import sqlite3
import warnings
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROJECT_ROOT, RANDOM_STATE

logger = logging.getLogger(__name__)


class TabularDataLoader(Protocol):
    """Common interface for loaders that return tabular pandas data."""

    def load(self) -> pd.DataFrame:
        ...


def resolve_path(path: str | Path, base_path: str | Path = PROJECT_ROOT) -> Path:
    path = Path(path).expanduser()

    if not path.is_absolute():
        path = Path(base_path).expanduser() / path

    return path.resolve()


# ── Data-splitting utilities (moved from preprocessing.py — F-08) ────


def split_features_target(
    dataframe: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in dataframe.columns:
        raise KeyError(f"Target column not found: {target_column}")

    return dataframe.drop(columns=[target_column]), dataframe[target_column]


def train_test_split_dataframe(
    dataframe: pd.DataFrame,
    target_column: str,
    *,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    stratify: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    features, target = split_features_target(dataframe, target_column)
    stratify_values = target if stratify else None

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values,
    )


# ── Data loaders ─────────────────────────────────────────────────────


class CSVDataLoader:
    def __init__(
        self,
        file_path: str | Path,
        base_path: str | Path = PROJECT_ROOT,
        **read_csv_kwargs: Any,
    ) -> None:
        self.file_path = resolve_path(file_path, base_path)
        self.read_csv_kwargs = read_csv_kwargs

        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.file_path, **self.read_csv_kwargs)


class ParquetDataLoader:
    def __init__(
        self,
        file_path: str | Path,
        base_path: str | Path = PROJECT_ROOT,
        **read_parquet_kwargs: Any,
    ) -> None:
        self.file_path = resolve_path(file_path, base_path)
        self.read_parquet_kwargs = read_parquet_kwargs

        if not self.file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {self.file_path}")

    def load(self) -> pd.DataFrame:
        return pd.read_parquet(self.file_path, **self.read_parquet_kwargs)


class SQLiteDataLoader:
    def __init__(self, db_path: str | Path, base_path: str | Path = PROJECT_ROOT) -> None:
        self.db_path = resolve_path(db_path, base_path)

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

    def load(self, table_name: str | None = None) -> pd.DataFrame:
        if table_name is not None:
            return self.load_table(table_name)

        tables = self.list_tables()
        if len(tables) != 1:
            raise ValueError(
                "SQLiteDataLoader.load() requires table_name when the database "
                f"contains {len(tables)} tables."
            )

        return self.load_table(tables[0])

    def load_table(
        self,
        table_name: str,
        columns: list[str] | None = None,
        where: str | None = None,
        params: tuple[Any, ...] | dict[str, Any] | None = None,
        predicates: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Load rows from a single table.

        .. warning:: SQL injection risk

            The ``where`` argument is interpolated directly into the query
            string. The ``params`` argument protects parameter values only,
            not the query structure; a UNION injection or structural modification
            can still occur via the ``where`` argument itself if it is built using
            untrusted input.

            For a safe, structured approach, prefer using the ``predicates``
            argument, which builds the WHERE clause internally and parameterises
            all values.

        :param table_name: Name of the table to load.
        :param columns: List of columns to retrieve.
        :param where: Raw WHERE clause string (e.g. "age > ?").
        :param params: Parameters for the raw WHERE clause.
        :param predicates: A dictionary of column-value mappings (e.g. {"status": "active"}).
                           Values can be single elements, None (translates to IS NULL), or
                           lists/tuples (translates to IN).
        :param limit: Maximum number of rows to return.
        """
        if where and predicates:
            raise ValueError("Cannot specify both 'where' and 'predicates'.")
        if predicates and params is not None:
            raise ValueError("Cannot specify both 'predicates' and 'params'.")

        selected_columns = "*"
        if columns:
            selected_columns = ", ".join(self._quote_identifier(column) for column in columns)

        query = f"SELECT {selected_columns} FROM {self._quote_identifier(table_name)}"
        query_params = None

        if where:
            warnings.warn(
                "Using the raw 'where' argument exposes a risk of SQL injection. "
                "The 'params' argument protects values only, not the SQL structure. "
                "Prefer using 'predicates' for structured, safe filtering.",
                UserWarning,
                stacklevel=2,
            )
            if params is None:
                warnings.warn(
                    "load_table() received a 'where' clause without 'params'. "
                    "Use parameterised predicates (where='col > ?', params=(val,)) "
                    "to guard against SQL injection.",
                    UserWarning,
                    stacklevel=2,
                )
                logger.warning(
                    "Unparameterised WHERE clause passed to load_table: %r", where
                )
            query = f"{query} WHERE {where}"
            query_params = params
        elif predicates:
            where_clauses = []
            where_params = []
            for col, val in predicates.items():
                quoted_col = self._quote_identifier(col)
                if val is None:
                    where_clauses.append(f"{quoted_col} IS NULL")
                elif isinstance(val, (list, tuple)):
                    if len(val) == 0:
                        where_clauses.append("1 = 0")
                    else:
                        placeholders = ", ".join("?" for _ in val)
                        where_clauses.append(f"{quoted_col} IN ({placeholders})")
                        where_params.extend(val)
                else:
                    where_clauses.append(f"{quoted_col} = ?")
                    where_params.append(val)
            if where_clauses:
                query = f"{query} WHERE {' AND '.join(where_clauses)}"
                query_params = tuple(where_params)

        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be greater than or equal to 0.")
            query = f"{query} LIMIT {limit}"

        with sqlite3.connect(self.db_path) as connection:
            return pd.read_sql_query(query, connection, params=query_params)

    def load_raw_query(
        self,
        query: str,
        params: tuple[Any, ...] | dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Execute a raw SQL query.

        .. warning:: SQL injection risk

            Executing raw SQL queries is inherently risky if any part of the query
            string is built using untrusted input. Always use the ``params``
            argument to parameterise values.
        """
        with sqlite3.connect(self.db_path) as connection:
            return pd.read_sql_query(query, connection, params=params)

    def load_query(
        self,
        query: str,
        params: tuple[Any, ...] | dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Deprecated. Use load_raw_query instead."""
        warnings.warn(
            "load_query is deprecated and will be removed in a future release. "
            "Use load_raw_query instead for raw query execution.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.load_raw_query(query, params)

    def list_tables(self) -> list[str]:
        query = """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """

        with sqlite3.connect(self.db_path) as connection:
            result = pd.read_sql_query(query, connection)

        return result["name"].tolist()

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        escaped_identifier = identifier.replace('"', '""')
        return f'"{escaped_identifier}"'
