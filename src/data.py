from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol
import sqlite3

import pandas as pd

from src.config import PROJECT_ROOT


class TabularDataLoader(Protocol):
    """Common interface for loaders that return tabular pandas data."""

    def load(self) -> pd.DataFrame:
        ...


def resolve_path(path: str | Path, base_path: str | Path = PROJECT_ROOT) -> Path:
    path = Path(path).expanduser()

    if not path.is_absolute():
        path = Path(base_path).expanduser() / path

    return path.resolve()


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
        limit: int | None = None,
    ) -> pd.DataFrame:
        selected_columns = "*"
        if columns:
            selected_columns = ", ".join(self._quote_identifier(column) for column in columns)

        query = f"SELECT {selected_columns} FROM {self._quote_identifier(table_name)}"
        if where:
            query = f"{query} WHERE {where}"
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be greater than or equal to 0.")
            query = f"{query} LIMIT {limit}"

        with sqlite3.connect(self.db_path) as connection:
            return pd.read_sql_query(query, connection, params=params)

    def load_query(
        self,
        query: str,
        params: tuple[Any, ...] | dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as connection:
            return pd.read_sql_query(query, connection, params=params)

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
