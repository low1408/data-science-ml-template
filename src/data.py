from __future__ import annotations

from pathlib import Path
import sqlite3

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SQLiteDataLoader:
    def __init__(self, db_path: str | Path, base_path: str | Path = PROJECT_ROOT) -> None:
        db_path = Path(db_path).expanduser()

        if not db_path.is_absolute():
            db_path = Path(base_path).expanduser() / db_path

        self.db_path = db_path.resolve()

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

    def load_table(self, table_name: str) -> pd.DataFrame:
        query = f"SELECT * FROM {self._quote_identifier(table_name)}"

        with sqlite3.connect(self.db_path) as connection:
            return pd.read_sql_query(query, connection)

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
