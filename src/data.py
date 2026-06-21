from __future__ import annotations

from pathlib import Path
import sqlite3

import pandas as pd


class SQLiteDataLoader:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

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