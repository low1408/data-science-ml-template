from __future__ import annotations

import pandas as pd

from src.preprocessing import infer_feature_columns


def test_infer_feature_columns_skips_target_and_assigns_roles():
    dataframe = pd.DataFrame(
        {
            "age": [20.0, 30.0, 40.0],
            "region_code": [1, 2, 1],
            "active": [True, False, True],
            "is_member": [1, 0, 1],
            "segment": ["a", "b", "a"],
            "created_at": pd.to_datetime(["2024-01-01", "2024-01-02", None]),
            "target": [0, 1, 0],
        }
    )

    columns = infer_feature_columns(
        dataframe,
        target_column="target",
        categorical_max_unique=2,
    )

    assert columns.numeric == ("age",)
    assert columns.categorical == ("region_code", "segment")
    assert columns.boolean == ("active", "is_member")
    assert columns.datetime == ("created_at",)
