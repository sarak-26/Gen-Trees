from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd


def get_training_schema(df: pd.DataFrame) -> dict[str, Any]:
    schema = df.attrs.get("training_schema")
    if isinstance(schema, dict):
        return schema

    return {
        "columns": {
            column: {
                "semantic_type": "categorical"
                if (
                    pd.api.types.is_bool_dtype(df[column])
                    or pd.api.types.is_object_dtype(df[column])
                    or pd.api.types.is_string_dtype(df[column])
                    or isinstance(df[column].dtype, pd.CategoricalDtype)
                )
                else "numerical",
                "processed_dtype": str(df[column].dtype),
            }
            for column in df.columns
        }
    }


def build_sdv_metadata(df: pd.DataFrame):
    from sdv.metadata import SingleTableMetadata

    schema = get_training_schema(df)
    metadata = SingleTableMetadata()

    for column in df.columns:
        column_schema = schema.get("columns", {}).get(column, {})
        semantic_type = column_schema.get("semantic_type", "numerical")
        sdtype = "categorical" if semantic_type == "categorical" else "numerical"
        metadata.add_column(column_name=column, sdtype=sdtype)

    metadata.validate()
    return metadata


def _coerce_backend_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    adapted = df.copy()
    schema = get_training_schema(df)

    for column, column_schema in schema.get("columns", {}).items():
        if column not in adapted.columns:
            continue
        if column_schema.get("semantic_type") in {"date", "numerical"}:
            adapted[column] = pd.to_numeric(adapted[column], errors="coerce").astype("float64")

    adapted.attrs.update(df.attrs)
    return adapted


def adapt_for_ctabganplus(df: pd.DataFrame) -> pd.DataFrame:
    return _coerce_backend_numeric_columns(df)


def adapt_for_synthcity(df: pd.DataFrame) -> pd.DataFrame:
    adapted = _coerce_backend_numeric_columns(df)
    cache_root = (Path("workspace") / "cache").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("KEOPS_CACHE_FOLDER", str(cache_root / "keops"))
    return adapted
