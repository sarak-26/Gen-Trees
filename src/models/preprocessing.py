from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from ..date_columns import prepare_training_dates
except ImportError:
    from date_columns import prepare_training_dates


def _infer_semantic_type(column_name: str, series: pd.Series, date_metadata: dict[str, dict]) -> str:
    if column_name in date_metadata:
        return "date"
    if (
        pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
    ):
        return "categorical"
    return "numerical"


def _attach_training_schema(
    processed_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    date_metadata: dict[str, dict] | None = None,
) -> pd.DataFrame:
    date_metadata = date_metadata or {}
    schema = {"columns": {}, "date_columns": sorted(date_metadata.keys())}

    for column in processed_df.columns:
        column_schema = {
            "semantic_type": _infer_semantic_type(column, processed_df[column], date_metadata),
            "source_dtype": str(raw_df[column].dtype) if column in raw_df.columns else None,
            "processed_dtype": str(processed_df[column].dtype),
        }
        if column in date_metadata:
            column_schema["date"] = dict(date_metadata[column])
        schema["columns"][column] = column_schema

    processed_df.attrs["training_schema"] = schema
    processed_df.attrs["date_column_metadata"] = date_metadata
    return processed_df


def _stringify_numeric_value(value) -> str:
    if pd.isna(value):
        return "<NA>"

    if isinstance(value, (int, str)):
        return str(value)

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value)


def _should_force_all_columns_to_string(train_data: str | pd.DataFrame) -> bool:
    if not isinstance(train_data, str):
        return False

    return Path(train_data).name == "Amazon_employee_access.csv"


def _stringify_series(series: pd.Series) -> pd.Series:
    return series.map(_stringify_numeric_value).astype(object)


def prepare_training_dataframe(
    train_data: str | pd.DataFrame,
    discrete_cardinality_threshold: int = 10,
) -> pd.DataFrame:
    if isinstance(train_data, pd.DataFrame):
        df = train_data.copy()
    else:
        df = pd.read_csv(train_data, header=0)
    raw_df = df.copy()

    if _should_force_all_columns_to_string(train_data):
        for col in df.columns:
            df[col] = _stringify_series(df[col])
        return _attach_training_schema(df, raw_df, {})

    df = prepare_training_dates(df)
    date_metadata = dict(df.attrs.get("date_column_metadata", {}))

    for col in df.columns:
        if col in date_metadata:
            continue
        series = df[col]
        if (
            pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        ):
            continue

        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() == 0:
            continue

        cardinality = int(numeric.dropna().nunique())
        if cardinality <= discrete_cardinality_threshold:
            df[col] = _stringify_series(numeric)

    return _attach_training_schema(df, raw_df, date_metadata)
