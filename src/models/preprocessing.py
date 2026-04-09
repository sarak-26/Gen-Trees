from __future__ import annotations

import pandas as pd


def _stringify_numeric_value(value) -> str:
    if pd.isna(value):
        return "<NA>"

    if isinstance(value, (int, str)):
        return str(value)

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value)


def prepare_training_dataframe(
    train_data: str | pd.DataFrame,
    discrete_cardinality_threshold: int = 10,
) -> pd.DataFrame:
    if isinstance(train_data, pd.DataFrame):
        df = train_data.copy()
    else:
        df = pd.read_csv(train_data, header=0)

    for col in df.columns:
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
            df[col] = numeric.map(_stringify_numeric_value).astype(object)

    return df
