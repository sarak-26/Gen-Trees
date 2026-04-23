import os
import random
from typing import Any

import numpy as np
import pandas as pd

try:
    from model.ctabgan import CTABGAN
    from model.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer
except Exception as exc:
    CTABGAN = None
    CTABGANSynthesizer = None
    _CTABGAN_IMPORT_ERROR = exc
else:
    _CTABGAN_IMPORT_ERROR = None

try:
    from ..date_columns import finalize_synthetic_dates
except ImportError:
    from date_columns import finalize_synthetic_dates

try:
    from .backend_adapters import adapt_for_ctabganplus
except ImportError:
    from backend_adapters import adapt_for_ctabganplus

try:
    from .preprocessing import prepare_training_dataframe
except ImportError:
    from preprocessing import prepare_training_dataframe


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_effectively_integer(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.dropna()
    if numeric.empty:
        return False
    return ((numeric % 1) == 0).all()


def _build_ctabgan_config(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        raise ValueError("Input training data must be a non-empty pandas DataFrame.")

    categorical_columns: list[str] = []
    log_columns: list[str] = []
    mixed_columns: dict[str, list[Any]] = {}
    general_columns: list[str] = []
    non_categorical_columns: list[str] = []
    integer_columns: list[str] = []

    row_count = len(df)

    for column in df.columns:
        series = df[column]
        non_null = series.dropna()

        if (
            pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        ):
            categorical_columns.append(column)
            continue

        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() == 0:
            categorical_columns.append(column)
            continue

        if _is_effectively_integer(series):
            integer_columns.append(column)

        cardinality = int(numeric.dropna().nunique())
        if 0 < cardinality <= min(10, max(3, row_count // 20)):
            categorical_columns.append(column)
            continue

        if (numeric > 0).all():
            log_columns.append(column)

        general_columns.append(column)

    target_column = df.columns[-1]
    target_series = df[target_column]
    target_numeric = pd.to_numeric(target_series, errors="coerce")
    target_unique = int(target_series.dropna().nunique())
    target_is_classification = (
        pd.api.types.is_bool_dtype(target_series)
        or pd.api.types.is_object_dtype(target_series)
        or pd.api.types.is_string_dtype(target_series)
        or pd.api.types.is_categorical_dtype(target_series)
        or (target_numeric.notna().sum() > 0 and target_unique <= min(20, max(3, row_count // 10)))
    )

    if target_is_classification:
        if target_column not in categorical_columns:
            categorical_columns.append(target_column)
        problem_type = {"Classification": target_column}
    else:
        problem_type = {"Regression": target_column}

    return {
        "categorical_columns": categorical_columns,
        "log_columns": log_columns,
        "mixed_columns": mixed_columns,
        "general_columns": general_columns,
        "non_categorical_columns": non_categorical_columns,
        "integer_columns": integer_columns,
        "problem_type": problem_type,
    }


def generate(train_data, n_generated, output_dir, *, seed: int = 42):
    if CTABGAN is None:
        raise RuntimeError(
            "CTABGAN+ backend is unavailable. Ensure its local dependencies are installed."
        ) from _CTABGAN_IMPORT_ERROR

    df = adapt_for_ctabganplus(prepare_training_dataframe(train_data))
    _seed_everything(seed)
    config = _build_ctabgan_config(df)

    model = CTABGAN.__new__(CTABGAN)
    model.__name__ = "CTABGAN"
    model.synthesizer = CTABGANSynthesizer()
    model.test_ratio = 0.20
    model.categorical_columns = config["categorical_columns"]
    model.log_columns = config["log_columns"]
    model.mixed_columns = config["mixed_columns"]
    model.general_columns = config["general_columns"]
    model.non_categorical_columns = config["non_categorical_columns"]
    model.integer_columns = config["integer_columns"]
    model.problem_type = config["problem_type"]

    model.raw_df = df.copy()

    model.fit()
    new_data = model.synthesizer.sample(int(n_generated))
    new_data = model.data_prep.inverse_prep(new_data)
    new_data.columns = df.columns
    new_data = finalize_synthetic_dates(new_data, df)

    float_cols = new_data.select_dtypes(include="float").columns
    new_data[float_cols] = new_data[float_cols].round(3)

    os.makedirs("synthetic_data", exist_ok=True)
    output_path = os.path.join("synthetic_data", f"{output_dir}")
    new_data.to_csv(output_path, index=False)
    return new_data


if __name__ == "__main__":
    generate("data/Employee.csv", 1500, "Employee_CTABGANPlus.csv")
