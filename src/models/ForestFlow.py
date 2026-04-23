from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from ForestDiffusion import ForestDiffusionModel
except ModuleNotFoundError:
    ForestDiffusionModel = None

try:
    from ..date_columns import finalize_synthetic_dates
except ImportError:
    from date_columns import finalize_synthetic_dates

try:
    from .preprocessing import prepare_training_dataframe
except ImportError:
    from preprocessing import prepare_training_dataframe

ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = ROOT / "synthetic_data"


def _resolve_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _resolve_model_kwargs(model_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    resolved = dict(model_kwargs or {})
    env_overrides = {
        "n_estimators": ("FORESTFLOW_N_ESTIMATORS", 100),
        "max_depth": ("FORESTFLOW_MAX_DEPTH", 7),
    }

    for key, (env_name, _) in env_overrides.items():
        if key not in resolved and os.getenv(env_name):
            resolved[key] = _resolve_int_env(env_name, env_overrides[key][1])

    if "model" not in resolved and os.getenv("FORESTFLOW_MODEL"):
        resolved["model"] = os.getenv("FORESTFLOW_MODEL", "xgboost")

    return resolved


def _is_categorical(series: pd.Series) -> bool:
    return (
        pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def _encode_categorical(series: pd.Series) -> tuple[pd.Series, list[Any]]:
    categories = series.dropna().unique().tolist()
    mapping = {value: code for code, value in enumerate(categories)}
    return series.map(mapping).astype(float), categories


def _classify_numeric(series: pd.Series, threshold: int) -> tuple[str, list[Any] | None]:
    numeric = pd.to_numeric(series, errors="coerce")
    non_null = numeric.dropna()
    if non_null.empty or not np.allclose(non_null, np.round(non_null)):
        return "float", None

    unique_values = sorted(non_null.unique().tolist())
    if len(unique_values) <= 2:
        return "binary", unique_values
    if len(unique_values) <= threshold:
        return "categorical", unique_values
    return "integer", None


def _encode_dataframe(df: pd.DataFrame, discrete_cardinality_threshold: int):
    encoded = pd.DataFrame(index=df.index)
    metadata: list[dict[str, Any]] = []
    index_map = {"binary": [], "categorical": [], "integer": []}

    for idx, column in enumerate(df.columns):
        series = df[column]
        info: dict[str, Any] = {"name": column, "kind": "float", "categories": None}

        if _is_categorical(series):
            encoded[column], info["categories"] = _encode_categorical(series)
            info["kind"] = "binary" if len(info["categories"]) <= 2 else "categorical"
        else:
            numeric = pd.to_numeric(series, errors="coerce").astype(float)
            info["kind"], info["categories"] = _classify_numeric(series, discrete_cardinality_threshold)
            if info["kind"] in {"binary", "categorical"}:
                mapping = {value: code for code, value in enumerate(info["categories"])}
                encoded[column] = numeric.map(mapping).astype(float)
            else:
                encoded[column] = numeric

        if info["kind"] in index_map:
            index_map[info["kind"]].append(idx)
        metadata.append(info)

    return encoded, metadata, index_map["binary"], index_map["categorical"], index_map["integer"]


def _decode_column(values: np.ndarray, info: dict[str, Any]) -> list[Any] | np.ndarray:
    if info["kind"] in {"binary", "categorical"}:
        categories = info["categories"] or []
        codes = np.clip(np.rint(values).astype(int), 0, max(len(categories) - 1, 0))
        return [categories[code] if categories else pd.NA for code in codes]
    if info["kind"] == "integer":
        return np.rint(values).astype(int)
    return values


def _decode_dataframe(array: np.ndarray, metadata, columns) -> pd.DataFrame:
    decoded = {
        info["name"]: _decode_column(array[:, idx], info)
        for idx, info in enumerate(metadata)
    }
    return pd.DataFrame(decoded, columns=columns)


def generate(
    train_data,
    n_generated,
    output_dir,
    *,
    n_t: int = 10,
    duplicate_K: int = 20,
    n_batch: int = 1,
    n_jobs: int = 1,
    seed: int = 42,
    discrete_cardinality_threshold: int = 10,
    model_kwargs: dict | None = None,
):
    if ForestDiffusionModel is None:
        raise ModuleNotFoundError("ForestDiffusion is not installed.")

    df = prepare_training_dataframe(
        train_data, discrete_cardinality_threshold=discrete_cardinality_threshold
    )
    if df.empty:
        raise ValueError("Input training data must be a non-empty pandas DataFrame.")

    encoded, metadata, bin_indexes, cat_indexes, int_indexes = _encode_dataframe(
        df, discrete_cardinality_threshold=discrete_cardinality_threshold
    )

    env_n_t = _resolve_int_env("FORESTFLOW_N_T", n_t)
    env_duplicate_k = _resolve_int_env("FORESTFLOW_DUPLICATE_K", duplicate_K)
    env_n_batch = _resolve_int_env("FORESTFLOW_N_BATCH", n_batch)
    env_n_jobs = _resolve_int_env("FORESTFLOW_N_JOBS", n_jobs)
    resolved_model_kwargs = _resolve_model_kwargs(model_kwargs)

    kind_to_columns = {
        "binary": [metadata[idx]["name"] for idx in bin_indexes],
        "categorical": [metadata[idx]["name"] for idx in cat_indexes],
        "integer": [metadata[idx]["name"] for idx in int_indexes],
    }
    print(
        "[ForestFlow] column kinds="
        f"binary={kind_to_columns['binary']} "
        f"categorical={kind_to_columns['categorical']} "
        f"integer={kind_to_columns['integer']}",
        flush=True,
    )

    estimated_training_rows = len(df) * env_duplicate_k
    print(
        "[ForestFlow] build config="
        f"n_t={env_n_t} duplicate_K={env_duplicate_k} n_batch={env_n_batch} n_jobs={env_n_jobs} "
        f"estimated_fits={env_n_t} estimated_training_rows_per_fit~={estimated_training_rows} "
        f"extra_model_kwargs={resolved_model_kwargs}",
        flush=True,
    )
    print(
        f"[ForestFlow] seed={seed} n_jobs={env_n_jobs} rows={len(df)} cols={len(df.columns)} starting model build",
        flush=True,
    )
    model = ForestDiffusionModel(
        encoded.to_numpy(dtype=float),
        n_t=env_n_t,
        duplicate_K=env_duplicate_k,
        n_batch=env_n_batch,
        diffusion_type="flow",
        bin_indexes=bin_indexes,
        cat_indexes=cat_indexes,
        int_indexes=int_indexes,
        n_jobs=env_n_jobs,
        seed=seed,
        **resolved_model_kwargs,
    )
    print(f"[ForestFlow] seed={seed} starting sample generation", flush=True)
    samples = model.generate(batch_size=int(n_generated))
    print(f"[ForestFlow] seed={seed} sample generation complete", flush=True)
    new_data = _decode_dataframe(np.asarray(samples, dtype=float), metadata, list(df.columns))
    new_data = finalize_synthetic_dates(new_data, df)

    float_cols = new_data.select_dtypes(include="float").columns
    new_data[float_cols] = new_data[float_cols].round(3)

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SYNTHETIC_DIR / output_dir
    new_data.to_csv(output_path, index=False)
    print(f"[ForestFlow] seed={seed} wrote {output_path}", flush=True)
    return new_data


if __name__ == "__main__":
    generate("data/Employee.csv", 500, "Employee_ForestFlow.csv")
