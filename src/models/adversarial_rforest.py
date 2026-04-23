import os
import random

import numpy as np
import pandas as pd
from arfpy import arf

try:
    from ..date_columns import finalize_synthetic_dates
except ImportError:
    from date_columns import finalize_synthetic_dates

try:
    from .preprocessing import prepare_training_dataframe
except ImportError:
    from preprocessing import prepare_training_dataframe

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _continuous_columns(df: pd.DataFrame) -> list[str]:
    continuous_cols: list[str] = []
    for col in df.columns:
        series = df[col]
        if (
            pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        ):
            continue
        continuous_cols.append(col)
    return continuous_cols


def _integer_like_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    integer_like: list[str] = []
    for col in columns:
        numeric = pd.to_numeric(df[col], errors="coerce").dropna()
        if numeric.empty:
            continue
        if np.all(np.isclose(numeric, np.round(numeric))):
            integer_like.append(col)
    return integer_like


def _local_leaf_bounds(model, continuous_cols: list[str]) -> pd.DataFrame:
    pred = model.clf.apply(model.x_real)
    frames: list[pd.DataFrame] = []
    for tree in range(model.num_trees):
        node_frame = model.x_real.loc[:, continuous_cols].copy()
        node_frame["nodeid"] = pred[:, tree]
        long = node_frame.melt(id_vars="nodeid", var_name="variable", value_name="value")
        stats = (
            long.groupby(["nodeid", "variable"], as_index=False)
            .agg(local_min=("value", "min"), local_max=("value", "max"))
        )
        stats["tree"] = tree
        frames.append(stats)
    return pd.concat(frames, ignore_index=True)


def _apply_finite_bounds(
    model,
    train_df: pd.DataFrame,
    *,
    finite_bounds: str = "global",
    epsilon: float = 1e-14,
) -> None:
    if finite_bounds == "no" or model.params.empty:
        return

    continuous_cols = _continuous_columns(train_df)
    if not continuous_cols:
        return

    params = model.params.copy()

    if finite_bounds == "global":
        global_min = train_df[continuous_cols].min().to_dict()
        global_max = train_df[continuous_cols].max().to_dict()
        min_mask = np.isneginf(params["min"])
        max_mask = np.isposinf(params["max"])
        params.loc[min_mask, "min"] = params.loc[min_mask, "variable"].map(global_min)
        params.loc[max_mask, "max"] = params.loc[max_mask, "variable"].map(global_max)
    elif finite_bounds == "local":
        local_bounds = _local_leaf_bounds(model, continuous_cols)
        params = params.merge(local_bounds, on=["tree", "nodeid", "variable"], how="left")
        min_mask = np.isneginf(params["min"])
        max_mask = np.isposinf(params["max"])
        params.loc[min_mask, "min"] = params.loc[min_mask, "local_min"]
        params.loc[max_mask, "max"] = params.loc[max_mask, "local_max"]
        params = params.drop(columns=["local_min", "local_max"])
    else:
        raise ValueError("finite_bounds must be one of: 'global', 'local', 'no'")

    finite_mask = np.isfinite(params["min"]) & np.isfinite(params["max"])
    widths = params.loc[finite_mask, "max"] - params.loc[finite_mask, "min"]
    padding = widths * (epsilon / 2.0)
    params.loc[finite_mask, "min"] = params.loc[finite_mask, "min"] - padding
    params.loc[finite_mask, "max"] = params.loc[finite_mask, "max"] + padding

    # arfpy can emit degenerate leaf fits with sd <= 0 or NaN, which later
    # causes scipy.truncnorm to raise. Keep those leaves effectively point-mass.
    invalid_sd_mask = ~np.isfinite(params["sd"]) | (params["sd"] <= 0)
    if invalid_sd_mask.any():
        params.loc[invalid_sd_mask, "sd"] = 1e-9
        same_bounds = invalid_sd_mask & np.isfinite(params["min"]) & np.isfinite(params["max"]) & (
            np.isclose(params["min"], params["max"])
        )
        params.loc[same_bounds, "min"] = params.loc[same_bounds, "mean"] - 1e-9
        params.loc[same_bounds, "max"] = params.loc[same_bounds, "mean"] + 1e-9

    model.params = params


def _postprocess_generated_data(new_data: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    continuous_cols = _continuous_columns(train_df)
    if not continuous_cols:
        return new_data

    lower = train_df[continuous_cols].min()
    upper = train_df[continuous_cols].max()
    new_data[continuous_cols] = new_data[continuous_cols].clip(lower=lower, upper=upper, axis=1)

    integer_like_cols = _integer_like_columns(train_df, continuous_cols)
    if integer_like_cols:
        new_data[integer_like_cols] = new_data[integer_like_cols].round(0)

    float_cols = new_data.select_dtypes(include="float").columns
    if len(float_cols) > 0:
        new_data[float_cols] = new_data[float_cols].round(3)

    return new_data


def generate(
    train_data,
    n_generated,
    output_dir,
    *,
    seed: int = 42,
    finite_bounds: str = "global",
    epsilon: float = 1e-14,
):
    # iris = load_iris()
    # print(iris['feature_names'])
    # df = pd.DataFrame(iris['data'], columns=iris['feature_names'])

    df = prepare_training_dataframe(train_data)
    _seed_everything(seed)
    myarf = arf.arf(x = df)
    myarf.forde()
    _apply_finite_bounds(myarf, df, finite_bounds=finite_bounds, epsilon=epsilon)
    new_data = myarf.forge(n = n_generated)
    new_data = _postprocess_generated_data(new_data, df)
    new_data = finalize_synthetic_dates(new_data, df)
    output_dir = os.path.join('synthetic_data', f'{output_dir}')
    new_data.to_csv(output_dir, index=False)
    return new_data

if __name__ == "__main__":
    generate('data/kaggle/ibm_hr.csv', 1500, 'kaggle_arf.csv')
