import os
import random

import numpy as np
import pandas as pd

try:
    from sdv.single_table import GaussianCopulaSynthesizer
except ModuleNotFoundError as exc:
    GaussianCopulaSynthesizer = None
    _SDV_IMPORT_ERROR = exc
else:
    _SDV_IMPORT_ERROR = None

try:
    from ..date_columns import finalize_synthetic_dates
except ImportError:
    from date_columns import finalize_synthetic_dates

try:
    from .backend_adapters import build_sdv_metadata
except ImportError:
    from backend_adapters import build_sdv_metadata

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


def generate(train_data, n_generated, output_dir, *, seed: int = 42):
    if GaussianCopulaSynthesizer is None:
        raise ModuleNotFoundError(
            "GaussianCopula generation requires the 'sdv' package to be installed."
        ) from _SDV_IMPORT_ERROR

    df = prepare_training_dataframe(train_data)
    _seed_everything(seed)

    metadata = build_sdv_metadata(df)
    model = GaussianCopulaSynthesizer(metadata)
    model.fit(df)
    new_data = model.sample(n_generated)
    new_data = finalize_synthetic_dates(new_data, df)
    float_cols = new_data.select_dtypes(include="float").columns
    new_data[float_cols] = new_data[float_cols].round(3)

    os.makedirs("synthetic_data", exist_ok=True)
    output_path = os.path.join("synthetic_data", f"{output_dir}")
    new_data.to_csv(output_path, index=False)
    return new_data


if __name__ == "__main__":
    generate("data/kaggle/ibm_hr.csv", 1500, "kaggle_GaussianCopula.csv")
