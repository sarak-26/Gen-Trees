import os
import random

import numpy as np
import pandas as pd
from sdv.single_table import CTGANSynthesizer

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
    df = prepare_training_dataframe(train_data)
    _seed_everything(seed)

    metadata = build_sdv_metadata(df)
    model = CTGANSynthesizer(metadata)
    model.fit(df)
    new_data = model.sample(n_generated)
    new_data = finalize_synthetic_dates(new_data, df)
    float_cols = new_data.select_dtypes(include='float').columns
    new_data[float_cols] = new_data[float_cols].round(3)
    output_dir = os.path.join('synthetic_data', f'{output_dir}')
    new_data.to_csv(output_dir, index=False)
    return new_data

if __name__ == "__main__":
    generate('data/kaggle/ibm_hr.csv', 1500, 'kaggle_CTGAN.csv')


    
