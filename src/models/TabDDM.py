import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    from ..date_columns import finalize_synthetic_dates
except ImportError:
    from date_columns import finalize_synthetic_dates

try:
    from .backend_adapters import adapt_for_synthcity
except ImportError:
    from backend_adapters import adapt_for_synthcity

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


class TabDDM:
    def __init__(
        self,
        plugin_name: str = "ddpm",
        plugin_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if plugin_name != "ddpm":
            raise ValueError("TabDDM only supports the SynthCity 'ddpm' plugin.")

        self.plugin_name = plugin_name
        self.plugin_kwargs = plugin_kwargs or {}
        self.plugin: Any = None
        self.backend = "uninitialized"

    def _build_plugin(self) -> Any:
        try:
            from synthcity.plugins.generic.plugin_ddpm import TabDDPMPlugin
        except Exception as exc:
            raise RuntimeError(
                "SynthCity TabDDM backend is unavailable. "
                "Install and configure synthcity dependencies to use this model."
            ) from exc

        plugin_kwargs = {
            "n_iter": 2000,
            "batch_size": 256,
            "lr": 1e-5,
            **self.plugin_kwargs,
        }
        plugin_kwargs.setdefault("workspace", Path("workspace"))

        return TabDDPMPlugin(**plugin_kwargs)

    def fit(self, df: pd.DataFrame) -> "TabDDM":
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Input training data must be a non-empty pandas DataFrame.")

        self.plugin = self._build_plugin()
        self.plugin.fit(df)
        self.backend = f"synthcity:{self.plugin_name}"
        return self

    def sample(self, n: int) -> pd.DataFrame:
        if int(n) <= 0:
            raise ValueError("n must be a positive integer.")
        if self.plugin is None:
            raise RuntimeError("Model must be fitted before sampling.")

        synthetic = self.plugin.generate(count=int(n))
        if hasattr(synthetic, "dataframe"):
            dataframe_attr = synthetic.dataframe
            return dataframe_attr() if callable(dataframe_attr) else dataframe_attr
        if hasattr(synthetic, "data"):
            return synthetic.data
        if isinstance(synthetic, pd.DataFrame):
            return synthetic
        raise TypeError(
            "Unexpected SynthCity generate output; expected a DataLoader-like object or DataFrame."
        )


def generate(train_data, n_generated, output_dir, *, seed: int = 42):
    df = adapt_for_synthcity(prepare_training_dataframe(train_data))
    _seed_everything(seed)
    model = TabDDM(
        plugin_name="ddpm",
        plugin_kwargs={"n_iter": 2000, "batch_size": 256, "lr": 1e-5, "random_state": seed},
    )
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
    generate("data/ibm_hr.csv", 1500, "ibm_hr_TabDDM.csv")
