import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class _PreprocessState:
    numeric_cols: List[str]
    categorical_cols: List[str]
    numeric_mean: np.ndarray
    numeric_std: np.ndarray
    categorical_levels: Dict[str, List[str]]
    categorical_slices: Dict[str, Tuple[int, int]]
    n_numeric: int
    n_total: int
    original_columns: List[str]


def _infer_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    categorical_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
    numeric_cols = [col for col in df.columns if col not in categorical_cols]
    return numeric_cols, categorical_cols


def _fit_transform(df: pd.DataFrame) -> Tuple[np.ndarray, _PreprocessState]:
    df = df.copy()
    original_columns = list(df.columns)
    numeric_cols, categorical_cols = _infer_columns(df)

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].astype(str).fillna("MISSING")

    if numeric_cols:
        x_num = df[numeric_cols].to_numpy(dtype=np.float32)
        num_mean = x_num.mean(axis=0)
        num_std = x_num.std(axis=0)
        num_std = np.where(num_std < 1e-6, 1.0, num_std)
        x_num = (x_num - num_mean) / num_std
    else:
        x_num = np.zeros((len(df), 0), dtype=np.float32)
        num_mean = np.zeros((0,), dtype=np.float32)
        num_std = np.ones((0,), dtype=np.float32)

    cat_parts = []
    levels_map: Dict[str, List[str]] = {}
    slices: Dict[str, Tuple[int, int]] = {}
    cursor = x_num.shape[1]

    for col in categorical_cols:
        values = df[col].astype(str)
        levels = sorted(values.unique().tolist())
        level_to_idx = {value: idx for idx, value in enumerate(levels)}
        idx_values = values.map(level_to_idx).to_numpy()
        one_hot = np.zeros((len(df), len(levels)), dtype=np.float32)
        one_hot[np.arange(len(df)), idx_values] = 1.0
        cat_parts.append(one_hot)
        levels_map[col] = levels
        slices[col] = (cursor, cursor + len(levels))
        cursor += len(levels)

    x_cat = np.concatenate(cat_parts, axis=1) if cat_parts else np.zeros((len(df), 0), dtype=np.float32)
    x_all = np.concatenate([x_num, x_cat], axis=1).astype(np.float32)

    state = _PreprocessState(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        numeric_mean=num_mean,
        numeric_std=num_std,
        categorical_levels=levels_map,
        categorical_slices=slices,
        n_numeric=x_num.shape[1],
        n_total=x_all.shape[1],
        original_columns=original_columns,
    )
    return x_all, state


def _inverse_transform(x: np.ndarray, state: _PreprocessState) -> pd.DataFrame:
    output = pd.DataFrame(index=np.arange(len(x)))

    if state.n_numeric > 0:
        x_num = x[:, : state.n_numeric]
        denormalized = x_num * state.numeric_std + state.numeric_mean
        for idx, col in enumerate(state.numeric_cols):
            output[col] = denormalized[:, idx]

    for col in state.categorical_cols:
        start, end = state.categorical_slices[col]
        logits = x[:, start:end]
        idx_values = logits.argmax(axis=1)
        levels = state.categorical_levels[col]
        output[col] = [levels[idx] for idx in idx_values]

    return output[state.original_columns]


class _Denoiser(nn.Module):
    def __init__(self, input_dim: int, n_steps: int, hidden_dim: int = 256, time_dim: int = 64):
        super().__init__()
        self.time_embed = nn.Embedding(n_steps, time_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_embed = self.time_embed(t)
        return self.net(torch.cat([x, t_embed], dim=1))


class _FallbackTabDDM:
    def __init__(
        self,
        n_steps: int = 150,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        hidden_dim: int = 256,
        random_state: int = 42,
    ):
        self.n_steps = int(n_steps)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.hidden_dim = int(hidden_dim)
        self.random_state = int(random_state)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[_Denoiser] = None
        self.state: Optional[_PreprocessState] = None

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

    def fit(self, df: pd.DataFrame) -> "_FallbackTabDDM":
        x, self.state = _fit_transform(df)
        dataset = TensorDataset(torch.from_numpy(x))
        batch_size = max(1, min(self.batch_size, len(dataset)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        self.model = _Denoiser(
            input_dim=self.state.n_total,
            n_steps=self.n_steps,
            hidden_dim=self.hidden_dim,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        betas = torch.linspace(1e-4, 2e-2, self.n_steps, device=self.device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.model.train()
        for _ in range(self.epochs):
            for (x0,) in loader:
                x0 = x0.to(self.device)
                t = torch.randint(0, self.n_steps, (x0.shape[0],), device=self.device)
                noise = torch.randn_like(x0)
                alpha_bar_t = alpha_bar[t].unsqueeze(1)
                xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise
                pred_noise = self.model(xt, t)
                loss = F.mse_loss(pred_noise, noise)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        return self

    @torch.no_grad()
    def sample(self, n: int) -> pd.DataFrame:
        if self.model is None or self.state is None:
            raise RuntimeError("Fallback TabDDM must be fitted before sampling.")

        self.model.eval()
        betas = torch.linspace(1e-4, 2e-2, self.n_steps, device=self.device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        xt = torch.randn((int(n), self.state.n_total), device=self.device)
        for step in reversed(range(self.n_steps)):
            t = torch.full((int(n),), step, device=self.device, dtype=torch.long)
            pred_noise = self.model(xt, t)

            beta_t = betas[step]
            alpha_t = alphas[step]
            alpha_bar_t = alpha_bar[step]
            mean = (1.0 / torch.sqrt(alpha_t)) * (
                xt - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise
            )
            if step > 0:
                xt = mean + torch.sqrt(beta_t) * torch.randn_like(xt)
            else:
                xt = mean

        sample_array = xt.detach().cpu().numpy().astype(np.float32)
        return _inverse_transform(sample_array, self.state)


class TabDDM:
    def __init__(
        self,
        plugin_name: str = "ddpm",
        plugin_kwargs: Optional[Dict] = None,
        fallback_kwargs: Optional[Dict] = None,
    ):
        self.plugin_name = plugin_name
        self.plugin_kwargs = plugin_kwargs or {}
        self.fallback_kwargs = fallback_kwargs or {}
        self.plugin: Any = None
        self.backend = "uninitialized"
        self.fallback_model: Optional[_FallbackTabDDM] = None
        self._plugins_factory: Any = None
        self._dataloader_cls: Any = None

    def _try_load_synthcity(self) -> bool:
        try:
            from synthcity.plugins import Plugins
            from synthcity.plugins.core.dataloader import GenericDataLoader
        except Exception:
            return False
        self._plugins_factory = Plugins
        self._dataloader_cls = GenericDataLoader
        return True

    def fit(self, df: pd.DataFrame) -> "TabDDM":
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Input training data must be a non-empty pandas DataFrame.")

        if self._try_load_synthcity():
            try:
                available = self._plugins_factory().list()
                if self.plugin_name in available:
                    loader = self._dataloader_cls(df)
                    plugins = self._plugins_factory()
                    try:
                        self.plugin = plugins.get(self.plugin_name, **self.plugin_kwargs)
                    except TypeError:
                        self.plugin = plugins.get(self.plugin_name)
                    self.plugin.fit(loader)
                    self.backend = f"synthcity:{self.plugin_name}"
                    return self
            except Exception as exc:
                warnings.warn(
                    f"SynthCity backend failed ({exc!r}); switching to fallback TabDDM implementation.",
                    RuntimeWarning,
                )

        self.fallback_model = _FallbackTabDDM(**self.fallback_kwargs)
        self.fallback_model.fit(df)
        self.backend = "fallback"
        return self

    def sample(self, n: int) -> pd.DataFrame:
        if int(n) <= 0:
            raise ValueError("n must be a positive integer.")

        if self.backend.startswith("synthcity"):
            synthetic = self.plugin.generate(count=int(n))
            if hasattr(synthetic, "dataframe"):
                dataframe_attr = synthetic.dataframe
                return dataframe_attr() if callable(dataframe_attr) else dataframe_attr
            if hasattr(synthetic, "data"):
                return synthetic.data
            raise TypeError(
                "Unexpected SynthCity generate output; expected object with dataframe() or data."
            )

        if self.fallback_model is None:
            raise RuntimeError("Model must be fitted before sampling.")
        return self.fallback_model.sample(int(n))


def generate(train_data, n_generated, output_dir):
    df = pd.read_csv(train_data, header=0)
    model = TabDDM(
        plugin_name="ddpm",
        plugin_kwargs={"n_iter": 1000, "batch_size": 256, "lr": 1e-3},
        fallback_kwargs={"n_steps": 150, "epochs": 100, "batch_size": 256, "lr": 1e-3},
    )
    model.fit(df)
    new_data = model.sample(n_generated)

    float_cols = new_data.select_dtypes(include="float").columns
    new_data[float_cols] = new_data[float_cols].round(3)

    os.makedirs("synthetic_data", exist_ok=True)
    output_path = os.path.join("synthetic_data", f"{output_dir}")
    new_data.to_csv(output_path)
    return new_data


if __name__ == "__main__":
    generate("data/kaggle/ibm_hr.csv", 1500, "kaggle_TabDDM2.csv")
