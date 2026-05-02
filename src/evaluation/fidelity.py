"""
python fidelity_amended.py \
  --real data/real.csv \
  --synthetic data/fake.csv \
  --output eval.txt \
  --output_json eval.json
"""

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from ..date_columns import apply_date_metadata_and_encode, detect_and_encode_date_columns
    from .preprocessing import (
        build_global_utility_preprocessor,
        drop_leaky_columns,
        infer_types_by_numeric_coercion,
        one_hot_encode_aligned,
        remove_id_like_and_constant,
        split_numeric_by_cardinality,
    )
except ImportError:
    from date_columns import apply_date_metadata_and_encode, detect_and_encode_date_columns
    from preprocessing import (
        build_global_utility_preprocessor,
        drop_leaky_columns,
        infer_types_by_numeric_coercion,
        one_hot_encode_aligned,
        remove_id_like_and_constant,
        split_numeric_by_cardinality,
    )


def _trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.ndim != 1 or x.ndim != 1:
        raise ValueError("Trapezoid integral expects 1D inputs.")
    if len(y) != len(x):
        raise ValueError("Trapezoid integral expects x and y of equal length.")
    if len(y) < 2:
        return 0.0
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5))


def evaluate_numerical_features(samples: np.ndarray, data: np.ndarray):
    samples = pd.to_numeric(pd.Series(samples), errors="coerce").to_numpy(dtype=float)
    data = pd.to_numeric(pd.Series(data), errors="coerce").to_numpy(dtype=float)

    samples = samples[np.isfinite(samples)]
    data = data[np.isfinite(data)]
    if len(samples) < 2 or len(data) < 2:
        return None, None, None

    ks_statistic, _ = ks_2samp(samples, data)
    w_distance = wasserstein_distance(samples, data)
    real_iqr = float(np.percentile(data, 75) - np.percentile(data, 25))
    w_distance_normalized_iqr = float(w_distance / real_iqr) if real_iqr > 0 else None
    return float(ks_statistic), float(w_distance), w_distance_normalized_iqr


def evaluate_date_features(samples: np.ndarray, data: np.ndarray):
    samples = pd.to_numeric(pd.Series(samples), errors="coerce").to_numpy(dtype=float)
    data = pd.to_numeric(pd.Series(data), errors="coerce").to_numpy(dtype=float)

    samples = samples[np.isfinite(samples)]
    data = data[np.isfinite(data)]
    if len(samples) < 2 or len(data) < 2:
        return None, None

    real_min = float(np.min(data))
    real_max = float(np.max(data))
    real_range = real_max - real_min

    ks_statistic, _ = ks_2samp(samples, data)

    if real_range > 0:
        samples_norm = (samples - real_min) / real_range
        data_norm = (data - real_min) / real_range
        w_distance = float(wasserstein_distance(samples_norm, data_norm))
    else:
        w_distance = 0.0 if np.allclose(samples, real_min) else None

    return float(ks_statistic), w_distance


def evaluate_categorical_tv(samples: np.ndarray, data: np.ndarray):
    samples = pd.Series(samples, dtype="string").fillna("<NA>").str.strip().to_numpy(dtype=object)
    data = pd.Series(data, dtype="string").fillna("<NA>").str.strip().to_numpy(dtype=object)

    cats = pd.Index(pd.Series(np.concatenate([samples, data], axis=0)).unique()).tolist()
    count_obs = Counter(samples)
    count_exp = Counter(data)

    s = np.array([count_obs[c] for c in cats], dtype=float)
    r = np.array([count_exp[c] for c in cats], dtype=float)
    if s.sum() == 0 or r.sum() == 0:
        return None

    s = s / s.sum()
    r = r / r.sum()
    tv = 0.5 * np.abs(s - r).sum()
    return float(tv)


def _load_with_date_metadata(real_path: str, synth_path: str):
    real_df = drop_leaky_columns(pd.read_csv(real_path))
    synth_df = drop_leaky_columns(pd.read_csv(synth_path))

    real_df, date_metadata = detect_and_encode_date_columns(real_df)
    synth_df = apply_date_metadata_and_encode(synth_df, date_metadata)
    return real_df, synth_df, date_metadata


def _detect_temporal_orderings(real_df: pd.DataFrame, date_cols: list[str], threshold: float = 0.95):
    relationships = []
    for left in date_cols:
        for right in date_cols:
            if left == right:
                continue
            pair = pd.DataFrame({"left": real_df[left], "right": real_df[right]}).dropna()
            if len(pair) == 0:
                continue

            left_before = float((pair["left"] <= pair["right"]).mean())
            if left_before >= threshold:
                relationships.append({"left": left, "right": right, "direction": "<=", "real_fraction": left_before})
    unique_relationships = {}
    for rel in relationships:
        key = (rel["left"], rel["right"], rel["direction"])
        unique_relationships[key] = rel
    return list(unique_relationships.values())


def _temporal_consistency_scores(synth_df: pd.DataFrame, relationships: list[dict]):
    scores = {}
    values = []
    for rel in relationships:
        left = rel["left"]
        right = rel["right"]
        pair = pd.DataFrame({"left": synth_df[left], "right": synth_df[right]}).dropna()
        if len(pair) == 0:
            score = None
            valid_rows = 0
        else:
            score = float((pair["left"] <= pair["right"]).mean())
            valid_rows = int(len(pair))
            values.append(score)
        scores[f"{left} <= {right}"] = {
            "score": score,
            "valid_rows": valid_rows,
            "real_fraction": rel["real_fraction"],
        }
    overall = float(np.mean(values)) if values else None
    return {"overall": overall, "pairs": scores}


class FeedForwardEmbedder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        rep_dim: int = 32,
        num_hidden: int = 64,
        num_layers: int = 3,
        activation: str = "ReLU",
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        act_cls = getattr(nn, activation)

        layers = []
        in_dim = input_dim
        for _ in range(max(1, num_layers - 1)):
            layers.append(nn.Linear(in_dim, num_hidden))
            layers.append(act_cls())
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            in_dim = num_hidden

        layers.append(nn.Linear(in_dim, rep_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Ruff et al. (2018) "Deep One-Class Classification", ICML
class DeepSVDDEmbedder:
    def __init__(
        self,
        input_dim: int,
        rep_dim: int = 32,
        num_hidden: int = 64,
        num_layers: int = 3,
        activation: str = "ReLU",
        dropout_prob: float = 0.0,
        nu: float = 0.01,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        epochs: int = 200,
        warm_up_epochs: int = 10,
        seed: int = 0,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.nu = nu
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.warm_up_epochs = warm_up_epochs
        self.seed = seed

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.model = FeedForwardEmbedder(
            input_dim=input_dim,
            rep_dim=rep_dim,
            num_hidden=num_hidden,
            num_layers=num_layers,
            activation=activation,
            dropout_prob=dropout_prob,
        ).to(self.device)

        self.center_ = None
        self.radius_ = 0.0

    @staticmethod
    def _dist2(z, c):
        return torch.sum((z - c) ** 2, dim=1)

    @staticmethod
    def _get_radius(dist2: np.ndarray, nu: float) -> float:
        return float(np.sqrt(np.quantile(dist2, 1 - nu)))

    def _initialize_center(self, X_tensor: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            z = self.model(X_tensor)
            c = z.mean(dim=0)

        eps = 1e-6
        c[(torch.abs(c) < eps) & (c < 0)] = -eps
        c[(torch.abs(c) < eps) & (c >= 0)] = eps
        return c.detach()

    def _soft_boundary_loss(self, z: torch.Tensor, c: torch.Tensor, R: torch.Tensor):
        dist2 = self._dist2(z, c)
        scores = dist2 - R**2
        loss = R**2 + (1.0 / self.nu) * torch.mean(torch.maximum(torch.zeros_like(scores), scores))
        return loss, dist2

    def fit(self, X_real: np.ndarray):
        X_real = np.asarray(X_real, dtype=np.float32)
        X_tensor = torch.tensor(X_real, dtype=torch.float32, device=self.device)

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.center_ = self._initialize_center(X_tensor)
        R = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        n = X_tensor.shape[0]
        for epoch in range(self.epochs):
            self.model.train()
            perm = torch.randperm(n, device=self.device)
            all_dist2 = []

            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb = X_tensor[idx]

                optimizer.zero_grad()
                z = self.model(xb)
                loss, dist2 = self._soft_boundary_loss(z, self.center_, R)
                loss.backward()
                optimizer.step()

                all_dist2.append(dist2.detach().cpu().numpy())

            all_dist2 = np.concatenate(all_dist2, axis=0)
            if epoch >= self.warm_up_epochs:
                self.radius_ = self._get_radius(all_dist2, self.nu)
                R = torch.tensor(self.radius_, dtype=torch.float32, device=self.device)
            else:
                self.radius_ = float(R.item())

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            z = self.model(X_tensor).cpu().numpy()
        return z

    def fit_transform(self, X_real: np.ndarray) -> np.ndarray:
        self.fit(X_real)
        return self.transform(X_real)


# Alaa et al. (2022) "How Faithful is your Synthetic Data?", ICML
def compute_alpha_precision_curve(z_real: np.ndarray, z_syn: np.ndarray, alphas=None):
    if alphas is None:
        alphas = np.linspace(0.05, 1.0, 20)

    c_r = z_real.mean(axis=0)
    d_real = np.linalg.norm(z_real - c_r, axis=1)
    d_syn = np.linalg.norm(z_syn - c_r, axis=1)

    p_alpha = []
    radii = []
    for alpha in alphas:
        r_alpha = float(np.quantile(d_real, alpha))
        radii.append(r_alpha)
        p_alpha.append(float(np.mean(d_syn <= r_alpha)))

    return np.asarray(alphas), np.asarray(p_alpha), c_r, np.asarray(radii)


def compute_beta_recall_curve(z_real: np.ndarray, z_syn: np.ndarray, betas=None, k=5):
    if betas is None:
        betas = np.linspace(0.05, 1.0, 20)

    c_g = z_syn.mean(axis=0)
    d_syn_from_cg = np.linalg.norm(z_syn - c_g, axis=1)

    nn_real = NearestNeighbors(n_neighbors=min(k + 1, len(z_real)))
    nn_real.fit(z_real)
    dist_real, _ = nn_real.kneighbors(z_real)
    local_idx = min(k, dist_real.shape[1] - 1)
    local_scale = dist_real[:, local_idx]

    r_beta_vals = []
    radii = []
    for beta in betas:
        r_beta = float(np.quantile(d_syn_from_cg, beta))
        radii.append(r_beta)
        z_syn_beta = z_syn[d_syn_from_cg <= r_beta]

        if len(z_syn_beta) == 0:
            r_beta_vals.append(0.0)
            continue

        nn_syn = NearestNeighbors(n_neighbors=1)
        nn_syn.fit(z_syn_beta)
        dist_to_syn, _ = nn_syn.kneighbors(z_real)
        dist_to_syn = dist_to_syn[:, 0]
        r_beta_vals.append(float(np.mean(dist_to_syn <= local_scale)))

    return np.asarray(betas), np.asarray(r_beta_vals), c_g, np.asarray(radii)


def integrated_alpha_precision(alphas: np.ndarray, p_alpha: np.ndarray) -> float:
    delta = _trapezoid_integral(np.abs(p_alpha - alphas), alphas)
    return float(1 - 2 * delta)


def integrated_beta_recall(betas: np.ndarray, r_beta: np.ndarray) -> float:
    delta = _trapezoid_integral(np.abs(r_beta - betas), betas)
    return float(1 - 2 * delta)


def alpha_beta_metrics(
    X_real,
    X_synth,
    k=5,
    rep_dim=32,
    hidden_dim=64,
    num_layers=3,
    activation="ReLU",
    dropout_prob=0.0,
    nu=0.01,
    lr=1e-3,
    weight_decay=1e-5,
    batch_size=256,
    epochs=200,
    warm_up_epochs=10,
    seed=0,
):
    embedder = DeepSVDDEmbedder(
        input_dim=X_real.shape[1],
        rep_dim=min(rep_dim, X_real.shape[1]) if rep_dim is not None else min(32, X_real.shape[1]),
        num_hidden=hidden_dim,
        num_layers=num_layers,
        activation=activation,
        dropout_prob=dropout_prob,
        nu=nu,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        epochs=epochs,
        warm_up_epochs=warm_up_epochs,
        seed=seed,
    )

    z_real = embedder.fit_transform(X_real)
    z_syn = embedder.transform(X_synth)

    alphas, p_alpha, c_r, alpha_radii = compute_alpha_precision_curve(z_real, z_syn)
    betas, r_beta, c_g, beta_radii = compute_beta_recall_curve(z_real, z_syn, k=k)

    return {
        "alpha_precision": float(p_alpha[-1]),
        "beta_recall": float(r_beta[-1]),
        "IP_alpha": integrated_alpha_precision(alphas, p_alpha),
        "IR_beta": integrated_beta_recall(betas, r_beta),
        "alphas": alphas.tolist(),
        "alpha_precision_curve": p_alpha.tolist(),
        "alpha_radii": alpha_radii.tolist(),
        "betas": betas.tolist(),
        "beta_recall_curve": r_beta.tolist(),
        "beta_radii": beta_radii.tolist(),
        "embedder": {
            "rep_dim": int(z_real.shape[1]),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "activation": activation,
            "dropout_prob": float(dropout_prob),
            "nu": float(nu),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "warm_up_epochs": int(warm_up_epochs),
            "radius": float(embedder.radius_),
            "k_recall": int(k),
        },
    }


def discriminator_eval_with_permutation(X_real, X_synth, test_size=0.3, seed=0, permutations=50):
    X = np.vstack([X_synth, X_real]).astype(float)
    y = np.concatenate([np.zeros(len(X_synth)), np.ones(len(X_real))]).astype(int)

    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=5000, solver="saga", random_state=seed)
    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]

    auc = float(roc_auc_score(y_test, p))
    acc = float(accuracy_score(y_test, (p >= 0.5).astype(int)))

    c = float(y_test.mean())
    pmse = float(np.mean((p - c) ** 2))

    rng = np.random.default_rng(seed)
    pmse_null = []
    for _ in range(permutations):
        y_perm = rng.permutation(y_train)
        m = LogisticRegression(max_iter=5000, solver="saga", random_state=seed)
        m.fit(X_train, y_perm)
        p_perm = m.predict_proba(X_test)[:, 1]
        pmse_null.append(float(np.mean((p_perm - c) ** 2)))

    null_mean = float(np.mean(pmse_null))
    ratio = float(pmse / null_mean) if null_mean > 0 else float("inf")
    percentile = float((np.sum(np.array(pmse_null) <= pmse) / len(pmse_null)) * 100.0)

    return {
        "auc": auc,
        "accuracy": acc,
        "pmse_test": pmse,
        "pmse_null_mean": null_mean,
        "pmse_ratio_perm": ratio,
        "pmse_null_percentile": percentile,
        "permutations": permutations,
        "test_size": test_size,
    }


def _best_balanced_accuracy(Xtr, ytr, Xte, yte, preprocessor, seed=0):
    ytr_series = pd.Series(ytr)
    classes = ytr_series.dropna().unique()
    if len(classes) < 2:
        return None

    models = [
        LogisticRegression(max_iter=5000, solver="saga", random_state=seed),
        HistGradientBoostingClassifier(random_state=seed),
    ]

    best = -np.inf
    any_fit = False
    for m in models:
        try:
            pipe = Pipeline([("prep", preprocessor), ("model", m)])
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            score = balanced_accuracy_score(yte, pred)
            best = max(best, score)
            any_fit = True
        except (TypeError, ValueError):
            continue

    return float(best) if any_fit else None


def _best_rmse(Xtr, ytr, Xte, yte, preprocessor, seed=0):
    models = [
        Ridge(random_state=seed),
        HistGradientBoostingRegressor(random_state=seed),
    ]
    best = np.inf
    any_fit = False
    for m in models:
        try:
            pipe = Pipeline([("prep", preprocessor), ("model", m)])
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            rmse = np.sqrt(mean_squared_error(yte, pred))
            best = min(best, rmse)
            any_fit = True
        except (TypeError, ValueError):
            continue
    return float(best) if any_fit else None


def compute_global_utility_metric(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    categorical_cols,
    numeric_cols,
    test_size: float = 0.2,
    seed: int = 0,
    synth_train_cap: int | None = 5000,
    min_real_train_rows: int = 200,
    min_real_test_rows: int = 100,
    min_synth_train_rows: int = 200,
):
    cols = [c for c in real_df.columns if c in synth_df.columns]
    real_df = real_df[cols].copy()
    synth_df = synth_df[cols].copy()

    Dref, Dtest = train_test_split(real_df, test_size=test_size, random_state=seed)

    per_col = {}
    utilities = []
    skipped = {"too_few_rows": 0, "single_class_or_failed_fit": 0}

    for target in cols:
        feature_cols = [c for c in cols if c != target]
        if not feature_cols:
            continue

        Dref_t = Dref.dropna(subset=[target] + feature_cols)
        Dtest_t = Dtest.dropna(subset=[target] + feature_cols)
        synth_t = synth_df.dropna(subset=[target] + feature_cols)

        if synth_train_cap is not None and len(synth_t) > synth_train_cap:
            synth_t = synth_t.sample(n=synth_train_cap, random_state=seed)

        if (
            len(Dref_t) < min_real_train_rows
            or len(Dtest_t) < min_real_test_rows
            or len(synth_t) < min_synth_train_rows
        ):
            skipped["too_few_rows"] += 1
            continue

        X_ref_tr, y_ref_tr = Dref_t[feature_cols], Dref_t[target]
        X_syn_tr, y_syn_tr = synth_t[feature_cols], synth_t[target]
        X_te, y_te = Dtest_t[feature_cols], Dtest_t[target]

        preprocessor = build_global_utility_preprocessor(
            feature_cols=feature_cols,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            real_df=real_df,
            synth_df=synth_df,
        )

        is_cat = target in (categorical_cols or [])
        if is_cat:
            perf_ref = _best_balanced_accuracy(X_ref_tr, y_ref_tr, X_te, y_te, preprocessor, seed=seed)
            perf_syn = _best_balanced_accuracy(X_syn_tr, y_syn_tr, X_te, y_te, preprocessor, seed=seed)

            if perf_ref is None or perf_syn is None or perf_ref <= 0:
                skipped["single_class_or_failed_fit"] += 1
                continue

            util = perf_syn / perf_ref
        else:
            y_ref_tr = pd.to_numeric(pd.Series(y_ref_tr), errors="coerce")
            y_syn_tr = pd.to_numeric(pd.Series(y_syn_tr), errors="coerce")
            y_te = pd.to_numeric(pd.Series(y_te), errors="coerce")

            ok_ref = y_ref_tr.notna()
            ok_syn = y_syn_tr.notna()
            ok_te = y_te.notna()

            X_ref_tr, y_ref_tr = X_ref_tr.loc[ok_ref], y_ref_tr.loc[ok_ref]
            X_syn_tr, y_syn_tr = X_syn_tr.loc[ok_syn], y_syn_tr.loc[ok_syn]
            X_te, y_te = X_te.loc[ok_te], y_te.loc[ok_te]

            if (
                len(X_ref_tr) < min_real_train_rows
                or len(X_syn_tr) < min_synth_train_rows
                or len(X_te) < min_real_test_rows
            ):
                skipped["too_few_rows"] += 1
                continue

            perf_ref = _best_rmse(X_ref_tr, y_ref_tr, X_te, y_te, preprocessor, seed=seed)
            perf_syn = _best_rmse(X_syn_tr, y_syn_tr, X_te, y_te, preprocessor, seed=seed)

            if perf_ref is None or perf_syn is None or perf_syn <= 0:
                skipped["single_class_or_failed_fit"] += 1
                continue

            util = perf_ref / perf_syn

        per_col[target] = {
            "type": "categorical" if is_cat else "numerical",
            "perf_ref": float(perf_ref),
            "perf_synth": float(perf_syn),
            "utility": float(util),
        }
        utilities.append(float(util))

    global_utility = float(np.mean(utilities)) if utilities else None

    return {
        "global_utility": global_utility,
        "num_columns_used": int(len(utilities)),
        "per_column": per_col,
        "skipped": skipped,
        "settings": {
            "test_size": test_size,
            "seed": seed,
            "synth_train_cap": synth_train_cap,
            "min_real_train_rows": int(min_real_train_rows),
            "min_real_test_rows": int(min_real_test_rows),
            "min_synth_train_rows": int(min_synth_train_rows),
        },
    }


@dataclass
class EvalConfig:
    numeric_threshold: float = 0.95
    k: int = 5
    rep_dim: int = 32
    hidden_dim: int = 64
    num_layers: int = 3
    activation: str = "ReLU"
    dropout_prob: float = 0.0
    nu: float = 0.01
    svdd_lr: float = 1e-3
    svdd_weight_decay: float = 1e-5
    svdd_batch_size: int = 256
    svdd_epochs: int = 200
    svdd_warm_up_epochs: int = 10
    seed: int = 0
    permutations: int = 50
    id_unique_threshold: float = 0.98
    global_utility_test_size: float = 0.2
    synth_train_cap: int | None = 5000
    global_utility_min_real_train_rows: int = 200
    global_utility_min_real_test_rows: int = 100
    global_utility_min_synth_train_rows: int = 200


def run_evaluation(
    real_path,
    synth_path,
    config: EvalConfig | None = None,
    *,
    categorical_cols=None,
    numeric_cols=None,
):
    if config is None:
        config = EvalConfig()

    real_df, synth_df, date_metadata = _load_with_date_metadata(real_path, synth_path)
    date_cols = [column for column in date_metadata if column in real_df.columns and column in synth_df.columns]

    real_df, synth_df, categorical_cols, numeric_cols = infer_types_by_numeric_coercion(
        real_df,
        synth_df,
        numeric_threshold=config.numeric_threshold,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
    )
    if date_cols:
        numeric_cols = list(dict.fromkeys(list(numeric_cols) + [column for column in date_cols if column in real_df.columns]))
        categorical_cols = [column for column in categorical_cols if column not in date_cols]

    real_df, synth_df, numeric_cols, categorical_cols, dropped = remove_id_like_and_constant(
        real_df,
        synth_df,
        numeric_cols,
        categorical_cols,
        id_unique_threshold=config.id_unique_threshold,
        protected_cols=date_cols,
    )
    discrete_ordinal_cols, continuous_numeric_cols = split_numeric_by_cardinality(
        real_df,
        synth_df,
        numeric_cols=numeric_cols,
    )
    date_cols = [column for column in date_cols if column in numeric_cols]
    continuous_numeric_cols = [column for column in continuous_numeric_cols if column not in date_cols]
    discrete_ordinal_cols = [column for column in discrete_ordinal_cols if column not in date_cols]
    temporal_orderings = _detect_temporal_orderings(real_df, date_cols)

    results = {
        "paths": {"real": real_path, "synthetic": synth_path},
        "columns": {
            "numeric": continuous_numeric_cols,
            "categorical": categorical_cols,
            "discrete_ordinal": discrete_ordinal_cols,
            "date": date_cols,
        },
        "dropped_columns": dropped,
        "encoding": {"one_hot_feature_count": None},
        "marginal_numeric": {},
        "marginal_categorical_tv": {},
        "marginal_discrete_ordinal": {},
        "marginal_date": {},
        "temporal_consistency": {},
        "support": {},
        "discriminator": {},
        "global_utility": {},
        "settings": {
            "numeric_threshold": config.numeric_threshold,
            "id_unique_threshold": config.id_unique_threshold,
            "k": config.k,
            "rep_dim": config.rep_dim,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "activation": config.activation,
            "dropout_prob": config.dropout_prob,
            "nu": config.nu,
            "svdd_lr": config.svdd_lr,
            "svdd_weight_decay": config.svdd_weight_decay,
            "svdd_batch_size": config.svdd_batch_size,
            "svdd_epochs": config.svdd_epochs,
            "svdd_warm_up_epochs": config.svdd_warm_up_epochs,
            "seed": config.seed,
            "permutations": config.permutations,
            "global_utility_test_size": config.global_utility_test_size,
            "synth_train_cap": config.synth_train_cap,
            "global_utility_min_real_train_rows": config.global_utility_min_real_train_rows,
            "global_utility_min_real_test_rows": config.global_utility_min_real_test_rows,
            "global_utility_min_synth_train_rows": config.global_utility_min_synth_train_rows,
            "date_column_metadata": date_metadata,
            "temporal_orderings": temporal_orderings,
        },
    }

    for c in continuous_numeric_cols:
        ks, w, w_norm_iqr = evaluate_numerical_features(synth_df[c].values, real_df[c].values)
        results["marginal_numeric"][c] = {
            "ks": ks,
            "wasserstein": w,
            "wasserstein_normalized_iqr": w_norm_iqr,
        }

    for c in discrete_ordinal_cols:
        ks, w, w_norm_iqr = evaluate_numerical_features(synth_df[c].values, real_df[c].values)
        tv = evaluate_categorical_tv(synth_df[c].values, real_df[c].values)
        results["marginal_discrete_ordinal"][c] = {
            "ks": ks,
            "wasserstein": w,
            "wasserstein_normalized_iqr": w_norm_iqr,
            "tv": tv,
        }

    for c in date_cols:
        ks, w_norm = evaluate_date_features(synth_df[c].values, real_df[c].values)
        results["marginal_date"][c] = {
            "ks": ks,
            "wasserstein_normalized": w_norm,
        }

    for c in categorical_cols:
        tv = evaluate_categorical_tv(synth_df[c].values, real_df[c].values)
        results["marginal_categorical_tv"][c] = tv

    results["temporal_consistency"] = _temporal_consistency_scores(synth_df, temporal_orderings)

    X_real, X_synth, feature_names = one_hot_encode_aligned(real_df, synth_df, categorical_cols, numeric_cols)
    results["encoding"]["one_hot_feature_count"] = int(len(feature_names))

    results["discriminator"] = discriminator_eval_with_permutation(
        X_real=X_real, X_synth=X_synth, seed=config.seed, permutations=config.permutations
    )

    results["support"] = alpha_beta_metrics(
        X_real=X_real,
        X_synth=X_synth,
        k=config.k,
        rep_dim=config.rep_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        activation=config.activation,
        dropout_prob=config.dropout_prob,
        nu=config.nu,
        lr=config.svdd_lr,
        weight_decay=config.svdd_weight_decay,
        batch_size=config.svdd_batch_size,
        epochs=config.svdd_epochs,
        warm_up_epochs=config.svdd_warm_up_epochs,
        seed=config.seed,
    )

    results["global_utility"] = compute_global_utility_metric(
        real_df=real_df,
        synth_df=synth_df,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        test_size=config.global_utility_test_size,
        seed=config.seed,
        synth_train_cap=config.synth_train_cap,
        min_real_train_rows=config.global_utility_min_real_train_rows,
        min_real_test_rows=config.global_utility_min_real_test_rows,
        min_synth_train_rows=config.global_utility_min_synth_train_rows,
    )

    return results


def _is_finite_numeric(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    ) and math.isfinite(float(value))


def _aggregate_values(values):
    present = [value for value in values if value is not None]
    if not present:
        return None

    if all(isinstance(value, dict) for value in present):
        keys = list(dict.fromkeys(key for value in present for key in value.keys()))
        return {key: _aggregate_values([value.get(key) if value is not None else None for value in values]) for key in keys}

    if all(isinstance(value, list) for value in present):
        lengths = {len(value) for value in present}
        if len(lengths) == 1:
            length = lengths.pop()
            return [_aggregate_values([value[idx] if value is not None else None for value in values]) for idx in range(length)]
        return present[0]

    if all(_is_finite_numeric(value) for value in present):
        return float(np.mean([float(value) for value in present]))

    if all(value == present[0] for value in present):
        return present[0]

    return present[0]


def _std_values(values):
    present = [value for value in values if value is not None]
    if not present:
        return None

    if all(isinstance(value, dict) for value in present):
        keys = list(dict.fromkeys(key for value in present for key in value.keys()))
        return {
            key: _std_values([value.get(key) if value is not None else None for value in values])
            for key in keys
        }

    if all(isinstance(value, list) for value in present):
        lengths = {len(value) for value in present}
        if len(lengths) == 1:
            length = lengths.pop()
            return [
                _std_values([value[idx] if value is not None else None for value in values])
                for idx in range(length)
            ]
        return None

    if all(_is_finite_numeric(value) for value in present):
        return float(np.std([float(value) for value in present], ddof=0))

    return None


def aggregate_evaluation_results(results_list: list[dict], seeds: list[int] | None = None) -> dict:
    if not results_list:
        raise ValueError("results_list must contain at least one evaluation result.")

    aggregated = _aggregate_values(results_list)
    std_tree = _std_values(results_list)
    real_paths = [result.get("paths", {}).get("real") for result in results_list]
    synthetic_paths = [result.get("paths", {}).get("synthetic") for result in results_list]

    aggregated.setdefault("paths", {})
    aggregated["paths"]["real"] = real_paths[0]
    aggregated["paths"]["synthetic"] = f"{len(results_list)} synthetic runs averaged"
    aggregated["multi_run"] = {
        "num_runs": len(results_list),
        "seeds": list(seeds or []),
        "real_paths": real_paths,
        "synthetic_paths": synthetic_paths,
        "std": std_tree,
    }
    if "settings" in aggregated:
        aggregated["settings"]["seed"] = list(seeds or [])

    return aggregated


def _get_nested(mapping: dict | None, *keys):
    current = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _format_stat(mean, std=None, *, precision: int = 6, suffix: str = "") -> str:
    if mean is None:
        return "NA"
    if _is_finite_numeric(mean):
        mean_text = f"{float(mean):.{precision}f}"
        if _is_finite_numeric(std):
            mean_text = f"{mean_text} +/- {float(std):.{precision}f}"
        return f"{mean_text}{suffix}"
    return f"{mean}{suffix}"


def format_results(results: dict) -> str:
    lines = []
    lines.append("=== Evaluation Summary ===")
    metadata = results.get("metadata", {})
    if metadata.get("model_name"):
        lines.append(f"Model:     {metadata['model_name']}")
    if metadata.get("dataset_name"):
        lines.append(f"Dataset:   {metadata['dataset_name']}")
    lines.append(f"Real:      {results['paths']['real']}")
    lines.append(f"Synthetic: {results['paths']['synthetic']}")
    multi_run = results.get("multi_run")
    if multi_run:
        lines.append(f"Runs:      {multi_run.get('num_runs')} (seeds={multi_run.get('seeds', [])})")
    lines.append("")
    lines.append("Columns:")
    lines.append(f"  Numeric:     {results['columns']['numeric']}")
    lines.append(f"  Categorical: {results['columns']['categorical']}")
    lines.append(f"  Discrete/Ordinal: {results['columns'].get('discrete_ordinal', [])}")
    lines.append(f"  Date:        {results['columns'].get('date', [])}")
    lines.append(f"  One-hot features: {results['encoding']['one_hot_feature_count']}")
    if results.get("dropped_columns"):
        lines.append(f"  Dropped: {results['dropped_columns']}")
    lines.append("")

    d = results["discriminator"]
    d_std = _get_nested(results.get("multi_run"), "std", "discriminator") or {}
    lines.append("Discriminator (logistic regression, held-out test + permutation null):")
    lines.append(f"  AUC:               {_format_stat(d.get('auc'), d_std.get('auc'))}")
    lines.append(f"  Accuracy:          {_format_stat(d.get('accuracy'), d_std.get('accuracy'))}")
    lines.append(f"  pMSE (test):       {_format_stat(d.get('pmse_test'), d_std.get('pmse_test'))}")
    lines.append(f"  pMSE null mean:    {_format_stat(d.get('pmse_null_mean'), d_std.get('pmse_null_mean'))}")
    lines.append(f"  pMSE ratio (perm): {_format_stat(d.get('pmse_ratio_perm'), d_std.get('pmse_ratio_perm'))}")
    lines.append(
        f"  Null percentile:   {_format_stat(d.get('pmse_null_percentile'), d_std.get('pmse_null_percentile'), precision=1, suffix='th')}"
    )
    lines.append("")

    s = results["support"]
    s_std = _get_nested(results.get("multi_run"), "std", "support") or {}
    lines.append("Support / coverage (paper-style one-class embedding):")
    lines.append(
        f"  Alpha precision (alpha=1.0): {_format_stat(s.get('alpha_precision'), s_std.get('alpha_precision'))}"
    )
    lines.append(
        f"  Beta recall (beta=1.0):      {_format_stat(s.get('beta_recall'), s_std.get('beta_recall'))}"
    )
    lines.append(f"  Integrated IP_alpha:         {_format_stat(s.get('IP_alpha'), s_std.get('IP_alpha'))}")
    lines.append(f"  Integrated IR_beta:          {_format_stat(s.get('IR_beta'), s_std.get('IR_beta'))}")
    lines.append(
        f"  Embedder radius:             {_format_stat(_get_nested(s, 'embedder', 'radius'), _get_nested(s_std, 'embedder', 'radius'))}"
    )
    lines.append(f"  Alpha grid:                  {s['alphas']}")
    lines.append(f"  P_alpha curve:               {s['alpha_precision_curve']}")
    lines.append(f"  Beta grid:                   {s['betas']}")
    lines.append(f"  R_beta curve:                {s['beta_recall_curve']}")
    lines.append("")

    gu = results.get("global_utility", {})
    gu_std = _get_nested(results.get("multi_run"), "std", "global_utility") or {}
    lines.append("Global Utility (TabStruct-style):")
    lines.append(f"  Global utility: {_format_stat(gu.get('global_utility'), gu_std.get('global_utility'))}")
    lines.append(f"  Columns used:   {_format_stat(gu.get('num_columns_used'), gu_std.get('num_columns_used'))}")
    lines.append(f"  Synth train cap:{gu.get('settings', {}).get('synth_train_cap')}")
    if gu.get("skipped"):
        skipped_std = gu_std.get("skipped", {}) if isinstance(gu_std, dict) else {}
        lines.append(
            f"  Skipped (too few rows): {_format_stat(gu['skipped'].get('too_few_rows', 0), skipped_std.get('too_few_rows'), precision=2)}"
        )
        lines.append(
            "  Skipped (single-class/fit fail): "
            f"{_format_stat(gu['skipped'].get('single_class_or_failed_fit', 0), skipped_std.get('single_class_or_failed_fit'), precision=2)}"
        )

    per_col = gu.get("per_column", {})
    if per_col:
        vals = [(k, v.get("utility")) for k, v in per_col.items() if v.get("utility") is not None]
        vals_sorted = sorted(vals, key=lambda x: x[1])
        if vals_sorted:
            lines.append("  Per-column utility (lowest 5):")
            for k, u in vals_sorted[:5]:
                lines.append(
                    f"    {k}: {_format_stat(u, _get_nested(gu_std, 'per_column', k, 'utility'))}"
                )
            lines.append("  Per-column utility (highest 5):")
            for k, u in vals_sorted[-5:][::-1]:
                lines.append(
                    f"    {k}: {_format_stat(u, _get_nested(gu_std, 'per_column', k, 'utility'))}"
                )
    lines.append("")

    lines.append("Marginal (numeric):")
    marginal_numeric_std = _get_nested(results.get("multi_run"), "std", "marginal_numeric") or {}
    for col, m in results["marginal_numeric"].items():
        lines.append(
            f"  {col}: KS={_format_stat(m.get('ks'), _get_nested(marginal_numeric_std, col, 'ks'))}, "
            f"Wasserstein={_format_stat(m.get('wasserstein'), _get_nested(marginal_numeric_std, col, 'wasserstein'))}, "
            "Wasserstein/IQR="
            f"{_format_stat(m.get('wasserstein_normalized_iqr'), _get_nested(marginal_numeric_std, col, 'wasserstein_normalized_iqr'))}"
        )
    lines.append("")
    lines.append("Marginal (discrete/ordinal):")
    marginal_discrete_std = _get_nested(results.get("multi_run"), "std", "marginal_discrete_ordinal") or {}
    for col, m in results.get("marginal_discrete_ordinal", {}).items():
        lines.append(
            f"  {col}: KS={_format_stat(m.get('ks'), _get_nested(marginal_discrete_std, col, 'ks'))}, "
            f"Wasserstein={_format_stat(m.get('wasserstein'), _get_nested(marginal_discrete_std, col, 'wasserstein'))}, "
            "Wasserstein/IQR="
            f"{_format_stat(m.get('wasserstein_normalized_iqr'), _get_nested(marginal_discrete_std, col, 'wasserstein_normalized_iqr'))}, "
            f"TV={_format_stat(m.get('tv'), _get_nested(marginal_discrete_std, col, 'tv'))}"
        )
    lines.append("")
    lines.append("Marginal (date as Unix timestamps):")
    marginal_date_std = _get_nested(results.get("multi_run"), "std", "marginal_date") or {}
    for col, m in results.get("marginal_date", {}).items():
        lines.append(
            f"  {col}: KS={_format_stat(m.get('ks'), _get_nested(marginal_date_std, col, 'ks'))}, "
            "Wasserstein[0,1]="
            f"{_format_stat(m.get('wasserstein_normalized'), _get_nested(marginal_date_std, col, 'wasserstein_normalized'))}"
        )
    lines.append("")
    lines.append("Temporal consistency:")
    tc = results.get("temporal_consistency", {})
    tc_std = _get_nested(results.get("multi_run"), "std", "temporal_consistency") or {}
    lines.append(f"  Overall: {_format_stat(tc.get('overall'), tc_std.get('overall'))}")
    for pair_name, values in tc.get("pairs", {}).items():
        lines.append(
            f"  {pair_name}: score={_format_stat(values.get('score'), _get_nested(tc_std, 'pairs', pair_name, 'score'))}, "
            f"valid_rows={values.get('valid_rows')}, real_fraction={_format_stat(values.get('real_fraction'))}"
        )
    lines.append("")
    lines.append("Marginal (categorical TV distance):")
    marginal_cat_std = _get_nested(results.get("multi_run"), "std", "marginal_categorical_tv") or {}
    for col, tv in results["marginal_categorical_tv"].items():
        lines.append(f"  {col}: TV={_format_stat(tv, marginal_cat_std.get(col))}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Synthetic-vs-real evaluation with paper-style alpha/beta metrics.")
    parser.add_argument("--real", required=True, help="Path to real.csv")
    parser.add_argument("--synthetic", required=True, help="Path to synthetic.csv")
    parser.add_argument("--categorical", nargs="*", default=None, help="Categorical columns (optional).")
    parser.add_argument("--numeric", nargs="*", default=None, help="Numeric columns (optional).")
    parser.add_argument(
        "--numeric_threshold",
        type=float,
        default=0.95,
        help="Numeric coercion threshold in BOTH datasets (default 0.95).",
    )
    parser.add_argument("--k", type=int, default=5, help="k for beta-recall local coverage.")
    parser.add_argument("--rep_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--activation", type=str, default="ReLU")
    parser.add_argument("--dropout_prob", type=float, default=0.0)
    parser.add_argument("--nu", type=float, default=0.01)
    parser.add_argument("--svdd_lr", type=float, default=1e-3)
    parser.add_argument("--svdd_weight_decay", type=float, default=1e-5)
    parser.add_argument("--svdd_batch_size", type=int, default=256)
    parser.add_argument("--svdd_epochs", type=int, default=200)
    parser.add_argument("--svdd_warm_up_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--permutations", type=int, default=50)
    parser.add_argument("--id_unique_threshold", type=float, default=0.98)
    parser.add_argument("--global_utility_test_size", type=float, default=0.2)
    parser.add_argument("--global_utility_min_real_train_rows", type=int, default=200)
    parser.add_argument("--global_utility_min_real_test_rows", type=int, default=100)
    parser.add_argument("--global_utility_min_synth_train_rows", type=int, default=200)
    parser.add_argument(
        "--synth_train_cap",
        type=int,
        default=5000,
        help="Cap synthetic rows used per target for global utility (speed). Use 0 or negative to disable.",
    )
    parser.add_argument("--output", default="evaluation.txt")
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    synth_cap = None if args.synth_train_cap is None or args.synth_train_cap <= 0 else args.synth_train_cap

    config = EvalConfig(
        numeric_threshold=args.numeric_threshold,
        k=args.k,
        rep_dim=args.rep_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        activation=args.activation,
        dropout_prob=args.dropout_prob,
        nu=args.nu,
        svdd_lr=args.svdd_lr,
        svdd_weight_decay=args.svdd_weight_decay,
        svdd_batch_size=args.svdd_batch_size,
        svdd_epochs=args.svdd_epochs,
        svdd_warm_up_epochs=args.svdd_warm_up_epochs,
        seed=args.seed,
        permutations=args.permutations,
        id_unique_threshold=args.id_unique_threshold,
        global_utility_test_size=args.global_utility_test_size,
        synth_train_cap=synth_cap,
        global_utility_min_real_train_rows=args.global_utility_min_real_train_rows,
        global_utility_min_real_test_rows=args.global_utility_min_real_test_rows,
        global_utility_min_synth_train_rows=args.global_utility_min_synth_train_rows,
    )
    results = run_evaluation(
        args.real,
        args.synthetic,
        config,
        categorical_cols=args.categorical,
        numeric_cols=args.numeric,
    )

    out = format_results(results)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(out)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved: {args.output}" + (f" and {args.output_json}" if args.output_json else ""))


if __name__ == "__main__":
    main()
