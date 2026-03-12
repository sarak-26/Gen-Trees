"""
python fidelity_amended.py \
  --real data/real.csv \
  --synthetic data/fake.csv \
  --output eval.txt \
  --output_json eval.json
"""

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import ks_2samp, wasserstein_distance

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------
# Marginal distribution
# -----------------------
def evaluate_numerical_features(samples: np.ndarray, data: np.ndarray):
    samples = pd.to_numeric(pd.Series(samples), errors="coerce").to_numpy(dtype=float)
    data = pd.to_numeric(pd.Series(data), errors="coerce").to_numpy(dtype=float)

    samples = samples[np.isfinite(samples)]
    data = data[np.isfinite(data)]
    if len(samples) < 2 or len(data) < 2:
        return None, None

    ks_statistic, _ = ks_2samp(samples, data)
    w_distance = wasserstein_distance(samples, data)
    return float(ks_statistic), float(w_distance)


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


# -----------------------
# Paper-faithful embedding + alpha/beta metrics
# -----------------------
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
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.nu = nu
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.warm_up_epochs = warm_up_epochs

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
    delta = np.trapz(np.abs(p_alpha - alphas), alphas)
    return float(1 - 2 * delta)


def integrated_beta_recall(betas: np.ndarray, r_beta: np.ndarray) -> float:
    delta = np.trapz(np.abs(r_beta - betas), betas)
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


# -----------------------
# Cleaning & Encoding
# -----------------------
def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop_cols = [c for c in df.columns if c.strip().lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    return df


def strip_object_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype("string").str.strip()
    return df


def infer_types_by_numeric_coercion(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    numeric_threshold: float = 0.95,
    categorical_cols=None,
    numeric_cols=None,
):
    shared = sorted(set(real_df.columns).intersection(set(synth_df.columns)))
    if not shared:
        raise ValueError("real.csv and synthetic.csv share no columns.")
    real_df = real_df[shared].copy()
    synth_df = synth_df[shared].copy()

    real_df = strip_object_whitespace(real_df)
    synth_df = strip_object_whitespace(synth_df)

    if categorical_cols is not None or numeric_cols is not None:
        if categorical_cols is None:
            categorical_cols = [c for c in shared if c not in (numeric_cols or [])]
        if numeric_cols is None:
            numeric_cols = [c for c in shared if c not in (categorical_cols or [])]
        return real_df, synth_df, categorical_cols, numeric_cols

    categorical_cols = []
    numeric_cols = []

    for c in shared:
        r_num = pd.to_numeric(real_df[c], errors="coerce")
        s_num = pd.to_numeric(synth_df[c], errors="coerce")

        r_ok = float(r_num.notna().mean())
        s_ok = float(s_num.notna().mean())

        if r_ok >= numeric_threshold and s_ok >= numeric_threshold:
            numeric_cols.append(c)
            real_df[c] = r_num
            synth_df[c] = s_num
        else:
            categorical_cols.append(c)

    return real_df, synth_df, categorical_cols, numeric_cols


def remove_id_like_and_constant(real_df, synth_df, numeric_cols, categorical_cols, id_unique_threshold=0.98):
    to_drop = []

    def nunique_ratio(s: pd.Series) -> float:
        s2 = s.dropna()
        if len(s2) == 0:
            return 0.0
        return float(s2.nunique()) / float(len(s2))

    for c in numeric_cols + categorical_cols:
        r = nunique_ratio(real_df[c])
        s = nunique_ratio(synth_df[c])

        if r >= id_unique_threshold and s >= id_unique_threshold:
            to_drop.append(c)
            continue
        if real_df[c].dropna().nunique() <= 1 and synth_df[c].dropna().nunique() <= 1:
            to_drop.append(c)

    if to_drop:
        real_df = real_df.drop(columns=to_drop, errors="ignore")
        synth_df = synth_df.drop(columns=to_drop, errors="ignore")
        numeric_cols = [c for c in numeric_cols if c not in to_drop]
        categorical_cols = [c for c in categorical_cols if c not in to_drop]

    return real_df, synth_df, numeric_cols, categorical_cols, to_drop


def one_hot_encode_aligned(real_df, synth_df, categorical_cols, numeric_cols):
    combined = pd.concat(
        [synth_df[categorical_cols + numeric_cols], real_df[categorical_cols + numeric_cols]],
        axis=0,
        ignore_index=True,
    )
    X = pd.get_dummies(combined, columns=categorical_cols, dummy_na=True)

    X_synth = X.iloc[: len(synth_df)].to_numpy(dtype=float)
    X_real = X.iloc[len(synth_df):].to_numpy(dtype=float)
    return X_real, X_synth, X.columns.tolist()


# -----------------------
# Discriminator metrics (train/test + permutation null)
# -----------------------
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


# -----------------------
# Global Utility (TabStruct-style) -- robust to single-class targets
# -----------------------
def _build_global_utility_preprocessor(feature_cols, categorical_cols, numeric_cols):
    cat = [c for c in feature_cols if c in (categorical_cols or [])]
    num = [c for c in feature_cols if c in (numeric_cols or [])]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", StandardScaler(with_mean=True, with_std=True), num),
        ],
        remainder="drop",
    )


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
        except ValueError:
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
        except ValueError:
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

        if len(Dref_t) < 200 or len(Dtest_t) < 200 or len(synth_t) < 200:
            skipped["too_few_rows"] += 1
            continue

        X_ref_tr, y_ref_tr = Dref_t[feature_cols], Dref_t[target]
        X_syn_tr, y_syn_tr = synth_t[feature_cols], synth_t[target]
        X_te, y_te = Dtest_t[feature_cols], Dtest_t[target]

        preprocessor = _build_global_utility_preprocessor(
            feature_cols=feature_cols,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
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

            if len(X_ref_tr) < 200 or len(X_syn_tr) < 200 or len(X_te) < 200:
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
        "settings": {"test_size": test_size, "seed": seed, "synth_train_cap": synth_train_cap},
    }


# -----------------------
# Full evaluation
# -----------------------
def run_evaluation(
    real_path,
    synth_path,
    categorical_cols=None,
    numeric_cols=None,
    numeric_threshold=0.95,
    k=5,
    rep_dim=32,
    hidden_dim=64,
    num_layers=3,
    activation="ReLU",
    dropout_prob=0.0,
    nu=0.01,
    svdd_lr=1e-3,
    svdd_weight_decay=1e-5,
    svdd_batch_size=256,
    svdd_epochs=200,
    svdd_warm_up_epochs=10,
    seed=0,
    permutations=50,
    id_unique_threshold=0.98,
    global_utility_test_size=0.2,
    synth_train_cap=5000,
):
    real_df = drop_leaky_columns(pd.read_csv(real_path))
    synth_df = drop_leaky_columns(pd.read_csv(synth_path))

    real_df, synth_df, categorical_cols, numeric_cols = infer_types_by_numeric_coercion(
        real_df,
        synth_df,
        numeric_threshold=numeric_threshold,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
    )

    real_df, synth_df, numeric_cols, categorical_cols, dropped = remove_id_like_and_constant(
        real_df, synth_df, numeric_cols, categorical_cols, id_unique_threshold=id_unique_threshold
    )

    results = {
        "paths": {"real": real_path, "synthetic": synth_path},
        "columns": {"numeric": numeric_cols, "categorical": categorical_cols},
        "dropped_columns": dropped,
        "encoding": {"one_hot_feature_count": None},
        "marginal_numeric": {},
        "marginal_categorical_tv": {},
        "support": {},
        "discriminator": {},
        "global_utility": {},
        "settings": {
            "numeric_threshold": numeric_threshold,
            "id_unique_threshold": id_unique_threshold,
            "k": k,
            "rep_dim": rep_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "activation": activation,
            "dropout_prob": dropout_prob,
            "nu": nu,
            "svdd_lr": svdd_lr,
            "svdd_weight_decay": svdd_weight_decay,
            "svdd_batch_size": svdd_batch_size,
            "svdd_epochs": svdd_epochs,
            "svdd_warm_up_epochs": svdd_warm_up_epochs,
            "seed": seed,
            "permutations": permutations,
            "global_utility_test_size": global_utility_test_size,
            "synth_train_cap": synth_train_cap,
        },
    }

    for c in numeric_cols:
        ks, w = evaluate_numerical_features(synth_df[c].values, real_df[c].values)
        results["marginal_numeric"][c] = {"ks": ks, "wasserstein": w}

    for c in categorical_cols:
        tv = evaluate_categorical_tv(synth_df[c].values, real_df[c].values)
        results["marginal_categorical_tv"][c] = tv

    X_real, X_synth, feature_names = one_hot_encode_aligned(real_df, synth_df, categorical_cols, numeric_cols)
    results["encoding"]["one_hot_feature_count"] = int(len(feature_names))

    results["discriminator"] = discriminator_eval_with_permutation(
        X_real=X_real, X_synth=X_synth, seed=seed, permutations=permutations
    )

    results["support"] = alpha_beta_metrics(
        X_real=X_real,
        X_synth=X_synth,
        k=k,
        rep_dim=rep_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation=activation,
        dropout_prob=dropout_prob,
        nu=nu,
        lr=svdd_lr,
        weight_decay=svdd_weight_decay,
        batch_size=svdd_batch_size,
        epochs=svdd_epochs,
        warm_up_epochs=svdd_warm_up_epochs,
    )

    results["global_utility"] = compute_global_utility_metric(
        real_df=real_df,
        synth_df=synth_df,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        test_size=global_utility_test_size,
        seed=seed,
        synth_train_cap=synth_train_cap,
    )

    return results


def format_results(results: dict) -> str:
    lines = []
    lines.append("=== Evaluation Summary ===")
    lines.append(f"Real:      {results['paths']['real']}")
    lines.append(f"Synthetic: {results['paths']['synthetic']}")
    lines.append("")
    lines.append("Columns:")
    lines.append(f"  Numeric:     {results['columns']['numeric']}")
    lines.append(f"  Categorical: {results['columns']['categorical']}")
    lines.append(f"  One-hot features: {results['encoding']['one_hot_feature_count']}")
    if results.get("dropped_columns"):
        lines.append(f"  Dropped: {results['dropped_columns']}")
    lines.append("")

    d = results["discriminator"]
    lines.append("Discriminator (logistic regression, held-out test + permutation null):")
    lines.append(f"  AUC:               {d['auc']:.6f}")
    lines.append(f"  Accuracy:          {d['accuracy']:.6f}")
    lines.append(f"  pMSE (test):       {d['pmse_test']:.6f}")
    lines.append(f"  pMSE null mean:    {d['pmse_null_mean']:.6f}")
    lines.append(f"  pMSE ratio (perm): {d['pmse_ratio_perm']:.6f}")
    lines.append(f"  Null percentile:   {d['pmse_null_percentile']:.1f}th")
    lines.append("")

    s = results["support"]
    lines.append("Support / coverage (paper-style one-class embedding):")
    lines.append(f"  Alpha precision (alpha=1.0): {s['alpha_precision']:.6f}")
    lines.append(f"  Beta recall (beta=1.0):      {s['beta_recall']:.6f}")
    lines.append(f"  Integrated IP_alpha:         {s['IP_alpha']:.6f}")
    lines.append(f"  Integrated IR_beta:          {s['IR_beta']:.6f}")
    lines.append(f"  Embedder radius:             {s['embedder']['radius']:.6f}")
    lines.append(f"  Alpha grid:                  {s['alphas']}")
    lines.append(f"  P_alpha curve:               {s['alpha_precision_curve']}")
    lines.append(f"  Beta grid:                   {s['betas']}")
    lines.append(f"  R_beta curve:                {s['beta_recall_curve']}")
    lines.append("")

    gu = results.get("global_utility", {})
    lines.append("Global Utility (TabStruct-style):")
    lines.append(f"  Global utility: {gu.get('global_utility')}")
    lines.append(f"  Columns used:   {gu.get('num_columns_used')}")
    lines.append(f"  Synth train cap:{gu.get('settings', {}).get('synth_train_cap')}")
    if gu.get("skipped"):
        lines.append(f"  Skipped (too few rows): {gu['skipped'].get('too_few_rows', 0)}")
        lines.append(f"  Skipped (single-class/fit fail): {gu['skipped'].get('single_class_or_failed_fit', 0)}")

    per_col = gu.get("per_column", {})
    if per_col:
        vals = [(k, v.get("utility")) for k, v in per_col.items() if v.get("utility") is not None]
        vals_sorted = sorted(vals, key=lambda x: x[1])
        if vals_sorted:
            lines.append("  Per-column utility (lowest 5):")
            for k, u in vals_sorted[:5]:
                lines.append(f"    {k}: {u:.6f}")
            lines.append("  Per-column utility (highest 5):")
            for k, u in vals_sorted[-5:][::-1]:
                lines.append(f"    {k}: {u:.6f}")
    lines.append("")

    lines.append("Marginal (numeric):")
    for col, m in results["marginal_numeric"].items():
        lines.append(
            f"  {col}: KS={m['ks'] if m['ks'] is not None else 'NA'}, "
            f"Wasserstein={m['wasserstein'] if m['wasserstein'] is not None else 'NA'}"
        )
    lines.append("")
    lines.append("Marginal (categorical TV distance):")
    for col, tv in results["marginal_categorical_tv"].items():
        lines.append(f"  {col}: TV={tv if tv is not None else 'NA'}")
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

    results = run_evaluation(
        real_path=args.real,
        synth_path=args.synthetic,
        categorical_cols=args.categorical,
        numeric_cols=args.numeric,
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
    )

    out = format_results(results)
    # print("\n" + out)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(out)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved: {args.output}" + (f" and {args.output_json}" if args.output_json else ""))


if __name__ == "__main__":
    main()
