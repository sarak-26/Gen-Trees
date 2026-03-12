from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


# ============================================================
# 1) Repro utilities
# ============================================================

def make_rng(seed: int) -> np.random.Generator:
    """Create a deterministic random number generator (RNG)."""
    return np.random.default_rng(int(seed))


def sha256_bytes(b: bytes) -> str:
    """SHA256 hash of bytes (useful for hashing configs)."""
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    """SHA256 hash of a file (useful for reproducibility verification)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json(obj: Any) -> str:
    """Stable JSON string so the hash doesn't change due to key order."""
    return json.dumps(obj, indent=2, sort_keys=True, default=str)


# ============================================================
# 2) Config helpers
# ============================================================

def load_yaml(path: Path) -> dict:
    """Load YAML file into a Python dict."""
    return yaml.safe_load(path.read_text())


def deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge override into base. Override wins.
    Useful for applying preset override_knobs cleanly.
    """
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def get_vocab_values(template_doc: dict, feature_spec: dict) -> List[str]:
    """
    For categorical features, fetch allowed values from template vocabulary.
    Example: values_from_vocab: regions -> template.vocabulary.regions
    """
    vocab_key = feature_spec.get("values_from_vocab")
    if not vocab_key:
        raise ValueError("categorical feature missing values_from_vocab")

    vocab = template_doc["template"]["vocabulary"]
    if vocab_key not in vocab:
        raise ValueError(f"values_from_vocab '{vocab_key}' not in vocabulary")

    return list(vocab[vocab_key])


# ============================================================
# 3) Small distribution helpers
# ============================================================

def categorical_probs_skewed(n: int, skew: float) -> np.ndarray:
    """
    Make category probabilities with controllable skew.
    - skew ~ 0 => near-uniform
    - skew ~ 1 => very concentrated on early categories
    """
    skew = float(skew)
    skew = min(max(skew, 0.0), 1.0)

    # Zipf-like weights: exponent controls how steep the long tail is
    exponent = 0.2 + 3.0 * skew  # range ~ 0.2..3.2
    w = 1.0 / (np.arange(1, n + 1) ** exponent)
    return w / w.sum()

#Random sampling
def choose_categories(rng: np.random.Generator, values: List[str], n_rows: int, p: np.ndarray) -> np.ndarray:
    """Sample categorical values from probability vector p."""
    return rng.choice(values, size=n_rows, p=p)


def sample_gaussian_mixture_truncated(
    rng: np.random.Generator,
    n_rows: int,
    min_val: float,
    max_val: float,
    means: List[float],
    stds: List[float],
    weights: List[float],
    integer: bool,
) -> np.ndarray:
    """
    Sample from a Gaussian mixture model, then truncate to [min_val, max_val].
    Useful for realistic age distributions (non-uniform).
    """
    means = np.array(means, dtype=float)
    stds = np.array(stds, dtype=float)
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()

    comps = rng.choice(len(weights), size=n_rows, p=weights)
    x = rng.normal(loc=means[comps], scale=stds[comps], size=n_rows)

    x = np.clip(x, min_val, max_val)
    if integer:
        x = np.round(x).astype(int)
    return x


def sample_gamma_truncated(
    rng: np.random.Generator,
    n_rows: int,
    shape: float,
    scale: float,
    min_val: float,
    max_val: float,
) -> np.ndarray:
    """
    Gamma distribution is right-skewed (good for tenure).
    Then truncate to bounds.
    """
    x = rng.gamma(shape=float(shape), scale=float(scale), size=n_rows)
    return np.clip(x, min_val, max_val)


# def normal_by_category_truncated(
#     rng: np.random.Generator,
#     categories: np.ndarray,
#     by: Dict[str, Dict[str, float]],
#     min_val: float,
#     max_val: float,
#     noise: float,
# ) -> np.ndarray:
#     """
#     Generate a continuous variable with a different Normal(mean, sd) per category.
#     Example: hours_worked depends on employment_type.
#     """
#     out = np.empty(len(categories), dtype=float)
#     for cat, params in by.items():
#         mask = categories == cat
#         if mask.any():
#             mu = float(params["mean"])
#             sd = float(params["std"])
#             out[mask] = rng.normal(mu, sd, size=mask.sum())

#     # Truncate to bounds
#     out = np.clip(out, min_val, max_val)

    # # Optional extra global noise knob (makes generation harder)
    # if noise and float(noise) > 0:
    #     out = out + rng.normal(0.0, float(noise), size=len(out))
    #     out = np.clip(out, min_val, max_val)

    # return out

def normal_by_category_truncated(rng, rows: pd.DataFrame, by, base_shift, min_val, max_val, noise):
    out = np.full(len(rows), np.nan)
    base_feature = by['feature_name']
    shift_features = base_shift.keys()

    for cat, params in by['categories'].items():
        for feature in shift_features:
            for cat2, params2 in base_shift[feature].items():
                mask_shift = np.array([
                    (row[base_feature] == cat and row[feature] == cat2)
                    for _, row in rows.iterrows()
                ])
                if mask_shift.any():
                    mu = params["mean"] + params2['mean_shift']
                    sd = params["std"]
                    out[mask_shift] = rng.normal(mu, sd, size=mask_shift.sum())
    
    out = np.clip(out, min_val, max_val)

    if (noise is not None) and float(noise) > 0:
        out = out + rng.normal(0.0, float(noise), size = len(out))
        out = np.clip(out, min_val, max_val)
    
    return out


# ============================================================
# 4) Correlation-strength helper
# ============================================================

def zscore(x: np.ndarray) -> np.ndarray:
    """Standardize array to mean 0, std 1 (safe for constant arrays)."""
    x = np.asarray(x, dtype=float)
    std = x.std()
    if std < 1e-12:
        return np.zeros_like(x)
    return (x - x.mean()) / std


# ============================================================
# 5) Lognormal generation with anchor median + correlation_strength
# ============================================================

def lognormal_anchor_median_with_corr(
    rng: np.random.Generator,
    median: np.ndarray,
    sigma: float,
    corr_strength: float,
    parent_signal: np.ndarray | None,
) -> np.ndarray:
    """
    Base: X ~ LogNormal(mu, sigma), median(X) = exp(mu).
    So mu_base = log(median).

    correlation_strength:
    - If parent_signal is provided, we add corr_strength * z(parent_signal) to mu.
      That means: when parent is high, earnings shift upward more consistently.
    """
    median = np.maximum(median, 1e-6)
    mu = np.log(median)

    if parent_signal is not None:
        mu = mu + float(corr_strength) * zscore(parent_signal)

    return rng.lognormal(mean=mu, sigma=float(sigma), size=len(median))


def apply_outliers_lognormal(
    rng: np.random.Generator,
    x: np.ndarray,
    outlier_rate: float,
) -> np.ndarray:
    """
    Outlier injection: pick a fraction of rows and multiply by a heavy-tailed factor.
    """
    rate = float(outlier_rate)
    if rate <= 0:
        return x

    n = len(x)
    k = int(np.floor(rate * n))
    if k <= 0:
        return x

    idx = rng.choice(n, size=k, replace=False)

    # Heavy-tailed multiplier (mostly > 1)
    mult = rng.lognormal(mean=0.0, sigma=0.9, size=k)

    x2 = x.copy()
    x2[idx] = x2[idx] * mult
    return x2


# ============================================================
# 6) Missingness (MCAR)
# ============================================================

def apply_missingness(
    rng: np.random.Generator,
    df: pd.DataFrame,
    rate: float,
    protected_cols: List[str] | None = None,
) -> pd.DataFrame:
    """
    Apply Missing Completely At Random (MCAR) missingness to non-protected columns.
    missingness_rate is a global knob in your template.
    """
    rate = float(rate)
    if rate <= 0:
        return df

    protected = set(protected_cols or [])
    cols = [c for c in df.columns if c not in protected]
    if not cols:
        return df

    out = df.copy()
    n = len(out)
    for c in cols:
        mask = rng.uniform(size=n) < rate
        out.loc[mask, c] = np.nan
    return out


# ============================================================
# 7) Formula evaluation (for median_formula)
# ============================================================

def evaluate_formula(formula: str, df: pd.DataFrame) -> np.ndarray:
    """
    Minimal evaluator for arithmetic formulas referencing column names.
    Example: "18 * hours_worked"

    NOTE: We restrict builtins to reduce risk.
    """
    allowed_names = {col: df[col].to_numpy() for col in df.columns}
    return eval(formula, {"__builtins__": {}}, allowed_names)


# ============================================================
# 8) Main generation per preset
# ============================================================

def generate_dataset_from_preset(template_doc: dict, preset_name: str, out_dir: Path) -> Tuple[Path, dict]:
    template = template_doc["template"]
    presets = template_doc.get("presets", {})
    if preset_name not in presets:
        raise ValueError(f"Preset '{preset_name}' not found. Available: {list(presets.keys())}")

    preset = presets[preset_name]

    # Defaults
    defaults = template.get("defaults", {})
    seed = int(defaults.get("seed", 0))
    output_format = defaults.get("output_format", "csv")
    train_frac = float(defaults.get("train_frac", 0.8))

    # Effective knobs = template knobs + preset overrides
    knobs = copy.deepcopy(template.get("global_knobs", {}))
    knobs = deep_merge(knobs, preset.get("override_knobs", {}))

    n_rows = int(preset["n_rows"])

    # Which features to generate
    include = preset.get("include_features", "*")
    feature_specs = template["features"]

    if include == "*" or include == ["*"]:
        feature_order = list(feature_specs.keys())
    else:
        feature_order = list(include)

    # RNG for data generation
    rng = make_rng(seed)

    df = pd.DataFrame(index=np.arange(n_rows))

    # Generate each feature in order
    for feat in feature_order:
        spec = feature_specs[feat]
        kind = spec["kind"]

        # -------------------------
        # Integer features
        # -------------------------
        if kind == "integer":
            dist = spec["distribution"]

            if dist["type"] == "gaussian_mixture_truncated":
                df[feat] = sample_gaussian_mixture_truncated(
                    rng=rng,
                    n_rows=n_rows,
                    min_val=float(dist["min"]),
                    max_val=float(dist["max"]),
                    means=dist["means"],
                    stds=dist["stds"],
                    weights=dist["weights"],
                    integer=True,
                )
            else:
                raise ValueError(f"Unsupported integer distribution: {dist['type']}")

        # -------------------------
        # Categorical features
        # -------------------------
        elif kind == "categorical":
            values = get_vocab_values(template_doc, spec)
            dist = spec.get("distribution", {"type": "categorical"})

            if dist["type"] == "categorical":
                probs_map = dist.get("probs")
                if probs_map:
                    p = np.array([float(probs_map[v]) for v in values], dtype=float)
                    p = p / p.sum()
                else:
                    p = np.ones(len(values), dtype=float) / len(values)

                df[feat] = choose_categories(rng, values, n_rows, p)

            elif dist["type"] == "categorical_skewed":
                knob_name = dist["skew_from_knob"]
                skew = float(knobs.get(knob_name, 0.7))
                p = categorical_probs_skewed(len(values), skew)
                df[feat] = choose_categories(rng, values, n_rows, p)

            else:
                raise ValueError(f"Unsupported categorical distribution: {dist['type']}")

        # -------------------------
        # Continuous features
        # -------------------------
        elif kind == "continuous":

            # A) "distribution" style (direct draw)
            if "distribution" in spec:
                dist = spec["distribution"]

                if dist["type"] == "gamma_truncated":
                    df[feat] = sample_gamma_truncated(
                        rng=rng,
                        n_rows=n_rows,
                        shape=float(dist["shape"]),
                        scale=float(dist["scale"]),
                        min_val=float(dist["min"]),
                        max_val=float(dist["max"]),
                    )
                else:
                    raise ValueError(f"Unsupported continuous distribution: {dist['type']}")

            # B) "generation" style (depends_on other columns)
            elif "generation" in spec:
                gen = spec["generation"]
                depends_on = spec.get("depends_on", [])

                if gen["type"] == "normal_by_category_truncated":
                    # Example: hours_worked depends on employment_type (categorical)
                    if len(depends_on) != 1:
                        raise ValueError(f"{feat}: normal_by_category_truncated expects depends_on with 1 feature")

                    dep_col = depends_on[0]
                    if dep_col not in df.columns:
                        raise ValueError(f"{feat} depends_on {dep_col} which hasn't been generated yet")

                    # noise knob
                    noise_knob = spec.get("noise_from_knob", "noise")
                    noise = float(knobs.get(noise_knob, 0.0))

                    df[feat] = normal_by_category_truncated(
                        rng=rng,
                        categories=df[dep_col].to_numpy(),
                        by=gen["by"],
                        min_val=float(gen["min"]),
                        max_val=float(gen["max"]),
                        noise=noise,
                    )

                elif gen["type"] == "lognormal_anchor_median":
                    # Example: weekly_earnings depends on hours_worked (continuous)
                    # 1) compute median from formula
                    median = evaluate_formula(gen["median_formula"], df)

                    # 2) correlation_strength knob (applies extra coupling to parent signal)
                    corr_strength = float(knobs.get("correlation_strength", 0.0))

                    # pick a parent signal if possible:
                    # if depends_on contains a numeric parent, use the first one
                    parent_signal = None
                    if depends_on:
                        # hours_worked is numeric, so this works nicely
                        parent_signal = df[depends_on[0]].to_numpy(dtype=float)

                    # 3) sample lognormal anchored at median, with correlation adjustment
                    sigma = float(gen.get("sigma_base", 0.35))
                    x = lognormal_anchor_median_with_corr(
                        rng=rng,
                        median=median,
                        sigma=sigma,
                        corr_strength=corr_strength,
                        parent_signal=parent_signal,
                    )

                    # 4) outliers from knob
                    out_cfg = gen.get("outliers", {})
                    if out_cfg:
                        rate_knob = out_cfg.get("rate_from_knob")
                        if rate_knob:
                            out_rate = float(knobs.get(rate_knob, 0.0))
                            x = apply_outliers_lognormal(rng, x, out_rate)

                    # 5) minimum bound
                    minv = float(gen.get("min", -np.inf))
                    x = np.clip(x, minv, np.inf)

                    df[feat] = x

                else:
                    raise ValueError(f"Unsupported continuous generation: {gen['type']}")

            else:
                raise ValueError(f"{feat}: continuous feature must have 'distribution' or 'generation'")

        else:
            raise ValueError(f"Unsupported kind: {kind} for feature {feat}")

    # Apply global missingness knob (MCAR)
    miss_rate = float(knobs.get("missing_rate", 0.0))

    # Protect a couple important columns from missingness by default
    protected = ["region", "employment_type"]
    df = apply_missingness(rng, df, rate=miss_rate, protected_cols=protected)

    # Deterministic train/holdout split:
    # Use a separate RNG stream so the split is stable even if generation changes later.
    idx = np.arange(n_rows)
    rng_split = make_rng(seed + 10_000)
    rng_split.shuffle(idx)

    n_train = int(np.floor(train_frac * n_rows))
    train_idx = idx[:n_train]
    hold_idx = idx[n_train:]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_hold = df.iloc[hold_idx].reset_index(drop=True)

    # Write outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{preset_name}_seed{seed}"

    if output_format == "csv":
        data_path = out_dir / f"{stem}.csv"
        train_path = out_dir / f"{stem}_train.csv"
        hold_path = out_dir / f"{stem}_holdout.csv"
        df.to_csv(data_path, index=False)
        df_train.to_csv(train_path, index=False)
        df_hold.to_csv(hold_path, index=False)

    elif output_format == "parquet":
        data_path = out_dir / f"{stem}.parquet"
        train_path = out_dir / f"{stem}_train.parquet"
        hold_path = out_dir / f"{stem}_holdout.parquet"
        df.to_parquet(data_path, index=False)
        df_train.to_parquet(train_path, index=False)
        df_hold.to_parquet(hold_path, index=False)

    else:
        raise ValueError(f"Unsupported output_format: {output_format}")

    # Metadata for academic reproducibility
    meta = {
        "preset": preset_name,
        "seed": seed,
        "n_rows": n_rows,
        "train_frac": train_frac,
        "output_format": output_format,
        "included_features": feature_order,
        "effective_knobs": knobs,
        "files": {
            "full": data_path.name,
            "train": train_path.name,
            "holdout": hold_path.name,
        },
        "hashes": {
            "full_sha256": sha256_file(data_path),
            "train_sha256": sha256_file(train_path),
            "holdout_sha256": sha256_file(hold_path),
            "template_sha256": sha256_bytes(stable_json(template_doc).encode("utf-8")),
        },
        "summary": {
            "columns": list(df.columns),
            "null_rate_overall": float(df.isna().mean().mean()),
        },
        "notes": {
            "correlation_strength_effect": (
                "Applied only when a feature uses lognormal_anchor_median and has depends_on; "
                "we add correlation_strength * z(parent) to lognormal mu."
            )
        },
    }

    meta_path = out_dir / f"{stem}.meta.json"
    meta_path.write_text(stable_json(meta))

    return data_path, meta


# ============================================================
# 9) CLI entry point
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, help="Path to dataset_template.yaml")
    ap.add_argument("--preset", default="ALL", help="Preset name (D1/D2/D3) or ALL")
    ap.add_argument("--out_dir", default="data/generated", help="Output directory")
    args = ap.parse_args()

    template_path = Path(args.template)
    out_dir = Path(args.out_dir)

    template_doc = load_yaml(template_path)

    if args.preset == "ALL":
        preset_names = list(template_doc.get("presets", {}).keys())
        if not preset_names:
            raise ValueError("No presets found in template file.")

        for p in preset_names:
            data_path, meta = generate_dataset_from_preset(template_doc, p, out_dir)
            print(f"[{p}] wrote {data_path} (sha256={meta['hashes']['full_sha256']})")
    else:
        data_path, meta = generate_dataset_from_preset(template_doc, args.preset, out_dir)
        print(f"[{args.preset}] wrote {data_path} (sha256={meta['hashes']['full_sha256']})")


if __name__ == "__main__":
    main()
