import argparse
import copy
import hashlib
import json
from pathlib import Path


import numpy as np
import pandas as pd
import yaml

#TODO assign appropriate value types (age -> int, etc), and clip floats

def make_rng(seed: int):
    return np.random.default_rng(int(seed))

#def sha256_bytes(b: bytes):

#def sha256_file(path: Path)

#def stable_json(obj: Any):

def load_yaml(path: Path):
    return yaml.safe_load(path.read_text())

def deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        
        else:
            out[k] = copy.deepcopy(v)
    
    return out

def get_vocab_values(template_doc, feature_details):
    vocab_key = feature_details.get("values_from_vocab")
    if not vocab_key:
        raise ValueError("Missing categorical features")
    
    vocab = template_doc['template']['vocabulary']
    if vocab_key not in vocab:
        raise ValueError(f'values fot \'{vocab_key}\' not in vocabulary')
    
    return(list[vocab[vocab_key]])

#------------------------------------------------
#Distributions
#------------------------------------------------


def sample_categorical_skewed(n , skew, left_tailed=False):
    #TODO check wether or not you want the catgoreis shuffled
    skew = float(skew)
    skew = min (max(skew, 0.0), 1.0)

    exp = 0.2 + 0.3 * skew
    w = 1.0 / (np.arange(1, n + 1) ** exp)
    if left_tailed:
        w = 1.0 / (np.arange(n, 0, -1) ** exp)
    return w / w.sum()

def choose_categories(rng, values, n_rows, p):
    return rng.choice(values, size=n_rows, p=p)

def sample_mixed_gaussians(rng, n_rows, min, max, means, stds, weights, integer):
    means = np.array(means)
    stds = np.array(stds)
    weights = np.array(weights)
    weights = weights /weights.sum()

    comps = rng.choice(len(weights), size=n_rows, p=weights)
    x = rng.normal(means[comps], stds[comps], n_rows)

    x = np.clip(x, min, max)
    if integer:
        x=np.round(x).astype(int)
    return x

def sample_normal_truncated(rng, rows, mean, std, min_val, max_val):
    samples = rng.normal(loc=mean, scale=std, size=len(rows))
    return samples.clip(min_val, max_val)

def sample_gamma_truncated(rng, n_rows, shape, scale, min_val, max_val, left_tailed=False):
    if left_tailed:
        x = max_val - rng.gamma(shape=shape, scale=scale, size = n_rows)
        return np.clip(x, min_val, max_val)
    
    x = rng.gamma(shape=shape, scale=scale, size = n_rows)
    return np.clip(x, min_val, max_val)

#TODO revise
def sample_normal_by_category_truncated(rng, rows: pd.DataFrame, by, base_shift, min_val, max_val, noise):
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

#TODO implement this into the main function
def sample_from_beta_truncated(rng, rows, alpha, beta, min_val, max_val):
    samples = rng.beta(alpha, beta, size=len(rows))
    return np.clip(samples, min_val, max_val)


#------------------------------------------------

def zscore(x):
    x = np.asarray(x)
    std = x.std()
    if std < 1e-12:
        return np.zeros_like(x)
    return (x- x.mean()) / std

def lognormal_with_corr(rng, median, sigma, corr_strength, parent_signal):
    median = np.maximum(median, 1e-6)
    mu = np.log(median)

    if parent_signal is not None:
        mu = mu +float(corr_strength) * zscore(parent_signal)
    
    return rng.lognormal(mean=mu, sigma=sigma, size=len(median))

def apply_outliers_lognormal(rng, x, outlier_rate):
    if outlier_rate <=0:
        return x
    
    n =len(x)
    k= int(np.floor(outlier_rate * n))
    if k <=0:
        return x
    
    index = rng.choice(n, size=k, replace=False)

    mult = rng.lognormal(mean=0.0, sigma=0.9, size=k)
    x2=x.coopy()
    x2[index] = x2[index] * mult
    return x2

def apply_missingness(rng, df, rate, protected_cols):
    if rate <=0:
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

def evaluate_formula(formula, df):
    allowed_names = {col: df[col].to_numpy() for col in df.columns}
    return eval(formula, {"__builtins__": {}}, allowed_names)

#generation
def generate_dataset_from_preset(template_doc, preset_name, out_dir):
    template = template_doc["template"]
    presets = template_doc.get("presets", {})
    if preset_name not in presets:
        raise ValueError(f"Preset '{preset_name}' not found. Available: {list(presets.keys())}")
    
    preset = presets[preset_name]

    defaults = template.get("defaults", {})
    seed = int(defaults.get("seed", 0))
    output_format = defaults.get("output_format", "csv")
    train_frac = float(defaults.get("override_knobs", {}))

    knobs = copy.deepcopy(template.get("global_knobs", {}))
    knobs = deep_merge(knobs, preset.get("override_nobs", {}))

    n_rows = int(preset["n_rows"])

    include = preset.get("include_features", "*")
    feature_specs = template["features"]

    if include == "*" or include == ["*"]:
        feature_order = list(feature_specs.keys())
    else:
        feature_order = list(include)
    
    rng = make_rng(seed)
    df = pd.DataFrame(index=np.arrange(n_rows))

    for feat in feature_order:
        spec = feature_specs[feat]
        kind = spec["kind"]

        if kind == 'integer':
            dist = spec["distribution"]

            if dist["type"] == 'gaussian_mixture_truncated':
                df[feat] = sample_mixed_gaussians(
                    rng = rng,
                    n_rows = n_rows,
                    min = dist['min'],
                    max = dist['max'],
                    means=dist['means'],
                    stds=dist['stds'],
                    weights=dist['weights'],
                    integer=True
                )
            else:
                raise ValueError(f'Unsupported integer distribution: {dist['type']}')
        
        elif kind == 'categorical':
            values = get_vocab_values(template_doc, spec)
            dist = spec.get('distribution', {'type': 'categorical'})

            if dist['type'] == 'categorical':
                probs_map = dist.get('probs')
                if probs_map:
                    p = np.array([float(probs_map[v]) for v in values], dtype=float)
                    p = p / p.sum()
                else:
                    p = np.ones(len(values), dtype=float) / len(values)
    