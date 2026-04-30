"""
Forest-Flow Explainability Analysis
=====================================
This script implements the explainability analysis suggested directly by the
Forest-Flow paper (Jolicoeur-Martineau et al., AISTATS 2024):

    "To better understand the data generation process, one could extract
    feature importance across all XGBoost models. Instead of average, one
    could also get a different plot per noise-level or variable predicted."
    — Section 5, Conclusion

Two experiments are produced:

  1. Average SHAP importance (Figure 4 replication)
     Replicates the paper's Figure 4 on the IBM HR dataset instead of Iris.
     SHAP values are computed for each of the n_t XGBoost models and averaged
     across all models and all training rows to give a single global importance
     ranking. This is the experiment the paper explicitly demonstrates.

  2. Per-noise-level SHAP importance heatmap (paper's suggested extension)
     The paper notes one could get "a different plot per noise-level". This
     experiment does exactly that — showing how feature importance shifts
     across the n_t noise levels from t=1 (pure noise) to t=1/n_t (near-real).
     This is unique to Forest-Flow among all models evaluated: ARF and
     Generative Forests produce a single static importance ranking, whereas
     Forest-Flow produces one per noise level, revealing which features govern
     coarse structure formation vs. fine-detail refinement.

Note on local explainability:
  The Forest-Flow architecture does not natively expose a mechanism for tracing
  the generation of individual synthetic records. The intermediate states of the
  flow process are not retained after generation, making post-hoc local
  attribution infeasible without modifying the sampling procedure itself.
  This script therefore focuses exclusively on global explainability, consistent
  with what the paper claims is possible.

Dependencies:
    pip install ForestDiffusion shap pandas numpy matplotlib seaborn scikit-learn

Usage:
    python forestflow_explainability.py

    Edit DATA_PATH below to point at your dataset.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ── ForestDiffusion ───────────────────────────────────────────────────────────
try:
    from ForestDiffusion import ForestDiffusionModel
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "ForestDiffusion is not installed. Run: pip install ForestDiffusion"
    )

# ── SHAP ─────────────────────────────────────────────────────────────────────
try:
    import shap
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "shap is not installed. Run: pip install shap"
    )

# ── Your preprocessing helper (same as ForestFlow.py uses) ───────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from preprocessing import prepare_training_dataframe
except ImportError:
    # Fallback: identity function if preprocessing is not available
    def prepare_training_dataframe(path, **kwargs):
        return pd.read_csv(path) if isinstance(path, str) else path


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH                    = "data/ibm_hr.csv"
N_T                          = 10      # noise levels (keep low for speed)
DUPLICATE_K                  = 10      # duplicate_K (paper uses 100; lower = faster)
SEED                         = 42
DISCRETE_CARDINALITY_THRESH  = 10
# How many training rows to use for SHAP (full dataset is slow; 200 is enough)
SHAP_SAMPLE_SIZE             = 200


# ─────────────────────────────────────────────────────────────────────────────
# 0.  ENCODE DATA  (mirrors ForestFlow.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def _is_categorical(series):
    return (
        pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )

def _encode_categorical(series):
    categories = series.dropna().unique().tolist()
    mapping = {v: i for i, v in enumerate(categories)}
    return series.map(mapping).astype(float), categories

def _classify_numeric(series, threshold):
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

def encode_dataframe(df, threshold):
    encoded = pd.DataFrame(index=df.index)
    metadata = []
    bin_idx, cat_idx, int_idx = [], [], []
    for i, col in enumerate(df.columns):
        s = df[col]
        info = {"name": col, "kind": "float", "categories": None}
        if _is_categorical(s):
            encoded[col], info["categories"] = _encode_categorical(s)
            info["kind"] = "binary" if len(info["categories"]) <= 2 else "categorical"
        else:
            numeric = pd.to_numeric(s, errors="coerce").astype(float)
            info["kind"], info["categories"] = _classify_numeric(s, threshold)
            if info["kind"] in {"binary", "categorical"}:
                mapping = {v: c for c, v in enumerate(info["categories"])}
                encoded[col] = numeric.map(mapping).astype(float)
            else:
                encoded[col] = numeric
        if info["kind"] == "binary":
            bin_idx.append(i)
        elif info["kind"] == "categorical":
            cat_idx.append(i)
        elif info["kind"] == "integer":
            int_idx.append(i)
        metadata.append(info)
    return encoded, metadata, bin_idx, cat_idx, int_idx


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD + ENCODE + FIT
# ─────────────────────────────────────────────────────────────────────────────
print("── Loading data ──")
try:
    df = prepare_training_dataframe(DATA_PATH,
                                    discrete_cardinality_threshold=DISCRETE_CARDINALITY_THRESH)
except Exception:
    df = pd.read_csv(DATA_PATH)

# Drop constant columns
constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
if constant_cols:
    print(f"  Dropping constant columns: {constant_cols}")
    df = df.drop(columns=constant_cols)

print(f"  Dataset shape: {df.shape}")

print("\n── Encoding ──")
encoded, metadata, bin_idx, cat_idx, int_idx = encode_dataframe(
    df, DISCRETE_CARDINALITY_THRESH
)
feature_names = list(encoded.columns)
X_array = encoded.to_numpy(dtype=float)

print("\n── Fitting ForestDiffusionModel (Forest-Flow) ──")
print(f"  n_t={N_T}, duplicate_K={DUPLICATE_K}, seed={SEED}")
model = ForestDiffusionModel(
    X_array,
    n_t=N_T,
    duplicate_K=DUPLICATE_K,
    diffusion_type="flow",
    bin_indexes=bin_idx,
    cat_indexes=cat_idx,
    int_indexes=int_idx,
    n_jobs=-1,
    seed=SEED,
)
print("  Fit complete.")

# ── Verify that models are accessible ────────────────────────────────────────
# ForestDiffusion 1.x stores per-noise-level XGBoost Boosters in regr_[0]
# (regr_ is a list of batches; batch 0 holds the n_t models for flow fitting).
if not hasattr(model, "regr_") or not model.regr_:
    raise AttributeError(
        "ForestDiffusionModel does not expose a 'regr_' attribute. "
        "Check your ForestDiffusion version — the internal attribute may have changed."
    )

xgb_models = model.regr_[0]   # list of n_t XGBoost Booster objects
n_models = len(xgb_models)
print(f"\n  Accessible XGBoost models: {n_models} (one per noise level)")
print(f"  Noise levels: t = 1/{N_T}, 2/{N_T}, ..., {N_T}/{N_T}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SHAP VALUES PER MODEL
#     For each of the n_t XGBoost models, compute SHAP values on a sample of
#     the encoded training data. Each model predicts the vector field (target
#     is a d-dimensional output), so we average SHAP absolute values across
#     output dimensions to get a single importance per feature per model.
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n── Computing SHAP values (sample size = {SHAP_SAMPLE_SIZE}) ──")

# Sample rows for SHAP (using full dataset is slow).
# Use model.X1 — the internally scaled array the Boosters were trained on —
# so pred_contribs shape matches the model's expected feature space.
np.random.seed(SEED)
sample_idx = np.random.choice(len(X_array),
                               size=min(SHAP_SAMPLE_SIZE, len(X_array)),
                               replace=False)
X_sample = model.X1[sample_idx]

# importance_matrix[t, f] = mean |SHAP| for feature f at noise level t
importance_matrix = np.zeros((n_models, len(feature_names)))

import xgboost as xgb

for t_idx, xgb_model in enumerate(xgb_models):
    noise_level = (t_idx + 1) / N_T
    print(f"  Computing SHAP for model {t_idx+1}/{n_models}  (t={noise_level:.2f})", end="\r")

    try:
        # Use XGBoost's native pred_contribs (avoids SHAP/XGBoost version conflicts).
        # Returns (n_samples, n_outputs, n_features + 1); last column is the bias term.
        dm = xgb.DMatrix(X_sample)
        contribs = xgb_model.predict(dm, pred_contribs=True)
        arr = np.array(contribs)

        if arr.ndim == 3:
            # (n_samples, n_outputs, n_features+1) — drop bias, average over outputs
            mean_abs = np.mean(np.abs(arr[:, :, :-1]), axis=(0, 1))
        elif arr.ndim == 2:
            # (n_samples, n_features+1) — single output, drop bias
            mean_abs = np.mean(np.abs(arr[:, :-1]), axis=0)
        else:
            mean_abs = np.abs(arr).flatten()[:len(feature_names)]

        importance_matrix[t_idx] = mean_abs[:len(feature_names)]

    except Exception as e:
        print(f"\n  Warning: SHAP failed for model {t_idx+1}: {e}")
        try:
            scores = xgb_model.get_score(importance_type="gain")
            fi = np.array([scores.get(f"f{i}", 0.0) for i in range(len(feature_names))])
            importance_matrix[t_idx] = fi[:len(feature_names)]
        except Exception:
            pass

print(f"\n  SHAP computation complete.")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  EXPERIMENT 1 — AVERAGE SHAP IMPORTANCE BAR CHART
#     Replication of the paper's Figure 4 on IBM HR instead of Iris.
#     Average importance across all n_t models.
# ─────────────────────────────────────────────────────────────────────────────

avg_importance = importance_matrix.mean(axis=0)
fi_series = pd.Series(avg_importance, index=feature_names).sort_values(ascending=False)

print("\n" + "═" * 60)
print("AVERAGE SHAP IMPORTANCE  (across all noise levels)")
print("═" * 60)
print(fi_series.round(4).to_string())

fig, ax = plt.subplots(figsize=(8, max(4, len(fi_series) * 0.28)))
fi_series.sort_values().plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
ax.set_xlabel("Mean |SHAP value|", fontsize=11)
ax.set_title(
    "Forest-Flow: Average Feature Importance (SHAP)\n"
    "Averaged across all XGBoost models (all noise levels)",
    fontsize=12
)
ax.axvline(avg_importance.mean(), color="red", linestyle="--", linewidth=1,
           label=f"Mean = {avg_importance.mean():.4f}")
ax.legend()
plt.tight_layout()
plt.savefig("ff_average_shap_importance.png", dpi=150)
plt.close()
print("\nSaved → ff_average_shap_importance.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  EXPERIMENT 2 — PER-NOISE-LEVEL SHAP HEATMAP
#     The paper's suggested extension: "one could also get a different plot
#     per noise-level". This heatmap shows all noise levels simultaneously,
#     revealing how feature relevance evolves across the generative trajectory.
#
#     Interpretation:
#       t close to 1  =  model is correcting noisy / coarse structure
#       t close to 0  =  model is refining fine detail near the real distribution
# ─────────────────────────────────────────────────────────────────────────────

# Normalise each noise level row so colours reflect relative importance
# at that level (not absolute magnitude, which tends to dominate at low t)
row_maxes = importance_matrix.max(axis=1, keepdims=True)
row_maxes[row_maxes == 0] = 1.0
importance_normalised = importance_matrix / row_maxes

# Sort features by average importance for readability
sorted_feature_idx = np.argsort(avg_importance)[::-1]
sorted_feature_names = [feature_names[i] for i in sorted_feature_idx]
sorted_matrix = importance_normalised[:, sorted_feature_idx]  # (n_t, n_features)

noise_level_labels = [f"t={((i+1)/N_T):.2f}" for i in range(N_T)]

fig, ax = plt.subplots(figsize=(max(10, len(feature_names) * 0.35), max(4, N_T * 0.55)))
sns.heatmap(
    sorted_matrix,
    ax=ax,
    xticklabels=sorted_feature_names,
    yticklabels=noise_level_labels,
    cmap="YlOrRd",
    linewidths=0.3,
    linecolor="white",
    cbar_kws={"label": "Normalised |SHAP| (per noise level)"},
    vmin=0, vmax=1,
)
ax.set_xlabel("Feature", fontsize=11)
ax.set_ylabel("Noise level t  (t=1 → noise,  t≈0 → real data)", fontsize=11)
ax.set_title(
    "Forest-Flow: Feature Importance Across Noise Levels\n"
    "(each row normalised — shows relative importance at each generative step)",
    fontsize=12
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig("ff_noise_level_shap_heatmap.png", dpi=150)
plt.close()
print("Saved → ff_noise_level_shap_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DIAGNOSTIC: PRINT TOP FEATURES AT EARLY VS LATE NOISE LEVELS
#     Useful for writing up the results — shows concretely whether different
#     features dominate at different stages of generation.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═" * 60)
print("TOP 5 FEATURES BY NOISE LEVEL (unnormalised mean |SHAP|)")
print("═" * 60)
for t_idx in range(N_T):
    noise_level = (t_idx + 1) / N_T
    top5 = pd.Series(importance_matrix[t_idx], index=feature_names)\
             .sort_values(ascending=False).head(5)
    top5_str = ", ".join([f"{n} ({v:.4f})" for n, v in top5.items()])
    print(f"  t={noise_level:.2f}:  {top5_str}")


print("\n✓  Forest-Flow explainability outputs generated successfully.")
print("   Files produced:")
print("     ff_average_shap_importance.png   — Figure 4 replication (IBM HR)")
print("     ff_noise_level_shap_heatmap.png  — per-noise-level importance heatmap")
print("\n   Citation for experiments:")
print("     Jolicoeur-Martineau et al. (2024)")
print("     'Generating and Imputing Tabular Data via Diffusion and")
print("      Flow-based Gradient-Boosted Trees', AISTATS 2024.")
