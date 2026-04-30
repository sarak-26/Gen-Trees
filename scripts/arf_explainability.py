"""
ARF Explainability Analysis
============================
This script demonstrates three explainability techniques for Adversarial Random Forests
using the arfpy Python package (Blesch & Wright, 2023).

Techniques covered:
  1. Row tracing  — explain why a specific synthetic row was generated as it was,
                    by identifying the leaf partition that produced it and the real
                    training records that inhabit that leaf.
  2. Feature importance — rank features by how often they are used to split the
                          data across the final ARF's trees (global explainability).
  3. Leaf partition diagram — visualise the decision-rule boundary for a chosen
                              synthetic row, showing the real records that share
                              its leaf and the bounds on each feature.

Dependencies:
    pip install arfpy scikit-learn pandas numpy matplotlib seaborn
"""

import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from arfpy import arf as arfpy

# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP  — allow imports from src/ and src/models/
# ─────────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src", "models"))

from preprocessing import prepare_training_dataframe
from adversarial_rforest import (
    _seed_everything,
    _apply_finite_bounds,
    _postprocess_generated_data,
)

# ─────────────────────────────────────────────────────────────────────────────
# 0.  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "ibm_hr.csv")
SEED      = 42
N_SYNTH   = 500

print(f"Loading data from: {DATA_PATH}")
df = prepare_training_dataframe(DATA_PATH)

FEATURE_COLS = df.columns.tolist()
print(f"Dataset shape: {df.shape}")
print(df.head(3))

# ─────────────────────────────────────────────────────────────────────────────
# 1.  TRAIN ARF  →  FORDE  →  FORGE
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Training ARF ──")
_seed_everything(SEED)
my_arf = arfpy.arf(x=df, num_trees=50, min_node_size=5, verbose=True)

print("\n── Estimating leaf distributions (FORDE) ──")
my_arf.forde()
_apply_finite_bounds(my_arf, df, finite_bounds="global")

print("\n── Generating synthetic data (FORGE) ──")
df_synth = my_arf.forge(n=N_SYNTH)
df_synth = _postprocess_generated_data(df_synth, df)
print(f"Generated {len(df_synth)} synthetic rows.")
print(df_synth.head(3))

# Save the analysed row (#0) plus 4 additional rows as a small sample
TRACED_ROW_IDX = 0
other_idxs = [i for i in range(1, len(df_synth)) if i != TRACED_ROW_IDX][:4]
sample_rows = df_synth.iloc[[TRACED_ROW_IDX] + other_idxs].reset_index(drop=True)
sample_path = os.path.join(_PROJECT_ROOT, "synthetic_data", "arf_explainability_sample_ibm_hr.csv")
sample_rows.to_csv(sample_path, index=False)
print(f"Saved sample rows (including traced row) → {sample_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  ROW TRACING
#     For a chosen synthetic row, identify:
#       (a) which leaf in the final ARF it falls into (per tree)
#       (b) the decision rules that define that leaf
#       (c) the real training records in the same leaf
# ─────────────────────────────────────────────────────────────────────────────

def _encode_for_clf(df_rows, arf_model):
    """Encode DataFrame rows using arfpy's internal encoding (cat.codes for categoricals)."""
    encoded = df_rows[arf_model.x_real.columns].copy()
    for col in arf_model.x_real.columns:
        if arf_model.factor_cols[col]:
            encoded[col] = pd.Categorical(encoded[col], categories=arf_model.levels[col]).codes
    return encoded


def trace_synthetic_row(arf_model, real_df, synth_row, row_idx=0, max_trees=3):
    """
    Trace a single synthetic row back through the ARF's final forest.

    Parameters
    ----------
    arf_model  : trained arfpy.arf object
    real_df    : the original training DataFrame (same columns)
    synth_row  : a one-row DataFrame representing the synthetic record
    row_idx    : label for printing (which synthetic row number this is)
    max_trees  : how many trees to show decision paths for (keeps output readable)

    Returns
    -------
    traces : list of dicts, one per tree, each containing
             'leaf_id', 'n_real_in_leaf', 'leaf_real_records', 'decision_path'
    """
    final_forest  = arf_model.clf
    X_real        = arf_model.x_real                  # already encoded, shape (n, 35)
    X_synth       = _encode_for_clf(synth_row, arf_model)
    feature_names = X_real.columns.tolist()
    traces = []

    for tree_idx, estimator in enumerate(final_forest.estimators_[:max_trees]):
        leaf_id_real  = estimator.apply(X_real)
        leaf_id_synth = estimator.apply(X_synth)[0]

        real_in_leaf_mask = (leaf_id_real == leaf_id_synth)
        real_in_leaf      = real_df[real_in_leaf_mask.tolist()]

        node_indicator = estimator.decision_path(X_synth)
        node_ids       = node_indicator.indices

        tree_struct = estimator.tree_
        rules = []
        for node_id in node_ids[:-1]:
            feat_idx  = tree_struct.feature[node_id]
            feature   = feature_names[feat_idx]
            threshold = tree_struct.threshold[node_id]
            val       = X_synth.values[0, feat_idx]
            direction = "<=" if val <= threshold else ">"
            rules.append(f"  {feature} {direction} {threshold:.3f}  (row value: {val:.3f})")

        traces.append({
            "tree_idx"         : tree_idx,
            "leaf_id"          : leaf_id_synth,
            "n_real_in_leaf"   : int(real_in_leaf_mask.sum()),
            "leaf_real_records": real_in_leaf,
            "decision_rules"   : rules,
        })

    return traces


# Pick the first synthetic row as our example
target_row = df_synth.iloc[[0]]

print("\n" + "═" * 60)
print(f"ROW TRACING  —  Synthetic row #0")
print("═" * 60)
print("Synthetic record values:")
print(target_row.to_string(index=False))

traces = trace_synthetic_row(my_arf, df, target_row, row_idx=0, max_trees=3)

for t in traces:
    print(f"\n  Tree {t['tree_idx']}  →  Leaf node #{t['leaf_id']}")
    print(f"  Real training records sharing this leaf: {t['n_real_in_leaf']}")
    print("  Decision path from root:")
    for rule in t['decision_rules']:
        print(rule)
    if t['n_real_in_leaf'] > 0:
        print("  Leaf real-record summary:")
        print(t['leaf_real_records'].describe().round(3).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FEATURE IMPORTANCE
#     Aggregate split frequency across all trees in the final ARF forest.
#     Columns used more often to split are more important to the data structure.
# ─────────────────────────────────────────────────────────────────────────────

def arf_feature_importance(arf_model, feature_names=None):
    """
    Compute mean impurity-decrease feature importance from the final ARF forest.
    This reflects which features were most useful for distinguishing real from
    synthetic data — i.e. which features carry the most structural information.
    If feature_names is given, filters to that subset; otherwise returns all features.
    """
    all_names   = arf_model.x_real.columns.tolist()
    fi          = pd.Series(arf_model.clf.feature_importances_, index=all_names)
    if feature_names is not None:
        fi = fi[feature_names]
    return fi.sort_values(ascending=False)


numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
fi = arf_feature_importance(my_arf)   # show all features in the bar chart

print("\n" + "═" * 60)
print("FEATURE IMPORTANCE  (mean impurity decrease, final ARF forest)")
print("═" * 60)
print(fi.round(4).to_string())

fig, ax = plt.subplots(figsize=(8, 4))
fi.sort_values().plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
ax.set_xlabel("Mean Impurity Decrease", fontsize=11)
ax.set_title("ARF Feature Importance\n(contribution to distinguishing real vs synthetic)", fontsize=12)
ax.axvline(fi.mean(), color="red", linestyle="--", linewidth=1, label=f"Mean = {fi.mean():.4f}")
ax.legend()
plt.tight_layout()
out_fi = os.path.join(_PROJECT_ROOT, "results", "arf_feature_importance.png")
plt.savefig(out_fi, dpi=150)
plt.close()
print(f"Saved → {out_fi}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LEAF PARTITION DIAGRAM
#     For the traced synthetic row, visualise the leaf boundary and the real
#     records that share it, in the 2D subspace of the two most important features.
# ─────────────────────────────────────────────────────────────────────────────

def plot_leaf_partition(arf_model, real_df, synth_row, feature_names,
                        tree_idx=0, top_features=None):
    """
    Plot the leaf partition for a chosen synthetic row in the 2D subspace
    of the two most important numeric features.

    Highlights:
      • Grey  points : all real training records
      • Blue  points : real records in the same leaf as the synthetic row
      • Red   star   : the synthetic row itself
      • Dashed box   : the bounding box of the leaf (min/max of real records in leaf)
    """
    final_forest  = arf_model.clf
    real_numeric  = real_df.select_dtypes(include=[np.number])
    synth_numeric = synth_row.select_dtypes(include=[np.number])
    shared_numeric = [c for c in real_numeric.columns if c in synth_numeric.columns]

    # Use top 2 numeric features by importance if not specified
    if top_features is None:
        fi = arf_feature_importance(arf_model, shared_numeric)
        top_features = fi.index[:2].tolist()

    f1, f2 = top_features[0], top_features[1]

    # Use full encoded features for leaf assignment
    X_real_enc  = arf_model.x_real
    X_synth_enc = _encode_for_clf(synth_row, arf_model)

    estimator     = final_forest.estimators_[tree_idx]
    leaf_id_real  = estimator.apply(X_real_enc)
    leaf_id_synth = estimator.apply(X_synth_enc)[0]
    in_leaf_mask  = (leaf_id_real == leaf_id_synth)

    # Build leaf bounding box from real records in the leaf
    leaf_real = real_numeric[in_leaf_mask.tolist()]
    x_min, x_max = leaf_real[f1].min(), leaf_real[f1].max()
    y_min, y_max = leaf_real[f2].min(), leaf_real[f2].max()
    # Add a small margin
    mx = (x_max - x_min) * 0.15 or 0.1
    my = (y_max - y_min) * 0.15 or 0.1

    fig, ax = plt.subplots(figsize=(8, 6))

    # All real records
    ax.scatter(real_numeric[f1], real_numeric[f2],
               c="lightgrey", s=30, zorder=1, label="All real records")

    # Real records in the same leaf
    ax.scatter(leaf_real[f1], leaf_real[f2],
               c="steelblue", s=50, zorder=2, label=f"Real records in leaf #{leaf_id_synth}")

    # Synthetic row
    sx = float(synth_row[f1].values[0])
    sy = float(synth_row[f2].values[0])
    ax.scatter(sx, sy, marker="*", c="red", s=250, zorder=3, label="Synthetic row #0")

    # Leaf bounding box
    rect = mpatches.FancyBboxPatch(
        (x_min - mx, y_min - my),
        (x_max - x_min) + 2 * mx,
        (y_max - y_min) + 2 * my,
        boxstyle="round,pad=0.01",
        linewidth=2, edgecolor="red", facecolor="none",
        linestyle="--", zorder=4
    )
    ax.add_patch(rect)

    ax.set_xlabel(f1, fontsize=11)
    ax.set_ylabel(f2, fontsize=11)
    ax.set_title(
        f"ARF Leaf Partition Diagram  —  Tree {tree_idx}, Leaf #{leaf_id_synth}\n"
        f"{in_leaf_mask.sum()} real records share this leaf with the synthetic row",
        fontsize=12
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(_PROJECT_ROOT, "results", "arf_leaf_partition.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → {out}")


plot_leaf_partition(my_arf, df, target_row, numeric_features, tree_idx=0)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  BONUS: LEAF DISTRIBUTION COMPARISON
#     For the traced leaf, plot real vs synthetic marginal distributions
#     side-by-side for the top 4 numeric features.
# ─────────────────────────────────────────────────────────────────────────────

def plot_leaf_distributions(arf_model, real_df, synth_df, feature_names,
                             tree_idx=0, top_n=4):
    """
    For the leaf that the first synthetic row falls into, compare the marginal
    distribution of real records in the leaf vs the full synthetic dataset,
    for the top_n most important features.
    """
    real_numeric  = real_df.select_dtypes(include=[np.number])
    synth_numeric = synth_df.select_dtypes(include=[np.number])
    shared_numeric = [c for c in real_numeric.columns if c in synth_numeric.columns]

    fi        = arf_feature_importance(arf_model, shared_numeric)
    top_feats = fi.index[:top_n].tolist()

    # Use full encoded features for leaf assignment
    X_real_enc     = arf_model.x_real
    X_synth_row_enc = _encode_for_clf(synth_df.iloc[[0]], arf_model)

    estimator     = arf_model.clf.estimators_[tree_idx]
    leaf_id_real  = estimator.apply(X_real_enc)
    leaf_id_synth = estimator.apply(X_synth_row_enc)[0]
    in_leaf_mask  = (leaf_id_real == leaf_id_synth)
    leaf_real     = real_numeric[in_leaf_mask.tolist()]

    fig, axes = plt.subplots(1, top_n, figsize=(4 * top_n, 4), sharey=False)
    fig.suptitle(
        f"Leaf #{leaf_id_synth} (Tree {tree_idx}) — Real vs Synthetic Marginal Distributions\n"
        f"Real records in leaf: {in_leaf_mask.sum()}  |  Total synthetic: {len(synth_df)}",
        fontsize=12
    )

    for ax, feat in zip(axes, top_feats):
        real_vals  = leaf_real[feat].dropna()
        synth_vals = synth_numeric[feat].dropna()

        ax.hist(real_vals,  bins=15, alpha=0.6, color="steelblue", label="Real (in leaf)", density=True)
        ax.hist(synth_vals, bins=15, alpha=0.5, color="tomato",    label="Synthetic (all)", density=True)
        ax.axvline(float(synth_df.iloc[0][feat]), color="darkred",
                   linestyle="--", linewidth=1.5, label="Synth row #0")
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    plt.tight_layout()
    out = os.path.join(_PROJECT_ROOT, "results", "arf_leaf_distributions.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → {out}")


plot_leaf_distributions(my_arf, df, df_synth, numeric_features, tree_idx=0, top_n=4)

print("\n✓  All explainability outputs generated successfully.")
print("   Files produced:")
print("     results/arf_feature_importance.png  — global feature importance bar chart")
print("     results/arf_leaf_partition.png      — 2D leaf boundary with real/synthetic overlap")
print("     results/arf_leaf_distributions.png  — marginal distributions inside the traced leaf")
