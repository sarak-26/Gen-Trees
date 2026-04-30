"""
GenForest Explainability Analysis
===================================
Explainability experiments for the custom GenerativeForest implementation,
mirroring the three analyses run for ARF:

  1. Feature importance   — how often each feature is used as a split variable
                            across all trees, weighted by empirical mass of the
                            leaf being split (heavier splits = more influential).

  2. Row tracing          — re-run STARUPDATE for a chosen synthetic row and
                            record every split decision taken, producing a
                            human-readable decision path and the real training
                            records that share the final partition cell.

  3. Partition cell diagram — visualise the final partition cell (and its real
                              records) in the 2D subspace of the two most
                              important features, analogous to the ARF leaf
                              partition diagram.

Usage:
    python genforest_explainability.py

    Edit the DATA_PATH and N_SYNTH constants below to point at your dataset.

Dependencies (beyond GenForests.py):
    pip install pandas numpy matplotlib seaborn scikit-learn
"""

import math
import sys
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── import your GenerativeForest ──────────────────────────────────────────────
# Adjust the import to match how GenForests.py sits in your project.
# If this script is in the same folder as GenForests.py:
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'models')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from GenForests import GenerativeForest, FeatureInfo, Constraint, SplitTest, Node, PartitionCell

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH  = "data/ibm_hr.csv"   # path to your CSV
N_SYNTH    = 500                         # synthetic rows to generate
SYNTH_ROW  = 0                           # which synthetic row to trace
SEED       = 2026

# ─────────────────────────────────────────────────────────────────────────────
# 0. LOAD + TRAIN
# ─────────────────────────────────────────────────────────────────────────────
print("── Loading data ──")
df = pd.read_csv(DATA_PATH)

# Drop constant columns that carry no information (same as ARF experiment)
constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
if constant_cols:
    print(f"  Dropping constant columns: {constant_cols}")
    df = df.drop(columns=constant_cols)

print(f"  Dataset shape: {df.shape}")

print("\n── Fitting GenerativeForest ──")
gf = GenerativeForest(
    n_trees=50,
    n_splits=800,  
    max_numeric_splits=16, 
    prior_real=0.6,    
    random_state=SEED,
)
gf.fit(df)

print(f"\n── Generating {N_SYNTH} synthetic rows ──")
df_synth = gf.sample(N_SYNTH)
print(df_synth.head(3))


# ─────────────────────────────────────────────────────────────────────────────
# 1. FEATURE IMPORTANCE
#    For every internal (non-leaf) node across all trees, record which feature
#    was split on. Weight each split by the empirical_count of the node being
#    split, so splits on denser regions of the data count more.
#    This is directly analogous to impurity-decrease importance in sklearn.
# ─────────────────────────────────────────────────────────────────────────────

def gf_feature_importance(gf: GenerativeForest) -> pd.Series:
    """
    Compute feature importance as normalised split frequency across all trees.

    For each internal node we add 1 to the tally of the feature used for the
    split. empirical_count is zeroed out after a split is applied in GenForests,
    so weighting by it would always give zero. The final scores are normalised to sum to 1.
    """
    importance: Dict[str, float] = {info.name: 0.0 for info in gf.feature_info}

    for tree in gf.trees:
        for node in tree.values():
            if node.split_test is not None:           # internal node
                importance[node.split_test.feature_name] += 1

    total = sum(importance.values())
    if total > 0:
        importance = {k: v / total for k, v in importance.items()}

    fi = pd.Series(importance).sort_values(ascending=False)
    return fi


fi = gf_feature_importance(gf)

print("\n" + "═" * 60)
print("FEATURE IMPORTANCE  (split-frequency weighted by empirical mass)")
print("═" * 60)
print(fi.round(4).to_string())

fig, ax = plt.subplots(figsize=(8, max(4, len(fi) * 0.28)))
fi.sort_values().plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
ax.set_xlabel("Normalised Importance", fontsize=11)
ax.set_title(
    "Generative Forest Feature Importance\n"
    "(normalised split frequency across all trees)",
    fontsize=12
)
ax.axvline(fi.mean(), color="red", linestyle="--", linewidth=1,
           label=f"Mean = {fi.mean():.4f}")
ax.legend()
plt.tight_layout()
plt.savefig("results/gf_feature_importance.png", dpi=150)
plt.close()
print("Saved → results/gf_feature_importance.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. ROW TRACING
#    Re-run the STARUPDATE sampling procedure for a chosen synthetic row,
#    recording every split decision made in each tree.
#
#    The key difference vs ARF: in a GenerativeForest there is no single leaf
#    per row — instead a row is the result of n_trees sequential binary
#    decisions. The final "partition cell" is the intersection of one leaf per
#    tree. We recover this by simulating STARUPDATE deterministically given
#    the actual feature values of the target synthetic row.
# ─────────────────────────────────────────────────────────────────────────────

def _value_goes_left(test: SplitTest, value: Any) -> bool:
    """Deterministically evaluate the split direction for a known value."""
    if pd.isna(value):
        return test.missing_go_left
    if test.kind == "categorical":
        return value in set(test.left_values)
    return float(value) >= test.threshold


def trace_row(gf: GenerativeForest, synth_row: pd.Series) -> Dict:
    """
    Trace a synthetic row back through the GenerativeForest.

    Returns
    -------
    dict with keys:
        'tree_paths'   : per-tree list of (feature, direction, threshold/value, node_id)
        'final_cell'   : the PartitionCell the row belongs to
        'real_in_cell' : rows from the training data that fall in the same cell
        'constraints'  : the per-feature bounds/allowed-sets of the final cell
    """
    # Build a lookup: for each tree, map node_id -> node
    # Trace the path the row would take through each tree
    tree_paths = []
    final_leaf_ids = []

    for t, tree in enumerate(gf.trees):
        # Find root
        root_id = None
        for node_id, node in tree.items():
            if node.parent is None:
                root_id = node_id
                break

        path = []
        current_id = root_id
        while True:
            node = tree[current_id]
            if node.is_leaf:
                final_leaf_ids.append(current_id)
                break
            test = node.split_test
            val  = synth_row[test.feature_name] if test.feature_name in synth_row.index else np.nan
            go_left = _value_goes_left(test, val)

            if test.kind == "categorical":
                direction = f"== {val}"
                threshold_str = f"left_set={list(test.left_values)}"
            else:
                direction = f">= {test.threshold:.3f}" if go_left else f"< {test.threshold:.3f}"
                threshold_str = f"threshold={test.threshold:.3f}"

            path.append({
                "tree"       : t,
                "node_id"    : current_id,
                "feature"    : test.feature_name,
                "value"      : val,
                "direction"  : direction,
                "split_info" : threshold_str,
                "go_left"    : go_left,
            })

            current_id = node.left_id if go_left else node.right_id

        tree_paths.append(path)

    # Find the partition cell that matches these leaf ids
    final_leaf_tuple = tuple(final_leaf_ids)
    final_cell = None
    for cell in gf.cells:
        if cell.leaf_ids == final_leaf_tuple:
            final_cell = cell
            break

    # If no exact cell match (can happen if cell was split away), find the
    # closest cell by majority leaf overlap
    if final_cell is None:
        best_overlap = -1
        for cell in gf.cells:
            overlap = sum(a == b for a, b in zip(cell.leaf_ids, final_leaf_tuple))
            if overlap > best_overlap:
                best_overlap = overlap
                final_cell = cell

    real_in_cell = gf.X.iloc[final_cell.row_idx] if final_cell is not None else pd.DataFrame()

    return {
        "tree_paths"   : tree_paths,
        "final_cell"   : final_cell,
        "real_in_cell" : real_in_cell,
        "final_leaf_ids": final_leaf_ids,
    }


# ── Run trace on synthetic row #SYNTH_ROW ────────────────────────────────────
target_row = df_synth.iloc[SYNTH_ROW]

print("\n" + "═" * 60)
print(f"ROW TRACING  —  Synthetic row #{SYNTH_ROW}")
print("═" * 60)
print("Synthetic record values:")
print(target_row.to_string())

trace = trace_row(gf, target_row)

# Print a readable summary — show first 3 trees to keep output manageable
N_TREES_TO_SHOW = min(3, gf.n_trees)
for t_idx in range(N_TREES_TO_SHOW):
    path = trace["tree_paths"][t_idx]
    leaf_id = trace["final_leaf_ids"][t_idx]
    print(f"\n  Tree {t_idx}  →  reached leaf node #{leaf_id}  ({len(path)} splits traversed)")
    for step in path:
        print(f"    [{step['node_id']}] {step['feature']} = {step['value']}  →  {step['direction']}  ({step['split_info']})")

cell = trace["final_cell"]
real_in_cell = trace["real_in_cell"]
print(f"\n  Final partition cell:")
print(f"    Real training records in cell : {len(real_in_cell)}")
print(f"    Uniform mass of cell          : {cell.u_mass:.6f}" if cell else "    No cell found.")

if len(real_in_cell) > 0:
    print("\n  Summary of real records in the same partition cell:")
    numeric_real = real_in_cell.select_dtypes(include=[np.number])
    print(numeric_real.describe().round(3).to_string())

# Print the final constraints (bounds) on numeric features for the cell
if cell is not None:
    print("\n  Feature constraints of the final partition cell (numeric features):")
    for j, info in enumerate(gf.feature_info):
        if info.kind in ("int", "float"):
            c = cell.constraints[j]
            print(f"    {info.name:30s}  [{c.low:.3f},  {c.high:.3f})")
        else:
            c = cell.constraints[j]
            vals = sorted(str(v) for v in c.allowed)
            if len(vals) <= 6:
                print(f"    {info.name:30s}  {{{', '.join(vals)}}}")
            else:
                print(f"    {info.name:30s}  {len(vals)} allowed values")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PARTITION CELL DIAGRAM
#    Visualise the final partition cell in the 2D subspace of the two most
#    important features, showing all real records, real records in the cell,
#    and the synthetic row itself. The dashed box marks the cell's constraint
#    bounds for those two features — directly analogous to the ARF leaf diagram.
# ─────────────────────────────────────────────────────────────────────────────

def plot_partition_cell(
    gf: GenerativeForest,
    fi: pd.Series,
    real_df: pd.DataFrame,
    synth_row: pd.Series,
    trace: Dict,
    top_features: Optional[List[str]] = None,
):
    """
    Plot the partition cell for the traced synthetic row in 2D.

    Uses the two highest-importance numeric features as axes.
    The dashed red box shows the constraint bounds of the final partition cell
    on those two features.
    """
    # Pick two numeric features with highest importance
    numeric_feature_names = [
        info.name for info in gf.feature_info if info.kind in ("int", "float")
    ]
    if top_features is None:
        ranked = [f for f in fi.index if f in numeric_feature_names]
        top_features = ranked[:2]

    f1, f2 = top_features[0], top_features[1]
    cell = trace["final_cell"]
    real_in_cell = trace["real_in_cell"]

    # Get cell bounds for the two axis features
    j1 = gf.col_names.index(f1)
    j2 = gf.col_names.index(f2)
    c1 = cell.constraints[j1]
    c2 = cell.constraints[j2]

    fig, ax = plt.subplots(figsize=(9, 7))

    # All real records
    ax.scatter(real_df[f1], real_df[f2],
               c="lightgrey", s=30, zorder=1, label="All real records")

    # Real records in the same partition cell
    if len(real_in_cell) > 0:
        ax.scatter(real_in_cell[f1], real_in_cell[f2],
                   c="steelblue", s=60, zorder=2,
                   label=f"Real records in partition cell (n={len(real_in_cell)})")

    # Synthetic row
    sx = float(synth_row[f1]) if f1 in synth_row.index else np.nan
    sy = float(synth_row[f2]) if f2 in synth_row.index else np.nan
    ax.scatter(sx, sy, marker="*", c="red", s=300, zorder=3,
               label=f"Synthetic row #{SYNTH_ROW}")

    # Cell bounding box from constraints
    x_lo, x_hi = c1.low, c1.high
    y_lo, y_hi = c2.low, c2.high
    mx = max((x_hi - x_lo) * 0.08, 0.05)
    my = max((y_hi - y_lo) * 0.08, 0.05)

    rect = mpatches.FancyBboxPatch(
        (x_lo - mx, y_lo - my),
        (x_hi - x_lo) + 2 * mx,
        (y_hi - y_lo) + 2 * my,
        boxstyle="round,pad=0.01",
        linewidth=2, edgecolor="red", facecolor="none",
        linestyle="--", zorder=4,
    )
    ax.add_patch(rect)

    ax.set_xlabel(f1, fontsize=11)
    ax.set_ylabel(f2, fontsize=11)
    ax.set_title(
        f"Generative Forest Partition Cell Diagram\n"
        f"{len(real_in_cell)} real records share this cell with synthetic row #{SYNTH_ROW}",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("results/gf_partition_cell.png", dpi=150)
    plt.close()
    print("Saved → results/gf_partition_cell.png")


plot_partition_cell(gf, fi, df, target_row, trace)


# ─────────────────────────────────────────────────────────────────────────────
# 4. DECISION PATH SUMMARY PLOT
#    For the traced row, draw a compact horizontal diagram showing which
#    feature was used at each split step across the first N trees.
#    This makes the multi-tree intersection structure of GenForests visible.
# ─────────────────────────────────────────────────────────────────────────────

def plot_decision_paths(trace: Dict, n_trees: int = 5, max_depth: int = 8):
    """
    Horizontal bar chart showing the sequence of features split on for each
    tree during STARUPDATE for the traced row. Each row = one tree, each
    column = one split step. Cells are coloured by the feature used.
    """
    tree_paths = trace["tree_paths"][:n_trees]
    all_features = sorted({step["feature"] for path in tree_paths for step in path})
    cmap   = plt.cm.get_cmap("tab20", len(all_features))
    f_color = {f: cmap(i) for i, f in enumerate(all_features)}

    fig, ax = plt.subplots(figsize=(max(8, max_depth * 1.2), n_trees * 0.7 + 1.5))
    ax.set_xlim(-0.5, max_depth - 0.5)
    ax.set_ylim(-0.5, n_trees - 0.5)

    for t_idx, path in enumerate(tree_paths):
        for depth, step in enumerate(path[:max_depth]):
            feat  = step["feature"]
            color = f_color[feat]
            rect  = mpatches.FancyBboxPatch(
                (depth - 0.45, t_idx - 0.4), 0.9, 0.8,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor="white", linewidth=0.8
            )
            ax.add_patch(rect)
            ax.text(depth, t_idx, feat[:12], ha="center", va="center",
                    fontsize=6.5, color="white", fontweight="bold")

    ax.set_yticks(range(n_trees))
    ax.set_yticklabels([f"Tree {i}" for i in range(n_trees)], fontsize=9)
    ax.set_xticks(range(max_depth))
    ax.set_xticklabels([f"Split {i+1}" for i in range(max_depth)], fontsize=8, rotation=30)
    ax.set_title(
        f"Decision Path per Tree  —  Synthetic Row #{SYNTH_ROW}\n"
        f"(each cell = feature split on at that depth)",
        fontsize=12
    )

    # Legend
    handles = [mpatches.Patch(color=f_color[f], label=f) for f in all_features]
    ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=7, title="Feature")

    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("results/gf_decision_paths.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → results/gf_decision_paths.png")


plot_decision_paths(trace, n_trees=min(8, gf.n_trees), max_depth=8)


print("\n✓  All GenForest explainability outputs generated.")
print("   Files produced:")
print("     results/gf_feature_importance.png  — global feature importance bar chart")
print("     results/gf_partition_cell.png      — 2D partition cell boundary diagram")
print("     results/gf_decision_paths.png      — per-tree decision path heatmap")
