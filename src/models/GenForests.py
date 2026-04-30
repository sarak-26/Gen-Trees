from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from ..date_columns import finalize_synthetic_dates
except ImportError:
    from date_columns import finalize_synthetic_dates

try:
    from .preprocessing import prepare_training_dataframe
except ImportError:
    from preprocessing import prepare_training_dataframe


# -----------------------------------------------------------------------------
# Generative Forests
#
# Faithful implementation of the paper's core algorithm:
# - GF.BOOST training over a forest partition P(G)
# - choose the heaviest leaf among all trees
# - split by minimizing the partition objective in Equation (4)
# - keep only positive-empirical-mass partition cells
# - sample with INIT / STARUPDATE style stochastic descent
#
# Practical implementation details follow the appendix when specified:
# - feature domain learned from training data
# - variable types: nominal / integer / float
# - continuous cut-points are evenly spaced
# - for large categorical cardinality, evaluate a random subset of tests
#
# Missing values:
# - handled paper-faithfully: rows with a missing value for the feature
#   being split are excluded from that split's empirical mass computation.
#   They remain in the partition cell and continue to inform splits on
#   other features where they have observed values.
#   No routing, no imputation, no indicators — the model is agnostic to
#   whether missingness is random or structural.
# -----------------------------------------------------------------------------


@dataclass
class FeatureInfo:
    name: str
    kind: str  # "categorical", "int", "float"
    categories: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass
class Constraint:
    kind: str
    allowed: Optional[set] = None
    low: Optional[float] = None
    high: Optional[float] = None

    def copy(self) -> "Constraint":
        return Constraint(
            kind=self.kind,
            allowed=None if self.allowed is None else set(self.allowed),
            low=self.low,
            high=self.high,
        )

    def intersect_test(self, test: "SplitTest", go_left: bool) -> Optional["Constraint"]:
        out = self.copy()
        if self.kind == "categorical":
            left_set = set(test.left_values)
            right_set = set(test.all_values) - left_set
            chosen = left_set if go_left else right_set
            out.allowed &= chosen
            return out if out.allowed else None

        if go_left:
            out.low = max(out.low, test.threshold)
        else:
            out.high = min(out.high, test.threshold)

        if out.low >= out.high:
            return None
        return out

    def contains_value(self, value: Any) -> bool:
        if pd.isna(value):
            return False
        if self.kind == "categorical":
            return value in self.allowed
        return self.low <= float(value) < self.high

    def uniform_mass(self, feature: FeatureInfo) -> float:
        if self.kind == "categorical":
            total = len(feature.categories) if feature.categories is not None else 0
            return 0.0 if total == 0 else len(self.allowed) / total

        denom = float(feature.max_value - feature.min_value)
        if denom <= 0:
            return 1.0
        return max(0.0, (self.high - self.low) / denom)

    def sample(self, rng: np.random.Generator, feature: FeatureInfo) -> Any:
        if self.kind == "categorical":
            vals = list(self.allowed)
            return vals[int(rng.integers(0, len(vals)))]

        if self.kind == "int":
            lo = int(math.ceil(self.low))
            hi = int(math.floor(self.high - 1e-12))
            if hi < lo:
                hi = lo
            return int(rng.integers(lo, hi + 1))

        return float(rng.uniform(self.low, self.high))


@dataclass(frozen=True)
class SplitTest:
    feature_idx: int
    feature_name: str
    kind: str  # "categorical" or "numeric"
    threshold: Optional[float] = None
    left_values: Optional[Tuple[Any, ...]] = None
    all_values: Optional[Tuple[Any, ...]] = None
    # missing_go_left removed: missing rows are excluded from splits entirely,
    # so there is no direction to assign.

    def go_left(self, value: Any) -> bool:
        # Called only during STARUPDATE on fully observed constraints —
        # NaN should never reach here during generation.
        if self.kind == "categorical":
            return value in set(self.left_values)
        return float(value) >= self.threshold


@dataclass
class Node:
    node_id: int
    tree_id: int
    depth: int
    constraint_by_feature: Dict[int, Constraint]
    parent: Optional[int] = None
    split_test: Optional[SplitTest] = None
    left_id: Optional[int] = None
    right_id: Optional[int] = None
    empirical_count: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.split_test is None


@dataclass
class PartitionCell:
    leaf_ids: Tuple[int, ...]
    row_idx: np.ndarray        # all rows in this cell, including those with missing values
    constraints: Dict[int, Constraint]
    u_mass: float

    @property
    def count(self) -> int:
        return int(self.row_idx.size)


class GenerativeForest:
    def __init__(
        self,
        n_trees: int = 100,
        n_splits: int = 400,
        prior_real: float = 0.5,
        max_numeric_splits: int = 16,
        max_categorical_values: int = 22,
        max_candidate_tests: int = 1000,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        if not (0.0 < prior_real < 1.0):
            raise ValueError("prior_real must be in (0, 1).")

        self.n_trees = int(n_trees)
        self.n_splits = int(n_splits)
        self.prior_real = float(prior_real)
        self.max_numeric_splits = int(max_numeric_splits)
        self.max_categorical_values = int(max_categorical_values)
        self.max_candidate_tests = int(max_candidate_tests)
        self.verbose = bool(verbose)

        self.rng = np.random.default_rng(random_state)
        self.py_rng = random.Random(random_state)

        self.feature_info: List[FeatureInfo] = []
        self.col_names: List[str] = []
        self.X: Optional[pd.DataFrame] = None
        self.X_values: Optional[np.ndarray] = None
        self.trees: List[Dict[int, Node]] = []
        self.root_constraints: Dict[int, Constraint] = {}
        self.cells: List[PartitionCell] = []
        self.next_node_id: int = 0
        self._root_ids: List[int] = []   # cached for O(1) lookup during sampling
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame) -> "GenerativeForest":
        X = pd.DataFrame(X).copy().reset_index(drop=True)
        if len(X) == 0:
            raise ValueError("Training data is empty.")

        print(f"[GenForest] Starting fit: {len(X)} rows x {len(X.columns)} cols | "
              f"trees={self.n_trees}, splits={self.n_splits}")

        self.X = X
        self.X_values = X.to_numpy(dtype=object)
        self.col_names = list(X.columns)
        self.feature_info = self._infer_feature_info(X)
        self.root_constraints = self._make_root_constraints()
        self.trees = []
        self.cells = []
        self.next_node_id = 0
        self._root_ids = []

        n_rows = len(X)
        root_ids: List[int] = []

        for t in range(self.n_trees):
            root = self._new_node(
                tree_id=t,
                depth=0,
                constraint_by_feature=self._copy_constraints(self.root_constraints),
            )
            root.empirical_count = n_rows
            self.trees.append({root.node_id: root})
            root_ids.append(root.node_id)

        self._root_ids = root_ids   # cache here, never changes after this point

        print(f"[GenForest] Initialized {self.n_trees} trees — running GF.BOOST splits ...")

        self.cells = [
            PartitionCell(
                leaf_ids=tuple(root_ids),
                row_idx=np.arange(n_rows, dtype=np.int32),
                constraints=self._copy_constraints(self.root_constraints),
                u_mass=1.0,
            )
        ]

        log_interval = max(1, self.n_splits // 10)
        for step in range(self.n_splits):
            chosen = self._choose_heaviest_leaf()
            if chosen is None:
                print(f"[GenForest] Early stop at split {step} — no splittable leaf remaining")
                break

            tree_id, leaf_id = chosen
            best_test = self._best_split_for_leaf(tree_id, leaf_id)
            if best_test is None:
                self.trees[tree_id][leaf_id].empirical_count = -1
                continue

            self._apply_split(tree_id, leaf_id, best_test)

            if (step + 1) % log_interval == 0 or step + 1 == self.n_splits:
                print(f"[GenForest]   split {step + 1:>4}/{self.n_splits} | "
                      f"cells={len(self.cells)} | tree={tree_id} | "
                      f"feature='{best_test.feature_name}'")

            if self.verbose and ((step + 1) % 25 == 0 or step + 1 == self.n_splits):
                print(f"[GF] split {step + 1}/{self.n_splits} | cells={len(self.cells)}")

        print(f"[GenForest] Fit complete — {len(self.cells)} partition cells")
        self._fitted = True
        return self

    def sample(self, n: int) -> pd.DataFrame:
        self._check_fitted()
        if n <= 0:
            return pd.DataFrame(columns=self.col_names)

        print(f"[GenForest] Sampling {n} rows ...")
        log_interval = max(1, n // 10)
        rows = []
        for i in range(n):
            rows.append(self._sample_one_starupdate())
            if (i + 1) % log_interval == 0 or i + 1 == n:
                print(f"[GenForest]   sampled {i + 1}/{n}")
        print(f"[GenForest] Sampling complete")
        return pd.DataFrame(rows, columns=self.col_names)

    def fit_sample(self, X: pd.DataFrame, n: Optional[int] = None) -> pd.DataFrame:
        self.fit(X)
        return self.sample(len(X) if n is None else n)

    # ------------------------------------------------------------------
    # Training internals
    # ------------------------------------------------------------------
    def _infer_feature_info(self, X: pd.DataFrame) -> List[FeatureInfo]:
        infos: List[FeatureInfo] = []
        for c in X.columns:
            s = X[c]

            if (
                pd.api.types.is_bool_dtype(s)
                or pd.api.types.is_object_dtype(s)
                or isinstance(s.dtype, pd.CategoricalDtype)   # replaces deprecated is_categorical_dtype
                or pd.api.types.is_string_dtype(s)
            ):
                cats = list(pd.Series(s).dropna().astype(object).unique())
                infos.append(FeatureInfo(name=c, kind="categorical", categories=cats))
            elif pd.api.types.is_integer_dtype(s):
                nonmissing = s.dropna()
                if len(nonmissing) == 0:
                    raise ValueError(f"Column '{c}' has only missing values.")
                infos.append(
                    FeatureInfo(
                        name=c,
                        kind="int",
                        min_value=float(nonmissing.min()),
                        max_value=float(nonmissing.max()) + 1.0,
                    )
                )
            else:
                nonmissing = s.dropna()
                if len(nonmissing) == 0:
                    raise ValueError(f"Column '{c}' has only missing values.")
                lo = float(nonmissing.min())
                hi = float(nonmissing.max())
                if hi <= lo:
                    hi = lo + 1e-9
                infos.append(
                    FeatureInfo(
                        name=c,
                        kind="float",
                        min_value=lo,
                        max_value=hi,
                    )
                )
        return infos

    def _make_root_constraints(self) -> Dict[int, Constraint]:
        out: Dict[int, Constraint] = {}
        for j, info in enumerate(self.feature_info):
            if info.kind == "categorical":
                out[j] = Constraint(kind="categorical", allowed=set(info.categories))
            else:
                out[j] = Constraint(kind=info.kind, low=info.min_value, high=info.max_value)
        return out

    def _copy_constraints(self, d: Dict[int, Constraint]) -> Dict[int, Constraint]:
        return {k: v.copy() for k, v in d.items()}

    def _new_node(
        self,
        tree_id: int,
        depth: int,
        constraint_by_feature: Dict[int, Constraint],
        parent: Optional[int] = None,
    ) -> Node:
        node = Node(
            node_id=self.next_node_id,
            tree_id=tree_id,
            depth=depth,
            constraint_by_feature=constraint_by_feature,
            parent=parent,
        )
        self.next_node_id += 1
        return node

    def _choose_heaviest_leaf(self) -> Optional[Tuple[int, int]]:
        best: Optional[Tuple[int, int]] = None
        best_count = -1
        for tree_id, tree in enumerate(self.trees):
            for node_id, node in tree.items():
                if node.is_leaf and node.empirical_count > best_count:
                    best = (tree_id, node_id)
                    best_count = node.empirical_count
        return best if best_count > 0 else None

    def _bayes_risk(self, p: float) -> float:
        # Log-loss Bayes risk — matches the paper's primary example (Lemma D,
        # KL divergence bound) and the entropic losses used in CART/C4.5.
        # Strictly proper and differentiable, satisfying Theorem 5.2's conditions.
        eps = 1e-12
        p = max(eps, min(1.0 - eps, p))
        return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))

    def _cell_loss(self, r_mass: float, u_mass: float) -> float:
        m_mass = self.prior_real * r_mass + (1.0 - self.prior_real) * u_mass
        if m_mass <= 0:
            return 0.0
        p = self.prior_real * r_mass / m_mass
        return m_mass * self._bayes_risk(p)

    def _split_rows(self, row_idx: np.ndarray, test: SplitTest) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split row_idx into left and right branches for the given test.

        Missing values for the split feature are excluded from both branches.
        They remain in the parent partition cell and continue to contribute
        to splits on other features where they have observed values.
        This is the paper-faithful treatment: the empirical mass ratio R[C_left]/R[C]
        is computed only over rows where the split feature is observed.
        """
        col = self.X_values[row_idx, test.feature_idx]
        observed_mask = ~pd.isnull(col)
        observed_idx = row_idx[observed_mask]
        col_observed = col[observed_mask]

        if observed_idx.size == 0:
            return np.empty(0, dtype=row_idx.dtype), np.empty(0, dtype=row_idx.dtype)

        if test.kind == "categorical":
            left_set = set(test.left_values)
            if len(left_set) == 1:
                mask_left = col_observed == next(iter(left_set))
            else:
                mask_left = np.vectorize(lambda v: v in left_set)(col_observed)
        else:
            mask_left = col_observed.astype(float) >= test.threshold

        mask_left = mask_left.astype(bool)
        return observed_idx[mask_left], observed_idx[~mask_left]

    def _apply_split(self, tree_id: int, leaf_id: int, test: SplitTest) -> None:
        tree = self.trees[tree_id]
        leaf = tree[leaf_id]

        left_constraints = self._copy_constraints(leaf.constraint_by_feature)
        right_constraints = self._copy_constraints(leaf.constraint_by_feature)

        left_constraints[test.feature_idx] = left_constraints[test.feature_idx].intersect_test(test, True)
        right_constraints[test.feature_idx] = right_constraints[test.feature_idx].intersect_test(test, False)

        if left_constraints[test.feature_idx] is None or right_constraints[test.feature_idx] is None:
            leaf.empirical_count = -1
            return

        left = self._new_node(
            tree_id=tree_id,
            depth=leaf.depth + 1,
            constraint_by_feature=left_constraints,
            parent=leaf_id,
        )
        right = self._new_node(
            tree_id=tree_id,
            depth=leaf.depth + 1,
            constraint_by_feature=right_constraints,
            parent=leaf_id,
        )

        tree[left.node_id] = left
        tree[right.node_id] = right

        leaf.split_test = test
        leaf.left_id = left.node_id
        leaf.right_id = right.node_id

        new_cells: List[PartitionCell] = []
        left_total = 0
        right_total = 0

        for cell in self.cells:
            if cell.leaf_ids[tree_id] != leaf_id:
                new_cells.append(cell)
                continue

            left_rows, right_rows = self._split_rows(cell.row_idx, test)

            # Rows missing the split feature — keep them in both child cells
            # so they remain available for future splits on other features.
            col = self.X_values[cell.row_idx, test.feature_idx]
            missing_rows = cell.row_idx[pd.isnull(col)]

            old_constraint = cell.constraints[test.feature_idx]
            old_feature_mass = old_constraint.uniform_mass(self.feature_info[test.feature_idx])
            if old_feature_mass <= 0:
                continue

            # Left child cell: observed-left rows + missing rows
            left_all = np.concatenate([left_rows, missing_rows]) if missing_rows.size > 0 else left_rows
            if left_all.size > 0:
                cdict = self._copy_constraints(cell.constraints)
                cdict[test.feature_idx] = cdict[test.feature_idx].intersect_test(test, True)
                leaf_ids = list(cell.leaf_ids)
                leaf_ids[tree_id] = left.node_id
                left_u = cell.u_mass / old_feature_mass * cdict[test.feature_idx].uniform_mass(
                    self.feature_info[test.feature_idx]
                )
                new_cells.append(PartitionCell(tuple(leaf_ids), left_all, cdict, left_u))
                # empirical count tracks only observed rows, consistent with _cell_loss
                left_total += int(left_rows.size)

            # Right child cell: observed-right rows + missing rows
            right_all = np.concatenate([right_rows, missing_rows]) if missing_rows.size > 0 else right_rows
            if right_all.size > 0:
                cdict = self._copy_constraints(cell.constraints)
                cdict[test.feature_idx] = cdict[test.feature_idx].intersect_test(test, False)
                leaf_ids = list(cell.leaf_ids)
                leaf_ids[tree_id] = right.node_id
                right_u = cell.u_mass / old_feature_mass * cdict[test.feature_idx].uniform_mass(
                    self.feature_info[test.feature_idx]
                )
                new_cells.append(PartitionCell(tuple(leaf_ids), right_all, cdict, right_u))
                right_total += int(right_rows.size)

        self.cells = new_cells
        left.empirical_count = left_total
        right.empirical_count = right_total
        leaf.empirical_count = 0

    def _best_split_for_leaf(self, tree_id: int, leaf_id: int) -> Optional[SplitTest]:
        affected = [cell for cell in self.cells if cell.leaf_ids[tree_id] == leaf_id]
        if not affected:
            return None

        leaf = self.trees[tree_id][leaf_id]
        candidates = self._candidate_splits_for_leaf(leaf, affected)
        if not candidates:
            return None

        n_rows = len(self.X)
        # current_loss uses observed-only counts (cell.count excludes nothing here
        # at evaluation time — the exclusion happens inside _split_rows per test).
        current_loss = sum(self._cell_loss(cell.count / n_rows, cell.u_mass) for cell in affected)

        best_gain = -np.inf
        best_test: Optional[SplitTest] = None

        self.py_rng.shuffle(candidates)
        candidates = candidates[: self.max_candidate_tests]

        for test in candidates:
            new_loss = 0.0

            for cell in affected:
                # _split_rows excludes missing rows — only observed rows contribute
                # to the empirical mass ratio, keeping R honest.
                left_rows, right_rows = self._split_rows(cell.row_idx, test)

                old_constraint = cell.constraints[test.feature_idx]
                old_feature_mass = old_constraint.uniform_mass(self.feature_info[test.feature_idx])

                left_constraint = old_constraint.intersect_test(test, True)
                right_constraint = old_constraint.intersect_test(test, False)
                if left_constraint is None or right_constraint is None:
                    new_loss = np.inf
                    break

                if old_feature_mass <= 0:
                    new_loss = np.inf
                    break

                left_u = cell.u_mass / old_feature_mass * left_constraint.uniform_mass(
                    self.feature_info[test.feature_idx]
                )
                right_u = cell.u_mass / old_feature_mass * right_constraint.uniform_mass(
                    self.feature_info[test.feature_idx]
                )

                if left_rows.size > 0:
                    new_loss += self._cell_loss(left_rows.size / n_rows, left_u)
                if right_rows.size > 0:
                    new_loss += self._cell_loss(right_rows.size / n_rows, right_u)

            gain = current_loss - new_loss
            if gain > best_gain + 1e-15:
                best_gain = gain
                best_test = test

        return best_test if best_test is not None and best_gain > 1e-15 else None

    def _candidate_splits_for_leaf(self, leaf: Node, affected: List[PartitionCell]) -> List[SplitTest]:
        rows = np.unique(np.concatenate([c.row_idx for c in affected]))
        if rows.size <= 1:
            return []

        candidates: List[SplitTest] = []

        for j, info in enumerate(self.feature_info):
            cons = leaf.constraint_by_feature[j]
            col = self.X_values[rows, j]

            if info.kind == "categorical":
                active_in_col = {v for v in col if not pd.isna(v)}
                active_values = [v for v in cons.allowed if v in active_in_col]
                if len(active_values) <= 1:
                    continue

                tests = active_values
                if len(active_values) > self.max_categorical_values:
                    tests = active_values.copy()
                    self.py_rng.shuffle(tests)
                    tests = tests[: min(len(tests), self.max_candidate_tests)]

                for v in tests:
                    candidates.append(
                        SplitTest(
                            feature_idx=j,
                            feature_name=info.name,
                            kind="categorical",
                            left_values=(v,),
                            all_values=tuple(cons.allowed),
                        )
                    )
            else:
                low = cons.low
                high = cons.high
                if not (high > low):
                    continue

                thresholds = np.linspace(low, high, self.max_numeric_splits + 2)[1:-1]
                thresholds = np.unique(thresholds)

                for thr in thresholds:
                    if not (low < float(thr) < high):
                        continue
                    candidates.append(
                        SplitTest(
                            feature_idx=j,
                            feature_name=info.name,
                            kind="numeric",
                            threshold=float(thr),
                        )
                    )

        return candidates

    # _default_missing_direction removed — no longer needed.

    # ------------------------------------------------------------------
    # Sampling with INIT / STARUPDATE
    # ------------------------------------------------------------------
    def _sample_one_starupdate(self) -> Dict[str, Any]:
        n_rows = len(self.X)
        current_rows = np.arange(n_rows, dtype=np.int32)
        current_constraints = self._copy_constraints(self.root_constraints)

        # O(1) root lookup via cached _root_ids (was O(nodes) per tree per sample).
        star_node_ids = list(self._root_ids)
        done = [False] * self.n_trees

        next_tree = 0
        unfinished = self.n_trees

        while unfinished > 0:
            for _ in range(self.n_trees):
                t = next_tree
                next_tree = (next_tree + 1) % self.n_trees
                if not done[t]:
                    break
            else:
                break

            node = self.trees[t][star_node_ids[t]]
            if node.is_leaf:
                done[t] = True
                unfinished -= 1
                continue

            test = node.split_test
            left_constraint = current_constraints[test.feature_idx].intersect_test(test, True)
            right_constraint = current_constraints[test.feature_idx].intersect_test(test, False)
            if left_constraint is None or right_constraint is None:
                raise RuntimeError("Invalid split encountered during STARUPDATE sampling.")

            # During sampling there are no missing values — we are sampling from
            # constraints, not routing real data points. _split_rows correctly
            # excludes NaN but current_rows here indexes training data used only
            # for the Bernoulli probability; missing rows for this feature are
            # excluded from the ratio, which is the correct paper behaviour.
            left_rows, right_rows = self._split_rows(current_rows, test)

            observed_total = left_rows.size + right_rows.size
            if observed_total == 0:
                # No observed values for this feature in current support —
                # fall back to uniform coin flip rather than crashing.
                p_left = 0.5
            else:
                p_left = left_rows.size / observed_total

            go_left = bool(self.rng.random() < p_left)

            if go_left:
                current_rows = left_rows
                current_constraints[test.feature_idx] = left_constraint
                star_node_ids[t] = node.left_id
            else:
                current_rows = right_rows
                current_constraints[test.feature_idx] = right_constraint
                star_node_ids[t] = node.right_id

            if self.trees[t][star_node_ids[t]].is_leaf:
                done[t] = True
                unfinished -= 1

        row: Dict[str, Any] = {}
        for j, info in enumerate(self.feature_info):
            row[info.name] = current_constraints[j].sample(self.rng, info)
        return row

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("GenerativeForest is not fitted yet.")


def generate(train_data, n_generated, output_dir, *, seed: int = 42):
    print(f"[GenForest] Loading training data from '{train_data}' ...")
    df = prepare_training_dataframe(train_data)
    print(f"[GenForest] Loaded {len(df)} rows, {len(df.columns)} columns | seed={seed}")

    gf = GenerativeForest(
        n_trees=50,
        n_splits=300,
        max_numeric_splits=12,
        random_state=seed,
    )

    gf.fit(df)
    synthetic = gf.sample(n_generated)

    print(f"[GenForest] Finalising date columns ...")
    synthetic = finalize_synthetic_dates(synthetic, df)

    output_path = os.path.join('synthetic_data', f'{output_dir}')
    synthetic.to_csv(output_path, index=False)
    print(f"[GenForest] Saved {len(synthetic)} synthetic rows to '{output_path}'")


if __name__ == "__main__":
    generate('data/kaggle/ibm_hr.csv', 1500, 'kaggle_GenForest.csv')