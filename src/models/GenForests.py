from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
# - supported during fitting by routing missing values to a deterministic
#   default branch chosen from observed data for the tested split.
#   The paper states training supports missing values, but does not fully
#   spell out the engineering details, so this is a standard tree-style choice.
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
    missing_go_left: bool = True

    def go_left(self, value: Any) -> bool:
        if pd.isna(value):
            return self.missing_go_left
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
    row_idx: np.ndarray
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
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame) -> "GenerativeForest":
        X = pd.DataFrame(X).copy().reset_index(drop=True)
        if len(X) == 0:
            raise ValueError("Training data is empty.")

        self.X = X
        self.X_values = X.to_numpy(dtype=object)
        self.col_names = list(X.columns)
        self.feature_info = self._infer_feature_info(X)
        self.root_constraints = self._make_root_constraints()
        self.trees = []
        self.cells = []
        self.next_node_id = 0

        root_ids: List[int] = []
        n_rows = len(X)

        for t in range(self.n_trees):
            root = self._new_node(
                tree_id=t,
                depth=0,
                constraint_by_feature=self._copy_constraints(self.root_constraints),
            )
            root.empirical_count = n_rows
            self.trees.append({root.node_id: root})
            root_ids.append(root.node_id)

        self.cells = [
            PartitionCell(
                leaf_ids=tuple(root_ids),
                row_idx=np.arange(n_rows, dtype=np.int32),
                constraints=self._copy_constraints(self.root_constraints),
                u_mass=1.0,
            )
        ]

        for step in range(self.n_splits):
            chosen = self._choose_heaviest_leaf()
            if chosen is None:
                break

            tree_id, leaf_id = chosen
            best_test = self._best_split_for_leaf(tree_id, leaf_id)
            if best_test is None:
                self.trees[tree_id][leaf_id].empirical_count = -1
                continue

            self._apply_split(tree_id, leaf_id, best_test)

            if self.verbose and ((step + 1) % 25 == 0 or step + 1 == self.n_splits):
                print(f"[GF] split {step + 1}/{self.n_splits} | cells={len(self.cells)}")

        self._fitted = True
        return self

    def sample(self, n: int) -> pd.DataFrame:
        self._check_fitted()
        if n <= 0:
            return pd.DataFrame(columns=self.col_names)

        rows = [self._sample_one_starupdate() for _ in range(n)]
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
                or pd.api.types.is_categorical_dtype(s)
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
        # Gini / square-loss Bayes risk.
        return p * (1.0 - p)

    def _cell_loss(self, r_mass: float, u_mass: float) -> float:
        m_mass = self.prior_real * r_mass + (1.0 - self.prior_real) * u_mass
        if m_mass <= 0:
            return 0.0
        p = self.prior_real * r_mass / m_mass
        return m_mass * self._bayes_risk(p)

    def _best_split_for_leaf(self, tree_id: int, leaf_id: int) -> Optional[SplitTest]:
        affected = [cell for cell in self.cells if cell.leaf_ids[tree_id] == leaf_id]
        if not affected:
            return None

        leaf = self.trees[tree_id][leaf_id]
        candidates = self._candidate_splits_for_leaf(leaf, affected)
        if not candidates:
            return None

        n_rows = len(self.X)
        current_loss = sum(self._cell_loss(cell.count / n_rows, cell.u_mass) for cell in affected)

        best_gain = -np.inf
        best_test: Optional[SplitTest] = None

        # Appendix-style implementation note: shuffle and evaluate up to
        # MAXIMAL_NUMBER_OF_SPLIT_TESTS_TRIES_PER_BOOSTING_ITERATION.
        self.py_rng.shuffle(candidates)
        candidates = candidates[: self.max_candidate_tests]

        for test in candidates:
            new_loss = 0.0

            for cell in affected:
                left_rows, right_rows = self._split_rows(cell.row_idx, test)

                old_constraint = cell.constraints[test.feature_idx]
                old_feature_mass = old_constraint.uniform_mass(self.feature_info[test.feature_idx])

                # Consistency says children partition the current support.
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

                # Keep only positive empirical mass cells, exactly as the paper notes.
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
                active_values = [v for v in cons.allowed if v in set(pd.Series(col).dropna().astype(object))]
                if len(active_values) <= 1:
                    continue

                tests = active_values
                if len(active_values) > self.max_categorical_values:
                    tests = active_values.copy()
                    self.py_rng.shuffle(tests)
                    tests = tests[: min(len(tests), self.max_candidate_tests)]

                for v in tests:
                    base_test = SplitTest(
                        feature_idx=j,
                        feature_name=info.name,
                        kind="categorical",
                        left_values=(v,),
                        all_values=tuple(cons.allowed),
                        missing_go_left=True,
                    )
                    miss_left = self._default_missing_direction(rows, base_test)
                    candidates.append(
                        SplitTest(
                            feature_idx=j,
                            feature_name=info.name,
                            kind="categorical",
                            left_values=(v,),
                            all_values=tuple(cons.allowed),
                            missing_go_left=miss_left,
                        )
                    )
            else:
                low = cons.low
                high = cons.high
                if not (high > low):
                    continue

                # Appendix: evenly spaced splits.
                thresholds = np.linspace(low, high, self.max_numeric_splits + 2)[1:-1]
                thresholds = np.unique(thresholds)

                for thr in thresholds:
                    if not (low < float(thr) < high):
                        continue
                    base_test = SplitTest(
                        feature_idx=j,
                        feature_name=info.name,
                        kind="numeric",
                        threshold=float(thr),
                        missing_go_left=True,
                    )
                    miss_left = self._default_missing_direction(rows, base_test)
                    candidates.append(
                        SplitTest(
                            feature_idx=j,
                            feature_name=info.name,
                            kind="numeric",
                            threshold=float(thr),
                            missing_go_left=miss_left,
                        )
                    )

        return candidates

    def _default_missing_direction(self, rows: np.ndarray, test: SplitTest) -> bool:
        col = self.X_values[rows, test.feature_idx]
        observed_mask = np.array([not pd.isna(v) for v in col], dtype=bool)
        if observed_mask.sum() == 0:
            return True

        observed = col[observed_mask]
        if test.kind == "categorical":
            left_set = set(test.left_values)
            go_left = np.array([v in left_set for v in observed], dtype=bool)
        else:
            vals = np.asarray(observed, dtype=float)
            go_left = vals >= test.threshold

        left_count = int(go_left.sum())
        right_count = int((~go_left).sum())
        return left_count >= right_count

    def _split_rows(self, row_idx: np.ndarray, test: SplitTest) -> Tuple[np.ndarray, np.ndarray]:
        col = self.X_values[row_idx, test.feature_idx]

        if test.kind == "categorical":
            left_set = set(test.left_values)
            mask_left = np.array(
                [test.missing_go_left if pd.isna(v) else (v in left_set) for v in col],
                dtype=bool,
            )
        else:
            mask_left = np.array(
                [test.missing_go_left if pd.isna(v) else (float(v) >= test.threshold) for v in col],
                dtype=bool,
            )

        return row_idx[mask_left], row_idx[~mask_left]

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

            old_constraint = cell.constraints[test.feature_idx]
            old_feature_mass = old_constraint.uniform_mass(self.feature_info[test.feature_idx])
            if old_feature_mass <= 0:
                continue

            if left_rows.size > 0:
                cdict = self._copy_constraints(cell.constraints)
                cdict[test.feature_idx] = cdict[test.feature_idx].intersect_test(test, True)
                leaf_ids = list(cell.leaf_ids)
                leaf_ids[tree_id] = left.node_id
                left_u = cell.u_mass / old_feature_mass * cdict[test.feature_idx].uniform_mass(
                    self.feature_info[test.feature_idx]
                )
                new_cells.append(PartitionCell(tuple(leaf_ids), left_rows, cdict, left_u))
                left_total += int(left_rows.size)

            if right_rows.size > 0:
                cdict = self._copy_constraints(cell.constraints)
                cdict[test.feature_idx] = cdict[test.feature_idx].intersect_test(test, False)
                leaf_ids = list(cell.leaf_ids)
                leaf_ids[tree_id] = right.node_id
                right_u = cell.u_mass / old_feature_mass * cdict[test.feature_idx].uniform_mass(
                    self.feature_info[test.feature_idx]
                )
                new_cells.append(PartitionCell(tuple(leaf_ids), right_rows, cdict, right_u))
                right_total += int(right_rows.size)

        self.cells = new_cells
        left.empirical_count = left_total
        right.empirical_count = right_total
        leaf.empirical_count = 0

    # ------------------------------------------------------------------
    # Sampling with INIT / STARUPDATE
    # ------------------------------------------------------------------
    def _sample_one_starupdate(self) -> Dict[str, Any]:
        n_rows = len(self.X)
        current_rows = np.arange(n_rows, dtype=np.int32)
        current_constraints = self._copy_constraints(self.root_constraints)

        star_node_ids = [self._root_id_of_tree(t) for t in range(self.n_trees)]
        done = [False] * self.n_trees

        # Any admissible sequence works; the paper states generation probability
        # does not depend on the tree-choice schedule.
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

            left_rows, right_rows = self._split_rows(current_rows, test)

            # Step 1 of STARUPDATE:
            # Bernoulli head probability is R[left ∩ C] / R[C].
            # With empirical R, this is count ratio in current_rows.
            if current_rows.size == 0:
                raise RuntimeError("STARUPDATE reached an empty empirical support.")
            p_left = left_rows.size / current_rows.size

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

    def _root_id_of_tree(self, tree_id: int) -> int:
        # roots are the unique nodes with parent=None
        for node_id, node in self.trees[tree_id].items():
            if node.parent is None:
                return node_id
        raise RuntimeError(f"Tree {tree_id} has no root.")

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("GenerativeForest is not fitted yet.")


def generate(train_data, n_generated, output_dir, *, seed: int = 42):
    df = prepare_training_dataframe(train_data)

    gf = GenerativeForest(
        n_trees=50,
        n_splits=300,
        max_numeric_splits=12,
        random_state=seed,
    )

    gf.fit(df)
    synthetic = gf.sample(n_generated)
    synthetic = finalize_synthetic_dates(synthetic, df)
    output_dir = os.path.join('synthetic_data', f'{output_dir}')
    synthetic.to_csv(output_dir, index=False)


if __name__ == "__main__":
    generate('data/kaggle/ibm_hr.csv', 1500, 'kaggle_GenForest.csv')
