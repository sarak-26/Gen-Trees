"""
python src/evaluation/privacy_eval.py \
  --train_csv data/kaggle/ibm_hr.csv \
  --synth_csv synthetic_data/kaggle_arf.csv \
  --output_txt results/privacy_results_kaggle_arf.txt
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================================================
# Helpers
# =========================================================

def split_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def make_preprocessor(df_reference: pd.DataFrame) -> ColumnTransformer:
    numeric_cols, categorical_cols = split_column_types(df_reference)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def align_columns(train_df: pd.DataFrame, synth_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    missing_in_synth = [c for c in train_df.columns if c not in synth_df.columns]
    extra_in_synth = [c for c in synth_df.columns if c not in train_df.columns]

    if missing_in_synth:
        raise ValueError(f"Synthetic CSV is missing columns present in training CSV: {missing_in_synth}")

    if extra_in_synth:
        synth_df = synth_df.drop(columns=extra_in_synth)

    synth_df = synth_df[train_df.columns].copy()
    return train_df.copy(), synth_df


def nearest_neighbor_distances(X_query: np.ndarray, X_reference: np.ndarray, k: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X_reference)
    distances, _ = nn.kneighbors(X_query)
    return distances


# =========================================================
# Metrics
# =========================================================

def exact_match_count(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> Dict[str, float]:
    real_records = Counter(map(tuple, real_df.astype(object).to_numpy()))
    synth_records = Counter(map(tuple, synth_df.astype(object).to_numpy()))

    matches = 0
    for row, synth_count in synth_records.items():
        matches += min(synth_count, real_records.get(row, 0))

    return {
        "count": int(matches),
        "share_of_synth": float(matches / max(len(synth_df), 1)),
        "share_of_real": float(matches / max(len(real_df), 1)),
    }


def dcr_summary(X_query: np.ndarray, X_reference: np.ndarray, percentile: float = 5.0) -> Dict[str, float]:
    dists = nearest_neighbor_distances(X_query, X_reference, k=1).ravel()
    return {
        "mean": float(np.mean(dists)),
        "median": float(np.median(dists)),
        "min": float(np.min(dists)),
        f"p{int(percentile)}": float(np.percentile(dists, percentile)),
    }


def train_train_dcr(X_train: np.ndarray, percentile: float = 5.0) -> Dict[str, float]:
    dists = nearest_neighbor_distances(X_train, X_train, k=2)[:, 1]
    return {
        "mean": float(np.mean(dists)),
        "median": float(np.median(dists)),
        "min": float(np.min(dists)),
        f"p{int(percentile)}": float(np.percentile(dists, percentile)),
    }


def share_synth_closer_to_train_than_holdout(
    X_synth: np.ndarray,
    X_train: np.ndarray,
    X_holdout: np.ndarray,
) -> float:
    d_train = nearest_neighbor_distances(X_synth, X_train, k=1).ravel()
    d_holdout = nearest_neighbor_distances(X_synth, X_holdout, k=1).ravel()
    return float(np.mean(d_train < d_holdout))


def nndr_against_reference(
    X_query: np.ndarray,
    X_reference: np.ndarray,
    percentile: float = 5.0,
) -> Dict[str, float]:
    nn1 = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn1.fit(X_reference)
    d1, nn1_idx = nn1.kneighbors(X_query)
    d1 = d1.ravel()
    nn1_idx = nn1_idx.ravel()

    nn2 = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn2.fit(X_reference)
    d2_all = nn2.kneighbors(X_reference)[0]
    d2 = d2_all[nn1_idx, 1]

    eps = 1e-12
    ratios = d1 / np.maximum(d2, eps)

    return {
        "mean": float(np.mean(ratios)),
        "median": float(np.median(ratios)),
        "min": float(np.min(ratios)),
        f"p{int(percentile)}": float(np.percentile(ratios, percentile)),
    }


@dataclass
class MIAResult:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    true_positive_rate: float
    false_positive_rate: float
    tp: int
    fp: int
    tn: int
    fn: int


def distance_based_mia(
    X_train: np.ndarray,
    X_holdout: np.ndarray,
    X_synth: np.ndarray,
    threshold_quantile: float = 0.05,
    random_state: int = 42,
) -> MIAResult:
    rng = np.random.default_rng(random_state)

    attack_size = min(len(X_train), len(X_holdout))
    train_idx = rng.choice(len(X_train), size=attack_size, replace=False)
    holdout_idx = rng.choice(len(X_holdout), size=attack_size, replace=False)

    X_attack_train = X_train[train_idx]
    X_attack_holdout = X_holdout[holdout_idx]

    train_dists = nearest_neighbor_distances(X_attack_train, X_synth, k=1).ravel()
    holdout_dists = nearest_neighbor_distances(X_attack_holdout, X_synth, k=1).ravel()

    threshold = float(np.quantile(train_dists, threshold_quantile))

    train_pred_member = train_dists < threshold
    holdout_pred_member = holdout_dists < threshold

    tp = int(np.sum(train_pred_member))
    fn = int(np.sum(~train_pred_member))
    fp = int(np.sum(holdout_pred_member))
    tn = int(np.sum(~holdout_pred_member))

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    tpr = recall
    fpr = fp / max(fp + tn, 1)

    return MIAResult(
        threshold=threshold,
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        true_positive_rate=float(tpr),
        false_positive_rate=float(fpr),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def repeated_distance_based_mia(
    X_train: np.ndarray,
    X_holdout: np.ndarray,
    X_synth: np.ndarray,
    n_runs: int = 20,
    threshold_quantile: float = 0.05,
    random_state: int = 42,
) -> Dict[str, float]:
    results = []
    for i in range(n_runs):
        results.append(
            distance_based_mia(
                X_train=X_train,
                X_holdout=X_holdout,
                X_synth=X_synth,
                threshold_quantile=threshold_quantile,
                random_state=random_state + i,
            )
        )

    return {
        "avg_accuracy": float(np.mean([r.accuracy for r in results])),
        "avg_precision": float(np.mean([r.precision for r in results])),
        "avg_recall": float(np.mean([r.recall for r in results])),
        "avg_tpr": float(np.mean([r.true_positive_rate for r in results])),
        "avg_fpr": float(np.mean([r.false_positive_rate for r in results])),
        "avg_threshold": float(np.mean([r.threshold for r in results])),
    }


# =========================================================
# Main evaluation
# =========================================================

def format_privacy_results(results: Dict[str, object]) -> str:
    interpretation = []
    metadata = results.get("metadata", {})
    paths = results.get("paths", {})

    if metadata or paths:
        interpretation.append("Privacy Evaluation")
        interpretation.append("=" * 80)
        if metadata.get("dataset_name"):
            interpretation.append(f"Dataset: {metadata['dataset_name']}")
        if metadata.get("model_name"):
            interpretation.append(f"Model: {metadata['model_name']}")
        if paths.get("train"):
            interpretation.append(f"Train CSV: {paths['train']}")
        if paths.get("holdout"):
            interpretation.append(f"Holdout CSV: {paths['holdout']}")
        if paths.get("synthetic"):
            interpretation.append(f"Synthetic CSV: {paths['synthetic']}")
        interpretation.append("")

    interpretation.append("Privacy Metric Interpretation")
    interpretation.append("=" * 80)
    interpretation.append("")
    interpretation.append(
        f"Exact matches (train vs synth): {results['exact_match_train_vs_synth']['count']} "
        f"(lower is better, ideally 0)"
    )
    interpretation.append(
        f"Exact matches (holdout vs synth): {results['exact_match_holdout_vs_synth']['count']}"
    )
    interpretation.append("")
    interpretation.append(
        f"Median DCR train-train:  {results['dcr_train_to_train']['median']:.6f}"
    )
    interpretation.append(
        f"Median DCR train-synth:  {results['dcr_train_to_synth']['median']:.6f}"
    )
    interpretation.append(
        f"Median DCR holdout-synth:{results['dcr_holdout_to_synth']['median']:.6f}"
    )
    interpretation.append(
        "Interpretation: train-synth distances should generally not be unusually smaller "
        "than holdout-synth distances."
    )
    interpretation.append("")
    interpretation.append(
        f"Share synth closer to train than holdout: "
        f"{results['share_synth_closer_to_train_than_holdout']:.6f}"
    )
    interpretation.append(
        "Interpretation: values near 0.5 are usually more reassuring; much higher values "
        "can indicate training-specific leakage."
    )
    interpretation.append("")
    interpretation.append(
        f"Median NNDR vs train:   {results['nndr_vs_train']['median']:.6f}"
    )
    interpretation.append(
        f"Median NNDR vs holdout: {results['nndr_vs_holdout']['median']:.6f}"
    )
    interpretation.append(
        "Interpretation: higher NNDR is generally better; very low values may indicate "
        "synthetic points lying unusually close to specific real records or outliers."
    )
    interpretation.append("")
    interpretation.append(
        f"MIA average accuracy:  {results['distance_based_mia']['avg_accuracy']:.6f}"
    )
    interpretation.append(
        f"MIA average precision: {results['distance_based_mia']['avg_precision']:.6f}"
    )
    interpretation.append(
        "Interpretation: values near 0.5 suggest attack performance close to chance."
    )
    interpretation.append("")
    return "\n".join(interpretation)


def run_privacy_evaluation(
    train_csv: str,
    synth_csv: str,
    holdout_csv: str | None = None,
    holdout_fraction: float = 0.2,
    random_state: int = 42,
    mia_runs: int = 20,
    mia_threshold_quantile: float = 0.05,
) -> Dict[str, object]:
    train_df = pd.read_csv(train_csv)
    synth_df = pd.read_csv(synth_csv)

    train_df, synth_df = align_columns(train_df, synth_df)

    if holdout_csv is not None:
        holdout_df = pd.read_csv(holdout_csv)
        train_df, holdout_df = align_columns(train_df, holdout_df)
        real_train_df = train_df
        real_holdout_df = holdout_df
    else:
        # Create internal train/holdout split from real data
        real_train_df, real_holdout_df = train_test_split(
            train_df,
            test_size=holdout_fraction,
            random_state=random_state,
            shuffle=True,
        )

    real_train_df = real_train_df.reset_index(drop=True)
    real_holdout_df = real_holdout_df.reset_index(drop=True)
    synth_df = synth_df.reset_index(drop=True)

    # Fit preprocessing on real data only
    real_all_df = pd.concat([real_train_df, real_holdout_df], axis=0, ignore_index=True)
    preprocessor = make_preprocessor(real_all_df)
    preprocessor.fit(real_all_df)

    X_train = preprocessor.transform(real_train_df)
    X_holdout = preprocessor.transform(real_holdout_df)
    X_synth = preprocessor.transform(synth_df)

    results = {
        "data_summary": {
            "training_csv_rows": int(len(train_df)),
            "synthetic_csv_rows": int(len(synth_df)),
            "num_columns": int(train_df.shape[1]),
            "holdout_fraction": float(holdout_fraction),
            "real_train_rows": int(len(real_train_df)),
            "real_holdout_rows": int(len(real_holdout_df)),
        },
        "exact_match_train_vs_synth": exact_match_count(real_train_df, synth_df),
        "exact_match_holdout_vs_synth": exact_match_count(real_holdout_df, synth_df),
        "dcr_train_to_synth": dcr_summary(X_synth, X_train),
        "dcr_holdout_to_synth": dcr_summary(X_synth, X_holdout),
        "dcr_train_to_train": train_train_dcr(X_train),
        "share_synth_closer_to_train_than_holdout": share_synth_closer_to_train_than_holdout(
            X_synth, X_train, X_holdout
        ),
        "nndr_vs_train": nndr_against_reference(X_synth, X_train),
        "nndr_vs_holdout": nndr_against_reference(X_synth, X_holdout),
        "distance_based_mia": repeated_distance_based_mia(
            X_train=X_train,
            X_holdout=X_holdout,
            X_synth=X_synth,
            n_runs=mia_runs,
            threshold_quantile=mia_threshold_quantile,
            random_state=random_state,
        ),
    }

    return results


def evaluate_privacy(
    train_csv: str,
    synth_csv: str,
    output_txt: str,
    holdout_csv: str | None = None,
    holdout_fraction: float = 0.2,
    random_state: int = 42,
    mia_runs: int = 20,
    mia_threshold_quantile: float = 0.05,
) -> None:
    results = run_privacy_evaluation(
        train_csv=train_csv,
        synth_csv=synth_csv,
        holdout_csv=holdout_csv,
        holdout_fraction=holdout_fraction,
        random_state=random_state,
        mia_runs=mia_runs,
        mia_threshold_quantile=mia_threshold_quantile,
    )
    interpretation = format_privacy_results(results)

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("SYNTHETIC DATA PRIVACY EVALUATION\n")
        f.write("=" * 80 + "\n\n")
        f.write("RAW RESULTS (JSON)\n")
        f.write("-" * 80 + "\n")
        f.write(json.dumps(results, indent=2))
        f.write("\n\n")
        f.write(interpretation)
        f.write("\n")

    print(f"Done. Results written to: {output_txt}")


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate privacy metrics for synthetic tabular data.")
    parser.add_argument("--train_csv", required=True, help="Path to the real training CSV file")
    parser.add_argument("--synth_csv", required=True, help="Path to the synthetic CSV file")
    parser.add_argument("--output_txt", required=True, help="Path to output TXT file")
    parser.add_argument("--holdout_csv", default=None, help="Optional external holdout CSV file")
    parser.add_argument("--holdout_fraction", type=float, default=0.2, help="Fraction of training data used as holdout")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--mia_runs", type=int, default=20, help="Number of repeated MIA runs")
    parser.add_argument("--mia_threshold_quantile", type=float, default=0.05, help="Threshold quantile for MIA")

    args = parser.parse_args()

    if not (0.0 < args.holdout_fraction < 1.0):
        raise ValueError("holdout_fraction must be between 0 and 1.")

    if not (0.0 < args.mia_threshold_quantile < 1.0):
        raise ValueError("mia_threshold_quantile must be between 0 and 1.")

    evaluate_privacy(
        train_csv=args.train_csv,
        synth_csv=args.synth_csv,
        output_txt=args.output_txt,
        holdout_csv=args.holdout_csv,
        holdout_fraction=args.holdout_fraction,
        random_state=args.random_state,
        mia_runs=args.mia_runs,
        mia_threshold_quantile=args.mia_threshold_quantile,
    )


if __name__ == "__main__":
    main()
