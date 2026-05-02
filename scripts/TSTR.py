import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

SYNTH_DIR = "synthetic_data"   # directory containing synthetic CSVs
SYNTH_SEED = 42                # which seed suffix to use for synthetic files
N_SEEDS = 5                    # number of random seeds to average over
TEST_SIZE = 0.25               # fraction of real data held out for testing

DATASETS = {
    "ibm_hr": {
        "real_path": "data/ibm_hr.csv",
        "target":    "Attrition",
        "drop_cols": ["EmployeeCount", "Over18", "StandardHours"],  # zero-variance
        "task":      "classification",
    },
}

MODELS = [
    "arf",
    "GenForest",
    "ForestFlow",
    "GaussianCopula",
    "CTGAN",
    "TabDDM",
    "TVAE",
]


def load_and_clean(path: str, drop_cols: list) -> pd.DataFrame:
    df = pd.read_csv(path)
    # strip unnamed index columns
    df = df.loc[:, ~df.columns.str.lower().str.startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    # strip whitespace from string columns
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    return df


def encode_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    # unknown_value=-1 is treated as NaN by HistGradientBoostingClassifier
    cat_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    X_train = X_train.copy()
    X_test = X_test.copy()

    if cat_cols:
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=np.nan,
        )
        X_train[cat_cols] = enc.fit_transform(X_train[cat_cols].astype(str))
        X_test[cat_cols] = enc.transform(X_test[cat_cols].astype(str))

    for c in num_cols:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

    return X_train, X_test


def encode_target(y: pd.Series) -> np.ndarray:
    if y.dtype == object or y.dtype.name == "string":
        classes = sorted(y.dropna().unique())
        mapping = {c: i for i, c in enumerate(classes)}
        return y.map(mapping).to_numpy(dtype=float)
    return y.to_numpy(dtype=float)


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict:
    n_classes = len(np.unique(y_true[~np.isnan(y_true)]))
    auc = roc_auc_score(
        y_true, y_prob if n_classes == 2 else y_prob,
        multi_class="ovr", average="macro"
    )
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bac = balanced_accuracy_score(y_true, y_pred)
    return {"auc": auc, "f1": f1, "bac": bac}


def run_single(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    seed: int,
) -> dict:
    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=seed,
        early_stopping=False,
    )
    # drop rows where target is NaN (can occur in synthetic data)
    mask = ~np.isnan(y_train)
    clf.fit(X_train[mask], y_train[mask])

    y_prob = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)

    if y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]

    return evaluate(y_test, y_prob, y_pred)


def main():
    all_results = []

    for ds_key, cfg in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_key}  |  Target: {cfg['target']}")
        print(f"{'='*60}")

        real_df = load_and_clean(cfg["real_path"], cfg["drop_cols"])

        if cfg["target"] not in real_df.columns:
            print(f"  [SKIP] Target column '{cfg['target']}' not found.")
            continue

        # drop rows with missing target in real data
        real_df = real_df.dropna(subset=[cfg["target"]])

        y_real_all = encode_target(real_df[cfg["target"]])
        X_real_all = real_df.drop(columns=[cfg["target"]])

        # --- TRTR ceiling across seeds ---
        trtr_metrics = {"auc": [], "f1": [], "bac": []}

        for seed in range(N_SEEDS):
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=TEST_SIZE, random_state=seed
            )
            train_idx, test_idx = next(sss.split(X_real_all, y_real_all))

            X_tr, X_te = X_real_all.iloc[train_idx], X_real_all.iloc[test_idx]
            y_tr, y_te = y_real_all[train_idx], y_real_all[test_idx]

            X_tr_enc, X_te_enc = encode_features(X_tr, X_te)
            m = run_single(X_tr_enc, y_tr, X_te_enc, y_te, seed)
            for k in trtr_metrics:
                trtr_metrics[k].append(m[k])

        trtr_mean = {k: np.mean(v) for k, v in trtr_metrics.items()}
        print(f"\n  TRTR (ceiling): AUC={trtr_mean['auc']:.3f}  "
              f"F1={trtr_mean['f1']:.3f}  BAC={trtr_mean['bac']:.3f}")

        all_results.append({
            "dataset": ds_key,
            "model": "TRTR_ceiling",
            "auc": trtr_mean["auc"],
            "f1": trtr_mean["f1"],
            "bac": trtr_mean["bac"],
            "auc_ratio": 1.0,
            "f1_ratio": 1.0,
            "bac_ratio": 1.0,
        })

        # --- TSTR per model ---
        for model in MODELS:
            synth_fname = os.path.join(SYNTH_DIR, f"{ds_key}_{model}_seed{SYNTH_SEED}.csv")
            if not os.path.exists(synth_fname):
                print(f"  [SKIP] {synth_fname} not found.")
                continue

            synth_df = load_and_clean(synth_fname, cfg["drop_cols"])

            if cfg["target"] not in synth_df.columns:
                print(f"  [SKIP] Target not in synthetic file for {model}.")
                continue

            # align columns to real data
            shared_cols = [c for c in X_real_all.columns if c in synth_df.columns]
            X_synth = synth_df[shared_cols]
            y_synth = encode_target(synth_df[cfg["target"]])

            tstr_metrics = {"auc": [], "f1": [], "bac": []}

            for seed in range(N_SEEDS):
                # test set is always from real data, same split as TRTR
                sss = StratifiedShuffleSplit(
                    n_splits=1, test_size=TEST_SIZE, random_state=seed
                )
                _, test_idx = next(sss.split(X_real_all, y_real_all))
                X_te = X_real_all.iloc[test_idx]
                y_te = y_real_all[test_idx]

                # train on full synthetic, test on real held-out
                X_tr_enc, X_te_enc = encode_features(X_synth, X_te)
                m = run_single(X_tr_enc, y_synth, X_te_enc, y_te, seed)
                for k in tstr_metrics:
                    tstr_metrics[k].append(m[k])

            tstr_mean = {k: np.mean(v) for k, v in tstr_metrics.items()}
            ratios = {k: tstr_mean[k] / trtr_mean[k] for k in tstr_mean}

            print(f"  {model:<16} TSTR: AUC={tstr_mean['auc']:.3f}  "
                  f"F1={tstr_mean['f1']:.3f}  BAC={tstr_mean['bac']:.3f}  "
                  f"| AUC ratio={ratios['auc']:.3f}  F1 ratio={ratios['f1']:.3f}")

            all_results.append({
                "dataset": ds_key,
                "model": model,
                "auc": tstr_mean["auc"],
                "f1": tstr_mean["f1"],
                "bac": tstr_mean["bac"],
                "auc_ratio": ratios["auc"],
                "f1_ratio": ratios["f1"],
                "bac_ratio": ratios["bac"],
            })

    # --- Save results ---
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("tstr_results.csv", index=False)
    print(f"\nResults saved to tstr_results.csv")

    # --- Summary table ---
    print(f"\n{'='*60}")
    print("Summary: AUC ratio (TSTR / TRTR ceiling)")
    print(f"{'='*60}")
    pivot = results_df[results_df["model"] != "TRTR_ceiling"].pivot(
        index="model", columns="dataset", values="auc_ratio"
    )
    print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()