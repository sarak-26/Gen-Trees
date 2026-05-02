import argparse
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from sklearn.model_selection import train_test_split

from .pipeline import (
    MODEL_MODULES,
    RESULTS_DIR,
    ROOT,
    SYNTHETIC_DIR,
    _discover_datasets,
    _load_generate_function,
    _resolve_datasets,
    _resolve_models,
)


def _split_real_dataset(
    dataset_path: Path,
    holdout_fraction: float,
    random_state: int,
    split_dir: Path,
) -> tuple[Path, Path, int, int]:
    data = pd.read_csv(dataset_path)
    train_df, holdout_df = train_test_split(
        data,
        test_size=holdout_fraction,
        random_state=random_state,
        shuffle=True,
    )
    train_df = train_df.reset_index(drop=True)
    holdout_df = holdout_df.reset_index(drop=True)

    train_path = split_dir / f"{dataset_path.stem}_train.csv"
    holdout_path = split_dir / f"{dataset_path.stem}_holdout.csv"
    train_df.to_csv(train_path, index=False)
    holdout_df.to_csv(holdout_path, index=False)
    return train_path, holdout_path, len(train_df), len(holdout_df)


def _build_synthetic_filename(dataset_name: str, model_name: str) -> str:
    return f"{dataset_name}_{model_name}_privacy.csv"


def _generate_synthetic_data(
    model_name: str,
    module_path: str,
    dataset_name: str,
    train_path: Path,
    n_rows: int,
) -> Path:
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    output_name = _build_synthetic_filename(dataset_name, model_name)
    generate = _load_generate_function(module_path)
    generate(str(train_path), int(n_rows), output_name)
    return SYNTHETIC_DIR / output_name


def _run_privacy_mlflow(
    train_path: Path,
    holdout_path: Path,
    synthetic_path: Path,
    dataset_name: str,
    model_name: str,
    experiment_name: str,
    tracking_uri: str | None,
    holdout_fraction: float,
    random_state: int,
    mia_runs: int,
    mia_threshold_quantile: float,
    save_txt: bool,
    save_json: bool,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "src.evaluation.privacy_mlflow",
        "--train-csv",
        str(train_path),
        "--synthetic",
        str(synthetic_path),
        "--holdout-csv",
        str(holdout_path),
        "--dataset-name",
        dataset_name,
        "--model-name",
        model_name,
        "--run-name",
        f"{dataset_name}_{model_name}_privacy",
        "--experiment-name",
        experiment_name,
        "--holdout-fraction",
        str(holdout_fraction),
        "--random-state",
        str(random_state),
        "--mia-runs",
        str(mia_runs),
        "--mia-threshold-quantile",
        str(mia_threshold_quantile),
    ]

    if tracking_uri:
        cmd.extend(["--tracking-uri", tracking_uri])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if save_txt:
        txt_path = RESULTS_DIR / f"privacy_{dataset_name}_{model_name}.txt"
        cmd.extend(["--output", str(txt_path)])
    if save_json:
        json_path = RESULTS_DIR / f"privacy_{dataset_name}_{model_name}.json"
        cmd.extend(["--output-json", str(json_path)])

    subprocess.run(cmd, cwd=ROOT, check=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split real data, generate synthetic data from the training split, and log privacy metrics to MLflow."
    )
    parser.add_argument("--model", required=True, help="Model name or ALL.")
    parser.add_argument("--dataset", required=True, help="Dataset name, dataset path, or ALL.")
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Synthetic row count. Defaults to the number of rows in the training split.",
    )
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--mia-runs", type=int, default=20)
    parser.add_argument("--mia-threshold-quantile", type=float, default=0.05)
    parser.add_argument("--save-txt", action="store_true", help="Also save a local text report in results/.")
    parser.add_argument("--save-json", action="store_true", help="Also save a local JSON report in results/.")
    parser.add_argument(
        "--experiment-name",
        default="synthetic-data-privacy",
        help="MLflow experiment name passed through to privacy_mlflow.py.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Optional MLflow tracking URI. Defaults to sqlite:///mlflow.db via privacy_mlflow.py.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if not (0.0 < args.holdout_fraction < 1.0):
        raise SystemExit("holdout_fraction must be between 0 and 1.")
    if not (0.0 < args.mia_threshold_quantile < 1.0):
        raise SystemExit("mia_threshold_quantile must be between 0 and 1.")

    model_selections = _resolve_models(args.model)
    dataset_selections = _resolve_datasets(args.dataset)

    with TemporaryDirectory() as tmp_dir:
        split_dir = Path(tmp_dir)

        for dataset_name, dataset_path in dataset_selections:
            train_path, holdout_path, train_rows, holdout_rows = _split_real_dataset(
                dataset_path=dataset_path,
                holdout_fraction=args.holdout_fraction,
                random_state=args.random_state,
                split_dir=split_dir,
            )
            n_rows = args.rows if args.rows is not None else train_rows

            for model_name, module_path in model_selections:
                print(
                    f"\n[Privacy Pipeline] dataset={dataset_name} model={model_name} "
                    f"train_rows={train_rows} holdout_rows={holdout_rows} synth_rows={n_rows}"
                )
                synthetic_path = _generate_synthetic_data(
                    model_name=model_name,
                    module_path=module_path,
                    dataset_name=dataset_name,
                    train_path=train_path,
                    n_rows=n_rows,
                )
                _run_privacy_mlflow(
                    train_path=train_path,
                    holdout_path=holdout_path,
                    synthetic_path=synthetic_path,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    experiment_name=args.experiment_name,
                    tracking_uri=args.tracking_uri,
                    holdout_fraction=args.holdout_fraction,
                    random_state=args.random_state,
                    mia_runs=args.mia_runs,
                    mia_threshold_quantile=args.mia_threshold_quantile,
                    save_txt=args.save_txt,
                    save_json=args.save_json,
                )


if __name__ == "__main__":
    main()
