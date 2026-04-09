import argparse
import json
import math
import re
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    from .privacy_eval import format_privacy_results, run_privacy_evaluation
except ImportError:
    from privacy_eval import format_privacy_results, run_privacy_evaluation


def _require_mlflow():
    try:
        import mlflow
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "mlflow is not installed in the active environment. "
            "Install it first, for example: ./.venv/bin/pip install mlflow"
        ) from exc
    return mlflow


def _set_or_restore_experiment(mlflow, experiment_name: str) -> None:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None and getattr(experiment, "lifecycle_stage", None) == "deleted":
        client = mlflow.tracking.MlflowClient()
        client.restore_experiment(experiment.experiment_id)
    mlflow.set_experiment(experiment_name)


def _sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return sanitized.strip("._-") or "unknown"


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _collect_summary_metrics(results: dict) -> dict[str, float]:
    metrics = {}
    candidates = {
        "exact_match_train.count": results.get("exact_match_train_vs_synth", {}).get("count"),
        "exact_match_train.share_of_synth": results.get("exact_match_train_vs_synth", {}).get("share_of_synth"),
        "exact_match_train.share_of_real": results.get("exact_match_train_vs_synth", {}).get("share_of_real"),
        "exact_match_holdout.count": results.get("exact_match_holdout_vs_synth", {}).get("count"),
        "exact_match_holdout.share_of_synth": results.get("exact_match_holdout_vs_synth", {}).get("share_of_synth"),
        "exact_match_holdout.share_of_real": results.get("exact_match_holdout_vs_synth", {}).get("share_of_real"),
        "dcr_train_to_synth.mean": results.get("dcr_train_to_synth", {}).get("mean"),
        "dcr_train_to_synth.median": results.get("dcr_train_to_synth", {}).get("median"),
        "dcr_train_to_synth.min": results.get("dcr_train_to_synth", {}).get("min"),
        "dcr_holdout_to_synth.mean": results.get("dcr_holdout_to_synth", {}).get("mean"),
        "dcr_holdout_to_synth.median": results.get("dcr_holdout_to_synth", {}).get("median"),
        "dcr_holdout_to_synth.min": results.get("dcr_holdout_to_synth", {}).get("min"),
        "dcr_train_to_train.mean": results.get("dcr_train_to_train", {}).get("mean"),
        "dcr_train_to_train.median": results.get("dcr_train_to_train", {}).get("median"),
        "dcr_train_to_train.min": results.get("dcr_train_to_train", {}).get("min"),
        "share_synth_closer_to_train_than_holdout": results.get("share_synth_closer_to_train_than_holdout"),
        "nndr_vs_train.mean": results.get("nndr_vs_train", {}).get("mean"),
        "nndr_vs_train.median": results.get("nndr_vs_train", {}).get("median"),
        "nndr_vs_train.min": results.get("nndr_vs_train", {}).get("min"),
        "nndr_vs_holdout.mean": results.get("nndr_vs_holdout", {}).get("mean"),
        "nndr_vs_holdout.median": results.get("nndr_vs_holdout", {}).get("median"),
        "nndr_vs_holdout.min": results.get("nndr_vs_holdout", {}).get("min"),
        "mia.avg_accuracy": results.get("distance_based_mia", {}).get("avg_accuracy"),
        "mia.avg_precision": results.get("distance_based_mia", {}).get("avg_precision"),
        "mia.avg_recall": results.get("distance_based_mia", {}).get("avg_recall"),
        "mia.avg_tpr": results.get("distance_based_mia", {}).get("avg_tpr"),
        "mia.avg_fpr": results.get("distance_based_mia", {}).get("avg_fpr"),
        "mia.avg_threshold": results.get("distance_based_mia", {}).get("avg_threshold"),
        "rows.training_csv": results.get("data_summary", {}).get("training_csv_rows"),
        "rows.synthetic_csv": results.get("data_summary", {}).get("synthetic_csv_rows"),
        "rows.real_train": results.get("data_summary", {}).get("real_train_rows"),
        "rows.real_holdout": results.get("data_summary", {}).get("real_holdout_rows"),
        "columns.count": results.get("data_summary", {}).get("num_columns"),
    }

    for key, value in candidates.items():
        if _is_finite_number(value):
            metrics[key] = float(value)
    return metrics


def _log_metrics(mlflow, metrics: dict[str, float], batch_size: int = 100) -> None:
    items = list(metrics.items())
    for start in range(0, len(items), batch_size):
        mlflow.log_metrics(dict(items[start:start + batch_size]))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run privacy evaluation and log results to MLflow.")
    parser.add_argument("--train-csv", required=True, help="Path to the real training CSV file used for privacy evaluation.")
    parser.add_argument("--synthetic", required=True, help="Path to the synthetic CSV file.")
    parser.add_argument("--holdout-csv", default=None, help="Optional external holdout CSV file.")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--mia-runs", type=int, default=20)
    parser.add_argument("--mia-threshold-quantile", type=float, default=0.05)
    parser.add_argument("--experiment-name", default="synthetic-data-privacy")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--tracking-uri", default=None, help="Example: sqlite:///mlflow.db")
    parser.add_argument(
        "--artifact-subdir",
        default="privacy_evaluation",
        help="Artifact subdirectory inside the MLflow run.",
    )
    parser.add_argument("--output", default=None, help="Optional local text report path.")
    parser.add_argument("--output-json", default=None, help="Optional local JSON report path.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    mlflow = _require_mlflow()
    tracking_uri = args.tracking_uri or f"sqlite:///{Path('mlflow.db').resolve()}"
    mlflow.set_tracking_uri(tracking_uri)
    _set_or_restore_experiment(mlflow, args.experiment_name)

    model_name = args.model_name or _sanitize_name(Path(args.synthetic).stem)
    dataset_name = args.dataset_name or _sanitize_name(Path(args.train_csv).stem)
    run_name = args.run_name or f"{dataset_name}_{model_name}_privacy"

    results = run_privacy_evaluation(
        train_csv=args.train_csv,
        synth_csv=args.synthetic,
        holdout_csv=args.holdout_csv,
        holdout_fraction=args.holdout_fraction,
        random_state=args.random_state,
        mia_runs=args.mia_runs,
        mia_threshold_quantile=args.mia_threshold_quantile,
    )
    results["metadata"] = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "run_name": run_name,
        "experiment_name": args.experiment_name,
    }
    results["paths"] = {
        "train": args.train_csv,
        "holdout": args.holdout_csv,
        "synthetic": args.synthetic,
    }
    report_text = format_privacy_results(results)
    summary_metrics = _collect_summary_metrics(results)

    params = {
        "train_csv": args.train_csv,
        "synthetic_path": args.synthetic,
        "holdout_csv": "None" if args.holdout_csv is None else args.holdout_csv,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "holdout_fraction": args.holdout_fraction,
        "random_state": args.random_state,
        "mia_runs": args.mia_runs,
        "mia_threshold_quantile": args.mia_threshold_quantile,
    }
    tags = {
        "task": "synthetic_data_privacy_evaluation",
        "model_name": model_name,
        "dataset_name": dataset_name,
    }

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        _log_metrics(mlflow, summary_metrics)

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            txt_path = tmp_path / "privacy_evaluation.txt"
            json_path = tmp_path / "privacy_evaluation.json"
            meta_path = tmp_path / "run_metadata.json"

            txt_path.write_text(report_text, encoding="utf-8")
            json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            meta_path.write_text(
                json.dumps(
                    {
                        "model_name": model_name,
                        "dataset_name": dataset_name,
                        "tracking_uri": tracking_uri,
                        "experiment_name": args.experiment_name,
                        "run_name": run_name,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            mlflow.log_artifact(str(txt_path), artifact_path=args.artifact_subdir)
            mlflow.log_artifact(str(json_path), artifact_path=args.artifact_subdir)
            mlflow.log_artifact(str(meta_path), artifact_path=args.artifact_subdir)

    if args.output:
        Path(args.output).write_text(report_text, encoding="utf-8")
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(report_text)
    print("")
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Run name: {run_name}")


if __name__ == "__main__":
    main()
