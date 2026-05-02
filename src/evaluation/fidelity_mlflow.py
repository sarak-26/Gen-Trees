import argparse
import json
import math
import re
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    from .fidelity import EvalConfig, aggregate_evaluation_results, format_results, run_evaluation
except ImportError:
    from fidelity import EvalConfig, aggregate_evaluation_results, format_results, run_evaluation


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


def _infer_model_name(synthetic_path: str) -> str:
    return _sanitize_name(Path(synthetic_path).stem)


def _infer_dataset_name(real_path: str) -> str:
    return _sanitize_name(Path(real_path).stem)


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _collect_summary_metrics(results: dict) -> dict[str, float]:
    metrics = {}

    discriminator = results.get("discriminator", {})
    support = results.get("support", {})
    utility = results.get("global_utility", {})

    summary_candidates = {
        "discriminator.auc": discriminator.get("auc"),
        "discriminator.accuracy": discriminator.get("accuracy"),
        "discriminator.pmse_test": discriminator.get("pmse_test"),
        "discriminator.pmse_null_mean": discriminator.get("pmse_null_mean"),
        "discriminator.pmse_ratio_perm": discriminator.get("pmse_ratio_perm"),
        "discriminator.pmse_null_percentile": discriminator.get("pmse_null_percentile"),
        "support.alpha_precision": support.get("alpha_precision"),
        "support.beta_recall": support.get("beta_recall"),
        "support.IP_alpha": support.get("IP_alpha"),
        "support.IR_beta": support.get("IR_beta"),
        "support.embedder_radius": support.get("embedder", {}).get("radius"),
        "global_utility.score": utility.get("global_utility"),
        "global_utility.num_columns_used": utility.get("num_columns_used"),
        "columns.numeric_count": len(results.get("columns", {}).get("numeric", [])),
        "columns.categorical_count": len(results.get("columns", {}).get("categorical", [])),
        "columns.discrete_ordinal_count": len(results.get("columns", {}).get("discrete_ordinal", [])),
        "columns.dropped_count": len(results.get("dropped_columns", [])),
        "encoding.one_hot_feature_count": results.get("encoding", {}).get("one_hot_feature_count"),
    }

    for key, value in summary_candidates.items():
        if _is_finite_number(value):
            metrics[key] = float(value)

    return metrics


def _collect_per_column_metrics(results: dict) -> dict[str, float]:
    metrics = {}

    for column, values in results.get("marginal_numeric", {}).items():
        column_name = _sanitize_name(column)
        ks = values.get("ks")
        wasserstein = values.get("wasserstein")
        wasserstein_normalized_iqr = values.get("wasserstein_normalized_iqr")
        if _is_finite_number(ks):
            metrics[f"marginal_numeric.{column_name}.ks"] = float(ks)
        if _is_finite_number(wasserstein):
            metrics[f"marginal_numeric.{column_name}.wasserstein"] = float(wasserstein)
        if _is_finite_number(wasserstein_normalized_iqr):
            metrics[f"marginal_numeric.{column_name}.wasserstein_normalized_iqr"] = float(
                wasserstein_normalized_iqr
            )

    for column, values in results.get("marginal_discrete_ordinal", {}).items():
        column_name = _sanitize_name(column)
        ks = values.get("ks")
        wasserstein = values.get("wasserstein")
        wasserstein_normalized_iqr = values.get("wasserstein_normalized_iqr")
        tv = values.get("tv")
        if _is_finite_number(ks):
            metrics[f"marginal_discrete_ordinal.{column_name}.ks"] = float(ks)
        if _is_finite_number(wasserstein):
            metrics[f"marginal_discrete_ordinal.{column_name}.wasserstein"] = float(wasserstein)
        if _is_finite_number(wasserstein_normalized_iqr):
            metrics[f"marginal_discrete_ordinal.{column_name}.wasserstein_normalized_iqr"] = float(
                wasserstein_normalized_iqr
            )
        if _is_finite_number(tv):
            metrics[f"marginal_discrete_ordinal.{column_name}.tv"] = float(tv)

    for column, value in results.get("marginal_categorical_tv", {}).items():
        column_name = _sanitize_name(column)
        if _is_finite_number(value):
            metrics[f"marginal_categorical.{column_name}.tv"] = float(value)

    for column, values in results.get("marginal_date", {}).items():
        column_name = _sanitize_name(column)
        ks = values.get("ks")
        wasserstein_normalized = values.get("wasserstein_normalized")
        if _is_finite_number(ks):
            metrics[f"marginal_date.{column_name}.ks"] = float(ks)
        if _is_finite_number(wasserstein_normalized):
            metrics[f"marginal_date.{column_name}.wasserstein_normalized"] = float(wasserstein_normalized)

    temporal = results.get("temporal_consistency", {})
    overall = temporal.get("overall")
    if _is_finite_number(overall):
        metrics["temporal_consistency.overall"] = float(overall)
    for pair_name, values in temporal.get("pairs", {}).items():
        score = values.get("score")
        if _is_finite_number(score):
            metrics[f"temporal_consistency.{_sanitize_name(pair_name)}.score"] = float(score)

    for column, values in results.get("global_utility", {}).get("per_column", {}).items():
        column_name = _sanitize_name(column)
        utility = values.get("utility")
        perf_ref = values.get("perf_ref")
        perf_synth = values.get("perf_synth")
        if _is_finite_number(utility):
            metrics[f"global_utility.per_column.{column_name}.utility"] = float(utility)
        if _is_finite_number(perf_ref):
            metrics[f"global_utility.per_column.{column_name}.perf_ref"] = float(perf_ref)
        if _is_finite_number(perf_synth):
            metrics[f"global_utility.per_column.{column_name}.perf_synth"] = float(perf_synth)

    return metrics


def _log_metrics(mlflow, metrics: dict[str, float], batch_size: int = 100) -> None:
    items = list(metrics.items())
    for start in range(0, len(items), batch_size):
        mlflow.log_metrics(dict(items[start:start + batch_size]))


def _aggregate_metric_dicts(metric_dicts: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    keys = sorted({key for metric_dict in metric_dicts for key in metric_dict})
    mean_metrics: dict[str, float] = {}
    std_metrics: dict[str, float] = {}

    for key in keys:
        values = [metric_dict[key] for metric_dict in metric_dicts if key in metric_dict]
        if not values:
            continue
        mean_value = float(sum(values) / len(values))
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        mean_metrics[key] = mean_value
        std_metrics[f"{key}.std"] = float(math.sqrt(variance))

    return mean_metrics, std_metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fidelity evaluation and log results to MLflow.")
    parser.add_argument("--real", required=True, help="Path to real.csv")
    parser.add_argument("--synthetic", nargs="+", required=True, help="One or more paths to synthetic CSV files.")
    parser.add_argument("--categorical", nargs="*", default=None, help="Categorical columns (optional).")
    parser.add_argument("--numeric", nargs="*", default=None, help="Numeric columns (optional).")
    parser.add_argument("--numeric_threshold", type=float, default=0.95)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--rep_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--activation", type=str, default="ReLU")
    parser.add_argument("--dropout_prob", type=float, default=0.0)
    parser.add_argument("--nu", type=float, default=0.01)
    parser.add_argument("--svdd_lr", type=float, default=1e-3)
    parser.add_argument("--svdd_weight_decay", type=float, default=1e-5)
    parser.add_argument("--svdd_batch_size", type=int, default=256)
    parser.add_argument("--svdd_epochs", type=int, default=200)
    parser.add_argument("--svdd_warm_up_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Optional per-run seeds matching --synthetic.")
    parser.add_argument("--permutations", type=int, default=50)
    parser.add_argument("--id_unique_threshold", type=float, default=0.98)
    parser.add_argument("--global_utility_test_size", type=float, default=0.2)
    parser.add_argument("--global_utility_min_real_train_rows", type=int, default=200)
    parser.add_argument("--global_utility_min_real_test_rows", type=int, default=100)
    parser.add_argument("--global_utility_min_synth_train_rows", type=int, default=200)
    parser.add_argument("--synth_train_cap", type=int, default=5000)

    parser.add_argument("--experiment-name", default="synthetic-data-fidelity")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--tracking-uri", default=None, help="Example: sqlite:///mlflow.db")
    parser.add_argument(
        "--artifact-subdir",
        default="evaluation",
        help="Artifact subdirectory inside the MLflow run.",
    )
    parser.add_argument(
        "--skip-per-column-metrics",
        action="store_true",
        help="Skip logging per-column metrics to keep runs lighter.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional local text report path, in addition to MLflow artifact logging.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional local JSON report path, in addition to MLflow artifact logging.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    mlflow = _require_mlflow()

    tracking_uri = args.tracking_uri or f"sqlite:///{Path('mlflow.db').resolve()}"
    mlflow.set_tracking_uri(tracking_uri)
    _set_or_restore_experiment(mlflow, args.experiment_name)

    synthetic_paths = args.synthetic
    seeds = args.seeds if args.seeds else [args.seed] * len(synthetic_paths)
    if len(seeds) != len(synthetic_paths):
        raise SystemExit("When provided, --seeds must contain exactly one seed per --synthetic path.")

    model_name = args.model_name or _infer_model_name(synthetic_paths[0])
    dataset_name = args.dataset_name or _infer_dataset_name(args.real)
    run_name = args.run_name or f"{dataset_name}_{model_name}"
    synth_cap = None if args.synth_train_cap is None or args.synth_train_cap <= 0 else args.synth_train_cap

    base_config = EvalConfig(
        numeric_threshold=args.numeric_threshold,
        k=args.k,
        rep_dim=args.rep_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        activation=args.activation,
        dropout_prob=args.dropout_prob,
        nu=args.nu,
        svdd_lr=args.svdd_lr,
        svdd_weight_decay=args.svdd_weight_decay,
        svdd_batch_size=args.svdd_batch_size,
        svdd_epochs=args.svdd_epochs,
        svdd_warm_up_epochs=args.svdd_warm_up_epochs,
        permutations=args.permutations,
        id_unique_threshold=args.id_unique_threshold,
        global_utility_test_size=args.global_utility_test_size,
        synth_train_cap=synth_cap,
        global_utility_min_real_train_rows=args.global_utility_min_real_train_rows,
        global_utility_min_real_test_rows=args.global_utility_min_real_test_rows,
        global_utility_min_synth_train_rows=args.global_utility_min_synth_train_rows,
    )

    run_results = []
    for synthetic_path, seed in zip(synthetic_paths, seeds):
        run_results.append(
            run_evaluation(
                args.real,
                synthetic_path,
                replace(base_config, seed=seed),
                categorical_cols=args.categorical,
                numeric_cols=args.numeric,
            )
        )

    results = run_results[0] if len(run_results) == 1 else aggregate_evaluation_results(run_results, seeds=seeds)
    results["metadata"] = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "run_name": run_name,
    }

    report_text = format_results(results)
    summary_metric_runs = [_collect_summary_metrics(result) for result in run_results]
    summary_metrics, summary_std_metrics = _aggregate_metric_dicts(summary_metric_runs)
    if args.skip_per_column_metrics:
        per_column_metrics = {}
        per_column_std_metrics = {}
    else:
        per_column_metric_runs = [_collect_per_column_metrics(result) for result in run_results]
        per_column_metrics, per_column_std_metrics = _aggregate_metric_dicts(per_column_metric_runs)

    params = {
        "real_path": args.real,
        "synthetic_paths": ",".join(synthetic_paths),
        "model_name": model_name,
        "dataset_name": dataset_name,
        "numeric_threshold": args.numeric_threshold,
        "k": args.k,
        "rep_dim": args.rep_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "activation": args.activation,
        "dropout_prob": args.dropout_prob,
        "nu": args.nu,
        "svdd_lr": args.svdd_lr,
        "svdd_weight_decay": args.svdd_weight_decay,
        "svdd_batch_size": args.svdd_batch_size,
        "svdd_epochs": args.svdd_epochs,
        "svdd_warm_up_epochs": args.svdd_warm_up_epochs,
        "seed": args.seed,
        "seeds": ",".join(str(seed) for seed in seeds),
        "num_runs": len(run_results),
        "permutations": args.permutations,
        "id_unique_threshold": args.id_unique_threshold,
        "global_utility_test_size": args.global_utility_test_size,
        "global_utility_min_real_train_rows": args.global_utility_min_real_train_rows,
        "global_utility_min_real_test_rows": args.global_utility_min_real_test_rows,
        "global_utility_min_synth_train_rows": args.global_utility_min_synth_train_rows,
        "synth_train_cap": "None" if synth_cap is None else synth_cap,
    }

    if args.categorical:
        params["categorical_columns"] = ",".join(args.categorical)
    if args.numeric:
        params["numeric_columns"] = ",".join(args.numeric)

    tags = {
        "task": "synthetic_data_fidelity_evaluation",
        "model_name": model_name,
        "dataset_name": dataset_name,
        "aggregation": "mean_across_runs" if len(run_results) > 1 else "single_run",
    }

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        _log_metrics(mlflow, summary_metrics)
        if len(run_results) > 1:
            _log_metrics(mlflow, summary_std_metrics)
        if per_column_metrics:
            _log_metrics(mlflow, per_column_metrics)
            if len(run_results) > 1:
                _log_metrics(mlflow, per_column_std_metrics)

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            txt_path = tmp_path / "evaluation.txt"
            json_path = tmp_path / "evaluation.json"
            per_run_path = tmp_path / "evaluation_runs.json"
            meta_path = tmp_path / "run_metadata.json"

            txt_path.write_text(report_text, encoding="utf-8")
            json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            per_run_path.write_text(json.dumps(run_results, indent=2), encoding="utf-8")
            meta_path.write_text(
                json.dumps(
                    {
                        "model_name": model_name,
                        "dataset_name": dataset_name,
                        "tracking_uri": tracking_uri,
                        "experiment_name": args.experiment_name,
                        "run_name": run_name,
                        "synthetic_paths": synthetic_paths,
                        "seeds": seeds,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            mlflow.log_artifact(str(txt_path), artifact_path=args.artifact_subdir)
            mlflow.log_artifact(str(json_path), artifact_path=args.artifact_subdir)
            mlflow.log_artifact(str(per_run_path), artifact_path=args.artifact_subdir)
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
