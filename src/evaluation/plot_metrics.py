import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MLRUNS_DIR = ROOT / "mlruns"
DEFAULT_OUTPUT_DIR = ROOT / "results"
MPL_CACHE_DIR = ROOT / ".cache" / "matplotlib"
FONTCONFIG_CACHE_DIR = ROOT / ".cache" / "fontconfig"

MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
FONTCONFIG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("FONTCONFIG_PATH", "/opt/homebrew/etc/fonts")
os.environ.setdefault("FONTCONFIG_FILE", "/opt/homebrew/etc/fonts/fonts.conf")

import matplotlib.pyplot as plt
DEFAULT_METRICS = [
    "support.alpha_precision",
    "support.beta_recall",
    "discriminator.auc",
    "global_utility.global_utility",
]


def _flatten_numeric_metrics(data: dict, prefix: str = "") -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            metrics.update(_flatten_numeric_metrics(value, full_key))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[full_key] = float(value)
    return metrics


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _collect_mlflow_runs(mlruns_dir: Path) -> list[dict]:
    runs: list[dict] = []
    for eval_path in sorted(mlruns_dir.glob("*/*/artifacts/evaluation/evaluation.json")):
        metadata_path = eval_path.with_name("run_metadata.json")
        if not metadata_path.exists():
            continue

        evaluation = _load_json(eval_path)
        metadata = _load_json(metadata_path)
        runs.append(
            {
                "source": eval_path,
                "dataset_name": metadata.get("dataset_name", "unknown"),
                "model_name": metadata.get("model_name", "unknown"),
                "run_name": metadata.get("run_name", eval_path.parent.parent.name),
                "metrics": _flatten_numeric_metrics(evaluation),
            }
        )
    return runs


def _collect_local_jsons(paths: list[Path]) -> list[dict]:
    runs: list[dict] = []
    for path in paths:
        evaluation = _load_json(path)
        dataset_name = Path(evaluation.get("paths", {}).get("real", path.stem)).stem
        model_name = Path(evaluation.get("paths", {}).get("synthetic", path.stem)).stem
        runs.append(
            {
                "source": path,
                "dataset_name": dataset_name,
                "model_name": model_name,
                "run_name": f"{dataset_name}_{model_name}",
                "metrics": _flatten_numeric_metrics(evaluation),
            }
        )
    return runs


def _filter_runs(runs: list[dict], dataset: str | None, models: list[str] | None) -> list[dict]:
    filtered = runs
    if dataset:
        filtered = [run for run in filtered if run["dataset_name"].lower() == dataset.lower()]
    if models:
        wanted = {model.lower() for model in models}
        filtered = [run for run in filtered if run["model_name"].lower() in wanted]
    return filtered


def _list_metrics(runs: list[dict]) -> list[str]:
    metrics = set()
    for run in runs:
        metrics.update(run["metrics"].keys())
    return sorted(metrics)


def _build_plot_title(dataset: str | None, metrics: list[str]) -> str:
    dataset_label = dataset if dataset else "all_datasets"
    if len(metrics) == 1:
        return f"{dataset_label}: {metrics[0]}"
    if len(metrics) == 2:
        return f"{dataset_label}: {metrics[0]} vs {metrics[1]}"
    return f"{dataset_label}: metric comparison"


def _sanitize_filename_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value)


def _finalize_figure(fig, output: Path | None = None, show: bool = False) -> None:
    fig.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {output}")
    if show:
        plt.show()


def _plot_single_metric(runs: list[dict], metric: str, title: str):
    labels = [run["model_name"] for run in runs]
    values = [run["metrics"].get(metric, np.nan) for run in runs]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=30)
    return fig


def _plot_two_metrics(runs: list[dict], metrics: list[str], title: str):
    x_metric, y_metric = metrics
    fig, ax = plt.subplots(figsize=(7, 6))

    for run in runs:
        x = run["metrics"].get(x_metric)
        y = run["metrics"].get(y_metric)
        if x is None or y is None:
            continue
        ax.scatter(x, y, label=run["model_name"], s=80)
        ax.annotate(run["model_name"], (x, y), xytext=(4, 4), textcoords="offset points")

    ax.set_title(title)
    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    return fig


def _plot_multiple_metrics(runs: list[dict], metrics: list[str], title: str):
    labels = [run["model_name"] for run in runs]
    x = np.arange(len(labels))
    width = 0.8 / max(len(metrics), 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    for index, metric in enumerate(metrics):
        values = [run["metrics"].get(metric, np.nan) for run in runs]
        offset = (index - (len(metrics) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, label=metric)

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.legend()
    return fig


def _default_output_path(dataset: str | None, metrics: list[str]) -> Path:
    metric_label = "_vs_".join(_sanitize_filename_part(metric) for metric in metrics[:3])
    if len(metrics) > 3:
        metric_label += "_etc"
    dataset_label = _sanitize_filename_part(dataset or "all_datasets")
    return DEFAULT_OUTPUT_DIR / f"plot_{dataset_label}_{metric_label}.png"


def load_runs(mlruns_dir: Path | str = DEFAULT_MLRUNS_DIR, input_json: list[Path | str] | None = None) -> list[dict]:
    if input_json:
        return _collect_local_jsons([Path(path).resolve() for path in input_json])
    return _collect_mlflow_runs(Path(mlruns_dir))


def available_datasets(runs: list[dict]) -> list[str]:
    return sorted({run["dataset_name"] for run in runs})


def available_models(runs: list[dict], dataset: str | None = None) -> list[str]:
    filtered = _filter_runs(runs, dataset=dataset, models=None)
    return sorted({run["model_name"] for run in filtered})


def available_metrics(runs: list[dict], dataset: str | None = None, models: list[str] | None = None) -> list[str]:
    filtered = _filter_runs(runs, dataset=dataset, models=models)
    return _list_metrics(filtered)


def build_plot_figure(
    runs: list[dict],
    dataset: str | None = None,
    models: list[str] | None = None,
    metrics: list[str] | None = None,
):
    filtered_runs = _filter_runs(runs, dataset=dataset, models=models)
    if not filtered_runs:
        raise ValueError("No evaluation runs matched the requested filters.")

    chosen_metrics = metrics or DEFAULT_METRICS
    title = _build_plot_title(dataset, chosen_metrics)

    if len(chosen_metrics) == 1:
        return _plot_single_metric(filtered_runs, chosen_metrics[0], title)
    if len(chosen_metrics) == 2:
        return _plot_two_metrics(filtered_runs, chosen_metrics, title)
    return _plot_multiple_metrics(filtered_runs, chosen_metrics, title)


def save_figure(figure, output: Path | str) -> Path:
    output_path = Path(output).resolve()
    _finalize_figure(figure, output=output_path, show=False)
    return output_path


def create_plot(
    runs: list[dict],
    dataset: str | None = None,
    models: list[str] | None = None,
    metrics: list[str] | None = None,
    output: Path | str | None = None,
) -> Path:
    filtered_runs = _filter_runs(runs, dataset=dataset, models=models)
    if not filtered_runs:
        raise ValueError("No evaluation runs matched the requested filters.")

    chosen_metrics = metrics or DEFAULT_METRICS
    output_path = Path(output).resolve() if output else _default_output_path(dataset, chosen_metrics)
    fig = build_plot_figure(filtered_runs, dataset=dataset, models=None, metrics=chosen_metrics)
    save_figure(fig, output_path)
    plt.close(fig)

    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create simple Matplotlib comparisons from evaluation JSON files or MLflow evaluation artifacts."
    )
    parser.add_argument(
        "--mlruns-dir",
        default=str(DEFAULT_MLRUNS_DIR),
        help="Root mlruns directory to scan for evaluation artifacts.",
    )
    parser.add_argument(
        "--input-json",
        nargs="*",
        default=None,
        help="Optional evaluation.json files to plot instead of scanning mlruns.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset filter, for example ibm_hr.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model filter, for example CTGAN TVAE GaussianCopula arf.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Metric paths to plot. One metric gives a bar chart, two give a scatter plot, three or more give grouped bars.",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="Print all discovered metric paths and exit.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output PNG path. Defaults to results/plot_<dataset>_<metric>.png.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    runs = load_runs(mlruns_dir=args.mlruns_dir, input_json=args.input_json)

    runs = _filter_runs(runs, dataset=args.dataset, models=args.models)
    if not runs:
        raise SystemExit("No evaluation runs matched the requested filters.")

    if args.list_metrics:
        for metric in _list_metrics(runs):
            print(metric)
        return

    create_plot(runs=runs, dataset=args.dataset, models=None, metrics=args.metrics, output=args.output)


if __name__ == "__main__":
    main()
