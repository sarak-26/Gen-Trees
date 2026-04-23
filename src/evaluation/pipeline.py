import argparse
import importlib
import inspect
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SYNTHETIC_DIR = ROOT / "synthetic_data"
RESULTS_DIR = ROOT / "results"

MODEL_MODULES = {
    "ctgan": ("CTGAN", "src.models.CTGAN"),
    "ctabganplus": ("CTABGANPlus", "src.models.CTABGANPlus"),
    "tvae": ("TVAE", "src.models.TVAE"),
    "gaussiancopula": ("GaussianCopula", "src.models.GaussianCopula"),
    "forestflow": ("ForestFlow", "src.models.ForestFlow"),
    "genforest": ("GenForest", "src.models.GenForests"),
    "arf": ("arf", "src.models.adversarial_rforest"),
    "tabddm": ("TabDDM", "src.models.TabDDM"),
}


def _discover_datasets() -> dict[str, Path]:
    datasets: dict[str, Path] = {}
    for csv_path in sorted(DATA_DIR.rglob("*.csv")):
        if "generated" in csv_path.parts:
            continue
        datasets[csv_path.stem.lower()] = csv_path
    return datasets


def _resolve_models(selection: str) -> list[tuple[str, str]]:
    if selection.upper() == "ALL":
        return list(MODEL_MODULES.values())

    key = selection.strip().lower()
    aliases = {
        "ctabgan+": "ctabganplus",
        "ctabgan_plus": "ctabganplus",
        "ctabgan": "ctabganplus",
        "forest_flow": "forestflow",
        "forest-flows": "forestflow",
        "forestflows": "forestflow",
        "forestdiffusion": "forestflow",
        "gaussian_copula": "gaussiancopula",
        "copula": "gaussiancopula",
        "genforests": "genforest",
        "adversarial_rforest": "arf",
        "random_forest": "arf",
    }
    key = aliases.get(key, key)
    if key not in MODEL_MODULES:
        available = ", ".join(name for name, _ in MODEL_MODULES.values())
        raise SystemExit(f"Unknown model '{selection}'. Available: ALL, {available}")
    return [MODEL_MODULES[key]]


def _resolve_datasets(selection: str) -> list[tuple[str, Path]]:
    datasets = _discover_datasets()
    if selection.upper() == "ALL":
        return [(path.stem, path) for path in datasets.values()]

    candidate = Path(selection)
    if candidate.exists():
        return [(candidate.stem, candidate.resolve())]

    key = selection.strip().lower()
    if key not in datasets:
        available = ", ".join(sorted(path.stem for path in datasets.values()))
        raise SystemExit(f"Unknown dataset '{selection}'. Available: ALL, {available}")
    path = datasets[key]
    return [(path.stem, path)]


def _load_generate_function(module_path: str):
    module = importlib.import_module(module_path)
    generate = getattr(module, "generate", None)
    if generate is None:
        raise SystemExit(f"Module '{module_path}' does not expose a generate(...) function.")
    return generate


def _build_seeded_synthetic_filename(dataset_name: str, model_name: str, seed: int) -> str:
    return f"{dataset_name}_{model_name}_seed{seed}.csv"


def _generate_synthetic_data(
    model_name: str,
    module_path: str,
    dataset_name: str,
    dataset_path: Path,
    n_rows: int,
    seed: int,
) -> Path:
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    output_name = _build_seeded_synthetic_filename(dataset_name, model_name, seed)
    generate = _load_generate_function(module_path)
    signature = inspect.signature(generate)
    if "seed" in signature.parameters:
        generate(str(dataset_path), int(n_rows), output_name, seed=seed)
    else:
        generate(str(dataset_path), int(n_rows), output_name)
    return SYNTHETIC_DIR / output_name


def _run_fidelity_mlflow(
    real_path: Path,
    synthetic_paths: list[Path],
    dataset_name: str,
    model_name: str,
    experiment_name: str,
    tracking_uri: str | None,
    save_txt: bool,
    seeds: list[int],
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "src.evaluation.fidelity_mlflow",
        "--real",
        str(real_path),
        "--synthetic",
        *(str(synthetic_path) for synthetic_path in synthetic_paths),
        "--dataset-name",
        dataset_name,
        "--model-name",
        model_name,
        "--run-name",
        f"{dataset_name}_{model_name}",
        "--experiment-name",
        experiment_name,
        "--seeds",
        *(str(seed) for seed in seeds),
    ]

    if tracking_uri:
        cmd.extend(["--tracking-uri", tracking_uri])

    if save_txt:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        txt_path = RESULTS_DIR / f"eval_{dataset_name}_{model_name}.txt"
        cmd.extend(["--output", str(txt_path)])

    subprocess.run(cmd, cwd=ROOT, check=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for one or more model/dataset pairs and log fidelity results to MLflow."
    )
    parser.add_argument("--model", required=True, help="Model name or ALL.")
    parser.add_argument("--dataset", required=True, help="Dataset name, dataset path, or ALL.")
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Synthetic row count. Defaults to the number of rows in the real dataset.",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Also save the formatted evaluation report to results/eval_<dataset>_<model>.txt.",
    )
    parser.add_argument(
        "--experiment-name",
        default="synthetic-data-fidelity",
        help="MLflow experiment name passed through to fidelity_mlflow.py.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Optional MLflow tracking URI. Defaults to sqlite:///mlflow.db via fidelity_mlflow.py.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[2026, 2003, 42],
        help="Seeds used for repeated generation and evaluation. Defaults to 2026 2003 42.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    model_selections = _resolve_models(args.model)
    dataset_selections = _resolve_datasets(args.dataset)

    for dataset_name, dataset_path in dataset_selections:
        real_rows = len(pd.read_csv(dataset_path))
        n_rows = args.rows if args.rows is not None else real_rows

        for model_name, module_path in model_selections:
            print(f"\n[Pipeline] dataset={dataset_name} model={model_name} rows={n_rows}")
            synthetic_paths = [
                _generate_synthetic_data(
                    model_name=model_name,
                    module_path=module_path,
                    dataset_name=dataset_name,
                    dataset_path=dataset_path,
                    n_rows=n_rows,
                    seed=seed,
                )
                for seed in args.seeds
            ]
            _run_fidelity_mlflow(
                real_path=dataset_path,
                synthetic_paths=synthetic_paths,
                dataset_name=dataset_name,
                model_name=model_name,
                experiment_name=args.experiment_name,
                tracking_uri=args.tracking_uri,
                save_txt=args.save_txt,
                seeds=args.seeds,
            )


if __name__ == "__main__":
    main()
