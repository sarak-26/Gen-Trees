import gc
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "cost_results"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

RESULTS_DIR.mkdir(exist_ok=True)

DATASETS: list[str] = [
    "employee_data",
]

MODELS: list[str] = ["arf", "GenForest", "ForestFlow", "CTGAN", "TVAE", "TabDDM"]

SEEDS: list[int] = [42]

N_SYNTHETIC: int = 1000

RAW_CSV = RESULTS_DIR / "cost_results_raw.csv"
SUMMARY_CSV = RESULTS_DIR / "cost_results_summary.csv"
RANKING_TXT = RESULTS_DIR / "cost_results_ranking.txt"


def _detect_mps() -> bool:
    try:
        import torch
        return bool(torch.backends.mps.is_available())
    except ImportError:
        return False

MPS_AVAILABLE: bool = _detect_mps()


def load_dataset(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def dataset_info(df: pd.DataFrame) -> dict[str, int]:
    n_rows, n_cols = df.shape
    n_cat = int(
        (df.dtypes == object).sum()
        + sum(isinstance(dt, pd.CategoricalDtype) for dt in df.dtypes)
    )
    return {"n_rows": n_rows, "n_cols": n_cols, "n_cat_cols": n_cat}


class _CPUMonitor:
    def __init__(self) -> None:
        self._proc = psutil.Process()
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._samples.append(self._proc.cpu_percent(interval=None))
            time.sleep(0.2)

    def start(self) -> None:
        self._proc.cpu_percent(interval=None)   # prime the counter
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        self._thread.join(timeout=2.0)
        return (sum(self._samples) / len(self._samples)) if self._samples else 0.0


class _PeakRAMTracker:
    def __init__(self) -> None:
        self._proc = psutil.Process()
        self._peak_mb = self._rss_mb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _rss_mb(self) -> float:
        return self._proc.memory_info().rss / (1024 ** 2)

    def _run(self) -> None:
        while not self._stop.is_set():
            current = self._rss_mb()
            if current > self._peak_mb:
                self._peak_mb = current
            time.sleep(0.1)

    def start(self) -> None:
        self._peak_mb = self._rss_mb()
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        self._thread.join(timeout=2.0)
        return self._peak_mb


def train_model(model_name: str, train_df: pd.DataFrame, seed: int) -> Any:
    if model_name == "arf":
        return _train_arf(train_df, seed)
    if model_name == "GenForest":
        return _train_genforest(train_df, seed)
    if model_name == "ForestFlow":
        return _train_forestflow(train_df, seed)
    if model_name == "CTGAN":
        return _train_sdv(train_df, seed, "CTGAN")
    if model_name == "TVAE":
        return _train_sdv(train_df, seed, "TVAE")
    if model_name == "TabDDM":
        return _train_tabddm(train_df, seed)
    raise ValueError(f"Unknown model: {model_name!r}")


def sample_model(model_name: str, fitted_model: Any, n_samples: int) -> pd.DataFrame:
    if model_name == "arf":
        return _sample_arf(fitted_model, n_samples)
    if model_name == "GenForest":
        return fitted_model.sample(n=n_samples)
    if model_name == "ForestFlow":
        return _sample_forestflow(fitted_model, n_samples)
    if model_name in ("CTGAN", "TVAE"):
        return fitted_model.sample(num_rows=n_samples)
    if model_name == "TabDDM":
        return fitted_model.sample(n=n_samples)
    raise ValueError(f"Unknown model: {model_name!r}")


def _train_arf(train_df: pd.DataFrame, seed: int) -> Any:
    from arfpy import arf as arfmod
    from src.models.preprocessing import prepare_training_dataframe
    from src.models.adversarial_rforest import (
        _apply_finite_bounds,
        _seed_everything,
    )

    df = prepare_training_dataframe(train_df.copy())
    _seed_everything(seed)
    model = arfmod.arf(x=df)
    model.forde()
    _apply_finite_bounds(model, df, finite_bounds="global")
    if not model.params.empty and "sd" in model.params.columns:
        model.params = model.params.assign(
            sd=model.params["sd"].clip(lower=1e-9).fillna(1e-9)
        )
    # stash preprocessed training frame for postprocessing in _sample_arf
    model._benchmark_train_df = df
    return model


def _train_genforest(train_df: pd.DataFrame, seed: int) -> Any:
    from src.models.GenForests import GenerativeForest
    from src.models.preprocessing import prepare_training_dataframe

    df = prepare_training_dataframe(train_df.copy())
    model = GenerativeForest(random_state=seed, verbose=False)
    model.fit(df)
    return model


def _train_forestflow(train_df: pd.DataFrame, seed: int) -> Any:
    # ForestDiffusionModel trains inside __init__, so instantiation IS training
    try:
        from ForestDiffusion import ForestDiffusionModel
    except ModuleNotFoundError as exc:
        raise RuntimeError("ForestDiffusion is not installed.") from exc

    from src.models.preprocessing import prepare_training_dataframe
    from src.models.ForestFlow import _encode_dataframe

    df = prepare_training_dataframe(train_df.copy())
    encoded, metadata, bin_indexes, cat_indexes, int_indexes = _encode_dataframe(
        df, discrete_cardinality_threshold=10
    )

    model = ForestDiffusionModel(
        encoded.to_numpy(dtype=float),
        n_t=10,
        duplicate_K=10,
        n_batch=1,
        diffusion_type="flow",
        bin_indexes=bin_indexes,
        cat_indexes=cat_indexes,
        int_indexes=int_indexes,
        n_jobs=1,   # single-threaded — avoids multiprocessing on MacBook Air M1
        seed=seed,
        max_depth=4,
        n_estimators=50,
    )
    # stash decoding context for _sample_forestflow
    model._benchmark_metadata = metadata
    model._benchmark_columns = list(df.columns)
    model._benchmark_train_df = df
    return model


def _train_sdv(train_df: pd.DataFrame, seed: int, model_type: str) -> Any:
    from src.models.preprocessing import prepare_training_dataframe
    from src.models.backend_adapters import build_sdv_metadata

    df = prepare_training_dataframe(train_df.copy())
    metadata = build_sdv_metadata(df)

    if model_type == "CTGAN":
        from sdv.single_table import CTGANSynthesizer
        synthesizer = CTGANSynthesizer(metadata, verbose=False)
    else:
        from sdv.single_table import TVAESynthesizer
        synthesizer = TVAESynthesizer(metadata)

    synthesizer.fit(df)
    return synthesizer


def _train_tabddm(train_df: pd.DataFrame, seed: int) -> Any:
    from src.models.TabDDM import TabDDM
    from src.models.preprocessing import prepare_training_dataframe

    df = prepare_training_dataframe(train_df.copy())
    model = TabDDM()
    model.fit(df)
    return model


def _sample_arf(model: Any, n_samples: int) -> pd.DataFrame:
    from src.models.adversarial_rforest import _postprocess_generated_data
    from src.date_columns import finalize_synthetic_dates

    new_data = model.forge(n=n_samples)
    train_df = getattr(model, "_benchmark_train_df", new_data)
    new_data = _postprocess_generated_data(new_data, train_df)
    return finalize_synthetic_dates(new_data, train_df)


def _sample_forestflow(model: Any, n_samples: int) -> pd.DataFrame:
    from src.models.ForestFlow import _decode_dataframe
    from src.date_columns import finalize_synthetic_dates

    raw = model.generate(batch_size=n_samples)
    decoded = _decode_dataframe(
        np.asarray(raw, dtype=float),
        model._benchmark_metadata,
        model._benchmark_columns,
    )
    float_cols = decoded.select_dtypes(include="float").columns
    decoded[float_cols] = decoded[float_cols].round(3)
    return finalize_synthetic_dates(decoded, model._benchmark_train_df)


def run_single(
    model_name: str,
    dataset_name: str,
    seed: int,
    n_synthetic: int,
    train_df: pd.DataFrame,
) -> dict[str, Any]:
    info = dataset_info(train_df)
    record: dict[str, Any] = {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "n_rows": info["n_rows"],
        "n_cols": info["n_cols"],
        "n_cat_cols": info["n_cat_cols"],
        "train_time_s": None,
        "gen_time_s": None,
        "samples_per_sec": None,
        "peak_ram_mb": None,
        "avg_cpu_pct": None,
        "mps_used": MPS_AVAILABLE,
        "error": None,
    }

    ram_tracker = _PeakRAMTracker()
    cpu_monitor = _CPUMonitor()

    try:
        gc.collect()
        baseline_mb = psutil.Process().memory_info().rss / (1024 ** 2)

        ram_tracker.start()
        cpu_monitor.start()

        t0 = time.perf_counter()
        fitted_model = train_model(model_name, train_df, seed)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        synthetic_df = sample_model(model_name, fitted_model, n_synthetic)
        gen_time = time.perf_counter() - t1

        peak_mb = ram_tracker.stop()
        avg_cpu = cpu_monitor.stop()

        record.update({
            "train_time_s": round(train_time, 4),
            "gen_time_s": round(gen_time, 4),
            "samples_per_sec": round(n_synthetic / gen_time, 2) if gen_time > 0 else None,
            "peak_ram_mb": round(max(peak_mb - baseline_mb, 0.0), 2),
            "avg_cpu_pct": round(avg_cpu, 2),
        })

        del fitted_model, synthetic_df

    except Exception as exc:
        record["error"] = traceback.format_exc(limit=5).strip().splitlines()[-1]
        print(f"    ERROR: {exc}")

    finally:
        try:
            ram_tracker.stop()
        except Exception:
            pass
        try:
            cpu_monitor.stop()
        except Exception:
            pass
        gc.collect()

    return record


def run_benchmark(
    datasets: list[str] = DATASETS,
    models: list[str] = MODELS,
    seeds: list[int] = SEEDS,
    n_synthetic: int = N_SYNTHETIC,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    total = len(datasets) * len(models) * len(seeds)
    done = 0
    width = len(str(total))

    for dataset_name in datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 60}")

        try:
            train_df = load_dataset(dataset_name)
        except FileNotFoundError as exc:
            print(f"  Skipping — {exc}")
            continue

        for model_name in models:
            for seed in seeds:
                done += 1
                print(
                    f"  [{done:{width}}/{total}]  {model_name:<12}  seed={seed} ...",
                    end=" ",
                    flush=True,
                )
                record = run_single(model_name, dataset_name, seed, n_synthetic, train_df)
                records.append(record)

                if record["error"] is None:
                    print(
                        f"OK  "
                        f"train={record['train_time_s']:.2f}s  "
                        f"gen={record['gen_time_s']:.2f}s  "
                        f"RAM Δ={record['peak_ram_mb']:.1f} MB  "
                        f"CPU={record['avg_cpu_pct']:.0f}%"
                    )
                else:
                    print("FAILED")

    return pd.DataFrame(records)


def build_summary(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = ["train_time_s", "gen_time_s", "samples_per_sec", "peak_ram_mb"]
    ok = raw[raw["error"].isna()].copy()

    agg = (
        ok
        .groupby(["model", "dataset"])[metrics]
        .agg(["mean", "std"])
    )
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    return agg.reset_index()


def build_ranking(summary: pd.DataFrame) -> str:
    model_agg = (
        summary
        .groupby("model")
        .agg(
            train_time      = ("train_time_s_mean",    "mean"),
            gen_time        = ("gen_time_s_mean",       "mean"),
            peak_ram        = ("peak_ram_mb_mean",      "mean"),
            samples_per_sec = ("samples_per_sec_mean",  "mean"),
        )
        .reset_index()
    )

    def _section(col: str, label: str, ascending: bool) -> list[str]:
        note = "lower is better" if ascending else "higher is better"
        ranked = (
            model_agg[["model", col]]
            .dropna()
            .sort_values(col, ascending=ascending)
            .reset_index(drop=True)
        )
        lines = [f"Ranked by {label} ({note}):"]
        for i, row in ranked.iterrows():
            lines.append(f"  {i + 1}. {row['model']:<14}  {row[col]:>10.3f}")
        return lines + [""]

    lines = (
        ["=" * 60, "  Model Ranking — averaged across all datasets", "=" * 60, ""]
        + _section("train_time",      "Average Training Time (s)",    ascending=True)
        + _section("gen_time",        "Average Generation Time (s)",  ascending=True)
        + _section("peak_ram",        "Average Peak RAM delta (MB)",  ascending=True)
        + _section("samples_per_sec", "Samples Generated per Second", ascending=False)
    )
    return "\n".join(lines)


if __name__ == "__main__":
    print(f"Apple MPS available : {MPS_AVAILABLE}")
    print(f"Benchmarking        : {len(MODELS)} models × {len(DATASETS)} datasets × {len(SEEDS)} seeds")
    print(f"Synthetic rows/run  : {N_SYNTHETIC}")

    raw_df = run_benchmark()

    if RAW_CSV.exists():
        existing = pd.read_csv(RAW_CSV)
        raw_df = pd.concat([existing, raw_df], ignore_index=True)
    raw_df.to_csv(RAW_CSV, index=False)
    print(f"\nRaw results  → {RAW_CSV}")

    summary_df = build_summary(raw_df)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Summary      → {SUMMARY_CSV}")

    ranking_text = build_ranking(summary_df)
    print("\n" + ranking_text)
    RANKING_TXT.write_text(ranking_text)
    print(f"Ranking      → {RANKING_TXT}")
