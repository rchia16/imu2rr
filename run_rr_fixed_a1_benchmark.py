#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_SUBJECTS = "S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29"
DEFAULT_SEEDS = "0 1 2 3 4"
DEFAULT_BASELINE_MODELS = "resnet1d cnn_gru tcn inceptiontime stft_cnn patchtst crossmodal_rr"
PRIMARY_REFERENCE = "F0 original RR readout"
RECOGNIZED_F0_METRIC_SOURCES = {"native_rr_head", "original_rr_readout", "a1_gaussian_kl"}


def parse_list(text: str | Sequence[str]) -> List[str]:
    if isinstance(text, (list, tuple)):
        items: List[str] = []
        for item in text:
            items.extend(parse_list(str(item)))
        return items
    return [x for x in str(text).replace(",", " ").split() if x]


def run_command(cmd: Sequence[str], log_path: Path, *, dry_run: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[RUN]", " ".join(shlex.quote(x) for x in cmd), flush=True)
    if dry_run:
        with open(log_path, "w") as f:
            f.write("DRY RUN: " + " ".join(shlex.quote(x) for x in cmd) + "\n")
        return
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}. See {log_path}")


def run_job_batch(jobs: Sequence[Dict[str, object]], *, dry_run: bool = False) -> None:
    if dry_run:
        for job in jobs:
            run_command(job["cmd"], job["log_path"], dry_run=True)  # type: ignore[arg-type]
        return
    pending = list(jobs)
    running: List[Tuple[subprocess.Popen, Path, str]] = []
    max_parallel = max(1, len({str(job["device"]) for job in jobs}) if jobs else 1)
    failures: List[Tuple[str, int, Path]] = []
    while pending or running:
        while pending and len(running) < max_parallel:
            job = pending.pop(0)
            cmd = job["cmd"]  # type: ignore[assignment]
            log_path = Path(job["log_path"])  # type: ignore[arg-type]
            label = str(job["label"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            print("[RUN]", label, "->", " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
            log = open(log_path, "w")
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
            proc._benchmark_log_handle = log  # type: ignore[attr-defined]
            running.append((proc, log_path, label))
        next_running: List[Tuple[subprocess.Popen, Path, str]] = []
        for proc, log_path, label in running:
            status = proc.poll()
            if status is None:
                next_running.append((proc, log_path, label))
                continue
            handle = getattr(proc, "_benchmark_log_handle", None)
            if handle is not None:
                handle.close()
            if status != 0:
                failures.append((label, int(status), log_path))
        running = next_running
        if running:
            import time

            time.sleep(2.0)
    if failures:
        lines = [f"{label} failed with exit code {code}; log={log}" for label, code, log in failures]
        raise RuntimeError("One or more benchmark jobs failed:\n" + "\n".join(lines))


def metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for src, dst in (("MAE", "mae"), ("RMSE", "rmse"), ("Pearson_correlation", "corr"), ("rr_mae", "mae"), ("rr_rmse", "rmse"), ("rr_corr", "corr")):
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]
    return out


def load_baseline_rows(path: Path, seed: int) -> pd.DataFrame:
    df = metric_columns(pd.read_csv(path))
    df = df.copy()
    if "mode" in df.columns:
        df = df[df["mode"].astype(str) == "none"].copy()
    else:
        df["mode"] = "none"
    df["seed"] = df["seed"] if "seed" in df.columns else int(seed)
    df["family"] = "jbhi_baselines"
    df["benchmark_method"] = df["model"].astype(str)
    df.loc[df["model"] == "tcn", "benchmark_method"] = "TCN baseline"
    df.loc[df["model"] == "crossmodal_rr", "benchmark_method"] = "standard cross-modal model"
    return df


def load_f0_rows(path: Path, seed: int) -> pd.DataFrame:
    df = metric_columns(pd.read_csv(path))
    df = df.copy()
    df["seed"] = df["seed"] if "seed" in df.columns else int(seed)
    mode = df["mode"].astype(str) if "mode" in df.columns else pd.Series(["full"] * len(df), index=df.index)
    df = df[mode == "full"].copy()
    if df.empty:
        raise ValueError(f"{path} contains no F0 rows with mode=full.")
    if "metric_source" not in df.columns:
        df["metric_source"] = "native_rr_head"
    metric_sources = set(df["metric_source"].astype(str))
    unknown = sorted(metric_sources - RECOGNIZED_F0_METRIC_SOURCES)
    if unknown:
        raise ValueError(f"{path} contains unsupported metric_source value(s): {unknown}")

    rows: List[pd.DataFrame] = []
    native = df[df["metric_source"].astype(str) == "native_rr_head"].copy()
    if not native.empty:
        native["family"] = "f0_native"
        native["benchmark_method"] = "F0 native RR head"
        for src, dst in (("native_rr_mae", "mae"), ("native_rr_rmse", "rmse"), ("native_rr_corr", "corr")):
            if src in native.columns:
                native[dst] = native[src]
        rows.append(native)

    original = df[df["metric_source"].astype(str) == "original_rr_readout"].copy()
    if not original.empty:
        original["family"] = "f0"
        original["benchmark_method"] = "F0 original RR readout"
        for src, dst in (("rr_probe_post_mae", "mae"), ("rr_probe_post_rmse", "rmse"), ("rr_probe_post_corr", "corr")):
            if src in original.columns:
                original[dst] = original[src]
        rows.append(original)

    a1 = df[df["metric_source"].astype(str) == "a1_gaussian_kl"].copy()
    if not a1.empty:
        a1["family"] = "f0_a1"
        a1["benchmark_method"] = "F0 + A1 Gaussian-KL spectral readout"
        rows.append(a1)

    if not rows:
        raise ValueError(f"{path} contains no recognised F0 metric_source rows.")
    return pd.concat(rows, ignore_index=True)


def _require_unique_subject_rows(df: pd.DataFrame, *, source: str, path: Path, seed: int) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"{path} seed={seed} is missing required metric_source={source}.")
    duplicates = df[df.duplicated("subject", keep=False)]
    if not duplicates.empty:
        dup_subjects = sorted(duplicates["subject"].astype(str).unique())
        raise ValueError(f"{path} seed={seed} has duplicate {source} rows for subject(s): {dup_subjects}")
    return df.set_index("subject", drop=False)


def validate_integrated_f0_lineage(f0_rows: pd.DataFrame, *, path: Path, seed: int) -> None:
    if "metric_source" not in f0_rows.columns:
        raise ValueError(f"{path} seed={seed} is missing metric_source.")
    original = _require_unique_subject_rows(
        f0_rows[f0_rows["metric_source"].astype(str) == "original_rr_readout"],
        source="original_rr_readout",
        path=path,
        seed=seed,
    )
    a1 = _require_unique_subject_rows(
        f0_rows[f0_rows["metric_source"].astype(str) == "a1_gaussian_kl"],
        source="a1_gaussian_kl",
        path=path,
        seed=seed,
    )
    if set(original.index) != set(a1.index):
        missing_a1 = sorted(set(original.index) - set(a1.index))
        missing_original = sorted(set(a1.index) - set(original.index))
        raise ValueError(
            f"{path} seed={seed} has mismatched original/A1 subjects; "
            f"missing_a1={missing_a1}, missing_original={missing_original}"
        )
    required_cols = ("f0_checkpoint_sha256", "evaluation_indices_hash")
    for col in required_cols:
        if col not in original.columns or col not in a1.columns:
            raise ValueError(f"{path} seed={seed} is missing lineage column {col!r}.")
        original_values = original[col].astype(str)
        a1_values = a1.loc[original.index, col].astype(str)
        missing_values = original_values.eq("") | a1_values.eq("") | original_values.eq("nan") | a1_values.eq("nan")
        if bool(missing_values.any()):
            subjects = sorted(original.index[missing_values].astype(str).tolist())
            raise ValueError(f"{path} seed={seed} has empty lineage value {col!r} for subject(s): {subjects}")
        mismatch = original_values != a1_values
        if bool(mismatch.any()):
            subjects = sorted(original.index[mismatch].astype(str).tolist())
            raise ValueError(f"{path} seed={seed} original/A1 {col} mismatch for subject(s): {subjects}")


def load_a1_rows(path: Path, seed: int) -> pd.DataFrame:
    df = metric_columns(pd.read_csv(path))
    df = df[df["method"].astype(str) == "A1_gaussian_kl"].copy()
    df["seed"] = df["seed"] if "seed" in df.columns else int(seed)
    df["family"] = "legacy_phase_abc_a1"
    df["model"] = "crossmodal_rr"
    df["mode"] = "fixed_20_param_gaussian_kl"
    df["benchmark_method"] = "legacy Phase ABC A1 Gaussian-KL spectral readout"
    return df


def load_classical_rows(path: Path) -> pd.DataFrame:
    df = metric_columns(pd.read_csv(path))
    if "subject" not in df.columns:
        raise ValueError(f"Classical/DSP CSV {path} must include a subject column.")
    if "seed" not in df.columns:
        df["seed"] = 0
    df["family"] = "classical_dsp"
    df["model"] = df["model"] if "model" in df.columns else "classical_dsp"
    df["mode"] = df["mode"] if "mode" in df.columns else "none"
    df["benchmark_method"] = "classical/DSP baseline"
    return df


def bootstrap_ci(values: np.ndarray, resamples: int, seed: int) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(int(resamples), dtype=float)
    for i in range(int(resamples)):
        means[i] = np.mean(values[rng.integers(0, values.size, size=values.size)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def paired_signflip_p(diff: np.ndarray, permutations: int, seed: int) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("nan")
    observed = abs(float(np.mean(diff)))
    rng = np.random.default_rng(seed)
    null = np.empty(int(permutations), dtype=float)
    for i in range(int(permutations)):
        signs = rng.choice(np.array([-1.0, 1.0]), size=diff.size)
        null[i] = abs(float(np.mean(diff * signs)))
    return float((np.sum(null >= observed) + 1.0) / (len(null) + 1.0))


def aggregate_benchmark(
    root: Path,
    seeds: Sequence[int],
    *,
    classical_csv: Optional[Path],
    require_classical: bool,
    bootstrap_resamples: int,
    permutation_resamples: int,
) -> Dict[str, pd.DataFrame]:
    rows: List[pd.DataFrame] = []
    missing: List[str] = []
    optional_missing: List[str] = []
    if classical_csv is not None:
        rows.append(load_classical_rows(classical_csv))
    elif require_classical:
        raise FileNotFoundError("A classical/DSP baseline is required; pass --classical-csv with subject-level rows.")
    else:
        optional_missing.append("classical/DSP baseline not supplied; pass --classical-csv to include it.")

    for seed in seeds:
        seed_dir = root / f"seed_{seed:03d}"
        baseline_path = seed_dir / "baselines" / "subject_rows.csv"
        f0_path = seed_dir / "f0" / "subject_rows.csv"
        if baseline_path.exists():
            rows.append(load_baseline_rows(baseline_path, seed))
        else:
            missing.append(str(baseline_path))
        if f0_path.exists():
            f0_rows = load_f0_rows(f0_path, seed)
            validate_integrated_f0_lineage(f0_rows, path=f0_path, seed=int(seed))
            rows.append(f0_rows)
        else:
            missing.append(str(f0_path))

    if missing:
        raise FileNotFoundError("Required benchmark inputs are missing:\n" + "\n".join(missing))

    subject_seed = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if subject_seed.empty:
        raise RuntimeError("No benchmark rows were found to aggregate.")
    for col in ("mae", "rmse", "corr"):
        if col not in subject_seed.columns:
            subject_seed[col] = np.nan

    subject_mean = (
        subject_seed.groupby(["benchmark_method", "family", "model", "mode", "subject"], dropna=False, as_index=False)
        .agg(mae=("mae", "mean"), rmse=("rmse", "mean"), corr=("corr", "mean"), n_seeds=("seed", "nunique"))
    )

    table_rows = []
    for method, g in subject_mean.groupby("benchmark_method", dropna=False):
        mae = g["mae"].to_numpy(dtype=float)
        rmse = g["rmse"].to_numpy(dtype=float)
        corr = g["corr"].to_numpy(dtype=float)
        ci_lo, ci_hi = bootstrap_ci(mae, bootstrap_resamples, seed=0)
        table_rows.append(
            {
                "benchmark_method": method,
                "family": str(g["family"].iloc[0]),
                "model": str(g["model"].iloc[0]),
                "mode": str(g["mode"].iloc[0]),
                "n_subjects": int(g["subject"].nunique()),
                "mean_subject_seed_averaged_MAE": float(np.nanmean(mae)),
                "sd_subject_seed_averaged_MAE": float(np.nanstd(mae, ddof=1)) if np.isfinite(mae).sum() > 1 else 0.0,
                "bootstrap95_MAE_low": ci_lo,
                "bootstrap95_MAE_high": ci_hi,
                "mean_subject_seed_averaged_RMSE": float(np.nanmean(rmse)),
                "mean_subject_seed_averaged_corr": float(np.nanmean(corr)),
                "seed_averaging_before_inference": True,
            }
        )
    benchmark_table = pd.DataFrame(table_rows).sort_values("mean_subject_seed_averaged_MAE")

    test_rows = []
    wide = subject_mean.pivot_table(index="subject", columns="benchmark_method", values="mae", aggfunc="mean")
    reference = PRIMARY_REFERENCE
    if reference in wide.columns:
        for method in wide.columns:
            if method == reference:
                continue
            paired = wide[[reference, method]].dropna()
            if paired.empty:
                continue
            diff = paired[method].to_numpy(dtype=float) - paired[reference].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(diff, bootstrap_resamples, seed=len(test_rows) + 11)
            test_rows.append(
                {
                    "reference": reference,
                    "method": method,
                    "difference_method_minus_reference_MAE": float(np.mean(diff)),
                    "bootstrap95_low": lo,
                    "bootstrap95_high": hi,
                    "signflip_p_value": paired_signflip_p(diff, permutation_resamples, seed=len(test_rows) + 101),
                    "n_subjects": int(len(paired)),
                    "unit": "held-out subject after averaging seeds within subject",
                }
            )
    paired_tests = pd.DataFrame(test_rows)

    (root / "benchmark_tables").mkdir(parents=True, exist_ok=True)
    subject_seed.to_csv(root / "benchmark_tables" / "benchmark_subject_seed_rows.csv", index=False)
    subject_mean.to_csv(root / "benchmark_tables" / "benchmark_subject_seed_averaged_rows.csv", index=False)
    benchmark_table.to_csv(root / "benchmark_tables" / "benchmark_table.csv", index=False)
    paired_tests.to_csv(root / "benchmark_tables" / "benchmark_paired_tests_vs_f0_original.csv", index=False)
    (root / "benchmark_tables" / "benchmark_manifest.json").write_text(
        json.dumps(
            {
                "seeds": list(map(int, seeds)),
                "seed_averaging_rule": "Metrics are averaged within each held-out subject across seeds before confidence intervals or significance testing.",
                "primary_reference": PRIMARY_REFERENCE,
                "paired_difference_rule": "comparison MAE - F0 original RR readout MAE; negative means comparison is lower MAE.",
                "a1_hyperparameters": {
                    "gaussian_target_width_bpm": 1.0,
                    "trainable_parameters": 20,
                    "max_epochs": 30,
                    "patience": 5,
                    "checkpoint_selection": "source validation only",
                },
                "missing_optional_inputs": optional_missing,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return {
        "subject_seed": subject_seed,
        "subject_mean": subject_mean,
        "benchmark_table": benchmark_table,
        "paired_tests": paired_tests,
    }


def run_components(args: argparse.Namespace, seeds: Sequence[int]) -> None:
    py = args.python_bin
    subjects = parse_list(args.subjects)
    devices = parse_list(args.devices) if args.devices else [args.device]
    if not devices:
        devices = [args.device]
    job_counter = 0
    component_order = ["baselines", "f0"]
    enabled = {
        "baselines": bool(args.run_baselines),
        "f0": bool(args.run_f0),
    }
    stage_jobs: Dict[str, List[Dict[str, object]]] = {name: [] for name in component_order}
    for seed in seeds:
        seed_dir = Path(args.root) / f"seed_{seed:03d}"
        logs = seed_dir / "logs"
        if args.run_baselines:
            device = devices[job_counter % len(devices)]
            job_counter += 1
            stage_jobs["baselines"].append(
                {
                    "label": f"seed_{seed:03d}:baselines:{device}",
                    "device": device,
                    "log_path": logs / "baselines.log",
                    "cmd": [
                    py,
                    "-u",
                    "run_rr_jbhi_baselines.py",
                    "--models",
                    args.baseline_models,
                    "--subjects",
                    " ".join(subjects),
                    "--data-str",
                    args.data_str,
                    "--data-dir",
                    args.data_dir,
                    "--data-group",
                    args.data_group,
                    "--mdl-dir",
                    args.mdl_dir,
                    "--out-dir",
                    str(seed_dir / "baselines"),
                    "--epochs",
                    str(args.baseline_epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--seed",
                    str(seed),
                    "--device",
                    device,
                    "--num-workers",
                    str(args.num_workers),
                    "--prefetch-factor",
                    str(args.prefetch_factor),
                    "--pin-memory",
                    str(args.pin_memory),
                    "--persistent-workers",
                    str(args.persistent_workers),
                ],
                }
            )
        if args.run_f0:
            device = devices[job_counter % len(devices)]
            job_counter += 1
            stage_jobs["f0"].append(
                {
                    "label": f"seed_{seed:03d}:f0:{device}",
                    "device": device,
                    "log_path": logs / "f0.log",
                    "cmd": [
                    py,
                    "-u",
                    "run_rr_jbhi_f0_family.py",
                    "--variants",
                    "full",
                    "--subjects",
                    *subjects,
                    "--data-str",
                    args.data_str,
                    "--data-dir",
                    args.data_dir,
                    "--data-group",
                    args.data_group,
                    "--mdl-dir",
                    args.mdl_dir,
                    "--out-dir",
                    str(seed_dir / "f0"),
                    "--epochs",
                    str(args.f0_epochs),
                    "--batch-size",
                    str(args.f0_batch_size),
                    "--rr-probe-epochs",
                    str(args.f0_rr_probe_epochs),
                    "--target-calibration-windows",
                    str(args.f0_calibration_windows),
                    "--target-calibration-mode",
                    "random",
                    "--include-calibration-in-eval",
                    "--metric-source",
                    "all",
                    "--seed",
                    str(seed),
                    "--device",
                    device,
                    "--num-workers",
                    str(args.num_workers),
                    "--prefetch-factor",
                    str(args.prefetch_factor),
                    "--pin-memory",
                    str(args.pin_memory),
                    "--persistent-workers",
                    str(args.persistent_workers),
                ],
                }
            )
    for component in component_order:
        if enabled[component] and stage_jobs[component]:
            print(f"[COMPONENT] {component}: {len(stage_jobs[component])} job(s), devices={' '.join(devices)}", flush=True)
            run_job_batch(stage_jobs[component], dry_run=args.dry_run)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-hyperparameter RR benchmark with F0 + 20-parameter A1 readout.")
    parser.add_argument("--root", default="/projects/BLVMob/imu-rr-seated/results/rr_fixed_a1_benchmark")
    parser.add_argument("--stages", nargs="+", default=["baselines", "f0", "table"], help="Any of: baselines f0 table")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--subjects", default=DEFAULT_SUBJECTS)
    parser.add_argument("--baseline-models", default=DEFAULT_BASELINE_MODELS)
    parser.add_argument("--data-str", default="imu_filt")
    parser.add_argument("--data-dir", default="/projects/BLVMob/imu-rr-seated/Data")
    parser.add_argument("--data-group", default="mr")
    parser.add_argument("--mdl-dir", default="/projects/BLVMob/imu-rr-seated/models/imu_filt/loocv")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--devices", nargs="+", default=None, help="CUDA/torch devices used round-robin by seed/component jobs, e.g. cuda:0 cuda:1.")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--f0-batch-size", type=int, default=16)
    parser.add_argument("--readout-batch-size", type=int, default=16, help=argparse.SUPPRESS)
    parser.add_argument("--cached-feature-batch-size", type=int, default=128, help=argparse.SUPPRESS)
    parser.add_argument("--baseline-epochs", type=int, default=60)
    parser.add_argument("--f0-epochs", type=int, default=20)
    parser.add_argument("--f0-rr-probe-epochs", type=int, default=100)
    parser.add_argument("--f0-calibration-windows", type=int, default=32)
    parser.add_argument("--a1-checkpoint-root", default="auto", help=argparse.SUPPRESS)
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("IMU_DATALOADER_WORKERS", "0")))
    parser.add_argument("--prefetch-factor", type=int, default=int(os.environ.get("IMU_DATALOADER_PREFETCH", "1")))
    parser.add_argument("--pin-memory", type=int, default=int(os.environ.get("IMU_DATALOADER_PIN_MEMORY", "0")))
    parser.add_argument("--persistent-workers", type=int, default=int(os.environ.get("IMU_DATALOADER_PERSISTENT_WORKERS", "0")))
    parser.add_argument("--classical-csv", default=None, help="Optional subject-level classical/DSP baseline CSV.")
    parser.add_argument("--require-classical", action="store_true", help="Fail if --classical-csv is not supplied.")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--permutation-resamples", type=int, default=10000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    stages = set(parse_list(args.stages))
    invalid = stages - {"baselines", "f0", "table"}
    if invalid:
        raise SystemExit(f"Unsupported stage(s): {sorted(invalid)}. Standalone A1 is disabled; use the f0 stage with --metric-source all.")
    seeds = [int(x) for x in parse_list(args.seeds)]
    args.run_baselines = "baselines" in stages
    args.run_f0 = "f0" in stages
    args.run_a1 = False
    if args.run_baselines or args.run_f0:
        run_components(args, seeds)
    if "table" in stages:
        outputs = aggregate_benchmark(
            root,
            seeds,
            classical_csv=Path(args.classical_csv) if args.classical_csv else None,
            require_classical=bool(args.require_classical),
            bootstrap_resamples=int(args.bootstrap_resamples),
            permutation_resamples=int(args.permutation_resamples),
        )
        print(outputs["benchmark_table"].to_string(index=False))
        print(f"[DONE] wrote {root / 'benchmark_tables'}")


if __name__ == "__main__":
    main()
