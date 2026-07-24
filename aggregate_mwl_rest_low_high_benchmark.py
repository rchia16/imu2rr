#!/usr/bin/env python3
"""Aggregate rest/low/high MWL prediction rows from A1 and direct-IMU runs."""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from mwl_rest_low_high import REST_LOW_HIGH_CLASS_NAMES, mapping_manifest, parse_list


REQUIRED_PRED_COLS = {
    "subject",
    "seed",
    "sample_id",
    "true_class",
    "predicted_class",
    "prob_rest",
    "prob_low",
    "prob_high",
    "model_name",
    "checkpoint_or_run_path",
}
DIRECT_MODELS = ("resnet1d", "cnn_gru", "tcn", "inceptiontime", "stft_cnn")
PLANNED_COMPARATORS = (*DIRECT_MODELS, "a1_rr_only_logreg")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_prediction_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = sorted(REQUIRED_PRED_COLS - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required prediction columns: {missing}")
    out = df.copy()
    out["source_result_file"] = str(path)
    if "scope" not in out.columns:
        out["scope"] = "window"
    return out


def adapt_hierarchical_window_csv(path: Path, seed: Optional[int]) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"subject", "variant", "sample_index", "true_label", "pred_label", "p_rest", "p_low", "p_high"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is not an adaptable hierarchical MWL prediction file; missing {missing}")
    if seed is None:
        seed = 0
    out = pd.DataFrame(
        {
            "subject": df["subject"].astype(str),
            "seed": int(seed),
            "sample_id": df["subject"].astype(str) + ":window:" + df["sample_index"].astype(int).map(lambda x: f"{x:06d}"),
            "true_class": df["true_label"].astype(int),
            "predicted_class": df["pred_label"].astype(int),
            "prob_rest": df["p_rest"].astype(float),
            "prob_low": df["p_low"].astype(float),
            "prob_high": df["p_high"].astype(float),
            "model_name": df["variant"].astype(str),
            "checkpoint_or_run_path": str(path.parent),
            "source_result_file": str(path),
        }
    )
    if "raw_condition" in df.columns:
        out["raw_condition"] = df["raw_condition"].astype(str)
    out["scope"] = "window"
    return out


def discover_prediction_files(roots: Sequence[Path]) -> List[Path]:
    out: List[Path] = []

    def is_under(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

    for root in roots:
        if root.is_file():
            out.append(root)
        elif root.exists():
            rollup_files = sorted(root.rglob("mwl_predictions.csv"), key=lambda p: (len(p.parts), str(p)))
            selected_rollups: List[Path] = []
            for path in rollup_files:
                parent_dirs = [selected.parent for selected in selected_rollups]
                if any(is_under(path.parent, parent_dir) for parent_dir in parent_dirs):
                    continue
                selected_rollups.append(path)
            out.extend(selected_rollups)

            for path in sorted(root.rglob("*_mwl_predictions_window.csv")):
                if any(is_under(path.parent, selected.parent) for selected in selected_rollups):
                    continue
                out.append(path)
            out.extend(sorted(root.rglob("mwl_one_minute_prediction_rows.csv")))
    return list(dict.fromkeys(out))


def missing_prediction_root_statuses(roots: Sequence[Path], prediction_files: Sequence[Path]) -> List[Dict[str, object]]:
    statuses: List[Dict[str, object]] = []

    def contains(root: Path, path: Path) -> bool:
        if root.is_file():
            return root == path
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    for root in roots:
        if any(contains(root, path) for path in prediction_files):
            continue
        status: Dict[str, object] = {
            "path": str(root),
            "status": "missing",
            "reason": "no compatible MWL prediction files discovered",
        }
        run_status = root / "run_status.csv"
        if root.is_dir() and run_status.exists():
            status["run_status_file"] = str(run_status)
        statuses.append(status)
    return statuses


def load_all_predictions(paths: Sequence[Path]) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    frames: List[pd.DataFrame] = []
    statuses: List[Dict[str, object]] = []
    for path in paths:
        try:
            df = pd.read_csv(path, nrows=5)
            if REQUIRED_PRED_COLS.issubset(df.columns):
                full = load_prediction_csv(path)
            elif {"true_label", "pred_label", "p_rest", "p_low", "p_high", "variant"}.issubset(df.columns):
                seed = infer_seed_from_path(path)
                full = adapt_hierarchical_window_csv(path, seed)
            else:
                statuses.append({"path": str(path), "status": "skipped", "reason": "unrecognized schema"})
                continue
            frames.append(full)
            statuses.append({"path": str(path), "status": "loaded", "rows": int(len(full))})
        except Exception as exc:
            statuses.append({"path": str(path), "status": "failed", "reason": str(exc)})
    if not frames:
        raise RuntimeError("No compatible MWL prediction files were loaded.")
    return pd.concat(frames, ignore_index=True), statuses


def infer_seed_from_path(path: Path) -> Optional[int]:
    for part in path.parts:
        if part.startswith("seed_"):
            try:
                return int(part.split("_")[-1])
            except ValueError:
                pass
    return None


def validate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("true_class", "predicted_class", "seed"):
        out[col] = out[col].astype(int)
    for col in ("prob_rest", "prob_low", "prob_high"):
        out[col] = out[col].astype(float)
    if "scope" not in out.columns:
        out["scope"] = "window"
    out["scope"] = out["scope"].fillna("window").astype(str)
    bad_labels = set(out["true_class"].unique()).union(set(out["predicted_class"].unique())) - {0, 1, 2}
    if bad_labels:
        raise ValueError(f"Unexpected class ids outside 0/1/2: {sorted(bad_labels)}")
    dup_cols = ["scope", "model_name", "subject", "seed", "sample_id"]
    dups = out[out.duplicated(dup_cols, keep=False)]
    if not dups.empty:
        examples = dups[dup_cols].drop_duplicates().head(10).to_dict(orient="records")
        raise ValueError(f"Duplicate model/subject/seed/sample_id rows found, examples={examples}")
    return out


def metrics_dict(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    if y.size == 0:
        return {"accuracy": float("nan"), "balanced_accuracy": float("nan"), "macro_f1": float("nan")}
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "n_windows": int(y.size),
    }


def subject_seed_rows(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scope, model, subject, seed), g in preds.groupby(["scope", "model_name", "subject", "seed"], dropna=False):
        y = g["true_class"].to_numpy(dtype=int)
        pred = g["predicted_class"].to_numpy(dtype=int)
        rows.append({"scope": scope, "model_name": model, "subject": subject, "seed": int(seed), **metrics_dict(y, pred)})
    return pd.DataFrame(rows)


def seed_averaged_prediction_rows(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["scope", "model_name", "subject", "sample_id"]
    for keys, g in preds.groupby(group_cols, dropna=False):
        scope, model, subject, sample_id = keys
        true_values = sorted(g["true_class"].astype(int).unique())
        if len(true_values) != 1:
            raise ValueError(f"Conflicting true labels for {model} {subject} {sample_id}: {true_values}")
        probs = g[["prob_rest", "prob_low", "prob_high"]].to_numpy(dtype=float)
        p = probs.mean(axis=0)
        rows.append(
            {
                "model_name": model,
                "scope": scope,
                "subject": subject,
                "sample_id": sample_id,
                "seed_count": int(g["seed"].nunique()),
                "true_class": int(true_values[0]),
                "predicted_class": int(np.argmax(p)),
                "prob_rest": float(p[0]),
                "prob_low": float(p[1]),
                "prob_high": float(p[2]),
                "checkpoint_or_run_path": " | ".join(sorted(set(g["checkpoint_or_run_path"].astype(str)))),
            }
        )
    return pd.DataFrame(rows)


def subject_seed_averaged_rows(avg_preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scope, model, subject), g in avg_preds.groupby(["scope", "model_name", "subject"], dropna=False):
        y = g["true_class"].to_numpy(dtype=int)
        pred = g["predicted_class"].to_numpy(dtype=int)
        rows.append({"scope": scope, "model_name": model, "subject": subject, **metrics_dict(y, pred), "seed_averaged_inference": True})
    return pd.DataFrame(rows)


def per_class_table_from_rows(preds: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    rows = []
    group_cols = ["model_name", "subject"] if "scope" not in preds.columns else ["scope", "model_name", "subject"]
    for keys, g in preds.groupby(group_cols, dropna=False):
        if len(keys) == 3:
            scope_value, model, subject = keys
        else:
            model, subject = keys
            scope_value = scope
        precision, recall, f1, support = precision_recall_fscore_support(
            g["true_class"].astype(int),
            g["predicted_class"].astype(int),
            labels=[0, 1, 2],
            zero_division=0,
        )
        for class_id, name in enumerate(REST_LOW_HIGH_CLASS_NAMES):
            rows.append(
                {
                    "scope": scope,
                    "prediction_scope": scope_value,
                    "model_name": model,
                    "subject": subject,
                    "class_id": int(class_id),
                    "class_name": name,
                    "precision": float(precision[class_id]),
                    "recall": float(recall[class_id]),
                    "f1": float(f1[class_id]),
                    "support": int(support[class_id]),
                }
            )
    return pd.DataFrame(rows)


def confusion_rows(preds: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    rows = []
    group_cols = ["model_name", "subject"] if "scope" not in preds.columns else ["scope", "model_name", "subject"]
    for keys, g in preds.groupby(group_cols, dropna=False):
        if len(keys) == 3:
            scope_value, model, subject = keys
        else:
            model, subject = keys
            scope_value = scope
        cm = confusion_matrix(g["true_class"].astype(int), g["predicted_class"].astype(int), labels=[0, 1, 2])
        for i, true_name in enumerate(REST_LOW_HIGH_CLASS_NAMES):
            for j, pred_name in enumerate(REST_LOW_HIGH_CLASS_NAMES):
                rows.append(
                    {
                        "scope": scope,
                        "prediction_scope": scope_value,
                        "model_name": model,
                        "subject": subject,
                        "true_class": true_name,
                        "predicted_class": pred_name,
                        "count": int(cm[i, j]),
                    }
                )
    return pd.DataFrame(rows)


def benchmark_table(subject_avg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["scope", "model_name"] if "scope" in subject_avg.columns else ["model_name"]
    for keys, g in subject_avg.groupby(group_cols, dropna=False):
        if isinstance(keys, tuple):
            scope, model = keys
        else:
            scope, model = "window", keys
        rows.append(
            {
                "scope": scope,
                "model_name": model,
                "n_subjects": int(g["subject"].nunique()),
                "mean_subject_accuracy": float(g["accuracy"].mean()),
                "mean_subject_balanced_accuracy": float(g["balanced_accuracy"].mean()),
                "mean_subject_macro_f1": float(g["macro_f1"].mean()),
                "median_subject_macro_f1": float(g["macro_f1"].median()),
                "total_windows": int(g["n_windows"].sum()),
                "primary_metrics": "macro_f1,balanced_accuracy",
            }
        )
    return pd.DataFrame(rows).sort_values(["scope", "mean_subject_macro_f1", "mean_subject_balanced_accuracy"], ascending=[True, False, False])


def signflip_p(diff: np.ndarray, permutations: int, seed: int) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("nan")
    observed = abs(float(np.mean(diff)))
    rng = np.random.default_rng(int(seed))
    null = np.empty(int(permutations), dtype=float)
    for i in range(int(permutations)):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=diff.size)
        null[i] = abs(float(np.mean(diff * signs)))
    return float((np.sum(null >= observed) + 1.0) / (float(permutations) + 1.0))


def bootstrap_ci(values: np.ndarray, resamples: int, seed: int) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    boots = np.empty(int(resamples), dtype=float)
    for i in range(int(resamples)):
        boots[i] = np.median(values[rng.integers(0, values.size, values.size)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def holm_correction(p_values: Sequence[float]) -> List[float]:
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    finite_idx = np.flatnonzero(np.isfinite(p))
    if finite_idx.size == 0:
        return out.tolist()
    order = finite_idx[np.argsort(p[finite_idx])]
    adjusted = np.empty(order.size, dtype=float)
    for rank, idx in enumerate(order):
        adjusted[rank] = min(1.0, (order.size - rank) * p[idx])
    adjusted = np.maximum.accumulate(adjusted)
    for rank, idx in enumerate(order):
        out[idx] = adjusted[rank]
    return out.tolist()


def paired_tests(subject_avg: pd.DataFrame, best_a1: Optional[str], permutations: int, bootstrap_resamples: int) -> pd.DataFrame:
    if not best_a1:
        return pd.DataFrame()
    rows = []
    if "scope" not in subject_avg.columns:
        subject_avg = subject_avg.copy()
        subject_avg["scope"] = "window"
    for scope, scoped in subject_avg.groupby("scope", dropna=False):
        wide = scoped.pivot_table(index="subject", columns="model_name", values="macro_f1", aggfunc="mean")
        if best_a1 not in wide.columns:
            continue
        for comparator in PLANNED_COMPARATORS:
            if comparator not in wide.columns or comparator == best_a1:
                continue
            paired = wide[[best_a1, comparator]].dropna()
            if paired.empty:
                continue
            diff = paired[best_a1].to_numpy(dtype=float) - paired[comparator].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(diff, bootstrap_resamples, seed=len(rows) + 17)
            rows.append(
                {
                    "scope": scope,
                    "reference": best_a1,
                    "comparator": comparator,
                    "metric": "macro_f1",
                    "median_paired_difference_reference_minus_comparator": float(np.median(diff)),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "improved_subjects": int(np.sum(diff > 0)),
                    "tied_subjects": int(np.sum(np.isclose(diff, 0.0))),
                    "worsened_subjects": int(np.sum(diff < 0)),
                    "p_value": signflip_p(diff, permutations, seed=len(rows) + 101),
                    "n_subjects": int(diff.size),
                    "statistical_unit": "held-out subject",
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["holm_p_value"] = holm_correction(df["p_value"].tolist())
    return df


def choose_strongest_a1(table: pd.DataFrame) -> Optional[str]:
    a1 = table[table["model_name"].astype(str).str.startswith("a1_")].copy()
    if "scope" in a1.columns and "window" in set(a1["scope"].astype(str)):
        a1 = a1[a1["scope"].astype(str) == "window"].copy()
    if a1.empty:
        return None
    a1 = a1.sort_values(["mean_subject_macro_f1", "mean_subject_balanced_accuracy"], ascending=False)
    return str(a1.iloc[0]["model_name"])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate rest/low/high MWL benchmark predictions.")
    parser.add_argument("--prediction-roots", nargs="+", required=True, help="Prediction CSVs or roots containing mwl_predictions.csv files.")
    parser.add_argument("--out-dir", default="results/a1_mwl_probe/rest_low_high/combined_benchmark")
    parser.add_argument("--subjects", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--permutation-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    roots = [Path(p) for p in args.prediction_roots]
    files = discover_prediction_files(roots)
    root_status = missing_prediction_root_statuses(roots, files)
    preds, load_status = load_all_predictions(files)
    load_status.extend(root_status)
    preds = validate_predictions(preds)
    if args.subjects:
        keep_subjects = set(parse_list(args.subjects))
        preds = preds[preds["subject"].astype(str).isin(keep_subjects)].copy()
    if args.seeds:
        keep_seeds = {int(v) for v in parse_list(args.seeds)}
        preds = preds[preds["seed"].astype(int).isin(keep_seeds)].copy()
    if preds.empty:
        raise RuntimeError("No prediction rows remain after subject/seed filtering.")
    subject_seed = subject_seed_rows(preds)
    seed_avg_preds = seed_averaged_prediction_rows(preds)
    subject_avg = subject_seed_averaged_rows(seed_avg_preds)
    table = benchmark_table(subject_avg)
    per_class = pd.concat(
        [
            per_class_table_from_rows(preds, scope="window_seed"),
            per_class_table_from_rows(seed_avg_preds, scope="seed_averaged_window"),
        ],
        ignore_index=True,
    )
    confusions = pd.concat(
        [
            confusion_rows(preds, scope="window_seed"),
            confusion_rows(seed_avg_preds, scope="seed_averaged_window"),
        ],
        ignore_index=True,
    )
    best_a1 = choose_strongest_a1(table)
    tests = paired_tests(subject_avg, best_a1, int(args.permutation_resamples), int(args.bootstrap_resamples))
    preds.to_csv(out_dir / "mwl_prediction_rows.csv", index=False)
    subject_seed.to_csv(out_dir / "mwl_subject_seed_rows.csv", index=False)
    seed_avg_preds.to_csv(out_dir / "mwl_seed_averaged_prediction_rows.csv", index=False)
    subject_avg.to_csv(out_dir / "mwl_subject_seed_averaged_rows.csv", index=False)
    table.to_csv(out_dir / "mwl_benchmark_table.csv", index=False)
    per_class.to_csv(out_dir / "mwl_per_class_table.csv", index=False)
    confusions.to_csv(out_dir / "mwl_confusion_matrices.csv", index=False)
    tests.to_csv(out_dir / "mwl_paired_tests.csv", index=False)
    manifest = {
        "mapping": mapping_manifest(),
        "subjects": sorted(preds["subject"].astype(str).unique()),
        "seeds": sorted(int(v) for v in preds["seed"].unique()),
        "model_names": sorted(preds["model_name"].astype(str).unique()),
        "source_result_directories": [str(p) for p in roots],
        "source_result_files": sorted(preds["source_result_file"].astype(str).unique()),
        "load_status": load_status,
        "best_a1_method_for_planned_tests": best_a1,
        "planned_comparators": list(PLANNED_COMPARATORS),
        "seed_averaging_rule": "Rows are aligned by model_name, subject, sample_id; class probabilities are averaged across seeds before argmax and subject-level metrics.",
        "statistical_unit": "held-out subject",
        "git_commit": git_commit(),
        "hostname": platform.node(),
        "python_version": sys.version,
        "pandas_version": pd.__version__,
        "newly_run_or_reused": "Aggregator consumes supplied result directories non-destructively; run status is determined by source manifests when present.",
        "missing_or_failed_runs": [row for row in load_status if row.get("status") != "loaded"],
    }
    (out_dir / "mwl_benchmark_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(table.to_string(index=False))
    print(f"[DONE] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
