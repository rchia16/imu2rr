from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PRIMARY_COMPARISONS = [
    ("stage0", "direct", "hard_spectral"),
    ("stage0", "direct", "soft_spectral"),
    ("stage1", "direct_rr", "best_frozen_readout"),
    ("stage2", "best_frozen_readout", "best_rr_focused_model"),
    ("stage3", "P0_no_profile", "P1_real_subject_profile_film"),
    ("stage3", "P3_shuffled_source_profile_film", "P1_real_subject_profile_film"),
]


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def subject_seed_mae(df: pd.DataFrame, method_col: str, pred_col: str = "rr_pred") -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(["subject", "seed", method_col], dropna=False):
        subject, seed, method = keys
        err = g[pred_col].to_numpy(dtype=float) - g["rr_true"].to_numpy(dtype=float)
        rows.append(
            {
                "subject": subject,
                "seed": seed,
                "method": method,
                "MAE": float(np.mean(np.abs(err))),
                "RMSE": float(np.sqrt(np.mean(err * err))),
                "bias": float(np.mean(err)),
                "n_windows": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap_ci(diff: np.ndarray, resamples: int, seed: int) -> Tuple[float, float]:
    diff = np.asarray(diff, dtype=float).reshape(-1)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(int(resamples), dtype=float)
    n = diff.size
    for i in range(int(resamples)):
        means[i] = np.mean(diff[rng.integers(0, n, size=n)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def paired_summary(
    table: pd.DataFrame,
    method_a: str,
    method_b: str,
    *,
    stage: str,
    metric: str = "MAE",
    resamples: int = 10000,
    seed: int = 0,
) -> Dict[str, object]:
    wide = table.pivot_table(index=["subject", "seed"], columns="method", values=metric, aggfunc="mean")
    if method_b == "best_frozen_readout":
        candidates = [c for c in wide.columns if c != method_a]
        if not candidates:
            return {}
        method_b = wide[candidates].mean(axis=0).sort_values().index[0]
    if method_b == "best_rr_focused_model":
        candidates = [c for c in wide.columns if c != method_a]
        if not candidates:
            return {}
        method_b = wide[candidates].mean(axis=0).sort_values().index[0]
    if method_a not in wide.columns or method_b not in wide.columns:
        return {}
    diff = (wide[method_b] - wide[method_a]).dropna()
    if diff.empty:
        return {}
    lo, hi = paired_bootstrap_ci(diff.to_numpy(), resamples=resamples, seed=seed)
    by_seed = diff.reset_index(name="diff").groupby("seed", as_index=False)["diff"].mean()
    values = diff.to_numpy(dtype=float)
    q25, q75 = np.percentile(values, [25, 75])
    return {
        "stage": stage,
        "metric": metric,
        "method_a": method_a,
        "method_b": method_b,
        "difference_b_minus_a_mean": float(np.mean(values)),
        "difference_b_minus_a_sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "difference_b_minus_a_median": float(np.median(values)),
        "difference_b_minus_a_iqr": float(q75 - q25),
        "subject_seed_pairs": int(values.size),
        "subject_win_count": int(np.sum(values < 0.0)),
        "seed_wise_mean": float(by_seed["diff"].mean()) if not by_seed.empty else float("nan"),
        "seed_wise_sd": float(by_seed["diff"].std(ddof=1)) if len(by_seed) > 1 else 0.0,
        "bootstrap_ci95_low": lo,
        "bootstrap_ci95_high": hi,
        "interpretation": "negative favors method_b",
    }


def stage0_table(out_dir: Path) -> pd.DataFrame:
    path = out_dir / "stage0_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        for method, col in (
            ("direct", "direct_MAE"),
            ("hard_spectral", "hard_spectral_MAE"),
            ("soft_spectral", "soft_spectral_MAE"),
            ("oracle_true_spectrum_hard", "oracle_true_spectrum_hard_MAE"),
        ):
            if col in r:
                rows.append({"subject": r["subject"], "seed": r.get("seed", 0), "method": method, "MAE": r[col]})
    return pd.DataFrame(rows)


def stage1_table(out_dir: Path) -> pd.DataFrame:
    path = out_dir / "stage1_per_subject.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df.rename(columns={"method": "method"})[["subject", "seed", "method", "MAE", "RMSE", "bias"]]


def stage2_table(out_dir: Path) -> pd.DataFrame:
    path = out_dir / "stage2_per_subject.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "checkpoint_selection" in df.columns:
        df = df[df["checkpoint_selection"].isin(["best_rr_mae", "frozen_stage1_reference"])]
    df = df.copy()
    df["method"] = df["ablation"].astype(str)
    return df[["subject", "seed", "method", "MAE", "RMSE", "bias"]]


def stage3_table(out_dir: Path) -> pd.DataFrame:
    path = out_dir / "stage3_per_subject.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path).copy()
    df["method"] = df["control"].astype(str)
    return df[["subject", "seed", "method", "MAE", "RMSE", "bias"]]


def final_method_comparison(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for stage, table in tables.items():
        if table.empty:
            continue
        for method, g in table.groupby("method"):
            rows.append(
                {
                    "stage": stage,
                    "method": method,
                    "mean_subject_seed_MAE": float(g["MAE"].mean()),
                    "sd_subject_seed_MAE": float(g["MAE"].std(ddof=1)) if len(g) > 1 else 0.0,
                    "median_subject_seed_MAE": float(g["MAE"].median()),
                    "n_subject_seed": int(len(g)),
                }
            )
    return pd.DataFrame(rows)


def final_subject_comparison(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for stage, table in tables.items():
        if table.empty:
            continue
        for (subject, seed), g in table.groupby(["subject", "seed"]):
            best = g.sort_values("MAE").iloc[0]
            rows.append(
                {
                    "stage": stage,
                    "subject": subject,
                    "seed": seed,
                    "best_method": best["method"],
                    "best_MAE": float(best["MAE"]),
                }
            )
    return pd.DataFrame(rows)


def final_seed_summary(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for stage, table in tables.items():
        if table.empty:
            continue
        for (seed, method), g in table.groupby(["seed", "method"]):
            rows.append(
                {
                    "stage": stage,
                    "seed": seed,
                    "method": method,
                    "mean_MAE": float(g["MAE"].mean()),
                    "sd_MAE": float(g["MAE"].std(ddof=1)) if len(g) > 1 else 0.0,
                    "n_subjects": int(g["subject"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def run_analysis(out_dir: Path, bootstrap_resamples: int = 10000, seed: int = 0) -> Dict[str, pd.DataFrame]:
    out_dir = Path(out_dir)
    tables = {
        "stage0": stage0_table(out_dir),
        "stage1": stage1_table(out_dir),
        "stage2": stage2_table(out_dir),
        "stage3": stage3_table(out_dir),
    }
    final_method = final_method_comparison(tables)
    final_subject = final_subject_comparison(tables)
    final_seed = final_seed_summary(tables)
    final_method.to_csv(out_dir / "final_method_comparison.csv", index=False)
    final_subject.to_csv(out_dir / "final_subject_comparison.csv", index=False)
    final_seed.to_csv(out_dir / "final_seed_summary.csv", index=False)

    comparison_rows: List[dict] = []
    for stage, a, b in PRIMARY_COMPARISONS:
        table = tables.get(stage, pd.DataFrame())
        if table.empty:
            continue
        row = paired_summary(
            table,
            a,
            b,
            stage=stage,
            resamples=int(bootstrap_resamples),
            seed=int(seed) + len(comparison_rows),
        )
        if row:
            comparison_rows.append(row)
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(out_dir / "paired_subject_seed_comparisons.csv", index=False)
    write_json(
        out_dir / "analysis_manifest.json",
        {
            "bootstrap_resamples": int(bootstrap_resamples),
            "seed": int(seed),
            "unit_of_generalisation": "held-out subject; subject-seed pairs are paired before aggregation",
            "window_pooling_for_inference": False,
        },
    )
    return {
        "final_method_comparison": final_method,
        "final_subject_comparison": final_subject,
        "final_seed_summary": final_seed,
        "paired_subject_seed_comparisons": comparisons,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subject-level analysis for RR spectral readout experiments.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    outputs = run_analysis(Path(args.out_dir), bootstrap_resamples=args.bootstrap_resamples, seed=args.seed)
    print("Wrote analysis outputs:")
    for name, df in outputs.items():
        print(f"  {name}: {len(df)} rows")


if __name__ == "__main__":
    main()
