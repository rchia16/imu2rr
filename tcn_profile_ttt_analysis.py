#!/usr/bin/env python3
"""Subject-level analysis for the standalone TCN profile/TTT experiment."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PRIMARY_COMPARISONS = [
    ("C2_real_profile_affine", "C0_plain_tcn", "T1 versus T0"),
    ("C4_real_profile_film", "C0_plain_tcn", "T2 versus T0"),
    ("C6_real_profile_film_affine", "C0_plain_tcn", "T3 versus T0"),
    ("T4_affine_only_ttt", "C2_real_profile_affine", "T4 versus T1"),
    ("T5_film_affine_ttt", "C6_real_profile_film_affine", "T5 versus T3"),
    ("M2_t6_meta_trained_gate", "T5_film_affine_ttt", "T6 versus T5"),
    ("M2_t6_meta_trained_gate", "C6_real_profile_film_affine", "T6 versus T3"),
    ("M2_t6_meta_trained_gate", "M6_t6_shuffled_matched_profile", "T6 real profile versus shuffled profile"),
    ("M2_t6_meta_trained_gate", "M4_t6_gate_fixed_1", "T6 gated versus gate fixed to 1"),
    ("M2_t6_meta_trained_gate", "M3_t6_gate_fixed_0", "T6 gated versus gate fixed to 0"),
    ("A1_t6_profile_gated_mean_alignment", "M2_t6_meta_trained_gate", "T7 versus T6"),
    ("A1_t6_profile_gated_mean_alignment", "A3_t6_fixed_alpha_0_50", "T7 versus fixed mean alignment"),
    ("A1_t6_profile_gated_mean_alignment", "A7_t6_shuffled_profile_alpha", "T7 real profile versus shuffled-profile alpha"),
]


def bootstrap_ci(values: np.ndarray, seed: int, n_bootstrap: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(n_bootstrap), dtype=float)
    for i in range(int(n_bootstrap)):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = float(np.mean(sample))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def method_summary(df: pd.DataFrame, seed: int, n_bootstrap: int) -> pd.DataFrame:
    rows = []
    for method, g in df.groupby("method"):
        mae = g["mae"].astype(float).to_numpy()
        lo, hi = bootstrap_ci(mae, seed, n_bootstrap)
        rows.append({
            "method": method,
            "mean_mae": float(np.nanmean(mae)),
            "mae_std": float(np.nanstd(mae, ddof=0)),
            "median_mae": float(np.nanmedian(mae)),
            "mae_iqr": float(np.nanpercentile(mae, 75) - np.nanpercentile(mae, 25)),
            "rmse": float(np.nanmean(g["rmse"])),
            "rr_corr": float(np.nanmean(g["rr_corr"])),
            "bias": float(np.nanmean(g["bias"])),
            "median_ae": float(np.nanmean(g["median_ae"])),
            "p95_ae": float(np.nanmean(g["p95_ae"])),
            "n_subject_seed_units": int(len(g)),
            "bootstrap_mae_ci_low": lo,
            "bootstrap_mae_ci_high": hi,
        })
    return pd.DataFrame(rows)


def paired_comparisons(df: pd.DataFrame, seed: int, n_bootstrap: int) -> pd.DataFrame:
    key_cols = ["seed", "subject"]
    rows = []
    for method, baseline, label in PRIMARY_COMPARISONS:
        a = df[df["method"] == method][key_cols + ["mae"]].rename(columns={"mae": "method_mae"})
        b = df[df["method"] == baseline][key_cols + ["mae"]].rename(columns={"mae": "baseline_mae"})
        joined = a.merge(b, on=key_cols, how="inner")
        if joined.empty:
            rows.append({"comparison": label, "method": method, "baseline": baseline, "available": False})
            continue
        delta = joined["method_mae"].astype(float).to_numpy() - joined["baseline_mae"].astype(float).to_numpy()
        lo, hi = bootstrap_ci(delta, seed, n_bootstrap)
        rows.append({
            "comparison": label,
            "method": method,
            "baseline": baseline,
            "available": True,
            "mean_delta_mae": float(np.nanmean(delta)),
            "median_delta_mae": float(np.nanmedian(delta)),
            "subject_wins": int(np.sum(delta < 0)),
            "subject_losses": int(np.sum(delta > 0)),
            "largest_improvement": float(np.nanmin(delta)),
            "largest_degradation": float(np.nanmax(delta)),
            "paired_bootstrap_ci_low": lo,
            "paired_bootstrap_ci_high": hi,
            "n_pairs": int(len(delta)),
        })
    return pd.DataFrame(rows)


def reliability_analysis(run_dir: Path, subject_df: pd.DataFrame) -> pd.DataFrame:
    diag_path = run_dir / "ttt_diagnostics.csv"
    if not diag_path.exists():
        return pd.DataFrame([{"reason": "ttt_diagnostics.csv unavailable"}])
    diag = pd.read_csv(diag_path)
    if diag.empty:
        return pd.DataFrame([{"reason": "no TTT diagnostics rows"}])
    cols = [c for c in ["gate_value", "profile_distance", "ttt_loss_pre", "ttt_loss_post", "ttt_delta_norm"] if c in diag.columns]
    rows = []
    for col in cols:
        vals = pd.to_numeric(diag[col], errors="coerce")
        rows.append({"feature": col, "mean": float(vals.mean()), "std": float(vals.std(ddof=0)), "n": int(vals.notna().sum())})
    return pd.DataFrame(rows)


def shift_classification(subject_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, subject), g in subject_df.groupby(["seed", "subject"]):
        best = g.sort_values("mae").iloc[0]
        method = str(best["method"])
        if "affine" in method.lower():
            label = "RR gain/bias shift"
        elif "alignment" in method.lower():
            label = "latent mean shift"
        elif "film" in method.lower() or "t6" in method.lower():
            label = "mixed low-dimensional shift"
        else:
            label = "not explained by low-dimensional correction"
        rows.append({"seed": seed, "subject": subject, "best_method": method, "shift_classification": label})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--bootstrap-resamples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run_dir = Path(args.run_dir)
    subject_path = run_dir / "subject_rows.csv"
    if not subject_path.exists():
        raise FileNotFoundError(subject_path)
    subject_df = pd.read_csv(subject_path)
    summary = method_summary(subject_df, int(args.seed), int(args.bootstrap_resamples))
    paired = paired_comparisons(subject_df, int(args.seed) + 17, int(args.bootstrap_resamples))
    summary.to_csv(run_dir / "final_method_comparison.csv", index=False)
    paired.to_csv(run_dir / "final_subject_comparison.csv", index=False)
    seed_summary = subject_df.groupby(["seed", "method"], as_index=False).agg(mae_mean=("mae", "mean"), mae_std=("mae", "std"), n_subjects=("subject", "nunique"))
    seed_summary.to_csv(run_dir / "final_seed_summary.csv", index=False)
    reliability_analysis(run_dir, subject_df).to_csv(run_dir / "adaptation_reliability_analysis.csv", index=False)
    shift_classification(subject_df).to_csv(run_dir / "subject_shift_classification.csv", index=False)
    print(f"[DONE] wrote final analysis files in {run_dir}")


if __name__ == "__main__":
    main()
