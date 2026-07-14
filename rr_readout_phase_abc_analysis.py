from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PRIMARY_COMPARISONS = [
    ("A1_gaussian_kl", "A0_soft_spectral"),
    ("A2_wasserstein", "A0_soft_spectral"),
    ("A3_kl_rr_mae", "A0_soft_spectral"),
    ("A5_spec_linear_residual", "A0_soft_spectral"),
    ("A6_spec_hidden_conf_residual", "A0_soft_spectral"),
    ("A6_spec_hidden_conf_residual", "A4_hidden_only"),
    ("B3_soft_gate", "B0_always_spec"),
    ("B3_soft_gate", "B1_always_hybrid"),
    ("C1_feature_mean", "C0_none"),
    ("C2_temperature", "C0_none"),
    ("C3_affine_readout", "C0_none"),
    ("C4_confidence_gated_affine", "C0_none"),
]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def metric_table(path: Path, phase: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["phase"] = phase
    return df


def paired_bootstrap_ci(diff: np.ndarray, resamples: int, seed: int) -> Tuple[float, float]:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(int(resamples), dtype=float)
    for i in range(int(resamples)):
        means[i] = np.mean(diff[rng.integers(0, diff.size, size=diff.size)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def final_method_comparison(table: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    if table.empty:
        return pd.DataFrame()
    for (phase, method), g in table.groupby(["phase", "method"]):
        vals = g["MAE"].to_numpy(dtype=float)
        q25, q75 = np.percentile(vals, [25, 75]) if vals.size else (np.nan, np.nan)
        rows.append(
            {
                "phase": phase,
                "method": method,
                "mean_subject_seed_MAE": float(np.mean(vals)),
                "sd_subject_seed_MAE": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                "median_subject_seed_MAE": float(np.median(vals)),
                "iqr_subject_seed_MAE": float(q75 - q25),
                "RMSE": float(g["RMSE"].mean()) if "RMSE" in g else float("nan"),
                "bias": float(g["bias"].mean()) if "bias" in g else float("nan"),
                "Pearson_correlation": float(g["Pearson_correlation"].mean()) if "Pearson_correlation" in g else float("nan"),
                "number_of_subject_seed_rows": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def paired_comparisons(table: pd.DataFrame, resamples: int, seed: int) -> pd.DataFrame:
    rows: List[dict] = []
    if table.empty:
        return pd.DataFrame()
    for idx, (method_a, method_b) in enumerate(PRIMARY_COMPARISONS):
        sub = table[table["method"].isin([method_a, method_b])]
        if sub.empty:
            continue
        wide = sub.pivot_table(index=["subject", "seed"], columns="method", values="MAE", aggfunc="mean")
        if method_a not in wide.columns or method_b not in wide.columns:
            continue
        diff = (wide[method_a] - wide[method_b]).dropna()
        if diff.empty:
            continue
        values = diff.to_numpy(dtype=float)
        lo, hi = paired_bootstrap_ci(values, resamples, seed + idx)
        rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "difference_a_minus_b_mean": float(np.mean(values)),
                "difference_a_minus_b_sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "difference_a_minus_b_median": float(np.median(values)),
                "subject_seed_pairs": int(values.size),
                "bootstrap_ci95_low": lo,
                "bootstrap_ci95_high": hi,
                "interpretation": "negative favors method_a",
            }
        )
    return pd.DataFrame(rows)


def subject_winners(table: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    if table.empty:
        return pd.DataFrame()
    for (phase, subject, seed), g in table.groupby(["phase", "subject", "seed"]):
        best = g.sort_values("MAE").iloc[0]
        rows.append({"phase": phase, "subject": subject, "seed": seed, "best_method": best["method"], "best_MAE": float(best["MAE"])})
    return pd.DataFrame(rows)


def seed_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    if table.empty:
        return pd.DataFrame()
    for (phase, seed, method), g in table.groupby(["phase", "seed", "method"]):
        rows.append(
            {
                "phase": phase,
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
    tables = [
        metric_table(out_dir / "phase_a_per_subject.csv", "A"),
        metric_table(out_dir / "phase_b_per_subject.csv", "B"),
        metric_table(out_dir / "phase_c_per_subject.csv", "C"),
    ]
    all_subject = pd.concat([t for t in tables if not t.empty], ignore_index=True) if any(not t.empty for t in tables) else pd.DataFrame()
    final = final_method_comparison(all_subject)
    pairs = paired_comparisons(all_subject, int(bootstrap_resamples), int(seed))
    winners = subject_winners(all_subject)
    seeds = seed_summary(all_subject)
    final.to_csv(out_dir / "phase_abc_final_method_comparison.csv", index=False)
    pairs.to_csv(out_dir / "phase_abc_paired_comparisons.csv", index=False)
    winners.to_csv(out_dir / "phase_abc_subject_winners.csv", index=False)
    seeds.to_csv(out_dir / "phase_abc_seed_summary.csv", index=False)
    write_json(
        out_dir / "phase_abc_analysis_manifest.json",
        {
            "bootstrap_resamples": int(bootstrap_resamples),
            "seed": int(seed),
            "unit_of_generalisation": "held-out subject; subject-seed rows are paired before inference",
            "window_pooling_for_inference": False,
        },
    )
    return {
        "phase_abc_final_method_comparison": final,
        "phase_abc_paired_comparisons": pairs,
        "phase_abc_subject_winners": winners,
        "phase_abc_seed_summary": seeds,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subject-level analysis for RR readout Phase A/B/C outputs.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    outputs = run_analysis(Path(args.out_dir), bootstrap_resamples=args.bootstrap_resamples, seed=args.seed)
    for name, df in outputs.items():
        print(f"{name}: {len(df)} rows")


if __name__ == "__main__":
    main()
