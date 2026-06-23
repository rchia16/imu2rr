#!/usr/bin/env python3
"""Build JBHI-ready result tables from baseline and adaptation outputs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def baseline_table(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(root.glob("**/summary.csv")):
        if p.parent == root:
            continue
        df = pd.read_csv(p)
        if {"model", "mode", "mae_mean"}.issubset(df.columns):
            rows.append(df.assign(source="neural_baseline", file=str(p)))
    if not rows and (root / "summary.csv").exists():
        df = pd.read_csv(root / "summary.csv")
        if {"model", "mode", "mae_mean"}.issubset(df.columns):
            rows.append(df.assign(source="neural_baseline", file=str(root / "summary.csv")))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def adaptation_table(root: Path) -> pd.DataFrame:
    rows = []
    # prototype gate outputs
    for p in sorted(root.glob("**/adaptation_prototype_gate_summary.csv")):
        df = pd.read_csv(p)
        if "policy" in df.columns:
            rows.append(pd.DataFrame({
                "model": "crossmodal_rr",
                "mode": df["policy"],
                "mae_mean": df["post_mae_mean"],
                "mae_std": np.nan,
                "rmse_mean": np.nan,
                "corr_mean": np.nan,
                "n_subjects": df.get("n_subjects", np.nan),
                "source": "prototype_gate_adaptation",
                "file": str(p),
            }))
    # alpha learned outputs if present
    for p in sorted(root.glob("**/adaptation_learned_alpha_summary.csv")):
        df = pd.read_csv(p)
        if "mode" in df.columns:
            rows.append(pd.DataFrame({
                "model": "crossmodal_rr",
                "mode": df["mode"],
                "mae_mean": df["post_mae_mean"],
                "mae_std": np.nan,
                "rmse_mean": np.nan,
                "corr_mean": np.nan,
                "n_subjects": df.get("n_subjects", np.nan),
                "source": "learned_alpha_adaptation",
                "file": str(p),
            }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-root", default="/projects/BLVMob/imu-rr-seated/results/jbhi_rr_baselines")
    p.add_argument("--adaptation-root", default="/projects/BLVMob/imu-rr-seated/results")
    p.add_argument("--out-dir", default="/projects/BLVMob/imu-rr-seated/results/jbhi_tables")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    b = baseline_table(Path(args.baseline_root))
    a = adaptation_table(Path(args.adaptation_root))
    all_rows = pd.concat([x for x in [b, a] if not x.empty], ignore_index=True)
    if all_rows.empty:
        raise SystemExit("No baseline/adaptation summary files found.")
    all_rows = all_rows.sort_values("mae_mean")
    all_rows.to_csv(out_dir / "jbhi_main_results_table.csv", index=False)

    # Compact ranked table for manuscript draft.
    compact = all_rows[["model", "mode", "mae_mean", "mae_std", "rmse_mean", "corr_mean", "n_subjects", "source"]].copy()
    compact.to_csv(out_dir / "jbhi_main_results_table_compact.csv", index=False)
    print(compact.head(30).to_string(index=False))
    print(f"[DONE] wrote {out_dir}")


if __name__ == "__main__":
    main()
