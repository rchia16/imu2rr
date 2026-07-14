#!/usr/bin/env python3
"""Run the seeded F0 RR family: full model plus three ablations.

This helper runs the cross-modal LOSO benchmark once per requested variant and
writes:
  - per-variant raw outputs under <out-dir>/<variant>/
  - a combined subject-row export at <out-dir>/subject_rows.csv
  - a combined summary table at <out-dir>/summary.csv
  - a manifest describing the exact variant settings
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from vit_pressure_crossmodal_stft_rr_core import build_base_parser, run_loocv_experiment


DEFAULT_SUBJECTS = "S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29"
DEFAULT_VARIANTS = "full no_film no_tcn no_qkv"


VARIANT_CONFIGS: Dict[str, Dict[str, object]] = {
    "full": {
        "use_profile_film": True,
        "use_profile_qkv": True,
        "shared_profile_qkv": True,
        "profile_conditioning": "film_qkv",
        "use_tcn_token_mixer": True,
        "tcn_mixer_alpha": 0.05,
        "profile_qkv_layers": "last1",
        "profile_qkv_scale": 0.01,
        "profile_qkv_residual": True,
    },
    "no_film": {
        "use_profile_film": False,
        "use_profile_qkv": True,
        "shared_profile_qkv": False,
        "profile_conditioning": "qkv",
        "use_tcn_token_mixer": True,
        "tcn_mixer_alpha": 0.05,
        "profile_qkv_layers": "last1",
        "profile_qkv_scale": 0.01,
        "profile_qkv_residual": True,
    },
    "no_tcn": {
        "use_profile_film": True,
        "use_profile_qkv": True,
        "shared_profile_qkv": True,
        "profile_conditioning": "film_qkv",
        "use_tcn_token_mixer": False,
        "tcn_mixer_alpha": 0.05,
        "profile_qkv_layers": "last1",
        "profile_qkv_scale": 0.01,
        "profile_qkv_residual": True,
    },
    "no_qkv": {
        "use_profile_film": True,
        "use_profile_qkv": True,
        "shared_profile_qkv": True,
        "profile_conditioning": "film_qkv",
        "use_tcn_token_mixer": True,
        "tcn_mixer_alpha": 0.05,
        "profile_qkv_layers": "none",
        "profile_qkv_scale": 0.01,
        "profile_qkv_residual": True,
    },
}


def parse_list(text: str) -> List[str]:
    return [t.strip() for t in str(text).replace(",", " ").split() if t.strip()]


def build_variant_args(args, variant: str):
    variant_args = copy.deepcopy(args)
    variant_args.out_dir = str(Path(args.out_dir) / variant)
    for key, value in VARIANT_CONFIGS[variant].items():
        setattr(variant_args, key, value)
    return variant_args


def summarize_subject_rows(df: pd.DataFrame, *, family: str, mode: str, model: str, seed: int) -> Dict[str, object]:
    numeric_cols = [
        col
        for col in df.columns
        if col not in {"subject", "family", "model", "mode", "seed", "variant"}
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    row: Dict[str, object] = {
        "family": family,
        "model": model,
        "mode": mode,
        "variant": mode,
        "seed": int(seed),
        "n_subjects": int(df["subject"].nunique()),
    }
    for col in numeric_cols:
        row[f"{col}_mean"] = float(df[col].mean())
        row[f"{col}_std"] = float(df[col].std())
    return row


def normalize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {}
    for src, dst in (("rr_mae", "mae"), ("rr_rmse", "rmse"), ("rr_corr", "corr")):
        if src in out.columns and dst not in out.columns:
            rename_map[src] = dst
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def main() -> None:
    parser = build_base_parser(DEFAULT_SUBJECTS.split(), "rr_jbhi_f0_family")
    parser.set_defaults(decoder_mode="cross_attn", rr_head_type="mlp")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS, help="Space- or comma-separated F0 variants to run.")
    args = parser.parse_args()

    variants = parse_list(args.variants)
    invalid = [variant for variant in variants if variant not in VARIANT_CONFIGS]
    if invalid:
        raise SystemExit(f"Unsupported F0 variant(s): {invalid}. Valid variants: {sorted(VARIANT_CONFIGS)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_rows = []
    summary_rows = []
    variant_specs = []

    for variant in variants:
        variant_args = build_variant_args(args, variant)
        variant_dir = Path(variant_args.out_dir)
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_specs.append({"variant": variant, "out_dir": str(variant_dir), "config": VARIANT_CONFIGS[variant]})

        outputs = run_loocv_experiment(variant_args)
        subject_df = normalize_metrics(outputs["summary"])
        subject_df = subject_df.copy()
        subject_df["family"] = "f0"
        subject_df["model"] = "crossmodal_rr"
        subject_df["mode"] = variant
        subject_df["variant"] = variant
        subject_df["seed"] = int(args.seed)
        combined_rows.append(subject_df)

        summary_rows.append(
            summarize_subject_rows(
                subject_df,
                family="f0",
                mode=variant,
                model="crossmodal_rr",
                seed=int(args.seed),
            )
        )

    combined_subject_rows = pd.concat(combined_rows, ignore_index=True) if combined_rows else pd.DataFrame()
    combined_summary = pd.DataFrame(summary_rows)

    combined_subject_rows.to_csv(out_dir / "subject_rows.csv", index=False)
    combined_summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "family": "f0",
                "seed": int(args.seed),
                "variants": variants,
                "variant_configs": variant_specs,
                "out_dir": str(out_dir),
                "subjects": args.subjects,
                "data_str": args.data_str,
                "data_dir": args.data_dir,
                "data_group": args.data_group,
                "mdl_dir": args.mdl_dir,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(combined_summary.to_string(index=False))
    print(f"[DONE] wrote {out_dir}")


if __name__ == "__main__":
    main()
