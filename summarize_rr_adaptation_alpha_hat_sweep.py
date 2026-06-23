#!/usr/bin/env python3
"""Recursive summarizer for unsupervised RR feature-adaptive ladder sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

DEFAULT_ORDER = [
    "none",
    "direct_stft_rr",
    "hybrid_probe_stft_conf",
    "adapt_mean_alpha_000",
    "adapt_mean_alpha_025",
    "adapt_mean_alpha_050",
    "adapt_mean_alpha_075",
    "adapt_mean_alpha_100",
    "adapt_mean_fixed",
    "adapt_mean_profile_shrink",
    "affine_cal",
    "affine_cal_ridge",
    "affine_cal_monotone",
    "ridge_residual_cal",
    "rrbin_centroid_cal",
    "rrbin_ssa",
    "feature_mean_align",
    "feature_diag_align",
    "feature_rr_orthogonal_align",
    "feature_rrbin_diag_align",
    "ln_affine_cal",
    "ln_stft_rr_consistency",
    "lastblock_ln_affine_cal",
    "ln_unsup_stft_consistency",
    "lastblock_ln_unsup_stft_consistency",
    "lastblock_ln_unsup_stft_smooth",
    "lastblock_ln_unsup_stft_smooth_anchor",
    "profile_film_init_only",
    "profile_film_gated_init_only",
    "profile_film_oracle_init_only",
    "profile_film_unsup_stft_consistency",
    "profile_film_unsup_stft_smooth_anchor",
    "profile_film_unsup_readout_affine",
    "profile_film_unsup_sparc",
    "profile_film_gated_sparc",
    "profile_film_oracle_sparc",
    "ssa",
    "cmt",
    "ssa_cmt",
    "kin_ssa",
    "ssa_kin",
]

DIAG_COLS = [
    "affine_a",
    "affine_b",
    "affine_lambda_a",
    "affine_lambda_b",
    "affine_cal_mae",
    "affine_cal_rmse",
    "affine_cal_corr",
    "residual_n_features",
    "residual_ridge_lambda",
    "residual_coef_l2",
    "residual_intercept",
    "residual_cal_mae",
    "residual_cal_rmse",
    "residual_cal_corr",
    "rrbin_n_bins",
    "rrbin_shrink",
    "rrbin_global_offset",
    "rrbin_min_count",
    "rrbin_max_count",
    "rrbin_cal_mae",
    "rrbin_cal_rmse",
    "rrbin_cal_corr",
    "rrbin_moment_min_count",
    "rrbin_moment_max_count",
    "rrbin_used_all_target_unlabeled",
    "ssa_loss_last",
    "ssa_rank_used",
    "ssa_dim_weight_mean",
    "ssa_dim_weight_max",
    "ssa_ssa_loss_last",
    "ssa_ssa_rank_used",
    "ssa_ssa_dim_weight_mean",
    "ssa_ssa_dim_weight_max",
    "cmt_aug_n",
    "cmt_aug_rr_mean",
    "cmt_aug_rr_std",
    "cmt_acceptance_ratio",
    "kin_source_n",
    "feature_tta_epoch",
    "feature_loss_last",
    "feature_cal_loss_last",
    "feature_stft_consistency_last",
    "feature_drift_loss_last",
    "feature_smoothness_last",
    "feature_trainable_params",
    "feature_trainable_layernorm_params",
    "feature_trainable_spec_proj_params",
    "feature_trainable_last_block_params",
    "feature_trainable_rr_head_params",
    "feature_trainable_pressure_decoder_params",
    "feature_tta_epochs",
    "feature_tta_lr",
    "feature_affine_a",
    "feature_affine_b",
    "feature_affine_lambda_a",
    "feature_affine_lambda_b",
    "feature_stft_consistency_weight",
    "feature_source_anchor_weight",
    "feature_target_drift_weight",
    "feature_smoothness_weight",
    "feature_cal_mae",
    "feature_cal_rmse",
    "feature_cal_stft_rr_mae_vs_label",
    "feature_cal_aux_rr_mae_vs_label",
    "eval_stft_rr_mae_vs_label",
    "eval_aux_rr_mae_vs_label",
    "is_unsupervised_feature_tta",
    "uses_target_rr_labels_for_adaptation",
    "unsup_feature_tta_epoch",
    "unsup_feature_loss_last",
    "unsup_stft_consistency_last",
    "unsup_smoothness_last",
    "unsup_target_drift_last",
    "unsup_source_anchor_last",
    "unsup_stft_confidence_mean_last",
    "unsup_feature_tta_epochs",
    "unsup_feature_tta_lr",
    "unsup_stft_consistency_weight",
    "unsup_smoothness_weight",
    "unsup_source_anchor_weight",
    "unsup_target_drift_weight",
    "unsup_stft_confidence_floor",
    "unsup_stft_confidence_power",
    "eval_stft_confidence_mean",
    "profile_unsup_n_adapt_windows",
    "unsup_readout_affine_enabled",
    "unsup_aux_consistency_weight",
    "unsup_aux_consistency_last",
    "unsup_rr_range_weight",
    "unsup_rr_range_min_std",
    "unsup_rr_range_loss_last",
    "unsup_readout_affine_lambda_a",
    "unsup_readout_affine_lambda_b",
    "unsup_readout_affine_prior_last",
    "readout_affine_a",
    "readout_affine_b",
    "profile_delta_norm",
    "profile_vector_norm",
    "profile_vector_prior_l2",
    "profile_prior_loss_last",
    "unsup_reliable_window_ratio",
    "unsup_rr_disagreement_mean",
    "rr_probe_profile_init_mae",
    "rr_probe_profile_init_rmse",
    "rr_probe_profile_init_corr",
    "rr_probe_profile_candidate_mae",
    "rr_probe_profile_candidate_rmse",
    "rr_probe_profile_candidate_corr",
    "profile_candidate_eval_mae_vs_label",
    "profile_init_adapt_mae_vs_label",
    "profile_base_adapt_mae_vs_label",
    "profile_selected_profile",
    "analysis_oracle_uses_target_labels_for_selection",
    "profile_gate_n_windows",
    "profile_gate_init_shift_mean_bpm",
    "profile_gate_init_shift_p95_bpm",
    "profile_gate_max_init_shift_bpm",
    "profile_gate_aux_stft_disagreement_mean_bpm",
    "profile_gate_aux_stft_disagreement_p95_bpm",
    "profile_gate_aux_stft_disagreement_max_bpm",
    "profile_gate_stft_confidence_mean",
    "profile_gate_stft_confidence_frac_above_threshold",
    "profile_gate_min_stft_confidence",
    "profile_gate_profile_rms_z_max",
    "profile_gate_profile_max_abs_z",
    "profile_gate_shift_ok",
    "profile_gate_aux_stft_ok",
    "profile_gate_confidence_ok",
    "profile_gate_stats_ok",
    "profile_gate_pass",
    "profile_stats_z_rms",
    "profile_stats_z_max_abs",
    "profile_oracle_base_cal_mae",
    "profile_oracle_init_cal_mae",
    "profile_oracle_delta_init_minus_base_mae",
    "profile_oracle_cal_tolerance_bpm",
    "profile_oracle_pass",
    "profile_oracle_uses_target_labels",
    "hybrid_use_stft_fraction",
    "hybrid_stft_confidence_threshold",
    "hybrid_aux_stft_disagreement_bpm",
    "hybrid_probe_stft_disagreement_bpm",
    "hybrid_aux_stft_disagreement_mean_bpm",
    "hybrid_probe_stft_disagreement_mean_bpm",
    # Directional profile-shift diagnostics. Label-derived residual fields are
    # analysis-only and are never used for adaptation.
    "eval_direction_n",
    "eval_direction_profile_shift_signed_bpm_mean",
    "eval_direction_profile_shift_signed_bpm_median",
    "eval_direction_profile_shift_signed_bpm_std",
    "eval_direction_profile_shift_signed_bpm_p05",
    "eval_direction_profile_shift_signed_bpm_p95",
    "eval_direction_profile_shift_signed_bpm_abs_mean",
    "eval_direction_profile_shift_signed_bpm_abs_p95",
    "eval_direction_base_residual_signed_bpm_mean",
    "eval_direction_profile_residual_signed_bpm_mean",
    "eval_direction_direction_dot_mean",
    "eval_direction_direction_agree_frac",
    "eval_direction_overshoot_frac",
    "eval_direction_profile_reduces_abs_error_frac",
    "eval_direction_profile_delta_mae",
    "eval_direction_post_shift_signed_bpm_mean",
    "eval_direction_post_shift_signed_bpm_abs_mean",
    "eval_direction_post_direction_agree_frac",
    "eval_direction_post_overshoot_frac",
    "eval_direction_post_reduces_abs_error_frac",
    "eval_direction_post_delta_mae",
    "eval_direction_aux_minus_base_bpm_mean",
    "eval_direction_stft_minus_base_bpm_mean",
    "eval_direction_weighted_physio_minus_base_bpm_mean",
    "eval_direction_profile_aux_sign_agree_frac",
    "eval_direction_profile_stft_sign_agree_frac",
    "eval_direction_profile_weighted_physio_sign_agree_frac",
    "eval_direction_aux_base_residual_sign_agree_frac",
    "eval_direction_stft_base_residual_sign_agree_frac",
    "eval_direction_weighted_physio_base_residual_sign_agree_frac",
    "eval_direction_stft_confidence_mean",
    "eval_direction_stft_confidence_frac_gt_005",
    "eval_direction_stft_confidence_frac_gt_010",
    "cal_direction_n",
    "cal_direction_profile_shift_signed_bpm_mean",
    "cal_direction_profile_shift_signed_bpm_abs_mean",
    "cal_direction_base_residual_signed_bpm_mean",
    "cal_direction_direction_dot_mean",
    "cal_direction_direction_agree_frac",
    "cal_direction_overshoot_frac",
    "cal_direction_profile_reduces_abs_error_frac",
    "cal_direction_profile_delta_mae",
    "cal_direction_aux_minus_base_bpm_mean",
    "cal_direction_stft_minus_base_bpm_mean",
    "cal_direction_weighted_physio_minus_base_bpm_mean",
    "cal_direction_profile_aux_sign_agree_frac",
    "cal_direction_profile_stft_sign_agree_frac",
    "cal_direction_profile_weighted_physio_sign_agree_frac",
    "cal_direction_aux_base_residual_sign_agree_frac",
    "cal_direction_stft_base_residual_sign_agree_frac",
    "cal_direction_weighted_physio_base_residual_sign_agree_frac",
    "eval_direction_init_feature_shift_rr_weight_available",
    "eval_direction_init_feature_shift_norm_mean",
    "eval_direction_init_feature_shift_rr_parallel_mean",
    "eval_direction_init_feature_shift_rr_parallel_abs_mean",
    "eval_direction_init_feature_shift_orthogonal_norm_mean",
    "eval_direction_init_feature_shift_parallel_fraction_mean",
    "eval_direction_post_feature_shift_rr_weight_available",
    "eval_direction_post_feature_shift_norm_mean",
    "eval_direction_post_feature_shift_rr_parallel_mean",
    "eval_direction_post_feature_shift_rr_parallel_abs_mean",
    "eval_direction_post_feature_shift_orthogonal_norm_mean",
    "eval_direction_post_feature_shift_parallel_fraction_mean",

    # Source-distribution feature canonicalization diagnostics.
    "feature_align_n_source_windows",
    "feature_align_n_moment_windows",
    "feature_align_use_calibration_only",
    "feature_align_source_rms_std_mean",
    "feature_align_target_rms_std_mean",
    "feature_align_eval_dist_to_source_before",
    "feature_align_eval_dist_to_source_after",
    "feature_align_eval_dist_delta_after_minus_before",
    "feature_align_mean_shift_l2",
    "feature_align_diag_log_std_ratio_abs_mean",
    "feature_align_rr_weight_available",
    "feature_align_rr_orthogonal_fallback_diag",
    "feature_align_rrbin_n_bins",
    "feature_align_min_bin_count",
    "feature_align_shrink",
    "feature_align_rrbin_min_count",
    "feature_align_rrbin_max_count",
    "feature_align_eval_n",
    "feature_align_eval_pred_shift_signed_mean_bpm",
    "feature_align_eval_pred_shift_abs_mean_bpm",
    "feature_align_eval_pred_shift_p95_abs_bpm",
    "feature_align_eval_base_residual_signed_mean_bpm",
    "feature_align_eval_post_residual_signed_mean_bpm",
    "feature_align_eval_direction_dot_mean",
    "feature_align_eval_direction_agree_frac",
    "feature_align_eval_overshoot_frac",
    "feature_align_eval_reduces_abs_error_frac",
    "feature_align_eval_delta_mae",
    "feature_align_cal_n",
    "feature_align_cal_pred_shift_signed_mean_bpm",
    "feature_align_cal_pred_shift_abs_mean_bpm",
    "feature_align_cal_direction_dot_mean",
    "feature_align_cal_direction_agree_frac",
    "feature_align_cal_overshoot_frac",
    "feature_align_cal_reduces_abs_error_frac",
    "feature_align_cal_delta_mae",
    "feature_align_eval_feature_delta_n",
    "feature_align_eval_feature_delta_norm_mean",
    "feature_align_eval_feature_delta_norm_p95",
    "feature_align_eval_feature_delta_rr_weight_available",
    "feature_align_eval_feature_delta_rr_parallel_mean",
    "feature_align_eval_feature_delta_rr_parallel_abs_mean",
    "feature_align_eval_feature_delta_orthogonal_norm_mean",
    "feature_align_eval_feature_delta_parallel_fraction_mean",

    # Profile-conditioned feature adaptation diagnostics.
    "feature_adapt_mode",
    "feature_adapt_n_source_windows",
    "feature_adapt_n_moment_windows",
    "feature_adapt_use_calibration_only",
    "feature_adapt_profile_rms_z",
    "feature_adapt_profile_max_abs_z",
    "feature_adapt_feature_mean_shift_rms",
    "feature_adapt_diag_log_std_ratio_abs_mean",
    "feature_adapt_kind",
    "feature_adapt_alpha",
    "feature_adapt_eval_dist_to_source_before",
    "feature_adapt_eval_dist_to_source_after",
    "feature_adapt_eval_dist_delta_after_minus_before",
    "feature_adapt_mean_shift_l2",
    "feature_adapt_eval_n",
    "feature_adapt_eval_pred_shift_signed_mean_bpm",
    "feature_adapt_eval_pred_shift_abs_mean_bpm",
    "feature_adapt_eval_pred_shift_p95_abs_bpm",
    "feature_adapt_eval_base_residual_signed_mean_bpm",
    "feature_adapt_eval_post_residual_signed_mean_bpm",
    "feature_adapt_eval_direction_dot_mean",
    "feature_adapt_eval_direction_agree_frac",
    "feature_adapt_eval_overshoot_frac",
    "feature_adapt_eval_reduces_abs_error_frac",
    "feature_adapt_eval_delta_mae",
    "feature_adapt_cal_n",
    "feature_adapt_cal_pred_shift_signed_mean_bpm",
    "feature_adapt_cal_pred_shift_abs_mean_bpm",
    "feature_adapt_cal_direction_dot_mean",
    "feature_adapt_cal_direction_agree_frac",
    "feature_adapt_cal_overshoot_frac",
    "feature_adapt_cal_reduces_abs_error_frac",
    "feature_adapt_cal_delta_mae",
    "feature_adapt_eval_feature_delta_n",
    "feature_adapt_eval_feature_delta_norm_mean",
    "feature_adapt_eval_feature_delta_norm_p95",
    "feature_adapt_eval_feature_delta_rr_weight_available",
    "feature_adapt_eval_feature_delta_rr_parallel_mean",
    "feature_adapt_eval_feature_delta_rr_parallel_abs_mean",
    "feature_adapt_eval_feature_delta_orthogonal_norm_mean",
    "feature_adapt_eval_feature_delta_parallel_fraction_mean",
]


def _mode_from_path(root: Path, path: Path) -> str:
    rel = path.relative_to(root)
    if rel.parts and rel.parts[0] in {"chunks", "subjects"}:
        return root.name
    if len(rel.parts) >= 2 and rel.parts[1] in {"chunks", "subjects"}:
        return rel.parts[0]
    return path.parent.name


def _subject_from_path(root: Path, path: Path) -> Optional[str]:
    rel = path.relative_to(root)
    parts = list(rel.parts)
    if "subjects" in parts:
        idx = parts.index("subjects")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    stem = path.stem
    marker = "_metrics_"
    if marker in stem:
        return stem.split(marker, 1)[1]
    return None


def _read_summary(root: Path, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "subject" in df.columns:
        df = df[df["subject"].notna()].copy()
    if df.empty:
        return df
    mode = _mode_from_path(root, path)
    if "rr_tta_mode" in df.columns and df["rr_tta_mode"].notna().any():
        vals = df["rr_tta_mode"].dropna().astype(str).unique().tolist()
        if len(vals) == 1:
            mode = vals[0]
    df.insert(0, "mode_dir", _mode_from_path(root, path))
    df.insert(1, "mode", mode)
    df.insert(2, "summary_path", str(path))
    return df


def _read_metrics_json(root: Path, path: Path) -> pd.DataFrame:
    with open(path, "r") as f:
        metrics = json.load(f)
    if not isinstance(metrics, dict):
        return pd.DataFrame()
    subject = str(metrics.get("subject") or _subject_from_path(root, path) or "")
    if not subject:
        return pd.DataFrame()
    mode = str(metrics.get("rr_tta_mode") or _mode_from_path(root, path))
    df = pd.DataFrame([{**metrics, "subject": subject}])
    df.insert(0, "mode_dir", _mode_from_path(root, path))
    df.insert(1, "mode", mode)
    df.insert(2, "summary_path", str(path))
    return df


def collect_rows(root: Path) -> pd.DataFrame:
    csv_patterns = [
        "**/*rr_structured_adaptation_summary.csv",
        "**/*rr_feature_adaptive_summary.csv",
        "**/*faithful_rr_tta_summary.csv",
        "**/*learned_profile_rr_tta_summary.csv",
    ]
    json_patterns = [
        "**/subjects/*/rr_feature_adaptive/rr_*_metrics_*.json",
        "**/subjects/*/rr_structured_adaptation/rr_*_metrics_*.json",
    ]
    frames: List[pd.DataFrame] = []
    seen = set()
    for pattern in csv_patterns:
        for path in sorted(root.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            try:
                df = _read_summary(root, path)
            except Exception as exc:
                print(f"[WARN] failed to read {path}: {exc}")
                continue
            if not df.empty:
                frames.append(df)
    for pattern in json_patterns:
        for path in sorted(root.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            try:
                df = _read_metrics_json(root, path)
            except Exception as exc:
                print(f"[WARN] failed to read {path}: {exc}")
                continue
            if not df.empty:
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    # Chunk reruns can duplicate a subject/mode. Keep the last row by file/path order.
    if {"mode", "subject"}.issubset(combined.columns):
        combined = combined.drop_duplicates(subset=["mode", "subject"], keep="last")
    return combined


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def summarize(combined: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    if combined.empty:
        return pd.DataFrame()
    for mode, df in combined.groupby("mode", sort=False):
        pre_mae = _num(df, "rr_probe_pre_mae")
        post_mae = _num(df, "rr_probe_post_mae")
        delta_mae = post_mae - pre_mae
        pre_rmse = _num(df, "rr_probe_pre_rmse")
        post_rmse = _num(df, "rr_probe_post_rmse")
        pre_corr = _num(df, "rr_probe_pre_corr")
        post_corr = _num(df, "rr_probe_post_corr")
        row: Dict[str, float] = {
            "mode": mode,
            "n_subjects": int(len(df)),
            "pre_mae_mean": float(pre_mae.mean()),
            "pre_mae_std": float(pre_mae.std(ddof=0)),
            "post_mae_mean": float(post_mae.mean()),
            "post_mae_std": float(post_mae.std(ddof=0)),
            "delta_mae_post_minus_pre_mean": float(delta_mae.mean()),
            "delta_mae_post_minus_pre_median": float(delta_mae.median()),
            "subjects_improved_mae": int((delta_mae < 0).sum()),
            "subjects_worse_mae": int((delta_mae > 0).sum()),
            "pre_rmse_mean": float(pre_rmse.mean()),
            "post_rmse_mean": float(post_rmse.mean()),
            "delta_rmse_post_minus_pre_mean": float((post_rmse - pre_rmse).mean()),
            "pre_corr_mean": float(pre_corr.mean()),
            "post_corr_mean": float(post_corr.mean()),
            "delta_corr_post_minus_pre_mean": float((post_corr - pre_corr).mean()),
        }
        for col in ["rr_probe_n_calibration", "rr_probe_n_eval", "rr_probe_n_source"]:
            if col in df.columns:
                row[col.replace("rr_probe_", "") + "_mean"] = float(_num(df, col).mean())
        for col in DIAG_COLS:
            if col in df.columns:
                vals = _num(df, col)
                if vals.notna().any():
                    row[f"{col}_mean"] = float(vals.mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        order_map = {m: i for i, m in enumerate(DEFAULT_ORDER)}
        out["order"] = out["mode"].map(order_map).fillna(999).astype(int)
        out = out.sort_values(["order", "post_mae_mean", "mode"]).drop(columns=["order"])
        out["rank_by_post_mae"] = out["post_mae_mean"].rank(method="min").astype(int)
        out["rank_by_delta_mae"] = out["delta_mae_post_minus_pre_mean"].rank(method="min").astype(int)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize efficient RR config sweep outputs.")
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--out-csv", required=True, type=Path)
    p.add_argument("--combined-subject-csv", default=None, type=Path)
    args = p.parse_args()

    combined = collect_rows(args.root)
    if combined.empty:
        raise SystemExit(f"No summary CSV files found under {args.root}")
    if args.combined_subject_csv is not None:
        args.combined_subject_csv.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.combined_subject_csv, index=False)
        print(f"Wrote {args.combined_subject_csv}")
    summary = summarize(combined)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")
    with pd.option_context("display.max_columns", 100, "display.width", 260):
        print(summary)


if __name__ == "__main__":
    main()
