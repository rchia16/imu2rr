#!/usr/bin/env python3
"""
Self-contained efficient configuration sweep for UNSUPERVISED RR feature-adaptive ladder adaptation.

This version intentionally avoids importing from:
  - vit_pressure_crossmodal_stft_rr_structured_adaptation_sweep
  - vit_pressure_crossmodal_stft_rr_feature_adaptive_sweep

It keeps the efficient same-checkpoint design: for each held-out subject,
run_loocv_experiment trains/loads the reconstruction model once, then this hook
evaluates multiple unsupervised RR-TTA ladder configurations against that same checkpoint state.
Feature-adaptive modes restore the model weights after each mode so every mode
starts from the identical checkpoint.

Outputs:
  <sweep_root>/<mode>/chunks/<run_id>_rr_structured_adaptation_summary.csv
  <sweep_root>/<mode>/subjects/<subject>/rr_feature_adaptive/*
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_pressure_crossmodal_profile_encoder import (
    collect_profile_stats,
    normalize_profile_stats,
    profile_stats_diagnostics,
    profile_stats_dim as infer_profile_stats_dim,
    split_profile_stats,
)
from vit_pressure_crossmodal_stft_rr_rrprobe_tta_main import (
    SUBJECTS,
    FaithfulRRRegressor,
    LinearPredictor,
    TrainConfig,
    adapt_cmt_original_style,
    adapt_ssa_original,
    build_base_parser,
    collect_rr_arrays,
    pooled_features,
    predict_features,
    predict_rr,
    rr_metrics,
    rr_targets_from_batch,
    run_loocv_experiment,
    select_source_by_target_kinematics,
    split_target_calibration_eval,
    train_source_rr_regressor,
    unpack_batch,
)


# ---------------------------------------------------------------------------
# Structured readout-level adaptation implementation (inlined)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Mode-specific defaults for the unsupervised ladder
# ---------------------------------------------------------------------------

UNSUP_MODE_DEFAULTS = {
    # LN-only: test whether tiny feature-stat shifts plus STFT agreement help.
    "ln_unsup_stft_consistency": {
        "unsup_stft_consistency_weight": 0.05,
        "unsup_smoothness_weight": 0.0,
        "unsup_source_anchor_weight": 0.0,
        "unsup_target_drift_weight": 0.01,
        "unsup_feature_lr": 3e-5,
    },
    # Capacity step: final block + spec_proj, but only STFT consistency + drift.
    "lastblock_ln_unsup_stft_consistency": {
        "unsup_stft_consistency_weight": 0.05,
        "unsup_smoothness_weight": 0.0,
        "unsup_source_anchor_weight": 0.0,
        "unsup_target_drift_weight": 0.05,
        "unsup_feature_lr": 1e-5,
    },
    # Add temporal smoothness on ordered target windows.
    "lastblock_ln_unsup_stft_smooth": {
        "unsup_stft_consistency_weight": 0.05,
        "unsup_smoothness_weight": 0.01,
        "unsup_source_anchor_weight": 0.0,
        "unsup_target_drift_weight": 0.05,
        "unsup_feature_lr": 1e-5,
    },
    # Full unsupervised proposed mode:
    # STFT consistency + smoothness + source anchor + target drift.
    "lastblock_ln_unsup_stft_smooth_anchor": {
        "unsup_stft_consistency_weight": 0.05,
        "unsup_smoothness_weight": 0.01,
        "unsup_source_anchor_weight": 0.05,
        "unsup_target_drift_weight": 0.05,
        "unsup_feature_lr": 1e-5,
    },
    # Profile-FiLM modes: adapt only a target subject profile vector while the
    # checkpointed model and profile modules stay frozen.
    "profile_film_unsup_stft_consistency": {
        "use_profile_film": 1,
        "use_profile_qkv": 0,
        "profile_conditioning": "film",
        "adapt_profile_vector": 1,
        "adapt_profile_encoder": 0,
        "adapt_profile_film": 0,
        "unsup_stft_consistency_weight": 0.05,
        "unsup_smoothness_weight": 0.0,
        "unsup_source_anchor_weight": 0.0,
        "unsup_target_drift_weight": 0.01,
        "unsup_profile_prior_weight": 0.01,
        "unsup_feature_lr": 3e-4,
    },
    "profile_film_unsup_stft_smooth_anchor": {
        "use_profile_film": 1,
        "use_profile_qkv": 0,
        "profile_conditioning": "film",
        "adapt_profile_vector": 1,
        "adapt_profile_encoder": 0,
        "adapt_profile_film": 0,
        "unsup_stft_consistency_weight": 0.05,
        "unsup_smoothness_weight": 0.01,
        "unsup_source_anchor_weight": 0.05,
        "unsup_target_drift_weight": 0.05,
        "unsup_profile_prior_weight": 0.01,
        "unsup_feature_lr": 3e-4,
    },
    # SPARC-style profile/readout ladder:
    #   - model weights stay frozen
    #   - adapt only the target subject profile vector p_t
    #   - optionally learn a scalar final readout affine y_final = a*y_probe+b
    #   - use unlabeled STFT-derived RR as the physiological anchor
    #   - use range loss to discourage subject-level RR collapse
    "profile_film_unsup_readout_affine": {
        "use_profile_film": 1,
        "use_profile_qkv": 0,
        "profile_conditioning": "film",
        "adapt_profile_vector": 1,
        "adapt_profile_encoder": 0,
        "adapt_profile_film": 0,
        "unsup_readout_affine": 1,
        "unsup_stft_consistency_weight": 0.05,
        "unsup_aux_consistency_weight": 0.01,
        "unsup_smoothness_weight": 0.01,
        "unsup_source_anchor_weight": 0.05,
        "unsup_target_drift_weight": 0.05,
        "unsup_profile_prior_weight": 0.01,
        "unsup_rr_range_weight": 0.005,
        "unsup_feature_lr": 3e-4,
    },
    # SPARC hypothesis:
    # Subject shift is primarily a small profile/readout offset-gain problem.
    # We therefore freeze encoder, decoder, RR probe, profile encoder, and FiLM
    # weights, and adapt only p_t plus optional scalar readout affine a,b using
    # unlabeled target windows. Target RR labels are never used in the loss.
    "profile_film_unsup_sparc": {
        "use_profile_film": 1,
        "use_profile_qkv": 0,
        "profile_conditioning": "film",
        "adapt_profile_vector": 1,
        "adapt_profile_encoder": 0,
        "adapt_profile_film": 0,
        "unsup_readout_affine": 1,
        "unsup_stft_consistency_weight": 0.05,
        "unsup_aux_consistency_weight": 0.02,
        "unsup_smoothness_weight": 0.01,
        "unsup_source_anchor_weight": 0.05,
        "unsup_target_drift_weight": 0.05,
        "unsup_profile_prior_weight": 0.01,
        "unsup_rr_range_weight": 0.01,
        "unsup_feature_lr": 3e-4,
    },
    # Diagnostic Profile-FiLM ablations and safety-gated variants.
    # profile_film_init_only evaluates the initial profile vector p0 without
    # unsupervised updates. The gated/oracle modes decide whether to use the
    # Profile-FiLM candidate or fall back to the no-profile baseline.
    "profile_film_init_only": {
        "use_profile_film": 1,
        "use_profile_qkv": 0,
        "profile_conditioning": "film",
        "adapt_profile_vector": 0,
        "adapt_profile_encoder": 0,
        "adapt_profile_film": 0,
        "unsup_readout_affine": 0,
        "unsup_feature_lr": 3e-4,
    },
    "profile_film_gated_init_only": {
        "use_profile_film": 1,
        "use_profile_qkv": 0,
        "profile_conditioning": "film",
        "adapt_profile_vector": 0,
        "adapt_profile_encoder": 0,
        "adapt_profile_film": 0,
        "unsup_readout_affine": 0,
        "unsup_feature_lr": 3e-4,
    },
    "profile_film_oracle_init_only": {
        "use_profile_film": 1,
        "use_profile_qkv": 0,
        "profile_conditioning": "film",
        "adapt_profile_vector": 0,
        "adapt_profile_encoder": 0,
        "adapt_profile_film": 0,
        "unsup_readout_affine": 0,
        "unsup_feature_lr": 3e-4,
    },
    "profile_film_gated_sparc": {
        "use_profile_film": 1,
        "use_profile_qkv": 0,
        "profile_conditioning": "film",
        "adapt_profile_vector": 1,
        "adapt_profile_encoder": 0,
        "adapt_profile_film": 0,
        "unsup_readout_affine": 1,
        "unsup_stft_consistency_weight": 0.05,
        "unsup_aux_consistency_weight": 0.02,
        "unsup_smoothness_weight": 0.01,
        "unsup_source_anchor_weight": 0.05,
        "unsup_target_drift_weight": 0.05,
        "unsup_profile_prior_weight": 0.01,
        "unsup_rr_range_weight": 0.01,
        "unsup_feature_lr": 3e-4,
    },
    "profile_film_oracle_sparc": {
        "use_profile_film": 1,
        "use_profile_qkv": 0,
        "profile_conditioning": "film",
        "adapt_profile_vector": 1,
        "adapt_profile_encoder": 0,
        "adapt_profile_film": 0,
        "unsup_readout_affine": 1,
        "unsup_stft_consistency_weight": 0.05,
        "unsup_aux_consistency_weight": 0.02,
        "unsup_smoothness_weight": 0.01,
        "unsup_source_anchor_weight": 0.05,
        "unsup_target_drift_weight": 0.05,
        "unsup_profile_prior_weight": 0.01,
        "unsup_rr_range_weight": 0.01,
        "unsup_feature_lr": 3e-4,
    },
    "profile_qkv_unsup_stft_consistency": {
        "use_profile_film": 0,
        "use_profile_qkv": 1,
        "profile_conditioning": "qkv",
        "profile_qkv_layers": "last1",
        "adapt_profile_vector": 1,
        "adapt_profile_encoder": 0,
        "adapt_profile_film": 0,
        "unsup_stft_consistency_weight": 0.05,
        "unsup_smoothness_weight": 0.0,
        "unsup_source_anchor_weight": 0.0,
        "unsup_target_drift_weight": 0.01,
        "unsup_profile_prior_weight": 0.01,
        "unsup_attention_profile_weight": 0.0,
        "unsup_feature_lr": 3e-4,
    },
    "profile_qkv_unsup_stft_smooth_anchor": {
        "use_profile_film": 0,
        "use_profile_qkv": 1,
        "profile_conditioning": "qkv",
        "profile_qkv_layers": "last1",
        "adapt_profile_vector": 1,
        "adapt_profile_encoder": 0,
        "adapt_profile_film": 0,
        "unsup_stft_consistency_weight": 0.05,
        "unsup_smoothness_weight": 0.01,
        "unsup_source_anchor_weight": 0.05,
        "unsup_target_drift_weight": 0.05,
        "unsup_profile_prior_weight": 0.01,
        "unsup_attention_profile_weight": 0.0,
        "unsup_feature_lr": 3e-4,
    },
    "profile_qkv_unsup_stft_smooth_prior_attn": {
        "use_profile_film": 0,
        "use_profile_qkv": 1,
        "profile_conditioning": "qkv",
        "profile_qkv_layers": "last1",
        "adapt_profile_vector": 1,
        "adapt_profile_encoder": 0,
        "adapt_profile_film": 0,
        "unsup_stft_consistency_weight": 0.05,
        "unsup_smoothness_weight": 0.01,
        "unsup_source_anchor_weight": 0.05,
        "unsup_target_drift_weight": 0.05,
        "unsup_profile_prior_weight": 0.01,
        "unsup_attention_profile_weight": 0.005,
        "unsup_feature_lr": 3e-4,
    },
    "tcn_profile_film_qkv_last1_0p01": {
        "use_tcn_token_mixer": 1,
        "tcn_mixer_alpha": 0.05,
        "use_profile_film": 1,
        "use_profile_qkv": 1,
        "shared_profile_qkv": 1,
        "profile_conditioning": "film_qkv",
        "profile_qkv_layers": "last1",
        "profile_qkv_scale": 0.01,
        "profile_qkv_residual": 1,
        "adapt_profile_vector": 0,
        "adapt_profile_encoder": 0,
        "adapt_profile_film": 0,
        "unsup_readout_affine": 0,
    },
    "tcn_profile_film_qkv_last1_0p01_sparc_pt": {
        "use_tcn_token_mixer": 1,
        "tcn_mixer_alpha": 0.05,
        "use_profile_film": 1,
        "use_profile_qkv": 1,
        "shared_profile_qkv": 1,
        "profile_conditioning": "film_qkv",
        "profile_qkv_layers": "last1",
        "profile_qkv_scale": 0.01,
        "profile_qkv_residual": 1,
        "adapt_profile_vector": 1,
        "adapt_profile_encoder": 0,
        "adapt_profile_film": 0,
        "unsup_readout_affine": 0,
        "unsup_stft_consistency_weight": 0.05,
        "unsup_aux_consistency_weight": 0.02,
        "unsup_smoothness_weight": 0.01,
        "unsup_source_anchor_weight": 0.05,
        "unsup_target_drift_weight": 0.05,
        "unsup_profile_prior_weight": 0.01,
        "unsup_rr_range_weight": 0.01,
        "qkv_delta_budget_weight": 0.01,
        "unsup_feature_lr": 3e-4,
    },
    "tcn_profile_film_qkv_last1_0p01_sparc_pt_budget": {
        "use_tcn_token_mixer": 1,
        "tcn_mixer_alpha": 0.05,
        "use_profile_film": 1,
        "use_profile_qkv": 1,
        "shared_profile_qkv": 1,
        "profile_conditioning": "film_qkv",
        "profile_qkv_layers": "last1",
        "profile_qkv_scale": 0.01,
        "profile_qkv_residual": 1,
        "adapt_profile_vector": 1,
        "adapt_profile_encoder": 0,
        "adapt_profile_film": 0,
        "unsup_readout_affine": 0,
        "unsup_stft_consistency_weight": 0.05,
        "unsup_aux_consistency_weight": 0.02,
        "unsup_smoothness_weight": 0.01,
        "unsup_source_anchor_weight": 0.05,
        "unsup_target_drift_weight": 0.05,
        "unsup_profile_prior_weight": 0.01,
        "unsup_rr_range_weight": 0.01,
        "qkv_delta_budget_weight": 0.01,
        "profile_safety_budget": 1,
        "unsup_feature_lr": 3e-4,
    },
    "tcn_profile_film_qkv_last1_0p01_pt_no_stft": {
        "use_tcn_token_mixer": 1,
        "tcn_mixer_alpha": 0.05,
        "use_profile_film": 1,
        "use_profile_qkv": 1,
        "shared_profile_qkv": 1,
        "profile_conditioning": "film_qkv",
        "profile_qkv_layers": "last1",
        "profile_qkv_scale": 0.01,
        "profile_qkv_residual": 1,
        "adapt_profile_vector": 1,
        "adapt_profile_encoder": 0,
        "adapt_profile_film": 0,
        "unsup_readout_affine": 0,
        "unsup_stft_consistency_weight": 0.0,
        "unsup_aux_consistency_weight": 0.02,
        "unsup_smoothness_weight": 0.01,
        "unsup_source_anchor_weight": 0.05,
        "unsup_target_drift_weight": 0.05,
        "unsup_profile_prior_weight": 0.01,
        "unsup_rr_range_weight": 0.01,
        "qkv_delta_budget_weight": 0.01,
        "unsup_feature_lr": 3e-4,
    },
    "tcn_profile_film_qkv_last1_0p01_pt_aux_only": {
        "use_tcn_token_mixer": 1,
        "tcn_mixer_alpha": 0.05,
        "use_profile_film": 1,
        "use_profile_qkv": 1,
        "shared_profile_qkv": 1,
        "profile_conditioning": "film_qkv",
        "profile_qkv_layers": "last1",
        "profile_qkv_scale": 0.01,
        "profile_qkv_residual": 1,
        "adapt_profile_vector": 1,
        "adapt_profile_encoder": 0,
        "adapt_profile_film": 0,
        "unsup_readout_affine": 0,
        "unsup_stft_consistency_weight": 0.0,
        "unsup_aux_consistency_weight": 0.05,
        "unsup_smoothness_weight": 0.0,
        "unsup_source_anchor_weight": 0.0,
        "unsup_target_drift_weight": 0.01,
        "unsup_profile_prior_weight": 0.01,
        "unsup_rr_range_weight": 0.005,
        "qkv_delta_budget_weight": 0.01,
        "unsup_feature_lr": 3e-4,
    },
    "tcn_profile_film_qkv_last1_0p01_pt_reg_only": {
        "use_tcn_token_mixer": 1,
        "tcn_mixer_alpha": 0.05,
        "use_profile_film": 1,
        "use_profile_qkv": 1,
        "shared_profile_qkv": 1,
        "profile_conditioning": "film_qkv",
        "profile_qkv_layers": "last1",
        "profile_qkv_scale": 0.01,
        "profile_qkv_residual": 1,
        "adapt_profile_vector": 1,
        "adapt_profile_encoder": 0,
        "adapt_profile_film": 0,
        "unsup_readout_affine": 0,
        "unsup_stft_consistency_weight": 0.0,
        "unsup_aux_consistency_weight": 0.0,
        "unsup_smoothness_weight": 0.01,
        "unsup_source_anchor_weight": 0.05,
        "unsup_target_drift_weight": 0.05,
        "unsup_profile_prior_weight": 0.01,
        "unsup_rr_range_weight": 0.0,
        "qkv_delta_budget_weight": 0.01,
        "unsup_feature_lr": 3e-4,
    },
    "tcn_profile_film_qkv_last1_0p01_pt_no_stft_budget": {
        "use_tcn_token_mixer": 1,
        "tcn_mixer_alpha": 0.05,
        "use_profile_film": 1,
        "use_profile_qkv": 1,
        "shared_profile_qkv": 1,
        "profile_conditioning": "film_qkv",
        "profile_qkv_layers": "last1",
        "profile_qkv_scale": 0.01,
        "profile_qkv_residual": 1,
        "adapt_profile_vector": 1,
        "adapt_profile_encoder": 0,
        "adapt_profile_film": 0,
        "unsup_readout_affine": 0,
        "unsup_stft_consistency_weight": 0.0,
        "unsup_aux_consistency_weight": 0.02,
        "unsup_smoothness_weight": 0.01,
        "unsup_source_anchor_weight": 0.05,
        "unsup_target_drift_weight": 0.05,
        "unsup_profile_prior_weight": 0.01,
        "unsup_rr_range_weight": 0.01,
        "qkv_delta_budget_weight": 0.01,
        "profile_safety_budget": 1,
        "profile_safety_budget_use_stft_confidence": 0,
        "unsup_feature_lr": 3e-4,
    },
    "tcn_profile_film_clsa_qkv_last1": {
        "use_tcn_token_mixer": 1,
        "tcn_mixer_alpha": 0.05,
        "use_profile_film": 1,
        "use_profile_qkv": 1,
        "shared_profile_qkv": 1,
        "profile_conditioning": "film_qkv",
        "profile_qkv_mode": "clsa",
        "profile_qkv_layers": "last1",
        "profile_qkv_scale": 0.01,
        "profile_qkv_residual": 1,
        "profile_clsa_rank": 8,
        "profile_clsa_scale": 0.01,
        "profile_clsa_eta_max": 0.1,
        "profile_clsa_enable_fast_update": 1,
        "profile_clsa_gate_init_bias": -2.0,
        "profile_clsa_loss_weight": 0.0,
        "adapt_profile_vector": 0,
        "adapt_profile_encoder": 0,
        "adapt_profile_film": 0,
        "unsup_readout_affine": 0,
        "unsup_stft_consistency_weight": 0.0,
        "uses_stft_pseudotarget_for_tta": 0,
    },
    "tcn_profile_film_clsa_qkv_last1_no_fast_update": {
        "use_tcn_token_mixer": 1,
        "tcn_mixer_alpha": 0.05,
        "use_profile_film": 1,
        "use_profile_qkv": 1,
        "shared_profile_qkv": 1,
        "profile_conditioning": "film_qkv",
        "profile_qkv_mode": "clsa",
        "profile_qkv_layers": "last1",
        "profile_qkv_scale": 0.01,
        "profile_qkv_residual": 1,
        "profile_clsa_rank": 8,
        "profile_clsa_scale": 0.01,
        "profile_clsa_eta_max": 0.1,
        "profile_clsa_enable_fast_update": 0,
        "profile_clsa_gate_init_bias": -2.0,
        "profile_clsa_loss_weight": 0.0,
        "adapt_profile_vector": 0,
        "adapt_profile_encoder": 0,
        "adapt_profile_film": 0,
        "unsup_readout_affine": 0,
        "unsup_stft_consistency_weight": 0.0,
        "uses_stft_pseudotarget_for_tta": 0,
    },
    "tcn_clsa_qkv_last1_no_film": {
        "use_tcn_token_mixer": 1,
        "tcn_mixer_alpha": 0.05,
        "use_profile_film": 0,
        "use_profile_qkv": 1,
        "shared_profile_qkv": 0,
        "profile_conditioning": "qkv",
        "profile_qkv_mode": "clsa",
        "profile_qkv_layers": "last1",
        "profile_qkv_scale": 0.01,
        "profile_qkv_residual": 1,
        "profile_clsa_rank": 8,
        "profile_clsa_scale": 0.01,
        "profile_clsa_eta_max": 0.1,
        "profile_clsa_enable_fast_update": 1,
        "profile_clsa_gate_init_bias": -2.0,
        "profile_clsa_loss_weight": 0.0,
        "adapt_profile_vector": 0,
        "adapt_profile_encoder": 0,
        "adapt_profile_film": 0,
        "unsup_readout_affine": 0,
        "unsup_stft_consistency_weight": 0.0,
        "uses_stft_pseudotarget_for_tta": 0,
    },
    "tcn_profile_film_clsa_qkv_last1_rank4": {
        "use_tcn_token_mixer": 1,
        "tcn_mixer_alpha": 0.05,
        "use_profile_film": 1,
        "use_profile_qkv": 1,
        "shared_profile_qkv": 1,
        "profile_conditioning": "film_qkv",
        "profile_qkv_mode": "clsa",
        "profile_qkv_layers": "last1",
        "profile_qkv_scale": 0.01,
        "profile_qkv_residual": 1,
        "profile_clsa_rank": 4,
        "profile_clsa_scale": 0.01,
        "profile_clsa_eta_max": 0.1,
        "profile_clsa_enable_fast_update": 1,
        "profile_clsa_gate_init_bias": -2.0,
        "profile_clsa_loss_weight": 0.0,
        "adapt_profile_vector": 0,
        "adapt_profile_encoder": 0,
        "adapt_profile_film": 0,
        "unsup_readout_affine": 0,
        "unsup_stft_consistency_weight": 0.0,
        "uses_stft_pseudotarget_for_tta": 0,
    },
}


def _args_for_rr_tta_mode(args, mode: str):
    """
    Return a shallow-copied argparse namespace with mode-specific settings.

    NOTE:
      There are two user-facing switches in this codebase:
        --apply-mode-defaults      used by the profile-encoder CLI
        --use-unsup-mode-defaults  used by this ladder module

      Treat either one as permission to apply UNSUP_MODE_DEFAULTS, otherwise
      profile-FiLM/QKV ladder modes can accidentally run with generic weights.
    """
    local_args = copy.copy(args)
    local_args.rr_tta = str(mode)

    apply_defaults = (
        bool(getattr(args, "use_unsup_mode_defaults", False))
        or bool(getattr(args, "apply_mode_defaults", False))
    )

    if apply_defaults:
        defaults = UNSUP_MODE_DEFAULTS.get(str(mode), {})
        for key, value in defaults.items():
            setattr(local_args, key, value)
        setattr(local_args, "unsup_mode_defaults_applied", int(bool(defaults)))
    else:
        setattr(local_args, "unsup_mode_defaults_applied", 0)

    return local_args

# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.size < 2 or np.std(y_true) <= 1e-8 or np.std(y_pred) <= 1e-8:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])




def _summary_stats(prefix: str, x: np.ndarray) -> Dict[str, float]:
    """Compact signed and absolute distribution summary for directional diagnostics."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_p05": float("nan"),
            f"{prefix}_p25": float("nan"),
            f"{prefix}_p75": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_abs_mean": float("nan"),
            f"{prefix}_abs_p95": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_p05": float(np.percentile(x, 5)),
        f"{prefix}_p25": float(np.percentile(x, 25)),
        f"{prefix}_p75": float(np.percentile(x, 75)),
        f"{prefix}_p95": float(np.percentile(x, 95)),
        f"{prefix}_abs_mean": float(np.mean(np.abs(x))),
        f"{prefix}_abs_p95": float(np.percentile(np.abs(x), 95)),
    }


def _sign_agreement(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    """Fraction of finite samples where two signed directions agree."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    n = min(a.size, b.size)
    if n <= 0:
        return float("nan")
    a = a[:n]
    b = b[:n]
    keep = np.isfinite(a) & np.isfinite(b) & (np.abs(a) > eps) & (np.abs(b) > eps)
    if not np.any(keep):
        return float("nan")
    return float(np.mean(np.sign(a[keep]) == np.sign(b[keep])))


def _directional_shift_diagnostics(
    *,
    y_true: np.ndarray,
    rr_base: np.ndarray,
    rr_profile: np.ndarray,
    rr_post: Optional[np.ndarray] = None,
    rr_aux: Optional[np.ndarray] = None,
    rr_stft: Optional[np.ndarray] = None,
    stft_conf: Optional[np.ndarray] = None,
    prefix: str,
) -> Dict[str, float]:
    """Diagnose whether profile shifts move predictions in the label-residual direction.

    Label-derived fields are analysis diagnostics only. They are never used by
    adaptation or no-label gates in this module.
    """
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    rr_base = np.asarray(rr_base, dtype=np.float32).reshape(-1)
    rr_profile = np.asarray(rr_profile, dtype=np.float32).reshape(-1)
    n = min(y_true.size, rr_base.size, rr_profile.size)
    if n <= 0:
        return {f"{prefix}_n": 0.0}
    y_true = y_true[:n]
    rr_base = rr_base[:n]
    rr_profile = rr_profile[:n]

    profile_shift = rr_profile - rr_base
    base_residual = y_true - rr_base
    profile_residual = y_true - rr_profile
    direction_dot = profile_shift * base_residual
    overshoot = np.abs(profile_shift) > np.abs(base_residual)

    out: Dict[str, float] = {f"{prefix}_n": float(n)}
    out.update(_summary_stats(f"{prefix}_profile_shift_signed_bpm", profile_shift))
    out.update(_summary_stats(f"{prefix}_base_residual_signed_bpm", base_residual))
    out.update(_summary_stats(f"{prefix}_profile_residual_signed_bpm", profile_residual))
    out.update(_summary_stats(f"{prefix}_direction_dot", direction_dot))
    out[f"{prefix}_direction_dot_mean"] = float(np.mean(direction_dot))
    out[f"{prefix}_direction_agree_frac"] = _sign_agreement(profile_shift, base_residual)
    out[f"{prefix}_overshoot_frac"] = float(np.mean(overshoot))
    out[f"{prefix}_profile_reduces_abs_error_frac"] = float(
        np.mean(np.abs(profile_residual) < np.abs(base_residual))
    )
    out[f"{prefix}_profile_delta_mae"] = float(
        np.mean(np.abs(profile_residual)) - np.mean(np.abs(base_residual))
    )

    if rr_post is not None:
        rr_post = np.asarray(rr_post, dtype=np.float32).reshape(-1)[:n]
        post_shift = rr_post - rr_base
        post_residual = y_true - rr_post
        post_direction_dot = post_shift * base_residual
        out.update(_summary_stats(f"{prefix}_post_shift_signed_bpm", post_shift))
        out.update(_summary_stats(f"{prefix}_post_residual_signed_bpm", post_residual))
        out.update(_summary_stats(f"{prefix}_post_direction_dot", post_direction_dot))
        out[f"{prefix}_post_direction_agree_frac"] = _sign_agreement(post_shift, base_residual)
        out[f"{prefix}_post_overshoot_frac"] = float(np.mean(np.abs(post_shift) > np.abs(base_residual)))
        out[f"{prefix}_post_reduces_abs_error_frac"] = float(
            np.mean(np.abs(post_residual) < np.abs(base_residual))
        )
        out[f"{prefix}_post_delta_mae"] = float(
            np.mean(np.abs(post_residual)) - np.mean(np.abs(base_residual))
        )

    if rr_aux is not None:
        rr_aux = np.asarray(rr_aux, dtype=np.float32).reshape(-1)[:n]
        aux_dir = rr_aux - rr_base
        out.update(_summary_stats(f"{prefix}_aux_minus_base_bpm", aux_dir))
        out[f"{prefix}_profile_aux_sign_agree_frac"] = _sign_agreement(profile_shift, aux_dir)
        out[f"{prefix}_aux_base_residual_sign_agree_frac"] = _sign_agreement(aux_dir, base_residual)
    else:
        aux_dir = None

    if rr_stft is not None:
        rr_stft = np.asarray(rr_stft, dtype=np.float32).reshape(-1)[:n]
        stft_dir = rr_stft - rr_base
        out.update(_summary_stats(f"{prefix}_stft_minus_base_bpm", stft_dir))
        out[f"{prefix}_profile_stft_sign_agree_frac"] = _sign_agreement(profile_shift, stft_dir)
        out[f"{prefix}_stft_base_residual_sign_agree_frac"] = _sign_agreement(stft_dir, base_residual)
    else:
        stft_dir = None

    if aux_dir is not None and stft_dir is not None:
        weighted_dir = 0.7 * aux_dir + 0.3 * stft_dir
        out.update(_summary_stats(f"{prefix}_weighted_physio_minus_base_bpm", weighted_dir))
        out[f"{prefix}_profile_weighted_physio_sign_agree_frac"] = _sign_agreement(
            profile_shift, weighted_dir
        )
        out[f"{prefix}_weighted_physio_base_residual_sign_agree_frac"] = _sign_agreement(
            weighted_dir, base_residual
        )

    if stft_conf is not None:
        stft_conf = np.asarray(stft_conf, dtype=np.float32).reshape(-1)[:n]
        out.update(_summary_stats(f"{prefix}_stft_confidence", stft_conf))
        out[f"{prefix}_stft_confidence_frac_gt_005"] = float(np.mean(stft_conf >= 0.005))
        out[f"{prefix}_stft_confidence_frac_gt_010"] = float(np.mean(stft_conf >= 0.010))

    return out


def _get_rr_probe_weight(rr_model: FaithfulRRRegressor) -> Optional[np.ndarray]:
    """Return final linear RR readout direction when the probe exposes one."""
    reg = getattr(rr_model, "regressor", None)
    if reg is None:
        return None
    linear_layers = [m for m in reg.modules() if isinstance(m, nn.Linear)]
    if not linear_layers:
        return None
    w = linear_layers[-1].weight.detach().cpu().numpy()
    if w.ndim == 2 and w.shape[0] == 1:
        return w.reshape(-1).astype(np.float32)
    return None


def _feature_shift_alignment_diagnostics(
    *,
    z_base: np.ndarray,
    z_profile: np.ndarray,
    rr_weight: Optional[np.ndarray],
    prefix: str,
) -> Dict[str, float]:
    """Project representation shift onto the learned RR-probe readout direction."""
    if rr_weight is None:
        return {f"{prefix}_feature_shift_rr_weight_available": 0.0}
    z_base = np.asarray(z_base, dtype=np.float32)
    z_profile = np.asarray(z_profile, dtype=np.float32)
    rr_weight = np.asarray(rr_weight, dtype=np.float32).reshape(-1)
    if z_base.ndim != 2 or z_profile.ndim != 2 or rr_weight.size == 0:
        return {f"{prefix}_feature_shift_rr_weight_available": 0.0}
    n = min(z_base.shape[0], z_profile.shape[0])
    d = min(z_base.shape[1], z_profile.shape[1], rr_weight.size)
    if n <= 0 or d <= 0:
        return {f"{prefix}_feature_shift_rr_weight_available": 0.0}
    dz = z_profile[:n, :d] - z_base[:n, :d]
    w = rr_weight[:d]
    w = w / (np.linalg.norm(w) + 1e-8)
    parallel = dz @ w
    dz_norm = np.linalg.norm(dz, axis=1)
    orth_norm = np.sqrt(np.maximum(dz_norm ** 2 - parallel ** 2, 0.0))
    parallel_fraction = np.abs(parallel) / (dz_norm + 1e-8)
    out = {f"{prefix}_feature_shift_rr_weight_available": 1.0}
    out.update(_summary_stats(f"{prefix}_feature_shift_norm", dz_norm))
    out.update(_summary_stats(f"{prefix}_feature_shift_rr_parallel", parallel))
    out.update(_summary_stats(f"{prefix}_feature_shift_orthogonal_norm", orth_norm))
    out.update(_summary_stats(f"{prefix}_feature_shift_parallel_fraction", parallel_fraction))
    return out


def _linear_predict_from_features(
    rr_model: FaithfulRRRegressor,
    z_feature: np.ndarray,
    device: str,
    batch_size: int = 2048,
) -> np.ndarray:
    """Predict using rr_model.regressor when z_feature is already adapter-space z."""
    rr_model.eval()
    out: List[np.ndarray] = []
    with torch.no_grad():
        for st in range(0, z_feature.shape[0], int(batch_size)):
            z = torch.tensor(z_feature[st : st + int(batch_size)], dtype=torch.float32, device=device)
            y = rr_model.regressor(z).squeeze(-1)
            out.append(y.detach().cpu().numpy().reshape(-1))
    return np.concatenate(out, axis=0)


def _ridge_solve(
    x: np.ndarray,
    y: np.ndarray,
    l2: np.ndarray,
    prior: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Solve min ||Xb-y||^2 + (b-prior)^T diag(l2) (b-prior)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    l2 = np.asarray(l2, dtype=np.float64).reshape(-1)
    if prior is None:
        prior = np.zeros(x.shape[1], dtype=np.float64)
    else:
        prior = np.asarray(prior, dtype=np.float64).reshape(-1)
    a = x.T @ x + np.diag(l2)
    b = x.T @ y + np.diag(l2) @ prior
    try:
        coef = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(a) @ b
    return coef.astype(np.float32)


# ---------------------------------------------------------------------------
# Output-level affine calibration
# ---------------------------------------------------------------------------


@dataclass
class AffineCalibration:
    a: float
    b: float

    def apply(self, pred: np.ndarray) -> np.ndarray:
        return float(self.a) * np.asarray(pred).reshape(-1) + float(self.b)


def fit_affine_calibration(
    pred_cal: np.ndarray,
    y_cal: np.ndarray,
    *,
    lambda_a: float,
    lambda_b: float,
    monotone: bool = False,
    min_slope: float = 0.05,
) -> Tuple[AffineCalibration, Dict[str, float]]:
    pred_cal = np.asarray(pred_cal, dtype=np.float32).reshape(-1)
    y_cal = np.asarray(y_cal, dtype=np.float32).reshape(-1)
    if pred_cal.size == 0:
        return AffineCalibration(1.0, 0.0), {
            "affine_a": 1.0,
            "affine_b": 0.0,
            "affine_n_cal": 0,
            "affine_monotone_clipped": 0,
        }

    x = np.column_stack([pred_cal, np.ones_like(pred_cal)]).astype(np.float32)
    coef = _ridge_solve(
        x,
        y_cal,
        np.array([float(lambda_a), float(lambda_b)], dtype=np.float32),
        prior=np.array([1.0, 0.0], dtype=np.float32),
    )
    a, b = float(coef[0]), float(coef[1])
    clipped = 0
    if monotone and a < float(min_slope):
        clipped = 1
        a = float(min_slope)
        # Re-estimate intercept with the slope fixed, with ridge toward 0.
        resid = y_cal - a * pred_cal
        denom = float(pred_cal.size) + float(lambda_b)
        b = float(resid.sum() / max(1e-12, denom))
    cal_pred = a * pred_cal + b
    return AffineCalibration(a, b), {
        "affine_a": float(a),
        "affine_b": float(b),
        "affine_n_cal": int(pred_cal.size),
        "affine_lambda_a": float(lambda_a),
        "affine_lambda_b": float(lambda_b),
        "affine_monotone": int(bool(monotone)),
        "affine_monotone_clipped": int(clipped),
        "affine_cal_mae": float(np.mean(np.abs(cal_pred - y_cal))),
        "affine_cal_rmse": float(np.sqrt(np.mean((cal_pred - y_cal) ** 2))),
        "affine_cal_corr": _safe_corr(y_cal, cal_pred),
    }


# ---------------------------------------------------------------------------
# Ridge residual calibration on low-dimensional profile-quality descriptors
# ---------------------------------------------------------------------------


@dataclass
class SourceProfileStats:
    mean: np.ndarray
    std: np.ndarray
    rr_bin_edges: np.ndarray
    rr_bin_centroids: np.ndarray
    global_centroid: np.ndarray


def _make_quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    n_bins = max(1, int(n_bins))
    if values.size == 0:
        return np.array([-np.inf, np.inf], dtype=np.float32)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, qs).astype(np.float32)
    edges[0] = -np.inf
    edges[-1] = np.inf
    # Quantiles can collapse for narrow RR distributions. Make the binning robust.
    for i in range(1, len(edges) - 1):
        if not np.isfinite(edges[i]) or edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.float32(np.inf))
    return edges.astype(np.float32)


def _assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    edges = np.asarray(edges, dtype=np.float32).reshape(-1)
    if edges.size <= 2:
        return np.zeros(values.shape[0], dtype=np.int64)
    return np.clip(np.digitize(values, edges[1:-1], right=False), 0, edges.size - 2).astype(np.int64)


def _fit_source_profile_stats(
    z_source: np.ndarray,
    pred_source: np.ndarray,
    y_source: np.ndarray,
    n_bins: int,
) -> SourceProfileStats:
    z_source = np.asarray(z_source, dtype=np.float32)
    mean = z_source.mean(axis=0)
    std = z_source.std(axis=0) + 1e-6
    z_std = (z_source - mean.reshape(1, -1)) / std.reshape(1, -1)
    global_centroid = z_std.mean(axis=0)
    # Use true source RR to define physiology bins; use target predicted RR for target assignment.
    edges = _make_quantile_edges(y_source, int(n_bins))
    source_bins = _assign_bins(y_source, edges)
    centroids = []
    for b in range(edges.size - 1):
        mask = source_bins == b
        if mask.sum() == 0:
            centroids.append(global_centroid.copy())
        else:
            centroids.append(z_std[mask].mean(axis=0))
    return SourceProfileStats(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        rr_bin_edges=edges.astype(np.float32),
        rr_bin_centroids=np.stack(centroids, axis=0).astype(np.float32),
        global_centroid=global_centroid.astype(np.float32),
    )


def _quality_features(
    z: np.ndarray,
    pred: np.ndarray,
    stats: SourceProfileStats,
) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    zs = (z - stats.mean.reshape(1, -1)) / stats.std.reshape(1, -1)
    n_dim = max(1, z.shape[1])
    z_norm = np.sqrt(np.mean(zs * zs, axis=1))
    d_global = np.sqrt(np.mean((zs - stats.global_centroid.reshape(1, -1)) ** 2, axis=1))
    d_bins = []
    for c in stats.rr_bin_centroids:
        d_bins.append(np.sqrt(np.mean((zs - c.reshape(1, -1)) ** 2, axis=1)))
    d_bins_arr = np.stack(d_bins, axis=1) if d_bins else np.zeros((z.shape[0], 1), dtype=np.float32)
    nearest_dist = d_bins_arr.min(axis=1)
    nearest_bin = d_bins_arr.argmin(axis=1).astype(np.float32)
    assigned_bin = _assign_bins(pred, stats.rr_bin_edges).astype(np.float32)
    denom = max(1.0, float(max(1, stats.rr_bin_edges.size - 2)))
    q = np.column_stack([
        pred,
        pred * pred,
        z_norm,
        d_global,
        nearest_dist,
        nearest_bin / denom,
        assigned_bin / denom,
    ]).astype(np.float32)
    return q


@dataclass
class ResidualCalibrator:
    q_mean: np.ndarray
    q_std: np.ndarray
    coef: np.ndarray

    def apply(self, pred: np.ndarray, q: np.ndarray) -> np.ndarray:
        qn = (q - self.q_mean.reshape(1, -1)) / self.q_std.reshape(1, -1)
        design = np.column_stack([np.ones(qn.shape[0], dtype=np.float32), qn]).astype(np.float32)
        residual = design @ self.coef.reshape(-1, 1)
        return np.asarray(pred, dtype=np.float32).reshape(-1) + residual.reshape(-1)


def fit_ridge_residual_calibrator(
    q_source: np.ndarray,
    q_cal: np.ndarray,
    pred_cal: np.ndarray,
    y_cal: np.ndarray,
    *,
    ridge_lambda: float,
) -> Tuple[ResidualCalibrator, Dict[str, float]]:
    pred_cal = np.asarray(pred_cal, dtype=np.float32).reshape(-1)
    y_cal = np.asarray(y_cal, dtype=np.float32).reshape(-1)
    q_source = np.asarray(q_source, dtype=np.float32)
    q_cal = np.asarray(q_cal, dtype=np.float32)
    if pred_cal.size == 0:
        coef = np.zeros(q_cal.shape[1] + 1, dtype=np.float32)
        return ResidualCalibrator(q_source.mean(axis=0), q_source.std(axis=0) + 1e-6, coef), {
            "residual_n_cal": 0,
            "residual_n_features": int(q_cal.shape[1]),
        }
    q_mean = q_source.mean(axis=0)
    q_std = q_source.std(axis=0) + 1e-6
    qn = (q_cal - q_mean.reshape(1, -1)) / q_std.reshape(1, -1)
    design = np.column_stack([np.ones(qn.shape[0], dtype=np.float32), qn]).astype(np.float32)
    residual = y_cal - pred_cal
    l2 = np.full(design.shape[1], float(ridge_lambda), dtype=np.float32)
    l2[0] = 0.0
    coef = _ridge_solve(design, residual, l2)
    cal_post = pred_cal + design @ coef.reshape(-1, 1).reshape(-1)
    return ResidualCalibrator(q_mean.astype(np.float32), q_std.astype(np.float32), coef.astype(np.float32)), {
        "residual_n_cal": int(pred_cal.size),
        "residual_n_features": int(q_cal.shape[1]),
        "residual_ridge_lambda": float(ridge_lambda),
        "residual_coef_l2": float(np.sqrt(np.sum(coef[1:] * coef[1:]))),
        "residual_intercept": float(coef[0]),
        "residual_cal_mae": float(np.mean(np.abs(cal_post - y_cal))),
        "residual_cal_rmse": float(np.sqrt(np.mean((cal_post - y_cal) ** 2))),
        "residual_cal_corr": _safe_corr(y_cal, cal_post),
    }


# ---------------------------------------------------------------------------
# RR-bin calibration and conditional alignment
# ---------------------------------------------------------------------------


@dataclass
class RRBinOffsetCalibrator:
    edges: np.ndarray
    offsets: np.ndarray
    global_offset: float

    def apply(self, pred: np.ndarray) -> np.ndarray:
        pred = np.asarray(pred, dtype=np.float32).reshape(-1)
        bins = _assign_bins(pred, self.edges)
        return pred + self.offsets[bins]


def fit_rrbin_centroid_calibrator(
    pred_source: np.ndarray,
    pred_cal: np.ndarray,
    y_cal: np.ndarray,
    *,
    n_bins: int,
    shrink: float,
) -> Tuple[RRBinOffsetCalibrator, Dict[str, float]]:
    pred_source = np.asarray(pred_source, dtype=np.float32).reshape(-1)
    pred_cal = np.asarray(pred_cal, dtype=np.float32).reshape(-1)
    y_cal = np.asarray(y_cal, dtype=np.float32).reshape(-1)
    edges = _make_quantile_edges(pred_source, int(n_bins))
    n_effective = edges.size - 1
    residual = y_cal - pred_cal
    global_offset = float(residual.mean()) if residual.size else 0.0
    offsets = np.zeros(n_effective, dtype=np.float32)
    counts = np.zeros(n_effective, dtype=np.int64)
    bins = _assign_bins(pred_cal, edges) if pred_cal.size else np.zeros(0, dtype=np.int64)
    for b in range(n_effective):
        mask = bins == b
        counts[b] = int(mask.sum())
        if counts[b] == 0:
            offsets[b] = global_offset
        else:
            local = float(residual[mask].mean())
            offsets[b] = float((counts[b] * local + float(shrink) * global_offset) / (counts[b] + float(shrink)))
    cal_post = pred_cal + offsets[bins] if pred_cal.size else pred_cal
    return RRBinOffsetCalibrator(edges=edges.astype(np.float32), offsets=offsets.astype(np.float32), global_offset=global_offset), {
        "rrbin_n_bins": int(n_effective),
        "rrbin_shrink": float(shrink),
        "rrbin_global_offset": float(global_offset),
        "rrbin_min_count": int(counts.min()) if counts.size else 0,
        "rrbin_max_count": int(counts.max()) if counts.size else 0,
        "rrbin_counts_json": json.dumps(counts.tolist()),
        "rrbin_offsets_json": json.dumps([float(v) for v in offsets.tolist()]),
        "rrbin_cal_mae": float(np.mean(np.abs(cal_post - y_cal))) if pred_cal.size else float("nan"),
        "rrbin_cal_rmse": float(np.sqrt(np.mean((cal_post - y_cal) ** 2))) if pred_cal.size else float("nan"),
        "rrbin_cal_corr": _safe_corr(y_cal, cal_post) if pred_cal.size else float("nan"),
    }


@dataclass
class RRBinMomentStats:
    edges: np.ndarray
    source_mean: np.ndarray
    source_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    counts: np.ndarray


def fit_rrbin_moment_stats(
    z_source: np.ndarray,
    y_source: np.ndarray,
    z_target_moments: np.ndarray,
    pred_target_moments: np.ndarray,
    *,
    n_bins: int,
    min_count: int,
    shrink: float,
) -> RRBinMomentStats:
    z_source = np.asarray(z_source, dtype=np.float32)
    z_target_moments = np.asarray(z_target_moments, dtype=np.float32)
    edges = _make_quantile_edges(y_source, int(n_bins))
    n_effective = edges.size - 1
    d = z_source.shape[1]

    src_global_mean = z_source.mean(axis=0)
    src_global_std = z_source.std(axis=0) + 1e-6
    tgt_global_mean = z_target_moments.mean(axis=0) if z_target_moments.shape[0] else src_global_mean.copy()
    tgt_global_std = z_target_moments.std(axis=0) + 1e-6 if z_target_moments.shape[0] else src_global_std.copy()

    source_mean = np.zeros((n_effective, d), dtype=np.float32)
    source_std = np.zeros((n_effective, d), dtype=np.float32)
    target_mean = np.zeros((n_effective, d), dtype=np.float32)
    target_std = np.zeros((n_effective, d), dtype=np.float32)
    counts = np.zeros(n_effective, dtype=np.int64)

    source_bins = _assign_bins(y_source, edges)
    target_bins = _assign_bins(pred_target_moments, edges) if z_target_moments.shape[0] else np.zeros(0, dtype=np.int64)

    for b in range(n_effective):
        smask = source_bins == b
        if smask.sum() >= 2:
            source_mean[b] = z_source[smask].mean(axis=0)
            source_std[b] = z_source[smask].std(axis=0) + 1e-6
        else:
            source_mean[b] = src_global_mean
            source_std[b] = src_global_std

        tmask = target_bins == b
        counts[b] = int(tmask.sum())
        if counts[b] >= int(min_count):
            local_mean = z_target_moments[tmask].mean(axis=0)
            local_std = z_target_moments[tmask].std(axis=0) + 1e-6 if counts[b] >= 2 else tgt_global_std
            alpha = float(counts[b]) / max(1e-12, float(counts[b]) + float(shrink))
            target_mean[b] = alpha * local_mean + (1.0 - alpha) * tgt_global_mean
            target_std[b] = alpha * local_std + (1.0 - alpha) * tgt_global_std
        else:
            target_mean[b] = tgt_global_mean
            target_std[b] = tgt_global_std
    return RRBinMomentStats(
        edges=edges.astype(np.float32),
        source_mean=source_mean.astype(np.float32),
        source_std=source_std.astype(np.float32),
        target_mean=target_mean.astype(np.float32),
        target_std=target_std.astype(np.float32),
        counts=counts.astype(np.int64),
    )


def apply_rrbin_moment_alignment(z: np.ndarray, pred: np.ndarray, stats: RRBinMomentStats) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    bins = _assign_bins(pred, stats.edges)
    out = np.empty_like(z, dtype=np.float32)
    for b in range(stats.edges.size - 1):
        mask = bins == b
        if not np.any(mask):
            continue
        out[mask] = (
            (z[mask] - stats.target_mean[b].reshape(1, -1))
            / stats.target_std[b].reshape(1, -1)
            * stats.source_std[b].reshape(1, -1)
            + stats.source_mean[b].reshape(1, -1)
        )
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Source-distribution feature canonicalization test modes
# ---------------------------------------------------------------------------

# Externally, these are adaptation modes. The legacy feature_* names are kept
# as aliases so previous alignment sweeps remain reproducible.
FEATURE_ADAPTATION_MODES = {
    # Conservative test-time feature-mean alignment before the frozen RR readout.
    "feature_mean_align_alpha050",
    "feature_mean_align_alpha075",
    "feature_mean_align_alpha100",
    "feature_mean_align_profile_shrink",
    # New focused tests.
    "adapt_mean_alpha_000",
    "adapt_mean_alpha_025",
    "adapt_mean_alpha_050",
    "adapt_mean_alpha_075",
    "adapt_mean_alpha_100",
    "adapt_mean_fixed",
    "adapt_mean_profile_shrink",
    # Backward-compatible alignment aliases.
    "feature_mean_align",
    "feature_diag_align",
    "feature_rr_orthogonal_align",
    "feature_rrbin_diag_align",
}
FEATURE_ALIGNMENT_MODES = FEATURE_ADAPTATION_MODES

FEATURE_MEAN_ALIGNMENT_MODES = {
    "feature_mean_align_alpha050",
    "feature_mean_align_alpha075",
    "feature_mean_align_alpha100",
    "feature_mean_align_profile_shrink",
}


def _finite_rows(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    if z.ndim != 2 or z.shape[0] == 0:
        return np.zeros(z.shape[0] if z.ndim == 2 else 0, dtype=bool)
    return np.isfinite(z).all(axis=1)


def _feature_moments(z: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=np.float32)
    keep = _finite_rows(z)
    if z.ndim != 2 or z.shape[0] == 0 or not np.any(keep):
        d = int(z.shape[1]) if z.ndim == 2 else 0
        return np.zeros(d, dtype=np.float32), np.ones(d, dtype=np.float32)
    zz = z[keep]
    return zz.mean(axis=0).astype(np.float32), (zz.std(axis=0) + float(eps)).astype(np.float32)


def compute_feature_mean_alignment_stats(
    z_source: np.ndarray,
    z_target: np.ndarray,
    eps: float = 1e-6,
) -> Dict[str, object]:
    """Compute no-label target-to-source feature mean alignment statistics."""
    z_source = np.asarray(z_source, dtype=np.float32)
    z_target = np.asarray(z_target, dtype=np.float32)
    if z_source.ndim != 2:
        raise ValueError(f"Expected z_source to be 2D, got shape={z_source.shape}")
    if z_target.ndim != 2:
        raise ValueError(f"Expected z_target to be 2D, got shape={z_target.shape}")
    if z_source.shape[1] != z_target.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch for alignment: source={z_source.shape[1]} target={z_target.shape[1]}"
        )
    src_keep = _finite_rows(z_source)
    tgt_keep = _finite_rows(z_target)
    if not np.any(src_keep):
        raise ValueError("No finite source rows available for feature mean alignment.")
    if not np.any(tgt_keep):
        raise ValueError("No finite target rows available for feature mean alignment.")

    source_mean = z_source[src_keep].mean(axis=0).astype(np.float32)
    target_mean = z_target[tgt_keep].mean(axis=0).astype(np.float32)
    source_std = (z_source[src_keep].std(axis=0) + float(eps)).astype(np.float32)
    target_std = (z_target[tgt_keep].std(axis=0) + float(eps)).astype(np.float32)
    delta = (source_mean - target_mean).astype(np.float32)
    dim = max(1, int(source_mean.size))
    mean_shift_norm = float(np.linalg.norm(target_mean - source_mean) / np.sqrt(dim))
    var_shift_norm = float(np.linalg.norm(target_std - source_std) / np.sqrt(dim))
    source_scale = float(np.mean(source_std) + float(eps))
    normalized_mean_shift = float(mean_shift_norm / source_scale)
    normalized_var_shift = float(var_shift_norm / source_scale)
    return {
        "source_mean": source_mean,
        "target_mean": target_mean,
        "source_std": source_std,
        "target_std": target_std,
        "delta": delta,
        "mean_shift_norm": mean_shift_norm,
        "var_shift_norm": var_shift_norm,
        "normalized_mean_shift": normalized_mean_shift,
        "normalized_var_shift": normalized_var_shift,
        "source_n": int(src_keep.sum()),
        "target_n": int(tgt_keep.sum()),
        "feature_dim": int(source_mean.size),
    }


def resolve_feature_alignment_alpha(
    mode: str,
    stats: Dict[str, object],
    max_alpha: float = 0.75,
    eps: float = 1e-6,
) -> float:
    mode = str(mode).lower()
    lookup = {
        "none": 0.0,
        "feature_mean_align_alpha050": 0.50,
        "feature_mean_align_alpha075": 0.75,
        "feature_mean_align_alpha100": 1.00,
    }
    if mode in lookup:
        return float(lookup[mode])
    if mode == "feature_mean_align_profile_shrink":
        mean_shift = float(stats.get("normalized_mean_shift", 0.0))
        var_shift = float(stats.get("normalized_var_shift", 0.0))
        alpha_raw = mean_shift / (mean_shift + var_shift + float(eps))
        return float(np.clip(alpha_raw, 0.0, float(max_alpha)))
    raise ValueError(f"Unsupported feature mean alignment mode={mode!r}")


def apply_feature_mean_alignment(
    z: np.ndarray,
    stats: Dict[str, object],
    alpha: float,
) -> np.ndarray:
    delta = np.asarray(stats["delta"], dtype=np.float32).reshape(1, -1)
    return (np.asarray(z, dtype=np.float32) + float(alpha) * delta).astype(np.float32)


def _feature_alignment_stats_diagnostics(
    *,
    mode: str,
    stats: Dict[str, object],
    alpha: float,
    target_source: str,
) -> Dict[str, object]:
    delta = np.asarray(stats.get("delta", np.zeros(0, dtype=np.float32)), dtype=np.float32).reshape(-1)
    return {
        "feature_align_mode": str(mode).lower(),
        "feature_align_alpha": float(alpha),
        "feature_align_mean_shift_norm": float(stats.get("mean_shift_norm", float("nan"))),
        "feature_align_var_shift_norm": float(stats.get("var_shift_norm", float("nan"))),
        "feature_align_normalized_mean_shift": float(stats.get("normalized_mean_shift", float("nan"))),
        "feature_align_normalized_var_shift": float(stats.get("normalized_var_shift", float("nan"))),
        "feature_align_delta_norm": float(np.linalg.norm(delta)) if delta.size else float("nan"),
        "feature_align_source_n": int(stats.get("source_n", 0)),
        "feature_align_target_n": int(stats.get("target_n", 0)),
        "feature_align_feature_dim": int(stats.get("feature_dim", delta.size)),
        "feature_align_target_source": str(target_source),
    }


def _attach_feature_alignment_metric_aliases(metrics: Dict[str, object], mode: str) -> None:
    if "feature_align_mode" not in metrics:
        metrics["feature_align_mode"] = str(mode).lower()
    if "feature_align_alpha" not in metrics:
        metrics["feature_align_alpha"] = 0.0 if str(mode).lower() == "none" else float("nan")
    for key in (
        "feature_align_mean_shift_norm",
        "feature_align_var_shift_norm",
        "feature_align_normalized_mean_shift",
        "feature_align_normalized_var_shift",
        "feature_align_delta_norm",
    ):
        metrics.setdefault(key, float("nan"))
    for key in ("feature_align_source_n", "feature_align_target_n", "feature_align_feature_dim"):
        metrics.setdefault(key, 0)
    metrics.setdefault("feature_align_target_source", "")

    pre_mae = float(metrics.get("rr_probe_pre_mae", float("nan")))
    post_mae = float(metrics.get("rr_probe_post_mae", float("nan")))
    pre_corr = float(metrics.get("rr_probe_pre_corr", float("nan")))
    post_corr = float(metrics.get("rr_probe_post_corr", float("nan")))
    metrics["feature_align_pre_mae"] = pre_mae
    metrics["feature_align_post_mae"] = post_mae
    metrics["feature_align_delta_mae_vs_none"] = (
        float(post_mae - pre_mae) if np.isfinite(pre_mae) and np.isfinite(post_mae) else float("nan")
    )
    metrics["feature_align_pre_corr"] = pre_corr
    metrics["feature_align_post_corr"] = post_corr
    metrics["feature_align_delta_corr_vs_none"] = (
        float(post_corr - pre_corr) if np.isfinite(pre_corr) and np.isfinite(post_corr) else float("nan")
    )


def _feature_alignment_use_calibration_only(args) -> bool:
    feature_flag = getattr(args, "feature_align_use_calibration_only", None)
    adapt_flag = getattr(args, "adaptation_use_calibration_only", None)
    if feature_flag is False or adapt_flag is False:
        return False
    if adapt_flag is not None:
        return bool(adapt_flag)
    if feature_flag is not None:
        return bool(feature_flag)
    return True


def _align_mean(z: np.ndarray, src_mean: np.ndarray, tgt_mean: np.ndarray) -> np.ndarray:
    return (np.asarray(z, dtype=np.float32) - tgt_mean.reshape(1, -1) + src_mean.reshape(1, -1)).astype(np.float32)


def _align_diag(
    z: np.ndarray,
    src_mean: np.ndarray,
    src_std: np.ndarray,
    tgt_mean: np.ndarray,
    tgt_std: np.ndarray,
) -> np.ndarray:
    return (((np.asarray(z, dtype=np.float32) - tgt_mean.reshape(1, -1)) / tgt_std.reshape(1, -1))
            * src_std.reshape(1, -1) + src_mean.reshape(1, -1)).astype(np.float32)


def _align_mean_shrink(
    z: np.ndarray,
    src_mean: np.ndarray,
    tgt_mean: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Shrinkage version of target-to-source feature-mean canonicalization."""
    a = float(np.clip(alpha, 0.0, 1.0))
    return (np.asarray(z, dtype=np.float32) + a * (src_mean.reshape(1, -1) - tgt_mean.reshape(1, -1))).astype(np.float32)


def _alpha_from_adaptation_mode(mode: str) -> Optional[float]:
    lookup = {
        "feature_mean_align_alpha050": 0.50,
        "feature_mean_align_alpha075": 0.75,
        "feature_mean_align_alpha100": 1.00,
        "adapt_mean_alpha_000": 0.0,
        "adapt_mean_alpha_025": 0.25,
        "adapt_mean_alpha_050": 0.50,
        "adapt_mean_alpha_075": 0.75,
        "adapt_mean_alpha_100": 1.00,
        "adapt_mean_fixed": 1.00,
        "feature_mean_align": 1.00,
    }
    return lookup.get(str(mode).lower())


def _sigmoid_scalar(x: float) -> float:
    if not np.isfinite(x):
        return 0.5
    if x >= 40.0:
        return 1.0
    if x <= -40.0:
        return 0.0
    return float(1.0 / (1.0 + np.exp(-x)))


def _profile_shrink_alpha(
    *,
    feature_mean_shift_rms: float,
    diag_log_std_ratio_abs_mean: float,
    profile_rms_z: Optional[float],
    args,
) -> float:
    """No-label profile-conditioned shrinkage for mean canonicalization.

    The gate increases alpha when the target feature mean is displaced from the
    source distribution, but suppresses alpha when the profile/statistics look
    highly out-of-source or when variance shift suggests mean-only correction is
    insufficient. The weights are transparent hyperparameters; source-subject
    pseudo-target training can later learn/replace them.
    """
    alpha_min = float(getattr(args, "adapt_profile_alpha_min", 0.0))
    alpha_max = float(getattr(args, "adapt_profile_alpha_max", 1.0))
    alpha_min = float(np.clip(alpha_min, 0.0, 1.0))
    alpha_max = float(np.clip(alpha_max, alpha_min, 1.0))

    shift_mid = float(getattr(args, "adapt_profile_shift_mid", 0.10))
    shift_scale = max(1e-6, float(getattr(args, "adapt_profile_shift_scale", 0.05)))
    z_safe = float(getattr(args, "adapt_profile_z_safe", 3.0))
    z_scale = max(1e-6, float(getattr(args, "adapt_profile_z_scale", 1.0)))
    std_scale = max(1e-6, float(getattr(args, "adapt_profile_std_ratio_scale", 1.0)))
    bias = float(getattr(args, "adapt_profile_alpha_bias", 0.0))

    shift_term = (float(feature_mean_shift_rms) - shift_mid) / shift_scale
    profile_z = float(profile_rms_z) if profile_rms_z is not None and np.isfinite(profile_rms_z) else z_safe
    profile_penalty = max(0.0, profile_z - z_safe) / z_scale
    std_penalty = max(0.0, float(diag_log_std_ratio_abs_mean)) / std_scale
    raw = bias + shift_term - profile_penalty - std_penalty
    return alpha_min + (alpha_max - alpha_min) * _sigmoid_scalar(raw)


def _rr_probe_unit_direction(rr_model: FaithfulRRRegressor, dim: int) -> Optional[np.ndarray]:
    reg = getattr(rr_model, "regressor", None)
    if reg is None:
        return None
    linear_layers = [m for m in reg.modules() if isinstance(m, nn.Linear)]
    if not linear_layers:
        return None
    w = linear_layers[-1].weight.detach().cpu().numpy()
    if w.ndim != 2 or w.shape[0] != 1:
        return None
    w = w.reshape(-1).astype(np.float32)[:dim]
    n = float(np.linalg.norm(w))
    if not np.isfinite(n) or n <= 1e-8:
        return None
    return (w / n).astype(np.float32)


def _project_orthogonal(z: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    parallel_scalar = z @ w
    orth = z - np.outer(parallel_scalar, w).astype(np.float32)
    return parallel_scalar.astype(np.float32), orth.astype(np.float32)


def _apply_rr_orthogonal_diag_alignment(
    z_eval: np.ndarray,
    z_source: np.ndarray,
    z_target_moments: np.ndarray,
    rr_model: FaithfulRRRegressor,
    eps: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    d = int(z_eval.shape[1])
    w = _rr_probe_unit_direction(rr_model, d)
    if w is None:
        src_mean, src_std = _feature_moments(z_source, eps)
        tgt_mean, tgt_std = _feature_moments(z_target_moments, eps)
        return _align_diag(z_eval, src_mean, src_std, tgt_mean, tgt_std), {
            "feature_align_rr_weight_available": 0.0,
            "feature_align_rr_orthogonal_fallback_diag": 1.0,
        }
    par_eval, orth_eval = _project_orthogonal(z_eval[:, :d], w)
    _par_src, orth_src = _project_orthogonal(z_source[:, :d], w)
    _par_tgt, orth_tgt = _project_orthogonal(z_target_moments[:, :d], w)
    src_mean, src_std = _feature_moments(orth_src, eps)
    tgt_mean, tgt_std = _feature_moments(orth_tgt, eps)
    orth_aligned = _align_diag(orth_eval, src_mean, src_std, tgt_mean, tgt_std)
    # Diagonal scaling in coordinate space can reintroduce a small component
    # along w; remove it so the original RR-probe parallel coordinate is preserved.
    orth_aligned = orth_aligned - np.outer(orth_aligned @ w, w).astype(np.float32)
    z_aligned = np.outer(par_eval, w).astype(np.float32) + orth_aligned.astype(np.float32)
    return z_aligned.astype(np.float32), {
        "feature_align_rr_weight_available": 1.0,
        "feature_align_rr_orthogonal_fallback_diag": 0.0,
    }


def _rms_dist_to_source(z: np.ndarray, src_mean: np.ndarray, src_std: np.ndarray) -> float:
    z = np.asarray(z, dtype=np.float32)
    if z.ndim != 2 or z.shape[0] == 0:
        return float("nan")
    zz = (z - src_mean.reshape(1, -1)) / src_std.reshape(1, -1)
    return float(np.sqrt(np.mean(zz * zz)))


def _prediction_shift_diagnostics(
    *,
    y_true: np.ndarray,
    pred_base: np.ndarray,
    pred_post: np.ndarray,
    prefix: str,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    pred_base = np.asarray(pred_base, dtype=np.float32).reshape(-1)
    pred_post = np.asarray(pred_post, dtype=np.float32).reshape(-1)
    n = min(y_true.size, pred_base.size, pred_post.size)
    if n <= 0:
        return {f"{prefix}_n": 0.0}
    y_true = y_true[:n]
    pred_base = pred_base[:n]
    pred_post = pred_post[:n]
    shift = pred_post - pred_base
    base_resid = y_true - pred_base
    post_resid = y_true - pred_post
    agree = np.isfinite(shift) & np.isfinite(base_resid) & (np.abs(shift) > 1e-6) & (np.abs(base_resid) > 1e-6)
    return {
        f"{prefix}_n": float(n),
        f"{prefix}_pred_shift_signed_mean_bpm": float(np.mean(shift)),
        f"{prefix}_pred_shift_abs_mean_bpm": float(np.mean(np.abs(shift))),
        f"{prefix}_pred_shift_p95_abs_bpm": float(np.percentile(np.abs(shift), 95)),
        f"{prefix}_base_residual_signed_mean_bpm": float(np.mean(base_resid)),
        f"{prefix}_post_residual_signed_mean_bpm": float(np.mean(post_resid)),
        f"{prefix}_direction_dot_mean": float(np.mean(shift * base_resid)),
        f"{prefix}_direction_agree_frac": float(np.mean(np.sign(shift[agree]) == np.sign(base_resid[agree]))) if np.any(agree) else float("nan"),
        f"{prefix}_overshoot_frac": float(np.mean(np.abs(shift) > np.abs(base_resid))),
        f"{prefix}_reduces_abs_error_frac": float(np.mean(np.abs(post_resid) < np.abs(base_resid))),
        f"{prefix}_delta_mae": float(np.mean(np.abs(post_resid)) - np.mean(np.abs(base_resid))),
    }


def _feature_delta_diagnostics(
    z_before: np.ndarray,
    z_after: np.ndarray,
    rr_model: FaithfulRRRegressor,
    prefix: str,
) -> Dict[str, float]:
    z_before = np.asarray(z_before, dtype=np.float32)
    z_after = np.asarray(z_after, dtype=np.float32)
    n = min(z_before.shape[0], z_after.shape[0])
    d = min(z_before.shape[1], z_after.shape[1]) if z_before.ndim == 2 and z_after.ndim == 2 else 0
    if n <= 0 or d <= 0:
        return {f"{prefix}_feature_delta_n": 0.0}
    dz = z_after[:n, :d] - z_before[:n, :d]
    dz_norm = np.linalg.norm(dz, axis=1)
    dz_rms = float(np.sqrt(np.mean(dz * dz))) if dz.size else 0.0
    out: Dict[str, float] = {
        f"{prefix}_feature_delta_n": float(n),
        f"{prefix}_feature_delta_rms": dz_rms,
        f"{prefix}_hidden_delta_rms": dz_rms,
        f"{prefix}_feature_delta_norm_mean": float(np.mean(dz_norm)),
        f"{prefix}_feature_delta_norm_p95": float(np.percentile(dz_norm, 95)),
    }
    w = _rr_probe_unit_direction(rr_model, d)
    if w is not None:
        par = dz @ w
        orth = np.sqrt(np.maximum(dz_norm * dz_norm - par * par, 0.0))
        frac = np.abs(par) / (dz_norm + 1e-8)
        out.update({
            f"{prefix}_feature_delta_rr_weight_available": 1.0,
            f"{prefix}_feature_delta_rr_parallel_mean": float(np.mean(par)),
            f"{prefix}_feature_delta_rr_parallel_abs_mean": float(np.mean(np.abs(par))),
            f"{prefix}_feature_delta_orthogonal_norm_mean": float(np.mean(orth)),
            f"{prefix}_feature_delta_parallel_fraction_mean": float(np.mean(frac)),
        })
    else:
        out[f"{prefix}_feature_delta_rr_weight_available"] = 0.0
    return out


def apply_feature_alignment_mode(
    *,
    mode: str,
    z_source: np.ndarray,
    y_source: np.ndarray,
    z_target_moments: np.ndarray,
    pred_target_moments: np.ndarray,
    z_eval: np.ndarray,
    pred_eval: np.ndarray,
    rr_model: FaithfulRRRegressor,
    args,
    profile_rms_z: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Unsupervised source-distribution canonicalization of frozen target features."""
    eps = float(getattr(args, "feature_align_diag_eps", 1e-6))
    src_mean, src_std = _feature_moments(z_source, eps)
    tgt_mean, tgt_std = _feature_moments(z_target_moments, eps)
    mode = str(mode).lower()
    use_cal_only = _feature_alignment_use_calibration_only(args)
    feature_mean_shift_rms = float(np.linalg.norm(src_mean - tgt_mean) / np.sqrt(max(1, src_mean.size))) if src_mean.size else float("nan")
    diag_log_std_ratio_abs_mean = float(np.mean(np.abs(np.log((tgt_std + eps) / (src_std + eps))))) if src_std.size else float("nan")
    mean_align_stats = compute_feature_mean_alignment_stats(z_source, z_target_moments, eps=eps)
    extra: Dict[str, float] = {
        "feature_align_mode": mode,
        "feature_align_n_source_windows": int(z_source.shape[0]),
        "feature_align_n_moment_windows": int(z_target_moments.shape[0]),
        "feature_align_use_calibration_only": int(use_cal_only),
        "feature_align_source_rms_std_mean": float(np.mean(src_std)) if src_std.size else float("nan"),
        "feature_align_target_rms_std_mean": float(np.mean(tgt_std)) if tgt_std.size else float("nan"),
        "feature_align_eval_dist_to_source_before": _rms_dist_to_source(z_eval, src_mean, src_std),
        # New adaptation-prefixed aliases for cleaner downstream tables.
        "feature_adapt_mode": mode,
        "feature_adapt_n_source_windows": int(z_source.shape[0]),
        "feature_adapt_n_moment_windows": int(z_target_moments.shape[0]),
        "feature_adapt_use_calibration_only": int(use_cal_only),
        "feature_adapt_profile_rms_z": float(profile_rms_z) if profile_rms_z is not None and np.isfinite(profile_rms_z) else float("nan"),
        "feature_adapt_feature_mean_shift_rms": feature_mean_shift_rms,
        "feature_adapt_diag_log_std_ratio_abs_mean": diag_log_std_ratio_abs_mean,
    }
    alpha = _alpha_from_adaptation_mode(mode)
    if mode == "feature_mean_align_profile_shrink":
        alpha = resolve_feature_alignment_alpha(
            mode,
            mean_align_stats,
            max_alpha=float(getattr(args, "feature_align_profile_shrink_max_alpha", 0.75)),
            eps=eps,
        )
    elif mode == "adapt_mean_profile_shrink":
        alpha = _profile_shrink_alpha(
            feature_mean_shift_rms=feature_mean_shift_rms,
            diag_log_std_ratio_abs_mean=diag_log_std_ratio_abs_mean,
            profile_rms_z=profile_rms_z,
            args=args,
        )
    if alpha is not None:
        if mode in FEATURE_MEAN_ALIGNMENT_MODES:
            z_aligned = apply_feature_mean_alignment(z_eval, mean_align_stats, alpha)
        else:
            z_aligned = _align_mean_shrink(z_eval, src_mean, tgt_mean, alpha)
        extra["feature_align_kind"] = "mean_shrink"
        extra["feature_align_alpha"] = float(alpha)
        extra["feature_adapt_kind"] = "mean_shrink"
        extra["feature_adapt_alpha"] = float(alpha)
        extra.update(_feature_alignment_stats_diagnostics(
            mode=mode,
            stats=mean_align_stats,
            alpha=float(alpha),
            target_source=str(getattr(args, "_feature_align_target_source", "unknown-unlabeled")),
        ))
    elif mode == "feature_diag_align":
        z_aligned = _align_diag(z_eval, src_mean, src_std, tgt_mean, tgt_std)
        extra["feature_align_kind"] = "diag"
        extra.update(_feature_alignment_stats_diagnostics(
            mode=mode,
            stats=mean_align_stats,
            alpha=1.0,
            target_source=str(getattr(args, "_feature_align_target_source", "unknown-unlabeled")),
        ))
    elif mode == "feature_rr_orthogonal_align":
        z_aligned, rr_extra = _apply_rr_orthogonal_diag_alignment(
            z_eval, z_source, z_target_moments, rr_model, eps
        )
        extra.update(rr_extra)
        extra["feature_align_kind"] = "rr_orthogonal_diag"
    elif mode == "feature_rrbin_diag_align":
        n_bins = int(getattr(args, "feature_align_rrbin_n_bins", getattr(args, "rrbin_n_bins", 3)))
        min_count = int(getattr(args, "feature_align_min_bin_count", getattr(args, "rrbin_min_bin_count", 4)))
        shrink = float(getattr(args, "feature_align_shrink", getattr(args, "rrbin_shrink", 8.0)))
        moment_stats = fit_rrbin_moment_stats(
            z_source,
            y_source,
            z_target_moments,
            pred_target_moments,
            n_bins=n_bins,
            min_count=min_count,
            shrink=shrink,
        )
        z_aligned = apply_rrbin_moment_alignment(z_eval, pred_eval, moment_stats)
        extra.update({
            "feature_align_kind": "rrbin_diag",
            "feature_align_rrbin_n_bins": int(moment_stats.edges.size - 1),
            "feature_align_min_bin_count": int(min_count),
            "feature_align_shrink": float(shrink),
            "feature_align_rrbin_min_count": int(moment_stats.counts.min()) if moment_stats.counts.size else 0,
            "feature_align_rrbin_max_count": int(moment_stats.counts.max()) if moment_stats.counts.size else 0,
            "feature_align_rrbin_counts_json": json.dumps(moment_stats.counts.tolist()),
        })
    else:
        raise ValueError(f"Unsupported feature alignment mode={mode!r}")

    dist_after = _rms_dist_to_source(z_aligned, src_mean, src_std)
    dist_delta = dist_after - extra["feature_align_eval_dist_to_source_before"]
    extra.update({
        "feature_align_eval_dist_to_source_after": dist_after,
        "feature_align_eval_dist_delta_after_minus_before": dist_delta,
        "feature_align_mean_shift_l2": float(np.linalg.norm(src_mean - tgt_mean)),
        "feature_align_diag_log_std_ratio_abs_mean": diag_log_std_ratio_abs_mean,
        "feature_adapt_eval_dist_to_source_before": extra["feature_align_eval_dist_to_source_before"],
        "feature_adapt_eval_dist_to_source_after": dist_after,
        "feature_adapt_eval_dist_delta_after_minus_before": dist_delta,
        "feature_adapt_mean_shift_l2": float(np.linalg.norm(src_mean - tgt_mean)),
    })
    return z_aligned.astype(np.float32), extra


# ---------------------------------------------------------------------------
# Evaluation hook
# ---------------------------------------------------------------------------


def rr_structured_adaptation_evaluate(
    model: nn.Module,
    train_loader,
    test_loader,
    subject: str,
    device: str,
    args,
    shared_context: Optional[SubjectEvalContext] = None,
    out_dir: Optional[Path] = None,
) -> Dict[str, float]:
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    mode = str(args.rr_tta).lower()
    need_kin = mode in {"kin_ssa", "ssa_kin"}
    ctx = shared_context or _build_subject_eval_context(
        model, train_loader, test_loader, device, args, include_kinematics=need_kin
    )
    x_source, y_source, k_source = ctx.x_source, ctx.y_source, ctx.k_source
    x_target, y_target, k_target = ctx.x_target, ctx.y_target, ctx.k_target
    x_cal, y_cal, k_cal = ctx.x_cal, ctx.y_cal, ctx.k_cal
    x_eval, y_eval, k_eval, cal_idx = ctx.x_eval, ctx.y_eval, ctx.k_eval, ctx.cal_idx
    rr_model = FaithfulRRRegressor(x_source.shape[1]).to(device)
    rr_model.load_state_dict(ctx.rr_model_state, strict=True)
    pred_pre = predict_rr(rr_model, x_eval, device)
    pred_source = predict_rr(rr_model, x_source, device)
    pred_cal = predict_rr(rr_model, x_cal, device) if x_cal.shape[0] else np.zeros(0, dtype=np.float32)
    pred_target = predict_rr(rr_model, x_target, device)

    metrics: Dict[str, float] = rr_metrics(y_eval, pred_pre, prefix="rr_probe_pre")
    _attach_base_pre_aliases(metrics)
    extra: Dict[str, float] = {}

    if mode == "none":
        pred_post = pred_pre
        try:
            z_source_none = predict_features(rr_model, x_source, device)
            use_cal_only = _feature_alignment_use_calibration_only(args)
            if use_cal_only and x_cal.shape[0]:
                z_target_none = predict_features(rr_model, x_cal, device)
                target_source = "calibration-unlabeled"
            else:
                z_target_none = predict_features(rr_model, x_target, device)
                target_source = "all-target-unlabeled"
            if z_target_none.shape[0] == 0:
                z_target_none = predict_features(rr_model, x_eval, device)
                target_source = "transductive-unlabeled"
            stats_none = compute_feature_mean_alignment_stats(
                z_source_none,
                z_target_none,
                eps=float(getattr(args, "feature_align_diag_eps", 1e-6)),
            )
            extra.update(_feature_alignment_stats_diagnostics(
                mode="none",
                stats=stats_none,
                alpha=0.0,
                target_source=target_source,
            ))
            extra.update({
                "feature_align_kind": "none",
                "feature_align_eval_feature_delta_rms": 0.0,
                "feature_align_eval_hidden_delta_rms": 0.0,
                "feature_align_eval_feature_delta_norm_mean": 0.0,
                "feature_align_eval_feature_delta_norm_p95": 0.0,
            })
        except Exception as exc:
            extra.update({
                "feature_align_mode": "none",
                "feature_align_alpha": 0.0,
                "feature_align_stats_error": str(exc),
            })

    elif mode == "affine_cal":
        cal, extra = fit_affine_calibration(
            pred_cal,
            y_cal,
            lambda_a=float(args.affine_lambda_a),
            lambda_b=float(args.affine_lambda_b),
            monotone=False,
        )
        pred_post = cal.apply(pred_pre)

    elif mode == "affine_cal_ridge":
        cal, extra = fit_affine_calibration(
            pred_cal,
            y_cal,
            lambda_a=float(args.affine_ridge_lambda_a),
            lambda_b=float(args.affine_ridge_lambda_b),
            monotone=False,
        )
        pred_post = cal.apply(pred_pre)

    elif mode == "affine_cal_monotone":
        cal, extra = fit_affine_calibration(
            pred_cal,
            y_cal,
            lambda_a=float(args.affine_ridge_lambda_a),
            lambda_b=float(args.affine_ridge_lambda_b),
            monotone=True,
            min_slope=float(args.affine_min_slope),
        )
        pred_post = cal.apply(pred_pre)

    elif mode == "ridge_residual_cal":
        z_source = predict_features(rr_model, x_source, device)
        z_cal = predict_features(rr_model, x_cal, device) if x_cal.shape[0] else np.zeros((0, x_source.shape[1]), dtype=np.float32)
        z_eval = predict_features(rr_model, x_eval, device)
        stats = _fit_source_profile_stats(z_source, pred_source, y_source, int(args.rrbin_n_bins))
        q_source = _quality_features(z_source, pred_source, stats)
        q_cal = _quality_features(z_cal, pred_cal, stats) if z_cal.shape[0] else np.zeros((0, q_source.shape[1]), dtype=np.float32)
        q_eval = _quality_features(z_eval, pred_pre, stats)
        calibrator, extra = fit_ridge_residual_calibrator(
            q_source,
            q_cal,
            pred_cal,
            y_cal,
            ridge_lambda=float(args.residual_ridge_lambda),
        )
        pred_post = calibrator.apply(pred_pre, q_eval)

    elif mode == "rrbin_centroid_cal":
        cal, extra = fit_rrbin_centroid_calibrator(
            pred_source,
            pred_cal,
            y_cal,
            n_bins=int(args.rrbin_n_bins),
            shrink=float(args.rrbin_shrink),
        )
        pred_post = cal.apply(pred_pre)

    elif mode == "rrbin_ssa":
        z_source = predict_features(rr_model, x_source, device)
        z_eval = predict_features(rr_model, x_eval, device)
        z_cal = predict_features(rr_model, x_cal, device) if x_cal.shape[0] else np.zeros((0, x_source.shape[1]), dtype=np.float32)
        if bool(args.rrbin_use_calibration_only):
            z_mom = z_cal
            pred_mom = pred_cal
        else:
            z_mom = predict_features(rr_model, x_target, device)
            pred_mom = pred_target
        moment_stats = fit_rrbin_moment_stats(
            z_source,
            y_source,
            z_mom,
            pred_mom,
            n_bins=int(args.rrbin_n_bins),
            min_count=int(args.rrbin_min_bin_count),
            shrink=float(args.rrbin_shrink),
        )
        z_eval_aligned = apply_rrbin_moment_alignment(z_eval, pred_pre, moment_stats)
        pred_post = _linear_predict_from_features(rr_model, z_eval_aligned, device)
        extra = {
            "rrbin_n_bins": int(moment_stats.edges.size - 1),
            "rrbin_min_bin_count": int(args.rrbin_min_bin_count),
            "rrbin_shrink": float(args.rrbin_shrink),
            "rrbin_moment_counts_json": json.dumps(moment_stats.counts.tolist()),
            "rrbin_moment_min_count": int(moment_stats.counts.min()) if moment_stats.counts.size else 0,
            "rrbin_moment_max_count": int(moment_stats.counts.max()) if moment_stats.counts.size else 0,
            "rrbin_used_all_target_unlabeled": int(not bool(args.rrbin_use_calibration_only)),
        }

    elif mode in FEATURE_ADAPTATION_MODES:
        z_source = predict_features(rr_model, x_source, device)
        z_eval = predict_features(rr_model, x_eval, device)
        z_cal = predict_features(rr_model, x_cal, device) if x_cal.shape[0] else np.zeros((0, x_source.shape[1]), dtype=np.float32)
        use_cal_only = _feature_alignment_use_calibration_only(args)
        target_source = "calibration-unlabeled"
        if use_cal_only:
            z_mom = z_cal
            pred_mom = pred_cal
            profile_windows = ctx.target_windows.subset(cal_idx) if cal_idx.size else ctx.eval_windows
        else:
            z_mom = predict_features(rr_model, x_target, device)
            pred_mom = pred_target
            profile_windows = ctx.target_windows
            target_source = "all-target-unlabeled"
        if z_mom.shape[0] == 0:
            z_mom = z_eval
            pred_mom = pred_pre
            profile_windows = ctx.eval_windows
            target_source = "transductive-unlabeled"

        profile_rms_z = None
        profile_max_abs_z = None
        try:
            _p_raw, p_norm = _profile_stats_from_windows(
                model,
                profile_windows,
                device,
                args,
                batch_size=int(getattr(args, "feature_eval_batch_size", 256)),
            )
            p_np = p_norm.detach().float().cpu().numpy().reshape(-1)
            if p_np.size:
                profile_rms_z = float(np.sqrt(np.mean(p_np * p_np)))
                profile_max_abs_z = float(np.max(np.abs(p_np)))
        except Exception:
            profile_rms_z = None
            profile_max_abs_z = None

        setattr(args, "_feature_align_target_source", target_source)
        z_eval_aligned, extra = apply_feature_alignment_mode(
            mode=mode,
            z_source=z_source,
            y_source=y_source,
            z_target_moments=z_mom,
            pred_target_moments=pred_mom,
            z_eval=z_eval,
            pred_eval=pred_pre,
            rr_model=rr_model,
            args=args,
            profile_rms_z=profile_rms_z,
        )
        extra["feature_align_target_source"] = target_source
        extra["feature_adapt_profile_max_abs_z"] = float(profile_max_abs_z) if profile_max_abs_z is not None and np.isfinite(profile_max_abs_z) else float("nan")
        pred_post = _linear_predict_from_features(rr_model, z_eval_aligned, device)
        extra.update(_prediction_shift_diagnostics(
            y_true=y_eval,
            pred_base=pred_pre,
            pred_post=pred_post,
            prefix="feature_align_eval",
        ))
        extra.update(_prediction_shift_diagnostics(
            y_true=y_eval,
            pred_base=pred_pre,
            pred_post=pred_post,
            prefix="feature_adapt_eval",
        ))
        extra.update(_feature_delta_diagnostics(
            z_before=z_eval,
            z_after=z_eval_aligned,
            rr_model=rr_model,
            prefix="feature_align_eval",
        ))
        extra.update(_feature_delta_diagnostics(
            z_before=z_eval,
            z_after=z_eval_aligned,
            rr_model=rr_model,
            prefix="feature_adapt_eval",
        ))
        if z_cal.shape[0] and y_cal.shape[0]:
            z_cal_aligned, _cal_extra_unused = apply_feature_alignment_mode(
                mode=mode,
                z_source=z_source,
                y_source=y_source,
                z_target_moments=z_mom,
                pred_target_moments=pred_mom,
                z_eval=z_cal,
                pred_eval=pred_cal,
                rr_model=rr_model,
                args=args,
                profile_rms_z=profile_rms_z,
            )
            pred_cal_aligned = _linear_predict_from_features(rr_model, z_cal_aligned, device)
            extra.update(_prediction_shift_diagnostics(
                y_true=y_cal,
                pred_base=pred_cal,
                pred_post=pred_cal_aligned,
                prefix="feature_align_cal",
            ))
            extra.update(_prediction_shift_diagnostics(
                y_true=y_cal,
                pred_base=pred_cal,
                pred_post=pred_cal_aligned,
                prefix="feature_adapt_cal",
            ))
        extra["uses_target_rr_labels_for_adaptation"] = 0

    elif mode == "ssa":
        rr_model, extra = adapt_ssa_original(
            rr_model,
            x_source,
            x_cal if bool(args.ssa_use_calibration_only) else x_target,
            args,
            device,
        )
        pred_post = predict_rr(rr_model, x_eval, device)

    elif mode == "kin_ssa":
        x_src_match, y_src_match = select_source_by_target_kinematics(
            x_source,
            y_source,
            k_source,
            k_cal,
            fraction=float(args.kin_source_fraction),
        )
        rr_model, extra = adapt_ssa_original(
            rr_model,
            x_src_match,
            x_cal if bool(args.ssa_use_calibration_only) else x_target,
            args,
            device,
        )
        extra["kin_source_n"] = int(x_src_match.shape[0])
        pred_post = predict_rr(rr_model, x_eval, device)

    elif mode == "cmt":
        predictor, extra = adapt_cmt_original_style(rr_model, x_source, y_source, x_cal, y_cal, args, device)
        pred_post = predict_rr(predictor, x_eval, device)

    elif mode == "ssa_cmt":
        rr_model, extra_ssa = adapt_ssa_original(
            rr_model,
            x_source,
            x_cal if bool(args.ssa_use_calibration_only) else x_target,
            args,
            device,
        )
        x_source_adapted = predict_features(rr_model, x_source, device)
        x_cal_adapted = predict_features(rr_model, x_cal, device) if x_cal.shape[0] else x_cal
        x_eval_adapted = predict_features(rr_model, x_eval, device)
        predictor, extra_cmt = adapt_cmt_original_style(
            rr_model,
            x_source_adapted,
            y_source,
            x_cal_adapted,
            y_cal,
            args,
            device,
        )
        pred_post = predict_rr(predictor, x_eval_adapted, device)
        extra = {**{f"ssa_{k}": v for k, v in extra_ssa.items()}, **extra_cmt}

    elif mode == "ssa_kin":
        x_src_match, y_src_match = select_source_by_target_kinematics(
            x_source,
            y_source,
            k_source,
            k_cal,
            fraction=float(args.kin_source_fraction),
        )
        rr_model, extra_ssa = adapt_ssa_original(
            rr_model,
            x_src_match,
            x_cal if bool(args.ssa_use_calibration_only) else x_target,
            args,
            device,
        )
        x_source_adapted = predict_features(rr_model, x_source, device)
        x_cal_adapted = predict_features(rr_model, x_cal, device) if x_cal.shape[0] else x_cal
        x_eval_adapted = predict_features(rr_model, x_eval, device)
        predictor, extra_cmt = adapt_cmt_original_style(
            rr_model,
            x_source_adapted,
            y_source,
            x_cal_adapted,
            y_cal,
            args,
            device,
        )
        pred_post = predict_rr(predictor, x_eval_adapted, device)
        extra = {**{f"ssa_{k}": v for k, v in extra_ssa.items()}, **extra_cmt, "kin_source_n": int(x_src_match.shape[0])}

    else:
        raise ValueError(f"Unsupported --rr-tta={args.rr_tta!r}")

    metrics.update(rr_metrics(y_eval, pred_post, prefix="rr_probe_post"))
    metrics.update(extra)
    _attach_feature_alignment_metric_aliases(metrics, mode)
    metrics.update({
        "rr_tta_mode": mode,
        "rr_probe_n_source": int(x_source.shape[0]),
        "rr_probe_n_target_total": int(x_target.shape[0]),
        "rr_probe_n_calibration": int(x_cal.shape[0]),
        "rr_probe_n_eval": int(x_eval.shape[0]),
        "rr_probe_n_features": int(x_source.shape[1]),
        "target_calibration_indices": json.dumps(cal_idx.tolist()),
        "eval_excludes_calibration": int(bool(args.exclude_calibration_from_eval)),
    })

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_df = pd.DataFrame({
            "rr_true": y_eval,
            "rr_pred_pre": pred_pre,
            "rr_pred_post": pred_post,
        })
        if mode in FEATURE_ADAPTATION_MODES:
            pred_df["feature_align_pred_shift"] = pred_post - pred_pre
            pred_df["feature_align_base_residual"] = y_eval - pred_pre
            pred_df["feature_align_post_residual"] = y_eval - pred_post
            pred_df["feature_align_direction_dot"] = pred_df["feature_align_pred_shift"] * pred_df["feature_align_base_residual"]
            pred_df["feature_align_direction_agree"] = (
                np.sign(pred_df["feature_align_pred_shift"].to_numpy()) == np.sign(pred_df["feature_align_base_residual"].to_numpy())
            ).astype(int)
            pred_df["feature_adapt_pred_shift"] = pred_df["feature_align_pred_shift"]
            pred_df["feature_adapt_base_residual"] = pred_df["feature_align_base_residual"]
            pred_df["feature_adapt_post_residual"] = pred_df["feature_align_post_residual"]
            pred_df["feature_adapt_direction_dot"] = pred_df["feature_align_direction_dot"]
            pred_df["feature_adapt_direction_agree"] = pred_df["feature_align_direction_agree"]
        pred_df.to_csv(out_dir / f"rr_structured_adaptation_predictions_{subject}.csv", index=False)
        with open(out_dir / f"rr_structured_adaptation_metrics_{subject}.json", "w") as f:
            json.dump(metrics, f, indent=2)
    return metrics


def rr_structured_adaptation_hook(model, sbj: str, train_loader, test_loader, device: str, args, sbj_dir: Path):
    metrics = rr_structured_adaptation_evaluate(
        model,
        train_loader,
        test_loader,
        sbj,
        device,
        args,
        out_dir=sbj_dir / "rr_structured_adaptation",
    )
    print(f"RR_STRUCTURED_ADAPTATION {sbj}: {metrics}")
    return {"__summary_name__": "rr_structured_adaptation_summary", "__summary_row__": {"subject": sbj, **metrics}}



def _add_argument_if_absent(parser, *name_or_flags, **kwargs):
    """Add an argparse option only if build_base_parser did not already define it."""
    for flag in name_or_flags:
        if isinstance(flag, str) and flag.startswith("-") and flag in parser._option_string_actions:
            return None
    return parser.add_argument(*name_or_flags, **kwargs)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Feature-space adaptation implementation (inlined)
# ---------------------------------------------------------------------------

SUPERVISED_FEATURE_MODES = {
    "ln_affine_cal",
    "ln_stft_rr_consistency",
    "lastblock_ln_affine_cal",
}

UNSUP_FEATURE_MODES = {
    "ln_unsup_stft_consistency",
    "lastblock_ln_unsup_stft_consistency",
    "lastblock_ln_unsup_stft_smooth",
    "lastblock_ln_unsup_stft_smooth_anchor",
}

PROFILE_VECTOR_UNSUP_MODES = {
    "profile_film_init_only",
    "profile_film_gated_init_only",
    "profile_film_oracle_init_only",
    "profile_film_unsup_stft_consistency",
    "profile_film_unsup_stft_smooth_anchor",
    "profile_film_unsup_readout_affine",
    "profile_film_unsup_sparc",
    "profile_film_gated_sparc",
    "profile_film_oracle_sparc",
    "profile_qkv_unsup_stft_consistency",
    "profile_qkv_unsup_stft_smooth_anchor",
    "profile_qkv_unsup_stft_smooth_prior_attn",
    "tcn_profile_film_qkv_last1_0p01",
    "tcn_profile_film_qkv_last1_0p01_sparc_pt",
    "tcn_profile_film_qkv_last1_0p01_sparc_pt_budget",
    "tcn_profile_film_qkv_last1_0p01_pt_no_stft",
    "tcn_profile_film_qkv_last1_0p01_pt_aux_only",
    "tcn_profile_film_qkv_last1_0p01_pt_reg_only",
    "tcn_profile_film_qkv_last1_0p01_pt_no_stft_budget",
    "tcn_profile_film_clsa_qkv_last1",
    "tcn_profile_film_clsa_qkv_last1_no_fast_update",
    "tcn_clsa_qkv_last1_no_film",
    "tcn_profile_film_clsa_qkv_last1_rank4",
}

READOUT_ABLATION_MODES = {
    "direct_stft_rr",
    "hybrid_probe_stft_conf",
}

FEATURE_ADAPTIVE_MODES = (
    SUPERVISED_FEATURE_MODES
    | UNSUP_FEATURE_MODES
    | PROFILE_VECTOR_UNSUP_MODES
    | READOUT_ABLATION_MODES
)

LASTBLOCK_FEATURE_MODES = {
    "lastblock_ln_affine_cal",
    "lastblock_ln_unsup_stft_consistency",
    "lastblock_ln_unsup_stft_smooth",
    "lastblock_ln_unsup_stft_smooth_anchor",
}


@dataclass
class TensorWindows:
    imu: torch.Tensor
    pressure: torch.Tensor
    rr: torch.Tensor

    def subset(self, idx: np.ndarray) -> "TensorWindows":
        if idx.size == 0:
            empty_imu = self.imu[:0]
            empty_pressure = self.pressure[:0]
            empty_rr = self.rr[:0]
            return TensorWindows(empty_imu, empty_pressure, empty_rr)
        tidx = torch.as_tensor(idx, dtype=torch.long)
        return TensorWindows(self.imu[tidx], self.pressure[tidx], self.rr[tidx])


@dataclass
class SubjectEvalContext:
    x_source: np.ndarray
    y_source: np.ndarray
    k_source: Optional[np.ndarray]
    x_target: np.ndarray
    y_target: np.ndarray
    k_target: Optional[np.ndarray]
    x_cal: np.ndarray
    y_cal: np.ndarray
    k_cal: Optional[np.ndarray]
    x_eval: np.ndarray
    y_eval: np.ndarray
    k_eval: Optional[np.ndarray]
    cal_idx: np.ndarray
    rr_model_state: Dict[str, torch.Tensor]
    target_windows: TensorWindows
    eval_windows: TensorWindows
    source_anchor_windows: TensorWindows


def _state_dict_cpu_clone(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Clone a full model state to CPU so feature-TTA can be undone safely."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _restore_state_dict(model: nn.Module, state: Dict[str, torch.Tensor], device: str) -> None:
    model.load_state_dict(state, strict=True)
    model.to(device)


def _clone_module_state_cpu(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


@torch.no_grad()
def _warm_lazy_modules(model: nn.Module, loader, device: str) -> None:
    """Ensure lazy spec_proj/pressure heads exist before trainable-param selection."""
    model.eval()
    for batch in loader:
        imu, _pressure, _cond, _br, _extra = unpack_batch(batch, device)
        _ = model(imu.float())
        return
    raise RuntimeError("No batches available to warm model lazy modules.")


@torch.no_grad()
def _collect_tensor_windows(model: nn.Module, loader, device: str, max_windows: int = 0) -> TensorWindows:
    """Collect ordered IMU, pressure, and RR labels as CPU tensors."""
    model.eval()
    imus: List[torch.Tensor] = []
    pressures: List[torch.Tensor] = []
    rrs: List[torch.Tensor] = []
    n_seen = 0
    for batch in loader:
        imu, pressure, _cond, br, _extra = unpack_batch(batch, device)
        imu = imu.float()
        pressure = pressure.float()
        rr = rr_targets_from_batch(pressure, br).view(-1).float()
        if max_windows > 0:
            remaining = int(max_windows) - n_seen
            if remaining <= 0:
                break
            if imu.size(0) > remaining:
                imu = imu[:remaining]
                pressure = pressure[:remaining]
                rr = rr[:remaining]
        imus.append(imu.detach().cpu())
        pressures.append(pressure.detach().cpu())
        rrs.append(rr.detach().cpu())
        n_seen += int(imu.size(0))
        if max_windows > 0 and n_seen >= int(max_windows):
            break
    if not imus:
        raise RuntimeError("No windows available for tensor collection.")
    return TensorWindows(
        imu=torch.cat(imus, dim=0),
        pressure=torch.cat(pressures, dim=0),
        rr=torch.cat(rrs, dim=0),
    )


@torch.no_grad()
def _pooled_features_from_windows(model: nn.Module, windows: TensorWindows, device: str, batch_size: int = 256) -> np.ndarray:
    model.eval()
    feats: List[np.ndarray] = []
    for st in range(0, int(windows.imu.size(0)), int(batch_size)):
        imu = windows.imu[st : st + int(batch_size)].to(device, non_blocking=True).float()
        hidden = model.encode(imu)
        z = pooled_features(hidden)
        feats.append(z.detach().cpu().numpy().astype(np.float32))
    if not feats:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(feats, axis=0).astype(np.float32)


class TorchAffineCalibrator(nn.Module):
    """Trainable scalar affine layer initialized to identity."""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.a * y + self.b

    def apply_np(self, y: np.ndarray) -> np.ndarray:
        return float(self.a.detach().cpu()) * np.asarray(y).reshape(-1) + float(self.b.detach().cpu())


def _count_trainable(model: nn.Module) -> Dict[str, int]:
    counts = {
        "feature_trainable_params": 0,
        "feature_trainable_layernorm_params": 0,
        "feature_trainable_spec_proj_params": 0,
        "feature_trainable_last_block_params": 0,
        "feature_trainable_rr_head_params": 0,
        "feature_trainable_pressure_decoder_params": 0,
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = int(p.numel())
        counts["feature_trainable_params"] += n
        if "spec_proj" in name:
            counts["feature_trainable_spec_proj_params"] += n
        if "rr_head" in name:
            counts["feature_trainable_rr_head_params"] += n
        if "pressure_mag_head" in name or "dec_rnn" in name or "out_deconv" in name:
            counts["feature_trainable_pressure_decoder_params"] += n
        if name.startswith("encoder.layers."):
            # Count only the numerically last block below if possible.
            pass
    for module_name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            for p in module.parameters(recurse=False):
                if p.requires_grad:
                    counts["feature_trainable_layernorm_params"] += int(p.numel())
    enc = getattr(model, "encoder", None)
    layers = getattr(enc, "layers", None)
    if layers is not None and len(layers) > 0:
        last_ids = {id(p) for p in layers[-1].parameters() if p.requires_grad}
        counts["feature_trainable_last_block_params"] = sum(int(p.numel()) for p in layers[-1].parameters() if id(p) in last_ids)
    return counts


def _configure_feature_trainability(model: nn.Module, mode: str) -> Dict[str, int]:
    """Freeze everything, then unfreeze only the requested feature-space subset."""
    for p in model.parameters():
        p.requires_grad = False

    # LayerNorm-only is the safe ViT analogue of BN affine adaptation.
    # Restrict this to the IMU feature extractor, not RR/pressure heads.
    feature_norm_prefixes = ("encoder", "transformer")
    blocked_norm_tokens = ("rr_head", "pressure_mag_head", "pressure_encoder", "pressure_proj", "dec_rnn")
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.LayerNorm):
            continue
        if not module_name.startswith(feature_norm_prefixes):
            continue
        if any(tok in module_name for tok in blocked_norm_tokens):
            continue
        for p in module.parameters(recurse=False):
            p.requires_grad = True

    if mode in LASTBLOCK_FEATURE_MODES:
        enc = getattr(model, "encoder", None)
        layers = getattr(enc, "layers", None)
        if layers is not None and len(layers) > 0:
            for p in layers[-1].parameters():
                p.requires_grad = True
        spec_proj = getattr(model, "spec_proj", None)
        if spec_proj is not None:
            for p in spec_proj.parameters():
                p.requires_grad = True

    return _count_trainable(model)


def _feature_param_names(model: nn.Module) -> List[str]:
    return [name for name, p in model.named_parameters() if p.requires_grad]


def _rr_from_reconstructed_stft(pred_logmag: torch.Tensor, br_fs: float = 18.0) -> torch.Tensor:
    """Estimate RR from reconstructed pressure log-magnitude STFT by dominant frequency."""
    if pred_logmag.ndim != 3:
        raise ValueError(f"Expected pred_logmag (B,T,F), got {tuple(pred_logmag.shape)}")
    n_freq = int(pred_logmag.size(-1))
    n_fft = max(2, 2 * (n_freq - 1))
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / float(br_fs)).to(pred_logmag.device)
    # Respiratory band aligned with the rest of the codebase.
    mask = (freqs >= 0.05) & (freqs <= 0.75)
    if not bool(mask.any()):
        return pred_logmag.new_zeros(pred_logmag.size(0))
    spectrum = pred_logmag.float().mean(dim=1)  # (B,F)
    local_idx = spectrum[:, mask].argmax(dim=1)
    rr_hz = freqs[mask][local_idx]
    return rr_hz * 60.0


def _set_rr_model_frozen(rr_model: FaithfulRRRegressor) -> None:
    rr_model.eval()
    for p in rr_model.parameters():
        p.requires_grad = False


def _run_model_rr_probe(
    model: nn.Module,
    rr_model: FaithfulRRRegressor,
    imu: torch.Tensor,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return RR-probe prediction, pooled feature, STFT-derived RR, and aux RR head."""
    pred_logmag, rr_aux, hidden = model(imu.to(device, non_blocking=True).float())
    z = pooled_features(hidden)
    rr_probe, _z_after_adapter = rr_model(z)
    rr_stft = _rr_from_reconstructed_stft(pred_logmag)
    return rr_probe.view(-1), z, rr_stft.view(-1), rr_aux.view(-1)


@torch.no_grad()
def _predict_feature_adapted(
    model: nn.Module,
    rr_model: FaithfulRRRegressor,
    affine: TorchAffineCalibrator,
    windows: TensorWindows,
    device: str,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    rr_model.eval()
    preds: List[np.ndarray] = []
    stft_rrs: List[np.ndarray] = []
    aux_rrs: List[np.ndarray] = []
    for st in range(0, int(windows.imu.size(0)), int(batch_size)):
        imu = windows.imu[st : st + int(batch_size)].to(device, non_blocking=True).float()
        rr_probe, _z, rr_stft, rr_aux = _run_model_rr_probe(model, rr_model, imu, device)
        rr_cal = affine(rr_probe)
        preds.append(rr_cal.detach().cpu().numpy().reshape(-1))
        stft_rrs.append(rr_stft.detach().cpu().numpy().reshape(-1))
        aux_rrs.append(rr_aux.detach().cpu().numpy().reshape(-1))
    return (
        np.concatenate(preds, axis=0),
        np.concatenate(stft_rrs, axis=0),
        np.concatenate(aux_rrs, axis=0),
    )


def _smoothness_l1(y: torch.Tensor) -> torch.Tensor:
    y = y.view(-1)
    if y.numel() < 2:
        return y.new_tensor(0.0)
    return (y[1:] - y[:-1]).abs().mean()


def adapt_feature_space_with_affine(
    model: nn.Module,
    rr_model: FaithfulRRRegressor,
    cal_windows: TensorWindows,
    source_anchor_windows: TensorWindows,
    mode: str,
    args,
    device: str,
) -> Tuple[TorchAffineCalibrator, Dict[str, float]]:
    """Perform calibration-prefix feature TTA. Leaves model in adapted state."""
    if cal_windows.imu.size(0) == 0:
        raise RuntimeError("Feature-adaptive modes require at least one target calibration window.")

    _set_rr_model_frozen(rr_model)
    train_counts = _configure_feature_trainability(model, mode)
    affine = TorchAffineCalibrator().to(device)

    model.eval()  # Disable dropout; gradients still flow through selected parameters.
    with torch.no_grad():
        cal_imu_full = cal_windows.imu.to(device, non_blocking=True).float()
        src_imu_full = source_anchor_windows.imu.to(device, non_blocking=True).float()
        _, z_cal_frozen, _stft_cal_frozen, _aux_cal_frozen = _run_model_rr_probe(model, rr_model, cal_imu_full, device)
        _, z_src_frozen, _stft_src_frozen, _aux_src_frozen = _run_model_rr_probe(model, rr_model, src_imu_full, device)
        z_cal_frozen = z_cal_frozen.detach()
        z_src_frozen = z_src_frozen.detach()

    params = [p for p in model.parameters() if p.requires_grad] + list(affine.parameters())
    if not params:
        raise RuntimeError(f"No trainable parameters selected for mode={mode}.")
    opt = torch.optim.AdamW(params, lr=float(args.feature_tta_lr), weight_decay=float(args.feature_tta_weight_decay))

    cal_y = cal_windows.rr.to(device, non_blocking=True).float().view(-1)
    src_imu = source_anchor_windows.imu.to(device, non_blocking=True).float()
    feature_batch_size = max(1, int(args.feature_tta_batch_size))
    n_cal = int(cal_windows.imu.size(0))
    rng = np.random.default_rng(int(args.seed))

    # Mode defaults: LayerNorm+STFT mode should turn on consistency even if the
    # caller forgets to set a nonzero weight; lastblock keeps caller control.
    stft_w = float(args.feature_stft_consistency_weight)
    if mode == "ln_stft_rr_consistency" and stft_w <= 0.0:
        stft_w = 0.05

    last_log: Dict[str, float] = {}
    for epoch in range(1, int(args.feature_tta_epochs) + 1):
        model.eval()
        order = rng.permutation(n_cal)
        epoch_losses: List[float] = []
        epoch_cal_losses: List[float] = []
        epoch_stft_losses: List[float] = []
        epoch_drift_losses: List[float] = []
        epoch_smooth_losses: List[float] = []
        for st in range(0, n_cal, feature_batch_size):
            idx_np = order[st : st + feature_batch_size]
            # cal_windows are intentionally stored on CPU to avoid keeping the full
            # subject in GPU memory. PyTorch requires tensor indices to live on the
            # same device as the indexed tensor, so use a CPU index for CPU windows
            # and a device index for tensors already moved to the GPU.
            idx_cpu = torch.as_tensor(idx_np, dtype=torch.long)
            idx_dev = idx_cpu.to(device)
            imu = cal_windows.imu[idx_cpu].to(device, non_blocking=True).float()
            y = cal_y[idx_dev]
            z_ref = z_cal_frozen[idx_dev]

            opt.zero_grad(set_to_none=True)
            rr_probe, z, rr_stft, _rr_aux = _run_model_rr_probe(model, rr_model, imu, device)
            rr_cal = affine(rr_probe)
            cal_loss = F.smooth_l1_loss(rr_cal, y)
            affine_reg = (
                float(args.feature_affine_lambda_a) * (affine.a - 1.0).pow(2)
                + float(args.feature_affine_lambda_b) * affine.b.pow(2)
            )
            target_drift = F.mse_loss(z, z_ref)
            stft_loss = F.smooth_l1_loss(rr_probe, rr_stft.detach())
            loss = cal_loss + affine_reg
            loss = loss + float(args.feature_target_drift_weight) * target_drift
            loss = loss + stft_w * stft_loss

            # Source-feature anchor: preserve source geometry after feature updates.
            if float(args.feature_source_anchor_weight) > 0.0 and src_imu.size(0) > 0:
                n_src = src_imu.size(0)
                src_take = min(n_src, max(1, int(args.feature_source_anchor_batch_size)))
                src_idx = torch.randint(0, n_src, (src_take,), device=device)
                _src_probe, z_src, _src_stft, _src_aux = _run_model_rr_probe(model, rr_model, src_imu[src_idx], device)
                source_drift = F.mse_loss(z_src, z_src_frozen[src_idx])
                loss = loss + float(args.feature_source_anchor_weight) * source_drift
            else:
                source_drift = target_drift.new_tensor(0.0)

            loss.backward()
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip))
            opt.step()

            epoch_losses.append(float(loss.detach().cpu()))
            epoch_cal_losses.append(float(cal_loss.detach().cpu()))
            epoch_stft_losses.append(float(stft_loss.detach().cpu()))
            epoch_drift_losses.append(float((target_drift + source_drift).detach().cpu()))

        # Smoothness is evaluated on the ordered calibration prefix once per epoch.
        # For lastblock mode it is also optimized in an extra full-prefix pass so the
        # temporal adjacency is meaningful.
        if float(args.feature_smoothness_weight) > 0.0:
            opt.zero_grad(set_to_none=True)
            rr_probe_full, _z_full, _rr_stft_full, _rr_aux_full = _run_model_rr_probe(model, rr_model, cal_imu_full, device)
            rr_cal_full = affine(rr_probe_full)
            smooth = _smoothness_l1(rr_cal_full)
            smooth_loss = float(args.feature_smoothness_weight) * smooth
            smooth_loss.backward()
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip))
            opt.step()
            epoch_smooth_losses.append(float(smooth.detach().cpu()))
        else:
            epoch_smooth_losses.append(0.0)

        last_log = {
            "feature_tta_epoch": int(epoch),
            "feature_loss_last": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
            "feature_cal_loss_last": float(np.mean(epoch_cal_losses)) if epoch_cal_losses else float("nan"),
            "feature_stft_consistency_last": float(np.mean(epoch_stft_losses)) if epoch_stft_losses else float("nan"),
            "feature_drift_loss_last": float(np.mean(epoch_drift_losses)) if epoch_drift_losses else float("nan"),
            "feature_smoothness_last": float(np.mean(epoch_smooth_losses)) if epoch_smooth_losses else 0.0,
        }

    with torch.no_grad():
        rr_cal_pred, rr_stft_cal, rr_aux_cal = _predict_feature_adapted(
            model, rr_model, affine, cal_windows, device, int(args.feature_eval_batch_size)
        )
    extra = {
        **train_counts,
        **last_log,
        "feature_tta_mode": mode,
        "feature_tta_epochs": int(args.feature_tta_epochs),
        "feature_tta_lr": float(args.feature_tta_lr),
        "feature_affine_a": float(affine.a.detach().cpu()),
        "feature_affine_b": float(affine.b.detach().cpu()),
        "feature_affine_lambda_a": float(args.feature_affine_lambda_a),
        "feature_affine_lambda_b": float(args.feature_affine_lambda_b),
        "feature_stft_consistency_weight": float(stft_w),
        "feature_source_anchor_weight": float(args.feature_source_anchor_weight),
        "feature_target_drift_weight": float(args.feature_target_drift_weight),
        "feature_smoothness_weight": float(args.feature_smoothness_weight),
        "feature_cal_mae": float(np.mean(np.abs(rr_cal_pred - cal_windows.rr.numpy().reshape(-1)))),
        "feature_cal_rmse": float(np.sqrt(np.mean((rr_cal_pred - cal_windows.rr.numpy().reshape(-1)) ** 2))),
        "feature_cal_stft_rr_mae_vs_label": float(np.mean(np.abs(rr_stft_cal - cal_windows.rr.numpy().reshape(-1)))),
        "feature_cal_aux_rr_mae_vs_label": float(np.mean(np.abs(rr_aux_cal - cal_windows.rr.numpy().reshape(-1)))),
        "feature_trainable_names_json": json.dumps(_feature_param_names(model)[:200]),
    }
    return affine, extra


def _make_eval_indices(n: int, cal_idx: np.ndarray, exclude_calibration_from_eval: bool) -> np.ndarray:
    if exclude_calibration_from_eval and cal_idx.size > 0:
        mask = np.ones(int(n), dtype=bool)
        mask[cal_idx] = False
        return np.where(mask)[0]
    return np.arange(int(n), dtype=np.int64)


def _rr_from_reconstructed_stft_with_confidence(
    pred_logmag: torch.Tensor,
    br_fs: float = 18.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate RR plus peak-confidence from reconstructed pressure log-STFT.

    Confidence is a simple peak-separation score in the respiratory band:
      (top1 - top2) / total_band_energy, clamped to [0, 1].
    This is intentionally lightweight so it can be used inside TTA loops.
    """
    if pred_logmag.ndim != 3:
        raise ValueError(f"Expected pred_logmag (B,T,F), got {tuple(pred_logmag.shape)}")
    n_freq = int(pred_logmag.size(-1))
    n_fft = max(2, 2 * (n_freq - 1))
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / float(br_fs)).to(pred_logmag.device)
    mask = (freqs >= 0.05) & (freqs <= 0.75)
    if not bool(mask.any()):
        return pred_logmag.new_zeros(pred_logmag.size(0)), pred_logmag.new_zeros(pred_logmag.size(0))

    spectrum = pred_logmag.float().mean(dim=1)  # (B,F)
    band = spectrum[:, mask].clamp_min(0.0)
    freqs_band = freqs[mask]
    local_idx = band.argmax(dim=1)
    rr_hz = freqs_band[local_idx]

    if band.size(1) >= 2:
        top2 = torch.topk(band, k=2, dim=1).values
        numerator = (top2[:, 0] - top2[:, 1]).clamp_min(0.0)
    else:
        numerator = band[:, 0].clamp_min(0.0)
    denom = band.sum(dim=1).clamp_min(1e-8)
    confidence = (numerator / denom).clamp(0.0, 1.0)
    return rr_hz * 60.0, confidence


def _run_model_rr_probe_with_confidence(
    model: nn.Module,
    rr_model: FaithfulRRRegressor,
    imu: torch.Tensor,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_logmag, rr_aux, hidden = model(imu.to(device, non_blocking=True).float())
    z = pooled_features(hidden)
    rr_probe, _z_after_adapter = rr_model(z)
    rr_stft, stft_conf = _rr_from_reconstructed_stft_with_confidence(pred_logmag)
    return rr_probe.view(-1), z, rr_stft.view(-1), rr_aux.view(-1), stft_conf.view(-1)


@torch.no_grad()
def _predict_feature_unsup(
    model: nn.Module,
    rr_model: FaithfulRRRegressor,
    windows: TensorWindows,
    device: str,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    rr_model.eval()
    preds: List[np.ndarray] = []
    stft_rrs: List[np.ndarray] = []
    aux_rrs: List[np.ndarray] = []
    confs: List[np.ndarray] = []
    for st in range(0, int(windows.imu.size(0)), int(batch_size)):
        imu = windows.imu[st : st + int(batch_size)].to(device, non_blocking=True).float()
        rr_probe, _z, rr_stft, rr_aux, conf = _run_model_rr_probe_with_confidence(model, rr_model, imu, device)
        preds.append(rr_probe.detach().cpu().numpy().reshape(-1))
        stft_rrs.append(rr_stft.detach().cpu().numpy().reshape(-1))
        aux_rrs.append(rr_aux.detach().cpu().numpy().reshape(-1))
        confs.append(conf.detach().cpu().numpy().reshape(-1))
    return (
        np.concatenate(preds, axis=0),
        np.concatenate(stft_rrs, axis=0),
        np.concatenate(aux_rrs, axis=0),
        np.concatenate(confs, axis=0),
    )


def _weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    per = F.smooth_l1_loss(pred.view(-1), target.view(-1), reduction="none")
    w = weight.view(-1).detach().clamp_min(0.0)
    return (per * w).sum() / w.sum().clamp_min(1e-8)


def _profile_normalizer_for_model(
    model: nn.Module,
    device: str,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    mean = getattr(model, "source_profile_mean", None)
    std = getattr(model, "source_profile_std", None)
    if mean is None or std is None:
        return None, None
    return mean.to(device), std.to(device)


def _normalize_profile_stats_for_model(
    model: nn.Module,
    profile_stats: torch.Tensor,
    device: str,
) -> torch.Tensor:
    source_mean, source_std = _profile_normalizer_for_model(model, device)
    if source_mean is None or source_std is None:
        return profile_stats
    return normalize_profile_stats(profile_stats, source_mean, source_std)


def _profile_stats_from_loader(
    model: nn.Module,
    loader,
    device: str,
    args,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_batches = int(getattr(args, "profile_stats_max_batches", 50))
    stats_raw = collect_profile_stats(
        model,
        loader,
        device,
        max_batches=None if max_batches <= 0 else max_batches,
    )
    stats_raw = stats_raw.to(device)
    stats_norm = _normalize_profile_stats_for_model(model, stats_raw, device)
    return stats_raw, stats_norm

@torch.no_grad()
def _profile_stats_from_windows(
    model: nn.Module,
    windows: TensorWindows,
    device: str,
    args,
    batch_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build raw/normalized profile stats from a selected unlabeled window set.

    This lets the profile ladder test:
      - full transductive subject profile
      - calibration-prefix profile
      - random/even representative calibration profile

    No target RR labels are used; RR values in TensorWindows are ignored here.
    """
    model.eval()
    z_list: List[torch.Tensor] = []
    aux_list: List[torch.Tensor] = []
    stft_list: List[torch.Tensor] = []
    conf_list: List[torch.Tensor] = []

    n = int(windows.imu.size(0))
    if n <= 0:
        return _profile_stats_from_loader(model, [], device, args)

    for st in range(0, n, int(batch_size)):
        imu = windows.imu[st : st + int(batch_size)].to(device, non_blocking=True).float()
        pred_logmag, rr_aux, hidden = model(imu)
        z = pooled_features(hidden)
        rr_stft, stft_conf = _rr_from_reconstructed_stft_with_confidence(pred_logmag)

        keep = (
            torch.isfinite(z).all(dim=1)
            & torch.isfinite(rr_aux.reshape(-1))
            & torch.isfinite(rr_stft.reshape(-1))
            & torch.isfinite(stft_conf.reshape(-1))
        )
        if not bool(keep.any()):
            continue

        z_list.append(z.detach()[keep])
        aux_list.append(rr_aux.detach().reshape(-1)[keep])
        stft_list.append(rr_stft.detach().reshape(-1)[keep])
        conf_list.append(stft_conf.detach().reshape(-1)[keep])

    if not z_list:
        stats_raw = torch.zeros(
            infer_profile_stats_dim(int(getattr(model, "d_model", 0))),
            device=device,
            dtype=torch.float32,
        )
    else:
        stats_raw = torch.cat(
            [
                torch.cat(z_list, dim=0).mean(dim=0),
                torch.cat(z_list, dim=0).std(dim=0, unbiased=False),
                torch.cat(aux_list, dim=0).mean().view(1),
                torch.cat(aux_list, dim=0).std(unbiased=False).view(1),
                torch.cat(stft_list, dim=0).mean().view(1),
                torch.cat(stft_list, dim=0).std(unbiased=False).view(1),
                (torch.cat(aux_list, dim=0) - torch.cat(stft_list, dim=0)).mean().view(1),
                (torch.cat(aux_list, dim=0) - torch.cat(stft_list, dim=0)).std(unbiased=False).view(1),
                torch.cat(conf_list, dim=0).mean().view(1),
                torch.cat(conf_list, dim=0).std(unbiased=False).view(1),
            ],
            dim=0,
        ).to(device)

    stats_norm = _normalize_profile_stats_for_model(model, stats_raw, device)
    return stats_raw, stats_norm


# This switch tests the calibration-window finding from the subject-shift study:
# first-window calibration can be biased, while random/even windows are more
# representative. Use --profile-unsup-adapt-scope calibration with
# --target-calibration-mode random/even to test that directly.
def _select_profile_adaptation_windows(
    target_windows: TensorWindows,
    eval_windows: TensorWindows,
    cal_idx: np.ndarray,
    args,
) -> Tuple[TensorWindows, str]:
    """Choose unlabeled windows used for profile/readout adaptation."""
    scope = str(getattr(args, "profile_unsup_adapt_scope", "full")).lower()
    if scope == "calibration":
        if cal_idx.size <= 0:
            return target_windows, "full_fallback_no_calibration"
        return target_windows.subset(cal_idx), "calibration"
    if scope == "eval":
        return eval_windows, "eval"
    return target_windows, "full"

def _profile_stats_scalar_diagnostics(
    model: nn.Module,
    profile_stats_raw: torch.Tensor,
    profile_stats_norm: torch.Tensor,
    profile_vector: torch.Tensor,
    args,
) -> Dict[str, float]:
    profile_dim = int(getattr(args, "profile_dim", getattr(model, "profile_dim", 0)))
    norm_diag = profile_stats_diagnostics(profile_stats_norm.detach().cpu(), profile_dim=profile_dim)
    out = {
        "profile_stats_dim": float(norm_diag.get("profile_stats_dim", 0)),
        "profile_dim": float(norm_diag.get("profile_dim", 0)),
        "profile_stats_norm_l2": float(norm_diag.get("profile_norm", 0.0)),
        "profile_stats_mean_abs": float(norm_diag.get("profile_mean_abs", 0.0)),
        "profile_stats_std": float(norm_diag.get("profile_std", 0.0)),
        "profile_stats_source_prior_l2": float(norm_diag.get("profile_source_prior_l2", 0.0)),
    }
    out.update({
        "profile_vector_norm": float(profile_vector.detach().float().norm(p=2).cpu()),
        "profile_vector_mean_abs": float(profile_vector.detach().float().abs().mean().cpu()),
        "profile_vector_std": float(profile_vector.detach().float().std(unbiased=False).cpu()),
        "profile_vector_prior_l2": float((profile_vector.detach().float() ** 2).mean().cpu()),
        "profile_norm": float(profile_vector.detach().float().norm(p=2).cpu()),
        "profile_mean_abs": float(profile_vector.detach().float().abs().mean().cpu()),
        "profile_std": float(profile_vector.detach().float().std(unbiased=False).cpu()),
        "profile_source_prior_l2": float((profile_vector.detach().float() ** 2).mean().cpu()),
    })
    latent_dim = int(getattr(model, "d_model", 0))
    try:
        split_raw = split_profile_stats(profile_stats_raw.detach().cpu().reshape(-1), latent_dim)
        split_norm = split_profile_stats(profile_stats_norm.detach().cpu().reshape(-1), latent_dim)
        out.update({
            "raw_profile_stats_rr_aux_mean_bpm": float(split_raw["rr_aux_mean"].item()),
            "raw_profile_stats_rr_stft_mean_bpm": float(split_raw["rr_stft_mean"].item()),
            "raw_profile_stats_rr_delta_mean_bpm": float(split_raw["rr_delta_mean"].item()),
            "raw_profile_stats_stft_confidence_mean": float(split_raw["stft_confidence_mean"].item()),
            "norm_profile_stats_rr_aux_mean": float(split_norm["rr_aux_mean"].item()),
            "norm_profile_stats_rr_stft_mean": float(split_norm["rr_stft_mean"].item()),
            "norm_profile_stats_rr_delta_mean": float(split_norm["rr_delta_mean"].item()),
            "norm_profile_stats_stft_confidence_mean": float(split_norm["stft_confidence_mean"].item()),
        })
    except Exception:
        pass
    return out


def _get_profile_attention_metrics(model: nn.Module) -> Dict[str, float]:
    if not hasattr(model, "last_profile_attention_metrics"):
        return {}
    try:
        metrics = model.last_profile_attention_metrics()
    except Exception:
        return {}
    return {str(k): float(v) for k, v in metrics.items()}


def _profile_clsa_config_diagnostics(model: nn.Module, args) -> Dict[str, float]:
    mode = str(getattr(model, "profile_qkv_mode", getattr(args, "profile_qkv_mode", "static")))
    return {
        "profile_clsa_enabled": int(mode == "clsa" and bool(getattr(model, "use_profile_qkv", False))),
        "profile_qkv_mode": mode,
        "profile_clsa_rank": int(getattr(model, "profile_clsa_rank", getattr(args, "profile_clsa_rank", 8))),
        "profile_clsa_scale": float(getattr(model, "profile_clsa_scale", getattr(args, "profile_clsa_scale", 0.01))),
        "profile_clsa_enable_fast_update": int(bool(getattr(model, "profile_clsa_enable_fast_update", getattr(args, "profile_clsa_enable_fast_update", 1)))),
    }


def _profile_clsa_metric_diagnostics(metric_rows: List[Dict[str, float]]) -> Dict[str, float]:
    merged = _merge_attention_metric_lists(metric_rows)
    out: Dict[str, float] = {}
    suffix = "_layer_"
    for key, value in merged.items():
        if key.startswith("profile_clsa_") and suffix in key:
            out[key.split(suffix, 1)[0]] = float(value)
    if "profile_clsa_loss" in out:
        out["profile_clsa_loss_mean"] = float(out["profile_clsa_loss"])
    return out


def _merge_attention_metric_lists(metric_lists: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_lists:
        return {}
    keys = sorted({k for row in metric_lists for k in row.keys()})
    out: Dict[str, float] = {}
    for key in keys:
        vals = [float(row[key]) for row in metric_lists if key in row]
        if vals:
            out[key] = float(np.mean(vals))
    return out


def _profile_conditioning_mode_for_rr_mode(mode: str) -> str:
    text = str(mode).lower().strip()
    if "film_qkv" in text or "profile_film_clsa_qkv" in text:
        return "film_qkv"
    if "profile_qkv_" in text or "clsa_qkv" in text:
        return "qkv"
    return "film"


@torch.no_grad()
def _collect_profile_attention_reference(
    model: nn.Module,
    windows: TensorWindows,
    profile_vector: torch.Tensor,
    device: str,
    batch_size: int,
    conditioning_mode: str,
) -> Dict[str, float]:
    model.eval()
    rows: List[Dict[str, float]] = []
    for st in range(0, int(windows.imu.size(0)), int(batch_size)):
        imu = windows.imu[st : st + int(batch_size)].to(device, non_blocking=True).float()
        p = profile_vector.expand(imu.size(0), -1)
        model.forward_profile_conditioned(imu, profile_vector=p, conditioning_mode=conditioning_mode)
        metrics = _get_profile_attention_metrics(model)
        if metrics:
            rows.append(metrics)
    return _merge_attention_metric_lists(rows)


def _profile_conditioned_rr_probe(
    model: nn.Module,
    rr_model: FaithfulRRRegressor,
    imu: torch.Tensor,
    profile_vector: torch.Tensor,
    device: str,
    conditioning_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    p = profile_vector.expand(imu.size(0), -1)
    pred_logmag, rr_aux, hidden, _p = model.forward_profile_conditioned(
        imu.to(device, non_blocking=True).float(),
        profile_vector=p,
        conditioning_mode=conditioning_mode,
    )
    z = pooled_features(hidden)
    rr_probe, _z_after_adapter = rr_model(z)
    rr_stft, stft_conf = _rr_from_reconstructed_stft_with_confidence(pred_logmag)
    attn_metrics = _get_profile_attention_metrics(model)
    return rr_probe.view(-1), z, rr_stft.view(-1), rr_aux.view(-1), stft_conf.view(-1), attn_metrics


def _build_subject_eval_context(
    model: nn.Module,
    train_loader,
    test_loader,
    device: str,
    args,
    *,
    include_kinematics: bool,
) -> SubjectEvalContext:
    x_source, y_source, k_source = collect_rr_arrays(
        model,
        train_loader,
        device,
        max_batches=int(args.rr_probe_source_batches),
        include_kinematics=include_kinematics,
    )
    x_target, y_target, k_target = collect_rr_arrays(
        model,
        test_loader,
        device,
        max_batches=0,
        include_kinematics=include_kinematics,
    )
    x_cal, y_cal, k_cal, x_eval, y_eval, k_eval, cal_idx = split_target_calibration_eval(
        x_target,
        y_target,
        k_target,
        int(args.target_calibration_windows),
        seed=int(args.seed),
        mode=str(args.target_calibration_mode),
        exclude_calibration_from_eval=bool(args.exclude_calibration_from_eval),
    )

    rr_model = FaithfulRRRegressor(x_source.shape[1]).to(device)
    rr_model = train_source_rr_regressor(
        rr_model,
        x_source,
        y_source,
        TrainConfig(
            epochs=int(args.rr_probe_epochs),
            lr=float(args.rr_probe_lr),
            weight_decay=float(args.rr_probe_weight_decay),
            batch_size=int(args.rr_probe_batch_size),
            grad_clip=float(args.grad_clip),
        ),
        device,
        train_adapter=bool(args.rr_probe_train_adapter),
    )
    _set_rr_model_frozen(rr_model)

    target_windows = _collect_tensor_windows(model, test_loader, device, max_windows=0)
    eval_idx = _make_eval_indices(target_windows.imu.size(0), cal_idx, bool(args.exclude_calibration_from_eval))
    eval_windows = target_windows.subset(eval_idx)
    source_anchor_windows = _collect_tensor_windows(
        model,
        train_loader,
        device,
        max_windows=int(args.feature_source_anchor_windows),
    )

    return SubjectEvalContext(
        x_source=x_source,
        y_source=y_source,
        k_source=k_source,
        x_target=x_target,
        y_target=y_target,
        k_target=k_target,
        x_cal=x_cal,
        y_cal=y_cal,
        k_cal=k_cal,
        x_eval=x_eval,
        y_eval=y_eval,
        k_eval=k_eval,
        cal_idx=cal_idx,
        rr_model_state=_clone_module_state_cpu(rr_model),
        target_windows=target_windows,
        eval_windows=eval_windows,
        source_anchor_windows=source_anchor_windows,
    )


@torch.no_grad()
def _predict_profile_unsup(
    model: nn.Module,
    rr_model: FaithfulRRRegressor,
    windows: TensorWindows,
    profile_vector: torch.Tensor,
    device: str,
    batch_size: int,
    conditioning_mode: str,
    readout_affine: Optional[TorchAffineCalibrator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    rr_model.eval()
    preds: List[np.ndarray] = []
    stft_rrs: List[np.ndarray] = []
    aux_rrs: List[np.ndarray] = []
    confs: List[np.ndarray] = []
    for st in range(0, int(windows.imu.size(0)), int(batch_size)):
        imu = windows.imu[st : st + int(batch_size)].to(device, non_blocking=True).float()
        _rr_probe, _z, rr_stft, rr_aux, conf, _attn = _profile_conditioned_rr_probe(
            model, rr_model, imu, profile_vector, device, conditioning_mode
        )
        rr_final = _rr_probe if readout_affine is None else readout_affine(_rr_probe)
        preds.append(rr_final.detach().cpu().numpy().reshape(-1))
        stft_rrs.append(rr_stft.detach().cpu().numpy().reshape(-1))
        aux_rrs.append(rr_aux.detach().cpu().numpy().reshape(-1))
        confs.append(conf.detach().cpu().numpy().reshape(-1))
    return (
        np.concatenate(preds, axis=0),
        np.concatenate(stft_rrs, axis=0),
        np.concatenate(aux_rrs, axis=0),
        np.concatenate(confs, axis=0),
    )


@torch.no_grad()
def _predict_profile_unsup_with_features(
    model: nn.Module,
    rr_model: FaithfulRRRegressor,
    windows: TensorWindows,
    profile_vector: torch.Tensor,
    device: str,
    batch_size: int,
    conditioning_mode: str,
    readout_affine: Optional[TorchAffineCalibrator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Profile-conditioned RR predictions plus pooled feature vectors."""
    model.eval()
    rr_model.eval()
    preds: List[np.ndarray] = []
    feats: List[np.ndarray] = []
    stft_rrs: List[np.ndarray] = []
    aux_rrs: List[np.ndarray] = []
    confs: List[np.ndarray] = []
    for st in range(0, int(windows.imu.size(0)), int(batch_size)):
        imu = windows.imu[st : st + int(batch_size)].to(device, non_blocking=True).float()
        rr_probe, z, rr_stft, rr_aux, conf, _attn = _profile_conditioned_rr_probe(
            model, rr_model, imu, profile_vector, device, conditioning_mode
        )
        rr_final = rr_probe if readout_affine is None else readout_affine(rr_probe)
        preds.append(rr_final.detach().cpu().numpy().reshape(-1))
        feats.append(z.detach().cpu().numpy().astype(np.float32))
        stft_rrs.append(rr_stft.detach().cpu().numpy().reshape(-1))
        aux_rrs.append(rr_aux.detach().cpu().numpy().reshape(-1))
        confs.append(conf.detach().cpu().numpy().reshape(-1))
    return (
        np.concatenate(preds, axis=0),
        np.concatenate(feats, axis=0).astype(np.float32),
        np.concatenate(stft_rrs, axis=0),
        np.concatenate(aux_rrs, axis=0),
        np.concatenate(confs, axis=0),
    )


def _attach_base_pre_aliases(metrics: Dict[str, float]) -> None:
    for key in ("mae", "rmse", "corr", "n"):
        src = f"rr_probe_pre_{key}"
        if src in metrics:
            metrics[f"rr_probe_base_pre_{key}"] = metrics[src]


def _mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    n = min(y_true.shape[0], y_pred.shape[0])
    if n <= 0:
        return float("nan")
    return float(np.mean(np.abs(y_pred[:n] - y_true[:n])))


def _bool_float(x: bool) -> float:
    return 1.0 if bool(x) else 0.0


def _profile_mode_init_only(mode: str) -> bool:
    mode = str(mode).lower()
    return mode in {
        "profile_film_init_only",
        "profile_film_gated_init_only",
        "profile_film_oracle_init_only",
        "tcn_profile_film_qkv_last1_0p01",
    }


def _profile_mode_gated(mode: str) -> bool:
    return str(mode).lower() in {"profile_film_gated_init_only", "profile_film_gated_sparc"}


def _profile_mode_oracle(mode: str) -> bool:
    return str(mode).lower() in {"profile_film_oracle_init_only", "profile_film_oracle_sparc"}


def _profile_stats_z_diagnostics(profile_stats_norm: torch.Tensor) -> Dict[str, float]:
    z = profile_stats_norm.detach().float().reshape(-1).cpu().numpy()
    if z.size == 0:
        return {
            "profile_stats_z_rms": float("nan"),
            "profile_stats_z_max_abs": float("nan"),
        }
    return {
        "profile_stats_z_rms": float(np.sqrt(np.mean(z * z))),
        "profile_stats_z_max_abs": float(np.max(np.abs(z))),
    }


def _profile_gate_diagnostics(
    args,
    profile_stats_norm: torch.Tensor,
    *,
    base_probe: np.ndarray,
    profile_init_probe: np.ndarray,
    aux_rr: np.ndarray,
    stft_rr: np.ndarray,
    stft_conf: np.ndarray,
) -> Dict[str, float]:
    base_probe = np.asarray(base_probe, dtype=np.float32).reshape(-1)
    profile_init_probe = np.asarray(profile_init_probe, dtype=np.float32).reshape(-1)
    aux_rr = np.asarray(aux_rr, dtype=np.float32).reshape(-1)
    stft_rr = np.asarray(stft_rr, dtype=np.float32).reshape(-1)
    stft_conf = np.asarray(stft_conf, dtype=np.float32).reshape(-1)
    n = min(
        base_probe.shape[0],
        profile_init_probe.shape[0],
        aux_rr.shape[0],
        stft_rr.shape[0],
        stft_conf.shape[0],
    )
    if n <= 0:
        n = 0
        shift = np.asarray([], dtype=np.float32)
        disagreement = np.asarray([], dtype=np.float32)
        conf = np.asarray([], dtype=np.float32)
    else:
        shift = np.abs(profile_init_probe[:n] - base_probe[:n])
        disagreement = np.abs(aux_rr[:n] - stft_rr[:n])
        conf = stft_conf[:n]

    stats_diag = _profile_stats_z_diagnostics(profile_stats_norm)
    max_shift = float(getattr(args, "profile_gate_max_init_shift_bpm", 0.75))
    max_disagree = float(getattr(args, "profile_gate_aux_stft_disagreement_bpm", 3.0))
    min_conf = float(getattr(args, "profile_gate_min_stft_confidence", 0.005))
    max_rms_z = float(getattr(args, "profile_gate_profile_rms_z_max", 3.0))
    max_abs_z = float(getattr(args, "profile_gate_profile_max_abs_z", 8.0))

    shift_mean = float(np.mean(shift)) if n > 0 else float("nan")
    shift_p95 = float(np.percentile(shift, 95)) if n > 0 else float("nan")
    disagreement_mean = float(np.mean(disagreement)) if n > 0 else float("nan")
    disagreement_p95 = float(np.percentile(disagreement, 95)) if n > 0 else float("nan")
    conf_mean = float(np.mean(conf)) if n > 0 else float("nan")
    conf_frac = float(np.mean(conf >= min_conf)) if n > 0 else 0.0

    shift_ok = bool(np.isfinite(shift_mean) and shift_mean <= max_shift)
    disagree_ok = bool(np.isfinite(disagreement_mean) and disagreement_mean <= max_disagree)
    # This intentionally uses the raw confidence, not the clamped/floored value
    # used in the unsupervised loss.
    conf_ok = bool(np.isfinite(conf_mean) and conf_mean >= min_conf)
    stats_ok = bool(
        np.isfinite(stats_diag["profile_stats_z_rms"])
        and np.isfinite(stats_diag["profile_stats_z_max_abs"])
        and stats_diag["profile_stats_z_rms"] <= max_rms_z
        and stats_diag["profile_stats_z_max_abs"] <= max_abs_z
    )
    gate_ok = bool(shift_ok and disagree_ok and conf_ok and stats_ok)

    return {
        **stats_diag,
        "profile_gate_n_windows": int(n),
        "profile_gate_init_shift_mean_bpm": shift_mean,
        "profile_gate_init_shift_p95_bpm": shift_p95,
        "profile_gate_max_init_shift_bpm": max_shift,
        "profile_gate_aux_stft_disagreement_mean_bpm": disagreement_mean,
        "profile_gate_aux_stft_disagreement_p95_bpm": disagreement_p95,
        "profile_gate_aux_stft_disagreement_max_bpm": max_disagree,
        "profile_gate_stft_confidence_mean": conf_mean,
        "profile_gate_stft_confidence_frac_above_threshold": conf_frac,
        "profile_gate_min_stft_confidence": min_conf,
        "profile_gate_profile_rms_z_max": max_rms_z,
        "profile_gate_profile_max_abs_z": max_abs_z,
        "profile_gate_shift_ok": _bool_float(shift_ok),
        "profile_gate_aux_stft_ok": _bool_float(disagree_ok),
        "profile_gate_confidence_ok": _bool_float(conf_ok),
        "profile_gate_stats_ok": _bool_float(stats_ok),
        "profile_gate_pass": _bool_float(gate_ok),
    }


def _profile_oracle_diagnostics(
    args,
    *,
    y_cal: np.ndarray,
    base_probe_cal: np.ndarray,
    profile_init_cal: np.ndarray,
) -> Dict[str, float]:
    tol = float(getattr(args, "profile_oracle_cal_tolerance_bpm", 0.05))
    base_mae = _mae_np(y_cal, base_probe_cal)
    init_mae = _mae_np(y_cal, profile_init_cal)
    passed = bool(np.isfinite(base_mae) and np.isfinite(init_mae) and init_mae <= base_mae - tol)
    return {
        "profile_oracle_base_cal_mae": float(base_mae),
        "profile_oracle_init_cal_mae": float(init_mae),
        "profile_oracle_delta_init_minus_base_mae": float(init_mae - base_mae) if np.isfinite(base_mae) and np.isfinite(init_mae) else float("nan"),
        "profile_oracle_cal_tolerance_bpm": float(tol),
        "profile_oracle_pass": _bool_float(passed),
        "profile_oracle_uses_target_labels": 1.0,
    }


def _profile_init_extra(
    model: nn.Module,
    target_profile_stats_raw: torch.Tensor,
    target_profile_stats_norm: torch.Tensor,
    p0: torch.Tensor,
    args,
    mode: str,
) -> Dict[str, float]:
    diag = _profile_stats_scalar_diagnostics(
        model,
        target_profile_stats_raw,
        target_profile_stats_norm,
        p0,
        args,
    )
    adapter_diag = _profile_adapter_delta_diagnostics(model, p0)
    return {
        **diag,
        **adapter_diag,
        **_profile_clsa_config_diagnostics(model, args),
        "feature_tta_mode": str(mode),
        "unsup_feature_tta_epochs": 0,
        "unsup_feature_tta_lr": 0.0,
        "unsup_stft_consistency_weight": 0.0,
        "uses_stft_pseudotarget_for_tta": 0,
        "unsup_smoothness_weight": 0.0,
        "unsup_source_anchor_weight": 0.0,
        "unsup_target_drift_weight": 0.0,
        "unsup_profile_prior_weight": 0.0,
        "unsup_attention_profile_weight": 0.0,
        "unsup_readout_affine_enabled": 0,
        "unsup_aux_consistency_weight": 0.0,
        "unsup_rr_range_weight": 0.0,
        "readout_affine_a": 1.0,
        "readout_affine_b": 0.0,
        "profile_delta_norm": 0.0,
        "profile_film_scale": float(getattr(model, "profile_film_scale", getattr(args, "profile_film_scale", 0.1))),
        "profile_film_placement": str(getattr(model, "profile_film_placement", getattr(args, "profile_film_placement", "token_pooled"))),
        "profile_film_residual_alpha": float(getattr(model, "profile_film_residual_alpha", getattr(args, "profile_film_residual_alpha", 0.1))),
        "profile_qkv_scale": float(getattr(model, "profile_qkv_scale", getattr(args, "profile_qkv_scale", 0.1))),
        "profile_qkv_layers_used": str(getattr(model, "profile_qkv_layers", getattr(args, "profile_qkv_layers", "last1"))),
        "profile_qkv_residual": int(bool(getattr(model, "profile_qkv_residual", getattr(args, "profile_qkv_residual", False)))),
        "profile_conditioning_mode": str(getattr(model, "profile_conditioning", "none")),
        "profile_safety_budget_use_stft_confidence": int(bool(getattr(args, "profile_safety_budget_use_stft_confidence", True))),
        "profile_unsup_episodic_batch": int(bool(getattr(args, "profile_unsup_episodic_batch", False))),
        "profile_unsup_episodic_batches": 0,
        "profile_adapted_vector": 0,
        "profile_adapted_encoder": 0,
        "profile_adapted_film": 0,
        "profile_trainable_params": 0,
        "readout_affine_trainable_params": 0,
        "feature_trainable_params": 0,
        "feature_trainable_layernorm_params": 0,
        "feature_trainable_spec_proj_params": 0,
        "feature_trainable_last_block_params": 0,
        "feature_trainable_rr_head_params": 0,
        "feature_trainable_pressure_decoder_params": 0,
        "feature_trainable_names_json": json.dumps([]),
    }


def _qkv_delta_norm_tensor(model: nn.Module, profile_vector: torch.Tensor) -> torch.Tensor:
    conditioners = getattr(model, "profile_qkv_conditioners", None)
    if conditioners is None or len(conditioners) == 0:
        return profile_vector.new_tensor(0.0)
    vals = []
    scale = float(getattr(model, "profile_qkv_scale", 0.0))
    for conditioner in conditioners.values():
        cq, ck, cv = conditioner(profile_vector)
        delta = torch.cat([cq, ck, cv], dim=-1) * scale
        vals.append(delta.float().norm(p=2, dim=-1))
    if not vals:
        return profile_vector.new_tensor(0.0)
    return torch.stack(vals, dim=0).mean()


@torch.no_grad()
def _profile_adapter_delta_diagnostics(model: nn.Module, profile_vector: torch.Tensor) -> Dict[str, float]:
    profile_vector = profile_vector.detach()
    out = {
        "profile_final_norm": float(profile_vector.float().norm(p=2).cpu()),
        "qkv_delta_norm_mean": 0.0,
        "qkv_delta_norm_p95": 0.0,
        "film_delta_norm_mean": 0.0,
        "film_delta_norm_p95": 0.0,
    }
    conditioners = getattr(model, "profile_qkv_conditioners", None)
    if conditioners is not None and len(conditioners) > 0:
        vals = []
        scale = float(getattr(model, "profile_qkv_scale", 0.0))
        for conditioner in conditioners.values():
            cq, ck, cv = conditioner(profile_vector)
            delta = torch.cat([cq, ck, cv], dim=-1) * scale
            vals.extend(delta.float().norm(p=2, dim=-1).detach().cpu().numpy().reshape(-1).tolist())
        if vals:
            arr = np.asarray(vals, dtype=np.float32)
            out["qkv_delta_norm_mean"] = float(np.mean(arr))
            out["qkv_delta_norm_p95"] = float(np.percentile(arr, 95))

    film_vals = []
    for attr in ("profile_film_tokens", "profile_film_pooled"):
        film = getattr(model, attr, None)
        linear = getattr(film, "to_gamma_beta", None)
        if linear is None:
            continue
        gb = linear(profile_vector)
        gamma_raw, beta_raw = gb.chunk(2, dim=-1)
        scale = float(getattr(film, "scale", getattr(model, "profile_film_scale", 0.0)))
        delta = torch.cat([scale * torch.tanh(gamma_raw), scale * torch.tanh(beta_raw)], dim=-1)
        film_vals.extend(delta.float().norm(p=2, dim=-1).detach().cpu().numpy().reshape(-1).tolist())
    if film_vals:
        arr = np.asarray(film_vals, dtype=np.float32)
        out["film_delta_norm_mean"] = float(np.mean(arr))
        out["film_delta_norm_p95"] = float(np.percentile(arr, 95))
    return out


def _budget_fallback_decision(
    *,
    args,
    p_delta_norm: float,
    qkv_delta_norm_p95: float,
    pred_shift_p95_abs_bpm: float,
    rr_range_violation_fraction: float,
    stft_confidence_mean: float,
) -> Tuple[int, str]:
    if not bool(getattr(args, "profile_safety_budget", False)):
        return 0, ""
    reasons = []
    if np.isfinite(p_delta_norm) and p_delta_norm > float(getattr(args, "profile_delta_budget_max", 3.0)):
        reasons.append("profile_delta_norm")
    if np.isfinite(qkv_delta_norm_p95) and qkv_delta_norm_p95 > float(getattr(args, "qkv_delta_budget_max", 1.0)):
        reasons.append("qkv_delta_norm_p95")
    if np.isfinite(pred_shift_p95_abs_bpm) and pred_shift_p95_abs_bpm > float(getattr(args, "pred_shift_budget_bpm", 3.0)):
        reasons.append("pred_shift_p95_abs_bpm")
    if np.isfinite(rr_range_violation_fraction) and rr_range_violation_fraction > float(getattr(args, "rr_range_violation_budget_frac", 0.05)):
        reasons.append("rr_range_violation_fraction")
    if (
        bool(getattr(args, "profile_safety_budget_use_stft_confidence", True))
        and np.isfinite(stft_confidence_mean)
        and stft_confidence_mean < float(getattr(args, "budget_stft_confidence_floor", 0.005))
    ):
        reasons.append("stft_confidence_mean")
    return int(bool(reasons)), ",".join(reasons)


def adapt_profile_vector_unsupervised(
    model: nn.Module,
    rr_model: FaithfulRRRegressor,
    target_loader,
    target_windows: TensorWindows,
    source_anchor_windows: TensorWindows,
    mode: str,
    args,
    device: str,
    target_profile_stats_raw: Optional[torch.Tensor] = None,
    target_profile_stats_norm: Optional[torch.Tensor] = None,
) -> Tuple[nn.Parameter, Optional[TorchAffineCalibrator], Dict[str, float]]:
    """Profile-vector-only unsupervised TTA. Leaves model weights frozen."""
    if target_windows.imu.size(0) == 0:
        raise RuntimeError("Profile-vector TTA needs target windows.")
    if not bool(getattr(model, "use_profile_conditioning", getattr(model, "use_profile_film", False))):
        raise RuntimeError("Profile-vector modes require profile conditioning to be enabled on the model.")
    if getattr(model, "profile_encoder", None) is None:
        raise RuntimeError("Profile-vector modes require an initialized profile_encoder.")

    _set_rr_model_frozen(rr_model)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    if target_profile_stats_raw is None or target_profile_stats_norm is None:
        target_profile_stats_raw, target_profile_stats_norm = _profile_stats_from_loader(
            model, target_loader, device, args
        )
    conditioning_mode = _profile_conditioning_mode_for_rr_mode(mode)
    with torch.no_grad():
        p0 = model.profile_encoder(target_profile_stats_norm.unsqueeze(0)).detach()
        target_imu_full = target_windows.imu.to(device, non_blocking=True).float()
        _rr0, z_target_frozen, _stft0, _aux0, _conf0, _attn0 = _profile_conditioned_rr_probe(
            model, rr_model, target_imu_full, p0, device, conditioning_mode
        )
        z_target_frozen = z_target_frozen.detach()
        src_imu_full = source_anchor_windows.imu.to(device, non_blocking=True).float()
        if src_imu_full.size(0) > 0:
            _rrs0, z_source_frozen, _stfts0, _auxs0, _confs0, _src_attn0 = _profile_conditioned_rr_probe(
                model, rr_model, src_imu_full, p0, device, conditioning_mode
            )
            z_source_frozen = z_source_frozen.detach()
        else:
            z_source_frozen = None

    p_target = nn.Parameter(p0.detach().clone())
    use_readout_affine = (
        bool(getattr(args, "unsup_readout_affine", False))
        or ("readout_affine" in str(mode))
        or ("sparc" in str(mode) and "sparc_pt" not in str(mode))
    )
    readout_affine = TorchAffineCalibrator().to(device) if use_readout_affine else None

    opt_params = [p_target]
    if readout_affine is not None:
        opt_params += list(readout_affine.parameters())

    opt = torch.optim.AdamW(
        opt_params,
        lr=float(args.unsup_feature_lr),
        weight_decay=float(args.unsup_feature_weight_decay),
    )

    n_target = int(target_windows.imu.size(0))
    bs = max(2, int(args.unsup_feature_batch_size))
    stft_w = float(args.unsup_stft_consistency_weight)
    uses_stft_pseudotarget = stft_w > 0.0
    smooth_w = float(args.unsup_smoothness_weight)
    drift_w = float(args.unsup_target_drift_weight)
    anchor_w = float(args.unsup_source_anchor_weight)
    prior_w = float(getattr(args, "unsup_profile_prior_weight", 0.01))
    attn_w = float(getattr(args, "unsup_attention_profile_weight", 0.0))
    qkv_budget_w = float(getattr(args, "qkv_delta_budget_weight", 0.0))

    aux_cons_w = float(getattr(args, "unsup_aux_consistency_weight", 0.0))
    range_w = float(getattr(args, "unsup_rr_range_weight", 0.0))
    min_rr_std = float(getattr(args, "unsup_rr_range_min_std", 1.0))
    aff_lam_a = float(getattr(args, "unsup_readout_affine_lambda_a", 1.0))
    aff_lam_b = float(getattr(args, "unsup_readout_affine_lambda_b", 0.1))

    conf_floor = float(args.unsup_stft_confidence_floor)
    conf_power = float(args.unsup_stft_confidence_power)
    rr_disagreement_threshold = float(getattr(args, "unsup_rr_disagreement_threshold", 12.0))
    rr_min_bpm = float(getattr(args, "unsup_rr_min_bpm", 4.0))
    rr_max_bpm = float(getattr(args, "unsup_rr_max_bpm", 45.0))
    max_temporal_jump_bpm = float(getattr(args, "unsup_max_temporal_jump_bpm", 10.0))
    source_attn_ref = (
        _collect_profile_attention_reference(
            model,
            source_anchor_windows,
            p0,
            device,
            batch_size=bs,
            conditioning_mode=conditioning_mode,
        )
        if bool(getattr(model, "use_profile_qkv", False)) and source_anchor_windows.imu.size(0) > 0
        else {}
    )

    last_log: Dict[str, float] = {}
    for epoch in range(1, int(args.unsup_feature_epochs) + 1):
        model.eval()
        epoch_losses: List[float] = []
        epoch_cons: List[float] = []
        epoch_smooth: List[float] = []
        epoch_drift: List[float] = []
        epoch_anchor: List[float] = []
        epoch_prior: List[float] = []
        epoch_attn: List[float] = []
        epoch_conf: List[float] = []
        epoch_reliable: List[float] = []
        epoch_rr_disagreement: List[float] = []
        epoch_aux_cons: List[float] = []
        epoch_range: List[float] = []
        epoch_affine_prior: List[float] = []
        epoch_qkv_budget: List[float] = []

        for st in range(0, n_target, bs):
            end = min(n_target, st + bs)
            idx_cpu = torch.arange(st, end, dtype=torch.long)
            idx_dev = idx_cpu.to(device)
            imu = target_windows.imu[idx_cpu].to(device, non_blocking=True).float()
            z_ref = z_target_frozen[idx_dev]

            opt.zero_grad(set_to_none=True)
            _rr_probe, z, rr_stft, rr_aux, conf, attn_metrics = _profile_conditioned_rr_probe(
                model, rr_model, imu, p_target, device, conditioning_mode
            )
            # The final evaluated signal is the frozen RR probe readout.
            # For SPARC/readout-affine modes, optimize this final readout directly
            # against the decoder-derived respiratory anchor rather than optimising
            # only the auxiliary RR head.
            rr_final = (
                _rr_probe
                if readout_affine is None
                else readout_affine(_rr_probe)
            )

            rr_loss_driver = (
                rr_final
                if use_readout_affine or not uses_stft_pseudotarget
                else rr_aux
            )

            disagreement = (rr_loss_driver - rr_stft.detach()).abs()
            reliable = torch.isfinite(rr_loss_driver)
            if uses_stft_pseudotarget:
                reliable = (
                    reliable
                    & torch.isfinite(rr_stft)
                    & torch.isfinite(conf)
                    & (disagreement <= rr_disagreement_threshold)
                )
            elif aux_cons_w > 0.0:
                reliable = reliable & torch.isfinite(rr_final) & torch.isfinite(rr_aux)
            reliable = reliable & (rr_loss_driver >= rr_min_bpm) & (rr_loss_driver <= rr_max_bpm)

            if rr_loss_driver.numel() > 1 and max_temporal_jump_bpm > 0.0:
                jump_ok = torch.ones_like(reliable, dtype=torch.bool)
                jump_ok[1:] = (
                    rr_loss_driver[1:] - rr_loss_driver[:-1].detach()
                ).abs() <= max_temporal_jump_bpm
                reliable = reliable & jump_ok

            if uses_stft_pseudotarget:
                conf_w = conf.detach().clamp(float(conf_floor), 1.0).pow(float(conf_power)) * reliable.float()
                consistency = _weighted_smooth_l1(
                    rr_loss_driver, rr_stft.detach(), conf_w
                )
            else:
                consistency = rr_loss_driver.new_tensor(0.0)

            if rr_loss_driver.numel() > 1:
                smooth_mask = (reliable[1:] & reliable[:-1]).float()
                smooth_per = F.smooth_l1_loss(
                    rr_loss_driver[1:],
                    rr_loss_driver[:-1].detach(),
                    reduction="none",
                )
                smoothness = (smooth_per * smooth_mask).sum() / smooth_mask.sum().clamp_min(1.0)
            else:
                smoothness = rr_loss_driver.new_tensor(0.0)

            target_drift = F.mse_loss(z, z_ref)
            profile_prior = F.mse_loss(p_target, p0.detach())
            qkv_budget = _qkv_delta_norm_tensor(model, p_target).pow(2)

            if attn_w > 0.0 and attn_metrics and source_attn_ref:
                attn_terms = []
                for key, src_val in source_attn_ref.items():
                    if key in attn_metrics:
                        attn_terms.append((float(attn_metrics[key]) - float(src_val)) ** 2)
                attention_profile = rr_loss_driver.new_tensor(
                    float(np.mean(attn_terms))
                ) if attn_terms else rr_loss_driver.new_tensor(0.0)
            else:
                attention_profile = rr_loss_driver.new_tensor(0.0)

            if anchor_w > 0.0 and z_source_frozen is not None and src_imu_full.size(0) > 0:
                n_src = src_imu_full.size(0)
                src_take = min(n_src, max(1, int(args.unsup_source_anchor_batch_size)))
                src_idx = torch.randint(0, n_src, (src_take,), device=device)
                _src_rr, z_src, _src_stft, _src_aux, _src_conf, _src_attn = _profile_conditioned_rr_probe(
                    model, rr_model, src_imu_full[src_idx], p_target, device, conditioning_mode
                )
                source_anchor = F.mse_loss(z_src, z_source_frozen[src_idx])
            else:
                source_anchor = target_drift.new_tensor(0.0)

            if aux_cons_w > 0.0 and (use_readout_affine or not uses_stft_pseudotarget):
                aux_consistency = _weighted_smooth_l1(
                    rr_final,
                    rr_aux.detach(),
                    reliable.float(),
                )
            else:
                aux_consistency = rr_loss_driver.new_tensor(0.0)

            if range_w > 0.0 and rr_loss_driver.numel() >= 2:
                valid_rr = rr_loss_driver[reliable] if bool(reliable.any()) else rr_loss_driver
                rr_std = valid_rr.float().std(unbiased=False)
                range_loss = F.relu(rr_loss_driver.new_tensor(min_rr_std) - rr_std).pow(2)
            else:
                range_loss = rr_loss_driver.new_tensor(0.0)

            if readout_affine is not None:
                affine_prior = (
                    aff_lam_a * (readout_affine.a - 1.0).pow(2)
                    + aff_lam_b * readout_affine.b.pow(2)
                )
            else:
                affine_prior = rr_loss_driver.new_tensor(0.0)

            loss = (
                stft_w * consistency
                + aux_cons_w * aux_consistency
                + smooth_w * smoothness
                + drift_w * target_drift
                + anchor_w * source_anchor
                + prior_w * profile_prior
                + attn_w * attention_profile
                + range_w * range_loss
                + affine_prior
                + qkv_budget_w * qkv_budget
            )
            loss.backward()
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(opt_params, float(args.grad_clip))
            opt.step()

            epoch_losses.append(float(loss.detach().cpu()))
            epoch_cons.append(float(consistency.detach().cpu()))
            epoch_smooth.append(float(smoothness.detach().cpu()))
            epoch_drift.append(float(target_drift.detach().cpu()))
            epoch_anchor.append(float(source_anchor.detach().cpu()))
            epoch_prior.append(float(profile_prior.detach().cpu()))
            epoch_attn.append(float(attention_profile.detach().cpu()))
            epoch_conf.append(float(conf.detach().cpu().mean()))
            epoch_reliable.append(float(reliable.float().mean().detach().cpu()))
            epoch_rr_disagreement.append(float(disagreement.mean().detach().cpu()))
            epoch_aux_cons.append(float(aux_consistency.detach().cpu()))
            epoch_range.append(float(range_loss.detach().cpu()))
            epoch_affine_prior.append(float(affine_prior.detach().cpu()))
            epoch_qkv_budget.append(float(qkv_budget.detach().cpu()))

        last_log = {
            "unsup_feature_tta_epoch": int(epoch),
            "unsup_feature_loss_last": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
            "unsup_stft_consistency_last": float(np.mean(epoch_cons)) if epoch_cons else float("nan"),
            "unsup_smoothness_last": float(np.mean(epoch_smooth)) if epoch_smooth else 0.0,
            "unsup_target_drift_last": float(np.mean(epoch_drift)) if epoch_drift else 0.0,
            "unsup_source_anchor_last": float(np.mean(epoch_anchor)) if epoch_anchor else 0.0,
            "profile_prior_loss_last": float(np.mean(epoch_prior)) if epoch_prior else 0.0,
            "attn_profile_loss_last": float(np.mean(epoch_attn)) if epoch_attn else 0.0,
            "unsup_aux_consistency_last": float(np.mean(epoch_aux_cons)) if epoch_aux_cons else 0.0,
            "unsup_rr_range_loss_last": float(np.mean(epoch_range)) if epoch_range else 0.0,
            "unsup_readout_affine_prior_last": float(np.mean(epoch_affine_prior)) if epoch_affine_prior else 0.0,
            "qkv_delta_budget_loss_last": float(np.mean(epoch_qkv_budget)) if epoch_qkv_budget else 0.0,
            "unsup_stft_confidence_mean_last": float(np.mean(epoch_conf)) if epoch_conf else float("nan"),
            "unsup_reliable_window_ratio": float(np.mean(epoch_reliable)) if epoch_reliable else float("nan"),
            "unsup_rr_disagreement_mean": float(np.mean(epoch_rr_disagreement)) if epoch_rr_disagreement else float("nan"),
        }

    diag = _profile_stats_scalar_diagnostics(
        model,
        target_profile_stats_raw,
        target_profile_stats_norm,
        p_target,
        args,
    )
    adapter_diag = _profile_adapter_delta_diagnostics(model, p_target)
    p_delta = p_target.detach() - p0.detach()

    n_affine_params = int(
        sum(p.numel() for p in readout_affine.parameters())
    ) if readout_affine is not None else 0
    n_profile_params = int(p_target.numel())
    n_trainable_total = n_profile_params + n_affine_params

    train_counts = {
        "feature_trainable_params": int(n_trainable_total),
        "feature_trainable_layernorm_params": 0,
        "feature_trainable_spec_proj_params": 0,
        "feature_trainable_last_block_params": 0,
        "feature_trainable_rr_head_params": 0,
        "feature_trainable_pressure_decoder_params": 0,
    }
    extra = {
        **train_counts,
        **last_log,
        **diag,
        **adapter_diag,
        **_profile_clsa_config_diagnostics(model, args),
        "feature_tta_mode": mode,
        "unsup_feature_tta_epochs": int(args.unsup_feature_epochs),
        "unsup_feature_tta_lr": float(args.unsup_feature_lr),
        "unsup_stft_consistency_weight": float(stft_w),
        "uses_stft_pseudotarget_for_tta": int(bool(uses_stft_pseudotarget)),
        "unsup_smoothness_weight": float(smooth_w),
        "unsup_source_anchor_weight": float(args.unsup_source_anchor_weight),
        "unsup_target_drift_weight": float(drift_w),
        "unsup_profile_prior_weight": float(prior_w),
        "unsup_attention_profile_weight": float(attn_w),
        "qkv_delta_budget_weight": float(qkv_budget_w),
        "unsup_stft_confidence_floor": float(conf_floor),
        "unsup_stft_confidence_power": float(conf_power),
        "unsup_rr_disagreement_threshold": float(rr_disagreement_threshold),
        "unsup_rr_min_bpm": float(rr_min_bpm),
        "unsup_rr_max_bpm": float(rr_max_bpm),
        "unsup_max_temporal_jump_bpm": float(max_temporal_jump_bpm),
        "unsup_readout_affine_enabled": int(bool(use_readout_affine)),
        "unsup_aux_consistency_weight": float(aux_cons_w),
        "unsup_rr_range_weight": float(range_w),
        "unsup_rr_range_min_std": float(min_rr_std),
        "unsup_readout_affine_lambda_a": float(aff_lam_a),
        "unsup_readout_affine_lambda_b": float(aff_lam_b),
        "readout_affine_a": float(readout_affine.a.detach().cpu()) if readout_affine is not None else 1.0,
        "readout_affine_b": float(readout_affine.b.detach().cpu()) if readout_affine is not None else 0.0,
        "profile_delta_norm": float(p_delta.norm(p=2).cpu()),
        "profile_film_scale": float(getattr(model, "profile_film_scale", getattr(args, "profile_film_scale", 0.1))),
        "profile_film_placement": str(getattr(model, "profile_film_placement", getattr(args, "profile_film_placement", "token_pooled"))),
        "profile_film_residual_alpha": float(getattr(model, "profile_film_residual_alpha", getattr(args, "profile_film_residual_alpha", 0.1))),
        "profile_qkv_scale": float(getattr(model, "profile_qkv_scale", getattr(args, "profile_qkv_scale", 0.1))),
        "profile_qkv_layers_used": str(getattr(model, "profile_qkv_layers", getattr(args, "profile_qkv_layers", "last1"))),
        "profile_qkv_residual": int(bool(getattr(model, "profile_qkv_residual", getattr(args, "profile_qkv_residual", False)))),
        "profile_conditioning_mode": str(getattr(model, "profile_conditioning", "none")),
        "profile_safety_budget_use_stft_confidence": int(bool(getattr(args, "profile_safety_budget_use_stft_confidence", True))),
        "profile_unsup_episodic_batch": int(bool(getattr(args, "profile_unsup_episodic_batch", False))),
        "profile_unsup_episodic_batches": 0,
        "profile_adapted_vector": 1,
        "profile_adapted_encoder": 0,
        "profile_adapted_film": 0,
        "profile_trainable_params": int(n_profile_params),
        "readout_affine_trainable_params": int(n_affine_params),
        "feature_trainable_names_json": json.dumps(
            ["p_target"] + (
                ["readout_affine.a", "readout_affine.b"] if readout_affine \
                is not None else []
            )
        ),
    }
    for key, value in source_attn_ref.items():
        extra[f"attn_distance_source_{key}"] = float(value)

    return p_target, readout_affine, extra


def _mean_episodic_profile_extras(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row.keys()})
    out: Dict[str, float] = {}
    for key in keys:
        vals = [row[key] for row in rows if key in row]
        if vals and all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals):
            out[key] = float(np.mean([float(v) for v in vals]))
        elif vals:
            out[key] = vals[-1]
    out["profile_unsup_episodic_batch"] = 1
    out["profile_unsup_episodic_batches"] = int(len(rows))
    return out


def adapt_profile_vector_episodic_batches(
    model: nn.Module,
    rr_model: FaithfulRRRegressor,
    target_loader,
    eval_windows: TensorWindows,
    source_anchor_windows: TensorWindows,
    mode: str,
    args,
    device: str,
    conditioning_mode: str,
    target_profile_stats_raw: torch.Tensor,
    target_profile_stats_norm: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Reset p_t for each target batch, adapt on that batch, then evaluate it."""
    if eval_windows.imu.size(0) == 0:
        raise RuntimeError("Episodic profile-vector TTT needs evaluation windows.")
    bs = max(1, int(args.unsup_feature_batch_size))
    preds: List[np.ndarray] = []
    feats: List[np.ndarray] = []
    stft_rrs: List[np.ndarray] = []
    aux_rrs: List[np.ndarray] = []
    confs: List[np.ndarray] = []
    extra_rows: List[Dict[str, float]] = []

    for st in range(0, int(eval_windows.imu.size(0)), bs):
        end = min(int(eval_windows.imu.size(0)), st + bs)
        batch_windows = eval_windows.subset(np.arange(st, end, dtype=np.int64))
        p_batch, readout_affine, extra = adapt_profile_vector_unsupervised(
            model,
            rr_model,
            target_loader,
            batch_windows,
            source_anchor_windows,
            mode,
            args,
            device,
            target_profile_stats_raw=target_profile_stats_raw,
            target_profile_stats_norm=target_profile_stats_norm,
        )
        pred, z, rr_stft, rr_aux, conf = _predict_profile_unsup_with_features(
            model,
            rr_model,
            batch_windows,
            p_batch,
            device,
            int(args.feature_eval_batch_size),
            conditioning_mode,
            readout_affine=readout_affine,
        )
        preds.append(pred)
        feats.append(z)
        stft_rrs.append(rr_stft)
        aux_rrs.append(rr_aux)
        confs.append(conf)
        extra_rows.append(extra)

    return (
        np.concatenate(preds, axis=0),
        np.concatenate(feats, axis=0).astype(np.float32),
        np.concatenate(stft_rrs, axis=0),
        np.concatenate(aux_rrs, axis=0),
        np.concatenate(confs, axis=0),
        _mean_episodic_profile_extras(extra_rows),
    )


def adapt_feature_space_unsupervised(
    model: nn.Module,
    rr_model: FaithfulRRRegressor,
    target_windows: TensorWindows,
    source_anchor_windows: TensorWindows,
    mode: str,
    args,
    device: str,
) -> Dict[str, float]:
    """Unsupervised feature TTA: no target RR labels and no affine calibration.

    Modes form a ladder:
      ln_unsup_stft_consistency: LN-only + RR-probe/STFT consistency
      lastblock_ln_unsup_stft_consistency: add final block + spec_proj capacity
      lastblock_ln_unsup_stft_smooth: add temporal smoothness
      lastblock_ln_unsup_stft_smooth_anchor: add source-feature anchor
    """
    if target_windows.imu.size(0) == 0:
        raise RuntimeError("Unsupervised feature-TTA needs target windows.")

    _set_rr_model_frozen(rr_model)
    train_counts = _configure_feature_trainability(model, mode)
    model.eval()

    with torch.no_grad():
        target_imu_full = target_windows.imu.to(device, non_blocking=True).float()
        src_imu_full = source_anchor_windows.imu.to(device, non_blocking=True).float()
        _rr0, z_target_frozen, _stft0, _aux0, _conf0 = _run_model_rr_probe_with_confidence(model, rr_model, target_imu_full, device)
        _rrs0, z_source_frozen, _stfts0, _auxs0, _confs0 = _run_model_rr_probe_with_confidence(model, rr_model, src_imu_full, device)
        z_target_frozen = z_target_frozen.detach()
        z_source_frozen = z_source_frozen.detach()

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError(f"No trainable parameters selected for unsupervised mode={mode}.")

    opt = torch.optim.AdamW(params, lr=float(args.unsup_feature_lr), weight_decay=float(args.unsup_feature_weight_decay))
    n_target = int(target_windows.imu.size(0))
    bs = max(2, int(args.unsup_feature_batch_size))

    stft_w = float(args.unsup_stft_consistency_weight)
    smooth_w = float(args.unsup_smoothness_weight)
    anchor_w = float(args.unsup_source_anchor_weight)
    drift_w = float(args.unsup_target_drift_weight)
    conf_floor = float(args.unsup_stft_confidence_floor)
    conf_power = float(args.unsup_stft_confidence_power)

    last_log: Dict[str, float] = {}
    for epoch in range(1, int(args.unsup_feature_epochs) + 1):
        model.eval()
        epoch_losses: List[float] = []
        epoch_cons: List[float] = []
        epoch_smooth: List[float] = []
        epoch_drift: List[float] = []
        epoch_anchor: List[float] = []
        epoch_conf: List[float] = []

        # Ordered batches preserve adjacent-window meaning for smoothness.
        for st in range(0, n_target, bs):
            end = min(n_target, st + bs)
            idx_cpu = torch.arange(st, end, dtype=torch.long)
            idx_dev = idx_cpu.to(device)
            imu = target_windows.imu[idx_cpu].to(device, non_blocking=True).float()
            z_ref = z_target_frozen[idx_dev]

            opt.zero_grad(set_to_none=True)
            rr_probe, z, rr_stft, _rr_aux, conf = _run_model_rr_probe_with_confidence(model, rr_model, imu, device)
            conf_w = conf.detach().clamp(float(conf_floor), 1.0).pow(float(conf_power))
            consistency = _weighted_smooth_l1(rr_probe, rr_stft.detach(), conf_w)
            smoothness = F.smooth_l1_loss(rr_probe[1:], rr_probe[:-1].detach()) if rr_probe.numel() > 1 else rr_probe.new_tensor(0.0)
            target_drift = F.mse_loss(z, z_ref)

            if anchor_w > 0.0 and src_imu_full.size(0) > 0:
                n_src = src_imu_full.size(0)
                src_take = min(n_src, max(1, int(args.unsup_source_anchor_batch_size)))
                src_idx = torch.randint(0, n_src, (src_take,), device=device)
                _src_rr, z_src, _src_stft, _src_aux, _src_conf = _run_model_rr_probe_with_confidence(model, rr_model, src_imu_full[src_idx], device)
                source_anchor = F.mse_loss(z_src, z_source_frozen[src_idx])
            else:
                source_anchor = target_drift.new_tensor(0.0)

            loss = (
                stft_w * consistency
                + smooth_w * smoothness
                + drift_w * target_drift
                + anchor_w * source_anchor
            )
            loss.backward()
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip))
            opt.step()

            epoch_losses.append(float(loss.detach().cpu()))
            epoch_cons.append(float(consistency.detach().cpu()))
            epoch_smooth.append(float(smoothness.detach().cpu()))
            epoch_drift.append(float(target_drift.detach().cpu()))
            epoch_anchor.append(float(source_anchor.detach().cpu()))
            epoch_conf.append(float(conf.detach().cpu().mean()))

        last_log = {
            "unsup_feature_tta_epoch": int(epoch),
            "unsup_feature_loss_last": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
            "unsup_stft_consistency_last": float(np.mean(epoch_cons)) if epoch_cons else float("nan"),
            "unsup_smoothness_last": float(np.mean(epoch_smooth)) if epoch_smooth else 0.0,
            "unsup_target_drift_last": float(np.mean(epoch_drift)) if epoch_drift else 0.0,
            "unsup_source_anchor_last": float(np.mean(epoch_anchor)) if epoch_anchor else 0.0,
            "unsup_stft_confidence_mean_last": float(np.mean(epoch_conf)) if epoch_conf else float("nan"),
        }

    return {
        **train_counts,
        **last_log,
        "feature_tta_mode": mode,
        "unsup_feature_tta_epochs": int(args.unsup_feature_epochs),
        "unsup_feature_tta_lr": float(args.unsup_feature_lr),
        "unsup_stft_consistency_weight": float(stft_w),
        "unsup_smoothness_weight": float(smooth_w),
        "unsup_source_anchor_weight": float(anchor_w),
        "unsup_target_drift_weight": float(drift_w),
        "unsup_stft_confidence_floor": float(conf_floor),
        "unsup_stft_confidence_power": float(conf_power),
        "feature_trainable_names_json": json.dumps(_feature_param_names(model)[:200]),
    }


def feature_adaptive_unsupervised_evaluate(
    model: nn.Module,
    train_loader,
    test_loader,
    subject: str,
    device: str,
    args,
    shared_context: Optional[SubjectEvalContext] = None,
    out_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """Evaluate an unsupervised feature-TTA mode and restore model weights."""
    mode = str(args.rr_tta).lower()
    conditioning_mode = _profile_conditioning_mode_for_rr_mode(mode)
    _warm_lazy_modules(model, train_loader, device)
    original_state = _state_dict_cpu_clone(model)

    try:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        ctx = shared_context or _build_subject_eval_context(
            model, train_loader, test_loader, device, args, include_kinematics=False
        )
        x_source, y_source = ctx.x_source, ctx.y_source
        x_target, y_target = ctx.x_target, ctx.y_target
        x_cal, y_cal, x_eval, y_eval, cal_idx = ctx.x_cal, ctx.y_cal, ctx.x_eval, ctx.y_eval, ctx.cal_idx
        rr_model = FaithfulRRRegressor(x_source.shape[1]).to(device)
        rr_model.load_state_dict(ctx.rr_model_state, strict=True)
        _set_rr_model_frozen(rr_model)

        pred_pre = predict_rr(rr_model, x_eval, device)
        metrics: Dict[str, float] = rr_metrics(y_eval, pred_pre, prefix="rr_probe_pre")
        _attach_base_pre_aliases(metrics)

        target_windows = ctx.target_windows
        eval_windows = ctx.eval_windows
        source_anchor_windows = ctx.source_anchor_windows

        extra = adapt_feature_space_unsupervised(
            model, rr_model, target_windows, source_anchor_windows, mode, args, device
        )
        pred_post, stft_rr_eval, aux_rr_eval, stft_conf_eval = _predict_feature_unsup(
            model, rr_model, eval_windows, device, int(args.feature_eval_batch_size)
        )
        y_eval_tensor = eval_windows.rr.detach().cpu().numpy().reshape(-1).astype(np.float32)
        y_eval_for_metrics = y_eval_tensor if y_eval_tensor.shape[0] == y_eval.shape[0] else y_eval.astype(np.float32)

        metrics.update(rr_metrics(y_eval_for_metrics, pred_post, prefix="rr_probe_post"))
        metrics.update(extra)
        metrics.update({
            "rr_tta_mode": mode,
            "is_unsupervised_feature_tta": 1,
            "uses_target_rr_labels_for_adaptation": 0,
            "rr_probe_n_source": int(x_source.shape[0]),
            "rr_probe_n_target_total": int(x_target.shape[0]),
            "rr_probe_n_calibration": int(x_cal.shape[0]),
            "rr_probe_n_eval": int(y_eval_for_metrics.shape[0]),
            "rr_probe_n_features": int(x_source.shape[1]),
            "target_calibration_indices": json.dumps(cal_idx.tolist()),
            "eval_excludes_calibration": int(bool(args.exclude_calibration_from_eval)),
            "eval_stft_rr_mae_vs_label": float(np.mean(np.abs(stft_rr_eval - y_eval_for_metrics))),
            "eval_aux_rr_mae_vs_label": float(np.mean(np.abs(aux_rr_eval - y_eval_for_metrics))),
            "eval_stft_confidence_mean": float(np.mean(stft_conf_eval)),
            "eval_probe_pre_matches_tensor_labels": int(y_eval_tensor.shape[0] == y_eval.shape[0]),
        })

        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({
                "rr_true": y_eval_for_metrics,
                "rr_pred_pre": pred_pre[: y_eval_for_metrics.shape[0]],
                "rr_pred_post": pred_post,
                "rr_stft_post": stft_rr_eval,
                "rr_aux_head_post": aux_rr_eval,
                "rr_stft_confidence": stft_conf_eval,
            }).to_csv(out_dir / f"rr_unsup_ladder_predictions_{subject}.csv", index=False)
            with open(out_dir / f"rr_unsup_ladder_metrics_{subject}.json", "w") as f:
                json.dump(metrics, f, indent=2)
        return metrics
    finally:
        _restore_state_dict(model, original_state, device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()


def readout_ablation_evaluate(
    model: nn.Module,
    train_loader,
    test_loader,
    subject: str,
    device: str,
    args,
    shared_context: Optional[SubjectEvalContext] = None,
    out_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """Evaluate readout-only ablations without changing model/profile state.

    direct_stft_rr: use dominant-frequency RR from reconstructed STFT.
    hybrid_probe_stft_conf: use STFT-RR only when raw confidence and
    disagreement checks pass; otherwise fall back to the frozen RR probe.
    """
    mode = str(args.rr_tta).lower()
    _warm_lazy_modules(model, train_loader, device)
    original_state = _state_dict_cpu_clone(model)
    try:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        ctx = shared_context or _build_subject_eval_context(
            model, train_loader, test_loader, device, args, include_kinematics=False
        )
        x_source, y_source = ctx.x_source, ctx.y_source
        x_target, x_cal, x_eval, y_eval, cal_idx = ctx.x_target, ctx.x_cal, ctx.x_eval, ctx.y_eval, ctx.cal_idx
        rr_model = FaithfulRRRegressor(x_source.shape[1]).to(device)
        rr_model.load_state_dict(ctx.rr_model_state, strict=True)
        _set_rr_model_frozen(rr_model)

        pred_pre = predict_rr(rr_model, x_eval, device)
        eval_windows = ctx.eval_windows
        pred_probe_eval, stft_rr_eval, aux_rr_eval, stft_conf_eval = _predict_feature_unsup(
            model,
            rr_model,
            eval_windows,
            device,
            int(args.feature_eval_batch_size),
        )
        y_eval_tensor = eval_windows.rr.detach().cpu().numpy().reshape(-1).astype(np.float32)
        y_eval_for_metrics = y_eval_tensor if y_eval_tensor.shape[0] == y_eval.shape[0] else y_eval.astype(np.float32)
        probe_for_eval = pred_probe_eval if pred_probe_eval.shape[0] == y_eval_for_metrics.shape[0] else pred_pre[: y_eval_for_metrics.shape[0]]

        if mode == "direct_stft_rr":
            pred_post = stft_rr_eval
            use_stft = np.ones_like(stft_rr_eval, dtype=bool)
        elif mode == "hybrid_probe_stft_conf":
            conf_thr = float(getattr(args, "hybrid_stft_confidence_threshold", 0.005))
            aux_thr = float(getattr(args, "hybrid_aux_stft_disagreement_bpm", 3.0))
            probe_thr = float(getattr(args, "hybrid_probe_stft_disagreement_bpm", 4.0))
            use_stft = (
                np.isfinite(stft_rr_eval)
                & np.isfinite(stft_conf_eval)
                & np.isfinite(aux_rr_eval)
                & np.isfinite(probe_for_eval)
                & (stft_conf_eval >= conf_thr)
                & (np.abs(aux_rr_eval - stft_rr_eval) <= aux_thr)
                & (np.abs(probe_for_eval - stft_rr_eval) <= probe_thr)
            )
            pred_post = np.where(use_stft, stft_rr_eval, probe_for_eval)
        else:
            raise ValueError(f"Unsupported readout ablation mode={mode!r}")

        metrics: Dict[str, float] = rr_metrics(y_eval_for_metrics, probe_for_eval, prefix="rr_probe_pre")
        _attach_base_pre_aliases(metrics)
        metrics.update(rr_metrics(y_eval_for_metrics, pred_post, prefix="rr_probe_post"))
        metrics.update({
            "rr_tta_mode": mode,
            "is_readout_ablation": 1,
            "uses_target_rr_labels_for_adaptation": 0,
            "rr_probe_n_source": int(x_source.shape[0]),
            "rr_probe_n_target_total": int(x_target.shape[0]),
            "rr_probe_n_calibration": int(x_cal.shape[0]),
            "rr_probe_n_eval": int(y_eval_for_metrics.shape[0]),
            "rr_probe_n_features": int(x_source.shape[1]),
            "target_calibration_indices": json.dumps(cal_idx.tolist()),
            "eval_excludes_calibration": int(bool(args.exclude_calibration_from_eval)),
            "eval_stft_rr_mae_vs_label": _mae_np(y_eval_for_metrics, stft_rr_eval),
            "eval_aux_rr_mae_vs_label": _mae_np(y_eval_for_metrics, aux_rr_eval),
            "eval_stft_confidence_mean": float(np.mean(stft_conf_eval)) if stft_conf_eval.size else float("nan"),
            "hybrid_use_stft_fraction": float(np.mean(use_stft)) if use_stft.size else 0.0,
            "hybrid_stft_confidence_threshold": float(getattr(args, "hybrid_stft_confidence_threshold", 0.005)),
            "hybrid_aux_stft_disagreement_bpm": float(getattr(args, "hybrid_aux_stft_disagreement_bpm", 3.0)),
            "hybrid_probe_stft_disagreement_bpm": float(getattr(args, "hybrid_probe_stft_disagreement_bpm", 4.0)),
            "hybrid_aux_stft_disagreement_mean_bpm": float(np.mean(np.abs(aux_rr_eval - stft_rr_eval))) if stft_rr_eval.size else float("nan"),
            "hybrid_probe_stft_disagreement_mean_bpm": float(np.mean(np.abs(probe_for_eval - stft_rr_eval))) if stft_rr_eval.size else float("nan"),
        })

        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({
                "rr_true": y_eval_for_metrics,
                "rr_pred_pre": probe_for_eval,
                "rr_pred_post": pred_post,
                "rr_stft_direct": stft_rr_eval,
                "rr_aux_head": aux_rr_eval,
                "rr_stft_confidence": stft_conf_eval,
                "hybrid_use_stft": use_stft.astype(int),
            }).to_csv(out_dir / f"rr_readout_ablation_predictions_{subject}.csv", index=False)
            with open(out_dir / f"rr_readout_ablation_metrics_{subject}.json", "w") as f:
                json.dump(metrics, f, indent=2)
        return metrics
    finally:
        _restore_state_dict(model, original_state, device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()


def profile_vector_unsupervised_evaluate(
    model: nn.Module,
    train_loader,
    test_loader,
    subject: str,
    device: str,
    args,
    shared_context: Optional[SubjectEvalContext] = None,
    out_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """Evaluate profile-vector ablations, unsupervised TTA, and safety-gated variants."""
    mode = str(args.rr_tta).lower()
    conditioning_mode = _profile_conditioning_mode_for_rr_mode(mode)
    _warm_lazy_modules(model, train_loader, device)
    original_state = _state_dict_cpu_clone(model)

    try:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        ctx = shared_context or _build_subject_eval_context(
            model, train_loader, test_loader, device, args, include_kinematics=False
        )
        x_source, y_source = ctx.x_source, ctx.y_source
        x_target, y_target = ctx.x_target, ctx.y_target
        x_cal, y_cal, x_eval, y_eval, cal_idx = ctx.x_cal, ctx.y_cal, ctx.x_eval, ctx.y_eval, ctx.cal_idx
        rr_model = FaithfulRRRegressor(x_source.shape[1]).to(device)
        rr_model.load_state_dict(ctx.rr_model_state, strict=True)
        _set_rr_model_frozen(rr_model)

        pred_pre = predict_rr(rr_model, x_eval, device)
        metrics: Dict[str, float] = rr_metrics(y_eval, pred_pre, prefix="rr_probe_pre")
        _attach_base_pre_aliases(metrics)

        target_windows = ctx.target_windows
        eval_windows = ctx.eval_windows
        source_anchor_windows = ctx.source_anchor_windows
        y_eval_tensor = eval_windows.rr.detach().cpu().numpy().reshape(-1).astype(np.float32)
        y_eval_for_metrics = y_eval_tensor if y_eval_tensor.shape[0] == y_eval.shape[0] else y_eval.astype(np.float32)

        adapt_windows, adapt_scope_used = _select_profile_adaptation_windows(
            target_windows,
            eval_windows,
            cal_idx,
            args,
        )

        target_profile_stats_raw, target_profile_stats_norm = _profile_stats_from_windows(
            model,
            adapt_windows,
            device,
            args,
            batch_size=int(args.feature_eval_batch_size),
        )
        with torch.no_grad():
            p0 = model.profile_encoder(target_profile_stats_norm.unsqueeze(0)).detach()

        z_base_eval = _pooled_features_from_windows(
            model,
            eval_windows,
            device,
            int(args.feature_eval_batch_size),
        )
        pred_profile_init, z_profile_init_eval, stft_rr_init_eval, aux_rr_init_eval, stft_conf_init_eval = _predict_profile_unsup_with_features(
            model,
            rr_model,
            eval_windows,
            p0,
            device,
            int(args.feature_eval_batch_size),
            conditioning_mode,
        )
        base_probe_adapt, _base_stft_adapt, _base_aux_adapt, _base_conf_adapt = _predict_feature_unsup(
            model,
            rr_model,
            adapt_windows,
            device,
            int(args.feature_eval_batch_size),
        )
        pred_profile_init_adapt, stft_rr_init_adapt, aux_rr_init_adapt, stft_conf_init_adapt = _predict_profile_unsup(
            model,
            rr_model,
            adapt_windows,
            p0,
            device,
            int(args.feature_eval_batch_size),
            conditioning_mode,
        )
        y_adapt = adapt_windows.rr.detach().cpu().numpy().reshape(-1).astype(np.float32)

        gate_diag = _profile_gate_diagnostics(
            args,
            target_profile_stats_norm,
            base_probe=base_probe_adapt,
            profile_init_probe=pred_profile_init_adapt,
            aux_rr=aux_rr_init_adapt,
            stft_rr=stft_rr_init_adapt,
            stft_conf=stft_conf_init_adapt,
        )
        oracle_diag = _profile_oracle_diagnostics(
            args,
            y_cal=y_adapt,
            base_probe_cal=base_probe_adapt,
            profile_init_cal=pred_profile_init_adapt,
        )

        attn_pre = _collect_profile_attention_reference(
            model,
            eval_windows,
            p0,
            device,
            batch_size=int(args.feature_eval_batch_size),
            conditioning_mode=conditioning_mode,
        )

        needs_adapt = not _profile_mode_init_only(mode)
        if needs_adapt:
            if bool(getattr(args, "profile_unsup_episodic_batch", False)):
                p_target = nn.Parameter(p0.detach().clone())
                readout_affine = None
                (
                    pred_candidate_post,
                    z_profile_post_eval,
                    stft_rr_eval,
                    aux_rr_eval,
                    stft_conf_eval,
                    extra,
                ) = adapt_profile_vector_episodic_batches(
                    model,
                    rr_model,
                    test_loader,
                    eval_windows,
                    source_anchor_windows,
                    mode,
                    args,
                    device,
                    conditioning_mode,
                    target_profile_stats_raw,
                    target_profile_stats_norm,
                )
                attn_post = attn_pre
            else:
                p_target, readout_affine, extra = adapt_profile_vector_unsupervised(
                    model,
                    rr_model,
                    test_loader,
                    adapt_windows,
                    source_anchor_windows,
                    mode,
                    args,
                    device,
                    target_profile_stats_raw=target_profile_stats_raw,
                    target_profile_stats_norm=target_profile_stats_norm,
                )
                pred_candidate_post, z_profile_post_eval, stft_rr_eval, aux_rr_eval, stft_conf_eval = _predict_profile_unsup_with_features(
                    model,
                    rr_model,
                    eval_windows,
                    p_target,
                    device,
                    int(args.feature_eval_batch_size),
                    conditioning_mode,
                    readout_affine=readout_affine,
                )
                attn_post = _collect_profile_attention_reference(
                    model,
                    eval_windows,
                    p_target.detach(),
                    device,
                    batch_size=int(args.feature_eval_batch_size),
                    conditioning_mode=conditioning_mode,
                )
        else:
            p_target = nn.Parameter(p0.detach().clone())
            readout_affine = None
            extra = _profile_init_extra(
                model,
                target_profile_stats_raw,
                target_profile_stats_norm,
                p0,
                args,
                mode,
            )
            pred_candidate_post = pred_profile_init
            z_profile_post_eval = z_profile_init_eval
            stft_rr_eval = stft_rr_init_eval
            aux_rr_eval = aux_rr_init_eval
            stft_conf_eval = stft_conf_init_eval
            attn_post = attn_pre

        profile_gate_pass = bool(gate_diag.get("profile_gate_pass", 0.0) >= 0.5)
        oracle_pass = bool(oracle_diag.get("profile_oracle_pass", 0.0) >= 0.5)
        selected = "profile"
        if _profile_mode_gated(mode):
            if profile_gate_pass:
                pred_post = pred_candidate_post
                selected = "profile"
            else:
                pred_post = pred_pre[: y_eval_for_metrics.shape[0]]
                selected = "none_gate_reject"
        elif _profile_mode_oracle(mode):
            if oracle_pass:
                pred_post = pred_candidate_post
                selected = "profile"
            else:
                pred_post = pred_pre[: y_eval_for_metrics.shape[0]]
                selected = "none_oracle_reject"
        elif mode == "profile_film_init_only":
            pred_post = pred_profile_init
            selected = "profile_init_only"
        else:
            pred_post = pred_candidate_post

        rr_base_eval = pred_pre[: y_eval_for_metrics.shape[0]]
        pred_shift = pred_post - rr_base_eval
        rr_range_violation = (
            (pred_post < float(getattr(args, "unsup_rr_min_bpm", 4.0)))
            | (pred_post > float(getattr(args, "unsup_rr_max_bpm", 45.0)))
            | (~np.isfinite(pred_post))
        )
        fallback_triggered, fallback_reason = _budget_fallback_decision(
            args=args,
            p_delta_norm=float(extra.get("profile_delta_norm", 0.0)),
            qkv_delta_norm_p95=float(extra.get("qkv_delta_norm_p95", 0.0)),
            pred_shift_p95_abs_bpm=float(np.percentile(np.abs(pred_shift), 95)) if pred_shift.size else float("nan"),
            rr_range_violation_fraction=float(np.mean(rr_range_violation)) if rr_range_violation.size else float("nan"),
            stft_confidence_mean=float(np.mean(stft_conf_eval)) if stft_conf_eval.size else float("nan"),
        )
        if fallback_triggered:
            pred_post = rr_base_eval
            selected = "none_budget_fallback"
            pred_shift = pred_post - rr_base_eval

        metrics.update(rr_metrics(y_eval_for_metrics, pred_post, prefix="rr_probe_post"))
        metrics.update(rr_metrics(y_eval_for_metrics, pred_profile_init, prefix="rr_probe_profile_init"))
        metrics.update(rr_metrics(y_eval_for_metrics, pred_candidate_post, prefix="rr_probe_profile_candidate"))
        metrics.update(extra)
        metrics.update(gate_diag)
        metrics.update(oracle_diag)

        # Directional-signature diagnostics: analysis-only labels are used here
        # to determine whether profile shifts align with true residuals. These
        # metrics are logged but never used for adaptation or gating.
        rr_base_eval = pred_pre[: y_eval_for_metrics.shape[0]]
        metrics.update(
            _directional_shift_diagnostics(
                y_true=y_eval_for_metrics,
                rr_base=rr_base_eval,
                rr_profile=pred_profile_init,
                rr_post=pred_candidate_post,
                rr_aux=aux_rr_eval,
                rr_stft=stft_rr_eval,
                stft_conf=stft_conf_eval,
                prefix="eval_direction",
            )
        )
        metrics.update(
            _directional_shift_diagnostics(
                y_true=y_adapt,
                rr_base=base_probe_adapt,
                rr_profile=pred_profile_init_adapt,
                rr_post=None,
                rr_aux=aux_rr_init_adapt,
                rr_stft=stft_rr_init_adapt,
                stft_conf=stft_conf_init_adapt,
                prefix="cal_direction",
            )
        )
        rr_weight = _get_rr_probe_weight(rr_model)
        metrics.update(
            _feature_shift_alignment_diagnostics(
                z_base=z_base_eval,
                z_profile=z_profile_init_eval,
                rr_weight=rr_weight,
                prefix="eval_direction_init",
            )
        )
        metrics.update(
            _feature_shift_alignment_diagnostics(
                z_base=z_base_eval,
                z_profile=z_profile_post_eval,
                rr_weight=rr_weight,
                prefix="eval_direction_post",
            )
        )

        metrics.update({
            "rr_tta_mode": mode,
            "is_unsupervised_feature_tta": 1,
            "is_profile_vector_tta": 1,
            "uses_target_rr_labels_for_adaptation": 0,
            "analysis_oracle_uses_target_labels_for_selection": int(_profile_mode_oracle(mode)),
            "profile_selection": selected,
            "profile_selected_profile": int(selected.startswith("profile")),
            "rr_probe_n_source": int(x_source.shape[0]),
            "rr_probe_n_target_total": int(x_target.shape[0]),
            "rr_probe_n_calibration": int(x_cal.shape[0]),
            "rr_probe_n_eval": int(y_eval_for_metrics.shape[0]),
            "rr_probe_n_features": int(x_source.shape[1]),
            "target_calibration_indices": json.dumps(cal_idx.tolist()),
            "eval_excludes_calibration": int(bool(args.exclude_calibration_from_eval)),
            "eval_stft_rr_mae_vs_label": _mae_np(y_eval_for_metrics, stft_rr_eval),
            "eval_aux_rr_mae_vs_label": _mae_np(y_eval_for_metrics, aux_rr_eval),
            "eval_stft_confidence_mean": float(np.mean(stft_conf_eval)) if stft_conf_eval.size else float("nan"),
            "eval_stft_confidence_p05": float(np.percentile(stft_conf_eval, 5)) if stft_conf_eval.size else float("nan"),
            "eval_stft_confidence_p95": float(np.percentile(stft_conf_eval, 95)) if stft_conf_eval.size else float("nan"),
            "profile_init_adapt_mae_vs_label": _mae_np(y_adapt, pred_profile_init_adapt),
            "profile_base_adapt_mae_vs_label": _mae_np(y_adapt, base_probe_adapt),
            "profile_candidate_eval_mae_vs_label": _mae_np(y_eval_for_metrics, pred_candidate_post),
            "eval_probe_pre_matches_tensor_labels": int(y_eval_tensor.shape[0] == y_eval.shape[0]),
            "profile_unsup_adapt_scope": adapt_scope_used,
            "profile_unsup_n_adapt_windows": int(adapt_windows.imu.size(0)),
            "pred_shift_signed_mean_bpm": float(np.mean(pred_shift)) if pred_shift.size else float("nan"),
            "pred_shift_abs_mean_bpm": float(np.mean(np.abs(pred_shift))) if pred_shift.size else float("nan"),
            "pred_shift_p95_abs_bpm": float(np.percentile(np.abs(pred_shift), 95)) if pred_shift.size else float("nan"),
            "stft_confidence_mean": float(np.mean(stft_conf_eval)) if stft_conf_eval.size else float("nan"),
            "stft_confidence_p05": float(np.percentile(stft_conf_eval, 5)) if stft_conf_eval.size else float("nan"),
            "stft_confidence_p95": float(np.percentile(stft_conf_eval, 95)) if stft_conf_eval.size else float("nan"),
            "probe_stft_disagreement_mean_bpm": float(np.mean(np.abs(pred_post - stft_rr_eval))) if stft_rr_eval.size else float("nan"),
            "rr_range_violation_fraction": float(np.mean(rr_range_violation)) if rr_range_violation.size else float("nan"),
            "fallback_triggered": int(fallback_triggered),
            "fallback_reason": fallback_reason,
        })
        for key, value in attn_pre.items():
            metrics[f"attn_distance_pre_{key}"] = float(value)
        for key, value in attn_post.items():
            metrics[f"attn_distance_post_{key}"] = float(value)
            if key in attn_pre:
                metrics[f"attn_distance_delta_{key}"] = float(value - attn_pre[key])
        metrics.update(_profile_clsa_metric_diagnostics([attn_post]))

        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            rr_base_eval = pred_pre[: y_eval_for_metrics.shape[0]]
            profile_shift = pred_profile_init - rr_base_eval
            candidate_shift = pred_candidate_post - rr_base_eval
            selected_shift = pred_post - rr_base_eval
            base_residual = y_eval_for_metrics - rr_base_eval
            profile_residual = y_eval_for_metrics - pred_profile_init
            candidate_residual = y_eval_for_metrics - pred_candidate_post
            selected_residual = y_eval_for_metrics - pred_post
            aux_minus_base = aux_rr_eval - rr_base_eval
            stft_minus_base = stft_rr_eval - rr_base_eval
            weighted_physio_minus_base = 0.7 * aux_minus_base + 0.3 * stft_minus_base

            pd.DataFrame({
                "subject": np.full(y_eval_for_metrics.shape[0], subject),
                "window_idx_eval": np.arange(y_eval_for_metrics.shape[0]),
                "rr_true": y_eval_for_metrics,
                "rr_pred_base": rr_base_eval,
                "rr_pred_pre": rr_base_eval,
                "rr_pred_profile_init": pred_profile_init,
                "rr_pred_profile_candidate_post": pred_candidate_post,
                "rr_pred_post": pred_post,
                "rr_stft_post": stft_rr_eval,
                "rr_aux_head_post": aux_rr_eval,
                "rr_stft_confidence": stft_conf_eval,
                "profile_selected_profile": np.full(y_eval_for_metrics.shape[0], int(selected.startswith("profile")), dtype=np.int64),
                "profile_shift_signed": profile_shift,
                "candidate_shift_signed": candidate_shift,
                "selected_shift_signed": selected_shift,
                "base_residual_signed": base_residual,
                "profile_residual_signed": profile_residual,
                "candidate_residual_signed": candidate_residual,
                "selected_residual_signed": selected_residual,
                "profile_direction_dot": profile_shift * base_residual,
                "candidate_direction_dot": candidate_shift * base_residual,
                "selected_direction_dot": selected_shift * base_residual,
                "profile_direction_agree": (np.sign(profile_shift) == np.sign(base_residual)).astype(int),
                "candidate_direction_agree": (np.sign(candidate_shift) == np.sign(base_residual)).astype(int),
                "selected_direction_agree": (np.sign(selected_shift) == np.sign(base_residual)).astype(int),
                "profile_overshoot": (np.abs(profile_shift) > np.abs(base_residual)).astype(int),
                "candidate_overshoot": (np.abs(candidate_shift) > np.abs(base_residual)).astype(int),
                "selected_overshoot": (np.abs(selected_shift) > np.abs(base_residual)).astype(int),
                "aux_minus_base": aux_minus_base,
                "stft_minus_base": stft_minus_base,
                "weighted_physio_minus_base": weighted_physio_minus_base,
                "profile_aux_sign_agree": (np.sign(profile_shift) == np.sign(aux_minus_base)).astype(int),
                "profile_stft_sign_agree": (np.sign(profile_shift) == np.sign(stft_minus_base)).astype(int),
                "profile_weighted_physio_sign_agree": (np.sign(profile_shift) == np.sign(weighted_physio_minus_base)).astype(int),
            }).to_csv(out_dir / f"rr_profile_vector_predictions_{subject}.csv", index=False)
            with open(out_dir / f"rr_profile_vector_metrics_{subject}.json", "w") as f:
                json.dump(metrics, f, indent=2)
        return metrics
    finally:
        _restore_state_dict(model, original_state, device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()


def feature_adaptive_evaluate(
    model: nn.Module,
    train_loader,
    test_loader,
    subject: str,
    device: str,
    args,
    shared_context: Optional[SubjectEvalContext] = None,
    out_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """Evaluate a feature-adaptive mode and restore model weights afterwards."""
    mode = str(args.rr_tta).lower()
    if mode in READOUT_ABLATION_MODES:
        return readout_ablation_evaluate(
            model, train_loader, test_loader, subject, device, args, shared_context=shared_context, out_dir=out_dir
        )
    if mode in PROFILE_VECTOR_UNSUP_MODES:
        return profile_vector_unsupervised_evaluate(
            model, train_loader, test_loader, subject, device, args, shared_context=shared_context, out_dir=out_dir
        )
    if mode in UNSUP_FEATURE_MODES:
        return feature_adaptive_unsupervised_evaluate(
            model, train_loader, test_loader, subject, device, args, shared_context=shared_context, out_dir=out_dir
        )
    if mode not in FEATURE_ADAPTIVE_MODES:
        return rr_structured_adaptation_evaluate(
            model, train_loader, test_loader, subject, device, args, shared_context=shared_context, out_dir=out_dir
        )

    _warm_lazy_modules(model, train_loader, device)
    original_state = _state_dict_cpu_clone(model)

    try:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        # Frozen source/target features for source RR probe training and baseline metrics.
        ctx = shared_context or _build_subject_eval_context(
            model, train_loader, test_loader, device, args, include_kinematics=False
        )
        x_source, y_source = ctx.x_source, ctx.y_source
        x_target, y_target = ctx.x_target, ctx.y_target
        x_cal, y_cal, x_eval, y_eval, cal_idx = ctx.x_cal, ctx.y_cal, ctx.x_eval, ctx.y_eval, ctx.cal_idx
        rr_model = FaithfulRRRegressor(x_source.shape[1]).to(device)
        rr_model.load_state_dict(ctx.rr_model_state, strict=True)
        _set_rr_model_frozen(rr_model)

        pred_pre = predict_rr(rr_model, x_eval, device)
        metrics: Dict[str, float] = rr_metrics(y_eval, pred_pre, prefix="rr_probe_pre")
        _attach_base_pre_aliases(metrics)

        # Ordered raw tensors for feature adaptation/evaluation.
        target_windows = ctx.target_windows
        cal_windows = target_windows.subset(cal_idx)
        eval_windows = ctx.eval_windows
        source_anchor_windows = ctx.source_anchor_windows

        affine, extra = adapt_feature_space_with_affine(
            model,
            rr_model,
            cal_windows,
            source_anchor_windows,
            mode,
            args,
            device,
        )
        pred_post, stft_rr_eval, aux_rr_eval = _predict_feature_adapted(
            model,
            rr_model,
            affine,
            eval_windows,
            device,
            int(args.feature_eval_batch_size),
        )
        y_eval_tensor = eval_windows.rr.detach().cpu().numpy().reshape(-1)
        # Guard against ordering mismatch between feature arrays and tensor windows.
        if y_eval_tensor.shape[0] == y_eval.shape[0]:
            y_eval_for_metrics = y_eval_tensor.astype(np.float32)
        else:
            y_eval_for_metrics = y_eval.astype(np.float32)

        metrics.update(rr_metrics(y_eval_for_metrics, pred_post, prefix="rr_probe_post"))
        metrics.update(extra)
        metrics.update({
            "rr_tta_mode": mode,
            "rr_probe_n_source": int(x_source.shape[0]),
            "rr_probe_n_target_total": int(x_target.shape[0]),
            "rr_probe_n_calibration": int(x_cal.shape[0]),
            "rr_probe_n_eval": int(y_eval_for_metrics.shape[0]),
            "rr_probe_n_features": int(x_source.shape[1]),
            "target_calibration_indices": json.dumps(cal_idx.tolist()),
            "eval_excludes_calibration": int(bool(args.exclude_calibration_from_eval)),
            "eval_stft_rr_mae_vs_label": float(np.mean(np.abs(stft_rr_eval - y_eval_for_metrics))),
            "eval_aux_rr_mae_vs_label": float(np.mean(np.abs(aux_rr_eval - y_eval_for_metrics))),
            "eval_probe_pre_matches_tensor_labels": int(y_eval_tensor.shape[0] == y_eval.shape[0]),
        })

        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({
                "rr_true": y_eval_for_metrics,
                "rr_pred_pre": pred_pre[: y_eval_for_metrics.shape[0]],
                "rr_pred_post": pred_post,
                "rr_stft_post": stft_rr_eval,
                "rr_aux_head_post": aux_rr_eval,
            }).to_csv(out_dir / f"rr_feature_adaptive_predictions_{subject}.csv", index=False)
            with open(out_dir / f"rr_feature_adaptive_metrics_{subject}.json", "w") as f:
                json.dump(metrics, f, indent=2)
        return metrics
    finally:
        _restore_state_dict(model, original_state, device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()


def feature_adaptive_hook(model, sbj: str, train_loader, test_loader, device: str, args, sbj_dir: Path):
    metrics = feature_adaptive_evaluate(
        model,
        train_loader,
        test_loader,
        sbj,
        device,
        args,
        out_dir=sbj_dir / "rr_feature_adaptive",
    )
    print(f"RR_FEATURE_ADAPTIVE {sbj}: {metrics}")
    # Keep the original summary name so existing summarizers still work.
    return {"__summary_name__": "rr_structured_adaptation_summary", "__summary_row__": {"subject": sbj, **metrics}}



# ---------------------------------------------------------------------------
# Efficient multi-mode same-checkpoint configuration sweep
# ---------------------------------------------------------------------------

DEFAULT_SWEEP_MODES = [
    "none",
    "feature_mean_align_alpha075",
    "feature_mean_align_alpha050",
    "feature_mean_align_alpha100",
    "feature_mean_align_profile_shrink",
    "adapt_mean_alpha_050",
    "adapt_mean_alpha_100",
    "adapt_mean_profile_shrink",
    "profile_film_init_only",
    "profile_film_unsup_sparc",
    "direct_stft_rr",
    "hybrid_probe_stft_conf",
]

FEATURE_MODES = FEATURE_ADAPTIVE_MODES


def _parse_modes(text: str) -> List[str]:
    toks: List[str] = []
    for part in str(text).replace(",", " ").split():
        part = part.strip()
        if part:
            toks.append(part)
    if not toks:
        raise ValueError("--rr-tta-modes did not contain any modes")
    return toks


def _apply_mode_defaults(args, mode: str):
    """Return a shallow args copy with safe per-mode defaults applied."""
    out = copy.copy(args)
    out.rr_tta = mode

    if not bool(getattr(args, "apply_mode_defaults", True)):
        return out

    # These match the intended direction-finding defaults from the mode-specific
    # bash branches, but are now applied inside the single-checkpoint sweep.
    if mode == "ln_affine_cal":
        out.feature_stft_consistency_weight = 0.0
        out.feature_smoothness_weight = 0.0
        out.feature_source_anchor_weight = 0.01
        out.feature_target_drift_weight = 0.01
    elif mode == "ln_stft_rr_consistency":
        out.feature_stft_consistency_weight = 0.05
        out.feature_smoothness_weight = 0.0
        out.feature_source_anchor_weight = 0.01
        out.feature_target_drift_weight = 0.01
    elif mode in LASTBLOCK_FEATURE_MODES:
        out.feature_stft_consistency_weight = 0.05
        out.feature_smoothness_weight = 0.01
        out.feature_source_anchor_weight = 0.05
        out.feature_target_drift_weight = 0.05
        out.feature_tta_lr = 1e-5
    elif mode == "rrbin_ssa":
        out.rrbin_use_calibration_only = True

    return out


def _flatten_for_csv(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, (list, tuple, dict)):
            out[k] = json.dumps(v)
        else:
            out[k] = v
    return out


def _write_summary_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = _flatten_for_csv(row)
    # Each chunk path is already unique per (mode, run_id). Re-running the same
    # subject should replace the prior row rather than append stale rows from
    # older executions, otherwise the aggregate CSV can disagree with current
    # per-subject logs.
    pd.DataFrame([row], columns=list(row.keys())).to_csv(path, index=False)


def rr_config_sweep_hook(model, sbj: str, train_loader, test_loader, device: str, args, sbj_dir: Path):
    modes = _parse_modes(args.rr_tta_modes)
    sweep_root = Path(args.sweep_root) if args.sweep_root else Path(args.out_dir)
    run_id = str(args.sweep_run_id or "single")
    include_kinematics = any(str(mode).lower() in {"kin_ssa", "ssa_kin"} for mode in modes)
    shared_context = _build_subject_eval_context(
        model,
        train_loader,
        test_loader,
        device,
        args,
        include_kinematics=include_kinematics,
    )

    hook_rows = []
    for mode in modes:
        mode = str(mode).strip()
        if not mode:
            continue

        mode_args = _args_for_rr_tta_mode(args, mode)
        mode_out_dir = sweep_root / mode / "subjects" / sbj
        print(f"[CONFIG_SWEEP] subject={sbj} mode={mode} out={mode_out_dir}")
        metrics = feature_adaptive_evaluate(
            model,
            train_loader,
            test_loader,
            sbj,
            device,
            mode_args,
            shared_context=shared_context,
            out_dir=mode_out_dir / "rr_feature_adaptive",
        )
        metrics.update({
            "rr_tta_mode": mode,
            "unsup_mode_defaults_applied": int(getattr(mode_args, "unsup_mode_defaults_applied", 0)),
            "effective_unsup_stft_consistency_weight": float(getattr(mode_args, "unsup_stft_consistency_weight", 0.0)),
            "effective_uses_stft_pseudotarget_for_tta": int(float(getattr(mode_args, "unsup_stft_consistency_weight", 0.0)) > 0.0),
            "effective_unsup_smoothness_weight": float(getattr(mode_args, "unsup_smoothness_weight", 0.0)),
            "effective_unsup_source_anchor_weight": float(getattr(mode_args, "unsup_source_anchor_weight", 0.0)),
            "effective_unsup_target_drift_weight": float(getattr(mode_args, "unsup_target_drift_weight", 0.0)),
            "effective_unsup_profile_prior_weight": float(getattr(mode_args, "unsup_profile_prior_weight", 0.0)),
            "effective_unsup_attention_profile_weight": float(getattr(mode_args, "unsup_attention_profile_weight", 0.0)),
            "effective_unsup_feature_lr": float(getattr(mode_args, "unsup_feature_lr", 0.0)),
            "effective_profile_safety_budget_use_stft_confidence": int(bool(getattr(mode_args, "profile_safety_budget_use_stft_confidence", True))),
            "effective_profile_qkv_mode": str(getattr(mode_args, "profile_qkv_mode", "static")),
            "effective_profile_clsa_rank": int(getattr(mode_args, "profile_clsa_rank", 8)),
            "effective_profile_clsa_scale": float(getattr(mode_args, "profile_clsa_scale", 0.01)),
            "effective_profile_clsa_enable_fast_update": int(bool(int(getattr(mode_args, "profile_clsa_enable_fast_update", 1)))),
        })
        row = {"subject": sbj, **metrics}
        chunk_summary = sweep_root / mode / "chunks" / f"{run_id}_rr_structured_adaptation_summary.csv"
        _write_summary_row(chunk_summary, row)
        hook_rows.append({"mode": mode, **row})
        print(f"RR_CONFIG_SWEEP subject={sbj} mode={mode}: {metrics}")

    # Return a compact row for the core runner's own summary. The real per-mode
    # summaries are written above.
    return {
        "__summary_name__": "rr_config_sweep_summary",
        "__summary_row__": {
            "subject": sbj,
            "n_modes": len(modes),
            "modes": " ".join(modes),
            "sweep_root": str(sweep_root),
            "sweep_run_id": run_id,
        },
    }


def add_common_adaptation_args(parser) -> None:
    # Source RR probe.
    parser.add_argument("--rr-probe-epochs", type=int, default=100)
    parser.add_argument("--rr-probe-lr", type=float, default=1e-3)
    parser.add_argument("--rr-probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--rr-probe-batch-size", type=int, default=256)
    parser.add_argument("--rr-probe-source-batches", type=int, default=0)
    parser.add_argument("--rr-probe-train-adapter", action="store_true")

    # Clean calibration/evaluation split.
    parser.add_argument("--target-calibration-windows", type=int, default=32)
    parser.add_argument("--target-calibration-mode", choices=["first", "random", "even"], default="first")
    parser.add_argument("--exclude-calibration-from-eval", action="store_true", default=True)
    parser.add_argument("--include-calibration-in-eval", dest="exclude_calibration_from_eval", action="store_false")

    parser.add_argument(
        "--rr-tta",
        default="none",
        choices=[
            "none",
            "affine_cal",
            "affine_cal_ridge",
            "affine_cal_monotone",
            "ridge_residual_cal",
            "rrbin_centroid_cal",
            "rrbin_ssa",
            "feature_mean_align_alpha050",
            "feature_mean_align_alpha075",
            "feature_mean_align_alpha100",
            "feature_mean_align_profile_shrink",
            "adapt_mean_alpha_000",
            "adapt_mean_alpha_025",
            "adapt_mean_alpha_050",
            "adapt_mean_alpha_075",
            "adapt_mean_alpha_100",
            "adapt_mean_fixed",
            "adapt_mean_profile_shrink",
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
            "profile_qkv_unsup_stft_consistency",
            "profile_qkv_unsup_stft_smooth_anchor",
            "profile_qkv_unsup_stft_smooth_prior_attn",
            "profile_film_unsup_readout_affine",
            "profile_film_unsup_sparc",
            "profile_film_gated_sparc",
            "profile_film_oracle_sparc",
            "tcn_profile_film_qkv_last1_0p01",
            "tcn_profile_film_qkv_last1_0p01_sparc_pt",
            "tcn_profile_film_qkv_last1_0p01_sparc_pt_budget",
            "tcn_profile_film_qkv_last1_0p01_pt_no_stft",
            "tcn_profile_film_qkv_last1_0p01_pt_aux_only",
            "tcn_profile_film_qkv_last1_0p01_pt_reg_only",
            "tcn_profile_film_qkv_last1_0p01_pt_no_stft_budget",
            "tcn_profile_film_clsa_qkv_last1",
            "tcn_profile_film_clsa_qkv_last1_no_fast_update",
            "tcn_clsa_qkv_last1_no_film",
            "tcn_profile_film_clsa_qkv_last1_rank4",
            "direct_stft_rr",
            "hybrid_probe_stft_conf",
            "ssa",
            "cmt",
            "ssa_cmt",
            "kin_ssa",
            "ssa_kin",
        ],
        help="Single-mode compatibility option. The efficient sweep uses --rr-tta-modes.",
    )

    # Structured sweep options retained for delegated modes.
    parser.add_argument("--affine-lambda-a", type=float, default=0.0)
    parser.add_argument("--affine-lambda-b", type=float, default=0.0)
    parser.add_argument("--affine-ridge-lambda-a", type=float, default=8.0)
    parser.add_argument("--affine-ridge-lambda-b", type=float, default=1.0)
    parser.add_argument("--affine-min-slope", type=float, default=0.05)
    parser.add_argument("--residual-ridge-lambda", type=float, default=10.0)
    parser.add_argument("--rrbin-n-bins", type=int, default=3)
    parser.add_argument("--rrbin-min-bin-count", type=int, default=4)
    parser.add_argument("--rrbin-shrink", type=float, default=8.0)
    parser.add_argument("--rrbin-use-calibration-only", action="store_true", default=True)
    parser.add_argument("--rrbin-use-all-target-unlabeled", dest="rrbin_use_calibration_only", action="store_false")

    # Feature-adaptive options.
    parser.add_argument("--feature-tta-epochs", type=int, default=10)
    parser.add_argument("--feature-tta-lr", type=float, default=3e-5)
    parser.add_argument("--feature-tta-weight-decay", type=float, default=0.0)
    parser.add_argument("--feature-tta-batch-size", type=int, default=32)
    parser.add_argument("--feature-eval-batch-size", type=int, default=256)
    parser.add_argument("--feature-affine-lambda-a", type=float, default=8.0)
    parser.add_argument("--feature-affine-lambda-b", type=float, default=1.0)
    parser.add_argument("--feature-source-anchor-windows", type=int, default=512)
    parser.add_argument("--feature-source-anchor-batch-size", type=int, default=128)
    parser.add_argument("--feature-source-anchor-weight", type=float, default=0.01)
    parser.add_argument("--feature-target-drift-weight", type=float, default=0.01)
    parser.add_argument("--feature-stft-consistency-weight", type=float, default=0.0)
    parser.add_argument("--feature-smoothness-weight", type=float, default=0.0)

    # Unsupervised feature-adaptive ladder options. These modes never use target
    # RR labels for adaptation; labels are used only for evaluation metrics.
    parser.add_argument("--unsup-feature-epochs", type=int, default=10)
    parser.add_argument("--unsup-feature-lr", type=float, default=1e-5)
    parser.add_argument("--unsup-feature-weight-decay", type=float, default=0.0)
    parser.add_argument("--unsup-feature-batch-size", type=int, default=32)
    parser.add_argument("--unsup-source-anchor-batch-size", type=int, default=128)
    parser.add_argument("--unsup-stft-consistency-weight", type=float, default=0.05)
    parser.add_argument("--unsup-smoothness-weight", type=float, default=0.0)
    parser.add_argument("--unsup-source-anchor-weight", type=float, default=0.0)
    parser.add_argument("--unsup-target-drift-weight", type=float, default=0.01)
    parser.add_argument("--unsup-profile-prior-weight", type=float, default=0.01)
    parser.add_argument("--unsup-attention-profile-weight", type=float, default=0.0)
    parser.add_argument("--unsup-stft-confidence-floor", type=float, default=0.1)
    parser.add_argument("--unsup-stft-confidence-power", type=float, default=1.0)
    parser.add_argument("--unsup-rr-disagreement-threshold", type=float, default=12.0)
    parser.add_argument("--unsup-rr-min-bpm", type=float, default=4.0)
    parser.add_argument("--unsup-rr-max-bpm", type=float, default=45.0)
    parser.add_argument("--unsup-max-temporal-jump-bpm", type=float, default=10.0)
    parser.add_argument("--qkv-delta-budget-weight", type=float, default=0.0)
    parser.add_argument("--qkv-delta-budget-max", type=float, default=1.0)
    parser.add_argument("--profile-delta-budget-max", type=float, default=3.0)
    parser.add_argument("--pred-shift-budget-bpm", type=float, default=3.0)
    parser.add_argument("--rr-range-violation-budget-frac", type=float, default=0.05)
    parser.add_argument("--budget-stft-confidence-floor", type=float, default=0.005)
    parser.add_argument("--profile-safety-budget", action="store_true", help="Enable unlabeled budget fallback for profile-vector modes.")
    parser.add_argument(
        "--profile-safety-budget-use-stft-confidence",
        action="store_true",
        default=True,
        help="Allow low STFT reconstruction confidence to trigger profile budget fallback.",
    )
    parser.add_argument(
        "--no-profile-safety-budget-use-stft-confidence",
        dest="profile_safety_budget_use_stft_confidence",
        action="store_false",
        help="Disable STFT-confidence veto in profile budget fallback; useful for no-STFT TTA ablations.",
    )
    # Readout ablation / hybrid controls.
    parser.add_argument("--hybrid-stft-confidence-threshold", type=float, default=0.005)
    parser.add_argument("--hybrid-aux-stft-disagreement-bpm", type=float, default=3.0)
    parser.add_argument("--hybrid-probe-stft-disagreement-bpm", type=float, default=4.0)

    # Profile-FiLM safety gate. These checks are unsupervised and use raw
    # reconstruction confidence rather than the clamped confidence used in the
    # training loss.
    parser.add_argument("--profile-gate-max-init-shift-bpm", type=float, default=0.75)
    parser.add_argument("--profile-gate-aux-stft-disagreement-bpm", type=float, default=3.0)
    parser.add_argument("--profile-gate-min-stft-confidence", type=float, default=0.005)
    parser.add_argument("--profile-gate-profile-rms-z-max", type=float, default=3.0)
    parser.add_argument("--profile-gate-profile-max-abs-z", type=float, default=8.0)

    # Analysis-only oracle selector. This records a label-using diagnostic and
    # is not a valid unsupervised deployment policy.
    parser.add_argument("--profile-oracle-cal-tolerance-bpm", type=float, default=0.05)
    parser.add_argument(
        "--log-directional-signatures",
        action="store_true",
        default=True,
        help="Log signed profile-shift, residual-alignment, aux/STFT direction, and RR-probe-alignment diagnostics.",
    )
    _add_argument_if_absent(parser, "--use-profile-film", action="store_true")
    _add_argument_if_absent(parser, "--use-profile-qkv", action="store_true")
    _add_argument_if_absent(parser, "--shared-profile-qkv", action="store_true")
    _add_argument_if_absent(parser, "--use-profile-lora", action="store_true")
    _add_argument_if_absent(parser, "--profile-conditioning", default="none", choices=["none", "film", "qkv", "film_qkv", "lora"])
    _add_argument_if_absent(parser, "--profile-dim", type=int, default=32)
    _add_argument_if_absent(parser, "--profile-stats-dim", type=int, default=0)
    _add_argument_if_absent(parser, "--profile-hidden-dim", type=int, default=128)
    _add_argument_if_absent(parser, "--profile-film-scale", type=float, default=0.1)
    _add_argument_if_absent(
        parser,
        "--profile-film-placement",
        default="token_pooled",
        choices=["token_pooled", "pooled_only", "late_token_only", "residual"],
    )
    _add_argument_if_absent(parser, "--profile-film-residual-alpha", type=float, default=0.1)
    _add_argument_if_absent(parser, "--profile-qkv-scale", type=float, default=0.1)
    _add_argument_if_absent(parser, "--profile-qkv-layers", default="last1", choices=["last1", "last2", "all"])
    _add_argument_if_absent(parser, "--profile-qkv-residual", action="store_true")
    _add_argument_if_absent(parser, "--profile-qkv-mode", default="static", choices=["static", "clsa"])
    _add_argument_if_absent(parser, "--profile-clsa-rank", type=int, default=8)
    _add_argument_if_absent(parser, "--profile-clsa-scale", type=float, default=0.01)
    _add_argument_if_absent(parser, "--profile-clsa-eta-max", type=float, default=0.1)
    _add_argument_if_absent(parser, "--profile-clsa-enable-fast-update", type=int, default=1)
    _add_argument_if_absent(parser, "--profile-clsa-gate-init-bias", type=float, default=-2.0)
    _add_argument_if_absent(parser, "--profile-clsa-loss-weight", type=float, default=0.0)
    _add_argument_if_absent(parser, "--profile-lora-rank", type=int, default=8)
    _add_argument_if_absent(parser, "--profile-lora-scale", type=float, default=0.05)
    _add_argument_if_absent(parser, "--profile-stats-max-batches", type=int, default=50)
    parser.add_argument("--adapt-profile-vector", action="store_true")
    parser.add_argument("--adapt-profile-encoder", action="store_true")
    parser.add_argument("--adapt-profile-film", action="store_true")
    _add_argument_if_absent(parser, "--use-tcn-token-mixer", action="store_true")
    _add_argument_if_absent(parser, "--tcn-mixer-alpha", type=float, default=0.05)
    _add_argument_if_absent(parser, "--tcn-mixer-hidden", type=int, default=32)
    _add_argument_if_absent(parser, "--tcn-mixer-layers", type=int, default=2)
    parser.add_argument(
        "--use-unsup-mode-defaults",
        action="store_true",
        help=(
            "Apply built-in mode-specific weights for the unsupervised ladder. "
            "Needed when --rr-tta-modes evaluates multiple modes inside one process."
        ),
    )

    # SPARC/readout-affine unsupervised controls.
    parser.add_argument(
        "--unsup-readout-affine",
        action="store_true",
        help=(
            "Adapt a scalar final readout y_final=a*y_probe+b together with the "
            "target profile vector. This targets subject offset/gain shift while "
            "keeping encoder, decoder, RR probe, profile encoder, and FiLM frozen."
        ),
    )
    parser.add_argument("--unsup-readout-affine-lambda-a", type=float, default=1.0)
    parser.add_argument("--unsup-readout-affine-lambda-b", type=float, default=0.1)
    parser.add_argument(
        "--unsup-aux-consistency-weight",
        type=float,
        default=0.0,
        help="Optional consistency between final readout RR and the model auxiliary RR head.",
    )
    parser.add_argument(
        "--unsup-rr-range-weight",
        type=float,
        default=0.0,
        help="Penalty against collapse of final RR variance across target windows.",
    )
    parser.add_argument(
        "--unsup-rr-range-min-std",
        type=float,
        default=1.0,
        help="Minimum target-window RR std in bpm before range-collapse penalty activates.",
    )
    parser.add_argument(
        "--profile-unsup-adapt-scope",
        choices=["full", "calibration", "eval"],
        default="full",
        help=(
            "Unlabeled windows used for profile/readout TTA. Use calibration with "
            "--target-calibration-mode random/even to test representative-subject calibration; "
            "use full for transductive subject-level TTA."
        ),
    )
    parser.add_argument(
        "--profile-unsup-episodic-batch",
        action="store_true",
        help="Reset and adapt the target profile vector independently for each evaluation batch.",
    )

    # Source-distribution feature canonicalization. These modes are unsupervised:
    # target labels are used only for evaluation diagnostics.
    parser.add_argument("--feature-align-use-calibration-only", action="store_true", default=True)
    parser.add_argument("--feature-align-use-all-target-unlabeled", dest="feature_align_use_calibration_only", action="store_false")
    parser.add_argument("--feature-align-rrbin-n-bins", type=int, default=3)
    parser.add_argument("--feature-align-min-bin-count", type=int, default=4)
    parser.add_argument("--feature-align-shrink", type=float, default=8.0)
    parser.add_argument("--feature-align-diag-eps", type=float, default=1e-6)
    parser.add_argument("--feature-align-profile-shrink-max-alpha", type=float, default=0.75)

    # Renamed adaptation options. The feature-align flags above remain as aliases.
    parser.add_argument("--adaptation-use-calibration-only", action="store_true", default=True)
    parser.add_argument("--adaptation-use-all-target-unlabeled", dest="adaptation_use_calibration_only", action="store_false")
    parser.add_argument("--adapt-profile-alpha-min", type=float, default=0.0)
    parser.add_argument("--adapt-profile-alpha-max", type=float, default=1.0)
    parser.add_argument("--adapt-profile-alpha-bias", type=float, default=0.0)
    parser.add_argument("--adapt-profile-shift-mid", type=float, default=0.10)
    parser.add_argument("--adapt-profile-shift-scale", type=float, default=0.05)
    parser.add_argument("--adapt-profile-z-safe", type=float, default=3.0)
    parser.add_argument("--adapt-profile-z-scale", type=float, default=1.0)
    parser.add_argument("--adapt-profile-std-ratio-scale", type=float, default=1.0)

    # SSA/CMT passthrough options retained for delegated modes.
    parser.add_argument("--ssa-epochs", type=int, default=20)
    parser.add_argument("--ssa-lr", type=float, default=1e-4)
    parser.add_argument("--ssa-weight-decay", type=float, default=0.0)
    parser.add_argument("--ssa-batch-size", type=int, default=256)
    _add_argument_if_absent(parser, "--ssa-rank", type=int, default=32)
    parser.add_argument("--ssa-weight-bias", type=float, default=0.0)
    parser.add_argument("--ssa-weight-exp", type=float, default=1.0)
    parser.add_argument("--ssa-normalize-weights", action="store_true")
    parser.add_argument("--ssa-use-calibration-only", action="store_true", default=True)
    parser.add_argument("--ssa-use-all-target-unlabeled", dest="ssa_use_calibration_only", action="store_false")

    parser.add_argument("--cmt-augment-size", type=int, default=4096)
    parser.add_argument("--cmt-include-original", action="store_true", default=True)
    parser.add_argument("--cmt-exclude-original", dest="cmt_include_original", action="store_false")
    parser.add_argument("--cmt-aug-max-iter", type=int, default=100)
    parser.add_argument("--cmt-novelty-nu", type=float, default=0.1)
    parser.add_argument("--cmt-flow-epochs", type=int, default=200)
    parser.add_argument("--cmt-flow-lr", type=float, default=1e-3)
    parser.add_argument("--cmt-flow-weight-decay", type=float, default=1e-5)
    parser.add_argument("--cmt-flow-batch-size", type=int, default=256)
    parser.add_argument("--cmt-flow-layers", type=int, default=6)
    parser.add_argument("--cmt-flow-hidden", type=int, default=256)
    parser.add_argument("--cmt-predictor-epochs", type=int, default=200)
    parser.add_argument("--cmt-predictor-lr", type=float, default=1e-3)
    parser.add_argument("--cmt-predictor-weight-decay", type=float, default=1e-4)
    parser.add_argument("--cmt-predictor-batch-size", type=int, default=256)
    parser.add_argument("--cmt-warm-start-source-head", action="store_true", default=True)
    parser.add_argument("--cmt-random-init-head", dest="cmt_warm_start_source_head", action="store_false")
    parser.add_argument("--kin-source-fraction", type=float, default=0.5)

def _patch_loocv_generator_for_eval_subjects(eval_subjects: List[str], full_subjects: List[str]) -> None:
    """Make run_loocv_experiment iterate only eval_subjects while training on full_subjects.

    The core runner historically uses one `subjects` list both as the set of
    held-out folds to iterate over and as the LOSO source cohort passed to the
    dataloader. A subject-level GPU scheduler needs these concepts separated:
    each process should evaluate one held-out subject, but its source train set
    must still be all other subjects from the full cohort.

    This patches the `loocv_generator` symbol used inside run_loocv_experiment
    and also dataloader.loocv_generator as a fallback. If the core runner does
    not use that symbol, this will fail early with the original behavior rather
    than silently producing an empty train cohort.
    """
    import dataloader as _dataloader

    eval_subjects = [str(s) for s in eval_subjects]
    full_subjects = [str(s) for s in full_subjects]
    missing = [s for s in eval_subjects if s not in full_subjects]
    if missing:
        raise SystemExit(
            f"--eval-subjects contains subjects not present in the full --subjects cohort: {missing}. "
            f"Full cohort: {full_subjects}"
        )
    if len(full_subjects) < 2:
        raise SystemExit("Full LOSO cohort must contain at least two subjects.")

    def _subject_loop_generator(
        subjects,
        data_str,
        val_split=0.25,
        batch_size=64,
        shuffle=True,
        drop_last=True,
        data_dir='/scratch/raqchia/',
        mdl_dir='/data/raqchia/',
        autoencoder=None,
        data_group=None,
        include_tlx=False,
        tlx_csv_path=None,
        **kwargs,
    ):
        # `subjects` from the core runner is intentionally ignored for source
        # cohort construction; full_subjects is the scientifically valid LOSO
        # cohort. eval_subjects controls only which held-out folds this process
        # evaluates.
        for sbj in eval_subjects:
            train_loader, val_loader, test_loader = _dataloader.build_loocv_loaders(
                sbj,
                full_subjects,
                data_str,
                val_split=val_split,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                data_dir=data_dir,
                mdl_dir=mdl_dir,
                autoencoder=autoencoder,
                data_group=data_group,
                include_tlx=include_tlx,
                tlx_csv_path=tlx_csv_path,
            )
            yield sbj, train_loader, val_loader, test_loader

    _dataloader.loocv_generator = _subject_loop_generator
    run_loocv_experiment.__globals__["loocv_generator"] = _subject_loop_generator


def main() -> None:
    # When this directional copy is executed directly, make imports inside
    # vit_pressure_crossmodal_profile_encoder resolve the ladder module name to
    # this patched module rather than the unpatched file on disk.
    import sys as _sys
    _sys.modules["vit_pressure_crossmodal_stft_rr_unsup_ladder_config_sweep"] = _sys.modules[__name__]
    from vit_pressure_crossmodal_profile_encoder import run_phase6_profile_encoder_cli

    run_phase6_profile_encoder_cli()


if __name__ == "__main__":
    main()
