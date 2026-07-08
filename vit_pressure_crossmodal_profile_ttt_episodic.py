#!/usr/bin/env python3
"""
Episodic test-time training (TTT) for profile-conditioned IMU->RR/STFT models.

Protocol:
  For each independent eval sample or small eval batch:
    1. Restore the frozen source-trained checkpoint.
    2. Use only that unlabeled input batch to initialize a fresh profile vector.
    3. Optimize a self-supervised loss on that same input batch.
    4. Predict RR for that same input batch.
    5. Discard the adapted profile vector before the next batch.

This is intentionally different from subject-level transductive TTA:
  - no future target windows are used for adapting a current window
  - no adapted state is carried between independent batches
  - target RR labels are never used for adaptation
  - calibration windows can still be excluded from evaluation, but are not used
    to adapt eval windows unless they are part of the current input batch

Recommended smoke run:
  python vit_pressure_crossmodal_profile_ttt_episodic.py \
    --subjects S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29 \
    --eval-subjects S13 \
    --data-str imu_filt \
    --data-dir /projects/BLVMob/imu-rr-seated/Data/ \
    --data-group mr \
    --mdl-dir /projects/BLVMob/imu-rr-seated/models/imu_filt/loocv \
    --out-dir smoke_profile_ttt_qkv_s13 \
    --device cuda:0 \
    --epochs 10 \
    --rr-probe-epochs 30 \
    --ttt-modes none profile_qkv_ttt_sample \
    --ttt-batch-size 1 \
    --ttt-inner-steps 5 \
    --profile-qkv-scale 0.03 \
    --profile-qkv-layers last1

Full profile-QKV episodic TTT:
  python vit_pressure_crossmodal_profile_ttt_episodic.py \
    --subjects S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29 \
    --eval-subjects S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29 \
    --data-str imu_filt \
    --data-dir /projects/BLVMob/imu-rr-seated/Data/ \
    --data-group mr \
    --mdl-dir /projects/BLVMob/imu-rr-seated/models/imu_filt/loocv \
    --out-dir profile_qkv_episodic_ttt \
    --device cuda:0 \
    --epochs 20 \
    --rr-probe-epochs 100 \
    --ttt-modes none profile_qkv_ttt_sample profile_qkv_ttt_batch \
    --ttt-batch-size 1 \
    --ttt-inner-steps 5 \
    --profile-qkv-scale 0.03 \
    --profile-qkv-layers last1

Batch episodic TTT:
  Use --ttt-batch-size 8 or 16. Each batch is still independent:
  the source checkpoint is restored before the next batch.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_pressure_crossmodal_profile_encoder import (
    build_profile_stats,
    estimate_rr_from_predicted_stft,
    normalize_profile_stats,
    profile_stats_dim,
    split_profile_stats,
)

# Prefer the project runner name used by the current phase-6 entrypoint.
try:
    from vit_pressure_crossmodal_stft_rr_rrprobe_tta_main import (
        SUBJECTS,
        FaithfulRRRegressor,
        TrainConfig,
        build_base_parser,
        collect_rr_arrays,
        pooled_features,
        predict_rr,
        rr_metrics,
        run_loocv_experiment,
        split_target_calibration_eval,
        train_source_rr_regressor,
    )
except ImportError:
    from vit_pressure_crossmodal_stft_rr_core import (
        SUBJECTS,
        FaithfulRRRegressor,
        TrainConfig,
        build_base_parser,
        collect_rr_arrays,
        pooled_features,
        predict_rr,
        rr_metrics,
        run_loocv_experiment,
        split_target_calibration_eval,
        train_source_rr_regressor,
    )

from vit_pressure_crossmodal_stft_rr_unsup_ladder_config_sweep import (
    TensorWindows,
    _clone_module_state_cpu,
    _collect_tensor_windows,
    _make_eval_indices,
    _patch_loocv_generator_for_eval_subjects,
    _restore_state_dict,
    _set_rr_model_frozen,
    _state_dict_cpu_clone,
    _warm_lazy_modules,
    add_common_adaptation_args,
)


TTT_PROFILE_MODES = {
    "profile_film_ttt_sample": "film",
    "profile_film_ttt_batch": "film",
    "profile_qkv_ttt_sample": "qkv",
    "profile_qkv_ttt_batch": "qkv",
}

TTT_ALL_MODES = {"none", *TTT_PROFILE_MODES.keys()}


def _parse_modes(text: str) -> List[str]:
    modes = [part.strip() for part in str(text).replace(",", " ").split() if part.strip()]
    if not modes:
        raise ValueError("--ttt-modes did not contain any modes.")
    bad = [m for m in modes if m not in TTT_ALL_MODES]
    if bad:
        raise ValueError(f"Unsupported TTT modes: {bad}. Valid modes: {sorted(TTT_ALL_MODES)}")
    return modes


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.size < 2 or np.std(y_true) <= 1e-8 or np.std(y_pred) <= 1e-8:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _state_dict_cpu_clone(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


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


def _weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    per = F.smooth_l1_loss(pred.reshape(-1), target.reshape(-1), reduction="none")
    w = weight.reshape(-1).detach().clamp_min(0.0)
    return (per * w).sum() / w.sum().clamp_min(1e-8)


def _rr_from_stft_with_confidence(
    pred_logmag: torch.Tensor,
    br_fs: float = 18.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rr, conf = estimate_rr_from_predicted_stft(
        pred_logmag,
        br_fs=float(br_fs),
        return_confidence=True,
    )
    return rr.reshape(-1), conf.reshape(-1)


def _profile_stats_from_unlabeled_batch(
    model: nn.Module,
    imu: torch.Tensor,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Build profile stats from only the current unlabeled test batch."""
    model.eval()
    with torch.no_grad():
        pred_logmag, rr_aux, hidden = model(imu.to(device, non_blocking=True).float())
        z = pooled_features(hidden)
        rr_stft, conf = _rr_from_stft_with_confidence(pred_logmag)
        keep = (
            torch.isfinite(z).all(dim=1)
            & torch.isfinite(rr_aux.reshape(-1))
            & torch.isfinite(rr_stft.reshape(-1))
            & torch.isfinite(conf.reshape(-1))
        )
        if not bool(keep.any()):
            latent_dim = int(getattr(model, "d_model", z.size(-1)))
            stats_raw = torch.zeros(profile_stats_dim(latent_dim), device=device, dtype=torch.float32)
        else:
            stats_raw = build_profile_stats(
                z[keep].detach(),
                rr_aux.reshape(-1)[keep].detach(),
                rr_stft.reshape(-1)[keep].detach(),
                conf.reshape(-1)[keep].detach(),
            )

        stats_norm = _normalize_profile_stats_for_model(model, stats_raw, device)

    diag = {
        "batch_profile_raw_rr_aux_mean": float("nan"),
        "batch_profile_raw_rr_stft_mean": float("nan"),
        "batch_profile_raw_rr_delta_mean": float("nan"),
        "batch_profile_raw_stft_confidence_mean": float("nan"),
    }
    try:
        split = split_profile_stats(stats_raw.detach().cpu().reshape(-1), int(getattr(model, "d_model", 0)))
        diag.update(
            {
                "batch_profile_raw_rr_aux_mean": float(split["rr_aux_mean"].item()),
                "batch_profile_raw_rr_stft_mean": float(split["rr_stft_mean"].item()),
                "batch_profile_raw_rr_delta_mean": float(split["rr_delta_mean"].item()),
                "batch_profile_raw_stft_confidence_mean": float(split["stft_confidence_mean"].item()),
            }
        )
    except Exception:
        pass
    return stats_raw, stats_norm, diag


def _profile_conditioned_rr_probe(
    model: nn.Module,
    rr_model: FaithfulRRRegressor,
    imu: torch.Tensor,
    profile_vector: torch.Tensor,
    device: str,
    conditioning_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return RR-probe, latent z, STFT-RR, auxiliary RR, and confidence."""
    p = profile_vector
    if p.ndim == 1:
        p = p.unsqueeze(0)
    if p.size(0) == 1 and imu.size(0) > 1:
        p = p.expand(imu.size(0), -1)

    pred_logmag, rr_aux, hidden, _ = model.forward_profile_conditioned(
        imu.to(device, non_blocking=True).float(),
        profile_vector=p,
        conditioning_mode=conditioning_mode,
    )
    z = pooled_features(hidden)
    rr_probe, _ = rr_model(z)
    rr_stft, conf = _rr_from_stft_with_confidence(pred_logmag)
    return rr_probe.reshape(-1), z, rr_stft.reshape(-1), rr_aux.reshape(-1), conf.reshape(-1)


@torch.no_grad()
def _base_rr_probe_batch(
    model: nn.Module,
    rr_model: FaithfulRRRegressor,
    imu: torch.Tensor,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    rr_model.eval()
    pred_logmag, rr_aux, hidden = model(imu.to(device, non_blocking=True).float())
    z = pooled_features(hidden)
    rr_probe, _ = rr_model(z)
    rr_stft, conf = _rr_from_stft_with_confidence(pred_logmag)
    return (
        rr_probe.detach().cpu().numpy().reshape(-1),
        rr_stft.detach().cpu().numpy().reshape(-1),
        rr_aux.detach().cpu().numpy().reshape(-1),
        conf.detach().cpu().numpy().reshape(-1),
    )


def _ttt_loss_for_batch(
    rr_probe: torch.Tensor,
    rr_aux: torch.Tensor,
    rr_stft: torch.Tensor,
    conf: torch.Tensor,
    profile_vector: torch.Tensor,
    p0: torch.Tensor,
    args,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    disagreement = (rr_aux.detach() - rr_stft.detach()).abs()
    reliable = torch.isfinite(rr_probe) & torch.isfinite(rr_aux) & torch.isfinite(rr_stft) & torch.isfinite(conf)
    reliable = reliable & (rr_aux >= float(args.ttt_rr_min_bpm)) & (rr_aux <= float(args.ttt_rr_max_bpm))
    reliable = reliable & (rr_stft >= float(args.ttt_rr_min_bpm)) & (rr_stft <= float(args.ttt_rr_max_bpm))
    reliable = reliable & (disagreement <= float(args.ttt_rr_disagreement_threshold))

    if rr_aux.numel() > 1 and float(args.ttt_max_temporal_jump_bpm) > 0:
        jump_ok = torch.ones_like(reliable, dtype=torch.bool)
        jump_ok[1:] = (rr_aux[1:] - rr_aux[:-1].detach()).abs() <= float(args.ttt_max_temporal_jump_bpm)
        reliable = reliable & jump_ok

    conf_w = conf.detach().clamp(float(args.ttt_stft_confidence_floor), 1.0).pow(
        float(args.ttt_stft_confidence_power)
    )
    conf_w = conf_w * reliable.float()

    loss_probe_stft = _weighted_smooth_l1(rr_probe, rr_stft.detach(), conf_w)
    loss_aux_stft = _weighted_smooth_l1(rr_aux, rr_stft.detach(), conf_w)
    profile_prior = F.mse_loss(profile_vector, p0.detach())

    if rr_probe.numel() > 1:
        smooth_mask = (reliable[1:] & reliable[:-1]).float()
        smooth_per = F.smooth_l1_loss(rr_probe[1:], rr_probe[:-1].detach(), reduction="none")
        smoothness = (smooth_per * smooth_mask).sum() / smooth_mask.sum().clamp_min(1.0)
    else:
        smoothness = rr_probe.new_tensor(0.0)

    loss = (
        float(args.ttt_probe_stft_weight) * loss_probe_stft
        + float(args.ttt_aux_stft_weight) * loss_aux_stft
        + float(args.ttt_profile_prior_weight) * profile_prior
        + float(args.ttt_smoothness_weight) * smoothness
    )

    parts = {
        "ttt_loss": float(loss.detach().cpu()),
        "ttt_probe_stft_loss": float(loss_probe_stft.detach().cpu()),
        "ttt_aux_stft_loss": float(loss_aux_stft.detach().cpu()),
        "ttt_profile_prior_loss": float(profile_prior.detach().cpu()),
        "ttt_smoothness_loss": float(smoothness.detach().cpu()),
        "ttt_reliable_ratio": float(reliable.float().mean().detach().cpu()),
        "ttt_confidence_mean": float(conf.detach().mean().cpu()),
        "ttt_rr_disagreement_mean": float(disagreement.detach().mean().cpu()),
    }
    return loss, parts


def _run_one_episodic_ttt_batch(
    model: nn.Module,
    rr_model: FaithfulRRRegressor,
    imu: torch.Tensor,
    device: str,
    conditioning_mode: str,
    args,
) -> Dict[str, object]:
    """Adapt a fresh profile vector on this batch only, then predict this same batch."""
    if getattr(model, "profile_encoder", None) is None:
        raise RuntimeError("Profile TTT requires model.profile_encoder.")

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    rr_model.eval()

    stats_raw, stats_norm, stats_diag = _profile_stats_from_unlabeled_batch(model, imu, device)

    with torch.no_grad():
        p0 = model.profile_encoder(stats_norm.unsqueeze(0)).detach()
        rr_init, _z_init, rr_stft_init, rr_aux_init, conf_init = _profile_conditioned_rr_probe(
            model,
            rr_model,
            imu,
            p0,
            device,
            conditioning_mode,
        )

    p_ttt = nn.Parameter(p0.detach().clone())
    opt = torch.optim.AdamW(
        [p_ttt],
        lr=float(args.ttt_lr),
        weight_decay=float(args.ttt_weight_decay),
    )

    last_parts: Dict[str, float] = {}
    for _step in range(1, int(args.ttt_inner_steps) + 1):
        opt.zero_grad(set_to_none=True)
        rr_probe, _z, rr_stft, rr_aux, conf = _profile_conditioned_rr_probe(
            model,
            rr_model,
            imu,
            p_ttt,
            device,
            conditioning_mode,
        )
        loss, parts = _ttt_loss_for_batch(rr_probe, rr_aux, rr_stft, conf, p_ttt, p0, args)
        if not torch.isfinite(loss) or not loss.requires_grad:
            last_parts = parts
            break
        loss.backward()
        if float(args.grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_([p_ttt], float(args.grad_clip))
        opt.step()
        last_parts = parts

    with torch.no_grad():
        rr_post, _z_post, rr_stft_post, rr_aux_post, conf_post = _profile_conditioned_rr_probe(
            model,
            rr_model,
            imu,
            p_ttt.detach(),
            device,
            conditioning_mode,
        )

    out = {
        "rr_profile_init": rr_init.detach().cpu().numpy().reshape(-1),
        "rr_pred_post": rr_post.detach().cpu().numpy().reshape(-1),
        "rr_stft_post": rr_stft_post.detach().cpu().numpy().reshape(-1),
        "rr_aux_post": rr_aux_post.detach().cpu().numpy().reshape(-1),
        "rr_conf_post": conf_post.detach().cpu().numpy().reshape(-1),
        "profile_norm_init": float(p0.detach().norm(p=2).cpu()),
        "profile_norm_post": float(p_ttt.detach().norm(p=2).cpu()),
        "profile_delta_norm": float((p_ttt.detach() - p0.detach()).norm(p=2).cpu()),
        **stats_diag,
        **last_parts,
    }
    return out


def evaluate_episodic_ttt_mode(
    model: nn.Module,
    rr_model: FaithfulRRRegressor,
    eval_windows: TensorWindows,
    y_eval: np.ndarray,
    mode: str,
    device: str,
    args,
    out_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """Run strict episodic TTT over eval windows."""
    mode = str(mode).lower().strip()
    conditioning_mode = TTT_PROFILE_MODES.get(mode, "none")
    original_state = _state_dict_cpu_clone(model)

    y_true_all: List[np.ndarray] = []
    pred_pre_all: List[np.ndarray] = []
    pred_init_all: List[np.ndarray] = []
    pred_post_all: List[np.ndarray] = []
    stft_post_all: List[np.ndarray] = []
    aux_post_all: List[np.ndarray] = []
    conf_post_all: List[np.ndarray] = []

    batch_rows: List[Dict[str, float]] = []
    sample_rows: List[Dict[str, float]] = []

    batch_size = 1 if mode.endswith("_sample") else int(args.ttt_batch_size)
    batch_size = max(1, int(batch_size))

    for batch_idx, st in enumerate(range(0, int(eval_windows.imu.size(0)), batch_size)):
        end = min(int(eval_windows.imu.size(0)), st + batch_size)

        # This is the core TTT reset: every independent sample/batch starts from
        # the source-trained checkpoint state.
        _restore_state_dict(model, original_state, device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        imu = eval_windows.imu[st:end].to(device, non_blocking=True).float()
        y_true = eval_windows.rr[st:end].detach().cpu().numpy().reshape(-1).astype(np.float32)

        rr_pre, stft_pre, aux_pre, conf_pre = _base_rr_probe_batch(model, rr_model, imu, device)

        if mode == "none":
            rr_init = rr_pre.copy()
            rr_post = rr_pre.copy()
            stft_post = stft_pre.copy()
            aux_post = aux_pre.copy()
            conf_post = conf_pre.copy()
            batch_extra = {
                "profile_norm_init": float("nan"),
                "profile_norm_post": float("nan"),
                "profile_delta_norm": 0.0,
                "ttt_loss": 0.0,
                "ttt_probe_stft_loss": 0.0,
                "ttt_aux_stft_loss": 0.0,
                "ttt_profile_prior_loss": 0.0,
                "ttt_smoothness_loss": 0.0,
                "ttt_reliable_ratio": 1.0,
                "ttt_confidence_mean": float(np.mean(conf_pre)),
                "ttt_rr_disagreement_mean": float(np.mean(np.abs(aux_pre - stft_pre))),
            }
        else:
            result = _run_one_episodic_ttt_batch(
                model,
                rr_model,
                imu,
                device,
                conditioning_mode,
                args,
            )
            rr_init = result["rr_profile_init"]
            rr_post = result["rr_pred_post"]
            stft_post = result["rr_stft_post"]
            aux_post = result["rr_aux_post"]
            conf_post = result["rr_conf_post"]
            batch_extra = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}

        y_true_all.append(y_true)
        pred_pre_all.append(rr_pre)
        pred_init_all.append(rr_init)
        pred_post_all.append(rr_post)
        stft_post_all.append(stft_post)
        aux_post_all.append(aux_post)
        conf_post_all.append(conf_post)

        batch_row = {
            "batch_idx": int(batch_idx),
            "start_idx": int(st),
            "end_idx": int(end),
            "n_batch": int(end - st),
            "batch_true_mean": float(np.mean(y_true)),
            "batch_pre_mae": float(np.mean(np.abs(rr_pre - y_true))),
            "batch_profile_init_mae": float(np.mean(np.abs(rr_init - y_true))),
            "batch_post_mae": float(np.mean(np.abs(rr_post - y_true))),
            **{k: float(v) for k, v in batch_extra.items() if isinstance(v, (int, float, np.floating))},
        }
        batch_rows.append(batch_row)

        for j in range(end - st):
            sample_rows.append(
                {
                    "global_eval_idx": int(st + j),
                    "batch_idx": int(batch_idx),
                    "rr_true": float(y_true[j]),
                    "rr_pred_pre": float(rr_pre[j]),
                    "rr_pred_profile_init": float(rr_init[j]),
                    "rr_pred_post": float(rr_post[j]),
                    "rr_stft_post": float(stft_post[j]),
                    "rr_aux_post": float(aux_post[j]),
                    "rr_stft_confidence": float(conf_post[j]),
                    "abs_err_pre": float(abs(rr_pre[j] - y_true[j])),
                    "abs_err_profile_init": float(abs(rr_init[j] - y_true[j])),
                    "abs_err_post": float(abs(rr_post[j] - y_true[j])),
                }
            )

    # Leave caller's model in clean source state after the whole mode.
    _restore_state_dict(model, original_state, device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    y_true_np = np.concatenate(y_true_all, axis=0)
    pred_pre_np = np.concatenate(pred_pre_all, axis=0)
    pred_init_np = np.concatenate(pred_init_all, axis=0)
    pred_post_np = np.concatenate(pred_post_all, axis=0)
    stft_post_np = np.concatenate(stft_post_all, axis=0)
    aux_post_np = np.concatenate(aux_post_all, axis=0)
    conf_post_np = np.concatenate(conf_post_all, axis=0)

    metrics: Dict[str, float] = {}
    metrics.update(rr_metrics(y_true_np, pred_pre_np, prefix="rr_probe_pre"))
    metrics.update(rr_metrics(y_true_np, pred_init_np, prefix="rr_probe_profile_init"))
    metrics.update(rr_metrics(y_true_np, pred_post_np, prefix="rr_probe_post"))

    batch_df = pd.DataFrame(batch_rows)
    metrics.update(
        {
            "rr_tta_mode": mode,
            "ttt_protocol": "episodic_reset_each_batch",
            "ttt_conditioning_mode": conditioning_mode,
            "ttt_batch_size": int(batch_size),
            "ttt_inner_steps": int(args.ttt_inner_steps),
            "ttt_lr": float(args.ttt_lr),
            "ttt_weight_decay": float(args.ttt_weight_decay),
            "ttt_uses_target_rr_labels_for_adaptation": 0,
            "ttt_uses_future_windows": 0,
            "ttt_carries_state_between_batches": 0,
            "ttt_restores_source_state_each_batch": 1,
            "ttt_adapted_object": "profile_vector_only" if mode != "none" else "none",
            "ttt_probe_stft_weight": float(args.ttt_probe_stft_weight),
            "ttt_aux_stft_weight": float(args.ttt_aux_stft_weight),
            "ttt_profile_prior_weight": float(args.ttt_profile_prior_weight),
            "ttt_smoothness_weight": float(args.ttt_smoothness_weight),
            "ttt_stft_confidence_floor": float(args.ttt_stft_confidence_floor),
            "ttt_stft_confidence_power": float(args.ttt_stft_confidence_power),
            "ttt_rr_disagreement_threshold": float(args.ttt_rr_disagreement_threshold),
            "eval_stft_rr_mae_vs_label": float(np.mean(np.abs(stft_post_np - y_true_np))),
            "eval_aux_rr_mae_vs_label": float(np.mean(np.abs(aux_post_np - y_true_np))),
            "eval_stft_confidence_mean": float(np.mean(conf_post_np)),
            "ttt_mean_profile_delta_norm": float(batch_df["profile_delta_norm"].mean()) if "profile_delta_norm" in batch_df else 0.0,
            "ttt_mean_reliable_ratio": float(batch_df["ttt_reliable_ratio"].mean()) if "ttt_reliable_ratio" in batch_df else float("nan"),
            "ttt_mean_loss": float(batch_df["ttt_loss"].mean()) if "ttt_loss" in batch_df else float("nan"),
            "ttt_n_eval": int(y_true_np.shape[0]),
            "ttt_n_batches": int(len(batch_rows)),
        }
    )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(sample_rows).to_csv(out_dir / "episodic_ttt_predictions.csv", index=False)
        pd.DataFrame(batch_rows).to_csv(out_dir / "episodic_ttt_batches.csv", index=False)
        with open(out_dir / "episodic_ttt_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


def build_subject_eval_context_for_ttt(
    model: nn.Module,
    train_loader,
    test_loader,
    device: str,
    args,
):
    """Build source RR probe and target eval tensors without using target labels for adaptation."""
    x_source, y_source, _ = collect_rr_arrays(
        model,
        train_loader,
        device,
        max_batches=int(args.rr_probe_source_batches),
        include_kinematics=False,
    )
    x_target, y_target, _ = collect_rr_arrays(
        model,
        test_loader,
        device,
        max_batches=0,
        include_kinematics=False,
    )

    x_cal, y_cal, _, x_eval, y_eval, _, cal_idx = split_target_calibration_eval(
        x_target,
        y_target,
        None,
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
    eval_idx = _make_eval_indices(
        int(target_windows.imu.size(0)),
        cal_idx,
        bool(args.exclude_calibration_from_eval),
    )
    eval_windows = target_windows.subset(eval_idx)

    return {
        "x_source": x_source,
        "y_source": y_source,
        "x_target": x_target,
        "y_target": y_target,
        "x_cal": x_cal,
        "y_cal": y_cal,
        "x_eval": x_eval,
        "y_eval": y_eval,
        "cal_idx": cal_idx,
        "rr_model": rr_model,
        "eval_windows": eval_windows,
    }


def episodic_ttt_sweep_hook(model, sbj: str, train_loader, test_loader, device: str, args, sbj_dir: Path):
    _warm_lazy_modules(model, train_loader, device)

    modes = _parse_modes(args.ttt_modes)
    sweep_root = Path(args.sweep_root) if str(args.sweep_root) else Path(args.out_dir)
    run_id = str(args.sweep_run_id or sbj)

    # Build source RR probe once per held-out subject. This uses only source labels.
    ctx = build_subject_eval_context_for_ttt(
        model,
        train_loader,
        test_loader,
        device,
        args,
    )

    rr_model: FaithfulRRRegressor = ctx["rr_model"]
    eval_windows: TensorWindows = ctx["eval_windows"]
    y_eval = ctx["y_eval"]
    cal_idx = ctx["cal_idx"]

    rows = []
    source_state = _state_dict_cpu_clone(model)

    for mode in modes:
        mode = str(mode).strip().lower()
        _restore_state_dict(model, source_state, device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        mode_dir = sweep_root / mode / "subjects" / sbj / "episodic_ttt"
        print(f"[EPISODIC_TTT] subject={sbj} mode={mode} out={mode_dir}")

        metrics = evaluate_episodic_ttt_mode(
            model=model,
            rr_model=rr_model,
            eval_windows=eval_windows,
            y_eval=y_eval,
            mode=mode,
            device=device,
            args=args,
            out_dir=mode_dir,
        )

        metrics.update(
            {
                "subject": sbj,
                "rr_tta_mode": mode,
                "rr_probe_n_source": int(ctx["x_source"].shape[0]),
                "rr_probe_n_target_total": int(ctx["x_target"].shape[0]),
                "rr_probe_n_calibration": int(ctx["x_cal"].shape[0]),
                "rr_probe_n_eval": int(eval_windows.rr.numel()),
                "rr_probe_n_features": int(ctx["x_source"].shape[1]),
                "target_calibration_indices": json.dumps(cal_idx.tolist()),
                "eval_excludes_calibration": int(bool(args.exclude_calibration_from_eval)),
                "profile_conditioning": str(getattr(model, "profile_conditioning", "none")),
                "profile_qkv_scale": float(getattr(model, "profile_qkv_scale", getattr(args, "profile_qkv_scale", 0.0))),
                "profile_qkv_layers": str(getattr(model, "profile_qkv_layers", getattr(args, "profile_qkv_layers", "none"))),
                "profile_film_scale": float(getattr(model, "profile_film_scale", getattr(args, "profile_film_scale", 0.0))),
            }
        )

        chunk = sweep_root / mode / "chunks" / f"{run_id}_episodic_ttt_summary.csv"
        chunk.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([metrics]).to_csv(chunk, index=False)

        with open(mode_dir / "run_protocol.json", "w") as f:
            json.dump(
                {
                    "protocol": "episodic_reset_each_batch",
                    "mode": mode,
                    "subject": sbj,
                    "uses_target_rr_labels_for_adaptation": False,
                    "uses_future_windows": False,
                    "carries_state_between_batches": False,
                    "restores_source_state_each_batch": True,
                    "adapted_object": "profile_vector_only" if mode != "none" else "none",
                    "ttt_batch_size": int(1 if mode.endswith("_sample") else args.ttt_batch_size),
                    "ttt_inner_steps": int(args.ttt_inner_steps),
                },
                f,
                indent=2,
            )

        rows.append(metrics)
        print(f"EPISODIC_TTT {sbj} {mode}: {metrics}")

    summary_df = pd.DataFrame(rows)
    local_summary = sbj_dir / "episodic_ttt_summary.csv"
    summary_df.to_csv(local_summary, index=False)

    return {
        "__summary_name__": "episodic_ttt_summary",
        "__summary_row__": {
            "subject": sbj,
            "n_modes": int(len(modes)),
            "modes": " ".join(modes),
            "sweep_root": str(sweep_root),
            "sweep_run_id": run_id,
            "ttt_protocol": "episodic_reset_each_batch",
        },
    }


def summarize_sweep_root(root: Path) -> None:
    rows = []
    for path in root.glob("*/chunks/*_episodic_ttt_summary.csv"):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not df.empty:
            rows.append(df)
    if not rows:
        print(f"[SUMMARY] No episodic TTT chunk files found under {root}")
        return

    subject_rows = pd.concat(rows, ignore_index=True)
    subject_rows.to_csv(root / "episodic_ttt_subject_rows.csv", index=False)

    group_cols = ["rr_tta_mode"]
    agg = (
        subject_rows.groupby(group_cols)
        .agg(
            n_subjects=("subject", "nunique"),
            rr_probe_pre_mae_mean=("rr_probe_pre_mae", "mean"),
            rr_probe_profile_init_mae_mean=("rr_probe_profile_init_mae", "mean"),
            rr_probe_post_mae_mean=("rr_probe_post_mae", "mean"),
            rr_probe_post_rmse_mean=("rr_probe_post_rmse", "mean"),
            rr_probe_post_corr_mean=("rr_probe_post_corr", "mean"),
            eval_stft_rr_mae_vs_label_mean=("eval_stft_rr_mae_vs_label", "mean"),
            eval_aux_rr_mae_vs_label_mean=("eval_aux_rr_mae_vs_label", "mean"),
            ttt_mean_profile_delta_norm_mean=("ttt_mean_profile_delta_norm", "mean"),
            ttt_mean_reliable_ratio_mean=("ttt_mean_reliable_ratio", "mean"),
        )
        .reset_index()
    )
    agg["post_minus_pre_mae"] = agg["rr_probe_post_mae_mean"] - agg["rr_probe_pre_mae_mean"]
    agg["post_minus_profile_init_mae"] = (
        agg["rr_probe_post_mae_mean"] - agg["rr_probe_profile_init_mae_mean"]
    )
    agg.to_csv(root / "episodic_ttt_comparison.csv", index=False)

    print("\n=== Episodic TTT comparison ===")
    print(agg.sort_values("rr_probe_post_mae_mean").to_string(index=False))
    print(f"\n[SUMMARY] wrote {root / 'episodic_ttt_subject_rows.csv'}")
    print(f"[SUMMARY] wrote {root / 'episodic_ttt_comparison.csv'}")


def add_ttt_args(parser) -> None:
    parser.add_argument(
        "--ttt-modes",
        default="none profile_film_ttt_sample",
        help=(
            "Space- or comma-separated modes. Valid: "
            "none profile_film_ttt_sample profile_film_ttt_batch "
            "profile_qkv_ttt_sample profile_qkv_ttt_batch"
        ),
    )
    parser.add_argument("--sweep-root", default="", help="Root for per-mode episodic TTT outputs. Defaults to --out-dir.")
    parser.add_argument("--sweep-run-id", default="single", help="Subject/job id for chunk summary filenames.")
    parser.add_argument(
        "--eval-subjects",
        nargs="+",
        default=None,
        help="Held-out subject(s) to evaluate while preserving the full --subjects cohort for LOSO training.",
    )

    parser.add_argument("--ttt-batch-size", type=int, default=1)
    parser.add_argument("--ttt-inner-steps", type=int, default=5)
    parser.add_argument("--ttt-lr", type=float, default=3e-4)
    parser.add_argument("--ttt-weight-decay", type=float, default=0.0)

    parser.add_argument("--ttt-probe-stft-weight", type=float, default=0.05)
    parser.add_argument("--ttt-aux-stft-weight", type=float, default=0.05)
    parser.add_argument("--ttt-profile-prior-weight", type=float, default=0.01)
    parser.add_argument("--ttt-smoothness-weight", type=float, default=0.0)

    parser.add_argument("--ttt-stft-confidence-floor", type=float, default=0.0)
    parser.add_argument("--ttt-stft-confidence-power", type=float, default=2.0)
    parser.add_argument("--ttt-rr-disagreement-threshold", type=float, default=12.0)
    parser.add_argument("--ttt-rr-min-bpm", type=float, default=4.0)
    parser.add_argument("--ttt-rr-max-bpm", type=float, default=45.0)
    parser.add_argument("--ttt-max-temporal-jump-bpm", type=float, default=10.0)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_base_parser(SUBJECTS, "profile_episodic_ttt")
    add_common_adaptation_args(parser)
    add_ttt_args(parser)

    args = parser.parse_args(list(argv) if argv is not None else None)

    modes = _parse_modes(args.ttt_modes)
    has_film = any("profile_film_" in mode for mode in modes)
    has_qkv = any("profile_qkv_" in mode for mode in modes)

    if has_film and has_qkv:
        raise SystemExit(
            "Do not mix profile_film_* and profile_qkv_* TTT modes in the same run. "
            "The source checkpoint trains one profile-conditioning family at a time."
        )

    if has_qkv:
        args.use_profile_qkv = True
        args.use_profile_film = False
        args.profile_conditioning = "qkv"
    elif has_film:
        args.use_profile_film = True
        args.use_profile_qkv = False
        args.profile_conditioning = "film"
    else:
        args.profile_conditioning = "none"

    if has_film or has_qkv:
        if int(getattr(args, "profile_stats_dim", 0)) <= 0:
            args.profile_stats_dim = profile_stats_dim(int(args.d_model))

    full_subjects = list(getattr(args, "subjects", []) or [])
    if len(full_subjects) < 2:
        raise SystemExit(
            "Need at least two subjects in --subjects because it defines the full LOSO source cohort. "
            "For subject-level scheduling, pass held-out subjects via --eval-subjects."
        )

    eval_subjects = list(args.eval_subjects or full_subjects)
    _patch_loocv_generator_for_eval_subjects(eval_subjects, full_subjects)
    args.subjects = eval_subjects

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_root = Path(args.sweep_root) if str(args.sweep_root) else out_dir
    args.sweep_root = str(sweep_root)

    protocol_manifest = {
        "script": "vit_pressure_crossmodal_profile_ttt_episodic.py",
        "ttt_protocol": "episodic_reset_each_batch",
        "modes": modes,
        "full_subjects": full_subjects,
        "eval_subjects": eval_subjects,
        "profile_conditioning": args.profile_conditioning,
        "profile_qkv_scale": float(getattr(args, "profile_qkv_scale", 0.0)),
        "profile_qkv_layers": str(getattr(args, "profile_qkv_layers", "none")),
        "profile_film_scale": float(getattr(args, "profile_film_scale", 0.0)),
        "ttt_batch_size": int(args.ttt_batch_size),
        "ttt_inner_steps": int(args.ttt_inner_steps),
        "ttt_lr": float(args.ttt_lr),
        "uses_target_rr_labels_for_adaptation": False,
        "uses_future_windows": False,
        "carries_state_between_batches": False,
        "restores_source_state_each_batch": True,
        "adapted_object": "profile_vector_only",
    }
    with open(out_dir / "episodic_ttt_protocol_manifest.json", "w") as f:
        json.dump(protocol_manifest, f, indent=2)

    run_loocv_experiment(args, pre_eval_hooks=[episodic_ttt_sweep_hook])
    summarize_sweep_root(sweep_root)


if __name__ == "__main__":
    main()
