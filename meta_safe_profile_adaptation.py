#!/usr/bin/env python3
"""Meta-safe low-capacity subject correction for IMU -> RR.

This experiment tests whether episodic source-subject training can make small
readout/profile corrections safer on held-out subjects. Test-time correction is
label-free: target RR labels are used only for final evaluation, never to decide
whether or how to adapt.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import BR_FS, M_DIR, SBJ_PROCESSED_DIR
from dataloader import LoadDataset, build_loocv_loaders, load_data, make_dataset
from vit_pressure_crossmodal_profile_encoder import (
    build_profile_stats,
    estimate_rr_from_predicted_stft,
    estimate_source_profile_normalizer,
    normalize_profile_stats,
    profile_stats_dim as infer_profile_stats_dim,
)
from vit_pressure_crossmodal_stft_rr_core import (
    TinyIMU2PressureViT,
    augment_imu,
    contrast_weight_for_epoch,
    default_subjects,
    infer_n_channels,
    pressure_stft_recon_loss,
    rr_targets_from_batch,
    set_seed,
    token_contrastive_loss,
    unpack_batch,
)


CORRECTION_MODES = ("none", "readout_affine", "profile_film", "qkv_last1_small")


@dataclass
class CorrectionResult:
    pred: torch.Tensor
    shift: torch.Tensor
    profile_norm: torch.Tensor
    update_norm: torch.Tensor
    fallback_mask: torch.Tensor


class ReadoutAffineCorrection(nn.Module):
    """Global bounded affine readout correction y' = a y + b."""

    def __init__(self, min_slope: float = 0.05):
        super().__init__()
        target_delta = max(1.0 - float(min_slope), 1e-4)
        init = math.log(math.exp(target_delta) - 1.0)
        self.log_slope_delta = nn.Parameter(torch.tensor(init, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(()))
        self.min_slope = float(min_slope)

    def slope(self) -> torch.Tensor:
        return self.min_slope + F.softplus(self.log_slope_delta)

    def forward(self, rr_base: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.slope()
        b = self.bias
        return a * rr_base + b, ((a - 1.0) ** 2 + b**2)

    def regularizer(self, lambda_a: float, lambda_b: float) -> torch.Tensor:
        a = self.slope()
        return float(lambda_a) * (a - 1.0) ** 2 + float(lambda_b) * self.bias**2


def _loader_kwargs(args) -> Dict[str, object]:
    n = max(0, int(args.num_workers))
    out: Dict[str, object] = {
        "num_workers": n,
        "pin_memory": bool(str(args.device).startswith("cuda")),
    }
    if n > 0:
        out["persistent_workers"] = True
        out["prefetch_factor"] = 2
    return out


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size <= 1 or np.std(y_true) <= 1e-8 or np.std(y_pred) <= 1e-8:
        return float("nan")
    return float(np.corrcoef(y_true.reshape(-1), y_pred.reshape(-1))[0, 1])


def finite_window_mask(imu: torch.Tensor, pressure: torch.Tensor, br: Optional[torch.Tensor]) -> torch.Tensor:
    """Keep windows that are finite in model inputs and RR labels."""
    keep = torch.isfinite(imu).flatten(1).all(dim=1)
    keep = keep & torch.isfinite(pressure).flatten(1).all(dim=1)
    if br is not None:
        keep = keep & torch.isfinite(br.reshape(-1)[: imu.size(0)])
    return keep


def filter_valid_windows(
    imu: torch.Tensor,
    pressure: torch.Tensor,
    br: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    keep = finite_window_mask(imu, pressure, br)
    if bool(keep.any()):
        imu = imu[keep]
        pressure = pressure[keep]
        if br is not None:
            br = br.reshape(-1)[: keep.numel()][keep]
    return imu, pressure, br, keep


def snapshot_params(params: Sequence[torch.nn.Parameter]) -> List[Optional[torch.Tensor]]:
    return [p.detach().clone() if p.requires_grad else None for p in params]


@torch.no_grad()
def restore_params(params: Sequence[torch.nn.Parameter], snapshot: Sequence[Optional[torch.Tensor]]) -> None:
    for p, old in zip(params, snapshot):
        if old is not None:
            p.copy_(old)


@torch.no_grad()
def params_are_finite(params: Sequence[torch.nn.Parameter]) -> bool:
    for p in params:
        if p.requires_grad and not bool(torch.isfinite(p).all()):
            return False
    return True


def safe_optimizer_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    params: Sequence[torch.nn.Parameter],
    args,
) -> Tuple[bool, float, str]:
    """Backward/step with guards against one bad batch corrupting the model."""
    if (not torch.isfinite(loss)) or abs(float(loss.detach().cpu())) > float(args.max_loss_value):
        optimizer.zero_grad(set_to_none=True)
        return False, float("nan"), "loss"

    snapshot = snapshot_params(params) if bool(args.rollback_nonfinite_step) else None
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    if float(args.grad_clip) > 0:
        grad_norm_t = torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip))
    else:
        sq = loss.new_tensor(0.0)
        for p in params:
            if p.grad is not None:
                sq = sq + p.grad.detach().float().pow(2).sum()
        grad_norm_t = torch.sqrt(sq)
    grad_norm = float(grad_norm_t.detach().cpu()) if torch.is_tensor(grad_norm_t) else float(grad_norm_t)
    if (not math.isfinite(grad_norm)) or grad_norm > float(args.max_grad_norm):
        if snapshot is not None:
            restore_params(params, snapshot)
        optimizer.zero_grad(set_to_none=True)
        return False, grad_norm, "grad"

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if bool(args.rollback_nonfinite_step) and not params_are_finite(params):
        if snapshot is not None:
            restore_params(params, snapshot)
        optimizer.state.clear()
        return False, grad_norm, "param"

    return True, grad_norm, ""


def build_subject_loader(subjects: Sequence[str], args, *, shuffle: bool, aug_ratio: float = 0.0) -> DataLoader:
    data_list = [load_data(str(s), data_dir=args.data_dir, data_group=args.data_group) for s in subjects]
    x, pressure, br, cond = make_dataset(
        data_list,
        args.data_str,
        label_encoder_dir=args.data_dir,
        data_group=args.data_group,
        is_train=shuffle,
    )
    ds = LoadDataset(x, pressure, cond, br, aug_ratio=aug_ratio, preserve_layout=False)
    return DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=shuffle,
        drop_last=False,
        **_loader_kwargs(args),
    )


def cycle_loader(loader: DataLoader) -> Iterator:
    while True:
        for batch in loader:
            yield batch


def build_model(n_channels: int, args, *, profile_enabled: bool) -> TinyIMU2PressureViT:
    profile_stats_dim = infer_profile_stats_dim(int(args.d_model)) if profile_enabled else 0
    profile_conditioning = "qkv" if profile_enabled else "none"
    return TinyIMU2PressureViT(
        input_channels=n_channels,
        d_model=int(args.d_model),
        pred_len=int(round(20 * BR_FS)),
        nhead=int(args.heads),
        num_layers=int(args.layers),
        decoder_mode=str(args.decoder_mode),
        decoder_layers=int(args.decoder_layers),
        rr_from=str(args.rr_from),
        imu_token_mixer=str(args.imu_token_mixer),
        rr_head_type=str(args.rr_head_type),
        rr_tcn_layers=int(args.rr_tcn_layers),
        rr_tcn_kernel_size=int(args.rr_tcn_kernel_size),
        rr_tcn_dropout=args.rr_tcn_dropout,
        use_profile_qkv=profile_enabled,
        use_profile_film=False,
        profile_conditioning=profile_conditioning,
        profile_dim=int(args.profile_dim),
        profile_stats_dim=profile_stats_dim,
        profile_hidden_dim=int(args.profile_hidden_dim),
        profile_film_scale=float(args.profile_film_scale),
        profile_film_placement=str(args.profile_film_placement),
        profile_film_residual_alpha=float(args.profile_film_residual_alpha),
        profile_qkv_scale=float(args.profile_qkv_scale),
        profile_qkv_layers=str(args.profile_qkv_layers),
        profile_qkv_residual=bool(args.profile_qkv_residual),
    )


def warmup_lazy_modules(model: nn.Module, loader: DataLoader, device: str) -> None:
    model.train()
    batch = next(iter(loader))
    imu, pressure, _, _, _ = unpack_batch(batch, device)
    with torch.no_grad():
        pred_logmag, _rr, hidden = model(imu[:1])
        _ = model.pressure_stft_target(pressure[:1], target_tokens=pred_logmag.size(1))
        _ = model.encode_pressure(pressure[:1], target_tokens=hidden.size(1))


def feature_moment_loss(source_h: torch.Tensor, target_h: torch.Tensor) -> torch.Tensor:
    zs = source_h.mean(dim=1).float()
    zt = target_h.mean(dim=1).float()
    return F.mse_loss(zs.mean(dim=0), zt.mean(dim=0)) + F.mse_loss(
        zs.std(dim=0, unbiased=False),
        zt.std(dim=0, unbiased=False),
    )


def batch_profile_stats(
    model: nn.Module,
    imu: torch.Tensor,
    pred_logmag: torch.Tensor,
    rr_pred: torch.Tensor,
    hidden: torch.Tensor,
    source_profile_mean: Optional[torch.Tensor],
    source_profile_std: Optional[torch.Tensor],
) -> torch.Tensor:
    z = hidden.mean(dim=1)
    rr_stft, stft_conf = estimate_rr_from_predicted_stft(
        pred_logmag,
        br_fs=float(BR_FS),
        return_confidence=True,
    )
    stats = build_profile_stats(
        z.detach(),
        rr_pred.detach().reshape(-1),
        rr_stft.detach().reshape(-1),
        stft_conf.detach().reshape(-1),
    )
    if source_profile_mean is not None and source_profile_std is not None:
        stats = normalize_profile_stats(stats, source_profile_mean, source_profile_std)
    return stats.unsqueeze(0).expand(imu.size(0), -1)


def apply_safety(
    base: torch.Tensor,
    proposed: torch.Tensor,
    profile_norm: torch.Tensor,
    update_norm: torch.Tensor,
    args,
    *,
    max_shift_override: Optional[float] = None,
) -> CorrectionResult:
    base = base.reshape(-1)
    proposed = proposed.reshape(-1)
    shift = proposed - base
    max_shift = float(args.max_rr_shift_bpm)
    if max_shift_override is not None:
        max_shift = min(max_shift, float(max_shift_override))
    max_shift = max(0.0, max_shift)

    if profile_norm.ndim == 0:
        profile_norm = profile_norm.expand_as(base)
    if update_norm.ndim == 0:
        update_norm = update_norm.expand_as(base)

    budget_bad = shift.abs() > max_shift
    budget_bad = budget_bad | (profile_norm.reshape(-1) > float(args.max_profile_norm))
    budget_bad = budget_bad | (update_norm.reshape(-1) > float(args.max_update_norm))

    if str(args.safety_mode) == "clamp":
        pred = base + shift.clamp(-max_shift, max_shift)
        fallback = torch.zeros_like(base, dtype=torch.bool)
        shift = pred - base
    else:
        pred = torch.where(budget_bad, base, proposed)
        fallback = budget_bad
        shift = pred - base

    return CorrectionResult(
        pred=pred,
        shift=shift,
        profile_norm=profile_norm.reshape(-1),
        update_norm=update_norm.reshape(-1),
        fallback_mask=fallback,
    )


def corrected_prediction(
    model: TinyIMU2PressureViT,
    affine: ReadoutAffineCorrection,
    mode: str,
    imu: torch.Tensor,
    base_rr: torch.Tensor,
    base_pred_logmag: torch.Tensor,
    base_hidden: torch.Tensor,
    args,
    source_profile_mean: Optional[torch.Tensor] = None,
    source_profile_std: Optional[torch.Tensor] = None,
) -> CorrectionResult:
    mode = str(mode)
    base_rr = base_rr.reshape(-1)
    zeros = torch.zeros_like(base_rr)

    if mode == "none":
        return CorrectionResult(base_rr, zeros, zeros, zeros, torch.zeros_like(base_rr, dtype=torch.bool))

    if mode == "readout_affine":
        proposed, update_scalar = affine(base_rr)
        profile_norm = torch.zeros_like(base_rr)
        update_norm = torch.sqrt(update_scalar.clamp_min(0.0)).expand_as(base_rr)
        return apply_safety(
            base_rr,
            proposed,
            profile_norm,
            update_norm,
            args,
            max_shift_override=float(args.affine_max_shift_bpm),
        )

    if mode in {"profile_film", "qkv_last1_small"}:
        if not bool(getattr(model, "use_profile_conditioning", False)):
            return CorrectionResult(base_rr, zeros, zeros, zeros, torch.ones_like(base_rr, dtype=torch.bool))
        stats = batch_profile_stats(
            model,
            imu,
            base_pred_logmag,
            base_rr,
            base_hidden,
            source_profile_mean,
            source_profile_std,
        )
        conditioning_mode = "film" if mode == "profile_film" else "qkv"
        _spec, rr_corr, _h_corr, profile_vector = model.forward_profile_conditioned(
            imu,
            profile_stats=stats,
            conditioning_mode=conditioning_mode,
        )
        profile_norm = profile_vector.float().norm(p=2, dim=1)
        profile_scale = min(1.0, float(args.max_profile_norm) / max(float(profile_norm.detach().max().cpu()), 1e-8))
        if profile_scale < 1.0:
            profile_vector = profile_vector * profile_scale
            _spec, rr_corr, _h_corr, profile_vector = model.forward_profile_conditioned(
                imu,
                profile_vector=profile_vector,
                conditioning_mode=conditioning_mode,
            )
            profile_norm = profile_vector.float().norm(p=2, dim=1)
        update_norm = (rr_corr.reshape(-1) - base_rr.reshape(-1)).abs()
        return apply_safety(base_rr, rr_corr.reshape(-1), profile_norm, update_norm, args)

    raise ValueError(f"Unknown correction mode: {mode}")


def source_training_loss(model: TinyIMU2PressureViT, batch, device: str, args, lambda_contrast: float) -> Optional[Tuple[torch.Tensor, Dict[str, float], torch.Tensor]]:
    imu, pressure, _, br, _ = unpack_batch(batch, device)
    imu, pressure, br, keep = filter_valid_windows(imu, pressure, br)
    if imu.size(0) == 0:
        return None
    pred_logmag, rr_pred, hidden = model(imu)
    loss, parts = pressure_stft_recon_loss(
        model,
        pred_logmag,
        pressure,
        rr_pred,
        br,
        lambda_stft=float(args.lambda_stft),
        lambda_rr=float(args.lambda_rr),
    )
    if not torch.isfinite(loss):
        return None
    l_contrast = pred_logmag.new_tensor(0.0)
    if lambda_contrast > 0:
        h_imu = model.encode(augment_imu(imu, shift_max=int(args.shift_max)))
        h_pressure = model.encode_pressure(pressure, target_tokens=h_imu.size(1))
        l_contrast = token_contrastive_loss(h_imu, h_pressure, temperature=float(args.temperature))
        loss = loss + float(lambda_contrast) * l_contrast
    if not torch.isfinite(loss):
        return None
    parts["contrast"] = float(l_contrast.detach().cpu())
    return loss, parts, hidden


def run_meta_episode(
    model: TinyIMU2PressureViT,
    affine: ReadoutAffineCorrection,
    source_batch,
    target_batch,
    mode: str,
    device: str,
    args,
) -> Optional[Tuple[torch.Tensor, Dict[str, float]]]:
    src_imu, _src_pressure, _, _src_br, _ = unpack_batch(source_batch, device)
    tgt_imu, tgt_pressure, _, tgt_br, _ = unpack_batch(target_batch, device)
    src_imu, _src_pressure, _src_br, _src_keep = filter_valid_windows(src_imu, _src_pressure, _src_br)
    tgt_imu, tgt_pressure, tgt_br, _tgt_keep = filter_valid_windows(tgt_imu, tgt_pressure, tgt_br)
    if src_imu.size(0) == 0 or tgt_imu.size(0) == 0:
        return None

    with torch.no_grad():
        _src_spec, _src_rr, src_hidden = model(src_imu)

    base_spec, base_rr, base_hidden = model(tgt_imu)
    rr_true = rr_targets_from_batch(tgt_pressure, tgt_br).reshape(-1)
    corr = corrected_prediction(
        model,
        affine,
        mode,
        tgt_imu,
        base_rr,
        base_spec,
        base_hidden,
        args,
    )
    base_mae = (base_rr.reshape(-1) - rr_true).abs().mean()
    adapt_mae = (corr.pred.reshape(-1) - rr_true).abs().mean()

    l_meta = F.smooth_l1_loss(corr.pred.reshape(-1), rr_true)
    l_harm = torch.relu(adapt_mae - base_mae.detach() + float(args.harm_margin))
    l_update = (corr.update_norm.float() ** 2).mean()
    l_moment = feature_moment_loss(src_hidden.detach(), base_hidden) if bool(args.use_moment_loss) else base_rr.new_tensor(0.0)

    loss = float(args.lambda_meta) * l_meta
    if bool(args.use_meta_no_harm):
        loss = loss + float(args.lambda_harm) * l_harm
    if bool(args.use_moment_loss):
        loss = loss + float(args.lambda_moment) * l_moment
    loss = loss + float(args.lambda_update) * l_update
    if mode == "readout_affine":
        loss = loss + affine.regularizer(float(args.affine_lambda_a), float(args.affine_lambda_b))
    if not torch.isfinite(loss):
        return None

    return loss, {
        "meta_mode": mode,
        "meta_base_mae": float(base_mae.detach().cpu()),
        "meta_adapt_mae": float(adapt_mae.detach().cpu()),
        "meta_no_harm": float(l_harm.detach().cpu()),
        "meta_moment": float(l_moment.detach().cpu()),
        "meta_update": float(l_update.detach().cpu()),
        "meta_fallback_rate": float(corr.fallback_mask.float().mean().detach().cpu()),
    }


def train_fold(
    model: TinyIMU2PressureViT,
    affine: ReadoutAffineCorrection,
    train_loader: DataLoader,
    val_loader: DataLoader,
    source_subject_loaders: Dict[str, DataLoader],
    source_subjects: Sequence[str],
    device: str,
    args,
    sbj_dir: Path,
) -> Tuple[TinyIMU2PressureViT, ReadoutAffineCorrection, List[Dict[str, float]]]:
    params = list(model.parameters()) + list(affine.parameters())
    opt = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_state: Optional[Dict[str, object]] = None
    best_val = float("inf")
    hist: List[Dict[str, float]] = []

    source_iters = {s: cycle_loader(loader) for s, loader in source_subject_loaders.items()}
    train_iter = cycle_loader(train_loader)
    steps_per_epoch = max(1, len(train_loader))
    if bool(args.subject_balanced):
        steps_per_epoch = max(1, sum(len(loader) for loader in source_subject_loaders.values()))
    if int(args.max_train_batches) > 0:
        steps_per_epoch = min(steps_per_epoch, int(args.max_train_batches))

    meta_modes = [m for m in args.correction_modes if m != "none"]
    if not meta_modes:
        meta_modes = ["readout_affine"]

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        affine.train()
        lambda_contrast = contrast_weight_for_epoch(args, epoch)
        totals: Dict[str, List[float]] = {
            "source_loss": [],
            "source_stft": [],
            "source_rr": [],
            "source_contrast": [],
            "meta_loss": [],
            "meta_base_mae": [],
            "meta_adapt_mae": [],
            "meta_no_harm": [],
            "meta_moment": [],
            "meta_update": [],
            "meta_fallback_rate": [],
            "skipped_source_batches": [],
            "skipped_meta_batches": [],
            "rolled_back_steps": [],
            "source_grad_norm": [],
            "meta_grad_norm": [],
        }

        for step in range(steps_per_epoch):
            if bool(args.subject_balanced):
                src_subject = random.choice(list(source_subjects))
                batch = next(source_iters[src_subject])
            else:
                batch = next(train_iter)
            opt.zero_grad(set_to_none=True)
            source_out = source_training_loss(model, batch, device, args, lambda_contrast)
            if source_out is None:
                totals["skipped_source_batches"].append(1.0)
                continue
            loss, parts, _hidden = source_out
            stepped, grad_norm, reason = safe_optimizer_step(loss, opt, params, args)
            if not stepped:
                totals["skipped_source_batches"].append(1.0)
                totals["rolled_back_steps"].append(1.0 if reason == "param" else 0.0)
                continue
            totals["source_loss"].append(float(loss.detach().cpu()))
            totals["source_stft"].append(parts["stft"])
            totals["source_rr"].append(parts["rr"])
            totals["source_contrast"].append(parts["contrast"])
            totals["skipped_source_batches"].append(0.0)
            totals["rolled_back_steps"].append(0.0)
            totals["source_grad_norm"].append(grad_norm)

            run_meta = int(args.meta_every) > 0 and (epoch % int(args.meta_every) == 0)
            if run_meta and step < int(args.meta_batches):
                for _ in range(max(1, int(args.meta_inner_repeats))):
                    pseudo_target = random.choice(list(source_subjects))
                    meta_train_candidates = [s for s in source_subjects if s != pseudo_target]
                    if not meta_train_candidates:
                        continue
                    meta_source = random.choice(meta_train_candidates)
                    mode = random.choice(meta_modes)
                    opt.zero_grad(set_to_none=True)
                    meta_out = run_meta_episode(
                        model,
                        affine,
                        next(source_iters[meta_source]),
                        next(source_iters[pseudo_target]),
                        mode,
                        device,
                        args,
                    )
                    if meta_out is None:
                        totals["skipped_meta_batches"].append(1.0)
                        continue
                    meta_loss, meta_parts = meta_out
                    stepped, grad_norm, reason = safe_optimizer_step(meta_loss, opt, params, args)
                    if not stepped:
                        totals["skipped_meta_batches"].append(1.0)
                        totals["rolled_back_steps"].append(1.0 if reason == "param" else 0.0)
                        continue
                    totals["meta_loss"].append(float(meta_loss.detach().cpu()))
                    totals["skipped_meta_batches"].append(0.0)
                    totals["rolled_back_steps"].append(0.0)
                    totals["meta_grad_norm"].append(grad_norm)
                    for key in (
                        "meta_base_mae",
                        "meta_adapt_mae",
                        "meta_no_harm",
                        "meta_moment",
                        "meta_update",
                        "meta_fallback_rate",
                    ):
                        totals[key].append(meta_parts[key])

        val_metrics = evaluate_mode(
            model,
            affine,
            val_loader,
            "none",
            device,
            args,
            source_profile_mean=None,
            source_profile_std=None,
        )
        row: Dict[str, float] = {"epoch": epoch, "lambda_contrast": float(lambda_contrast)}
        row.update({k: float(np.mean(v)) if v else float("nan") for k, v in totals.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))})
        hist.append(row)
        pd.DataFrame(hist).to_csv(sbj_dir / "history.csv", index=False)
        print(
            f"epoch {epoch:03d} | src {row['source_loss']:.4f} "
            f"meta_harm {row['meta_no_harm']:.4f} meta_adapt {row['meta_adapt_mae']:.3f} "
            f"val_mae {val_metrics['post_mae']:.3f}"
        )

        if val_metrics["post_mae"] < best_val:
            best_val = float(val_metrics["post_mae"])
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "affine": copy.deepcopy(affine.state_dict()),
                "epoch": epoch,
                "best_val": best_val,
            }

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        affine.load_state_dict(best_state["affine"])
        torch.save(best_state, sbj_dir / "best_model.pt")
    torch.save({"model": model.state_dict(), "affine": affine.state_dict(), "args": vars(args)}, sbj_dir / "last_model.pt")
    return model, affine, hist


@torch.no_grad()
def collect_profile_normalizer(model: nn.Module, train_loader: DataLoader, device: str, args) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not bool(getattr(model, "use_profile_conditioning", False)):
        return None, None
    max_batches = int(args.profile_stats_max_batches)
    norm = estimate_source_profile_normalizer(
        model,
        train_loader,
        device,
        max_batches=None if max_batches <= 0 else max_batches,
        br_fs=float(BR_FS),
    )
    return norm["source_profile_mean"].to(device), norm["source_profile_std"].to(device)


@torch.no_grad()
def evaluate_mode(
    model: TinyIMU2PressureViT,
    affine: ReadoutAffineCorrection,
    loader: DataLoader,
    mode: str,
    device: str,
    args,
    source_profile_mean: Optional[torch.Tensor],
    source_profile_std: Optional[torch.Tensor],
) -> Dict[str, float]:
    model.eval()
    affine.eval()
    y_true_all: List[np.ndarray] = []
    pre_all: List[np.ndarray] = []
    post_all: List[np.ndarray] = []
    shifts: List[np.ndarray] = []
    profile_norms: List[np.ndarray] = []
    update_norms: List[np.ndarray] = []
    fallback: List[np.ndarray] = []

    for batch_idx, batch in enumerate(loader):
        if int(args.max_eval_batches) > 0 and batch_idx >= int(args.max_eval_batches):
            break
        imu, pressure, _, br, _ = unpack_batch(batch, device)
        imu, pressure, br, keep = filter_valid_windows(imu, pressure, br)
        if imu.size(0) == 0:
            continue
        pred_logmag, base_rr, hidden = model(imu)
        rr_true = rr_targets_from_batch(pressure, br).reshape(-1)
        corr = corrected_prediction(
            model,
            affine,
            mode,
            imu,
            base_rr,
            pred_logmag,
            hidden,
            args,
            source_profile_mean=source_profile_mean,
            source_profile_std=source_profile_std,
        )
        finite = (
            torch.isfinite(rr_true)
            & torch.isfinite(base_rr.reshape(-1))
            & torch.isfinite(corr.pred.reshape(-1))
            & torch.isfinite(corr.shift.reshape(-1))
        )
        if not bool(finite.any()):
            continue
        y_true_all.append(rr_true[finite].detach().cpu().numpy())
        pre_all.append(base_rr.reshape(-1)[finite].detach().cpu().numpy())
        post_all.append(corr.pred.reshape(-1)[finite].detach().cpu().numpy())
        shifts.append(corr.shift.reshape(-1)[finite].detach().cpu().numpy())
        profile_norms.append(corr.profile_norm.reshape(-1)[finite].detach().cpu().numpy())
        update_norms.append(corr.update_norm.reshape(-1)[finite].detach().cpu().numpy())
        fallback.append(corr.fallback_mask.reshape(-1)[finite].detach().cpu().numpy().astype(np.float32))

    if not y_true_all:
        return {
            "pre_mae": float("nan"),
            "post_mae": float("nan"),
            "rmse": float("nan"),
            "corr": float("nan"),
            "n_windows": 0,
            "mean_abs_rr_shift": 0.0,
            "max_abs_rr_shift": 0.0,
            "profile_norm": 0.0,
            "update_norm": 0.0,
            "safety_fallback_rate": 0.0,
        }

    y_true = np.concatenate(y_true_all)
    pre = np.concatenate(pre_all)
    post = np.concatenate(post_all)
    shift = np.concatenate(shifts)
    profile_norm = np.concatenate(profile_norms)
    update_norm = np.concatenate(update_norms)
    fallback_arr = np.concatenate(fallback)
    pre_mae = float(np.mean(np.abs(pre - y_true)))
    post_mae = float(np.mean(np.abs(post - y_true)))
    rmse = float(np.sqrt(np.mean((post - y_true) ** 2)))
    return {
        "pre_mae": pre_mae,
        "post_mae": post_mae,
        "rmse": rmse,
        "corr": _safe_corr(y_true, post),
        "n_windows": int(y_true.shape[0]),
        "mean_abs_rr_shift": float(np.mean(np.abs(shift))) if shift.size else 0.0,
        "max_abs_rr_shift": float(np.max(np.abs(shift))) if shift.size else 0.0,
        "profile_norm": float(np.mean(profile_norm)) if profile_norm.size else 0.0,
        "update_norm": float(np.mean(update_norm)) if update_norm.size else 0.0,
        "safety_fallback_rate": float(np.mean(fallback_arr)) if fallback_arr.size else 0.0,
    }


def summarize(subject_rows: List[Dict[str, object]], out_dir: Path) -> None:
    df = pd.DataFrame(subject_rows)
    if df.empty:
        return
    none_by_subject = df[df["mode"] == "none"].set_index("subject")["post_mae"].to_dict()
    delta = []
    improved = []
    worse = []
    for _, row in df.iterrows():
        base = float(none_by_subject.get(row["subject"], row["post_mae"]))
        d = float(row["post_mae"]) - base
        delta.append(d)
        improved.append(bool(d < -1e-8))
        worse.append(bool(d > 1e-8))
    df["delta_vs_none"] = delta
    df["improved_vs_none"] = improved
    df["worse_vs_none"] = worse
    df.to_csv(out_dir / "subject_metrics.csv", index=False)

    summary_rows = []
    harm_rows = []
    budget_rows = []
    for mode, g in df.groupby("mode", sort=False):
        degradations = g["delta_vs_none"].astype(float)
        summary_rows.append(
            {
                "mode": mode,
                "n_subjects": int(g["subject"].nunique()),
                "post_mae_mean": float(g["post_mae"].mean()),
                "post_mae_std": float(g["post_mae"].std(ddof=0)),
                "delta_vs_none_mean": float(g["delta_vs_none"].mean()),
                "subjects_improved_vs_none": int(g["improved_vs_none"].sum()),
                "subjects_worse_vs_none": int(g["worse_vs_none"].sum()),
                "worst_subject_degradation": float(degradations.max()),
                "fallback_rate_mean": float(g["safety_fallback_rate"].mean()),
            }
        )
        harm_rows.append(
            {
                "mode": mode,
                "subjects_harmed": int(g["worse_vs_none"].sum()),
                "subjects_improved": int(g["improved_vs_none"].sum()),
                "mean_degradation_positive_only": float(degradations[degradations > 0].mean()) if bool((degradations > 0).any()) else 0.0,
                "worst_subject_degradation": float(degradations.max()),
            }
        )
        budget_rows.append(
            {
                "mode": mode,
                "mean_abs_rr_shift": float(g["mean_abs_rr_shift"].mean()),
                "max_abs_rr_shift": float(g["max_abs_rr_shift"].max()),
                "profile_norm_mean": float(g["profile_norm"].mean()),
                "update_norm_mean": float(g["update_norm"].mean()),
                "fallback_rate_mean": float(g["safety_fallback_rate"].mean()),
            }
        )

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(harm_rows).to_csv(out_dir / "harm_summary.csv", index=False)
    pd.DataFrame(budget_rows).to_csv(out_dir / "correction_budget_summary.csv", index=False)


def run(args) -> None:
    set_seed(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    eval_subjects = list(args.eval_subjects or args.subjects)
    meta_log_rows: List[Dict[str, object]] = []
    subject_rows: List[Dict[str, object]] = []

    for heldout in eval_subjects:
        source_subjects = [s for s in args.subjects if s != heldout]
        if not source_subjects:
            raise RuntimeError(f"Held-out subject {heldout} leaves no source subjects.")
        print(f"\n=== Held-out subject {heldout} | source={len(source_subjects)} ===")
        sbj_dir = out_dir / str(heldout)
        sbj_dir.mkdir(parents=True, exist_ok=True)

        train_loader, val_loader, test_loader = build_loocv_loaders(
            heldout,
            args.subjects,
            args.data_str,
            val_split=float(args.val_split),
            batch_size=int(args.batch_size),
            shuffle=True,
            drop_last=False,
            data_dir=args.data_dir,
            mdl_dir=args.mdl_dir or f"{M_DIR}/{args.data_str}/loocv",
            data_group=args.data_group,
            train_aug_ratio=float(args.train_aug_ratio),
        )
        source_subject_loaders = {
            s: build_subject_loader([s], args, shuffle=True, aug_ratio=0.0) for s in source_subjects
        }
        n_channels = infer_n_channels(train_loader)
        profile_enabled = any(m in args.correction_modes for m in ("profile_film", "qkv_last1_small"))
        model = build_model(n_channels, args, profile_enabled=profile_enabled).to(args.device)
        affine = ReadoutAffineCorrection(min_slope=float(args.affine_min_slope)).to(args.device)
        warmup_lazy_modules(model, train_loader, args.device)

        model, affine, hist = train_fold(
            model,
            affine,
            train_loader,
            val_loader,
            source_subject_loaders,
            source_subjects,
            args.device,
            args,
            sbj_dir,
        )
        for row in hist:
            meta_log_rows.append({"subject": heldout, **row})
            pd.DataFrame(meta_log_rows).to_csv(out_dir / "meta_train_log.csv", index=False)

        source_profile_mean, source_profile_std = collect_profile_normalizer(model, train_loader, args.device, args)
        for mode in args.correction_modes:
            metrics = evaluate_mode(
                model,
                affine,
                test_loader,
                mode,
                args.device,
                args,
                source_profile_mean=source_profile_mean,
                source_profile_std=source_profile_std,
            )
            row = {"subject": heldout, "mode": mode, **metrics}
            subject_rows.append(row)
            print(
                f"{heldout} {mode}: pre_mae={metrics['pre_mae']:.3f} "
                f"post_mae={metrics['post_mae']:.3f} fallback={metrics['safety_fallback_rate']:.3f}"
            )
            summarize(subject_rows, out_dir)

    if not meta_log_rows:
        pd.DataFrame().to_csv(out_dir / "meta_train_log.csv", index=False)
    summarize(subject_rows, out_dir)
    print(f"\nWrote outputs to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", default=default_subjects())
    parser.add_argument("--eval-subjects", nargs="+", default=None)
    parser.add_argument("--data-str", default="imu_filt", choices=["imu_filt", "imu_ica"])
    parser.add_argument("--data-group", default="mr", choices=["mr", "levels", "mr_levels"])
    parser.add_argument("--data-dir", default=SBJ_PROCESSED_DIR)
    parser.add_argument("--mdl-dir", default=None)
    parser.add_argument("--out-dir", default="results/meta_safe_profile_adaptation")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.25)
    parser.add_argument("--train-aug-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("IMU_DATALOADER_WORKERS", "0")))
    parser.add_argument("--max-train-batches", type=int, default=0, help="Debug limiter; 0 uses the full source epoch.")
    parser.add_argument("--max-eval-batches", type=int, default=0, help="Debug limiter for validation/test evaluation; 0 uses all batches.")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--decoder-mode", default="gru", choices=["none", "gru", "self_attn", "cross_attn"])
    parser.add_argument("--decoder-layers", type=int, default=1)
    parser.add_argument("--rr-from", default="encoder", choices=["encoder", "decoder", "both"])
    parser.add_argument("--imu-token-mixer", default="dwconv", choices=["none", "dwconv"])
    parser.add_argument("--rr-head-type", default="mlp", choices=["mlp", "token_tcn"])
    parser.add_argument("--rr-tcn-layers", type=int, default=2)
    parser.add_argument("--rr-tcn-kernel-size", type=int, default=3)
    parser.add_argument("--rr-tcn-dropout", type=float, default=None)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-stft", type=float, default=1.0)
    parser.add_argument("--lambda-rr", type=float, default=0.1)
    parser.add_argument("--lambda-contrast", type=float, default=0.05)
    parser.add_argument("--lambda-contrast-min", type=float, default=0.0)
    parser.add_argument("--contrast-warmup-epochs", type=int, default=5)
    parser.add_argument("--contrast-ramp-end-epoch", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--shift-max", type=int, default=24)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=100000.0, help="Skip an optimizer step if the pre-clipped gradient norm exceeds this value.")
    parser.add_argument("--max-loss-value", type=float, default=10000.0, help="Skip an optimizer step if the scalar loss exceeds this value or is non-finite.")
    parser.add_argument("--rollback-nonfinite-step", action="store_true", default=True, help="Restore parameters and clear Adam state if an optimizer step creates non-finite parameters.")
    parser.add_argument("--no-rollback-nonfinite-step", dest="rollback_nonfinite_step", action="store_false")

    parser.add_argument("--meta-every", type=int, default=1)
    parser.add_argument("--meta-batches", type=int, default=4)
    parser.add_argument("--meta-inner-repeats", type=int, default=1)
    parser.add_argument("--correction-modes", nargs="+", default=list(CORRECTION_MODES), choices=CORRECTION_MODES)
    parser.add_argument("--lambda-meta", type=float, default=1.0)
    parser.add_argument("--lambda-harm", type=float, default=1.0)
    parser.add_argument("--lambda-moment", type=float, default=0.01)
    parser.add_argument("--lambda-update", type=float, default=0.01)
    parser.add_argument("--harm-margin", type=float, default=0.0)

    parser.add_argument("--subject-balanced", action="store_true")
    parser.add_argument("--use-moment-loss", action="store_true")
    parser.add_argument("--use-meta-no-harm", action="store_true")

    parser.add_argument("--max-rr-shift-bpm", type=float, default=3.0)
    parser.add_argument("--max-profile-norm", type=float, default=3.0)
    parser.add_argument("--max-update-norm", type=float, default=1.0)
    parser.add_argument("--safety-mode", default="fallback", choices=["clamp", "fallback"])

    parser.add_argument("--affine-lambda-a", type=float, default=0.01)
    parser.add_argument("--affine-lambda-b", type=float, default=0.01)
    parser.add_argument("--affine-max-shift-bpm", type=float, default=3.0)
    parser.add_argument("--affine-min-slope", type=float, default=0.05)

    parser.add_argument("--profile-dim", type=int, default=32)
    parser.add_argument("--profile-hidden-dim", type=int, default=128)
    parser.add_argument("--profile-stats-max-batches", type=int, default=50)
    parser.add_argument("--profile-film-scale", type=float, default=0.1)
    parser.add_argument("--profile-film-placement", default="token_pooled", choices=["token_pooled", "pooled_only", "late_token_only", "residual"])
    parser.add_argument("--profile-film-residual-alpha", type=float, default=0.1)
    parser.add_argument("--profile-qkv-scale", type=float, default=0.01)
    parser.add_argument("--profile-qkv-layers", default="last1", choices=["last1", "last2", "all"])
    parser.add_argument("--profile-qkv-residual", action="store_true")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
