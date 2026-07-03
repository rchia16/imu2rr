#!/usr/bin/env python3
"""
Phase 1-6 profile-conditioning utilities and CLI entrypoint for RR/STFT experiments.

This module contains the reusable profile encoder pieces plus the phase-6
command-line entrypoint that runs the profile-aware LOSO sweep.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

PROFILE_STATS_SCALAR_DIM = 8
PROFILE_STATS_MIN_STD = 1e-6


def _rr_tta_mode_complete(sweep_root: Path, mode: str, subject: str) -> bool:
    metrics_dir = sweep_root / str(mode) / "subjects" / str(subject) / "rr_feature_adaptive"
    if not metrics_dir.is_dir():
        return False
    pattern = f"rr_*_metrics_{subject}.json"
    return any(path.is_file() and path.stat().st_size > 0 for path in metrics_dir.glob(pattern))


def _completed_rr_tta_modes(sweep_root: Path, modes: Sequence[str], subject: str) -> List[str]:
    return [str(mode) for mode in modes if _rr_tta_mode_complete(sweep_root, str(mode), str(subject))]


class PatientProfileEncoder(nn.Module):
    """Map low-dimensional subject/window summary statistics into a profile vector."""

    def __init__(self, in_dim: int, profile_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        if int(in_dim) <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        if int(profile_dim) <= 0:
            raise ValueError(f"profile_dim must be positive, got {profile_dim}")
        if int(hidden_dim) <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.in_dim = int(in_dim)
        self.profile_dim = int(profile_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.profile_dim),
        )

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        if stats.ndim == 1:
            stats = stats.unsqueeze(0)
        if stats.ndim != 2:
            raise ValueError(f"Expected stats with shape (B,D) or (D,), got {tuple(stats.shape)}")
        if stats.size(-1) != self.in_dim:
            raise ValueError(f"Expected stats last dim {self.in_dim}, got {stats.size(-1)}")
        return self.net(stats)


class ProfileFiLM(nn.Module):
    """Apply profile-conditioned affine modulation to token or pooled representations."""

    def __init__(self, profile_dim: int, d_model: int, scale: float = 0.1):
        super().__init__()
        if int(profile_dim) <= 0:
            raise ValueError(f"profile_dim must be positive, got {profile_dim}")
        if int(d_model) <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        self.profile_dim = int(profile_dim)
        self.d_model = int(d_model)
        self.scale = float(scale)
        self.to_gamma_beta = nn.Linear(self.profile_dim, 2 * self.d_model)

    def forward(self, h: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        if profile.ndim == 1:
            profile = profile.unsqueeze(0)
        if profile.ndim != 2:
            raise ValueError(f"Expected profile with shape (B,D) or (D,), got {tuple(profile.shape)}")
        if profile.size(-1) != self.profile_dim:
            raise ValueError(
                f"Expected profile last dim {self.profile_dim}, got {profile.size(-1)}"
            )
        if h.ndim not in (2, 3):
            raise ValueError(f"Expected h with shape (B,D) or (B,T,D), got {tuple(h.shape)}")
        if h.size(0) != profile.size(0):
            raise ValueError(
                f"Batch mismatch between h ({h.size(0)}) and profile ({profile.size(0)})"
            )
        if h.size(-1) != self.d_model:
            raise ValueError(f"Expected h last dim {self.d_model}, got {h.size(-1)}")

        gamma, beta = self.to_gamma_beta(profile).chunk(2, dim=-1)
        gamma = 1.0 + self.scale * torch.tanh(gamma)
        beta = self.scale * torch.tanh(beta)

        if h.ndim == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        return gamma * h + beta


def profile_stats_dim(latent_dim: int) -> int:
    """Return stats dimensionality for pooled latent size `latent_dim`."""
    if int(latent_dim) <= 0:
        raise ValueError(f"latent_dim must be positive, got {latent_dim}")
    return 2 * int(latent_dim) + PROFILE_STATS_SCALAR_DIM


def split_profile_stats(stats: torch.Tensor, latent_dim: int) -> Dict[str, torch.Tensor]:
    """Split a flat profile-stats vector into named latent and scalar blocks."""
    if stats.ndim != 1:
        raise ValueError(f"Expected 1D stats vector, got {tuple(stats.shape)}")
    latent_dim = int(latent_dim)
    expected_dim = profile_stats_dim(latent_dim)
    if stats.numel() != expected_dim:
        raise ValueError(f"Expected stats dim {expected_dim}, got {stats.numel()}")

    offset = 0
    z_mean = stats[offset : offset + latent_dim]
    offset += latent_dim
    z_std = stats[offset : offset + latent_dim]
    offset += latent_dim
    scalars = stats[offset:]
    return {
        "z_mean": z_mean,
        "z_std": z_std,
        "rr_aux_mean": scalars[0:1],
        "rr_aux_std": scalars[1:2],
        "rr_stft_mean": scalars[2:3],
        "rr_stft_std": scalars[3:4],
        "rr_delta_mean": scalars[4:5],
        "rr_delta_std": scalars[5:6],
        "stft_confidence_mean": scalars[6:7],
        "stft_confidence_std": scalars[7:8],
    }


def estimate_rr_from_predicted_stft(
    pred_logmag: torch.Tensor,
    br_fs: float = 18.0,
    return_confidence: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """Estimate RR from reconstructed pressure log-STFT using the sweep's band logic."""
    if pred_logmag.ndim != 3:
        raise ValueError(f"Expected pred_logmag (B,T,F), got {tuple(pred_logmag.shape)}")

    n_freq = int(pred_logmag.size(-1))
    n_fft = max(2, 2 * (n_freq - 1))
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / float(br_fs)).to(pred_logmag.device)
    mask = (freqs >= 0.05) & (freqs <= 0.75)
    if not bool(mask.any()):
        rr = pred_logmag.new_zeros(pred_logmag.size(0))
        conf = pred_logmag.new_zeros(pred_logmag.size(0))
        return (rr, conf) if return_confidence else rr

    spectrum = pred_logmag.float().mean(dim=1)
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
    rr = rr_hz * 60.0
    return (rr, confidence) if return_confidence else rr


def build_profile_stats(
    z: torch.Tensor,
    rr_aux: torch.Tensor,
    rr_stft: torch.Tensor,
    stft_conf: torch.Tensor,
) -> torch.Tensor:
    """Assemble the phase 1 statistics vector from pooled latents and RR summaries."""
    if z.ndim != 2:
        raise ValueError(f"Expected pooled latent z with shape (N,D), got {tuple(z.shape)}")

    rr_aux = rr_aux.reshape(-1)
    rr_stft = rr_stft.reshape(-1)
    stft_conf = stft_conf.reshape(-1)
    n = z.size(0)
    if rr_aux.numel() != n or rr_stft.numel() != n or stft_conf.numel() != n:
        raise ValueError(
            "Stats inputs must share the same number of windows: "
            f"z={n}, rr_aux={rr_aux.numel()}, rr_stft={rr_stft.numel()}, stft_conf={stft_conf.numel()}"
        )

    delta = rr_aux - rr_stft
    return torch.cat(
        [
            z.mean(dim=0),
            z.std(dim=0, unbiased=False),
            rr_aux.mean().view(1),
            rr_aux.std(unbiased=False).view(1),
            rr_stft.mean().view(1),
            rr_stft.std(unbiased=False).view(1),
            delta.mean().view(1),
            delta.std(unbiased=False).view(1),
            stft_conf.mean().view(1),
            stft_conf.std(unbiased=False).view(1),
        ],
        dim=0,
    )


def normalize_profile_stats(
    profile_stats: torch.Tensor,
    source_profile_mean: torch.Tensor,
    source_profile_std: torch.Tensor,
    eps: float = PROFILE_STATS_MIN_STD,
) -> torch.Tensor:
    """Normalize profile stats with a source-fold mean/std and a variance floor."""
    if profile_stats.shape != source_profile_mean.shape or profile_stats.shape != source_profile_std.shape:
        raise ValueError(
            "profile_stats, source_profile_mean, and source_profile_std must have identical shapes: "
            f"{tuple(profile_stats.shape)}, {tuple(source_profile_mean.shape)}, {tuple(source_profile_std.shape)}"
        )
    denom = source_profile_std.clamp_min(float(eps))
    return (profile_stats - source_profile_mean) / denom


def profile_stats_diagnostics(
    profile_stats: torch.Tensor,
    source_profile_mean: Optional[torch.Tensor] = None,
    source_profile_std: Optional[torch.Tensor] = None,
    profile_dim: Optional[int] = None,
    eps: float = PROFILE_STATS_MIN_STD,
) -> Dict[str, float]:
    """Summarize raw and normalized profile-stat magnitudes for logging."""
    stats = profile_stats.reshape(-1).float()
    diag: Dict[str, float] = {
        "profile_stats_dim": int(stats.numel()),
        "profile_dim": int(profile_dim) if profile_dim is not None else 0,
    }

    if source_profile_mean is None or source_profile_std is None:
        norm = stats
    else:
        norm = normalize_profile_stats(
            stats,
            source_profile_mean.reshape(-1).float(),
            source_profile_std.reshape(-1).float(),
            eps=float(eps),
        )

    diag.update(
        {
            "profile_norm": float(norm.norm(p=2).item()),
            "profile_mean_abs": float(norm.abs().mean().item()),
            "profile_std": float(norm.std(unbiased=False).item()),
            "profile_source_prior_l2": float((norm * norm).mean().item()),
        }
    )
    return diag


def _split_model_outputs(model_outputs: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(model_outputs, (tuple, list)) or len(model_outputs) < 3:
        raise ValueError(
            "Model must return at least (pred_logmag, rr_aux, hidden) for profile-stat collection."
        )
    pred_logmag, rr_aux, hidden = model_outputs[:3]
    if not isinstance(pred_logmag, torch.Tensor) or not isinstance(rr_aux, torch.Tensor) or not isinstance(hidden, torch.Tensor):
        raise TypeError("Model outputs must be tensors: (pred_logmag, rr_aux, hidden)")
    return pred_logmag, rr_aux, hidden


def pooled_features(hidden: torch.Tensor) -> torch.Tensor:
    return hidden.mean(dim=1)


def _default_profile_stats(
    model: nn.Module,
    device: str | torch.device,
) -> torch.Tensor:
    latent_dim = int(getattr(model, "d_model", 0))
    if latent_dim <= 0:
        raise RuntimeError("Unable to infer latent dim for default profile stats.")
    return torch.zeros(profile_stats_dim(latent_dim), device=device, dtype=torch.float32)


def _finite_window_mask(
    z: torch.Tensor,
    rr_aux: torch.Tensor,
    rr_stft: torch.Tensor,
    stft_conf: torch.Tensor,
) -> torch.Tensor:
    return (
        torch.isfinite(z).all(dim=1)
        & torch.isfinite(rr_aux.reshape(-1))
        & torch.isfinite(rr_stft.reshape(-1))
        & torch.isfinite(stft_conf.reshape(-1))
    )


def unpack_batch(
    batch: Iterable[torch.Tensor],
    device: str | torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if len(batch) == 4:
        imu, pressure, conds, br = batch
        tlx = None
    elif len(batch) == 5:
        imu, pressure, conds, br, tlx = batch
    else:
        raise ValueError(f"Expected 4 or 5 tensors from dataloader, got {len(batch)}")

    imu = imu.float().to(device)
    pressure = pressure.float().to(device)
    if pressure.ndim == 3:
        pressure = pressure.squeeze(-1)
    conds = conds.to(device)
    br = br.float().to(device)
    if tlx is not None:
        tlx = tlx.float().to(device)
    return imu, pressure, conds, br, tlx


@torch.no_grad()
def collect_profile_stats(
    model: nn.Module,
    loader,
    device: str | torch.device,
    max_batches: Optional[int] = None,
    br_fs: float = 18.0,
) -> torch.Tensor:
    """Collect phase 1 profile statistics from a loader using the current model."""
    model.eval()

    z_list = []
    rr_aux_list = []
    rr_stft_list = []
    conf_list = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break

        x, _pressure, _cond, _br, _tlx = unpack_batch(batch, device)

        pred_logmag, rr_aux, hidden = _split_model_outputs(model(x))
        z = pooled_features(hidden)
        rr_stft, stft_conf = estimate_rr_from_predicted_stft(
            pred_logmag,
            br_fs=float(br_fs),
            return_confidence=True,
        )

        keep = _finite_window_mask(z, rr_aux, rr_stft, stft_conf)
        if not bool(keep.any()):
            continue

        z = z[keep]
        rr_aux = rr_aux.reshape(-1)[keep]
        rr_stft = rr_stft.reshape(-1)[keep]
        stft_conf = stft_conf.reshape(-1)[keep]

        z_list.append(z.detach())
        rr_aux_list.append(rr_aux.detach())
        rr_stft_list.append(rr_stft.detach())
        conf_list.append(stft_conf.detach())

    if not z_list:
        return _default_profile_stats(model, device)

    z = torch.cat(z_list, dim=0)
    rr_aux = torch.cat(rr_aux_list, dim=0)
    rr_stft = torch.cat(rr_stft_list, dim=0)
    stft_conf = torch.cat(conf_list, dim=0)
    return build_profile_stats(z, rr_aux, rr_stft, stft_conf)


@torch.no_grad()
def estimate_source_profile_normalizer(
    model: nn.Module,
    loader,
    device: str | torch.device,
    max_batches: Optional[int] = None,
    br_fs: float = 18.0,
    min_std: float = PROFILE_STATS_MIN_STD,
) -> Dict[str, torch.Tensor]:
    """Estimate source-fold profile-stat mean/std from batch-level summary vectors."""
    model.eval()

    batch_stats: List[torch.Tensor] = []
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break

        x, _pressure, _cond, _br, _tlx = unpack_batch(batch, device)
        pred_logmag, rr_aux, hidden = _split_model_outputs(model(x))
        z = pooled_features(hidden)
        rr_stft, stft_conf = estimate_rr_from_predicted_stft(
            pred_logmag,
            br_fs=float(br_fs),
            return_confidence=True,
        )

        keep = _finite_window_mask(z, rr_aux, rr_stft, stft_conf)
        if not bool(keep.any()):
            continue

        batch_stats.append(
            build_profile_stats(
                z.detach()[keep],
                rr_aux.detach().reshape(-1)[keep],
                rr_stft.detach().reshape(-1)[keep],
                stft_conf.detach().reshape(-1)[keep],
            )
        )

    if not batch_stats:
        default_stats = _default_profile_stats(model, device)
        return {
            "source_profile_mean": default_stats.clone(),
            "source_profile_std": torch.ones_like(default_stats),
            "source_profile_stats": default_stats.clone(),
            "source_profile_stats_norm": default_stats.clone(),
            "source_profile_batch_stats": default_stats.unsqueeze(0),
        }

    batch_stats_tensor = torch.stack(batch_stats, dim=0)
    source_profile_mean = batch_stats_tensor.mean(dim=0)
    source_profile_std = batch_stats_tensor.std(dim=0, unbiased=False).clamp_min(float(min_std))
    aggregate_profile_stats = collect_profile_stats(
        model,
        loader,
        device,
        max_batches=max_batches,
        br_fs=br_fs,
    )
    aggregate_profile_stats_norm = normalize_profile_stats(
        aggregate_profile_stats,
        source_profile_mean,
        source_profile_std,
        eps=float(min_std),
    )
    return {
        "source_profile_mean": source_profile_mean,
        "source_profile_std": source_profile_std,
        "source_profile_stats": aggregate_profile_stats,
        "source_profile_stats_norm": aggregate_profile_stats_norm,
        "source_profile_batch_stats": batch_stats_tensor,
    }


def run_phase6_profile_encoder_cli(argv: Optional[Sequence[str]] = None) -> None:
    """Phase-6 executable entrypoint.

    The profile encoder is the public runnable surface. The heavy sweep
    implementation remains in the ladder module, but this function owns the CLI
    entrypoint and dispatches into that implementation.
    """
    sys.modules.setdefault("vit_pressure_crossmodal_profile_encoder", sys.modules[__name__])

    from vit_pressure_crossmodal_stft_rr_rrprobe_tta_main import (
        SUBJECTS,
        build_base_parser,
        run_loocv_experiment,
    )
    from vit_pressure_crossmodal_stft_rr_unsup_ladder_config_sweep import (
        DEFAULT_SWEEP_MODES,
        PROFILE_VECTOR_UNSUP_MODES,
        _parse_modes,
        _patch_loocv_generator_for_eval_subjects,
        add_common_adaptation_args,
        rr_config_sweep_hook,
    )

    parser = build_base_parser(SUBJECTS, "profile_encoder_rr_tta")
    add_common_adaptation_args(parser)
    parser.add_argument(
        "--rr-tta-modes",
        default=" ".join(DEFAULT_SWEEP_MODES),
        help="Space- or comma-separated RR-TTA modes to evaluate after each subject checkpoint is available.",
    )
    parser.add_argument("--sweep-root", default="", help="Root where per-mode summaries/predictions are written. Defaults to --out-dir.")
    parser.add_argument("--sweep-run-id", default="single", help="Unique id for this subject/job; used in chunk summary filenames.")
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip eval subjects whose requested RR-TTA modes already have per-subject metrics JSON outputs.",
    )
    parser.add_argument("--apply-mode-defaults", action="store_true", default=True)
    parser.add_argument("--no-apply-mode-defaults", dest="apply_mode_defaults", action="store_false")
    parser.add_argument(
        "--eval-subjects",
        nargs="+",
        default=None,
        help=(
            "Held-out subject(s) to evaluate in this process while preserving the full "
            "--subjects cohort for LOSO source training. This is the correct argument "
            "for subject-level GPU scheduling."
        ),
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    requested_modes = _parse_modes(args.rr_tta_modes)
    if str(getattr(args, "rr_tta", "none")) in PROFILE_VECTOR_UNSUP_MODES:
        requested_modes.append(str(args.rr_tta))
    has_profile_film_mode = any("profile_film_" in str(m) for m in requested_modes)
    has_profile_qkv_mode = any("profile_qkv_" in str(m) for m in requested_modes)
    if has_profile_film_mode and has_profile_qkv_mode:
        raise SystemExit(
            "FiLM and QKV profile modes must be run in separate jobs because the source checkpoint "
            "trains only one profile-conditioning family at a time."
        )
    if any(str(m) in PROFILE_VECTOR_UNSUP_MODES for m in requested_modes):
        if has_profile_qkv_mode:
            args.use_profile_qkv = True
            args.profile_conditioning = "qkv"
        else:
            args.use_profile_film = True
            if str(getattr(args, "profile_conditioning", "none")).lower().strip() == "none":
                args.profile_conditioning = "film"
        if int(getattr(args, "profile_stats_dim", 0)) <= 0:
            args.profile_stats_dim = profile_stats_dim(int(args.d_model))

    valid_modes = set(parser._option_string_actions["--rr-tta"].choices)
    bad = [m for m in _parse_modes(args.rr_tta_modes) if m not in valid_modes]
    if bad:
        raise SystemExit(f"Unsupported mode(s) in --rr-tta-modes: {bad}. Valid modes: {sorted(valid_modes)}")

    full_subjects = list(getattr(args, "subjects", []) or [])
    if len(full_subjects) < 2:
        raise SystemExit(
            "Need at least two subjects in --subjects because it defines the full LOSO source cohort. "
            "For subject-level scheduling, pass the held-out subject via --eval-subjects, not singleton --subjects."
        )

    eval_subjects = list(args.eval_subjects or full_subjects)
    if bool(getattr(args, "skip_completed", False)):
        sweep_root = Path(args.sweep_root) if str(args.sweep_root or "").strip() else Path(args.out_dir)
        requested_mode_set = _parse_modes(args.rr_tta_modes)
        remaining_subjects = []
        for subject in eval_subjects:
            completed = set(_completed_rr_tta_modes(sweep_root, requested_mode_set, str(subject)))
            missing = [mode for mode in requested_mode_set if mode not in completed]
            if missing:
                print(
                    f"[SKIP-COMPLETED] subject={subject} missing_modes={' '.join(missing)}",
                    flush=True,
                )
                remaining_subjects.append(subject)
            else:
                print(
                    f"[SKIP-COMPLETED] subject={subject} all requested modes complete under {sweep_root}",
                    flush=True,
                )
        eval_subjects = remaining_subjects
        if not eval_subjects:
            print(
                f"[SKIP-COMPLETED] all requested subjects complete under {sweep_root}; nothing to train",
                flush=True,
            )
            return
    _patch_loocv_generator_for_eval_subjects(eval_subjects, full_subjects)
    args.subjects = eval_subjects
    run_loocv_experiment(args, pre_eval_hooks=[rr_config_sweep_hook])


def main() -> None:
    run_phase6_profile_encoder_cli()


if __name__ == "__main__":
    main()
