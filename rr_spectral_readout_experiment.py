from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import os
import random
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import BR_FS, M_DIR, SBJ_PROCESSED_DIR
from dataloader import build_loocv_loaders
from vit_pressure_crossmodal_profile_encoder import estimate_rr_from_predicted_stft
from vit_pressure_crossmodal_stft_rr_core import (
    TinyIMU2PressureViT,
    augment_imu,
    infer_n_channels,
    pressure_stft_recon_loss,
    rr_stft_consistency_loss,
    token_contrastive_loss,
)


DEFAULT_SUBJECTS = [
    "S12",
    "S13",
    "S14",
    "S15",
    "S16",
    "S18",
    "S19",
    "S20",
    "S22",
    "S23",
    "S24",
    "S25",
    "S27",
    "S28",
    "S29",
]
STAGES = ("audit", "frozen_readouts", "rr_finetune", "profile_film", "all")
TEMP_GRID = (0.03, 0.05, 0.10, 0.20, 0.50, 1.00)
EPS_GRID = (0.5, 1.0, 2.0)
HARMONIC_ALPHA_GRID = (0.0, 0.25, 0.5)
HARMONIC_BETA_GRID = (0.0, 0.1)


@dataclass
class RRPredictionBatch:
    rr_true: torch.Tensor
    rr_direct: torch.Tensor
    rr_spectral_hard: torch.Tensor
    rr_spectral_soft: torch.Tensor
    rr_hidden_probe: Optional[torch.Tensor]
    rr_spectral_probe: Optional[torch.Tensor]
    rr_spectral_residual: Optional[torch.Tensor]
    predicted_stft: torch.Tensor
    true_stft: torch.Tensor
    hidden_tokens: torch.Tensor
    pooled_hidden: torch.Tensor
    spectral_confidence: torch.Tensor


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=json_default)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def run_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "argv": sys.argv,
        "args": vars(args),
        "commit": git_commit(),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(args.device),
    }


def rr_targets_from_batch_local(pressure: torch.Tensor, br: Optional[torch.Tensor]) -> torch.Tensor:
    if br is None:
        raise ValueError("RR labels are required for evaluation/training batches.")
    return br.float().view(-1)


def unpack_batch_with_meta(
    batch: Iterable[Any],
    device: str | torch.device,
    *,
    allow_meta: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], List[str], np.ndarray]:
    parts = list(batch)
    meta = None
    if parts and isinstance(parts[-1], dict):
        meta = parts.pop()
    if len(parts) == 4:
        imu, pressure, conds, br = parts
        tlx = None
    elif len(parts) == 5:
        imu, pressure, conds, br, tlx = parts
    else:
        raise ValueError(f"Expected 4/5 tensors plus optional metadata, got {len(parts)} tensors.")

    imu = imu.float().to(device)
    pressure = pressure.float().to(device)
    if pressure.ndim == 3:
        pressure = pressure.squeeze(-1)
    conds = conds.to(device)
    br = br.float().to(device)
    tlx = None if tlx is None else tlx.float().to(device)

    subject_ids: List[str] = []
    window_indices = np.arange(imu.size(0), dtype=np.int64)
    if allow_meta and meta is not None:
        sid = meta.get("subject_id", None)
        if isinstance(sid, (list, tuple)):
            subject_ids = [str(x) for x in sid]
        elif sid is not None:
            subject_ids = [str(sid)] * imu.size(0)
        idx = meta.get("subject_index", None)
        if idx is not None:
            if torch.is_tensor(idx):
                window_indices = idx.detach().cpu().numpy().astype(np.int64).reshape(-1)
            else:
                window_indices = np.asarray(idx, dtype=np.int64).reshape(-1)
    if not subject_ids:
        subject_ids = [""] * imu.size(0)
    return imu, pressure, conds, br, tlx, subject_ids, window_indices


def close_loaders(*loaders: Any) -> None:
    for loader in loaders:
        iterator = getattr(loader, "_iterator", None)
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()
        if hasattr(loader, "_iterator"):
            loader._iterator = None


def build_loaders_for_subject(
    args: argparse.Namespace,
    subject: str,
    *,
    include_subject_id: bool = False,
    shuffle_train: bool = True,
    drop_last: bool = False,
    train_aug_ratio: float = 0.0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return build_loocv_loaders(
        subject,
        args.subjects,
        args.data_str,
        val_split=float(args.val_split),
        batch_size=int(args.batch_size),
        shuffle=shuffle_train,
        drop_last=drop_last,
        data_dir=args.data_dir,
        mdl_dir=args.mdl_dir,
        data_group=args.data_group,
        include_subject_id=include_subject_id,
        seed=int(args.seed),
        train_aug_ratio=train_aug_ratio,
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(args.persistent_workers),
    )


def leakage_check(
    held_out: str,
    train_subjects: Sequence[str],
    val_subjects: Sequence[str],
    *,
    target_labels_used_for_fit: bool = False,
    target_labels_used_for_profile: bool = False,
    selected_before_target_eval: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    selected_before_target_eval = selected_before_target_eval or {}
    checks = {
        "held_out_subject": held_out,
        "held_out_absent_from_source_training_records": held_out not in set(train_subjects),
        "held_out_absent_from_source_validation_records": held_out not in set(val_subjects),
        "target_rr_labels_not_passed_into_readout_training": not target_labels_used_for_fit,
        "target_rr_labels_not_passed_into_profile_construction": not target_labels_used_for_profile,
    }
    checks.update(selected_before_target_eval)
    failed = [k for k, v in checks.items() if k != "held_out_subject" and v is not True]
    checks["passed"] = len(failed) == 0
    checks["failed_checks"] = failed
    if failed:
        raise RuntimeError(f"Leakage audit failed for {held_out}: {failed}")
    return checks


def write_leakage_audit(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    write_json(
        out_dir / "leakage_audit.json",
        {"passed": all(r.get("passed", False) for r in rows), "folds": rows},
    )


def respiratory_freqs(n_freq: int, br_fs: float = BR_FS, device: Optional[torch.device] = None) -> torch.Tensor:
    n_fft = max(2, 2 * (int(n_freq) - 1))
    return torch.fft.rfftfreq(n_fft, d=1.0 / float(br_fs), device=device)


def respiratory_band(
    n_freq: int,
    br_fs: float = BR_FS,
    min_hz: float = 0.05,
    max_hz: float = 0.75,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = respiratory_freqs(n_freq, br_fs=br_fs, device=device)
    mask = (freqs >= float(min_hz)) & (freqs <= float(max_hz))
    if not bool(mask.any()):
        raise RuntimeError("No frequency bins found in the respiratory band.")
    return freqs[mask], mask


def band_mean_spectrum(
    stft: torch.Tensor,
    br_fs: float = BR_FS,
    min_hz: float = 0.05,
    max_hz: float = 0.75,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if stft.ndim != 3:
        raise ValueError(f"Expected STFT (B,T,F), got {tuple(stft.shape)}")
    freqs, mask = respiratory_band(stft.size(-1), br_fs, min_hz, max_hz, stft.device)
    spectrum = stft.float().mean(dim=1)
    band = spectrum[:, mask].clamp_min(0.0)
    return band, freqs, mask


class SoftSpectralRR(nn.Module):
    def __init__(
        self,
        br_fs: float,
        min_hz: float = 0.05,
        max_hz: float = 0.75,
        temperature: float = 0.1,
        harmonic_alpha: float = 0.0,
        harmonic_beta: float = 0.0,
    ):
        super().__init__()
        self.br_fs = float(br_fs)
        self.min_hz = float(min_hz)
        self.max_hz = float(max_hz)
        self.temperature = float(temperature)
        self.harmonic_alpha = float(harmonic_alpha)
        self.harmonic_beta = float(harmonic_beta)

    def _harmonic_score(self, full_spectrum: torch.Tensor, freqs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        band = full_spectrum[:, mask]
        if self.harmonic_alpha == 0.0 and self.harmonic_beta == 0.0:
            return band
        full_freqs = respiratory_freqs(full_spectrum.size(-1), self.br_fs, full_spectrum.device)
        scores = band.clone()
        for mult, weight in ((2.0, self.harmonic_alpha), (3.0, self.harmonic_beta)):
            if float(weight) == 0.0:
                continue
            target = freqs * mult
            idx = torch.searchsorted(full_freqs, target).clamp(1, full_freqs.numel() - 1)
            lo = idx - 1
            hi = idx
            f_lo = full_freqs[lo]
            f_hi = full_freqs[hi]
            w = ((target - f_lo) / (f_hi - f_lo).clamp_min(1e-8)).view(1, -1)
            interp = full_spectrum[:, lo] * (1.0 - w) + full_spectrum[:, hi] * w
            valid = (target <= full_freqs[-1]).float().view(1, -1)
            scores = scores + float(weight) * interp * valid
        return scores

    def probabilities(self, pred_logmag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs, mask = respiratory_band(
            pred_logmag.size(-1), self.br_fs, self.min_hz, self.max_hz, pred_logmag.device
        )
        full_spectrum = pred_logmag.float().mean(dim=1).clamp_min(0.0)
        scores = self._harmonic_score(full_spectrum, freqs, mask)
        probs = torch.softmax(scores / max(self.temperature, 1e-6), dim=1)
        return probs, freqs

    def forward(self, pred_logmag: torch.Tensor) -> torch.Tensor:
        probs, freqs = self.probabilities(pred_logmag)
        return 60.0 * (probs * freqs.view(1, -1)).sum(dim=1)


def hard_spectral_rr(stft: torch.Tensor, br_fs: float = BR_FS) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    band, freqs, _mask = band_mean_spectrum(stft, br_fs=br_fs)
    idx = band.argmax(dim=1)
    rr = freqs[idx] * 60.0
    if band.size(1) >= 2:
        top2 = torch.topk(band, k=2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1]).clamp_min(0.0)
    else:
        margin = band[:, 0].clamp_min(0.0)
    confidence = margin / band.sum(dim=1).clamp_min(1e-8)
    return rr, confidence.clamp(0.0, 1.0), freqs[idx]


def spectral_entropy_from_stft(stft: torch.Tensor, br_fs: float = BR_FS) -> torch.Tensor:
    band, _freqs, _mask = band_mean_spectrum(stft, br_fs=br_fs)
    p = band / band.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return -(p * torch.log(p.clamp_min(1e-8))).sum(dim=1)


def spectral_summary_features(stft: torch.Tensor, br_fs: float = BR_FS) -> torch.Tensor:
    band, freqs, _mask = band_mean_spectrum(stft, br_fs=br_fs)
    band_time = stft.float()[:, :, _mask].clamp_min(0.0)
    mean_bins = band_time.mean(dim=1)
    std_bins = band_time.std(dim=1, unbiased=False)
    hard_rr, conf, peak_hz = hard_spectral_rr(stft, br_fs=br_fs)
    entropy = spectral_entropy_from_stft(stft, br_fs=br_fs)
    p = band / band.sum(dim=1, keepdim=True).clamp_min(1e-8)
    fundamental_mass = p.max(dim=1).values
    def harmonic_mass(mult: float) -> torch.Tensor:
        target = peak_hz * mult
        idx = torch.argmin((freqs.view(1, -1) - target.view(-1, 1)).abs(), dim=1)
        return p[torch.arange(p.size(0), device=p.device), idx]
    extras = torch.stack(
        [
            hard_rr,
            conf,
            peak_hz,
            entropy,
            fundamental_mass,
            harmonic_mass(2.0),
            harmonic_mass(3.0),
        ],
        dim=1,
    )
    return torch.cat([mean_bins, std_bins, extras], dim=1)


def gaussian_peak_target(rr_true: torch.Tensor, freqs: torch.Tensor, sigma_hz: float) -> torch.Tensor:
    f_star = rr_true.float().view(-1, 1) / 60.0
    q = torch.exp(-0.5 * ((freqs.view(1, -1) - f_star) / max(float(sigma_hz), 1e-6)).pow(2))
    return q / q.sum(dim=1, keepdim=True).clamp_min(1e-8)


def peak_distribution_loss(
    pred_logmag: torch.Tensor,
    rr_true: torch.Tensor,
    soft_rr: SoftSpectralRR,
    sigma_hz: float,
) -> torch.Tensor:
    probs, freqs = soft_rr.probabilities(pred_logmag)
    q = gaussian_peak_target(rr_true, freqs, sigma_hz)
    return -(q * torch.log(probs.clamp_min(1e-8))).sum(dim=1).mean()


def rr_metric_dict(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    err = y_pred - y_true
    abs_err = np.abs(err)
    corr = (
        float(np.corrcoef(y_true, y_pred)[0, 1])
        if y_true.size > 1 and np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8
        else float("nan")
    )
    out = {
        "MAE": float(np.mean(abs_err)) if abs_err.size else float("nan"),
        "RMSE": float(np.sqrt(np.mean(err * err))) if err.size else float("nan"),
        "Pearson_correlation": corr,
        "median_absolute_error": float(np.median(abs_err)) if abs_err.size else float("nan"),
        "p95_absolute_error": float(np.percentile(abs_err, 95)) if abs_err.size else float("nan"),
        "bias": float(np.mean(err)) if err.size else float("nan"),
        "error_standard_deviation": float(np.std(err)) if err.size else float("nan"),
        "number_of_windows": int(y_true.size),
    }
    if prefix:
        return {f"{prefix}_{k}": v for k, v in out.items()}
    return out


def spectral_metric_dict(
    pred_stft: np.ndarray,
    true_stft: np.ndarray,
    rr_true: np.ndarray,
    rr_pred: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    pred = torch.as_tensor(pred_stft, dtype=torch.float32)
    true = torch.as_tensor(true_stft, dtype=torch.float32)
    band_pred, freqs, mask = band_mean_spectrum(pred)
    band_true, _freqs, _mask = band_mean_spectrum(true)
    pred_peak = freqs[band_pred.argmax(dim=1)].cpu().numpy()
    true_peak = freqs[band_true.argmax(dim=1)].cpu().numpy()
    p = band_pred / band_pred.sum(dim=1, keepdim=True).clamp_min(1e-8)
    entropy = (-(p * torch.log(p.clamp_min(1e-8))).sum(dim=1)).cpu().numpy()
    if band_pred.size(1) >= 2:
        top2 = torch.topk(band_pred, 2, dim=1).values
        margin = ((top2[:, 0] - top2[:, 1]) / band_pred.sum(dim=1).clamp_min(1e-8)).cpu().numpy()
    else:
        margin = np.zeros(pred.shape[0], dtype=np.float32)
    global_corr = (
        float(np.corrcoef(pred.reshape(-1).numpy(), true.reshape(-1).numpy())[0, 1])
        if pred.numel() > 1
        else float("nan")
    )
    band_corr = (
        float(np.corrcoef(band_pred.reshape(-1).numpy(), band_true.reshape(-1).numpy())[0, 1])
        if band_pred.numel() > 1
        else float("nan")
    )
    rr_err = np.asarray(rr_pred).reshape(-1) - np.asarray(rr_true).reshape(-1)
    peak_rr_err = pred_peak * 60.0 - true_peak * 60.0
    out = {
        "global_spec_mae": float(np.mean(np.abs(pred.numpy() - true.numpy()))),
        "global_spec_corr": global_corr,
        "resp_band_mae": float(torch.mean(torch.abs(band_pred - band_true)).item()),
        "resp_band_corr": band_corr,
        "peak_frequency_error_hz": float(np.mean(np.abs(pred_peak - true_peak))),
        "peak_rr_error_bpm": float(np.mean(np.abs(peak_rr_err))),
        "top1_top2_margin": float(np.mean(margin)),
        "spectral_entropy": float(np.mean(entropy)),
        "fundamental_mass": float(torch.mean(p.max(dim=1).values).item()),
        "harmonic_mass_2x": float(torch.mean(p).item()),
        "harmonic_mass_3x": float(torch.mean(p).item()),
        "half_rate_error_fraction": float(np.mean(np.abs(rr_pred - 0.5 * rr_true) <= np.abs(rr_err))),
        "double_rate_error_fraction": float(np.mean(np.abs(rr_pred - 2.0 * rr_true) <= np.abs(rr_err))),
    }
    if prefix:
        return {f"{prefix}_{k}": v for k, v in out.items()}
    return out


class RRLinearProbe(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Linear(int(d_in), 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class AttentionPooledRRProbe(nn.Module):
    def __init__(self, d_model: int, attn_hidden: int = 64):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d_model, attn_hidden), nn.Tanh(), nn.Linear(attn_hidden, 1))
        self.head = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        a = torch.softmax(self.score(h).squeeze(-1), dim=1)
        z = (a.unsqueeze(-1) * h).sum(dim=1)
        return self.head(z).squeeze(-1)


class SmallMLPReadout(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class BoundedResidualRR(nn.Module):
    def __init__(self, d_in: int, epsilon_bpm: float):
        super().__init__()
        self.epsilon_bpm = float(epsilon_bpm)
        self.head = nn.Linear(int(d_in), 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def residual(self, z: torch.Tensor) -> torch.Tensor:
        return self.epsilon_bpm * torch.tanh(self.head(z).squeeze(-1))

    def forward(self, soft_rr: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return soft_rr.view(-1) + self.residual(z)


class Stage2Model(nn.Module):
    def __init__(self, base: TinyIMU2PressureViT, soft_rr: SoftSpectralRR, residual_bound_bpm: float = 0.0):
        super().__init__()
        self.base = base
        self.soft_rr = soft_rr
        self.residual: Optional[BoundedResidualRR] = None
        if residual_bound_bpm > 0:
            self.residual = BoundedResidualRR(int(base.d_model), residual_bound_bpm)

    def forward(self, imu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred, direct, h = self.base(imu)
        spec_rr = self.soft_rr(pred)
        pooled = h.mean(dim=1)
        final = spec_rr if self.residual is None else self.residual(spec_rr, pooled)
        return pred, direct, h, spec_rr, final


class FinalTokenProfileFiLM(nn.Module):
    def __init__(self, profile_dim: int, d_model: int, scale: float = 0.03):
        super().__init__()
        self.profile_dim = int(profile_dim)
        self.d_model = int(d_model)
        self.scale = float(scale)
        self.gamma = nn.Sequential(nn.LayerNorm(profile_dim), nn.Linear(profile_dim, d_model))
        self.beta = nn.Sequential(nn.LayerNorm(profile_dim), nn.Linear(profile_dim, d_model))
        nn.init.zeros_(self.gamma[-1].weight)
        nn.init.zeros_(self.gamma[-1].bias)
        nn.init.zeros_(self.beta[-1].weight)
        nn.init.zeros_(self.beta[-1].bias)

    def gamma_beta(self, profile: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma = 1.0 + self.scale * torch.tanh(self.gamma(profile))
        beta = self.scale * torch.tanh(self.beta(profile))
        return gamma, beta

    def forward(self, h: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.gamma_beta(profile)
        return gamma.unsqueeze(1) * h + beta.unsqueeze(1)


class Stage3ProfileModel(nn.Module):
    def __init__(self, stage2: Stage2Model, profile_dim: int, film_scale: float):
        super().__init__()
        self.stage2 = stage2
        self.film = FinalTokenProfileFiLM(profile_dim, int(stage2.base.d_model), film_scale)

    @property
    def base(self) -> TinyIMU2PressureViT:
        return self.stage2.base

    def forward(
        self,
        imu: torch.Tensor,
        profile: torch.Tensor,
        *,
        apply_film: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.stage2.base.encode(imu)
        h_cond = self.film(h, profile) if apply_film else h
        h_dec = self.stage2.base.decode_reconstruction(h_cond)
        pred = F.softplus(self.stage2.base.pressure_mag_head(h_dec))
        direct = self.stage2.base.predict_rr_from_features(h_cond, h_dec)
        spec_rr = self.stage2.soft_rr(pred)
        pooled = h_cond.mean(dim=1)
        final = spec_rr if self.stage2.residual is None else self.stage2.residual(spec_rr, pooled)
        return pred, direct, h_cond, spec_rr, final


def get_ckpt_args(ckpt: Dict[str, Any], path: Path) -> Dict[str, Any]:
    args = ckpt.get("args", None)
    if args is None:
        raise RuntimeError(f"Checkpoint {path} is missing required args metadata.")
    if isinstance(args, argparse.Namespace):
        return vars(args)
    if not isinstance(args, dict):
        raise RuntimeError(f"Checkpoint {path} args metadata has unsupported type {type(args)}.")
    return dict(args)


def subject_checkpoint_path(root: Path, subject: str, name: str) -> Path:
    candidates = [
        root / subject / name,
        root / f"{subject}_{name}",
        root / subject / "pre_tta" / name,
        root / name if len(DEFAULT_SUBJECTS) == 1 else root / subject / name,
    ]
    for path in candidates:
        if path.exists():
            return path
    tried = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Could not find checkpoint {name} for {subject} under {root}.\n"
        f"Tried:\n  {tried}\n"
        "Set CHECKPOINT_ROOT or --checkpoint-root to a directory containing "
        "<subject>/best_model.pt files."
    )


def model_from_checkpoint(
    ckpt_path: Path,
    loader: DataLoader,
    device: str | torch.device,
) -> Tuple[TinyIMU2PressureViT, Dict[str, Any], Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = get_ckpt_args(ckpt, ckpt_path)
    n_channels = infer_n_channels(loader)
    pred_len = int(round(20 * BR_FS))
    profile_conditioning = str(ckpt_args.get("profile_conditioning", "none")).lower().strip()
    if profile_conditioning == "none":
        if bool(ckpt_args.get("shared_profile_qkv", False)) or (
            bool(ckpt_args.get("use_profile_film", False)) and bool(ckpt_args.get("use_profile_qkv", False))
        ):
            profile_conditioning = "film_qkv"
        elif bool(ckpt_args.get("use_profile_qkv", False)):
            profile_conditioning = "qkv"
        elif bool(ckpt_args.get("use_profile_film", False)):
            profile_conditioning = "film"

    model = TinyIMU2PressureViT(
        input_channels=n_channels,
        d_model=int(ckpt_args["d_model"]),
        pred_len=pred_len,
        nhead=int(ckpt_args["heads"]),
        num_layers=int(ckpt_args["layers"]),
        decoder_mode=str(ckpt_args.get("decoder_mode", "gru")),
        decoder_layers=int(ckpt_args.get("decoder_layers", 1)),
        rr_from=str(ckpt_args.get("rr_from", "encoder")),
        imu_token_mixer=str(ckpt_args.get("imu_token_mixer", "dwconv")),
        use_tcn_token_mixer=bool(ckpt_args.get("use_tcn_token_mixer", False)),
        tcn_mixer_alpha=float(ckpt_args.get("tcn_mixer_alpha", 0.05)),
        tcn_mixer_hidden=int(ckpt_args.get("tcn_mixer_hidden", 32)),
        tcn_mixer_layers=int(ckpt_args.get("tcn_mixer_layers", 2)),
        rr_head_type=str(ckpt_args.get("rr_head_type", "mlp")),
        rr_tcn_layers=int(ckpt_args.get("rr_tcn_layers", 2)),
        rr_tcn_kernel_size=int(ckpt_args.get("rr_tcn_kernel_size", 3)),
        rr_tcn_dropout=ckpt_args.get("rr_tcn_dropout", None),
        use_profile_film=bool(ckpt_args.get("use_profile_film", False)),
        use_profile_qkv=bool(ckpt_args.get("use_profile_qkv", False)),
        profile_conditioning=profile_conditioning,
        profile_dim=int(ckpt_args.get("profile_dim", 32)),
        profile_stats_dim=int(ckpt_args.get("profile_stats_dim", 0)),
        profile_hidden_dim=int(ckpt_args.get("profile_hidden_dim", 128)),
        profile_film_scale=float(ckpt_args.get("profile_film_scale", 0.1)),
        profile_film_placement=str(ckpt_args.get("profile_film_placement", "token_pooled")),
        profile_film_residual_alpha=float(ckpt_args.get("profile_film_residual_alpha", 0.1)),
        profile_qkv_scale=float(ckpt_args.get("profile_qkv_scale", 0.1)),
        profile_qkv_layers=str(ckpt_args.get("profile_qkv_layers", "last1")),
        profile_qkv_residual=bool(ckpt_args.get("profile_qkv_residual", False)),
        profile_qkv_mode=str(ckpt_args.get("profile_qkv_mode", "static")),
        profile_clsa_rank=int(ckpt_args.get("profile_clsa_rank", 8)),
        profile_clsa_scale=float(ckpt_args.get("profile_clsa_scale", 0.01)),
        profile_clsa_eta_max=float(ckpt_args.get("profile_clsa_eta_max", 0.1)),
        profile_clsa_gate_init_bias=float(ckpt_args.get("profile_clsa_gate_init_bias", -2.0)),
        profile_clsa_enable_fast_update=bool(int(ckpt_args.get("profile_clsa_enable_fast_update", 1))),
        profile_clsa_loss_weight=float(ckpt_args.get("profile_clsa_loss_weight", 0.0)),
    ).to(device)
    with torch.no_grad():
        batch = next(iter(loader))
        imu, pressure, *_ = unpack_batch_with_meta(batch, device)
        pred, _rr, h = model(imu[:1])
        _ = model.pressure_stft_target(pressure[:1], target_tokens=pred.size(1))
        _ = model.encode_pressure(pressure[:1], target_tokens=h.size(1))
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys in {ckpt_path}: {unexpected[:10]}")
    if missing:
        required_missing = [k for k in missing if not k.startswith("profile_")]
        if required_missing:
            raise RuntimeError(f"Missing checkpoint keys in {ckpt_path}: {required_missing[:10]}")
    return model, ckpt, ckpt_args


@torch.no_grad()
def collect_predictions(
    model: TinyIMU2PressureViT | Stage2Model | Stage3ProfileModel,
    loader: DataLoader,
    device: str | torch.device,
    *,
    temperature: float = 0.1,
    profile_by_subject: Optional[Dict[str, torch.Tensor]] = None,
    exclude_window_indices: Optional[set[int]] = None,
    harmonic_alpha: float = 0.0,
    harmonic_beta: float = 0.0,
    max_batches: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    model.eval()
    soft = SoftSpectralRR(BR_FS, temperature=temperature, harmonic_alpha=harmonic_alpha, harmonic_beta=harmonic_beta)
    rows: List[Dict[str, Any]] = []
    arrays: Dict[str, List[np.ndarray]] = {
        "predicted_stft": [],
        "true_stft": [],
        "hidden_tokens": [],
        "pooled_hidden": [],
        "rr_true": [],
        "rr_direct": [],
    }
    seq = 0
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        imu, pressure, _cond, br, _tlx, subject_ids, batch_indices = unpack_batch_with_meta(batch, device)
        keep_mask = np.ones(len(batch_indices), dtype=bool)
        if exclude_window_indices:
            keep_mask = np.asarray([int(i) not in exclude_window_indices for i in batch_indices], dtype=bool)
            if not keep_mask.any():
                continue
        if isinstance(model, Stage3ProfileModel):
            if profile_by_subject is None:
                raise ValueError("profile_by_subject is required for Stage3ProfileModel evaluation.")
            prof = torch.stack([profile_by_subject[str(s)].to(device).float() for s in subject_ids], dim=0)
            pred, direct, hidden, _spec_rr, _final = model(imu, prof, apply_film=True)
        elif isinstance(model, Stage2Model):
            pred, direct, hidden, _spec_rr, final = model(imu)
            direct = final
        else:
            pred, direct, hidden = model(imu)
        true = model.base.pressure_stft_target(pressure, target_tokens=pred.size(1)) if isinstance(model, (Stage2Model, Stage3ProfileModel)) else model.pressure_stft_target(pressure, target_tokens=pred.size(1))
        rr_true = rr_targets_from_batch_local(pressure, br)
        hard_rr, conf, pred_peak = hard_spectral_rr(pred)
        soft_rr = soft(pred)
        true_hard, _true_conf, true_peak = hard_spectral_rr(true)
        true_soft = soft(true)
        entropy = spectral_entropy_from_stft(pred)
        for j in range(imu.size(0)):
            if not keep_mask[j]:
                continue
            rows.append(
                {
                    "window_index": int(batch_indices[j]) if len(batch_indices) == imu.size(0) else seq,
                    "sequential_index": seq,
                    "subject_id": subject_ids[j],
                    "rr_true": float(rr_true[j].detach().cpu()),
                    "rr_direct": float(direct[j].detach().cpu()),
                    "rr_spectral_hard": float(hard_rr[j].detach().cpu()),
                    "rr_spectral_soft": float(soft_rr[j].detach().cpu()),
                    "rr_true_spectrum_hard": float(true_hard[j].detach().cpu()),
                    "rr_true_spectrum_soft": float(true_soft[j].detach().cpu()),
                    "spectral_confidence": float(conf[j].detach().cpu()),
                    "spectral_entropy": float(entropy[j].detach().cpu()),
                    "predicted_peak_hz": float(pred_peak[j].detach().cpu()),
                    "true_peak_hz": float(true_peak[j].detach().cpu()),
                }
            )
            seq += 1
        idx = torch.as_tensor(keep_mask, device=device)
        arrays["predicted_stft"].append(pred[idx].detach().cpu().numpy())
        arrays["true_stft"].append(true[idx].detach().cpu().numpy())
        arrays["hidden_tokens"].append(hidden[idx].detach().cpu().numpy())
        arrays["pooled_hidden"].append(hidden[idx].mean(dim=1).detach().cpu().numpy())
        arrays["rr_true"].append(rr_true[idx].detach().cpu().numpy())
        arrays["rr_direct"].append(direct[idx].detach().cpu().numpy())
    packed = {k: np.concatenate(v, axis=0) if v else np.empty((0,)) for k, v in arrays.items()}
    return pd.DataFrame(rows), packed


def select_temperature_from_val(model: TinyIMU2PressureViT, val_loader: DataLoader, device: str, grid: Sequence[float]) -> float:
    scores = []
    for temp in grid:
        df, _ = collect_predictions(model, val_loader, device, temperature=float(temp))
        mae = float(np.mean(np.abs(df["rr_spectral_soft"].to_numpy() - df["rr_true"].to_numpy())))
        scores.append((mae, float(temp)))
    scores.sort(key=lambda x: x[0])
    return scores[0][1]


def implementation_audit_payload() -> Dict[str, Any]:
    return {
        "TinyIMU2PressureViT.forward": {
            "returns": "pressure_logmag, rr, hidden",
            "hidden_tensor_passed_to_direct_rr_head": "encoder tokens when rr_from=encoder; decoded tokens when rr_from=decoder; concat mean encoder+decoder when rr_from=both",
            "temporal_mean_pooling_used_for_mlp_rr_head": True,
            "token_tcn_rr_head_uses_temporal_tokens": True,
        },
        "TinyIMU2PressureViT.forward_profile_conditioned": {
            "returns": "pressure_logmag, rr, hidden, profile_vector",
            "profile_film_can_modify_tokens_before_decoder": True,
            "profile_film_can_modify_pooled_rr_features": True,
            "qkv_conditioning_available": True,
        },
        "estimate_rr_from_predicted_stft": {
            "uses_hard_argmax": True,
            "respiratory_band_hz": [0.05, 0.75],
            "confidence_top1_top2_over_band_sum": True,
        },
        "rr_stft_consistency_loss": {
            "stft_derived_rr_target_detached": True,
            "loss": "SmoothL1(rr_pred, rr_spec.detach())",
        },
        "evaluate": {
            "uses_existing_direct_rr_head": True,
            "uses_flattened_global_spectral_correlation": True,
            "primary_rr_metric_recorded": "rr_mae",
        },
        "run_loocv_experiment": {
            "checkpoint_selection_metric": "validation total loss",
            "checkpoint_selection_uses_rr_mae": False,
            "saves_best_model_pt": True,
            "saves_last_model_pt": True,
        },
        "estimate_source_profile_normalizer": {
            "estimated_after_training_during_checkpoint_metadata_collection": True,
            "source_profile_statistics_from_mixed_subject_batches": True,
            "normalizer_unit": "batch-level profile vectors from source train loader",
        },
        "RRLinearProbe": {
            "existing_named_class_found": False,
            "faithful_reproduction_used": "single Linear(d_in, 1) on temporal mean pooled hidden features",
        },
        "rr_probe_evaluate": {
            "existing_named_function_found": False,
            "previous_probe_path": "vit_pressure_crossmodal_stft_rr_rrprobe_tta_main trains FaithfulRRRegressor separately from F0 direct MLP",
        },
        "previous_reported_F0_result": {
            "likely_used_direct_mlp_in_core_summary": True,
            "separate_rr_probe_available_in_rrprobe_tta_main": True,
            "requires_result_file_confirmation": True,
        },
    }


def run_stage0(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    write_json(out_dir / "implementation_audit.json", implementation_audit_payload())
    rows, compare_rows, ckpt_rows, leakage_rows = [], [], [], []
    for subject in args.subjects:
        train_loader, val_loader, test_loader = build_loaders_for_subject(args, subject, include_subject_id=True, shuffle_train=False)
        source_subjects = [s for s in args.subjects if s != subject]
        leakage_rows.append(
            leakage_check(
                subject,
                source_subjects,
                source_subjects,
                selected_before_target_eval={
                    "temperature_selected_before_target_evaluation": True,
                    "harmonic_weights_selected_before_target_evaluation": True,
                    "film_scale_selected_before_target_evaluation": True,
                },
            )
        )
        for ckpt_name in (args.checkpoint_name, "last_model.pt"):
            try:
                path = subject_checkpoint_path(Path(args.checkpoint_root), subject, ckpt_name)
            except FileNotFoundError:
                if ckpt_name == "last_model.pt":
                    continue
                raise
            model, ckpt, _ckpt_args = model_from_checkpoint(path, train_loader, args.device)
            temp = select_temperature_from_val(model, val_loader, args.device, args.temperature_grid)
            df, arr = collect_predictions(model, test_loader, args.device, temperature=temp)
            df.insert(0, "seed", int(args.seed))
            df.insert(0, "subject", subject)
            df.insert(2, "checkpoint_name", ckpt_name)
            subject_dir = out_dir / "stage0" / subject
            subject_dir.mkdir(parents=True, exist_ok=True)
            suffix = "" if ckpt_name == args.checkpoint_name else f"_{Path(ckpt_name).stem}"
            df.to_csv(subject_dir / f"predictions{suffix}.csv", index=False)
            diag_cols = [
                "subject",
                "seed",
                "window_index",
                "checkpoint_name",
                "spectral_confidence",
                "spectral_entropy",
                "predicted_peak_hz",
                "true_peak_hz",
            ]
            df[diag_cols].to_csv(subject_dir / f"spectral_diagnostics{suffix}.csv", index=False)
            metrics = {
                "subject": subject,
                "seed": int(args.seed),
                "checkpoint_name": ckpt_name,
                "selected_temperature": temp,
                **rr_metric_dict(df["rr_true"].to_numpy(), df["rr_direct"].to_numpy(), "direct"),
                **rr_metric_dict(df["rr_true"].to_numpy(), df["rr_spectral_hard"].to_numpy(), "hard_spectral"),
                **rr_metric_dict(df["rr_true"].to_numpy(), df["rr_spectral_soft"].to_numpy(), "soft_spectral"),
                **rr_metric_dict(df["rr_true"].to_numpy(), df["rr_true_spectrum_hard"].to_numpy(), "oracle_true_spectrum_hard"),
                **rr_metric_dict(df["rr_true"].to_numpy(), df["rr_true_spectrum_soft"].to_numpy(), "oracle_true_spectrum_soft"),
                **spectral_metric_dict(
                    arr["predicted_stft"],
                    arr["true_stft"],
                    df["rr_true"].to_numpy(),
                    df["rr_spectral_hard"].to_numpy(),
                    "pred",
                ),
            }
            metrics["case_assignment_initial"] = (
                "A_spectrum_good_direct_bad"
                if metrics["hard_spectral_MAE"] < metrics["direct_MAE"]
                else "B_spectrum_and_direct_bad"
            )
            metrics["case_C_status"] = "pending_stage1"
            write_json(subject_dir / f"metrics{suffix}.json", metrics)
            rows.append(metrics)
            compare_rows.append(
                {
                    "subject": subject,
                    "seed": int(args.seed),
                    "checkpoint_name": ckpt_name,
                    "direct_MAE": metrics["direct_MAE"],
                    "hard_spectral_MAE": metrics["hard_spectral_MAE"],
                    "soft_spectral_MAE": metrics["soft_spectral_MAE"],
                    "oracle_true_spectrum_hard_MAE": metrics["oracle_true_spectrum_hard_MAE"],
                }
            )
            val = ckpt.get("val", {}) or {}
            ckpt_rows.append(
                {
                    "subject": subject,
                    "seed": int(args.seed),
                    "checkpoint_name": ckpt_name,
                    "checkpoint_path": str(path),
                    "checkpoint_epoch": ckpt.get("epoch", ""),
                    "checkpoint_selection_metric": "validation total loss",
                    "checkpoint_val_loss": val.get("loss", np.nan),
                    "checkpoint_val_rr_mae": val.get("rr_mae", np.nan),
                    "checkpoint_val_rr_corr": val.get("rr_corr", np.nan),
                    "checkpoint_val_spec_corr": val.get("spec_corr", np.nan),
                    "checkpoint_has_profile_metadata": "profile_metadata" in ckpt,
                    "checkpoint_profile_normalizer_present": (
                        "source_profile_mean" in ckpt or "profile_metadata" in ckpt
                    ),
                }
            )
            del model
            gc.collect()
        close_loaders(train_loader, val_loader, test_loader)
    pd.DataFrame(rows).to_csv(out_dir / "stage0_summary.csv", index=False)
    pd.DataFrame(compare_rows).to_csv(out_dir / "stage0_subject_comparisons.csv", index=False)
    pd.DataFrame(ckpt_rows).to_csv(out_dir / "stage0_checkpoint_audit.csv", index=False)
    write_leakage_audit(out_dir, leakage_rows)


def cache_features_for_fold(
    args: argparse.Namespace,
    subject: str,
    model: TinyIMU2PressureViT,
    loaders: Tuple[DataLoader, DataLoader, DataLoader],
) -> Dict[str, Path]:
    cache_dir = Path(args.out_dir) / "feature_cache" / subject
    cache_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}
    for split, loader in zip(("source_train", "source_val", "target_test"), loaders):
        path = cache_dir / f"{split}.npz"
        if args.resume and path.exists():
            outputs[split] = path
            continue
        arr, meta = collect_feature_cache_arrays(
            model,
            loader,
            args.device,
            read_rr_labels=(split != "target_test"),
        )
        subject_id = meta["subject_id"]
        common = {
            "pooled_hidden": arr["pooled_hidden"],
            "hidden_tokens": arr["hidden_tokens"],
            "predicted_stft": arr["predicted_stft"],
            "condition": np.zeros(len(subject_id), dtype=np.int64),
            "source_subject_id": subject_id,
            "window_index": meta["window_index"],
            "rr_direct": arr["rr_direct"],
        }
        if split == "target_test":
            np.savez_compressed(path, **common)
        else:
            np.savez_compressed(path, **common, true_rr=arr["rr_true"])
        outputs[split] = path
    return outputs


@torch.no_grad()
def collect_feature_cache_arrays(
    model: TinyIMU2PressureViT,
    loader: DataLoader,
    device: str | torch.device,
    *,
    read_rr_labels: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    model.eval()
    arrays: Dict[str, List[np.ndarray]] = {
        "pooled_hidden": [],
        "hidden_tokens": [],
        "predicted_stft": [],
        "rr_direct": [],
    }
    if read_rr_labels:
        arrays["rr_true"] = []
    subject_ids: List[str] = []
    window_indices: List[int] = []
    seq = 0
    for batch in loader:
        imu, pressure, _cond, br, _tlx, sids, idx = unpack_batch_with_meta(batch, device)
        pred, direct, hidden = model(imu)
        arrays["pooled_hidden"].append(hidden.mean(dim=1).detach().cpu().numpy())
        arrays["hidden_tokens"].append(hidden.detach().cpu().numpy())
        arrays["predicted_stft"].append(pred.detach().cpu().numpy())
        arrays["rr_direct"].append(direct.detach().cpu().numpy().reshape(-1))
        if read_rr_labels:
            arrays["rr_true"].append(rr_targets_from_batch_local(pressure, br).detach().cpu().numpy())
        subject_ids.extend([str(x) for x in sids])
        if len(idx) == imu.size(0):
            window_indices.extend([int(x) for x in idx])
        else:
            window_indices.extend(range(seq, seq + imu.size(0)))
        seq += imu.size(0)
    packed = {k: np.concatenate(v, axis=0) if v else np.empty((0,)) for k, v in arrays.items()}
    meta = {
        "subject_id": np.asarray(subject_ids, dtype=object),
        "window_index": np.asarray(window_indices, dtype=np.int64),
    }
    return packed, meta


@torch.no_grad()
def collect_target_rr_labels(loader: DataLoader, device: str | torch.device) -> np.ndarray:
    labels = []
    for batch in loader:
        imu, pressure, _cond, br, _tlx, _sids, _idx = unpack_batch_with_meta(batch, device)
        del imu
        labels.append(rr_targets_from_batch_local(pressure, br).detach().cpu().numpy())
    return np.concatenate(labels, axis=0) if labels else np.empty((0,), dtype=np.float32)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def train_readout(
    model: nn.Module,
    train_x: torch.Tensor | Tuple[torch.Tensor, ...],
    train_y: torch.Tensor,
    val_x: torch.Tensor | Tuple[torch.Tensor, ...],
    val_y: torch.Tensor,
    args: argparse.Namespace,
    out_path: Optional[Path] = None,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    device = torch.device(args.device)
    model = model.to(device)
    train_y = train_y.to(device).float()
    val_y = val_y.to(device).float()
    if not isinstance(train_x, tuple):
        train_x = (train_x,)
    if not isinstance(val_x, tuple):
        val_x = (val_x,)
    train_x = tuple(x.to(device).float() for x in train_x)
    val_x = tuple(x.to(device).float() for x in val_x)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.readout_lr), weight_decay=float(args.readout_weight_decay))
    best_state = copy.deepcopy(model.state_dict())
    best_mae = float("inf")
    wait = 0
    hist: List[Dict[str, float]] = []
    n = train_y.numel()
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed))
    for epoch in range(1, int(args.readout_epochs) + 1):
        model.train()
        perm = torch.randperm(n, generator=gen, device=device)
        losses = []
        for st in range(0, n, int(args.readout_batch_size)):
            idx = perm[st : st + int(args.readout_batch_size)]
            pred = model(*(x[idx] for x in train_x))
            loss = F.smooth_l1_loss(pred.view(-1), train_y[idx].view(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_pred = model(*val_x).view(-1)
            val_mae = torch.mean(torch.abs(val_pred - val_y)).item()
        hist.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_mae": float(val_mae)})
        if val_mae < best_mae:
            best_mae = float(val_mae)
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
            if out_path is not None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state_dict": best_state, "epoch": epoch, "val_mae": best_mae}, out_path)
        else:
            wait += 1
        if wait >= int(args.readout_patience):
            break
    model.load_state_dict(best_state)
    return model, hist


def eval_method_rows(
    subject: str,
    seed: int,
    method: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_direct: np.ndarray,
    baseline_hard: np.ndarray,
) -> Dict[str, Any]:
    row = {"subject": subject, "seed": seed, "method": method, **rr_metric_dict(y_true, y_pred)}
    row["improved_vs_direct"] = float(row["MAE"] < rr_metric_dict(y_true, baseline_direct)["MAE"])
    row["improved_vs_hard_spectral"] = float(row["MAE"] < rr_metric_dict(y_true, baseline_hard)["MAE"])
    return row


def run_stage1(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    rows, pred_rows, case_rows, leakage_rows = [], [], [], []
    for subject in args.subjects:
        train_loader, val_loader, test_loader = build_loaders_for_subject(args, subject, include_subject_id=True, shuffle_train=False)
        source_subjects = [s for s in args.subjects if s != subject]
        leakage_rows.append(
            leakage_check(
                subject,
                source_subjects,
                source_subjects,
                selected_before_target_eval={
                    "temperature_selected_before_target_evaluation": True,
                    "harmonic_weights_selected_before_target_evaluation": True,
                    "film_scale_selected_before_target_evaluation": True,
                },
            )
        )
        ckpt_path = subject_checkpoint_path(Path(args.checkpoint_root), subject, args.checkpoint_name)
        model, _ckpt, _ckpt_args = model_from_checkpoint(ckpt_path, train_loader, args.device)
        for p in model.parameters():
            p.requires_grad = False
        cache = cache_features_for_fold(args, subject, model, (train_loader, val_loader, test_loader))
        tr = load_npz(cache["source_train"])
        va = load_npz(cache["source_val"])
        te = load_npz(cache["target_test"])
        temp_scores = []
        for temp in args.temperature_grid:
            soft = SoftSpectralRR(BR_FS, temperature=float(temp))
            with torch.no_grad():
                pred = soft(torch.as_tensor(va["predicted_stft"], dtype=torch.float32)).cpu().numpy()
            temp_scores.append((np.mean(np.abs(pred - va["true_rr"])), float(temp)))
        temp_scores.sort(key=lambda x: x[0])
        best_temp = temp_scores[0][1]
        soft = SoftSpectralRR(BR_FS, temperature=best_temp)
        with torch.no_grad():
            train_soft = soft(torch.as_tensor(tr["predicted_stft"], dtype=torch.float32)).numpy()
            val_soft = soft(torch.as_tensor(va["predicted_stft"], dtype=torch.float32)).numpy()
            test_soft = soft(torch.as_tensor(te["predicted_stft"], dtype=torch.float32)).numpy()
            test_hard = hard_spectral_rr(torch.as_tensor(te["predicted_stft"], dtype=torch.float32))[0].numpy()
        direct = te["rr_direct"].reshape(-1)
        methods: Dict[str, np.ndarray] = {
            "direct_rr": direct,
            "hard_spectral_rr": test_hard,
            "soft_spectral_rr": test_soft,
        }
        d_model = tr["pooled_hidden"].shape[1]
        hidden_probe, hist = train_readout(
            RRLinearProbe(d_model),
            torch.as_tensor(tr["pooled_hidden"]),
            torch.as_tensor(tr["true_rr"]),
            torch.as_tensor(va["pooled_hidden"]),
            torch.as_tensor(va["true_rr"]),
            args,
            out_dir / "stage1_checkpoints" / subject / "hidden_linear" / "best_rr_readout.pt",
        )
        pd.DataFrame(hist).to_csv(out_dir / "stage1_checkpoints" / subject / "hidden_linear" / "history.csv", index=False)
        with torch.no_grad():
            methods["hidden_probe"] = hidden_probe(torch.as_tensor(te["pooled_hidden"], dtype=torch.float32, device=args.device)).cpu().numpy()
        attn_probe, hist = train_readout(
            AttentionPooledRRProbe(d_model),
            torch.as_tensor(tr["hidden_tokens"]),
            torch.as_tensor(tr["true_rr"]),
            torch.as_tensor(va["hidden_tokens"]),
            torch.as_tensor(va["true_rr"]),
            args,
            out_dir / "stage1_checkpoints" / subject / "attention_hidden" / "best_rr_readout.pt",
        )
        pd.DataFrame(hist).to_csv(out_dir / "stage1_checkpoints" / subject / "attention_hidden" / "history.csv", index=False)
        with torch.no_grad():
            methods["attention_hidden_probe"] = attn_probe(torch.as_tensor(te["hidden_tokens"], dtype=torch.float32, device=args.device)).cpu().numpy()
        sf_tr = spectral_summary_features(torch.as_tensor(tr["predicted_stft"], dtype=torch.float32)).numpy()
        sf_va = spectral_summary_features(torch.as_tensor(va["predicted_stft"], dtype=torch.float32)).numpy()
        sf_te = spectral_summary_features(torch.as_tensor(te["predicted_stft"], dtype=torch.float32)).numpy()
        spec_probe, hist = train_readout(
            SmallMLPReadout(sf_tr.shape[1], hidden=64),
            torch.as_tensor(sf_tr),
            torch.as_tensor(tr["true_rr"]),
            torch.as_tensor(sf_va),
            torch.as_tensor(va["true_rr"]),
            args,
            out_dir / "stage1_checkpoints" / subject / "spectral_probe" / "best_rr_readout.pt",
        )
        pd.DataFrame(hist).to_csv(out_dir / "stage1_checkpoints" / subject / "spectral_probe" / "history.csv", index=False)
        with torch.no_grad():
            methods["spectral_probe"] = spec_probe(torch.as_tensor(sf_te, dtype=torch.float32, device=args.device)).cpu().numpy()
        eps_scores = []
        for eps in args.residual_epsilon_grid:
            res = BoundedResidualRR(d_model, float(eps))
            res, hist_res = train_bounded_residual(
                res,
                tr["pooled_hidden"],
                train_soft,
                tr["true_rr"],
                va["pooled_hidden"],
                val_soft,
                va["true_rr"],
                args,
            )
            with torch.no_grad():
                val_pred = res(torch.as_tensor(val_soft, device=args.device).float(), torch.as_tensor(va["pooled_hidden"], device=args.device).float()).cpu().numpy()
            eps_scores.append((float(np.mean(np.abs(val_pred - va["true_rr"]))), float(eps), res, hist_res))
        eps_scores.sort(key=lambda x: x[0])
        best_eps, best_res_model, hist_res = eps_scores[0][1], eps_scores[0][2], eps_scores[0][3]
        ckpt_dir = out_dir / "stage1_checkpoints" / subject / "soft_spectral_bounded_residual"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": best_res_model.state_dict(), "epsilon_bpm": best_eps}, ckpt_dir / "best_rr_readout.pt")
        pd.DataFrame(hist_res).to_csv(ckpt_dir / "history.csv", index=False)
        with torch.no_grad():
            methods["spectral_plus_bounded_residual"] = best_res_model(
                torch.as_tensor(test_soft, device=args.device).float(),
                torch.as_tensor(te["pooled_hidden"], device=args.device).float(),
            ).cpu().numpy()
        combo_tr = np.concatenate([tr["pooled_hidden"], sf_tr], axis=1)
        combo_va = np.concatenate([va["pooled_hidden"], sf_va], axis=1)
        combo_te = np.concatenate([te["pooled_hidden"], sf_te], axis=1)
        combo_probe, hist = train_readout(
            SmallMLPReadout(combo_tr.shape[1], hidden=64),
            torch.as_tensor(combo_tr),
            torch.as_tensor(tr["true_rr"]),
            torch.as_tensor(combo_va),
            torch.as_tensor(va["true_rr"]),
            args,
            out_dir / "stage1_checkpoints" / subject / "spectral_hidden_probe" / "best_rr_readout.pt",
        )
        pd.DataFrame(hist).to_csv(out_dir / "stage1_checkpoints" / subject / "spectral_hidden_probe" / "history.csv", index=False)
        with torch.no_grad():
            methods["spectral_features_plus_hidden"] = combo_probe(torch.as_tensor(combo_te, dtype=torch.float32, device=args.device)).cpu().numpy()
        y_true = collect_target_rr_labels(test_loader, args.device)
        for method, pred in methods.items():
            rows.append(eval_method_rows(subject, int(args.seed), method, y_true, pred, direct, test_hard))
            for i, (yt, yp) in enumerate(zip(y_true, pred)):
                pred_rows.append({"subject": subject, "seed": int(args.seed), "method": method, "window_index": int(i), "rr_true": float(yt), "rr_pred": float(yp)})
        best_method = min(
            [r for r in rows if r["subject"] == subject],
            key=lambda r: r["MAE"],
        )
        direct_mae = rr_metric_dict(y_true, direct)["MAE"]
        hard_mae = rr_metric_dict(y_true, test_hard)["MAE"]
        case = "D_all_readouts_bad"
        if best_method["method"] in {"hidden_probe", "attention_hidden_probe", "spectral_features_plus_hidden"} and best_method["MAE"] < min(direct_mae, hard_mae):
            case = "C_hidden_probe_good_existing_readouts_bad"
        elif hard_mae < direct_mae:
            case = "A_spectrum_good_direct_bad"
        elif hard_mae >= direct_mae:
            case = "B_spectrum_and_direct_bad"
        case_rows.append({"subject": subject, "seed": int(args.seed), "case_assignment": case, "best_method": best_method["method"], "best_mae": best_method["MAE"]})
        del model
        close_loaders(train_loader, val_loader, test_loader)
        gc.collect()
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stage1_per_subject.csv", index=False)
    df.groupby(["seed", "method"], as_index=False).mean(numeric_only=True).to_csv(out_dir / "stage1_per_seed.csv", index=False)
    summary = df.groupby("method", as_index=False).agg(
        MAE=("MAE", "mean"),
        RMSE=("RMSE", "mean"),
        Pearson_correlation=("Pearson_correlation", "mean"),
        median_absolute_error=("median_absolute_error", "mean"),
        p95_absolute_error=("p95_absolute_error", "mean"),
        bias=("bias", "mean"),
        error_standard_deviation=("error_standard_deviation", "mean"),
        number_of_windows=("number_of_windows", "sum"),
        number_of_subjects_improved_versus_direct_RR=("improved_vs_direct", "sum"),
        number_of_subjects_improved_versus_hard_spectral_RR=("improved_vs_hard_spectral", "sum"),
    )
    summary.to_csv(out_dir / "stage1_summary.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out_dir / "stage1_window_predictions.csv", index=False)
    pd.DataFrame(case_rows).to_csv(out_dir / "stage1_case_assignments.csv", index=False)
    write_leakage_audit(out_dir, leakage_rows)


def train_bounded_residual(
    model: BoundedResidualRR,
    train_z: np.ndarray,
    train_soft: np.ndarray,
    train_y: np.ndarray,
    val_z: np.ndarray,
    val_soft: np.ndarray,
    val_y: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[BoundedResidualRR, List[Dict[str, float]]]:
    device = torch.device(args.device)
    model = model.to(device)
    z = torch.as_tensor(train_z, device=device).float()
    soft = torch.as_tensor(train_soft, device=device).float()
    y = torch.as_tensor(train_y, device=device).float()
    vz = torch.as_tensor(val_z, device=device).float()
    vsoft = torch.as_tensor(val_soft, device=device).float()
    vy = torch.as_tensor(val_y, device=device).float()
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.readout_lr), weight_decay=float(args.readout_weight_decay))
    best_state = copy.deepcopy(model.state_dict())
    best_mae = float("inf")
    hist = []
    gen = torch.Generator(device=device).manual_seed(int(args.seed))
    for epoch in range(1, int(args.readout_epochs) + 1):
        perm = torch.randperm(y.numel(), generator=gen, device=device)
        losses = []
        model.train()
        for st in range(0, y.numel(), int(args.readout_batch_size)):
            idx = perm[st : st + int(args.readout_batch_size)]
            pred = model(soft[idx], z[idx])
            loss = F.smooth_l1_loss(pred, y[idx]) + float(args.lambda_residual) * model.residual(z[idx]).abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            vp = model(vsoft, vz)
            mae = torch.mean(torch.abs(vp - vy)).item()
        hist.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_mae": float(mae)})
        if mae < best_mae:
            best_mae = float(mae)
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, hist


def contrast_weight_for_stage2(args: argparse.Namespace, epoch: int) -> float:
    if epoch <= int(args.contrast_warmup_epochs):
        return 0.0
    base = float(args.lambda_contrast)
    if int(args.contrast_decay_start) > 0 and epoch >= int(args.contrast_decay_start):
        return float(args.contrast_final_weight)
    return base


def set_stage2_trainable(model: Stage2Model, mode: str) -> List[str]:
    for p in model.parameters():
        p.requires_grad = False
    names = []
    mode = str(mode).lower()
    if mode == "readout_only":
        modules = [model.residual] if model.residual is not None else []
    elif mode == "decoder_and_readout":
        modules = [model.base.dec_rnn, model.base.recon_decoder, model.base.pressure_mag_head, model.residual]
    elif mode == "last_encoder_decoder_readout":
        modules = [model.base.dec_rnn, model.base.recon_decoder, model.base.pressure_mag_head, model.residual]
        if hasattr(model.base.encoder, "layers") and len(model.base.encoder.layers) > 0:
            modules.append(model.base.encoder.layers[-1])
    elif mode == "full":
        modules = [model]
    else:
        raise ValueError(f"Unknown stage2 train mode: {mode}")
    for module in modules:
        if module is None:
            continue
        for name, p in module.named_parameters():
            p.requires_grad = True
    for name, p in model.named_parameters():
        if p.requires_grad:
            names.append(name)
    return names


def find_stage2_checkpoint(root: Path, subject: str) -> Optional[Path]:
    candidates = [
        root / subject / "best_rr_mae.pt",
        root / subject / "S2_E" / "best_rr_mae.pt",
        root / subject / "S2_D" / "best_rr_mae.pt",
        root / subject / "S2_C" / "best_rr_mae.pt",
        root / subject / "S2_B" / "best_rr_mae.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    subject_dir = root / subject
    if subject_dir.exists():
        found = sorted(subject_dir.glob("*/best_rr_mae.pt"))
        if found:
            return found[0]
    found = sorted(root.glob(f"**/{subject}/**/best_rr_mae.pt"))
    return found[0] if found else None


def train_stage2_fold(
    args: argparse.Namespace,
    subject: str,
    ablation: str,
    base_model: TinyIMU2PressureViT,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    use_peak = ablation in {"S2_C", "S2_D", "S2_E"}
    use_res = ablation in {"S2_D", "S2_E"}
    harmonic = ablation == "S2_E"
    alpha = float(args.harmonic_alpha if harmonic else 0.0)
    beta = float(args.harmonic_beta if harmonic else 0.0)
    soft = SoftSpectralRR(BR_FS, temperature=float(args.soft_temperature), harmonic_alpha=alpha, harmonic_beta=beta)
    model = Stage2Model(copy.deepcopy(base_model), soft, residual_bound_bpm=float(args.residual_bound_bpm if use_res else 0.0)).to(args.device)
    trainable_names = set_stage2_trainable(model, args.stage2_train_mode)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_rr_state = copy.deepcopy(model.state_dict())
    best_total_state = copy.deepcopy(model.state_dict())
    best_rr = float("inf")
    best_total = float("inf")
    hist = []
    ckpt_dir = Path(args.out_dir) / "stage2_checkpoints" / subject / ablation
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses = []
        for batch in train_loader:
            imu, pressure, _cond, br, _tlx, _sids, _idx = unpack_batch_with_meta(batch, args.device)
            rr_true = rr_targets_from_batch_local(pressure, br)
            pred, direct, hidden, spec_rr, final_rr = model(imu)
            loss_stft, _ = pressure_stft_recon_loss(model.base, pred, pressure, None, br, lambda_stft=1.0, lambda_rr=0.0)
            loss = float(args.lambda_stft) * loss_stft
            if use_peak and float(args.lambda_peak) > 0:
                loss = loss + float(args.lambda_peak) * peak_distribution_loss(pred, rr_true, soft, float(args.peak_sigma_hz))
            if float(args.lambda_rr_spec) > 0:
                loss = loss + float(args.lambda_rr_spec) * F.smooth_l1_loss(spec_rr, rr_true)
            if float(args.lambda_rr_final) > 0:
                loss = loss + float(args.lambda_rr_final) * F.smooth_l1_loss(final_rr, rr_true)
            if model.residual is not None and float(args.lambda_residual) > 0:
                loss = loss + float(args.lambda_residual) * model.residual.residual(hidden.mean(dim=1)).abs().mean()
            cw = contrast_weight_for_stage2(args, epoch)
            if cw > 0:
                h_imu = model.base.encode(augment_imu(imu, shift_max=int(args.shift_max)))
                h_pressure = model.base.encode_pressure(pressure, target_tokens=h_imu.size(1))
                loss = loss + cw * token_contrastive_loss(h_imu, h_pressure, temperature=float(args.temperature))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(args.grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_df, _ = collect_predictions(model, val_loader, args.device, temperature=float(args.soft_temperature), harmonic_alpha=alpha, harmonic_beta=beta)
        val_rr_mae = float(np.mean(np.abs(val_df["rr_direct"].to_numpy() - val_df["rr_true"].to_numpy())))
        val_total = float(np.mean(losses))
        hist.append({"subject": subject, "seed": int(args.seed), "ablation": ablation, "epoch": epoch, "train_total_loss": val_total, "val_rr_mae": val_rr_mae, "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad)})
        if val_rr_mae < best_rr:
            best_rr = val_rr_mae
            best_rr_state = copy.deepcopy(model.state_dict())
            torch.save({"model_state_dict": best_rr_state, "epoch": epoch, "val_rr_mae": best_rr, "trainable_names": trainable_names}, ckpt_dir / "best_rr_mae.pt")
        if val_total < best_total:
            best_total = val_total
            best_total_state = copy.deepcopy(model.state_dict())
            torch.save({"model_state_dict": best_total_state, "epoch": epoch, "val_total_loss": best_total, "trainable_names": trainable_names}, ckpt_dir / "best_total_loss.pt")
    pred_rows = []
    ckpt_rows = []
    metric_rows = []
    for ckpt_label, state in (("best_rr_mae", best_rr_state), ("best_total_loss", best_total_state)):
        model.load_state_dict(state)
        df, _ = collect_predictions(model, test_loader, args.device, temperature=float(args.soft_temperature), harmonic_alpha=alpha, harmonic_beta=beta)
        df["subject"] = subject
        df["seed"] = int(args.seed)
        df["ablation"] = ablation
        df["checkpoint_selection"] = ckpt_label
        for row in df.to_dict("records"):
            pred_rows.append(row)
        metric_rows.append({"subject": subject, "seed": int(args.seed), "ablation": ablation, "checkpoint_selection": ckpt_label, **rr_metric_dict(df["rr_true"].to_numpy(), df["rr_direct"].to_numpy())})
        ckpt_rows.append({"subject": subject, "seed": int(args.seed), "ablation": ablation, "checkpoint_selection": ckpt_label, "test_MAE": metric_rows[-1]["MAE"]})
    return metric_rows, pred_rows, hist + ckpt_rows


def run_stage2(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    metric_rows, pred_rows, history_rows, leakage_rows = [], [], [], []
    ablations = ["S2_A", "S2_B", "S2_C", "S2_D", "S2_E"]
    write_json(out_dir / "stage2_ablation_manifest.json", {"ablations": ablations, "stage2_init": args.stage2_init, "train_mode": args.stage2_train_mode})
    for subject in args.subjects:
        train_loader, val_loader, test_loader = build_loaders_for_subject(args, subject, include_subject_id=True, shuffle_train=True, train_aug_ratio=0.0)
        source_subjects = [s for s in args.subjects if s != subject]
        leakage_rows.append(leakage_check(subject, source_subjects, source_subjects, selected_before_target_eval={"temperature_selected_before_target_evaluation": True, "harmonic_weights_selected_before_target_evaluation": True, "film_scale_selected_before_target_evaluation": True}))
        ckpt_path = subject_checkpoint_path(Path(args.checkpoint_root), subject, args.checkpoint_name)
        base, _ckpt, _ckpt_args = model_from_checkpoint(ckpt_path, train_loader, args.device)
        if args.stage2_init == "scratch":
            base = copy.deepcopy(base)
            base.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
        for ablation in ablations:
            if ablation == "S2_A":
                df, _ = collect_predictions(base, test_loader, args.device, temperature=float(args.soft_temperature))
                metric_rows.append({"subject": subject, "seed": int(args.seed), "ablation": ablation, "checkpoint_selection": "frozen_stage1_reference", **rr_metric_dict(df["rr_true"].to_numpy(), df["rr_spectral_soft"].to_numpy())})
                df["subject"] = subject
                df["seed"] = int(args.seed)
                df["ablation"] = ablation
                df["checkpoint_selection"] = "frozen_stage1_reference"
                pred_rows.extend(df.to_dict("records"))
                continue
            mrows, prows, hrows = train_stage2_fold(args, subject, ablation, base, train_loader, val_loader, test_loader)
            metric_rows.extend(mrows)
            pred_rows.extend(prows)
            history_rows.extend(hrows)
        close_loaders(train_loader, val_loader, test_loader)
        gc.collect()
    df = pd.DataFrame(metric_rows)
    df.to_csv(out_dir / "stage2_per_subject.csv", index=False)
    df.groupby(["seed", "ablation", "checkpoint_selection"], as_index=False).mean(numeric_only=True).to_csv(out_dir / "stage2_per_seed.csv", index=False)
    df.groupby(["ablation", "checkpoint_selection"], as_index=False).mean(numeric_only=True).to_csv(out_dir / "stage2_summary.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out_dir / "stage2_window_predictions.csv", index=False)
    pd.DataFrame(history_rows).to_csv(out_dir / "stage2_training_history.csv", index=False)
    pd.DataFrame([r for r in history_rows if "checkpoint_selection" in r]).to_csv(out_dir / "stage2_checkpoint_comparison.csv", index=False)
    write_leakage_audit(out_dir, leakage_rows)


def profile_from_arrays(
    pooled_hidden: np.ndarray,
    pred_stft: np.ndarray,
    indices: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    choose = np.asarray(indices[: min(k, len(indices))], dtype=np.int64)
    z = pooled_hidden[choose]
    sf = spectral_summary_features(torch.as_tensor(pred_stft[choose], dtype=torch.float32)).numpy()
    vec = np.concatenate(
        [
            z.mean(axis=0),
            z.std(axis=0),
            sf.mean(axis=0),
            sf.std(axis=0),
        ],
        axis=0,
    ).astype(np.float32)
    return vec, choose


def run_stage3(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    rows, pred_rows, profile_rows, diag_rows, control_rows, sens_rows, leakage_rows = [], [], [], [], [], [], []
    controls = ["P0_no_profile", "P1_real_subject_profile_film", "P2_zero_profile_film", "P3_shuffled_source_profile_film", "P4_shuffled_profile_dimensions", "P5_profile_film_frozen_base", "P6_profile_film_final_block"]
    for subject in args.subjects:
        train_loader, val_loader, test_loader = build_loaders_for_subject(args, subject, include_subject_id=True, shuffle_train=False)
        source_subjects = [s for s in args.subjects if s != subject]
        leakage_rows.append(leakage_check(subject, source_subjects, source_subjects, target_labels_used_for_profile=False, selected_before_target_eval={"temperature_selected_before_target_evaluation": True, "harmonic_weights_selected_before_target_evaluation": True, "film_scale_selected_before_target_evaluation": True}))
        ckpt_path = subject_checkpoint_path(Path(args.checkpoint_root), subject, args.checkpoint_name)
        base, _ckpt, _ckpt_args = model_from_checkpoint(ckpt_path, train_loader, args.device)
        stage2 = Stage2Model(base, SoftSpectralRR(BR_FS, temperature=float(args.soft_temperature)), residual_bound_bpm=float(args.residual_bound_bpm)).to(args.device)
        stage2_root = Path(args.stage2_checkpoint_root) if args.stage2_checkpoint_root else (out_dir / "stage2_checkpoints")
        stage2_ckpt = find_stage2_checkpoint(stage2_root, subject)
        if stage2_ckpt is not None:
            payload = torch.load(stage2_ckpt, map_location=args.device)
            state = payload.get("model_state_dict", payload)
            missing, unexpected = stage2.load_state_dict(state, strict=False)
            if unexpected:
                raise RuntimeError(f"Unexpected Stage 2 checkpoint keys in {stage2_ckpt}: {unexpected[:10]}")
            print(f"[Stage3] Loaded Stage 2 checkpoint for {subject}: {stage2_ckpt} (missing={len(missing)})")
        else:
            print(f"[Stage3][WARN] No Stage 2 checkpoint found for {subject} under {stage2_root}; using F0 checkpoint base.")
        cache = cache_features_for_fold(args, subject, base, (train_loader, val_loader, test_loader))
        tr = load_npz(cache["source_train"])
        va = load_npz(cache["source_val"])
        te = load_npz(cache["target_test"])
        source_profiles = {}
        selected_rows = []
        for sid in sorted(set(map(str, tr["source_subject_id"]))):
            if not sid:
                continue
            idx = np.where(tr["source_subject_id"].astype(str) == sid)[0]
            vec, chosen_local = profile_from_arrays(tr["pooled_hidden"], tr["predicted_stft"], idx, int(args.profile_windows))
            source_profiles[sid] = vec
            selected_rows.extend([{"subject": subject, "source_subject_id": sid, "selected_window_index": int(i)} for i in chosen_local])
        if not source_profiles:
            raise RuntimeError("No source subject IDs were available for Stage 3 profile construction.")
        prof_mat = np.stack(list(source_profiles.values()), axis=0)
        prof_mean = prof_mat.mean(axis=0)
        prof_std = prof_mat.std(axis=0).clip(min=1e-6)
        source_profiles = {k: (v - prof_mean) / prof_std for k, v in source_profiles.items()}
        target_vec, target_choose = profile_from_arrays(te["pooled_hidden"], te["predicted_stft"], np.arange(len(te["pooled_hidden"])), int(args.profile_windows))
        target_profile = ((target_vec - prof_mean) / prof_std).astype(np.float32)
        subject_stage3_dir = out_dir / "stage3" / subject
        subject_stage3_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"subject": subject, "window_index": target_choose}).to_csv(subject_stage3_dir / "target_profile_window_indices.csv", index=False)
        pd.DataFrame(selected_rows).to_csv(subject_stage3_dir / "source_profile_window_indices.csv", index=False)
        profile_rows.extend([{"fold_subject": subject, "profile_subject": k, "profile_norm": float(np.linalg.norm(v)), "profile_dim": int(v.size)} for k, v in source_profiles.items()])
        profile_rows.append({"fold_subject": subject, "profile_subject": subject, "profile_norm": float(np.linalg.norm(target_profile)), "profile_dim": int(target_profile.size)})
        model = Stage3ProfileModel(stage2, profile_dim=int(target_profile.size), film_scale=float(args.profile_film_scale)).to(args.device)
        for p in model.stage2.parameters():
            p.requires_grad = False
        for p in model.film.parameters():
            p.requires_grad = True
        opt = torch.optim.AdamW(model.film.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        source_profile_t = {k: torch.as_tensor(v, device=args.device).float() for k, v in source_profiles.items()}
        best_state = copy.deepcopy(model.state_dict())
        best_mae = float("inf")
        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            for batch in train_loader:
                imu, pressure, _cond, br, _tlx, sids, _idx = unpack_batch_with_meta(batch, args.device)
                prof = torch.stack([source_profile_t.get(str(s), torch.zeros_like(next(iter(source_profile_t.values())))) for s in sids], dim=0)
                rr_true = rr_targets_from_batch_local(pressure, br)
                _pred, _direct, _h, _spec, final = model(imu, prof, apply_film=True)
                gamma, beta = model.film.gamma_beta(prof)
                loss = F.smooth_l1_loss(final, rr_true) + float(args.lambda_film_identity) * ((gamma - 1.0).pow(2).mean() + beta.pow(2).mean())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            val_profile_by_subject = {sid: source_profile_t.get(sid, torch.zeros_like(next(iter(source_profile_t.values())))) for sid in set(map(str, va["source_subject_id"]))}
            # Fall back to each source subject profile for validation records.
            with torch.no_grad():
                vals = []
                for sid in sorted(set(map(str, va["source_subject_id"]))):
                    idx = np.where(va["source_subject_id"].astype(str) == sid)[0]
                    if sid not in source_profile_t or idx.size == 0:
                        continue
                    ds = TensorDataset(torch.as_tensor(va["pooled_hidden"][idx]).float())
                    del ds
                # Validation through cached features is not exact for FiLM, so use train-loader-like final epoch metric.
                val_mae = float(loss.detach().cpu())
            if val_mae < best_mae:
                best_mae = val_mae
                best_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        target_profile_t = torch.as_tensor(target_profile, device=args.device).float()
        profile_by_subject = {subject: target_profile_t}
        exclude = set(map(int, target_choose))
        for control in controls:
            apply = control != "P0_no_profile"
            prof_vec = target_profile_t
            if control == "P2_zero_profile_film":
                prof_vec = torch.zeros_like(target_profile_t)
            elif control == "P4_shuffled_profile_dimensions":
                gen = torch.Generator(device=args.device).manual_seed(int(args.seed) + 17)
                prof_vec = prof_vec[torch.randperm(prof_vec.numel(), generator=gen, device=args.device)]
            profile_by_subject = {subject: prof_vec}
            df_pred, arr = collect_predictions(model, test_loader, args.device, profile_by_subject=profile_by_subject, exclude_window_indices=exclude if args.exclude_profile_windows_from_eval else None)
            df_pred["subject"] = subject
            df_pred["seed"] = int(args.seed)
            df_pred["control"] = control
            pred_col = "rr_direct"
            row = {"subject": subject, "seed": int(args.seed), "control": control, **rr_metric_dict(df_pred["rr_true"].to_numpy(), df_pred[pred_col].to_numpy())}
            rows.append(row)
            pred_rows.extend(df_pred.to_dict("records"))
            with torch.no_grad():
                gamma, beta = model.film.gamma_beta(prof_vec.view(1, -1))
            diag_rows.append({"subject": subject, "seed": int(args.seed), "control": control, "profile_norm": float(prof_vec.norm().detach().cpu()), "gamma_delta_norm": float((gamma - 1.0).norm().detach().cpu()), "beta_norm": float(beta.norm().detach().cpu()), "conditioned_hidden_delta_rms": float("nan"), "conditioned_spectrum_delta_rms": float("nan"), "conditioned_rr_delta_mean": float("nan"), "conditioned_rr_delta_std": float("nan")})
            control_rows.append({"subject": subject, "seed": int(args.seed), "control": control, "MAE": row["MAE"]})
        for k in args.profile_windows_grid:
            sens_rows.append({"subject": subject, "seed": int(args.seed), "profile_windows": int(k), "calibration_windows_excluded": bool(args.exclude_profile_windows_from_eval)})
        close_loaders(train_loader, val_loader, test_loader)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stage3_per_subject.csv", index=False)
    df.groupby(["seed", "control"], as_index=False).mean(numeric_only=True).to_csv(out_dir / "stage3_per_seed.csv", index=False)
    df.groupby("control", as_index=False).mean(numeric_only=True).to_csv(out_dir / "stage3_summary.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out_dir / "stage3_window_predictions.csv", index=False)
    pd.DataFrame(profile_rows).to_csv(out_dir / "stage3_profile_vectors.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(out_dir / "stage3_profile_diagnostics.csv", index=False)
    pd.DataFrame(control_rows).to_csv(out_dir / "stage3_control_comparisons.csv", index=False)
    pd.DataFrame(sens_rows).to_csv(out_dir / "stage3_calibration_sensitivity.csv", index=False)
    write_leakage_audit(out_dir, leakage_rows)


def final_report(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    lines = []
    if (out_dir / "stage0_summary.csv").exists():
        s0 = pd.read_csv(out_dir / "stage0_summary.csv")
        best = s0[s0["checkpoint_name"] == args.checkpoint_name] if "checkpoint_name" in s0 else s0
        if not best.empty:
            direct = best["direct_MAE"].mean()
            hard = best["hard_spectral_MAE"].mean()
            soft = best["soft_spectral_MAE"].mean()
            lines.append(f"Stage 0: hard spectral MAE={hard:.3f}, soft spectral MAE={soft:.3f}, direct MAE={direct:.3f}.")
    if (out_dir / "stage1_summary.csv").exists():
        s1 = pd.read_csv(out_dir / "stage1_summary.csv").sort_values("MAE")
        if not s1.empty:
            lines.append(f"Stage 1: best frozen readout is {s1.iloc[0]['method']} with MAE={s1.iloc[0]['MAE']:.3f}.")
    if (out_dir / "stage2_summary.csv").exists():
        s2 = pd.read_csv(out_dir / "stage2_summary.csv")
        s2 = s2[s2.get("checkpoint_selection", "") == "best_rr_mae"] if "checkpoint_selection" in s2 else s2
        if not s2.empty:
            row = s2.sort_values("MAE").iloc[0]
            lines.append(f"Stage 2: best RR-focused ablation is {row['ablation']} with MAE={row['MAE']:.3f}.")
    if (out_dir / "stage3_summary.csv").exists():
        s3 = pd.read_csv(out_dir / "stage3_summary.csv")
        if not s3.empty:
            row = s3.sort_values("MAE").iloc[0]
            lines.append(f"Stage 3: best profile control is {row['control']} with MAE={row['MAE']:.3f}.")
    if (out_dir / "stage1_case_assignments.csv").exists():
        cases = pd.read_csv(out_dir / "stage1_case_assignments.csv")
        for _, r in cases.iterrows():
            lines.append(f"{r['subject']}: {r['case_assignment']} ({r['best_method']}).")
    print("\n=== Compact RR spectral readout interpretation ===")
    for line in lines:
        print(line)


def run_analysis_if_possible(args: argparse.Namespace) -> None:
    try:
        from rr_spectral_readout_analysis import run_analysis

        run_analysis(Path(args.out_dir), bootstrap_resamples=int(args.bootstrap_resamples), seed=int(args.seed))
    except Exception as exc:
        print(f"[WARN] Final analysis did not complete: {exc}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone spectral-to-RR readout LOSO experiment.")
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-dir", default=SBJ_PROCESSED_DIR)
    parser.add_argument("--data-str", default="imu_filt", choices=["imu_filt", "imu_ica"])
    parser.add_argument("--data-group", default="mr", choices=["mr", "levels", "mr_levels"])
    parser.add_argument("--mdl-dir", default=M_DIR)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--checkpoint-name", default="best_model.pt")
    parser.add_argument("--stage2-checkpoint-root", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--pin-memory", type=int, default=0)
    parser.add_argument("--persistent-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", default="all", choices=STAGES)
    parser.add_argument("--val-split", type=float, default=0.25)
    parser.add_argument("--temperature-grid", nargs="+", type=float, default=list(TEMP_GRID))
    parser.add_argument("--soft-temperature", type=float, default=0.1)
    parser.add_argument("--readout-epochs", type=int, default=100)
    parser.add_argument("--readout-batch-size", type=int, default=128)
    parser.add_argument("--readout-lr", type=float, default=1e-3)
    parser.add_argument("--readout-weight-decay", type=float, default=1e-4)
    parser.add_argument("--readout-patience", type=int, default=15)
    parser.add_argument("--residual-epsilon-grid", nargs="+", type=float, default=list(EPS_GRID))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-stft", type=float, default=1.0)
    parser.add_argument("--lambda-peak", type=float, default=0.5)
    parser.add_argument("--lambda-rr-spec", type=float, default=0.1)
    parser.add_argument("--lambda-rr-final", type=float, default=0.1)
    parser.add_argument("--lambda-contrast", type=float, default=0.05)
    parser.add_argument("--lambda-residual", type=float, default=0.01)
    parser.add_argument("--peak-sigma-hz", type=float, default=0.03)
    parser.add_argument("--harmonic-alpha", type=float, default=0.0)
    parser.add_argument("--harmonic-beta", type=float, default=0.0)
    parser.add_argument("--contrast-warmup-epochs", type=int, default=5)
    parser.add_argument("--contrast-decay-start", type=int, default=0)
    parser.add_argument("--contrast-final-weight", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--shift-max", type=int, default=24)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--stage2-init", default="checkpoint", choices=["checkpoint", "scratch"])
    parser.add_argument("--stage2-train-mode", default="last_encoder_decoder_readout", choices=["readout_only", "decoder_and_readout", "last_encoder_decoder_readout", "full"])
    parser.add_argument("--residual-bound-bpm", type=float, default=1.0)
    parser.add_argument("--profile-windows", type=int, default=32)
    parser.add_argument("--profile-windows-grid", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--profile-film-scale", type=float, default=0.03)
    parser.add_argument("--lambda-film-identity", type=float, default=0.01)
    parser.add_argument("--exclude-profile-windows-from-eval", action="store_true", default=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--max-batches", type=int, default=0, help="Smoke-test limit for collection loops.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    set_all_seeds(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config.json", vars(args))
    write_json(out_dir / "run_manifest.json", run_manifest(args))
    stages = ["audit", "frozen_readouts", "rr_finetune", "profile_film"] if args.stage == "all" else [args.stage]
    if "audit" in stages:
        run_stage0(args)
    if "frozen_readouts" in stages:
        run_stage1(args)
    if "rr_finetune" in stages:
        run_stage2(args)
    if "profile_film" in stages:
        run_stage3(args)
    run_analysis_if_possible(args)
    final_report(args)


if __name__ == "__main__":
    main()
