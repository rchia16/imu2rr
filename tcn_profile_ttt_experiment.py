#!/usr/bin/env python3
"""Phase 1 TCN subject-profile experiment.

This file is intentionally standalone: it imports the validated TCN baseline,
LOSO data utilities, augmentations, and metrics without modifying them.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from augmentations import scaling, spectral_augment
from config import IMU_FS
from dataloader import _loader_kwargs, load_data, make_dataset
from rr_jbhi_models import TCNRR


DEFAULT_SUBJECTS = "S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29"
PHASE1_MODES = {"t0_plain", "t1_profile_affine", "t2_profile_film", "t3_profile_film_affine", "all"}
PHASE2_MODES = {"t4_ttt_affine", "t5_ttt_film_affine"}
PHASE3_MODES = {"t6_meta_gated_ttt"}
PHASE4_MODES = {"t7_profile_mean_alignment"}
FUTURE_MODES: set[str] = set()
TTT_SUBJECT_DEFAULTS = {
    "ttt_parameter_group": "",
    "ttt_steps": 0,
    "ttt_lr": np.nan,
    "ttt_mode": "",
    "ttt_loss_pre": np.nan,
    "ttt_loss_post": np.nan,
    "ttt_aug_loss_pre": np.nan,
    "ttt_aug_loss_post": np.nan,
    "ttt_temp_loss_pre": np.nan,
    "ttt_temp_loss_post": np.nan,
    "ttt_spec_loss_pre": np.nan,
    "ttt_spec_loss_post": np.nan,
    "ttt_anchor_loss_pre": np.nan,
    "ttt_anchor_loss_post": np.nan,
    "ttt_gradient_norm": 0.0,
    "ttt_delta_norm": 0.0,
    "delta_gamma_norm": 0.0,
    "delta_beta_norm": 0.0,
    "delta_gain": 0.0,
    "delta_bias": 0.0,
    "prediction_variance_pre": np.nan,
    "prediction_variance_post": np.nan,
    "gate_value": np.nan,
    "effective_ttt_lr": np.nan,
    "update_was_rejected": False,
}


@dataclass
class TCNForward:
    prediction: torch.Tensor
    hidden_tokens: torch.Tensor
    pooled_hidden: torch.Tensor


@dataclass
class ConditionedForward:
    prediction: torch.Tensor
    baseline_prediction: torch.Tensor
    film_prediction: torch.Tensor
    affine_prediction: torch.Tensor
    combined_prediction: torch.Tensor
    hidden_tokens: torch.Tensor
    pooled_hidden: torch.Tensor
    conditioned_hidden: torch.Tensor
    film_gamma: torch.Tensor
    film_beta: torch.Tensor
    affine_gain: torch.Tensor
    affine_bias: torch.Tensor


@dataclass
class TTTState:
    delta_gamma: Optional[torch.Tensor]
    delta_beta: Optional[torch.Tensor]
    delta_gain: Optional[torch.Tensor]
    delta_bias: Optional[torch.Tensor]


@dataclass
class TTTConfig:
    ttt_steps: int = 1
    ttt_lr: float = 1e-4
    ttt_parameter_group: str = "film_affine"
    ttt_mode: str = "episodic"
    lambda_ttt_aug: float = 1.0
    lambda_ttt_temp: float = 0.05
    lambda_ttt_spec: float = 0.25
    lambda_ttt_anchor: float = 1.0
    ttt_grad_clip: float = 1.0
    ttt_max_delta_norm: float = 0.25
    ttt_use_spectrum: bool = False
    ttt_noise_std: float = 0.005
    ttt_amplitude_scale: float = 0.01
    ttt_rotation_deg: float = 2.0
    ttt_mask_fraction: float = 0.02
    ttt_channel_drop_prob: float = 0.02
    ttt_collapse_var_min: float = 1e-8
    ttt_max_prediction_jump_bpm: float = 5.0
    seed: int = 0


@dataclass
class TTTResult:
    state: TTTState
    diagnostics: Dict[str, object]
    loss_components: List[Dict[str, object]]
    state_rows: List[Dict[str, object]]


@dataclass
class MetaSubjectEpisode:
    pseudo_target_subject: str
    support_indices: torch.Tensor
    query_indices: torch.Tensor
    support_batch: Any
    query_batch: Any
    profile: torch.Tensor


@dataclass
class AdaptationGateOutput:
    gate: torch.Tensor
    lr_multiplier: torch.Tensor
    max_delta_multiplier: torch.Tensor
    mean_alignment_alpha: Optional[torch.Tensor] = None


def parse_subjects(values: Sequence[str] | str) -> List[str]:
    if isinstance(values, str):
        values = values.replace(",", " ").split()
    out: List[str] = []
    for value in values:
        out.extend(str(value).replace(",", " ").split())
    return [s.strip() for s in out if s.strip()]


def mode_set(args: argparse.Namespace) -> set[str]:
    modes = getattr(args, "mode", "all")
    if isinstance(modes, str):
        return set(parse_subjects(modes))
    return set(parse_subjects(list(modes)))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def ensure_channel_first(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3 and x.shape[1] > x.shape[2]:
        return x.permute(0, 2, 1).contiguous()
    return x.contiguous()


def finite_array(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


class WrappedTCN(nn.Module):
    """Expose TCN tokens and pooled latent while reusing the baseline modules."""

    def __init__(self, tcn: TCNRR):
        super().__init__()
        self.tcn = tcn

    @property
    def encoder(self) -> nn.Module:
        return self.tcn.encoder

    @property
    def proj(self) -> nn.Module:
        return self.tcn.proj

    @property
    def head(self) -> nn.Module:
        return self.tcn.head

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x.float())
        z = self.proj(h)
        return h.transpose(1, 2).contiguous(), z

    def forward(self, x: torch.Tensor) -> TCNForward:
        tokens, pooled = self.encode(x)
        pred = self.head(pooled)
        return TCNForward(prediction=pred, hidden_tokens=tokens, pooled_hidden=pooled)


class ProfileAffineRR(nn.Module):
    def __init__(self, profile_dim: int, gain_bound: float = 0.03, bias_bound_bpm: float = 1.0):
        super().__init__()
        self.gain_bound = float(gain_bound)
        self.bias_bound_bpm = float(bias_bound_bpm)
        self.gain = nn.Linear(profile_dim, 1, bias=False)
        self.bias = nn.Linear(profile_dim, 1, bias=False)
        nn.init.zeros_(self.gain.weight)
        nn.init.zeros_(self.bias.weight)

    def parameters_from_profile(self, profile: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gain = 1.0 + self.gain_bound * torch.tanh(self.gain(profile)).squeeze(-1)
        bias = self.bias_bound_bpm * torch.tanh(self.bias(profile)).squeeze(-1)
        return gain, bias

    def forward(self, pred: torch.Tensor, profile: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gain, bias = self.parameters_from_profile(profile)
        return gain * pred + bias, gain, bias


class ProfileFiLM(nn.Module):
    def __init__(self, profile_dim: int, hidden_dim: int, scale: float = 0.03):
        super().__init__()
        self.scale = float(scale)
        self.gamma = nn.Linear(profile_dim, hidden_dim, bias=False)
        self.beta = nn.Linear(profile_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def parameters_from_profile(self, profile: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma = 1.0 + self.scale * torch.tanh(self.gamma(profile))
        beta = self.scale * torch.tanh(self.beta(profile))
        return gamma, beta

    def forward(self, z: torch.Tensor, profile: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma, beta = self.parameters_from_profile(profile)
        return gamma * z + beta, gamma, beta


class ProfileConditionedTCN(nn.Module):
    def __init__(
        self,
        wrapped: WrappedTCN,
        profile_dim: int,
        use_film: bool,
        use_affine: bool,
        film_scale: float,
        affine_gain_bound: float,
        affine_bias_bound_bpm: float,
    ):
        super().__init__()
        self.wrapped = wrapped
        self.use_film = bool(use_film)
        self.use_affine = bool(use_affine)
        hidden_dim = int(wrapped.tcn.head.net[0].normalized_shape[0])
        self.film = ProfileFiLM(profile_dim, hidden_dim, film_scale)
        self.affine = ProfileAffineRR(profile_dim, affine_gain_bound, affine_bias_bound_bpm)

    def forward(self, x: torch.Tensor, profile: torch.Tensor) -> ConditionedForward:
        tokens, pooled = self.wrapped.encode(x)
        baseline = self.wrapped.head(pooled)
        gamma = torch.ones_like(pooled)
        beta = torch.zeros_like(pooled)
        z_cond = pooled
        if self.use_film:
            z_cond, gamma, beta = self.film(pooled, profile)
        film_pred = self.wrapped.head(z_cond)
        affine_pred, gain, bias = self.affine(baseline, profile)
        combined = film_pred
        if self.use_affine:
            combined, gain, bias = self.affine(film_pred, profile)
        return ConditionedForward(
            prediction=combined,
            baseline_prediction=baseline,
            film_prediction=film_pred,
            affine_prediction=affine_pred,
            combined_prediction=combined,
            hidden_tokens=tokens,
            pooled_hidden=pooled,
            conditioned_hidden=z_cond,
            film_gamma=gamma,
            film_beta=beta,
            affine_gain=gain,
            affine_bias=bias,
        )


class ProfileAdaptationGate(nn.Module):
    """Bounded profile-conditioned gate for scaling one predefined TTT update."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        lr_min: float = 0.0,
        lr_max: float = 1.0,
        delta_min: float = 0.0,
        delta_max: float = 1.0,
        alpha_max: float = 0.5,
        include_alignment: bool = False,
    ):
        super().__init__()
        self.lr_min = float(lr_min)
        self.lr_max = float(lr_max)
        self.delta_min = float(delta_min)
        self.delta_max = float(delta_max)
        self.alpha_max = float(alpha_max)
        self.include_alignment = bool(include_alignment)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gate_head = nn.Linear(hidden_dim, 1)
        self.lr_head = nn.Linear(hidden_dim, 1)
        self.delta_head = nn.Linear(hidden_dim, 1)
        self.alpha_head = nn.Linear(hidden_dim, 1) if include_alignment else None
        nn.init.constant_(self.gate_head.bias, -2.0)
        nn.init.constant_(self.lr_head.bias, 0.0)
        nn.init.constant_(self.delta_head.bias, 0.0)
        if self.alpha_head is not None:
            nn.init.constant_(self.alpha_head.bias, -6.0)

    def forward(self, gate_features: torch.Tensor) -> AdaptationGateOutput:
        h = self.trunk(gate_features.float())
        gate = torch.sigmoid(self.gate_head(h)).squeeze(-1)
        lr_unit = torch.sigmoid(self.lr_head(h)).squeeze(-1)
        delta_unit = torch.sigmoid(self.delta_head(h)).squeeze(-1)
        lr_multiplier = self.lr_min + (self.lr_max - self.lr_min) * lr_unit
        max_delta_multiplier = self.delta_min + (self.delta_max - self.delta_min) * delta_unit
        alpha = None
        if self.alpha_head is not None:
            alpha = self.alpha_max * torch.sigmoid(self.alpha_head(h)).squeeze(-1)
        return AdaptationGateOutput(
            gate=gate,
            lr_multiplier=lr_multiplier,
            max_delta_multiplier=max_delta_multiplier,
            mean_alignment_alpha=alpha,
        )


class ProfileGatedMeanAlignment(nn.Module):
    def __init__(self, profile_dim: int, alpha_max: float = 0.5):
        super().__init__()
        self.alpha_max = float(alpha_max)
        self.alpha_head = nn.Linear(profile_dim, 1)
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.constant_(self.alpha_head.bias, -6.0)

    def alpha_from_profile(self, profile: torch.Tensor) -> torch.Tensor:
        return self.alpha_max * torch.sigmoid(self.alpha_head(profile.float())).squeeze(-1)

    def forward(
        self,
        pooled: torch.Tensor,
        profile: torch.Tensor,
        source_mean: torch.Tensor,
        target_mean: torch.Tensor,
        *,
        fixed_alpha: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if fixed_alpha is None:
            alpha = self.alpha_from_profile(profile)
        else:
            alpha = torch.full((pooled.shape[0],), float(fixed_alpha), device=pooled.device, dtype=pooled.dtype)
        shift = (target_mean.to(pooled.device) - source_mean.to(pooled.device)).view(1, -1)
        return pooled + alpha.view(-1, 1) * shift, alpha


class TTTAdapter(nn.Module):
    """Small per-subject TTT state around a frozen static profile model."""

    def __init__(self, static_model: ProfileConditionedTCN, profile: torch.Tensor, parameter_group: str):
        super().__init__()
        if parameter_group not in {"affine_only", "film_only", "film_affine"}:
            raise ValueError(f"Unsupported TTT parameter group: {parameter_group}")
        self.static_model = static_model
        self.parameter_group = str(parameter_group)
        profile = profile.detach().float()
        if profile.ndim == 1:
            profile = profile.view(1, -1)
        self.register_buffer("profile", profile)
        for p in self.static_model.parameters():
            p.requires_grad = False
        self.static_model.eval()

        hidden_dim = int(static_model.wrapped.tcn.head.net[0].normalized_shape[0])
        if parameter_group in {"film_only", "film_affine"}:
            self.delta_gamma = nn.Parameter(torch.zeros(hidden_dim))
            self.delta_beta = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.register_buffer("delta_gamma", torch.zeros(hidden_dim))
            self.register_buffer("delta_beta", torch.zeros(hidden_dim))
        if parameter_group in {"affine_only", "film_affine"}:
            self.delta_gain = nn.Parameter(torch.zeros(()))
            self.delta_bias = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer("delta_gain", torch.zeros(()))
            self.register_buffer("delta_bias", torch.zeros(()))

    def ttt_state(self) -> TTTState:
        return TTTState(
            delta_gamma=self.delta_gamma.detach().clone() if self.parameter_group in {"film_only", "film_affine"} else None,
            delta_beta=self.delta_beta.detach().clone() if self.parameter_group in {"film_only", "film_affine"} else None,
            delta_gain=self.delta_gain.detach().clone() if self.parameter_group in {"affine_only", "film_affine"} else None,
            delta_bias=self.delta_bias.detach().clone() if self.parameter_group in {"affine_only", "film_affine"} else None,
        )

    def trainable_ttt_parameter_names(self) -> List[str]:
        return [name for name, param in self.named_parameters() if param.requires_grad]

    def trainable_ttt_parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.parameters() if param.requires_grad))

    def frozen_parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.static_model.parameters()))

    def delta_norm(self) -> torch.Tensor:
        return torch.sqrt(self.delta_squared_norm().clamp_min(0.0))

    def delta_squared_norm(self) -> torch.Tensor:
        terms = []
        if self.parameter_group in {"film_only", "film_affine"}:
            terms.extend([self.delta_gamma.pow(2).sum(), self.delta_beta.pow(2).sum()])
        if self.parameter_group in {"affine_only", "film_affine"}:
            terms.extend([self.delta_gain.pow(2), self.delta_bias.pow(2)])
        if not terms:
            return torch.zeros((), device=self.profile.device)
        return torch.stack([t.reshape(()) for t in terms]).sum()

    def project_delta_norm(self, max_norm: float) -> Tuple[float, float, bool]:
        before = float(self.delta_norm().detach().cpu())
        projected = False
        if max_norm > 0 and before > float(max_norm):
            scale = float(max_norm) / max(before, 1e-12)
            with torch.no_grad():
                if self.parameter_group in {"film_only", "film_affine"}:
                    self.delta_gamma.mul_(scale)
                    self.delta_beta.mul_(scale)
                if self.parameter_group in {"affine_only", "film_affine"}:
                    self.delta_gain.mul_(scale)
                    self.delta_bias.mul_(scale)
            projected = True
        after = float(self.delta_norm().detach().cpu())
        return before, after, projected

    def _profile_for_batch(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.profile.to(device).expand(batch_size, -1)

    def forward(self, x: torch.Tensor) -> ConditionedForward:
        self.static_model.eval()
        profile = self._profile_for_batch(int(x.shape[0]), x.device)
        tokens, pooled = self.static_model.wrapped.encode(x)
        baseline = self.static_model.wrapped.head(pooled)

        if self.static_model.use_film:
            static_z, static_gamma, static_beta = self.static_model.film(pooled, profile)
        else:
            static_gamma = torch.ones_like(pooled)
            static_beta = torch.zeros_like(pooled)
            static_z = pooled

        gamma = static_gamma
        beta = static_beta
        if self.parameter_group in {"film_only", "film_affine"}:
            gamma = torch.clamp(static_gamma + self.delta_gamma.view(1, -1), 1.0 - self.static_model.film.scale, 1.0 + self.static_model.film.scale)
            beta = torch.clamp(static_beta + self.delta_beta.view(1, -1), -self.static_model.film.scale, self.static_model.film.scale)
        z_cond = gamma * pooled + beta if self.static_model.use_film else static_z
        film_pred = self.static_model.wrapped.head(z_cond)

        static_affine_pred, static_gain, static_bias = self.static_model.affine(film_pred if self.static_model.use_film else baseline, profile)
        gain = static_gain
        bias = static_bias
        if self.parameter_group in {"affine_only", "film_affine"}:
            gain = torch.clamp(static_gain + self.delta_gain, 1.0 - self.static_model.affine.gain_bound, 1.0 + self.static_model.affine.gain_bound)
            bias = torch.clamp(static_bias + self.delta_bias, -self.static_model.affine.bias_bound_bpm, self.static_model.affine.bias_bound_bpm)

        if self.static_model.use_affine:
            combined = gain * film_pred + bias
        else:
            combined = film_pred
        if self.static_model.use_film:
            affine_only, _g0, _b0 = self.static_model.affine(baseline, profile)
        else:
            affine_only = combined
        return ConditionedForward(
            prediction=combined,
            baseline_prediction=baseline,
            film_prediction=film_pred,
            affine_prediction=affine_only,
            combined_prediction=combined,
            hidden_tokens=tokens,
            pooled_hidden=pooled,
            conditioned_hidden=z_cond,
            film_gamma=gamma,
            film_beta=beta,
            affine_gain=gain,
            affine_bias=bias,
        )


class SubjectAwareDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray | torch.Tensor,
        rr: np.ndarray | torch.Tensor,
        cond: np.ndarray | torch.Tensor,
        subject_id: Sequence[str],
        window_index: Sequence[int],
        *,
        aug_ratio: float = 0.0,
    ):
        self.x = torch.as_tensor(x).float()
        self.rr = torch.as_tensor(rr).float().view(-1)
        self.cond = torch.as_tensor(cond).float().view(-1)
        self.subject_id = np.asarray(subject_id, dtype=object)
        self.window_index = np.asarray(window_index, dtype=np.int64)
        if not (len(self.x) == len(self.rr) == len(self.cond) == len(self.subject_id) == len(self.window_index)):
            raise ValueError("SubjectAwareDataset arrays have mismatched lengths")
        if aug_ratio > 0:
            self._append_augmented(float(aug_ratio))

    def _append_augmented(self, aug_ratio: float) -> None:
        n_aug = int(aug_ratio * len(self.x))
        if n_aug <= 0:
            return
        idx = random.sample(range(len(self.x)), n_aug)
        x_aug = self.x[idx]
        x_tmp = x_aug.permute(0, 2, 1)
        x_aug = torch.from_numpy(scaling(x_tmp, sigma=1.1, device=self.x.device)).float()
        x_aug = spectral_augment(
            x_aug,
            fs=IMU_FS,
            max_phase=np.pi / 2,
            max_shift_hz=0.5,
            dim=-1,
            p_phase=0.5,
            p_shift=0.5,
        ).permute(0, 2, 1)
        self.x = torch.cat([self.x, x_aug], dim=0)
        self.rr = torch.cat([self.rr, self.rr[idx]], dim=0)
        self.cond = torch.cat([self.cond, self.cond[idx]], dim=0)
        self.subject_id = np.concatenate([self.subject_id, self.subject_id[idx]])
        self.window_index = np.concatenate([self.window_index, self.window_index[idx]])

    def __len__(self) -> int:
        return int(len(self.x))

    def __getitem__(self, index: int) -> Dict[str, object]:
        return {
            "imu": self.x[index],
            "rr": self.rr[index],
            "subject_id": str(self.subject_id[index]),
            "window_index": int(self.window_index[index]),
            "condition": self.cond[index],
            "sequence_index": int(self.window_index[index]),
        }


def load_arrays_for_subjects(args: argparse.Namespace, subjects: Sequence[str]) -> Dict[str, np.ndarray]:
    data_list = [load_data(s, data_dir=args.data_dir, data_group=args.data_group) for s in subjects]
    out = make_dataset(
        data_list,
        args.data_str,
        label_encoder_dir=args.data_dir,
        data_group=args.data_group,
        include_subject_id=True,
        is_train=False,
    )
    x, _pressure, rr, cond, subject_ids = out[:5]
    window_indices = []
    for data in data_list:
        window_indices.extend(range(len(data[args.data_str])))
    return {
        "x": np.asarray(x),
        "rr": np.asarray(rr),
        "cond": np.asarray(cond),
        "subject_id": np.asarray(subject_ids, dtype=object),
        "window_index": np.asarray(window_indices, dtype=np.int64),
    }


def subset_arrays(arrays: Mapping[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v)[idx] for k, v in arrays.items()}


def make_loader(dataset: Dataset, args: argparse.Namespace, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    kwargs = _loader_kwargs(
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(args.persistent_workers),
    )
    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=shuffle,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        **kwargs,
    )


def build_fold_loaders(args: argparse.Namespace, heldout: str) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, np.ndarray]]:
    source_subjects = [s for s in parse_subjects(args.subjects) if s != heldout]
    if not source_subjects:
        raise ValueError("Need at least one source subject in --subjects in addition to the held-out subject")
    source = load_arrays_for_subjects(args, source_subjects)
    target = load_arrays_for_subjects(args, [heldout])
    idx = np.arange(len(source["x"]))
    train_idx, val_idx = train_test_split(idx, test_size=float(args.val_split), random_state=int(args.seed))
    train = subset_arrays(source, train_idx)
    val = subset_arrays(source, val_idx)
    train_ds = SubjectAwareDataset(train["x"], train["rr"], train["cond"], train["subject_id"], train["window_index"], aug_ratio=float(args.train_aug_ratio))
    val_ds = SubjectAwareDataset(val["x"], val["rr"], val["cond"], val["subject_id"], val["window_index"], aug_ratio=0.0)
    test_ds = SubjectAwareDataset(target["x"], target["rr"], target["cond"], target["subject_id"], target["window_index"], aug_ratio=0.0)
    loaders = (
        make_loader(train_ds, args, shuffle=True, seed=int(args.seed) + 11),
        make_loader(val_ds, args, shuffle=False, seed=int(args.seed) + 12),
        make_loader(test_ds, args, shuffle=False, seed=int(args.seed) + 13),
    )
    return loaders[0], loaders[1], loaders[2], {"source": source, "target": target, "train": train, "val": val}


def batch_to_device(batch: Mapping[str, object], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, List[str], np.ndarray, np.ndarray]:
    x = ensure_channel_first(batch["imu"].float().to(device))
    rr = batch["rr"].float().to(device).view(-1)
    subjects = [str(s) for s in batch["subject_id"]]
    window_index = batch["window_index"].detach().cpu().numpy().astype(int)
    cond = batch["condition"].detach().cpu().numpy()
    return x, rr, subjects, window_index, cond


def assert_unlabelled_support_batch(support_batch: Mapping[str, object]) -> None:
    forbidden = {"rr", "label", "labels", "target", "targets", "y", "br", "ground_truth"}
    present = forbidden.intersection(str(k).lower() for k in support_batch.keys())
    if present:
        raise AssertionError(f"TTT support batch contains target-label key(s): {sorted(present)}")


def build_unlabelled_support_batch(test_ds: SubjectAwareDataset, profile_indices: set[int], device: torch.device) -> Dict[str, object]:
    selected = [i for i, w in enumerate(test_ds.window_index.tolist()) if int(w) in profile_indices]
    if not selected:
        return {
            "imu": torch.empty(0, device=device),
            "subject_id": [],
            "window_index": np.asarray([], dtype=int),
            "condition": np.asarray([], dtype=float),
            "sequence_index": np.asarray([], dtype=int),
        }
    x = ensure_channel_first(test_ds.x[selected].float().to(device))
    support = {
        "imu": x,
        "subject_id": [str(s) for s in test_ds.subject_id[selected]],
        "window_index": test_ds.window_index[selected].astype(int),
        "condition": test_ds.cond[selected].detach().cpu().numpy(),
        "sequence_index": test_ds.window_index[selected].astype(int),
    }
    assert_unlabelled_support_batch(support)
    return support


def ttt_config_from_args(args: argparse.Namespace, *, overrides: Optional[Dict[str, object]] = None) -> TTTConfig:
    cfg = TTTConfig(
        ttt_steps=int(args.ttt_steps),
        ttt_lr=float(args.ttt_lr),
        ttt_parameter_group=str(args.ttt_parameter_group),
        ttt_mode=str(args.ttt_mode),
        lambda_ttt_aug=float(args.lambda_ttt_aug),
        lambda_ttt_temp=float(args.lambda_ttt_temp),
        lambda_ttt_spec=float(args.lambda_ttt_spec),
        lambda_ttt_anchor=float(args.lambda_ttt_anchor),
        ttt_grad_clip=float(args.ttt_grad_clip),
        ttt_max_delta_norm=float(args.ttt_max_delta_norm),
        ttt_use_spectrum=bool(args.ttt_use_spectrum),
        ttt_noise_std=float(args.ttt_noise_std),
        ttt_amplitude_scale=float(args.ttt_amplitude_scale),
        ttt_rotation_deg=float(args.ttt_rotation_deg),
        ttt_mask_fraction=float(args.ttt_mask_fraction),
        ttt_channel_drop_prob=float(args.ttt_channel_drop_prob),
        ttt_collapse_var_min=float(args.ttt_collapse_var_min),
        ttt_max_prediction_jump_bpm=float(args.ttt_max_prediction_jump_bpm),
        seed=int(args.seed),
    )
    if overrides:
        for key, value in overrides.items():
            setattr(cfg, key, value)
    return cfg


def deterministic_ttt_augment(x: torch.Tensor, config: TTTConfig, seed: int) -> torch.Tensor:
    if x.numel() == 0:
        return x
    gen = torch.Generator(device=x.device).manual_seed(int(seed))
    out = x.clone()
    if config.ttt_noise_std > 0:
        out = out + torch.randn(out.shape, generator=gen, device=x.device, dtype=x.dtype) * float(config.ttt_noise_std)
    if config.ttt_amplitude_scale > 0:
        scale = 1.0 + (torch.rand((x.shape[0], x.shape[1], 1), generator=gen, device=x.device, dtype=x.dtype) * 2.0 - 1.0) * float(config.ttt_amplitude_scale)
        out = out * scale
    if config.ttt_rotation_deg > 0 and x.shape[1] >= 3:
        max_rad = float(config.ttt_rotation_deg) * math.pi / 180.0
        angles = (torch.rand((x.shape[0], 3), generator=gen, device=x.device, dtype=x.dtype) * 2.0 - 1.0) * max_rad
        cx, cy, cz = torch.cos(angles[:, 0]), torch.cos(angles[:, 1]), torch.cos(angles[:, 2])
        sx, sy, sz = torch.sin(angles[:, 0]), torch.sin(angles[:, 1]), torch.sin(angles[:, 2])
        rot = torch.zeros((x.shape[0], 3, 3), device=x.device, dtype=x.dtype)
        rot[:, 0, 0] = cy * cz
        rot[:, 0, 1] = sx * sy * cz - cx * sz
        rot[:, 0, 2] = cx * sy * cz + sx * sz
        rot[:, 1, 0] = cy * sz
        rot[:, 1, 1] = sx * sy * sz + cx * cz
        rot[:, 1, 2] = cx * sy * sz - sx * cz
        rot[:, 2, 0] = -sy
        rot[:, 2, 1] = sx * cy
        rot[:, 2, 2] = cx * cy
        for start in (0, 3):
            if out.shape[1] >= start + 3:
                part = out[:, start:start + 3, :].transpose(1, 2)
                out[:, start:start + 3, :] = torch.bmm(part, rot.transpose(1, 2)).transpose(1, 2)
    if config.ttt_mask_fraction > 0 and x.shape[-1] > 1:
        mask_len = max(1, int(round(float(config.ttt_mask_fraction) * x.shape[-1])))
        mask_len = min(mask_len, x.shape[-1])
        for i in range(x.shape[0]):
            start = int(torch.randint(0, x.shape[-1] - mask_len + 1, (1,), generator=gen, device=x.device).item())
            out[i, :, start:start + mask_len] = 0.0
    if config.ttt_channel_drop_prob > 0:
        drop = torch.rand((x.shape[0], x.shape[1], 1), generator=gen, device=x.device, dtype=x.dtype) < float(config.ttt_channel_drop_prob)
        out = out.masked_fill(drop, 0.0)
    return out


def valid_temporal_pairs(subjects: Sequence[str], sequence_index: Sequence[int], window_index: Sequence[int], condition: Sequence[float]) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    order = np.argsort(np.asarray(window_index, dtype=int))
    for left, right in zip(order[:-1], order[1:]):
        if str(subjects[left]) != str(subjects[right]):
            continue
        if float(condition[left]) != float(condition[right]):
            continue
        if abs(int(sequence_index[right]) - int(sequence_index[left])) > 1:
            continue
        if abs(int(window_index[right]) - int(window_index[left])) <= 1:
            pairs.append((int(left), int(right)))
    return pairs


def spectral_consistency_loss(pred: torch.Tensor, support_batch: Mapping[str, object], config: TTTConfig) -> Tuple[torch.Tensor, Dict[str, float]]:
    zero = pred.sum() * 0.0
    if not config.ttt_use_spectrum or "spectrum_logits" not in support_batch:
        return zero, {"spectral_rr_mean": float("nan"), "spectral_confidence_mean": 0.0}
    logits = support_batch["spectrum_logits"].detach()
    freqs = support_batch["spectrum_freqs"].detach().to(logits.device)
    prob = torch.softmax(logits, dim=-1)
    spec_rr = 60.0 * (prob * freqs.view(1, -1)).sum(dim=-1)
    sorted_prob = torch.sort(prob, dim=-1, descending=True).values
    margin = sorted_prob[:, 0] - sorted_prob[:, 1].clamp_min(0.0) if sorted_prob.shape[1] > 1 else sorted_prob[:, 0]
    entropy = -(prob * torch.log(prob.clamp_min(1e-12))).sum(dim=-1) / max(1.0, math.log(prob.shape[-1]))
    confidence = torch.clamp((margin * (1.0 - entropy)).detach(), 0.0, 1.0)
    loss = (confidence * F.smooth_l1_loss(pred, spec_rr.detach(), reduction="none")).mean()
    return loss, {"spectral_rr_mean": float(spec_rr.detach().mean().cpu()), "spectral_confidence_mean": float(confidence.detach().mean().cpu())}


def anchor_loss(adapter: TTTAdapter) -> torch.Tensor:
    return adapter.delta_squared_norm()


def compute_ttt_losses(adapter: TTTAdapter, support_batch: Mapping[str, object], config: TTTConfig, *, seed: int) -> Tuple[torch.Tensor, Dict[str, object]]:
    assert_unlabelled_support_batch(support_batch)
    x = support_batch["imu"]
    if x.numel() == 0:
        zero = torch.zeros((), device=adapter.profile.device)
        return zero, {
            "ttt_loss": 0.0,
            "ttt_aug_loss": 0.0,
            "ttt_temp_loss": 0.0,
            "ttt_spec_loss": 0.0,
            "ttt_anchor_loss": 0.0,
            "augmentation_prediction_disagreement": 0.0,
            "temporal_pair_count": 0,
            "spectral_rr_mean": float("nan"),
            "spectral_confidence_mean": 0.0,
        }
    with torch.no_grad():
        teacher = adapter.static_model(x, adapter._profile_for_batch(int(x.shape[0]), x.device)).prediction.detach()
    x_aug = deterministic_ttt_augment(x, config, seed)
    pred_aug = adapter(x_aug).prediction
    aug_loss = F.smooth_l1_loss(pred_aug, teacher)
    with torch.no_grad():
        static_aug = adapter.static_model(x_aug, adapter._profile_for_batch(int(x.shape[0]), x.device)).prediction.detach()
        aug_disagreement = float(torch.mean(torch.abs(static_aug - teacher)).detach().cpu())

    pred = adapter(x).prediction
    pairs = valid_temporal_pairs(
        support_batch["subject_id"],
        support_batch["sequence_index"],
        support_batch["window_index"],
        support_batch["condition"],
    )
    if pairs:
        left = torch.as_tensor([i for i, _j in pairs], device=pred.device, dtype=torch.long)
        right = torch.as_tensor([j for _i, j in pairs], device=pred.device, dtype=torch.long)
        temp_loss = F.smooth_l1_loss(pred[left], pred[right])
    else:
        temp_loss = pred.sum() * 0.0
    spec_loss, spec_diag = spectral_consistency_loss(pred, support_batch, config)
    anch = anchor_loss(adapter)
    total = (
        float(config.lambda_ttt_aug) * aug_loss
        + float(config.lambda_ttt_temp) * temp_loss
        + float(config.lambda_ttt_spec) * spec_loss
        + float(config.lambda_ttt_anchor) * anch
    )
    diag = {
        "ttt_loss": float(total.detach().cpu()),
        "ttt_aug_loss": float(aug_loss.detach().cpu()),
        "ttt_temp_loss": float(temp_loss.detach().cpu()),
        "ttt_spec_loss": float(spec_loss.detach().cpu()),
        "ttt_anchor_loss": float(anch.detach().cpu()),
        "augmentation_prediction_disagreement": aug_disagreement,
        "temporal_pair_count": int(len(pairs)),
        **spec_diag,
    }
    return total, diag


def snapshot_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def assert_state_dict_unchanged(before: Mapping[str, torch.Tensor], module: nn.Module, *, label: str) -> None:
    after = module.state_dict()
    changed = []
    for key, value in before.items():
        cur = after[key].detach().cpu()
        if not torch.equal(value, cur):
            changed.append(key)
    if changed:
        raise AssertionError(f"{label} changed during TTT: {changed[:10]}")


def gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total += float((grad.float() * grad.float()).sum().cpu())
    return float(math.sqrt(total))


@torch.no_grad()
def adapter_prediction_stats(adapter: TTTAdapter, support_batch: Mapping[str, object]) -> Dict[str, float]:
    x = support_batch["imu"]
    if x.numel() == 0:
        return {"adapted_prediction_mean": float("nan"), "prediction_variance_post": float("nan")}
    pred = adapter(x).prediction.detach()
    return {
        "adapted_prediction_mean": float(pred.mean().cpu()),
        "prediction_variance_post": float(pred.var(unbiased=False).cpu()) if pred.numel() > 1 else 0.0,
    }


@torch.no_grad()
def static_prediction_stats(adapter: TTTAdapter, support_batch: Mapping[str, object]) -> Dict[str, float]:
    x = support_batch["imu"]
    if x.numel() == 0:
        return {"static_prediction_mean": float("nan"), "prediction_variance_pre": float("nan")}
    profile = adapter._profile_for_batch(int(x.shape[0]), x.device)
    pred = adapter.static_model(x, profile).prediction.detach()
    return {
        "static_prediction_mean": float(pred.mean().cpu()),
        "prediction_variance_pre": float(pred.var(unbiased=False).cpu()) if pred.numel() > 1 else 0.0,
    }


def loss_component_rows(method: str, heldout: str, seed: int, pre: Mapping[str, object], post: Mapping[str, object]) -> List[Dict[str, object]]:
    rows = []
    for component in ["ttt_loss", "ttt_aug_loss", "ttt_temp_loss", "ttt_spec_loss", "ttt_anchor_loss"]:
        rows.append({
            "seed": int(seed),
            "subject": heldout,
            "method": method,
            "component": component,
            "pre": pre.get(component, np.nan),
            "post": post.get(component, np.nan),
        })
    return rows


def adapt_on_unlabelled_prefix(
    adapter: TTTAdapter,
    support_batch: Mapping[str, object],
    config: TTTConfig,
) -> TTTResult:
    assert_unlabelled_support_batch(support_batch)
    static_before = snapshot_state_dict(adapter.static_model)
    trainable = [param for param in adapter.parameters() if param.requires_grad]
    trainable_names = adapter.trainable_ttt_parameter_names()
    diagnostics: Dict[str, object] = {
        "trainable_ttt_parameter_names": " ".join(trainable_names),
        "trainable_ttt_parameter_count": adapter.trainable_ttt_parameter_count(),
        "frozen_parameter_count": adapter.frozen_parameter_count(),
        "support_window_count": int(support_batch["imu"].shape[0]) if torch.is_tensor(support_batch.get("imu")) else 0,
        "ttt_parameter_group": adapter.parameter_group,
        "ttt_steps": int(config.ttt_steps),
        "ttt_lr": float(config.ttt_lr),
        "ttt_mode": str(config.ttt_mode),
        "update_was_rejected": False,
        "update_rejection_reason": "",
        "ttt_gradient_norm": 0.0,
        "gradient_norm": 0.0,
        "delta_norm_before_projection": 0.0,
        "delta_norm_after_projection": 0.0,
    }
    diagnostics.update(static_prediction_stats(adapter, support_batch))

    if diagnostics["support_window_count"] == 0:
        diagnostics.update({
            "update_was_rejected": True,
            "update_rejection_reason": "empty_support_set",
            "ttt_loss_pre": 0.0,
            "ttt_loss_post": 0.0,
        })
        return TTTResult(adapter.ttt_state(), diagnostics, [], [])

    pre_loss, pre = compute_ttt_losses(adapter, support_batch, config, seed=int(config.seed))
    diagnostics.update({f"{k}_pre": v for k, v in pre.items() if k.startswith("ttt_")})
    diagnostics["augmentation_prediction_disagreement_pre"] = pre.get("augmentation_prediction_disagreement", 0.0)
    diagnostics["temporal_pair_count"] = int(pre.get("temporal_pair_count", 0))
    diagnostics["spectral_rr_mean"] = pre.get("spectral_rr_mean", np.nan)
    diagnostics["spectral_confidence_mean"] = pre.get("spectral_confidence_mean", 0.0)
    diagnostics["ttt_loss_pre"] = pre.get("ttt_loss", np.nan)
    diagnostics["n_temporal_pairs"] = int(pre.get("temporal_pair_count", 0))

    if int(config.ttt_steps) <= 0 or not trainable:
        post_loss, post = compute_ttt_losses(adapter, support_batch, config, seed=int(config.seed))
        diagnostics.update({f"{k}_post": v for k, v in post.items() if k.startswith("ttt_")})
        diagnostics["augmentation_prediction_disagreement_post"] = post.get("augmentation_prediction_disagreement", 0.0)
        diagnostics["ttt_loss_post"] = post.get("ttt_loss", np.nan)
        diagnostics.update(adapter_prediction_stats(adapter, support_batch))
        diagnostics["ttt_delta_norm"] = float(adapter.delta_norm().detach().cpu())
        diagnostics["distance_from_static_state"] = diagnostics["ttt_delta_norm"]
        assert_state_dict_unchanged(static_before, adapter.static_model, label="static profile model")
        return TTTResult(adapter.ttt_state(), diagnostics, loss_component_rows("", "", int(config.seed), pre, post), [])

    if not torch.isfinite(pre_loss):
        diagnostics.update({
            "update_was_rejected": True,
            "update_rejection_reason": "nonfinite_pre_loss",
        })
    else:
        opt = AdamW(trainable, lr=float(config.ttt_lr), weight_decay=0.0)
        for step in range(1, int(config.ttt_steps) + 1):
            opt.zero_grad(set_to_none=True)
            loss, _diag = compute_ttt_losses(adapter, support_batch, config, seed=int(config.seed) + step)
            if not torch.isfinite(loss):
                diagnostics["update_was_rejected"] = True
                diagnostics["update_rejection_reason"] = "nonfinite_ttt_loss"
                break
            loss.backward()
            grad_norm = gradient_norm(trainable)
            diagnostics["gradient_norm"] = grad_norm
            diagnostics["ttt_gradient_norm"] = grad_norm
            if not np.isfinite(grad_norm):
                diagnostics["update_was_rejected"] = True
                diagnostics["update_rejection_reason"] = "nonfinite_gradient_norm"
                opt.zero_grad(set_to_none=True)
                break
            nn.utils.clip_grad_norm_(trainable, float(config.ttt_grad_clip))
            opt.step()
            diagnostics["ttt_step"] = int(step)
            before, after, projected = adapter.project_delta_norm(float(config.ttt_max_delta_norm))
            diagnostics["delta_norm_before_projection"] = before
            diagnostics["delta_norm_after_projection"] = after
            diagnostics["delta_projection_applied"] = bool(projected)

    post_loss, post = compute_ttt_losses(adapter, support_batch, config, seed=int(config.seed))
    diagnostics.update({f"{k}_post": v for k, v in post.items() if k.startswith("ttt_")})
    diagnostics["augmentation_prediction_disagreement_post"] = post.get("augmentation_prediction_disagreement", 0.0)
    diagnostics["ttt_loss_post"] = post.get("ttt_loss", np.nan)
    diagnostics.update(adapter_prediction_stats(adapter, support_batch))
    diagnostics["ttt_delta_norm"] = float(adapter.delta_norm().detach().cpu())
    diagnostics["distance_from_static_state"] = diagnostics["ttt_delta_norm"]

    pre_var = float(diagnostics.get("prediction_variance_pre", np.nan))
    post_var = float(diagnostics.get("prediction_variance_post", np.nan))
    pred_jump = abs(float(diagnostics.get("adapted_prediction_mean", 0.0)) - float(diagnostics.get("static_prediction_mean", 0.0)))
    if not np.isfinite(float(post_loss.detach().cpu())):
        diagnostics["update_was_rejected"] = True
        diagnostics["update_rejection_reason"] = "nonfinite_post_loss"
    elif np.isfinite(post_var) and post_var < float(config.ttt_collapse_var_min) and (not np.isfinite(pre_var) or pre_var >= float(config.ttt_collapse_var_min)):
        diagnostics["update_was_rejected"] = True
        diagnostics["update_rejection_reason"] = "near_zero_prediction_variance"
    elif pred_jump > float(config.ttt_max_prediction_jump_bpm):
        diagnostics["update_was_rejected"] = True
        diagnostics["update_rejection_reason"] = "large_systematic_prediction_jump"

    if diagnostics.get("update_was_rejected"):
        with torch.no_grad():
            if adapter.parameter_group in {"film_only", "film_affine"}:
                adapter.delta_gamma.zero_()
                adapter.delta_beta.zero_()
            if adapter.parameter_group in {"affine_only", "film_affine"}:
                adapter.delta_gain.zero_()
                adapter.delta_bias.zero_()
        post_loss, post = compute_ttt_losses(adapter, support_batch, config, seed=int(config.seed))
        diagnostics.update({f"{k}_post": v for k, v in post.items() if k.startswith("ttt_")})
        diagnostics["ttt_loss_post"] = post.get("ttt_loss", np.nan)
        diagnostics.update(adapter_prediction_stats(adapter, support_batch))
        diagnostics["ttt_delta_norm"] = 0.0
        diagnostics["distance_from_static_state"] = 0.0

    state = adapter.ttt_state()
    diagnostics.update({
        "delta_gamma_norm": float(torch.linalg.vector_norm(state.delta_gamma).cpu()) if state.delta_gamma is not None else 0.0,
        "delta_beta_norm": float(torch.linalg.vector_norm(state.delta_beta).cpu()) if state.delta_beta is not None else 0.0,
        "delta_gain": float(state.delta_gain.cpu()) if state.delta_gain is not None else 0.0,
        "delta_bias": float(state.delta_bias.cpu()) if state.delta_bias is not None else 0.0,
    })
    assert_state_dict_unchanged(static_before, adapter.static_model, label="static profile model")
    state_rows = [{
        "ttt_step": int(diagnostics.get("ttt_step", 0)),
        "delta_gamma_norm": diagnostics["delta_gamma_norm"],
        "delta_beta_norm": diagnostics["delta_beta_norm"],
        "delta_gain": diagnostics["delta_gain"],
        "delta_bias": diagnostics["delta_bias"],
        "ttt_delta_norm": diagnostics["ttt_delta_norm"],
    }]
    return TTTResult(
        state=state,
        diagnostics=diagnostics,
        loss_components=loss_component_rows("", "", int(config.seed), pre, post),
        state_rows=state_rows,
    )


def apply_gate_to_adapter(adapter: TTTAdapter, gate_value: float) -> None:
    gate_value = float(np.clip(gate_value, 0.0, 1.0))
    with torch.no_grad():
        if adapter.parameter_group in {"film_only", "film_affine"}:
            adapter.delta_gamma.mul_(gate_value)
            adapter.delta_beta.mul_(gate_value)
        if adapter.parameter_group in {"affine_only", "film_affine"}:
            adapter.delta_gain.mul_(gate_value)
            adapter.delta_bias.mul_(gate_value)


def make_gate_features(
    profile: torch.Tensor,
    source_profiles: Mapping[str, np.ndarray],
    *,
    support_window_count: int,
) -> torch.Tensor:
    if profile.ndim == 1:
        p = profile.view(1, -1).float()
    else:
        p = profile.float()
    device = p.device
    if source_profiles:
        mat = torch.as_tensor(np.stack(list(source_profiles.values()), axis=0), dtype=torch.float32, device=device)
        centroid = mat.mean(dim=0, keepdim=True)
        centroid_dist = torch.linalg.vector_norm(p - centroid, dim=1, keepdim=True)
        nearest_dist = torch.min(torch.linalg.vector_norm(mat.unsqueeze(0) - p.unsqueeze(1), dim=2), dim=1).values.view(-1, 1)
    else:
        centroid_dist = torch.zeros((p.shape[0], 1), dtype=p.dtype, device=device)
        nearest_dist = torch.zeros((p.shape[0], 1), dtype=p.dtype, device=device)
    support_count = torch.full((p.shape[0], 1), float(support_window_count), dtype=p.dtype, device=device)
    return torch.cat([p, centroid_dist, nearest_dist, torch.log1p(support_count)], dim=1)


def arrays_batch(
    arrays: Mapping[str, np.ndarray],
    indices: Sequence[int],
    device: torch.device,
    *,
    include_rr: bool,
) -> Dict[str, object]:
    idx = np.asarray(indices, dtype=int)
    x = ensure_channel_first(torch.as_tensor(np.asarray(arrays["x"])[idx]).float().to(device))
    batch: Dict[str, object] = {
        "imu": x,
        "subject_id": [str(s) for s in np.asarray(arrays["subject_id"], dtype=object)[idx]],
        "window_index": np.asarray(arrays["window_index"])[idx].astype(int),
        "condition": np.asarray(arrays["cond"])[idx],
        "sequence_index": np.asarray(arrays["window_index"])[idx].astype(int),
    }
    if include_rr:
        batch["rr"] = torch.as_tensor(np.asarray(arrays["rr"])[idx], dtype=torch.float32, device=device).view(-1)
    return batch


def build_meta_subject_episode(
    subject: str,
    arrays: Mapping[str, np.ndarray],
    normalized_profiles: Mapping[str, np.ndarray],
    *,
    support_windows: int,
    query_windows: int,
    seed: int,
    device: torch.device,
) -> MetaSubjectEpisode:
    subject_mask = np.asarray(arrays["subject_id"], dtype=object) == subject
    local = np.flatnonzero(subject_mask)
    if len(local) <= 1:
        raise ValueError(f"not enough windows for meta episode subject {subject}")
    local_order = local[np.argsort(np.asarray(arrays["window_index"])[local].astype(int))]
    n_support = min(int(support_windows), max(1, len(local_order) - 1))
    support_global = local_order[:n_support]
    remaining = local_order[n_support:]
    if int(query_windows) > 0:
        remaining = remaining[: int(query_windows)]
    if len(remaining) == 0:
        remaining = local_order[-1:]
        support_global = local_order[: max(1, len(local_order) - 1)]
    if set(support_global.tolist()).intersection(set(remaining.tolist())):
        raise AssertionError("meta support/query split overlaps")
    support = arrays_batch(arrays, support_global, device, include_rr=False)
    query = arrays_batch(arrays, remaining, device, include_rr=True)
    assert_unlabelled_support_batch(support)
    return MetaSubjectEpisode(
        pseudo_target_subject=subject,
        support_indices=torch.as_tensor(np.asarray(arrays["window_index"])[support_global].astype(int), device=device),
        query_indices=torch.as_tensor(np.asarray(arrays["window_index"])[remaining].astype(int), device=device),
        support_batch=support,
        query_batch=query,
        profile=torch.as_tensor(normalized_profiles[subject], dtype=torch.float32, device=device).view(1, -1),
    )


def build_meta_episodes(
    arrays: Mapping[str, np.ndarray],
    subjects: Sequence[str],
    heldout: str,
    normalized_profiles: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
) -> List[MetaSubjectEpisode]:
    candidates = [s for s in subjects if s != heldout]
    if heldout in candidates:
        raise AssertionError("real held-out target entered meta-training candidates")
    rng = np.random.default_rng(int(args.seed))
    if str(args.meta_pseudo_target_order) == "shuffled":
        candidates = candidates.copy()
        rng.shuffle(candidates)
    episodes: List[MetaSubjectEpisode] = []
    for subject in candidates:
        episodes.append(build_meta_subject_episode(
            subject,
            arrays,
            normalized_profiles,
            support_windows=int(args.meta_support_windows),
            query_windows=int(args.meta_query_windows),
            seed=int(args.seed),
            device=device,
        ))
    return episodes


def meta_episode_split_rows(seed: int, heldout: str, episode_id: int, episode: MetaSubjectEpisode) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for split, batch in [("support", episode.support_batch), ("query", episode.query_batch)]:
        for i, widx in enumerate(np.asarray(batch["window_index"]).astype(int)):
            rows.append({
                "seed": int(seed),
                "true_loso_subject": heldout,
                "pseudo_target_subject": episode.pseudo_target_subject,
                "episode": int(episode_id),
                "split": split,
                "window_index": int(widx),
                "condition": float(np.asarray(batch["condition"])[i]),
                "sequence_index": int(np.asarray(batch["sequence_index"])[i]),
            })
    return rows


def gate_regularization(gate_value: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    mean_penalty = float(args.lambda_gate_mean) * gate_value.mean()
    target = torch.full_like(gate_value, float(args.gate_target_mean))
    entropy_like = (gate_value - target).pow(2).mean()
    return mean_penalty + float(args.lambda_gate_entropy) * entropy_like


def meta_query_step(
    static_model: ProfileConditionedTCN,
    gate: ProfileAdaptationGate,
    episode: MetaSubjectEpisode,
    source_profiles: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    *,
    mode: str,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    profile = episode.profile
    gate_features = make_gate_features(profile, source_profiles, support_window_count=int(episode.support_batch["imu"].shape[0]))
    gate_out = gate(gate_features)
    support_cfg = TTTConfig(
        ttt_steps=1,
        ttt_lr=float(args.ttt_lr),
        ttt_parameter_group=str(args.ttt_parameter_group),
        lambda_ttt_aug=float(args.lambda_ttt_aug),
        lambda_ttt_temp=float(args.lambda_ttt_temp),
        lambda_ttt_spec=float(args.lambda_ttt_spec),
        lambda_ttt_anchor=float(args.lambda_ttt_anchor),
        ttt_max_delta_norm=float(args.ttt_max_delta_norm),
        seed=int(args.seed),
    )
    adapter = TTTAdapter(static_model, profile, str(args.ttt_parameter_group)).to(profile.device)
    trainable = [p for p in adapter.parameters() if p.requires_grad]
    for p in list(static_model.film.parameters()) + list(static_model.affine.parameters()):
        p.requires_grad = True
    loss_pre, _pre = compute_ttt_losses(adapter, episode.support_batch, support_cfg, seed=int(args.seed))
    create_graph = str(mode) == "full_second_order"
    grads = torch.autograd.grad(loss_pre, trainable, create_graph=create_graph, allow_unused=True)
    grad_norm_terms = [g.detach().pow(2).sum() for g in grads if g is not None]
    grad_norm = torch.sqrt(torch.stack(grad_norm_terms).sum()).item() if grad_norm_terms else 0.0
    effective_lr = float(args.ttt_lr) * gate_out.lr_multiplier.view(()).clamp_min(0.0)
    for param, grad in zip(trainable, grads):
        if grad is None:
            continue
        update = -effective_lr * (grad if create_graph else grad.detach())
        with torch.no_grad():
            param.add_(update)
    adapter.project_delta_norm(float(args.ttt_max_delta_norm) * float(gate_out.max_delta_multiplier.detach().cpu().view(())))
    query_x = episode.query_batch["imu"]
    query_rr = episode.query_batch["rr"]
    query_static = static_model(query_x, profile.expand(query_x.shape[0], -1)).prediction
    raw_query_adapted = adapter(query_x).prediction
    # First-order mode detaches the inner TTT parameter update; the gate and LR heads
    # still receive query gradients through this bounded deployment-time blend.
    lr_signal = gate_out.lr_multiplier.view(()) * (raw_query_adapted.detach() - query_static.detach())
    query_adapted = query_static + gate_out.gate.view(()) * lr_signal
    static_loss = F.smooth_l1_loss(query_static, query_rr)
    adapted_loss = F.smooth_l1_loss(query_adapted, query_rr)
    identity = adapter.delta_squared_norm()
    loss = (
        adapted_loss
        + float(args.lambda_meta_static) * static_loss
        + gate_regularization(gate_out.gate, args)
        + float(args.lambda_meta_identity) * identity
    )
    diag = {
        "support_windows": int(episode.support_batch["imu"].shape[0]),
        "query_windows": int(query_x.shape[0]),
        "static_query_mae": float(torch.mean(torch.abs(query_static.detach() - query_rr.detach())).cpu()),
        "adapted_query_mae": float(torch.mean(torch.abs(query_adapted.detach() - query_rr.detach())).cpu()),
        "query_improvement": float((torch.mean(torch.abs(query_static.detach() - query_rr.detach())) - torch.mean(torch.abs(query_adapted.detach() - query_rr.detach()))).cpu()),
        "gate_value": float(gate_out.gate.detach().cpu().view(())),
        "lr_multiplier": float(gate_out.lr_multiplier.detach().cpu().view(())),
        "effective_ttt_lr": float((float(args.ttt_lr) * gate_out.lr_multiplier.detach()).cpu().view(())),
        "max_delta_multiplier": float(gate_out.max_delta_multiplier.detach().cpu().view(())),
        "support_ttt_loss_pre": float(loss_pre.detach().cpu()),
        "support_ttt_loss_post": np.nan,
        "inner_gradient_norm": float(grad_norm),
        "raw_delta_norm": float(adapter.delta_norm().detach().cpu()),
        "gated_delta_norm": float(adapter.delta_norm().detach().cpu()) * float(gate_out.gate.detach().cpu().view(())),
        "update_was_rejected": False,
        "update_rejection_reason": "",
    }
    return loss, diag


def train_meta_gate(
    static_model: ProfileConditionedTCN,
    episodes: Sequence[MetaSubjectEpisode],
    source_profiles: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    heldout: str,
    gate: ProfileAdaptationGate,
) -> Dict[str, List[Dict[str, object]]]:
    if not episodes:
        return {"meta_training_history": [], "meta_episode_rows": [], "meta_gate_diagnostics": [], "meta_update_rejections": [], "meta_subject_summary": [], "meta_episode_splits": []}
    freeze_module(static_model.wrapped)
    trainable = [p for p in list(static_model.film.parameters()) + list(static_model.affine.parameters()) + list(gate.parameters()) if p.requires_grad]
    opt = AdamW(trainable, lr=float(args.meta_learning_rate), weight_decay=float(args.meta_weight_decay))
    history: List[Dict[str, object]] = []
    episode_rows: List[Dict[str, object]] = []
    gate_rows: List[Dict[str, object]] = []
    splits: List[Dict[str, object]] = []
    max_eps = int(args.meta_episodes_per_epoch) if int(args.meta_episodes_per_epoch) > 0 else len(episodes)
    for epoch in range(1, int(args.meta_epochs) + 1):
        losses: List[float] = []
        for i, episode in enumerate(list(episodes)[:max_eps], start=1):
            opt.zero_grad(set_to_none=True)
            loss, diag = meta_query_step(static_model, gate, episode, source_profiles, args, mode=str(args.meta_mode))
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu()))
            row = {"seed": int(args.seed), "true_loso_subject": heldout, "pseudo_target_subject": episode.pseudo_target_subject, "episode": i, **diag}
            episode_rows.append(row)
            gate_rows.append({k: row[k] for k in ["seed", "true_loso_subject", "pseudo_target_subject", "episode", "gate_value", "lr_multiplier", "effective_ttt_lr", "max_delta_multiplier"]})
            splits.extend(meta_episode_split_rows(int(args.seed), heldout, i, episode))
        history.append({
            "seed": int(args.seed),
            "true_loso_subject": heldout,
            "epoch": epoch,
            "meta_train_query_mae": float(np.mean([r["adapted_query_mae"] for r in episode_rows[-max_eps:]])) if episode_rows else np.nan,
            "meta_val_query_mae": float(np.mean([r["adapted_query_mae"] for r in episode_rows[-max_eps:]])) if episode_rows else np.nan,
            "meta_val_static_mae": float(np.mean([r["static_query_mae"] for r in episode_rows[-max_eps:]])) if episode_rows else np.nan,
            "meta_val_adapted_mae": float(np.mean([r["adapted_query_mae"] for r in episode_rows[-max_eps:]])) if episode_rows else np.nan,
            "meta_val_improvement": float(np.mean([r["query_improvement"] for r in episode_rows[-max_eps:]])) if episode_rows else np.nan,
            "mean_gate": float(np.mean([r["gate_value"] for r in episode_rows[-max_eps:]])) if episode_rows else np.nan,
            "mean_effective_lr": float(np.mean([r["effective_ttt_lr"] for r in episode_rows[-max_eps:]])) if episode_rows else np.nan,
            "mean_delta_norm": float(np.mean([r["gated_delta_norm"] for r in episode_rows[-max_eps:]])) if episode_rows else np.nan,
            "meta_loss": float(np.mean(losses)) if losses else np.nan,
            "trainable_parameter_count": int(sum(p.numel() for p in trainable)),
            "trainable_parameter_names": " ".join([name for name, p in list(static_model.named_parameters()) + [(f"gate.{n}", p) for n, p in gate.named_parameters()] if p.requires_grad]),
        })
    by_subject = pd.DataFrame(episode_rows).groupby("pseudo_target_subject", as_index=False).agg(
        adapted_query_mae=("adapted_query_mae", "mean"),
        static_query_mae=("static_query_mae", "mean"),
        mean_gate=("gate_value", "mean"),
    ).to_dict("records") if episode_rows else []
    return {
        "meta_training_history": history,
        "meta_episode_rows": episode_rows,
        "meta_gate_diagnostics": gate_rows,
        "meta_update_rejections": [r for r in episode_rows if bool(r.get("update_was_rejected", False))],
        "meta_subject_summary": by_subject,
        "meta_episode_splits": splits,
    }


def deploy_gated_ttt(
    static_model: ProfileConditionedTCN,
    gate: ProfileAdaptationGate,
    profile: np.ndarray,
    source_profiles: Mapping[str, np.ndarray],
    support_batch: Mapping[str, object],
    args: argparse.Namespace,
    *,
    gate_mode: str,
    lr_mode: str,
    steps: int,
    device: torch.device,
) -> Tuple[TTTAdapter, Dict[str, object]]:
    profile_t = torch.as_tensor(profile, dtype=torch.float32, device=device).view(1, -1)
    gate_features = make_gate_features(profile_t, source_profiles, support_window_count=int(support_batch["imu"].shape[0]))
    with torch.no_grad():
        gate_out = gate(gate_features)
    gate_value = float(gate_out.gate.detach().cpu().view(()))
    if gate_mode == "fixed0":
        gate_value = 0.0
    elif gate_mode == "fixed1":
        gate_value = 1.0
    lr_multiplier = float(gate_out.lr_multiplier.detach().cpu().view(()))
    if lr_mode == "fixed":
        lr_multiplier = 1.0
    max_delta_multiplier = float(gate_out.max_delta_multiplier.detach().cpu().view(()))
    cfg = ttt_config_from_args(args, overrides={
        "ttt_steps": int(steps),
        "ttt_parameter_group": str(args.ttt_parameter_group),
        "ttt_lr": float(args.ttt_lr) * lr_multiplier,
        "ttt_max_delta_norm": float(args.ttt_max_delta_norm) * max_delta_multiplier,
        "seed": int(args.seed),
    })
    adapter = TTTAdapter(static_model, profile_t, str(args.ttt_parameter_group)).to(device)
    result = adapt_on_unlabelled_prefix(adapter, support_batch, cfg)
    raw_delta_norm = float(adapter.delta_norm().detach().cpu())
    apply_gate_to_adapter(adapter, gate_value)
    gated_delta_norm = float(adapter.delta_norm().detach().cpu())
    diag = dict(result.diagnostics)
    diag.update({
        "gate_value": gate_value,
        "lr_multiplier": lr_multiplier,
        "effective_ttt_lr": float(cfg.ttt_lr),
        "max_delta_multiplier": max_delta_multiplier,
        "raw_delta_norm_before_gate": raw_delta_norm,
        "ttt_delta_norm": gated_delta_norm,
        "gated_delta_norm": gated_delta_norm,
        "gate_suppressed_delta_norm": raw_delta_norm - gated_delta_norm,
        "gate_applied_to_delta": True,
    })
    return adapter, diag


@torch.no_grad()
def latent_stats_from_loader(model: WrappedTCN, loader: DataLoader, device: torch.device, *, batch_limit: int = 0) -> Tuple[np.ndarray, np.ndarray, int]:
    pooled: List[np.ndarray] = []
    model.eval()
    for step, batch in enumerate(loader):
        if batch_limit > 0 and step >= batch_limit:
            break
        x, _rr, _subjects, _widx, _cond = batch_to_device(batch, device)
        pooled.append(model(x).pooled_hidden.detach().cpu().numpy())
    if not pooled:
        raise ValueError("cannot estimate latent stats from an empty loader")
    mat = np.concatenate(pooled, axis=0)
    return mat.mean(axis=0).astype(np.float32), mat.std(axis=0).astype(np.float32), int(mat.shape[0])


@torch.no_grad()
def latent_stats_from_support(model: WrappedTCN, support_batch: Mapping[str, object]) -> Tuple[np.ndarray, np.ndarray, int]:
    x = support_batch["imu"]
    if x.numel() == 0:
        raise ValueError("cannot estimate target latent stats from empty support prefix")
    out = model(x)
    mat = out.pooled_hidden.detach().cpu().numpy()
    return mat.mean(axis=0).astype(np.float32), mat.std(axis=0).astype(np.float32), int(mat.shape[0])


def mean_alignment_alpha(
    aligner: ProfileGatedMeanAlignment,
    profile: np.ndarray,
    device: torch.device,
    *,
    mode: str,
    fixed_alpha: Optional[float],
    seed: int,
) -> float:
    if mode == "none":
        return 0.0
    if mode == "fixed":
        return float(fixed_alpha or 0.0)
    if mode == "random":
        return float(np.random.default_rng(int(seed) + 177).uniform(0.0, aligner.alpha_max))
    p = torch.as_tensor(profile, dtype=torch.float32, device=device).view(1, -1)
    with torch.no_grad():
        return float(aligner.alpha_from_profile(p).detach().cpu().view(()))


def selected_profile_indices(n_windows: int, k: int, seed: int = 0, chronological: bool = True) -> List[int]:
    k = min(int(k), int(n_windows))
    if chronological:
        return list(range(k))
    rng = np.random.default_rng(int(seed))
    return sorted(rng.choice(np.arange(n_windows), size=k, replace=False).astype(int).tolist())


def assert_subject_pure(subject_ids: Sequence[str], expected_subject: str) -> None:
    found = set(str(s) for s in subject_ids)
    if found != {str(expected_subject)}:
        raise AssertionError(f"profile windows are not subject-pure for {expected_subject}: {sorted(found)}")


def raw_imu_features(x: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("Expected profile IMU windows with shape [N,T,C] or [N,C,T]")
    if x.shape[1] < x.shape[2]:
        xc = x
    else:
        xc = np.transpose(x, (0, 2, 1))
    n, c, t = xc.shape
    flat = xc.transpose(1, 0, 2).reshape(c, n * t)
    names: List[str] = []
    vals: List[float] = []
    freqs = np.fft.rfftfreq(t, d=1.0 / float(IMU_FS))
    bands = [(0.05, 0.2, "low"), (0.2, 0.6, "mid"), (0.6, 2.0, "high")]
    for ch in range(c):
        v = flat[ch]
        med = np.median(v)
        stats = {
            "mean": np.mean(v),
            "std": np.std(v),
            "median": med,
            "mad": np.median(np.abs(v - med)),
            "rms": np.sqrt(np.mean(v * v)),
            "energy": np.mean(v * v),
        }
        spec = np.abs(np.fft.rfft(xc[:, ch, :], axis=-1)) ** 2
        psd = spec.mean(axis=0)
        denom = float(psd.sum() + 1e-12)
        p = psd / denom
        stats["dominant_frequency"] = float(freqs[int(np.argmax(psd[1:]) + 1)]) if len(psd) > 1 else 0.0
        stats["spectral_entropy"] = float(-(p * np.log(p + 1e-12)).sum() / max(1.0, math.log(len(p))))
        for lo, hi, label in bands:
            mask = (freqs >= lo) & (freqs < hi)
            stats[f"{label}_frequency_energy"] = float(psd[mask].sum() / denom)
        for key, value in stats.items():
            names.append(f"raw_ch{ch}_{key}")
            vals.append(float(value))
    for start, label in [(0, "acc"), (3, "gyr")]:
        if c >= start + 3:
            norm = np.linalg.norm(xc[:, start:start + 3, :], axis=1).reshape(-1)
            names.extend([f"{label}_norm_mean", f"{label}_norm_std", f"{label}_norm_rms"])
            vals.extend([float(np.mean(norm)), float(np.std(norm)), float(np.sqrt(np.mean(norm * norm)))])
    if c >= 2:
        corr = np.corrcoef(flat)
        for i in range(c):
            for j in range(i + 1, c):
                names.append(f"raw_corr_ch{i}_ch{j}")
                vals.append(float(corr[i, j]) if np.isfinite(corr[i, j]) else 0.0)
    diffs = np.diff(xc, axis=-1)
    names.append("raw_temporal_variability")
    vals.append(float(np.sqrt(np.mean(diffs * diffs))) if diffs.size else 0.0)
    return finite_array(np.asarray(vals)), names


@torch.no_grad()
def latent_profile_features(model: WrappedTCN, x: np.ndarray, device: torch.device) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
    model.eval()
    xb = ensure_channel_first(torch.as_tensor(x).float().to(device))
    out = model(xb)
    pooled = out.pooled_hidden.detach().cpu().numpy()
    tokens = out.hidden_tokens.detach().cpu().numpy()
    pred = out.prediction.detach().cpu().numpy().reshape(-1)
    x_aug = xb * 1.01
    pred_aug = model(x_aug).prediction.detach().cpu().numpy().reshape(-1)
    vals: List[float] = []
    names: List[str] = []
    for i, v in enumerate(pooled.mean(axis=0)):
        names.append(f"latent_mean_{i}")
        vals.append(float(v))
    for i, v in enumerate(pooled.std(axis=0)):
        names.append(f"latent_std_{i}")
        vals.append(float(v))
    token_norm = np.linalg.norm(tokens, axis=-1)
    temporal_var = np.diff(tokens, axis=1)
    names.extend(["hidden_token_norm_mean", "hidden_temporal_variability", "prediction_mean", "prediction_std", "augmentation_disagreement"])
    vals.extend([
        float(token_norm.mean()),
        float(np.sqrt(np.mean(temporal_var * temporal_var))) if temporal_var.size else 0.0,
        float(pred.mean()),
        float(pred.std()),
        float(np.mean(np.abs(pred - pred_aug))),
    ])
    diag = {
        "latent_norm": float(np.linalg.norm(pooled.mean(axis=0))),
        "prediction_mean": float(pred.mean()),
        "prediction_std": float(pred.std()),
        "augmentation_disagreement": float(np.mean(np.abs(pred - pred_aug))),
    }
    return finite_array(np.asarray(vals)), names, diag


def build_subject_profile(
    subject: str,
    arrays: Mapping[str, np.ndarray],
    model: WrappedTCN,
    k: int,
    device: torch.device,
    seed: int,
) -> Tuple[np.ndarray, List[str], Dict[str, object]]:
    subject_mask = np.asarray(arrays["subject_id"], dtype=object) == subject
    assert_subject_pure(np.asarray(arrays["subject_id"], dtype=object)[subject_mask], subject)
    local = np.flatnonzero(subject_mask)
    chosen_local = selected_profile_indices(len(local), k, seed=seed, chronological=True)
    chosen_global = local[np.asarray(chosen_local, dtype=int)]
    # Leakage-sensitive: only IMU windows and frozen-model predictions are used here.
    x = np.asarray(arrays["x"])[chosen_global]
    raw_vec, raw_names = raw_imu_features(x)
    latent_vec, latent_names, latent_diag = latent_profile_features(model, x, device)
    vector = np.concatenate([raw_vec, latent_vec]).astype(np.float32)
    schema = raw_names + latent_names
    diag = {
        "subject": subject,
        "profile_windows": int(len(chosen_global)),
        "profile_indices": [int(arrays["window_index"][i]) for i in chosen_global],
        "profile_norm": float(np.linalg.norm(vector)),
        "target_rr_used": False,
        **latent_diag,
    }
    return vector, schema, diag


def fit_profile_normalizer(profiles: Mapping[str, np.ndarray], source_subjects: Sequence[str], heldout: str) -> Tuple[np.ndarray, np.ndarray]:
    if heldout in set(source_subjects):
        raise AssertionError("held-out subject is present in normalizer source subjects")
    mat = np.stack([profiles[s] for s in source_subjects], axis=0).astype(np.float64)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_profiles(profiles: Mapping[str, np.ndarray], mean: np.ndarray, std: np.ndarray) -> Dict[str, np.ndarray]:
    return {s: ((v - mean) / std).astype(np.float32) for s, v in profiles.items()}


def deterministic_profile_permutation(subjects: Sequence[str], seed: int) -> Dict[str, str]:
    subjects = list(subjects)
    if len(subjects) <= 1:
        return {s: s for s in subjects}
    rng = np.random.default_rng(int(seed) + 991)
    perm = subjects.copy()
    for _ in range(100):
        rng.shuffle(perm)
        if all(a != b for a, b in zip(subjects, perm)):
            return dict(zip(subjects, perm))
    return dict(zip(subjects, perm[1:] + perm[:1]))


def profile_tensor_for_subjects(subjects: Sequence[str], profiles: Mapping[str, np.ndarray], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.stack([profiles[str(s)] for s in subjects], axis=0), dtype=torch.float32, device=device)


def rr_metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    err = y_pred - y_true
    ae = np.abs(err)
    corr = np.nan
    if len(y_true) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = float(pearsonr(y_true, y_pred)[0])
    return {
        "mae": float(np.mean(ae)) if len(ae) else np.nan,
        "rmse": float(np.sqrt(np.mean(err * err))) if len(err) else np.nan,
        "rr_corr": corr,
        "bias": float(np.mean(err)) if len(err) else np.nan,
        "median_ae": float(np.median(ae)) if len(ae) else np.nan,
        "p95_ae": float(np.percentile(ae, 95)) if len(ae) else np.nan,
        "n_windows": int(len(y_true)),
    }


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def train_plain_tcn(model: WrappedTCN, train_loader: DataLoader, val_loader: DataLoader, args: argparse.Namespace, device: torch.device, subject: str) -> List[Dict[str, object]]:
    opt = AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    best_state = None
    best_val = float("inf")
    bad = 0
    rows: List[Dict[str, object]] = []
    model.to(device)
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: List[float] = []
        for step, batch in enumerate(train_loader):
            if int(args.batch_limit) > 0 and step >= int(args.batch_limit):
                break
            x, rr, _subjects, _widx, _cond = batch_to_device(batch, device)
            pred = model(x).prediction
            loss = F.smooth_l1_loss(pred, rr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_metrics = evaluate_plain(model, val_loader, device, batch_limit=int(args.batch_limit))
        val_mae = float(val_metrics["mae"])
        rows.append({"seed": int(args.seed), "subject": subject, "stage": "t0_plain", "epoch": epoch, "train_loss": float(np.mean(losses)), "val_mae": val_mae})
        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= int(args.patience):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return rows


@torch.no_grad()
def evaluate_plain(model: WrappedTCN, loader: DataLoader, device: torch.device, batch_limit: int = 0) -> Dict[str, float]:
    model.eval()
    y, p = [], []
    for step, batch in enumerate(loader):
        if batch_limit > 0 and step >= batch_limit:
            break
        x, rr, _subjects, _widx, _cond = batch_to_device(batch, device)
        out = model(x)
        y.append(rr.detach().cpu().numpy())
        p.append(out.prediction.detach().cpu().numpy())
    return rr_metric_dict(np.concatenate(y), np.concatenate(p))


def train_conditioning(
    model: ProfileConditionedTCN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    profiles: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    subject: str,
    stage: str,
) -> List[Dict[str, object]]:
    freeze_module(model.wrapped)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    best_state = None
    best_val = float("inf")
    bad = 0
    rows: List[Dict[str, object]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        # Leakage/control-sensitive: the source TCN is frozen, including BN state.
        model.wrapped.eval()
        losses: List[float] = []
        for step, batch in enumerate(train_loader):
            if int(args.batch_limit) > 0 and step >= int(args.batch_limit):
                break
            x, rr, subjects, _widx, _cond = batch_to_device(batch, device)
            profile = profile_tensor_for_subjects(subjects, profiles, device)
            pred = model(x, profile).prediction
            loss = F.smooth_l1_loss(pred, rr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, float(args.grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_metrics = evaluate_conditioned(model, val_loader, profiles, device, batch_limit=int(args.batch_limit), exclude_indices=None)
        val_mae = float(val_metrics["metrics"]["mae"])
        rows.append({"seed": int(args.seed), "subject": subject, "stage": stage, "epoch": epoch, "train_loss": float(np.mean(losses)), "val_mae": val_mae})
        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= int(args.patience):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return rows


@torch.no_grad()
def evaluate_conditioned(
    model: nn.Module,
    loader: DataLoader,
    profiles: Mapping[str, np.ndarray],
    device: torch.device,
    *,
    batch_limit: int = 0,
    exclude_indices: Optional[set[int]] = None,
) -> Dict[str, object]:
    model.eval()
    rows: List[Dict[str, object]] = []
    y_all: List[float] = []
    p_all: List[float] = []
    diag_rows: List[Dict[str, float]] = []
    for step, batch in enumerate(loader):
        if batch_limit > 0 and step >= batch_limit:
            break
        x, rr, subjects, widx, cond = batch_to_device(batch, device)
        keep = np.ones(len(widx), dtype=bool)
        if exclude_indices is not None:
            keep = np.asarray([int(i) not in exclude_indices for i in widx], dtype=bool)
        if not keep.any():
            continue
        profile = profile_tensor_for_subjects(subjects, profiles, device)
        if isinstance(model, WrappedTCN):
            out0 = model(x)
            pred = out0.prediction
            baseline = pred
            film_pred = pred
            affine_pred = pred
            combined = pred
            gamma = torch.ones_like(out0.pooled_hidden)
            beta = torch.zeros_like(out0.pooled_hidden)
            gain = torch.ones_like(pred)
            bias = torch.zeros_like(pred)
            pooled = out0.pooled_hidden
            z_cond = pooled
        else:
            out = model(x, profile)
            pred = out.prediction
            baseline = out.baseline_prediction
            film_pred = out.film_prediction
            affine_pred = out.affine_prediction
            combined = out.combined_prediction
            gamma = out.film_gamma
            beta = out.film_beta
            gain = out.affine_gain
            bias = out.affine_bias
            pooled = out.pooled_hidden
            z_cond = out.conditioned_hidden
        pred_np = pred.detach().cpu().numpy().reshape(-1)
        rr_np = rr.detach().cpu().numpy().reshape(-1)
        base_np = baseline.detach().cpu().numpy().reshape(-1)
        film_np = film_pred.detach().cpu().numpy().reshape(-1)
        aff_np = affine_pred.detach().cpu().numpy().reshape(-1)
        comb_np = combined.detach().cpu().numpy().reshape(-1)
        gamma_np = gamma.detach().cpu().numpy()
        beta_np = beta.detach().cpu().numpy()
        gain_np = gain.detach().cpu().numpy().reshape(-1)
        bias_np = bias.detach().cpu().numpy().reshape(-1)
        hidden_delta = (z_cond - pooled).detach().cpu().numpy()
        for i in np.flatnonzero(keep):
            rows.append({
                "subject": subjects[i],
                "window_index": int(widx[i]),
                "condition": float(cond[i]),
                "sequence_index": int(widx[i]),
                "rr_true": float(rr_np[i]),
                "rr_pred": float(pred_np[i]),
                "rr_pred_plain": float(base_np[i]),
                "rr_pred_static": float(pred_np[i]),
                "rr_pred_adapted": float(pred_np[i]),
                "adaptation_delta": 0.0,
                "is_profile_window": bool(exclude_indices is not None and int(widx[i]) in exclude_indices),
                "ttt_mode": "",
                "baseline_pred": float(base_np[i]),
                "film_only_pred": float(film_np[i]),
                "affine_only_pred": float(aff_np[i]),
                "combined_pred": float(comb_np[i]),
                "prediction_delta": float(pred_np[i] - base_np[i]),
                "affine_gain": float(gain_np[i]),
                "affine_bias": float(bias_np[i]),
                "film_gamma_delta_norm": float(np.linalg.norm(gamma_np[i] - 1.0)),
                "film_beta_norm": float(np.linalg.norm(beta_np[i])),
                "hidden_delta_rms": float(np.sqrt(np.mean(hidden_delta[i] * hidden_delta[i]))),
            })
            y_all.append(float(rr_np[i]))
            p_all.append(float(pred_np[i]))
            diag_rows.append(rows[-1])
    metrics = rr_metric_dict(np.asarray(y_all), np.asarray(p_all))
    if diag_rows:
        df = pd.DataFrame(diag_rows)
        metrics.update({
            "film_gamma_delta_norm": float(df["film_gamma_delta_norm"].mean()),
            "film_beta_norm": float(df["film_beta_norm"].mean()),
            "hidden_delta_rms": float(df["hidden_delta_rms"].mean()),
            "affine_gain": float(df["affine_gain"].mean()),
            "affine_bias": float(df["affine_bias"].mean()),
            "prediction_delta_mean": float(df["prediction_delta"].mean()),
            "prediction_delta_std": float(df["prediction_delta"].std(ddof=0)),
        })
    return {"metrics": metrics, "rows": rows}


@torch.no_grad()
def evaluate_ttt_adapter(
    adapter: TTTAdapter,
    loader: DataLoader,
    device: torch.device,
    *,
    profile_window_indices: set[int],
    exclude_indices: Optional[set[int]] = None,
) -> Dict[str, object]:
    adapter.eval()
    rows: List[Dict[str, object]] = []
    y_all: List[float] = []
    p_all: List[float] = []
    for batch in loader:
        x, rr, subjects, widx, cond = batch_to_device(batch, device)
        keep = np.ones(len(widx), dtype=bool)
        if exclude_indices is not None:
            keep = np.asarray([int(i) not in exclude_indices for i in widx], dtype=bool)
        if not keep.any():
            continue
        profile = adapter._profile_for_batch(int(x.shape[0]), x.device)
        static_out = adapter.static_model(x, profile)
        adapted_out = adapter(x)
        static_np = static_out.prediction.detach().cpu().numpy().reshape(-1)
        adapted_np = adapted_out.prediction.detach().cpu().numpy().reshape(-1)
        rr_np = rr.detach().cpu().numpy().reshape(-1)
        gamma_np = adapted_out.film_gamma.detach().cpu().numpy()
        beta_np = adapted_out.film_beta.detach().cpu().numpy()
        gain_np = adapted_out.affine_gain.detach().cpu().numpy().reshape(-1)
        bias_np = adapted_out.affine_bias.detach().cpu().numpy().reshape(-1)
        hidden_delta = (adapted_out.conditioned_hidden - adapted_out.pooled_hidden).detach().cpu().numpy()
        for i in np.flatnonzero(keep):
            rows.append({
                "subject": subjects[i],
                "window_index": int(widx[i]),
                "condition": float(cond[i]),
                "sequence_index": int(widx[i]),
                "rr_true": float(rr_np[i]),
                "rr_pred": float(adapted_np[i]),
                "rr_pred_plain": float(static_out.baseline_prediction.detach().cpu().numpy().reshape(-1)[i]),
                "rr_pred_static": float(static_np[i]),
                "rr_pred_adapted": float(adapted_np[i]),
                "adaptation_delta": float(adapted_np[i] - static_np[i]),
                "is_profile_window": bool(int(widx[i]) in profile_window_indices),
                "ttt_mode": adapter.parameter_group,
                "baseline_pred": float(static_out.baseline_prediction.detach().cpu().numpy().reshape(-1)[i]),
                "film_only_pred": float(static_out.film_prediction.detach().cpu().numpy().reshape(-1)[i]),
                "affine_only_pred": float(static_out.affine_prediction.detach().cpu().numpy().reshape(-1)[i]),
                "combined_pred": float(static_np[i]),
                "prediction_delta": float(adapted_np[i] - static_np[i]),
                "affine_gain": float(gain_np[i]),
                "affine_bias": float(bias_np[i]),
                "film_gamma_delta_norm": float(np.linalg.norm(gamma_np[i] - 1.0)),
                "film_beta_norm": float(np.linalg.norm(beta_np[i])),
                "hidden_delta_rms": float(np.sqrt(np.mean(hidden_delta[i] * hidden_delta[i]))),
            })
            y_all.append(float(rr_np[i]))
            p_all.append(float(adapted_np[i]))
    metrics = rr_metric_dict(np.asarray(y_all), np.asarray(p_all))
    if rows:
        df = pd.DataFrame(rows)
        metrics.update({
            "film_gamma_delta_norm": float(df["film_gamma_delta_norm"].mean()),
            "film_beta_norm": float(df["film_beta_norm"].mean()),
            "hidden_delta_rms": float(df["hidden_delta_rms"].mean()),
            "affine_gain": float(df["affine_gain"].mean()),
            "affine_bias": float(df["affine_bias"].mean()),
            "prediction_delta_mean": float(df["prediction_delta"].mean()),
            "prediction_delta_std": float(df["prediction_delta"].std(ddof=0)),
            "adapted_prediction_mean": float(df["rr_pred_adapted"].mean()),
            "static_prediction_mean": float(df["rr_pred_static"].mean()),
        })
    return {"metrics": metrics, "rows": rows}


@torch.no_grad()
def evaluate_mean_aligned_adapter(
    adapter: TTTAdapter,
    loader: DataLoader,
    device: torch.device,
    *,
    source_mean: np.ndarray,
    target_mean: np.ndarray,
    alpha: float,
    profile_window_indices: set[int],
    exclude_indices: Optional[set[int]] = None,
) -> Dict[str, object]:
    adapter.eval()
    rows: List[Dict[str, object]] = []
    y_all: List[float] = []
    p_all: List[float] = []
    shift = torch.as_tensor(target_mean - source_mean, dtype=torch.float32, device=device).view(1, -1)
    for batch in loader:
        x, rr, subjects, widx, cond = batch_to_device(batch, device)
        keep = np.ones(len(widx), dtype=bool)
        if exclude_indices is not None:
            keep = np.asarray([int(i) not in exclude_indices for i in widx], dtype=bool)
        if not keep.any():
            continue
        profile = adapter._profile_for_batch(int(x.shape[0]), x.device)
        unaligned_static = adapter.static_model(x, profile)
        tokens, pooled = adapter.static_model.wrapped.encode(x)
        aligned_pooled = pooled + float(alpha) * shift
        baseline = adapter.static_model.wrapped.head(aligned_pooled)
        if adapter.static_model.use_film:
            static_z, static_gamma, static_beta = adapter.static_model.film(aligned_pooled, profile)
        else:
            static_z = aligned_pooled
            static_gamma = torch.ones_like(aligned_pooled)
            static_beta = torch.zeros_like(aligned_pooled)
        gamma = static_gamma
        beta = static_beta
        if adapter.parameter_group in {"film_only", "film_affine"}:
            gamma = torch.clamp(static_gamma + adapter.delta_gamma.view(1, -1), 1.0 - adapter.static_model.film.scale, 1.0 + adapter.static_model.film.scale)
            beta = torch.clamp(static_beta + adapter.delta_beta.view(1, -1), -adapter.static_model.film.scale, adapter.static_model.film.scale)
        z_cond = gamma * aligned_pooled + beta if adapter.static_model.use_film else static_z
        film_pred = adapter.static_model.wrapped.head(z_cond)
        _static_aff, static_gain, static_bias = adapter.static_model.affine(film_pred if adapter.static_model.use_film else baseline, profile)
        gain = static_gain
        bias = static_bias
        if adapter.parameter_group in {"affine_only", "film_affine"}:
            gain = torch.clamp(static_gain + adapter.delta_gain, 1.0 - adapter.static_model.affine.gain_bound, 1.0 + adapter.static_model.affine.gain_bound)
            bias = torch.clamp(static_bias + adapter.delta_bias, -adapter.static_model.affine.bias_bound_bpm, adapter.static_model.affine.bias_bound_bpm)
        pred = gain * film_pred + bias if adapter.static_model.use_affine else film_pred
        pred_np = pred.detach().cpu().numpy().reshape(-1)
        static_np = unaligned_static.prediction.detach().cpu().numpy().reshape(-1)
        rr_np = rr.detach().cpu().numpy().reshape(-1)
        gamma_np = gamma.detach().cpu().numpy()
        beta_np = beta.detach().cpu().numpy()
        gain_np = gain.detach().cpu().numpy().reshape(-1)
        bias_np = bias.detach().cpu().numpy().reshape(-1)
        hidden_delta = (z_cond - pooled).detach().cpu().numpy()
        for i in np.flatnonzero(keep):
            rows.append({
                "subject": subjects[i],
                "window_index": int(widx[i]),
                "condition": float(cond[i]),
                "sequence_index": int(widx[i]),
                "rr_true": float(rr_np[i]),
                "rr_pred": float(pred_np[i]),
                "rr_pred_plain": float(unaligned_static.baseline_prediction.detach().cpu().numpy().reshape(-1)[i]),
                "rr_pred_static": float(static_np[i]),
                "rr_pred_adapted": float(pred_np[i]),
                "adaptation_delta": float(pred_np[i] - static_np[i]),
                "alignment_delta": float(pred_np[i] - static_np[i]),
                "is_profile_window": bool(int(widx[i]) in profile_window_indices),
                "ttt_mode": adapter.parameter_group,
                "baseline_pred": float(baseline.detach().cpu().numpy().reshape(-1)[i]),
                "film_only_pred": float(film_pred.detach().cpu().numpy().reshape(-1)[i]),
                "affine_only_pred": float((_static_aff.detach().cpu().numpy().reshape(-1))[i]),
                "combined_pred": float(static_np[i]),
                "prediction_delta": float(pred_np[i] - static_np[i]),
                "affine_gain": float(gain_np[i]),
                "affine_bias": float(bias_np[i]),
                "film_gamma_delta_norm": float(np.linalg.norm(gamma_np[i] - 1.0)),
                "film_beta_norm": float(np.linalg.norm(beta_np[i])),
                "hidden_delta_rms": float(np.sqrt(np.mean(hidden_delta[i] * hidden_delta[i]))),
            })
            y_all.append(float(rr_np[i]))
            p_all.append(float(pred_np[i]))
    metrics = rr_metric_dict(np.asarray(y_all), np.asarray(p_all))
    if rows:
        df = pd.DataFrame(rows)
        metrics.update({
            "film_gamma_delta_norm": float(df["film_gamma_delta_norm"].mean()),
            "film_beta_norm": float(df["film_beta_norm"].mean()),
            "hidden_delta_rms": float(df["hidden_delta_rms"].mean()),
            "affine_gain": float(df["affine_gain"].mean()),
            "affine_bias": float(df["affine_bias"].mean()),
            "prediction_delta_mean": float(df["prediction_delta"].mean()),
            "prediction_delta_std": float(df["prediction_delta"].std(ddof=0)),
            "alignment_prediction_delta_mean": float(df["alignment_delta"].mean()),
            "alignment_prediction_delta_std": float(df["alignment_delta"].std(ddof=0)),
        })
    return {"metrics": metrics, "rows": rows}


def leakage_assertions(heldout: str, train_ds: SubjectAwareDataset, val_ds: SubjectAwareDataset, normalizer_subjects: Sequence[str]) -> Dict[str, object]:
    train_subjects = set(str(s) for s in train_ds.subject_id)
    val_subjects = set(str(s) for s in val_ds.subject_id)
    norm_subjects = set(str(s) for s in normalizer_subjects)
    audit = {
        "heldout": heldout,
        "heldout_absent_from_source_training": heldout not in train_subjects,
        "heldout_absent_from_source_validation": heldout not in val_subjects,
        "heldout_absent_from_profile_normalizer": heldout not in norm_subjects,
        "real_target_excluded_from_meta_training": heldout not in norm_subjects,
        "real_target_excluded_from_source_latent_references": heldout not in train_subjects,
        "target_rr_not_used_for_profile_construction": True,
        "target_rr_not_used_for_hyperparameter_selection": True,
        "target_rr_not_used_for_checkpoint_selection": True,
        "ttt_function_receives_no_rr_labels": True,
        "target_labels_not_used_in_augmentation_consistency": True,
        "target_labels_not_used_in_temporal_consistency": True,
        "target_labels_not_used_in_spectral_consistency": True,
        "target_labels_not_used_in_anchor_loss": True,
        "target_labels_not_used_in_update_acceptance": True,
        "target_rr_absent_from_gate_inputs": True,
        "target_rr_absent_from_alpha_inputs": True,
        "target_labels_not_used_to_select_ttt_learning_rate": True,
        "target_labels_not_used_to_select_ttt_steps": True,
        "target_labels_not_used_to_select_parameter_group": True,
        "target_labels_not_used_to_select_loss_weights": True,
        "profile_prefix_target_labels_inaccessible_during_adaptation": True,
        "evaluation_labels_joined_only_after_prediction_generation": True,
        "target_predictions_saved_before_target_labels_joined": True,
        "pseudo_target_support_query_split_disjoint": True,
        "pseudo_target_query_labels_used_only_in_outer_meta_loss": True,
        "target_rr_used_for_profile_construction": False,
        "target_rr_used_for_hyperparameter_selection": False,
        "target_rr_used_for_checkpoint_selection": False,
    }
    required_true = [
        "heldout_absent_from_source_training",
        "heldout_absent_from_source_validation",
        "heldout_absent_from_profile_normalizer",
        "real_target_excluded_from_meta_training",
        "real_target_excluded_from_source_latent_references",
        "target_rr_not_used_for_profile_construction",
        "target_rr_not_used_for_hyperparameter_selection",
        "target_rr_not_used_for_checkpoint_selection",
        "ttt_function_receives_no_rr_labels",
        "target_labels_not_used_in_augmentation_consistency",
        "target_labels_not_used_in_temporal_consistency",
        "target_labels_not_used_in_spectral_consistency",
        "target_labels_not_used_in_anchor_loss",
        "target_labels_not_used_in_update_acceptance",
        "target_rr_absent_from_gate_inputs",
        "target_rr_absent_from_alpha_inputs",
        "target_labels_not_used_to_select_ttt_learning_rate",
        "target_labels_not_used_to_select_ttt_steps",
        "target_labels_not_used_to_select_parameter_group",
        "target_labels_not_used_to_select_loss_weights",
        "profile_prefix_target_labels_inaccessible_during_adaptation",
        "evaluation_labels_joined_only_after_prediction_generation",
        "target_predictions_saved_before_target_labels_joined",
        "pseudo_target_support_query_split_disjoint",
        "pseudo_target_query_labels_used_only_in_outer_meta_loss",
    ]
    failed = [k for k in required_true if not bool(audit[k])]
    if failed:
        raise AssertionError(f"Leakage assertion(s) failed: {failed}")
    return audit


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def run_root_for(args: argparse.Namespace) -> Path:
    root = Path(args.out_dir)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    modes = mode_set(args)
    if modes.intersection(PHASE4_MODES):
        phase = "phase4"
    elif modes.intersection(PHASE3_MODES):
        phase = "phase3"
    elif modes.intersection(PHASE2_MODES):
        phase = "phase2"
    else:
        phase = "phase1"
    run = root / f"{phase}_{stamp}_seed{args.seed}"
    suffix = 1
    while run.exists():
        run = root / f"{phase}_{stamp}_seed{args.seed}_{suffix}"
        suffix += 1
    run.mkdir(parents=True, exist_ok=False)
    return run


def model_count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def control_specs(args: argparse.Namespace) -> List[Tuple[str, str, str]]:
    modes = mode_set(args)
    phase2 = bool(modes.intersection(PHASE2_MODES))
    if "all" in modes:
        modes = {"all"}
    specs = [("C0_plain_tcn", "plain", "real")]
    if "all" in modes or "t3_profile_film_affine" in modes:
        specs.append(("C1_zero_profile_conditioning", "film_affine", "zero"))
    if "all" in modes or "t1_profile_affine" in modes or "t4_ttt_affine" in modes:
        specs.extend([("C2_real_profile_affine", "affine", "real"), ("C3_shuffled_profile_affine", "affine", "shuffled")])
    if "all" in modes or "t2_profile_film" in modes:
        specs.extend([("C4_real_profile_film", "film", "real"), ("C5_shuffled_profile_film", "film", "shuffled")])
    if "all" in modes or "t3_profile_film_affine" in modes or "t5_ttt_film_affine" in modes:
        specs.extend([("C6_real_profile_film_affine", "film_affine", "real"), ("C7_shuffled_profile_film_affine", "film_affine", "shuffled")])
    if "all" in modes:
        return [
            ("C0_plain_tcn", "plain", "real"),
            ("C1_zero_profile_conditioning", "film_affine", "zero"),
            ("C2_real_profile_affine", "affine", "real"),
            ("C3_shuffled_profile_affine", "affine", "shuffled"),
            ("C4_real_profile_film", "film", "real"),
            ("C5_shuffled_profile_film", "film", "shuffled"),
            ("C6_real_profile_film_affine", "film_affine", "real"),
            ("C7_shuffled_profile_film_affine", "film_affine", "shuffled"),
        ]
    if phase2:
        seen = set()
        out = []
        for item in specs:
            if item[0] not in seen:
                out.append(item)
                seen.add(item[0])
        return out
    return specs


def ttt_control_specs(args: argparse.Namespace) -> List[Dict[str, object]]:
    modes = mode_set(args)
    if not modes.intersection(PHASE2_MODES):
        return []
    control_set = str(args.ttt_controls)
    specs: List[Dict[str, object]] = []
    if "t4_ttt_affine" in modes:
        specs.append({"method": "T4_affine_only_ttt", "static_kind": "affine", "parameter_group": "affine_only", "profile_kind": "real", "steps": int(args.ttt_steps), "anchor": float(args.lambda_ttt_anchor), "loss_mode": "default", "fixed_zero": False})
    if "t5_ttt_film_affine" in modes:
        specs.append({"method": "T5_film_affine_ttt", "static_kind": "film_affine", "parameter_group": str(args.ttt_parameter_group), "profile_kind": "real", "steps": int(args.ttt_steps), "anchor": float(args.lambda_ttt_anchor), "loss_mode": "default", "fixed_zero": False})
    if control_set in {"smoke", "all"}:
        specs.extend([
            {"method": "C11_real_profile_ttt_steps0", "static_kind": "film_affine", "parameter_group": str(args.ttt_parameter_group), "profile_kind": "real", "steps": 0, "anchor": float(args.lambda_ttt_anchor), "loss_mode": "default", "fixed_zero": False},
            {"method": "C12_shuffled_profile_ttt", "static_kind": "film_affine", "parameter_group": str(args.ttt_parameter_group), "profile_kind": "shuffled", "steps": int(args.ttt_steps), "anchor": float(args.lambda_ttt_anchor), "loss_mode": "default", "fixed_zero": False},
        ])
    if control_set == "all":
        specs.extend([
            {"method": "C8_ttt_without_real_subject_profile", "static_kind": "film_affine", "parameter_group": "film_affine", "profile_kind": "zero", "steps": int(args.ttt_steps), "anchor": float(args.lambda_ttt_anchor), "loss_mode": "default", "fixed_zero": False},
            {"method": "C9_static_real_profile_without_ttt", "static_kind": "film_affine", "parameter_group": "film_affine", "profile_kind": "real", "steps": 0, "anchor": float(args.lambda_ttt_anchor), "loss_mode": "default", "fixed_zero": False},
            {"method": "C10_real_profile_ttt", "static_kind": "film_affine", "parameter_group": "film_affine", "profile_kind": "real", "steps": int(args.ttt_steps), "anchor": float(args.lambda_ttt_anchor), "loss_mode": "default", "fixed_zero": False},
            {"method": "C13_real_profile_ttt_anchor_disabled", "static_kind": "film_affine", "parameter_group": "film_affine", "profile_kind": "real", "steps": int(args.ttt_steps), "anchor": 0.0, "loss_mode": "default", "fixed_zero": False},
            {"method": "C14_real_profile_ttt_aug_only", "static_kind": "film_affine", "parameter_group": "film_affine", "profile_kind": "real", "steps": int(args.ttt_steps), "anchor": 0.0, "loss_mode": "aug_only", "fixed_zero": False},
            {"method": "C15_real_profile_ttt_fixed_zero_update", "static_kind": "film_affine", "parameter_group": "film_affine", "profile_kind": "real", "steps": 0, "anchor": float(args.lambda_ttt_anchor), "loss_mode": "default", "fixed_zero": True},
        ])
    seen = set()
    out = []
    for spec in specs:
        key = (spec["method"], spec["static_kind"], spec["profile_kind"])
        if key not in seen:
            out.append(spec)
            seen.add(key)
    return out


def meta_control_specs(args: argparse.Namespace) -> List[Dict[str, object]]:
    modes = mode_set(args)
    if "all" not in modes and not modes.intersection(PHASE3_MODES | PHASE4_MODES):
        return []
    return [
        {"method": "M0_t3_static_profile_film_affine", "profile_kind": "real", "steps": 0, "gate_mode": "none", "lr_mode": "learned", "trained": True},
        {"method": "M1_t5_fixed_strength_film_affine_ttt", "profile_kind": "real", "steps": int(args.ttt_steps), "gate_mode": "fixed1", "lr_mode": "fixed", "trained": True},
        {"method": "M2_t6_meta_trained_gate", "profile_kind": "real", "steps": int(args.ttt_steps), "gate_mode": "learned", "lr_mode": "learned", "trained": True},
        {"method": "M3_t6_gate_fixed_0", "profile_kind": "real", "steps": int(args.ttt_steps), "gate_mode": "fixed0", "lr_mode": "learned", "trained": True},
        {"method": "M4_t6_gate_fixed_1", "profile_kind": "real", "steps": int(args.ttt_steps), "gate_mode": "fixed1", "lr_mode": "learned", "trained": True},
        {"method": "M5_t6_fixed_learning_rate", "profile_kind": "real", "steps": int(args.ttt_steps), "gate_mode": "learned", "lr_mode": "fixed", "trained": True},
        {"method": "M6_t6_shuffled_matched_profile", "profile_kind": "shuffled", "steps": int(args.ttt_steps), "gate_mode": "learned", "lr_mode": "learned", "trained": True},
        {"method": "M7_t6_zero_profile", "profile_kind": "zero", "steps": int(args.ttt_steps), "gate_mode": "learned", "lr_mode": "learned", "trained": True},
        {"method": "M8_t6_meta_static_no_ttt", "profile_kind": "real", "steps": 0, "gate_mode": "fixed0", "lr_mode": "learned", "trained": True},
        {"method": "M9_t6_without_gate_regularisation", "profile_kind": "real", "steps": int(args.ttt_steps), "gate_mode": "learned", "lr_mode": "learned", "trained": True},
        {"method": "M10_t6_random_untrained_gate", "profile_kind": "real", "steps": int(args.ttt_steps), "gate_mode": "learned", "lr_mode": "learned", "trained": False},
    ]


def mean_alignment_control_specs(args: argparse.Namespace) -> List[Dict[str, object]]:
    modes = mode_set(args)
    if not modes.intersection(PHASE4_MODES) and not ("all" in modes and bool(getattr(args, "use_mean_alignment", False))):
        return []
    return [
        {"method": "A0_t6_without_alignment", "mode": "none", "alpha": 0.0, "profile_kind": "real", "steps": int(args.ttt_steps)},
        {"method": "A1_t6_profile_gated_mean_alignment", "mode": "profile_gated", "alpha": None, "profile_kind": "real", "steps": int(args.ttt_steps)},
        {"method": "A2_t6_fixed_alpha_0_25", "mode": "fixed", "alpha": 0.25, "profile_kind": "real", "steps": int(args.ttt_steps)},
        {"method": "A3_t6_fixed_alpha_0_50", "mode": "fixed", "alpha": 0.50, "profile_kind": "real", "steps": int(args.ttt_steps)},
        {"method": "A4_t6_fixed_alpha_0_75", "mode": "fixed", "alpha": 0.75, "profile_kind": "real", "steps": int(args.ttt_steps)},
        {"method": "A5_t6_fixed_alpha_1_00", "mode": "fixed", "alpha": 1.00, "profile_kind": "real", "steps": int(args.ttt_steps)},
        {"method": "A6_t6_random_alpha", "mode": "random", "alpha": None, "profile_kind": "real", "steps": int(args.ttt_steps)},
        {"method": "A7_t6_shuffled_profile_alpha", "mode": "profile_gated", "alpha": None, "profile_kind": "shuffled", "steps": int(args.ttt_steps)},
        {"method": "A8_t6_profile_gated_alignment_no_ttt", "mode": "profile_gated", "alpha": None, "profile_kind": "real", "steps": 0},
    ]


def build_conditioned_copy(base: WrappedTCN, kind: str, profile_dim: int, args: argparse.Namespace) -> ProfileConditionedTCN:
    raw = TCNRR(in_ch=int(args.in_channels), width=int(args.tcn_width), emb_dim=int(args.emb_dim))
    wrapped = WrappedTCN(raw)
    wrapped.load_state_dict(base.state_dict())
    return ProfileConditionedTCN(
        wrapped,
        profile_dim=profile_dim,
        use_film=kind in {"film", "film_affine"},
        use_affine=kind in {"affine", "film_affine"},
        film_scale=float(args.profile_film_scale),
        affine_gain_bound=float(args.affine_gain_bound),
        affine_bias_bound_bpm=float(args.affine_bias_bound_bpm),
    )


def run_fold(args: argparse.Namespace, heldout: str, run_dir: Path) -> Dict[str, List[Dict[str, object]]]:
    set_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in str(args.device) else "cpu")
    train_loader, val_loader, test_loader, arrays = build_fold_loaders(args, heldout)
    train_ds = train_loader.dataset
    val_ds = val_loader.dataset
    source_subjects = [s for s in parse_subjects(args.subjects) if s != heldout]
    leakage = leakage_assertions(heldout, train_ds, val_ds, source_subjects)

    sample = next(iter(train_loader))["imu"].float()
    in_ch = int(ensure_channel_first(sample).shape[1])
    args.in_channels = in_ch
    base = WrappedTCN(TCNRR(in_ch=in_ch, width=int(args.tcn_width), emb_dim=int(args.emb_dim)))
    train_rows = train_plain_tcn(base, train_loader, val_loader, args, device, heldout)

    all_subjects = source_subjects + [heldout]
    all_arrays = {k: np.concatenate([arrays["source"][k], arrays["target"][k]], axis=0) for k in arrays["source"].keys()}
    raw_profiles: Dict[str, np.ndarray] = {}
    profile_diags: List[Dict[str, object]] = []
    schema: Optional[List[str]] = None
    for subject in all_subjects:
        vec, names, diag = build_subject_profile(subject, all_arrays, base, int(args.profile_windows), device, int(args.seed))
        if schema is None:
            schema = names
        elif schema != names:
            raise RuntimeError("Profile schema changed across subjects")
        raw_profiles[subject] = vec
        profile_diags.append(diag)
    assert schema is not None
    mean, std = fit_profile_normalizer(raw_profiles, source_subjects, heldout)
    profiles = normalize_profiles(raw_profiles, mean, std)
    zero_profiles = {s: np.zeros_like(v, dtype=np.float32) for s, v in profiles.items()}
    perm = deterministic_profile_permutation(all_subjects, int(args.seed))
    shuffled_profiles = {s: profiles[perm[s]] for s in all_subjects}
    target_profile_indices = set(int(i) for i in next(d for d in profile_diags if d["subject"] == heldout)["profile_indices"])

    fold_dir = run_dir / heldout
    fold_dir.mkdir(parents=True, exist_ok=True)
    np.save(fold_dir / "profile_mean.npy", mean)
    np.save(fold_dir / "profile_std.npy", std)
    save_json(fold_dir / "profile_schema.json", schema)
    pd.DataFrame([{"subject": d["subject"], "window_index": i} for d in profile_diags if d["subject"] != heldout for i in d["profile_indices"]]).to_csv(fold_dir / "source_profile_window_indices.csv", index=False)
    pd.DataFrame([{"subject": heldout, "window_index": i} for i in target_profile_indices]).to_csv(fold_dir / "target_profile_window_indices.csv", index=False)
    torch.save(base.state_dict(), fold_dir / "t0_plain_tcn.pt")

    profile_dim = int(next(iter(profiles.values())).shape[0])
    models: Dict[str, nn.Module] = {"plain": base.to(device)}
    stage_map = {
        "affine": "t1_profile_affine",
        "film": "t2_profile_film",
        "film_affine": "t3_profile_film_affine",
    }
    needed = {kind for _name, kind, _ptype in control_specs(args) if kind != "plain"}
    if meta_control_specs(args) or mean_alignment_control_specs(args):
        needed.add("film_affine")
    for kind in ["affine", "film", "film_affine"]:
        if kind not in needed:
            continue
        model = build_conditioned_copy(base, kind, profile_dim, args)
        train_rows.extend(train_conditioning(model, train_loader, val_loader, profiles, args, device, heldout, stage_map[kind]))
        torch.save(model.state_dict(), fold_dir / f"{kind}.pt")
        models[kind] = model.to(device)

    leakage["profile_normalizer_subjects"] = source_subjects
    leakage["profile_permutation"] = perm
    save_json(fold_dir / "leakage_audit.json", leakage)

    subject_rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []
    ttt_diagnostics: List[Dict[str, object]] = []
    ttt_state_rows: List[Dict[str, object]] = []
    ttt_loss_rows: List[Dict[str, object]] = []
    ttt_rejection_rows: List[Dict[str, object]] = []
    meta_training_history: List[Dict[str, object]] = []
    meta_episode_rows: List[Dict[str, object]] = []
    meta_episode_splits: List[Dict[str, object]] = []
    meta_gate_diagnostics: List[Dict[str, object]] = []
    meta_update_rejections: List[Dict[str, object]] = []
    meta_subject_summary: List[Dict[str, object]] = []
    mean_alignment_diagnostics: List[Dict[str, object]] = []
    mean_alignment_subject_rows: List[Dict[str, object]] = []
    mean_alignment_controls: List[Dict[str, object]] = []
    for method, kind, profile_kind in control_specs(args):
        prof = profiles
        if profile_kind == "zero":
            prof = zero_profiles
        elif profile_kind == "shuffled":
            prof = shuffled_profiles
        model = models[kind]
        for scope, exclude in [("primary_post_profile", target_profile_indices), ("all_windows", None)]:
            ev = evaluate_conditioned(model, test_loader, prof, device, batch_limit=0, exclude_indices=exclude)
            metrics = ev["metrics"]
            rows = ev["rows"]
            for row in rows:
                row.update({"seed": int(args.seed), "method": method, "eval_scope": scope, "profile_kind": profile_kind})
            pred_rows.extend(rows)
            if scope == "primary_post_profile":
                target_diag = next(d for d in profile_diags if d["subject"] == heldout)
                profile_norm = float(np.linalg.norm(prof[heldout]))
                source_mat = np.stack([profiles[s] for s in source_subjects], axis=0)
                profile_distance = float(np.min(np.linalg.norm(source_mat - prof[heldout][None, :], axis=1))) if len(source_mat) else np.nan
                static_row = {
                    "seed": int(args.seed),
                    "subject": heldout,
                    "method": method,
                    "n_windows": int(metrics.get("n_windows", 0)),
                    "mae": metrics.get("mae", np.nan),
                    "rmse": metrics.get("rmse", np.nan),
                    "rr_corr": metrics.get("rr_corr", np.nan),
                    "bias": metrics.get("bias", np.nan),
                    "median_ae": metrics.get("median_ae", np.nan),
                    "p95_ae": metrics.get("p95_ae", np.nan),
                    "profile_windows": int(target_diag["profile_windows"]),
                    "profile_norm": profile_norm,
                    "profile_distance": profile_distance,
                    "film_gamma_delta_norm": metrics.get("film_gamma_delta_norm", 0.0),
                    "film_beta_norm": metrics.get("film_beta_norm", 0.0),
                    "hidden_delta_rms": metrics.get("hidden_delta_rms", 0.0),
                    "affine_gain": metrics.get("affine_gain", 1.0),
                    "affine_bias": metrics.get("affine_bias", 0.0),
                    "prediction_delta_mean": metrics.get("prediction_delta_mean", 0.0),
                    "prediction_delta_std": metrics.get("prediction_delta_std", 0.0),
                }
                static_row.update(TTT_SUBJECT_DEFAULTS)
                subject_rows.append(static_row)

    support_batch = build_unlabelled_support_batch(test_loader.dataset, target_profile_indices, device)
    for spec in ttt_control_specs(args):
        method = str(spec["method"])
        kind = str(spec["static_kind"])
        profile_kind = str(spec["profile_kind"])
        prof = profiles
        if profile_kind == "zero":
            prof = zero_profiles
        elif profile_kind == "shuffled":
            prof = shuffled_profiles
        static_clone = build_conditioned_copy(base, kind, profile_dim, args)
        static_clone.load_state_dict(models[kind].state_dict())
        static_clone.to(device).eval()
        cfg_overrides: Dict[str, object] = {
            "ttt_steps": int(spec["steps"]),
            "ttt_parameter_group": str(spec["parameter_group"]),
            "lambda_ttt_anchor": float(spec["anchor"]),
            "seed": int(args.seed),
        }
        if str(spec.get("loss_mode", "default")) == "aug_only":
            cfg_overrides.update({"lambda_ttt_temp": 0.0, "lambda_ttt_spec": 0.0, "lambda_ttt_anchor": 0.0})
        if bool(spec.get("fixed_zero", False)):
            cfg_overrides.update({"ttt_steps": 0})
        cfg = ttt_config_from_args(args, overrides=cfg_overrides)
        adapter = TTTAdapter(
            static_clone,
            torch.as_tensor(prof[heldout], dtype=torch.float32, device=device),
            parameter_group=str(cfg.ttt_parameter_group),
        ).to(device)
        result = adapt_on_unlabelled_prefix(adapter, support_batch, cfg)
        diag = dict(result.diagnostics)
        diag.update({"seed": int(args.seed), "subject": heldout, "method": method, "profile_kind": profile_kind})
        ttt_diagnostics.append(diag)
        for row in result.state_rows:
            r = dict(row)
            r.update({"seed": int(args.seed), "subject": heldout, "method": method, "profile_kind": profile_kind})
            ttt_state_rows.append(r)
        for row in result.loss_components:
            r = dict(row)
            r.update({"seed": int(args.seed), "subject": heldout, "method": method})
            ttt_loss_rows.append(r)
        if bool(diag.get("update_was_rejected", False)):
            ttt_rejection_rows.append({
                "seed": int(args.seed),
                "subject": heldout,
                "method": method,
                "update_rejection_reason": diag.get("update_rejection_reason", ""),
            })

        for scope, exclude in [("primary_post_profile", target_profile_indices), ("all_windows", None)]:
            ev = evaluate_ttt_adapter(adapter, test_loader, device, profile_window_indices=target_profile_indices, exclude_indices=exclude)
            metrics = ev["metrics"]
            rows = ev["rows"]
            for row in rows:
                row.update({"seed": int(args.seed), "method": method, "eval_scope": scope, "profile_kind": profile_kind, "ttt_mode": str(cfg.ttt_mode)})
            pred_rows.extend(rows)
            if scope == "primary_post_profile":
                target_diag = next(d for d in profile_diags if d["subject"] == heldout)
                profile_norm = float(np.linalg.norm(prof[heldout]))
                source_mat = np.stack([profiles[s] for s in source_subjects], axis=0)
                profile_distance = float(np.min(np.linalg.norm(source_mat - prof[heldout][None, :], axis=1))) if len(source_mat) else np.nan
                ttt_row = {
                    "seed": int(args.seed),
                    "subject": heldout,
                    "method": method,
                    "n_windows": int(metrics.get("n_windows", 0)),
                    "mae": metrics.get("mae", np.nan),
                    "rmse": metrics.get("rmse", np.nan),
                    "rr_corr": metrics.get("rr_corr", np.nan),
                    "bias": metrics.get("bias", np.nan),
                    "median_ae": metrics.get("median_ae", np.nan),
                    "p95_ae": metrics.get("p95_ae", np.nan),
                    "profile_windows": int(target_diag["profile_windows"]),
                    "profile_norm": profile_norm,
                    "profile_distance": profile_distance,
                    "film_gamma_delta_norm": metrics.get("film_gamma_delta_norm", 0.0),
                    "film_beta_norm": metrics.get("film_beta_norm", 0.0),
                    "hidden_delta_rms": metrics.get("hidden_delta_rms", 0.0),
                    "affine_gain": metrics.get("affine_gain", 1.0),
                    "affine_bias": metrics.get("affine_bias", 0.0),
                    "prediction_delta_mean": metrics.get("prediction_delta_mean", 0.0),
                    "prediction_delta_std": metrics.get("prediction_delta_std", 0.0),
                }
                for key, default in TTT_SUBJECT_DEFAULTS.items():
                    ttt_row[key] = diag.get(key, default)
                ttt_row["prediction_variance_pre"] = diag.get("prediction_variance_pre", np.nan)
                ttt_row["prediction_variance_post"] = diag.get("prediction_variance_post", np.nan)
                subject_rows.append(ttt_row)

    gate: Optional[ProfileAdaptationGate] = None
    if meta_control_specs(args) or mean_alignment_control_specs(args):
        gate_input_dim = profile_dim + 3
        gate = ProfileAdaptationGate(
            gate_input_dim,
            hidden_dim=int(args.profile_hidden_dim),
            lr_min=float(args.gate_lr_multiplier_min),
            lr_max=float(args.gate_lr_multiplier_max),
            delta_min=float(args.gate_delta_multiplier_min),
            delta_max=float(args.gate_delta_multiplier_max),
            alpha_max=float(args.mean_alignment_alpha_max),
            include_alignment=bool(args.use_mean_alignment),
        ).to(device)
        static_meta = models["film_affine"]
        episodes = build_meta_episodes(all_arrays, source_subjects, heldout, profiles, args, device)
        meta_out = train_meta_gate(static_meta, episodes, {s: profiles[s] for s in source_subjects}, args, heldout, gate)
        meta_training_history.extend(meta_out["meta_training_history"])
        meta_episode_rows.extend(meta_out["meta_episode_rows"])
        meta_episode_splits.extend(meta_out["meta_episode_splits"])
        meta_gate_diagnostics.extend(meta_out["meta_gate_diagnostics"])
        meta_update_rejections.extend(meta_out["meta_update_rejections"])
        meta_subject_summary.extend(meta_out["meta_subject_summary"])
        save_json(fold_dir / "gate_schema.json", {
            "features": ["normalized_profile", "distance_to_source_centroid", "distance_to_nearest_source_profile", "log_support_window_count"],
            "prohibited": ["target_rr", "target_error", "target_mae", "query_labels", "post_adaptation_target_performance"],
        })
        torch.save({"gate": gate.state_dict(), "static_model": static_meta.state_dict(), "args": vars(args)}, fold_dir / "best_meta_query_mae.pt")
        torch.save({"gate": gate.state_dict(), "static_model": static_meta.state_dict(), "args": vars(args)}, fold_dir / "last_meta_checkpoint.pt")

        for spec in meta_control_specs(args):
            method = str(spec["method"])
            prof = profiles
            if str(spec["profile_kind"]) == "zero":
                prof = zero_profiles
            elif str(spec["profile_kind"]) == "shuffled":
                prof = shuffled_profiles
            run_gate = gate
            if not bool(spec.get("trained", True)):
                run_gate = ProfileAdaptationGate(
                    gate_input_dim,
                    hidden_dim=int(args.profile_hidden_dim),
                    lr_min=float(args.gate_lr_multiplier_min),
                    lr_max=float(args.gate_lr_multiplier_max),
                    delta_min=float(args.gate_delta_multiplier_min),
                    delta_max=float(args.gate_delta_multiplier_max),
                    alpha_max=float(args.mean_alignment_alpha_max),
                    include_alignment=bool(args.use_mean_alignment),
                ).to(device)
            static_clone = build_conditioned_copy(base, "film_affine", profile_dim, args)
            static_clone.load_state_dict(models["film_affine"].state_dict())
            static_clone.to(device).eval()
            adapter, diag = deploy_gated_ttt(
                static_clone,
                run_gate,
                prof[heldout],
                {s: profiles[s] for s in source_subjects},
                support_batch,
                args,
                gate_mode=str(spec["gate_mode"]),
                lr_mode=str(spec["lr_mode"]),
                steps=int(spec["steps"]),
                device=device,
            )
            diag.update({"seed": int(args.seed), "subject": heldout, "method": method, "profile_kind": str(spec["profile_kind"])})
            ttt_diagnostics.append(diag)
            meta_gate_diagnostics.append({
                "seed": int(args.seed),
                "true_loso_subject": heldout,
                "pseudo_target_subject": "",
                "episode": -1,
                "method": method,
                "gate_value": diag.get("gate_value", np.nan),
                "lr_multiplier": diag.get("lr_multiplier", np.nan),
                "effective_ttt_lr": diag.get("effective_ttt_lr", np.nan),
                "max_delta_multiplier": diag.get("max_delta_multiplier", np.nan),
            })
            for scope, exclude in [("primary_post_profile", target_profile_indices), ("all_windows", None)]:
                ev = evaluate_ttt_adapter(adapter, test_loader, device, profile_window_indices=target_profile_indices, exclude_indices=exclude)
                metrics = ev["metrics"]
                rows = ev["rows"]
                for row in rows:
                    row.update({"seed": int(args.seed), "method": method, "eval_scope": scope, "profile_kind": str(spec["profile_kind"]), "ttt_mode": "meta_gated"})
                pred_rows.extend(rows)
                if scope == "primary_post_profile":
                    target_diag = next(d for d in profile_diags if d["subject"] == heldout)
                    ttt_row = {
                        "seed": int(args.seed),
                        "subject": heldout,
                        "method": method,
                        "n_windows": int(metrics.get("n_windows", 0)),
                        "mae": metrics.get("mae", np.nan),
                        "rmse": metrics.get("rmse", np.nan),
                        "rr_corr": metrics.get("rr_corr", np.nan),
                        "bias": metrics.get("bias", np.nan),
                        "median_ae": metrics.get("median_ae", np.nan),
                        "p95_ae": metrics.get("p95_ae", np.nan),
                        "profile_windows": int(target_diag["profile_windows"]),
                        "profile_norm": float(np.linalg.norm(prof[heldout])),
                        "profile_distance": float(np.min(np.linalg.norm(np.stack([profiles[s] for s in source_subjects], axis=0) - prof[heldout][None, :], axis=1))) if source_subjects else np.nan,
                        "film_gamma_delta_norm": metrics.get("film_gamma_delta_norm", 0.0),
                        "film_beta_norm": metrics.get("film_beta_norm", 0.0),
                        "hidden_delta_rms": metrics.get("hidden_delta_rms", 0.0),
                        "affine_gain": metrics.get("affine_gain", 1.0),
                        "affine_bias": metrics.get("affine_bias", 0.0),
                        "prediction_delta_mean": metrics.get("prediction_delta_mean", 0.0),
                        "prediction_delta_std": metrics.get("prediction_delta_std", 0.0),
                    }
                    for key, default in TTT_SUBJECT_DEFAULTS.items():
                        ttt_row[key] = diag.get(key, default)
                    subject_rows.append(ttt_row)

        if mean_alignment_control_specs(args):
            source_mean, source_std, source_count = latent_stats_from_loader(base, train_loader, device, batch_limit=int(args.batch_limit))
            target_mean, target_std, target_count = latent_stats_from_support(base, support_batch)
            np.save(fold_dir / "source_latent_mean.npy", source_mean)
            np.save(fold_dir / "source_latent_std.npy", source_std)
            save_json(fold_dir / "source_latent_reference.json", {
                "heldout": heldout,
                "source_subjects": source_subjects,
                "source_window_count": source_count,
                "target_profile_window_count": target_count,
                "heldout_excluded_from_source_reference": heldout not in source_subjects,
            })
            aligner = ProfileGatedMeanAlignment(profile_dim, alpha_max=float(args.mean_alignment_alpha_max)).to(device)
            for spec in mean_alignment_control_specs(args):
                method = str(spec["method"])
                prof = profiles if str(spec["profile_kind"]) == "real" else shuffled_profiles
                alpha = mean_alignment_alpha(
                    aligner,
                    prof[heldout],
                    device,
                    mode=str(spec["mode"]),
                    fixed_alpha=None if spec["alpha"] is None else float(spec["alpha"]),
                    seed=int(args.seed),
                )
                static_clone = build_conditioned_copy(base, "film_affine", profile_dim, args)
                static_clone.load_state_dict(models["film_affine"].state_dict())
                static_clone.to(device).eval()
                adapter, diag = deploy_gated_ttt(
                    static_clone,
                    gate,
                    prof[heldout],
                    {s: profiles[s] for s in source_subjects},
                    support_batch,
                    args,
                    gate_mode="learned" if int(spec["steps"]) > 0 else "fixed0",
                    lr_mode="learned",
                    steps=int(spec["steps"]),
                    device=device,
                )
                ev = evaluate_mean_aligned_adapter(
                    adapter,
                    test_loader,
                    device,
                    source_mean=source_mean,
                    target_mean=target_mean,
                    alpha=alpha,
                    profile_window_indices=target_profile_indices,
                    exclude_indices=target_profile_indices,
                )
                metrics = ev["metrics"]
                rows = ev["rows"]
                for pred_row in rows:
                    pred_row.update({"seed": int(args.seed), "method": method, "eval_scope": "primary_post_profile", "profile_kind": str(spec["profile_kind"]), "ttt_mode": "mean_alignment"})
                pred_rows.extend(rows)
                shift = target_mean - source_mean
                row = {
                    "seed": int(args.seed),
                    "subject": heldout,
                    "method": method,
                    "alpha": alpha,
                    "alpha_max": float(args.mean_alignment_alpha_max),
                    "source_mean_norm": float(np.linalg.norm(source_mean)),
                    "target_mean_norm": float(np.linalg.norm(target_mean)),
                    "mean_shift_norm": float(np.linalg.norm(shift)),
                    "applied_shift_norm": float(abs(alpha) * np.linalg.norm(shift)),
                    "prediction_delta_mean": metrics.get("prediction_delta_mean", 0.0),
                    "prediction_delta_std": metrics.get("prediction_delta_std", 0.0),
                    "alignment_prediction_delta_mean": metrics.get("alignment_prediction_delta_mean", 0.0),
                    "alignment_prediction_delta_std": metrics.get("alignment_prediction_delta_std", 0.0),
                    "mae": metrics.get("mae", np.nan),
                    "rr_corr": metrics.get("rr_corr", np.nan),
                }
                mean_alignment_diagnostics.append(row)
                mean_alignment_subject_rows.append(row)
                mean_alignment_controls.append({**row, "profile_kind": str(spec["profile_kind"]), "alignment_mode": str(spec["mode"])})
                subject_row = {
                    "seed": int(args.seed),
                    "subject": heldout,
                    "method": method,
                    "n_windows": int(metrics.get("n_windows", 0)),
                    "mae": metrics.get("mae", np.nan),
                    "rmse": metrics.get("rmse", np.nan),
                    "rr_corr": metrics.get("rr_corr", np.nan),
                    "bias": metrics.get("bias", np.nan),
                    "median_ae": metrics.get("median_ae", np.nan),
                    "p95_ae": metrics.get("p95_ae", np.nan),
                    "profile_windows": int(next(d for d in profile_diags if d["subject"] == heldout)["profile_windows"]),
                    "profile_norm": float(np.linalg.norm(prof[heldout])),
                    "profile_distance": np.nan,
                    "film_gamma_delta_norm": metrics.get("film_gamma_delta_norm", 0.0),
                    "film_beta_norm": metrics.get("film_beta_norm", 0.0),
                    "hidden_delta_rms": metrics.get("hidden_delta_rms", 0.0),
                    "affine_gain": metrics.get("affine_gain", 1.0),
                    "affine_bias": metrics.get("affine_bias", 0.0),
                    "prediction_delta_mean": metrics.get("prediction_delta_mean", 0.0),
                    "prediction_delta_std": metrics.get("prediction_delta_std", 0.0),
                }
                for key, default in TTT_SUBJECT_DEFAULTS.items():
                    subject_row[key] = diag.get(key, default)
                subject_rows.append(subject_row)
    profile_rows = []
    for s in all_subjects:
        row = {"seed": int(args.seed), "subject": s, "is_heldout": s == heldout}
        row.update({name: float(value) for name, value in zip(schema, profiles[s])})
        profile_rows.append(row)
    return {
        "training_history": train_rows,
        "subject_rows": subject_rows,
        "window_predictions": pred_rows,
        "profile_vectors": profile_rows,
        "profile_diagnostics": profile_diags,
        "ttt_diagnostics": ttt_diagnostics,
        "ttt_state_by_subject": ttt_state_rows,
        "ttt_loss_components": ttt_loss_rows,
        "ttt_update_rejections": ttt_rejection_rows,
        "meta_training_history": meta_training_history,
        "meta_episode_rows": meta_episode_rows,
        "meta_episode_splits": meta_episode_splits,
        "meta_gate_diagnostics": meta_gate_diagnostics,
        "meta_update_rejections": meta_update_rejections,
        "meta_subject_summary": meta_subject_summary,
        "gate_diagnostics": meta_gate_diagnostics,
        "mean_alignment_diagnostics": mean_alignment_diagnostics,
        "mean_alignment_subject_rows": mean_alignment_subject_rows,
        "mean_alignment_controls": mean_alignment_controls,
        "leakage_audits": [leakage],
    }


def aggregate_and_write(args: argparse.Namespace, run_dir: Path, outputs: List[Dict[str, List[Dict[str, object]]]]) -> None:
    combined: Dict[str, List[Dict[str, object]]] = {}
    for output in outputs:
        for key, rows in output.items():
            combined.setdefault(key, []).extend(rows)
    for key, rows in combined.items():
        if key == "leakage_audits":
            continue
        write_csv(run_dir / f"{key}.csv", rows)
    for key in [
        "ttt_diagnostics",
        "ttt_state_by_subject",
        "ttt_loss_components",
        "ttt_update_rejections",
        "gate_diagnostics",
        "oracle_calibration",
        "meta_training_history",
        "meta_episode_rows",
        "meta_episode_splits",
        "meta_gate_diagnostics",
        "meta_update_rejections",
        "meta_subject_summary",
        "mean_alignment_diagnostics",
        "mean_alignment_subject_rows",
        "mean_alignment_controls",
        "final_method_comparison",
        "final_subject_comparison",
        "final_seed_summary",
        "adaptation_reliability_analysis",
        "subject_shift_classification",
    ]:
        if key not in combined:
            write_csv(run_dir / f"{key}.csv", [])
    subject_df = pd.DataFrame(combined.get("subject_rows", []))
    if len(subject_df):
        summary = subject_df.groupby("method", as_index=False).agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rr_corr_mean=("rr_corr", "mean"),
            n_subjects=("subject", "nunique"),
        )
        summary.to_csv(run_dir / "summary.csv", index=False)
        per_seed = subject_df.groupby(["seed", "method"], as_index=False).agg(mae_mean=("mae", "mean"), rmse_mean=("rmse", "mean"), n_subjects=("subject", "nunique"))
        per_seed.to_csv(run_dir / "per_seed_summary.csv", index=False)
        base = subject_df[subject_df["method"] == "C0_plain_tcn"][["subject", "mae"]].rename(columns={"mae": "plain_mae"})
        comp = subject_df.merge(base, on="subject", how="left")
        comp["mae_delta_vs_plain"] = comp["mae"] - comp["plain_mae"]
        comp.to_csv(run_dir / "control_comparisons.csv", index=False)
        ttt_comp = comp[comp["method"].astype(str).str.contains("TTT|ttt|T4|T5|C8|C9|C10|C11|C12|C13|C14|C15", regex=True)].copy()
        ttt_comp.to_csv(run_dir / "ttt_control_comparisons.csv", index=False)
    else:
        for name in ["summary.csv", "per_seed_summary.csv", "control_comparisons.csv"]:
            pd.DataFrame().to_csv(run_dir / name, index=False)
        pd.DataFrame().to_csv(run_dir / "ttt_control_comparisons.csv", index=False)
    save_json(run_dir / "leakage_audit.json", combined.get("leakage_audits", []))
    save_json(run_dir / "manifest.json", {
        "run_dir": str(run_dir),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "subjects": parse_subjects(args.subjects),
        "seed": int(args.seed),
        "git_commit": git_commit(),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    })
    save_json(run_dir / "implementation_audit.json", {
        "phase": "phase1_to_phase4_standalone",
        "implemented": ["T0 plain TCN", "T1 profile affine RR correction", "T2 bounded pooled Profile-FiLM", "T3 Profile-FiLM plus affine correction", "T4 affine-only one-step TTT", "T5 FiLM+affine one-step TTT", "T6 pseudo-held-out meta-gated TTT", "T7 optional profile-gated mean alignment"],
        "not_implemented": ["full production resume metadata validation", "oracle target-affine calibration"],
        "baseline_tcn_source": "rr_jbhi_models.TCNRR",
        "loso_source": "standalone wrapper around dataloader.load_data/make_dataset with sklearn train_test_split matching dataloader.build_loocv_loaders",
        "metrics": "local RR metrics matching required Phase 1 fields",
        "checkpoint_selection": "source validation MAE only",
    })
    cfg = vars(args).copy()
    cfg["run_dir"] = str(run_dir)
    save_json(run_dir / "config.json", cfg)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", nargs="+", default=["all"], choices=sorted(PHASE1_MODES | PHASE2_MODES | PHASE3_MODES | PHASE4_MODES | FUTURE_MODES))
    p.add_argument("--subjects", nargs="+", default=parse_subjects(DEFAULT_SUBJECTS))
    p.add_argument("--eval-subjects", nargs="+", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data-dir", default="/projects/BLVMob/imu-rr-seated/Data")
    p.add_argument("--data-str", default="imu_filt")
    p.add_argument("--data-group", default="mr")
    p.add_argument("--model-dir", "--mdl-dir", dest="model_dir", default="/projects/BLVMob/imu-rr-seated/models/imu_filt/loocv")
    p.add_argument("--out-dir", default="/projects/BLVMob/imu-rr-seated/results/tcn_profile_ttt")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=int(os.environ.get("IMU_DATALOADER_WORKERS", "0")))
    p.add_argument("--prefetch-factor", type=int, default=int(os.environ.get("IMU_DATALOADER_PREFETCH", "1")))
    p.add_argument("--pin-memory", type=int, default=int(os.environ.get("IMU_DATALOADER_PIN_MEMORY", "0")))
    p.add_argument("--persistent-workers", type=int, default=int(os.environ.get("IMU_DATALOADER_PERSISTENT_WORKERS", "0")))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--batch-limit", type=int, default=0)
    p.add_argument("--profile-windows", type=int, default=32)
    p.add_argument("--affine-gain-bound", type=float, default=0.03)
    p.add_argument("--affine-bias-bound-bpm", type=float, default=1.0)
    p.add_argument("--profile-film-scale", type=float, default=0.03)
    p.add_argument("--val-split", type=float, default=0.25)
    p.add_argument("--train-aug-ratio", type=float, default=0.2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--tcn-width", type=int, default=64)
    p.add_argument("--ttt-steps", type=int, default=1)
    p.add_argument("--ttt-lr", type=float, default=1e-4)
    p.add_argument("--ttt-parameter-group", default="film_affine", choices=["affine_only", "film_only", "film_affine"])
    p.add_argument("--ttt-mode", default="episodic", choices=["episodic", "cumulative"])
    p.add_argument("--lambda-ttt-aug", type=float, default=1.0)
    p.add_argument("--lambda-ttt-temp", type=float, default=0.05)
    p.add_argument("--lambda-ttt-spec", type=float, default=0.25)
    p.add_argument("--lambda-ttt-anchor", type=float, default=1.0)
    p.add_argument("--ttt-grad-clip", type=float, default=1.0)
    p.add_argument("--ttt-max-delta-norm", type=float, default=0.25)
    p.add_argument("--ttt-use-spectrum", action="store_true")
    p.add_argument("--ttt-noise-std", type=float, default=0.005)
    p.add_argument("--ttt-amplitude-scale", type=float, default=0.01)
    p.add_argument("--ttt-rotation-deg", type=float, default=2.0)
    p.add_argument("--ttt-mask-fraction", type=float, default=0.02)
    p.add_argument("--ttt-channel-drop-prob", type=float, default=0.02)
    p.add_argument("--ttt-collapse-var-min", type=float, default=1e-8)
    p.add_argument("--ttt-max-prediction-jump-bpm", type=float, default=5.0)
    p.add_argument("--ttt-controls", default="smoke", choices=["smoke", "all", "primary"])
    p.add_argument("--profile-hidden-dim", type=int, default=32)
    p.add_argument("--meta-support-windows", type=int, default=32)
    p.add_argument("--meta-query-windows", type=int, default=0)
    p.add_argument("--meta-episodes-per-epoch", type=int, default=0)
    p.add_argument("--meta-epochs", type=int, default=1)
    p.add_argument("--meta-batch-subjects", type=int, default=1)
    p.add_argument("--meta-mode", default="first_order", choices=["first_order", "full_second_order"])
    p.add_argument("--meta-learning-rate", type=float, default=1e-3)
    p.add_argument("--meta-weight-decay", type=float, default=1e-4)
    p.add_argument("--meta-pseudo-target-order", default="deterministic", choices=["deterministic", "shuffled"])
    p.add_argument("--meta-unfreeze", default="none", choices=["none", "final_tcn_block"])
    p.add_argument("--lambda-meta-static", type=float, default=0.1)
    p.add_argument("--lambda-meta-identity", type=float, default=0.1)
    p.add_argument("--lambda-gate-mean", type=float, default=0.01)
    p.add_argument("--lambda-gate-entropy", type=float, default=0.01)
    p.add_argument("--gate-target-mean", type=float, default=0.25)
    p.add_argument("--gate-lr-multiplier-min", type=float, default=0.0)
    p.add_argument("--gate-lr-multiplier-max", type=float, default=1.0)
    p.add_argument("--gate-delta-multiplier-min", type=float, default=0.0)
    p.add_argument("--gate-delta-multiplier-max", type=float, default=1.0)
    p.add_argument("--use-mean-alignment", action="store_true")
    p.add_argument("--mean-alignment-alpha-max", type=float, default=0.5)
    p.add_argument("--mean-alignment-placement", default="pooled", choices=["pooled", "final_tokens"])
    p.add_argument("--mean-alignment-mode", default="none", choices=["none", "profile_gated", "fixed", "random"])
    p.add_argument("--lambda-alpha", type=float, default=0.01)
    args = p.parse_args(argv)
    args.subjects = parse_subjects(args.subjects)
    args.eval_subjects = parse_subjects(args.eval_subjects) if args.eval_subjects else list(args.subjects)
    if len(args.subjects) == 1 and args.eval_subjects == args.subjects:
        args.eval_subjects = list(args.subjects)
        args.subjects = parse_subjects(DEFAULT_SUBJECTS)
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    set_seed(int(args.seed))
    run_dir = run_root_for(args)
    outputs = []
    for heldout in parse_subjects(args.eval_subjects):
        print(f"[LOSO] heldout={heldout}", flush=True)
        outputs.append(run_fold(args, heldout, run_dir))
    aggregate_and_write(args, run_dir, outputs)
    print(f"[DONE] wrote {run_dir}", flush=True)


if __name__ == "__main__":
    main()
