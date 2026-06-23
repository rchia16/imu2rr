#!/usr/bin/env python3
"""
Faithful RR-probe TTA baselines for IMU -> RR regression.

This script intentionally separates three things that were conflated in the
previous RR-probe TTA script:

1. none
   Source-trained frozen-encoder RR probe, evaluated on a held-out target split.

2. ssa
   Significant-subspace alignment following the official SSA design for
   regression TTA:
     - compute source feature mean/covariance;
     - take the top PCA basis;
     - weight PCA dimensions by abs(linear_regressor_weight @ basis) + bias;
     - align target features by symmetric diagonal-Gaussian KL in the PCA space;
     - use no target labels.

3. cmt
   Causal-mechanism-transfer-style few-shot domain adaptation following the
   released CMT augmenter structure:
     - fit an invertible ICA/flow model on source [feature, y] pairs;
     - encode the few labelled target calibration pairs;
     - generate latent-wise stochastic combinations;
     - invert them back to [feature, y] space;
     - reject generated points outside source support with a novelty detector;
     - fit a predictor on the augmented target-like pairs.

Important honesty note:
The public CMT package expects a user-supplied trainable invertible ICA object.
For this IMU/RR pipeline, this script supplies a tabular RealNVP as that
invertible object and implements the same augmenter mechanics locally. This is
faithful to the algorithmic interface and data-augmentation mechanism, but it is
not a byte-for-byte copy of the original package.

A fourth experimental mode, kin_ssa, is included only as a clearly labelled
research scaffold for your idea that subject kinematic shift should be aligned
unsupervised. Do not describe kin_ssa as original SSA/CMT.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from vit_pressure_crossmodal_stft_rr_core import (
    build_base_parser,
    default_subjects,
    pooled_features,
    rr_targets_from_batch,
    run_loocv_experiment,
    unpack_batch,
)

IMU_ISSUES_MR = [17, 26, 30]
IMU_ISSUES_L = [17, 21, 26, 30]
SUBJECTS = default_subjects(IMU_ISSUES_MR, IMU_ISSUES_L)


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def _imu_kinematic_summary(imu: torch.Tensor) -> torch.Tensor:
    """Simple subject-kinematic summary; used only by experimental kin_ssa.

    imu: (B,T,C)
    returns: (B, 5*C) = mean, std, rms, mean abs first-diff, std first-diff.
    """
    if imu.ndim != 3:
        raise ValueError(f"Expected imu (B,T,C), got {tuple(imu.shape)}")
    if imu.size(1) < imu.size(2):
        imu = imu.transpose(1, 2)
    d = imu[:, 1:, :] - imu[:, :-1, :]
    mean = imu.mean(dim=1)
    std = imu.std(dim=1, unbiased=False)
    rms = torch.sqrt((imu * imu).mean(dim=1).clamp_min(1e-12))
    d_abs = d.abs().mean(dim=1)
    d_std = d.std(dim=1, unbiased=False)
    return torch.cat([mean, std, rms, d_abs, d_std], dim=1)


@torch.no_grad()
def collect_rr_arrays(
    model: nn.Module,
    loader,
    device: str,
    max_batches: int = 0,
    include_kinematics: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Collect frozen IMU encoder features, RR labels, and optional kinematic summaries."""
    model.eval()
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ks: List[np.ndarray] = []
    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        imu, pressure, _, br, _ = unpack_batch(batch, device)
        hidden = model.encode(imu)
        z = pooled_features(hidden)
        rr_true = rr_targets_from_batch(pressure, br)
        xs.append(z.detach().cpu().numpy())
        ys.append(rr_true.detach().cpu().numpy().reshape(-1))
        if include_kinematics:
            ks.append(_imu_kinematic_summary(imu).detach().cpu().numpy())
    if not xs:
        raise RuntimeError("No batches available for RR feature extraction.")
    k_out = np.concatenate(ks, axis=0).astype(np.float32) if include_kinematics else None
    return (
        np.concatenate(xs, axis=0).astype(np.float32),
        np.concatenate(ys, axis=0).astype(np.float32),
        k_out,
    )


def split_target_calibration_eval(
    x: np.ndarray,
    y: np.ndarray,
    kinematic: Optional[np.ndarray],
    n_cal: int,
    *,
    seed: int,
    mode: str = "first",
    exclude_calibration_from_eval: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Split target into calibration/adaptation and held-out evaluation windows.

    Returns x_cal, y_cal, k_cal, x_eval, y_eval, k_eval, cal_indices.
    """
    n = x.shape[0]
    n_cal = max(0, min(int(n_cal), n))
    if n_cal == 0:
        cal_idx = np.zeros((0,), dtype=np.int64)
    elif mode == "first":
        cal_idx = np.arange(n_cal, dtype=np.int64)
    elif mode == "random":
        rng = np.random.default_rng(seed)
        cal_idx = np.sort(rng.choice(np.arange(n), size=n_cal, replace=False))
    elif mode == "even":
        cal_idx = np.unique(np.linspace(0, n - 1, n_cal).round().astype(np.int64))
        if cal_idx.size < n_cal:
            missing = np.setdiff1d(np.arange(n), cal_idx)
            cal_idx = np.sort(np.concatenate([cal_idx, missing[: n_cal - cal_idx.size]]))
    else:
        raise ValueError(f"Unsupported --target-calibration-mode={mode!r}")

    if exclude_calibration_from_eval and cal_idx.size > 0:
        mask = np.ones(n, dtype=bool)
        mask[cal_idx] = False
        eval_idx = np.where(mask)[0]
    else:
        eval_idx = np.arange(n, dtype=np.int64)

    if eval_idx.size == 0:
        raise RuntimeError(
            "No target evaluation windows remain. Reduce --target-calibration-windows "
            "or disable --exclude-calibration-from-eval."
        )

    k_cal = None if kinematic is None else kinematic[cal_idx]
    k_eval = None if kinematic is None else kinematic[eval_idx]
    return x[cal_idx], y[cal_idx], k_cal, x[eval_idx], y[eval_idx], k_eval, cal_idx


# ---------------------------------------------------------------------------
# Source RR regressor and metrics
# ---------------------------------------------------------------------------


class AffineFeatureAdapter(nn.Module):
    """Minimal feature-extractor parameters for TTA.

    Initialized as identity: z -> z * exp(log_scale) + bias.
    SSA adapts only these parameters by default while keeping the source-trained
    regressor fixed, matching the feature-alignment spirit of the SSA paper.
    """

    def __init__(self, d_in: int):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(d_in))
        self.bias = nn.Parameter(torch.zeros(d_in))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z * torch.exp(self.log_scale).view(1, -1) + self.bias.view(1, -1)


class FaithfulRRRegressor(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.adapter = AffineFeatureAdapter(d_in)
        self.regressor = nn.Linear(d_in, 1)

    def feature(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.feature(x)
        y = self.regressor(z).squeeze(-1)
        return y, z


class LinearPredictor(nn.Module):
    """Small predictor used for CMT augmented target-like features."""

    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Linear(d_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class TrainConfig:
    epochs: int
    lr: float
    weight_decay: float
    batch_size: int
    grad_clip: float


def train_source_rr_regressor(
    model: FaithfulRRRegressor,
    x_source: np.ndarray,
    y_source: np.ndarray,
    cfg: TrainConfig,
    device: str,
    *,
    train_adapter: bool = False,
) -> FaithfulRRRegressor:
    x = torch.tensor(x_source, dtype=torch.float32, device=device)
    y = torch.tensor(y_source, dtype=torch.float32, device=device)

    for p in model.parameters():
        p.requires_grad = True
    if not train_adapter:
        for p in model.adapter.parameters():
            p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    n = x.size(0)
    for _epoch in range(int(cfg.epochs)):
        model.train()
        perm = torch.randperm(n, device=device)
        for st in range(0, n, int(cfg.batch_size)):
            idx = perm[st : st + int(cfg.batch_size)]
            opt.zero_grad(set_to_none=True)
            pred, _ = model(x[idx])
            loss = F.smooth_l1_loss(pred, y[idx])
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()

    # keep source model in evaluation mode for subsequent frozen-head adaptation
    for p in model.parameters():
        p.requires_grad = True
    return model


@torch.no_grad()
def predict_rr(model: FaithfulRRRegressor | LinearPredictor, x: np.ndarray, device: str, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    preds = []
    for st in range(0, x.shape[0], batch_size):
        xb = torch.tensor(x[st : st + batch_size], dtype=torch.float32, device=device)
        out = model(xb)
        pred = out[0] if isinstance(out, tuple) else out
        preds.append(pred.detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds, axis=0)


@torch.no_grad()
def predict_features(model: FaithfulRRRegressor, x: np.ndarray, device: str, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    zs = []
    for st in range(0, x.shape[0], batch_size):
        xb = torch.tensor(x[st : st + batch_size], dtype=torch.float32, device=device)
        z = model.feature(xb)
        zs.append(z.detach().cpu().numpy())
    return np.concatenate(zs, axis=0)


def rr_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    err = y_pred - y_true
    corr = (
        float(np.corrcoef(y_true, y_pred)[0, 1])
        if y_true.size > 1 and np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8
        else float("nan")
    )
    return {
        f"{prefix}_mae": float(np.mean(np.abs(err))),
        f"{prefix}_rmse": float(np.sqrt(np.mean(err * err))),
        f"{prefix}_corr": corr,
        f"{prefix}_n": int(y_true.shape[0]),
    }


# ---------------------------------------------------------------------------
# Faithful SSA implementation
# ---------------------------------------------------------------------------


@dataclass
class SSAStats:
    mean: torch.Tensor       # (D,)
    basis: torch.Tensor      # (D,r)
    source_var: torch.Tensor # (r,)
    dim_weight: torch.Tensor # (r,)


def compute_ssa_stats_original(
    rr_model: FaithfulRRRegressor,
    x_source: np.ndarray,
    device: str,
    *,
    rank: int,
    weight_bias: float = 0.0,
    weight_exp: float = 1.0,
    normalize_weights: bool = False,
) -> SSAStats:
    """Source stats following the official SSA pipeline.

    Official feature_stats.py stores source feature mean, covariance eigvecs, and
    eigvals. TTA then aligns target projected moments in that PCA basis and
    weights dimensions by the regressor's sensitivity to each basis direction.
    """
    z_np = predict_features(rr_model, x_source, device)
    z = torch.tensor(z_np, dtype=torch.float32, device=device)
    mean = z.mean(dim=0)
    zc = z - mean.view(1, -1)
    denom = max(1, zc.size(0) - 1)
    cov = (zc.T @ zc) / float(denom)
    # eigh returns ascending eigenvalues; use largest first.
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    r = max(1, min(int(rank), zc.size(0) - 1, zc.size(1)))
    idx = order[:r]
    basis = eigvecs[:, idx].contiguous()
    source_var = eigvals[idx].clamp_min(1e-8)

    w = rr_model.regressor.weight.detach().view(1, -1).to(device)  # (1,D)
    dim_weight = torch.abs((w @ basis).view(-1)) + float(weight_bias)
    dim_weight = dim_weight.clamp_min(1e-12).pow(float(weight_exp))
    if normalize_weights:
        dim_weight = dim_weight / dim_weight.mean().clamp_min(1e-12)
    return SSAStats(mean.detach(), basis.detach(), source_var.detach(), dim_weight.detach())


def diagonal_gaussian_kl(mean_1: torch.Tensor, var_1: torch.Tensor, mean_2: torch.Tensor, var_2: torch.Tensor) -> torch.Tensor:
    """Elementwise KL[N(mean_1,var_1) || N(mean_2,var_2)]."""
    var_1 = var_1.clamp_min(1e-8)
    var_2 = var_2.clamp_min(1e-8)
    return 0.5 * (torch.log(var_2 / var_1) + (var_1 + (mean_1 - mean_2).pow(2)) / var_2 - 1.0)


def ssa_original_loss(z_target: torch.Tensor, stats: SSAStats) -> torch.Tensor:
    basis = stats.basis.to(z_target.device, z_target.dtype)
    mean = stats.mean.to(z_target.device, z_target.dtype)
    source_var = stats.source_var.to(z_target.device, z_target.dtype)
    weights = stats.dim_weight.to(z_target.device, z_target.dtype)

    a = (z_target - mean.view(1, -1)) @ basis
    target_mean = a.mean(dim=0)
    target_var = a.var(dim=0, unbiased=True).clamp_min(1e-8)
    source_mean = torch.zeros_like(target_mean)

    # Symmetric diagonal-Gaussian KL, per PCA dimension.
    kl_ts = diagonal_gaussian_kl(target_mean, target_var, source_mean, source_var)
    kl_st = diagonal_gaussian_kl(source_mean, source_var, target_mean, target_var)
    per_dim = 0.5 * (kl_ts + kl_st)
    return (weights * per_dim).sum()


def adapt_ssa_original(
    rr_model: FaithfulRRRegressor,
    x_source: np.ndarray,
    x_target_unlabeled: np.ndarray,
    args,
    device: str,
) -> Tuple[FaithfulRRRegressor, Dict[str, float]]:
    stats = compute_ssa_stats_original(
        rr_model,
        x_source,
        device,
        rank=int(args.ssa_rank),
        weight_bias=float(args.ssa_weight_bias),
        weight_exp=float(args.ssa_weight_exp),
        normalize_weights=bool(args.ssa_normalize_weights),
    )

    # Original TTA adapts feature-extractor parameters while regressor remains fixed.
    for p in rr_model.parameters():
        p.requires_grad = False
    for p in rr_model.adapter.parameters():
        p.requires_grad = True

    params = [p for p in rr_model.adapter.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(args.ssa_lr), weight_decay=float(args.ssa_weight_decay))
    xt = torch.tensor(x_target_unlabeled, dtype=torch.float32, device=device)
    bs = int(args.ssa_batch_size)
    rows = []
    for ep in range(1, int(args.ssa_epochs) + 1):
        rr_model.train()
        losses = []
        # bootstrap mini-batch updates; this matches TTA's batch-stat nature.
        n_steps = max(1, math.ceil(xt.size(0) / max(1, bs)))
        for _ in range(n_steps):
            idx = torch.randint(0, xt.size(0), (min(bs, xt.size(0)),), device=device)
            opt.zero_grad(set_to_none=True)
            _, zt = rr_model(xt[idx])
            loss = ssa_original_loss(zt, stats)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        rows.append(float(np.mean(losses)) if losses else float("nan"))

    for p in rr_model.parameters():
        p.requires_grad = True
    return rr_model, {
        "ssa_loss_last": rows[-1] if rows else float("nan"),
        "ssa_rank_used": int(stats.basis.shape[1]),
        "ssa_dim_weight_mean": float(stats.dim_weight.detach().cpu().mean()),
        "ssa_dim_weight_max": float(stats.dim_weight.detach().cpu().max()),
    }


# ---------------------------------------------------------------------------
# CMT-style augmenter: local implementation of the released augmenter mechanics
# ---------------------------------------------------------------------------


class AffineCoupling(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, mask_even: bool):
        super().__init__()
        mask = torch.zeros(dim)
        mask[::2] = 1.0 if mask_even else 0.0
        mask[1::2] = 0.0 if mask_even else 1.0
        self.register_buffer("mask", mask)
        self.net_s = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim))
        self.net_t = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim))

    def forward(self, z: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        m = self.mask.view(1, -1)
        z_masked = z * m
        s = torch.tanh(self.net_s(z_masked))
        t = self.net_t(z_masked)
        if not reverse:
            out = z_masked + (1.0 - m) * (z * torch.exp(s) + t)
            log_det = ((1.0 - m) * s).sum(dim=1)
        else:
            out = z_masked + (1.0 - m) * ((z - t) * torch.exp(-s))
            log_det = -((1.0 - m) * s).sum(dim=1)
        return out, log_det


class TabularRealNVP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 256, num_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([AffineCoupling(dim, hidden_dim, i % 2 == 0) for i in range(num_layers)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(x.size(0), device=x.device)
        z = x
        for layer in self.layers:
            z, ld = layer(z, reverse=False)
            log_det = log_det + ld
        return z, log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        for layer in reversed(self.layers):
            x, _ = layer(x, reverse=True)
        return x

    def nll(self, x: torch.Tensor) -> torch.Tensor:
        z, log_det = self.forward(x)
        log_pz = -0.5 * (z * z).sum(dim=1) - 0.5 * z.size(1) * math.log(2.0 * math.pi)
        return -(log_pz + log_det).mean()


class LocalICATransferAugmenter:
    """Local CMT augmenter matching the released ICATransferAugmenter mechanics.

    The original code does:
      data_src = hstack((X_src, Y_src)); fit invertible ICA + novelty detector;
      inputs = hstack((X_tgt, Y_tgt)); encode to latent e;
      stochastic/full latent dimension-wise combinations;
      invert; reject generated samples outside support; split back into X/Y.
    """

    def __init__(self, dim: int, args, device: str):
        self.dim = int(dim)
        self.args = args
        self.device = device
        self.scaler = StandardScaler()
        self.flow = TabularRealNVP(
            dim=dim,
            hidden_dim=int(args.cmt_flow_hidden),
            num_layers=int(args.cmt_flow_layers),
        ).to(device)
        self.novelty = OneClassSVM(nu=float(args.cmt_novelty_nu), gamma="auto")
        self.acceptance_ratio_: float = float("nan")

    def fit(self, x_src: np.ndarray, y_src: np.ndarray) -> None:
        data = np.hstack([x_src, y_src.reshape(-1, 1)]).astype(np.float32)
        data_s = self.scaler.fit_transform(data).astype(np.float32)
        self.novelty.fit(data_s)
        xt = torch.tensor(data_s, dtype=torch.float32, device=self.device)
        opt = torch.optim.AdamW(self.flow.parameters(), lr=float(self.args.cmt_flow_lr), weight_decay=float(self.args.cmt_flow_weight_decay))
        bs = int(self.args.cmt_flow_batch_size)
        n = xt.size(0)
        for _ep in range(int(self.args.cmt_flow_epochs)):
            perm = torch.randperm(n, device=self.device)
            for st in range(0, n, bs):
                idx = perm[st : st + bs]
                opt.zero_grad(set_to_none=True)
                loss = self.flow.nll(xt[idx])
                loss.backward()
                if float(self.args.grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(self.flow.parameters(), float(self.args.grad_clip))
                opt.step()

    @staticmethod
    def stochastic_combination(e: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
        perms = rng.integers(0, e.shape[0], size=(int(size), e.shape[1]))
        return np.hstack(tuple(e[perms[:, d], d][:, None] for d in range(e.shape[1]))).astype(np.float32)

    def augment_to_size(self, x_cal: np.ndarray, y_cal: np.ndarray, augment_size: int, *, seed: int, include_original: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if x_cal.shape[0] == 0:
            raise RuntimeError("CMT needs labelled target calibration windows; got zero.")
        data = np.hstack([x_cal, y_cal.reshape(-1, 1)]).astype(np.float32)
        data_s = self.scaler.transform(data).astype(np.float32)
        with torch.no_grad():
            e, _ = self.flow(torch.tensor(data_s, dtype=torch.float32, device=self.device))
        e_np = e.detach().cpu().numpy().astype(np.float32)

        rng = np.random.default_rng(seed)
        target_size = int(augment_size)
        ret = data_s.copy() if include_original else np.empty((0, data_s.shape[1]), dtype=np.float32)
        accepted_total = 0
        proposed_total = 0
        max_iter = int(self.args.cmt_aug_max_iter)
        for _ in range(max_iter):
            if ret.shape[0] >= target_size:
                break
            need = max(target_size - ret.shape[0], target_size)
            ebar = self.stochastic_combination(e_np, need, rng)
            with torch.no_grad():
                xbar_s = self.flow.inverse(torch.tensor(ebar, dtype=torch.float32, device=self.device)).detach().cpu().numpy()
            valid = self.novelty.predict(np.nan_to_num(xbar_s)) == 1
            valid &= ~np.isnan(xbar_s).any(axis=1)
            valid &= ~np.isinf(xbar_s).any(axis=1)
            accepted = xbar_s[valid].astype(np.float32)
            proposed_total += int(xbar_s.shape[0])
            accepted_total += int(accepted.shape[0])
            if accepted.shape[0] > 0:
                ret = np.vstack([ret, accepted])
        self.acceptance_ratio_ = accepted_total / max(1, proposed_total)
        if ret.shape[0] == 0:
            raise RuntimeError("CMT augmentation produced no accepted samples. Try increasing --cmt-novelty-nu or reducing flow epochs.")
        ret = ret[:target_size]
        out = self.scaler.inverse_transform(ret).astype(np.float32)
        return out[:, :-1], out[:, -1].reshape(-1)


def train_predictor_on_augmented(
    predictor: LinearPredictor,
    x_aug: np.ndarray,
    y_aug: np.ndarray,
    args,
    device: str,
) -> LinearPredictor:
    x = torch.tensor(x_aug, dtype=torch.float32, device=device)
    y = torch.tensor(y_aug, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(predictor.parameters(), lr=float(args.cmt_predictor_lr), weight_decay=float(args.cmt_predictor_weight_decay))
    bs = int(args.cmt_predictor_batch_size)
    n = x.size(0)
    for _ep in range(int(args.cmt_predictor_epochs)):
        predictor.train()
        perm = torch.randperm(n, device=device)
        for st in range(0, n, bs):
            idx = perm[st : st + bs]
            opt.zero_grad(set_to_none=True)
            pred = predictor(x[idx])
            loss = F.smooth_l1_loss(pred, y[idx])
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), float(args.grad_clip))
            opt.step()
    return predictor


def adapt_cmt_original_style(
    source_rr_model: FaithfulRRRegressor,
    x_source: np.ndarray,
    y_source: np.ndarray,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    args,
    device: str,
) -> Tuple[LinearPredictor, Dict[str, float]]:
    augmenter = LocalICATransferAugmenter(dim=x_source.shape[1] + 1, args=args, device=device)
    augmenter.fit(x_source, y_source)
    x_aug, y_aug = augmenter.augment_to_size(
        x_cal,
        y_cal,
        int(args.cmt_augment_size),
        seed=int(args.seed),
        include_original=bool(args.cmt_include_original),
    )
    predictor = LinearPredictor(x_source.shape[1]).to(device)
    if bool(args.cmt_warm_start_source_head):
        with torch.no_grad():
            predictor.net.weight.copy_(source_rr_model.regressor.weight.detach())
            predictor.net.bias.copy_(source_rr_model.regressor.bias.detach())
    predictor = train_predictor_on_augmented(predictor, x_aug, y_aug, args, device)
    return predictor, {
        "cmt_aug_n": int(x_aug.shape[0]),
        "cmt_aug_rr_mean": float(np.mean(y_aug)),
        "cmt_aug_rr_std": float(np.std(y_aug)),
        "cmt_acceptance_ratio": float(augmenter.acceptance_ratio_),
    }


# ---------------------------------------------------------------------------
# Experimental kinematic SSA scaffold, explicitly not original SSA/CMT.
# ---------------------------------------------------------------------------


def select_source_by_target_kinematics(
    x_source: np.ndarray,
    y_source: np.ndarray,
    k_source: np.ndarray,
    k_target: np.ndarray,
    *,
    fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select source samples nearest to target calibration kinematic centroid."""
    if k_source is None or k_target is None or k_target.shape[0] == 0:
        return x_source, y_source
    scaler = StandardScaler()
    ks = scaler.fit_transform(k_source)
    kt = scaler.transform(k_target)
    center = kt.mean(axis=0, keepdims=True)
    dist = ((ks - center) ** 2).mean(axis=1)
    n_keep = max(8, int(round(float(fraction) * x_source.shape[0])))
    n_keep = min(n_keep, x_source.shape[0])
    idx = np.argsort(dist)[:n_keep]
    return x_source[idx], y_source[idx]


# ---------------------------------------------------------------------------
# Hook and CLI
# ---------------------------------------------------------------------------


def faithful_rr_tta_evaluate(
    model: nn.Module,
    train_loader,
    test_loader,
    subject: str,
    device: str,
    args,
    out_dir: Optional[Path] = None,
) -> Dict[str, float]:
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    need_kin = str(args.rr_tta).lower() in {"kin_ssa", "ssa_kin"}
    x_source, y_source, k_source = collect_rr_arrays(
        model,
        train_loader,
        device,
        max_batches=int(args.rr_probe_source_batches),
        include_kinematics=need_kin,
    )
    x_target, y_target, k_target = collect_rr_arrays(
        model,
        test_loader,
        device,
        max_batches=0,
        include_kinematics=need_kin,
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

    pred_pre = predict_rr(rr_model, x_eval, device)
    metrics = rr_metrics(y_eval, pred_pre, prefix="rr_probe_pre")
    mode = str(args.rr_tta).lower()
    post_prefix = "rr_probe_post"
    extra: Dict[str, float] = {}

    if mode == "none":
        pred_post = pred_pre
    elif mode == "ssa":
        rr_model, extra = adapt_ssa_original(rr_model, x_source, x_cal if bool(args.ssa_use_calibration_only) else x_target, args, device)
        pred_post = predict_rr(rr_model, x_eval, device)
    elif mode == "kin_ssa":
        x_src_match, y_src_match = select_source_by_target_kinematics(
            x_source, y_source, k_source, k_cal, fraction=float(args.kin_source_fraction)
        )
        rr_model, extra = adapt_ssa_original(rr_model, x_src_match, x_cal if bool(args.ssa_use_calibration_only) else x_target, args, device)
        extra["kin_source_n"] = int(x_src_match.shape[0])
        pred_post = predict_rr(rr_model, x_eval, device)
    elif mode == "cmt":
        predictor, extra = adapt_cmt_original_style(rr_model, x_source, y_source, x_cal, y_cal, args, device)
        pred_post = predict_rr(predictor, x_eval, device)
    elif mode == "ssa_cmt":
        # Faithful composition: first unlabeled SSA feature alignment, then CMT
        # augmentation/predictor fitting on the same calibration split. CMT sees
        # source features after SSA adapter so the feature space is consistent.
        rr_model, extra_ssa = adapt_ssa_original(rr_model, x_source, x_cal if bool(args.ssa_use_calibration_only) else x_target, args, device)
        x_source_adapted = predict_features(rr_model, x_source, device)
        x_cal_adapted = predict_features(rr_model, x_cal, device) if x_cal.shape[0] else x_cal
        x_eval_adapted = predict_features(rr_model, x_eval, device)
        predictor, extra_cmt = adapt_cmt_original_style(rr_model, x_source_adapted, y_source, x_cal_adapted, y_cal, args, device)
        pred_post = predict_rr(predictor, x_eval_adapted, device)
        extra = {**{f"ssa_{k}": v for k, v in extra_ssa.items()}, **extra_cmt}
    elif mode == "ssa_kin":
        x_src_match, y_src_match = select_source_by_target_kinematics(
            x_source, y_source, k_source, k_cal, fraction=float(args.kin_source_fraction)
        )
        rr_model, extra_ssa = adapt_ssa_original(rr_model, x_src_match, x_cal if bool(args.ssa_use_calibration_only) else x_target, args, device)
        x_source_adapted = predict_features(rr_model, x_source, device)
        x_cal_adapted = predict_features(rr_model, x_cal, device) if x_cal.shape[0] else x_cal
        x_eval_adapted = predict_features(rr_model, x_eval, device)
        predictor, extra_cmt = adapt_cmt_original_style(rr_model, x_source_adapted, y_source, x_cal_adapted, y_cal, args, device)
        pred_post = predict_rr(predictor, x_eval_adapted, device)
        extra = {**{f"ssa_{k}": v for k, v in extra_ssa.items()}, **extra_cmt, "kin_source_n": int(x_src_match.shape[0])}
    else:
        raise ValueError(f"Unsupported --rr-tta={args.rr_tta!r}")

    metrics.update(rr_metrics(y_eval, pred_post, prefix=post_prefix))
    metrics.update(extra)
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
        pd.DataFrame({
            "rr_true": y_eval,
            "rr_pred_pre": pred_pre,
            "rr_pred_post": pred_post,
        }).to_csv(out_dir / f"faithful_rr_tta_predictions_{subject}.csv", index=False)
        with open(out_dir / f"faithful_rr_tta_metrics_{subject}.json", "w") as f:
            json.dump(metrics, f, indent=2)
    return metrics


def faithful_rr_tta_hook(model, sbj: str, train_loader, test_loader, device: str, args, sbj_dir: Path):
    metrics = faithful_rr_tta_evaluate(model, train_loader, test_loader, sbj, device, args, out_dir=sbj_dir / "faithful_rr_tta")
    print(f"FAITHFUL_RR_TTA {sbj}: {metrics}")
    return {"__summary_name__": "faithful_rr_tta_summary", "__summary_row__": {"subject": sbj, **metrics}}


def main() -> None:
    parser = build_base_parser(SUBJECTS, "faithful_rr_tta")

    # Source RR probe.
    parser.add_argument("--rr-probe-epochs", type=int, default=100)
    parser.add_argument("--rr-probe-lr", type=float, default=1e-3)
    parser.add_argument("--rr-probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--rr-probe-batch-size", type=int, default=256)
    parser.add_argument("--rr-probe-source-batches", type=int, default=0)
    parser.add_argument("--rr-probe-train-adapter", action="store_true", help="Train affine feature adapter on source; default keeps source adapter identity.")

    # Clean calibration/evaluation split.
    parser.add_argument("--target-calibration-windows", type=int, default=32)
    parser.add_argument("--target-calibration-mode", choices=["first", "random", "even"], default="first")
    parser.add_argument("--exclude-calibration-from-eval", action="store_true", default=True)
    parser.add_argument("--include-calibration-in-eval", dest="exclude_calibration_from_eval", action="store_false")

    # Adaptation mode.
    parser.add_argument("--rr-tta", default="none", choices=["none", "ssa", "cmt", "ssa_cmt", "kin_ssa", "ssa_kin"])

    # Faithful SSA knobs.
    parser.add_argument("--ssa-epochs", type=int, default=20)
    parser.add_argument("--ssa-lr", type=float, default=1e-4)
    parser.add_argument("--ssa-weight-decay", type=float, default=0.0)
    parser.add_argument("--ssa-batch-size", type=int, default=256)
    parser.add_argument("--ssa-weight-bias", type=float, default=0.0, help="Bias added to abs(regressor_weight @ basis), matching the original weighting form.")
    parser.add_argument("--ssa-weight-exp", type=float, default=1.0)
    parser.add_argument("--ssa-normalize-weights", action="store_true", help="Not original; useful only for stabilizing ablations.")
    parser.add_argument("--ssa-use-calibration-only", action="store_true", default=True)
    parser.add_argument("--ssa-use-all-target-unlabeled", dest="ssa_use_calibration_only", action="store_false")

    # CMT local invertible-ICA/augmenter knobs.
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

    # Experimental kinematic alignment scaffold.
    parser.add_argument("--kin-source-fraction", type=float, default=0.5, help="Experimental kin_ssa only: nearest source fraction by raw IMU kinematic summary.")

    args = parser.parse_args()
    run_loocv_experiment(args, pre_eval_hooks=[faithful_rr_tta_hook])


if __name__ == "__main__":
    main()
