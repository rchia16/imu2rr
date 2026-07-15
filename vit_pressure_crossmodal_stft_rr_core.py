from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
)



from utils import _filter_subjects_with_data
from dataloader import build_loocv_loaders
from config import BR_FS, SBJ_PROCESSED_DIR, M_DIR
from vit_pressure_crossmodal_profile_encoder import (
    PatientProfileEncoder,
    ProfileFiLM,
    build_profile_stats,
    estimate_rr_from_predicted_stft,
    estimate_source_profile_normalizer,
    normalize_profile_stats,
    profile_stats_diagnostics,
    profile_stats_dim as infer_profile_stats_dim,
    split_profile_stats,
)

TLX_CSV = "/projects/BLVMob/imu-rr-seated/Data/seated_tlx.csv"
DEFAULT_IMU_ISSUES_MR = [17, 26, 30]
DEFAULT_IMU_ISSUES_LEVELS = [17, 21, 26, 30]


def default_subjects(
    imu_issues_mr: Optional[List[int]] = None,
    imu_issues_levels: Optional[List[int]] = None,
) -> List[str]:
    issues_mr = DEFAULT_IMU_ISSUES_MR if imu_issues_mr is None else imu_issues_mr
    issues_levels = DEFAULT_IMU_ISSUES_LEVELS if imu_issues_levels is None else imu_issues_levels
    return _filter_subjects_with_data(
        ["S" + str(i).zfill(2) for i in range(12, 31)],
        excluded_subject_nums=issues_mr + issues_levels,
        data_dir=SBJ_PROCESSED_DIR,
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
    """Split target arrays into calibration and evaluation subsets.

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_fold_seed(base_seed: int, subject: str) -> int:
    subject_number = int(str(subject).lstrip("S"))
    return int(int(base_seed) * 1000 + subject_number)


def close_loaders(*loaders) -> None:
    for loader in loaders:
        iterator = getattr(loader, "_iterator", None)
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()
        if hasattr(loader, "_iterator"):
            loader._iterator = None


def _state_dict_cpu_clone(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _state_dict_matches(model: nn.Module, reference: Dict[str, torch.Tensor]) -> bool:
    current = model.state_dict()
    if set(current.keys()) != set(reference.keys()):
        return False
    for key, ref in reference.items():
        if not torch.equal(current[key].detach().cpu(), ref):
            return False
    return True


def _restore_model_state(model: nn.Module, reference: Dict[str, torch.Tensor], device: str) -> None:
    model.load_state_dict(reference, strict=True)
    model.to(device)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ProfileConditioner(nn.Module):
    """Generate additive Q/K/V conditioners from a low-dimensional profile vector."""

    def __init__(self, profile_dim: int, d_model: int, hidden_dim: int = 128):
        super().__init__()
        if int(profile_dim) <= 0:
            raise ValueError(f"profile_dim must be positive, got {profile_dim}")
        if int(d_model) <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if int(hidden_dim) <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        self.profile_dim = int(profile_dim)
        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.profile_dim),
            nn.Linear(self.profile_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 3 * self.d_model),
        )

    def forward(self, profile: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if profile.ndim == 1:
            profile = profile.unsqueeze(0)
        if profile.ndim != 2:
            raise ValueError(f"Expected profile with shape (B,D) or (D,), got {tuple(profile.shape)}")
        if profile.size(-1) != self.profile_dim:
            raise ValueError(
                f"Expected profile last dim {self.profile_dim}, got {profile.size(-1)}"
            )
        cq, ck, cv = self.net(profile).chunk(3, dim=-1)
        return cq, ck, cv


class ProfileCLSAQKVAdapter(nn.Module):
    """Profile-conditioned one-step CLSA-QKV residual adapter.

    This is a low-capacity fast branch: slow parameters are learned during
    source training, while per-sample fast weights are generated from the
    profile vector and updated once inside the forward pass using a token-level
    self-alignment loss. The fast weights are discarded after the forward pass.
    """

    def __init__(
        self,
        d_model: int,
        profile_dim: int,
        rank: int = 8,
        scale: float = 0.01,
        eta_max: float = 0.1,
        hidden_dim: int = 128,
        gate_init_bias: float = -2.0,
    ):
        super().__init__()
        if int(d_model) <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if int(profile_dim) <= 0:
            raise ValueError(f"profile_dim must be positive, got {profile_dim}")
        if int(rank) <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.d_model = int(d_model)
        self.profile_dim = int(profile_dim)
        self.rank = int(rank)
        self.scale = float(scale)
        self.eta_max = float(eta_max)
        self.hidden_dim = int(hidden_dim)

        self.norm = nn.LayerNorm(self.d_model)
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)

        # Keep the first implementation explicit and low-rank. hidden_dim is
        # retained in the signature for future MLP initializers, but the linear
        # map is intentionally minimal and easy to audit.
        self.profile_to_a = nn.Linear(self.profile_dim, self.d_model * self.rank)
        self.profile_to_b = nn.Linear(self.profile_dim, self.rank * self.d_model)
        self.profile_to_eta = nn.Linear(self.profile_dim, 1)
        self.profile_to_gate = nn.Linear(self.profile_dim, 1)

        # Conservative init: no-fast-update mode is an exact near-no-op at
        # initialization because B starts at zero and the gate starts mostly shut.
        nn.init.zeros_(self.profile_to_b.weight)
        nn.init.zeros_(self.profile_to_b.bias)
        nn.init.constant_(self.profile_to_gate.bias, float(gate_init_bias))

    def forward(
        self,
        h: torch.Tensor,
        profile: torch.Tensor,
        enable_fast_update: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if h.ndim != 3:
            raise ValueError(f"Expected h with shape (B,T,D), got {tuple(h.shape)}")
        if profile.ndim == 1:
            profile = profile.unsqueeze(0)
        if profile.ndim != 2:
            raise ValueError(f"Expected profile with shape (B,P) or (P,), got {tuple(profile.shape)}")
        if h.size(0) != profile.size(0):
            raise ValueError(f"Batch mismatch between h ({h.size(0)}) and profile ({profile.size(0)})")
        if h.size(-1) != self.d_model:
            raise ValueError(f"Expected h last dim {self.d_model}, got {h.size(-1)}")
        if profile.size(-1) != self.profile_dim:
            raise ValueError(f"Expected profile last dim {self.profile_dim}, got {profile.size(-1)}")

        # This forward can be called from evaluation helpers wrapped in
        # torch.no_grad(). The one-step local fast update still needs gradients
        # with respect to A_fast/B_fast, so locally re-enable autograd.
        with torch.enable_grad():
            bsz, _tokens, dim = h.shape
            x = self.norm(h)
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

            a0 = self.profile_to_a(profile).view(bsz, dim, self.rank)
            b0 = self.profile_to_b(profile).view(bsz, self.rank, dim)
            a0 = a0.requires_grad_(True)
            b0 = b0.requires_grad_(True)

            z0 = torch.bmm(torch.bmm(k, a0), b0)
            target = (v - k).detach()
            z0_ln = F.layer_norm(z0, z0.shape[-1:])
            clsa_loss = 0.5 * F.mse_loss(z0_ln, target)

            if bool(enable_fast_update):
                grad_a, grad_b = torch.autograd.grad(
                    clsa_loss,
                    [a0, b0],
                    create_graph=bool(self.training),
                    retain_graph=True,
                    allow_unused=False,
                )
                eta = F.softplus(self.profile_to_eta(profile)).clamp(max=self.eta_max).view(bsz, 1, 1)
                a1 = a0 - eta * grad_a
                b1 = b0 - eta * grad_b
                fast_update_norm = (
                    torch.linalg.vector_norm((eta * grad_a).reshape(bsz, -1), dim=1)
                    + torch.linalg.vector_norm((eta * grad_b).reshape(bsz, -1), dim=1)
                )
            else:
                eta = torch.zeros(bsz, 1, 1, device=h.device, dtype=h.dtype)
                a1, b1 = a0, b0
                fast_update_norm = torch.zeros(bsz, device=h.device, dtype=h.dtype)

            delta = torch.bmm(torch.bmm(q, a1), b1)
            gate = torch.sigmoid(self.profile_to_gate(profile)).view(bsz, 1, 1)
            out = h + float(self.scale) * gate * delta

            delta_norm = delta.detach().norm(dim=-1).reshape(-1)
            gate_flat = gate.detach().reshape(-1)
            eta_flat = eta.detach().reshape(-1)
            fast_update_norm_det = fast_update_norm.detach().reshape(-1)
            target_norm = target.detach().norm(dim=-1).reshape(-1)
            metrics = {
                "profile_clsa_loss": clsa_loss,
                "profile_clsa_gate_mean": gate_flat.mean(),
                "profile_clsa_gate_p05": torch.quantile(gate_flat, 0.05),
                "profile_clsa_gate_p95": torch.quantile(gate_flat, 0.95),
                "profile_clsa_eta_mean": eta_flat.mean(),
                "profile_clsa_delta_norm_mean": delta_norm.mean(),
                "profile_clsa_delta_norm_p95": torch.quantile(delta_norm, 0.95),
                "profile_clsa_target_norm_mean": target_norm.mean(),
                "profile_clsa_fast_update_norm_mean": fast_update_norm_det.mean(),
                "profile_clsa_fast_update_norm_p95": torch.quantile(fast_update_norm_det, 0.95),
            }
        return out, metrics


class ProfileLowRankAdapter(nn.Module):
    """Apply a profile-generated low-rank residual update to token features."""

    def __init__(self, profile_dim: int, d_model: int, rank: int = 8, scale: float = 0.05):
        super().__init__()
        if int(profile_dim) <= 0:
            raise ValueError(f"profile_dim must be positive, got {profile_dim}")
        if int(d_model) <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if int(rank) <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.profile_dim = int(profile_dim)
        self.d_model = int(d_model)
        self.rank = int(rank)
        self.scale = float(scale)
        self.norm = nn.LayerNorm(self.d_model)
        self.to_a = nn.Linear(self.profile_dim, self.d_model * self.rank)
        self.to_b = nn.Linear(self.profile_dim, self.rank * self.d_model)
        nn.init.zeros_(self.to_b.weight)
        nn.init.zeros_(self.to_b.bias)

    def forward(self, h: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        if profile.ndim == 1:
            profile = profile.unsqueeze(0)
        if profile.ndim != 2:
            raise ValueError(f"Expected profile with shape (B,D) or (D,), got {tuple(profile.shape)}")
        if h.ndim not in (2, 3):
            raise ValueError(f"Expected h with shape (B,D) or (B,T,D), got {tuple(h.shape)}")
        if h.size(0) != profile.size(0):
            raise ValueError(
                f"Batch mismatch between h ({h.size(0)}) and profile ({profile.size(0)})"
            )
        if h.size(-1) != self.d_model:
            raise ValueError(f"Expected h last dim {self.d_model}, got {h.size(-1)}")
        if profile.size(-1) != self.profile_dim:
            raise ValueError(
                f"Expected profile last dim {self.profile_dim}, got {profile.size(-1)}"
            )

        bsz = profile.size(0)
        a = self.to_a(profile).view(bsz, self.d_model, self.rank)
        b = self.to_b(profile).view(bsz, self.rank, self.d_model)
        h_norm = self.norm(h)
        if h_norm.ndim == 2:
            delta = torch.bmm(torch.bmm(h_norm.unsqueeze(1), a), b).squeeze(1)
        else:
            delta = torch.bmm(torch.bmm(h_norm, a), b)
        return h + self.scale * delta


class ReconstructionDecoder(nn.Module):
    """Configurable reconstruction decoder for IMU-token -> pressure-STFT-token prediction.

    The legacy GRU path stays in TinyIMU2PressureViT as `dec_rnn` for checkpoint
    compatibility. This wrapper handles only the non-GRU ablation modes.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 1,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        mode: str = "self_attn",
        max_query_len: int = 512,
        norm_first: bool = True,
    ):
        super().__init__()
        mode = str(mode).lower().strip()
        if mode not in {"self_attn", "cross_attn"}:
            raise ValueError(f"ReconstructionDecoder only handles self_attn/cross_attn, got {mode!r}")
        self.mode = mode
        self.d_model = int(d_model)
        self.max_query_len = int(max_query_len)

        if mode == "self_attn":
            layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=norm_first,
            )
            self.block = TransformerEncoder(layer, num_layers=num_layers)
        else:
            layer = TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=norm_first,
            )
            self.block = TransformerDecoder(layer, num_layers=num_layers)
            self.query_tokens = nn.Parameter(torch.randn(1, self.max_query_len, d_model) * 0.02)
            self.query_pos = PositionalEncoding(d_model, max_len=max_query_len)

    def _make_queries(self, batch_size: int, n_tokens: int, device, dtype) -> torch.Tensor:
        if n_tokens <= self.max_query_len:
            q = self.query_tokens[:, :n_tokens, :]
        else:
            q = F.interpolate(
                self.query_tokens.transpose(1, 2),
                size=n_tokens,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        q = q.to(device=device, dtype=dtype).expand(batch_size, -1, -1)
        return self.query_pos(q)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.mode == "self_attn":
            return self.block(h)
        if self.mode == "cross_attn":
            b, t, _ = h.shape
            q = self._make_queries(batch_size=b, n_tokens=t, device=h.device, dtype=h.dtype)
            return self.block(tgt=q, memory=h)
        raise RuntimeError(f"Unhandled decoder mode: {self.mode}")


class TokenTCNBlock(nn.Module):
    """Residual depthwise-separable temporal convolution over ViT tokens."""

    def __init__(self, d_model: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        if int(d_model) <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if int(kernel_size) <= 0 or int(kernel_size) % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        if int(dilation) <= 0:
            raise ValueError(f"dilation must be positive, got {dilation}")
        self.d_model = int(d_model)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        padding = (self.kernel_size - 1) * self.dilation // 2
        self.depthwise = nn.Conv1d(
            self.d_model,
            self.d_model,
            kernel_size=self.kernel_size,
            padding=padding,
            dilation=self.dilation,
            groups=self.d_model,
        )
        self.pointwise = nn.Conv1d(self.d_model, self.d_model, kernel_size=1)
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B,D,T), got {tuple(x.shape)}")
        if x.size(1) != self.d_model:
            raise ValueError(f"Expected channel dim {self.d_model}, got {x.size(1)}")
        y = self.depthwise(x)
        y = F.gelu(y)
        y = self.pointwise(y)
        y = self.dropout(y)
        y = x + y
        return self.norm(y.transpose(1, 2)).transpose(1, 2)


class TokenTCNRRHead(nn.Module):
    """RR head that models local token patterns before mean/std pooling."""

    def __init__(
        self,
        d_model: int,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if int(d_model) <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if int(num_layers) <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.kernel_size = int(kernel_size)
        self.blocks = nn.ModuleList(
            [
                TokenTCNBlock(
                    self.d_model,
                    kernel_size=self.kernel_size,
                    dilation=2 ** i,
                    dropout=float(dropout),
                )
                for i in range(self.num_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(2 * self.d_model),
            nn.Linear(2 * self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, 1),
        )

    def forward(self, h_vit: torch.Tensor) -> torch.Tensor:
        if h_vit.ndim != 3:
            raise ValueError(f"Expected h_vit with shape (B,T,D), got {tuple(h_vit.shape)}")
        if h_vit.size(-1) != self.d_model:
            raise ValueError(f"Expected token dim {self.d_model}, got {h_vit.size(-1)}")
        x = h_vit.transpose(1, 2).contiguous()
        for block in self.blocks:
            x = block(x)
        mean = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=False)
        pooled = torch.cat([mean, std], dim=-1)
        return self.head(pooled).squeeze(-1)


class DepthwiseTemporalTokenMixer(nn.Module):
    """Lightweight local temporal mixer for projected IMU STFT tokens."""

    def __init__(self, d_model: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        if int(kernel_size) < 1 or int(kernel_size) % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        padding = int(kernel_size) // 2
        self.norm = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=int(kernel_size),
            padding=padding,
            groups=d_model,
        )
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm(x).transpose(1, 2)
        y = self.depthwise(y)
        y = F.gelu(self.pointwise(y)).transpose(1, 2)
        return residual + self.dropout(y)


class RawTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        if int(kernel_size) < 1:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        padding = ((int(kernel_size) - 1) * int(dilation)) // 2
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=int(kernel_size),
            dilation=int(dilation),
            padding=padding,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if y.size(-1) != x.size(-1):
            y = F.interpolate(y, size=x.size(-1), mode="linear", align_corners=False)
        y = self.dropout(F.relu(self.norm(y)))
        return x + y


class ResidualIMUTemporalMixer(nn.Module):
    """Small residual temporal mixer on raw IMU before STFT tokenisation."""

    def __init__(
        self,
        channels: int,
        hidden: int = 32,
        layers: int = 2,
        alpha: float = 0.05,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channels = int(channels)
        self.hidden = int(hidden)
        self.layers_n = int(layers)
        self.alpha = float(alpha)
        blocks: List[nn.Module] = [
            nn.Conv1d(self.channels, self.hidden, kernel_size=1),
            nn.GELU(),
        ]
        for i in range(max(1, self.layers_n)):
            dilation = i + 1
            blocks.extend(
                [
                    nn.GroupNorm(1, self.hidden),
                    nn.Conv1d(
                        self.hidden,
                        self.hidden,
                        kernel_size=5,
                        padding=2 * dilation,
                        dilation=dilation,
                        groups=max(1, self.hidden),
                    ),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Conv1d(self.hidden, self.hidden, kernel_size=1),
                    nn.GELU(),
                ]
            )
        blocks.append(nn.Conv1d(self.hidden, self.channels, kernel_size=1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B,T,C), got {tuple(x.shape)}")
        if x.size(-1) != self.channels:
            raise ValueError(f"Expected {self.channels} IMU channels, got {x.size(-1)}")
        y = self.net(x.transpose(1, 2)).transpose(1, 2)
        if y.size(1) != x.size(1):
            y = F.interpolate(
                y.transpose(1, 2),
                size=x.size(1),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return x + float(self.alpha) * y


class TinyIMU2PressureViT(nn.Module):
    def __init__(
        self,
        input_channels: int = 6,
        d_model: int = 128,
        pred_len: int = 360,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        n_fft: int = 256,
        hop_length: int = 64,
        win_length: int = 256,
        use_profile_film: bool = False,
        use_profile_qkv: bool = False,
        profile_conditioning: str = "none",
        profile_dim: int = 32,
        profile_stats_dim: int = 0,
        profile_hidden_dim: int = 128,
        profile_film_scale: float = 0.1,
        profile_film_placement: str = "token_pooled",
        profile_film_residual_alpha: float = 0.1,
        profile_qkv_scale: float = 0.1,
        profile_qkv_layers: str = "last1",
        profile_qkv_residual: bool = False,
        profile_qkv_mode: str = "static",
        profile_clsa_rank: int = 8,
        profile_clsa_scale: float = 0.01,
        profile_clsa_eta_max: float = 0.1,
        profile_clsa_gate_init_bias: float = -2.0,
        profile_clsa_enable_fast_update: bool = True,
        profile_clsa_loss_weight: float = 0.0,
        decoder_mode: str = "gru",
        decoder_layers: int = 1,
        rr_from: str = "encoder",
        imu_token_mixer: str = "dwconv",
        use_tcn_token_mixer: bool = False,
        tcn_mixer_alpha: float = 0.05,
        tcn_mixer_hidden: int = 32,
        tcn_mixer_layers: int = 2,
        rr_head_type: str = "mlp",
        rr_tcn_layers: int = 2,
        rr_tcn_kernel_size: int = 3,
        rr_tcn_dropout: Optional[float] = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.d_model = d_model
        self.pred_len = pred_len
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        requested_profile_mode = str(profile_conditioning).lower().strip()
        if requested_profile_mode in {"", "auto"}:
            requested_profile_mode = "none"
        if requested_profile_mode == "none":
            if bool(use_profile_film) and bool(use_profile_qkv):
                requested_profile_mode = "film_qkv"
            elif bool(use_profile_qkv):
                requested_profile_mode = "qkv"
            elif bool(use_profile_film):
                requested_profile_mode = "film"
        if requested_profile_mode in {"qkv_film", "shared_film_qkv", "shared_profile_qkv"}:
            requested_profile_mode = "film_qkv"
        if requested_profile_mode not in {"none", "film", "qkv", "film_qkv"}:
            raise ValueError(
                f"profile_conditioning must be one of none/film/qkv/film_qkv, got {profile_conditioning!r}"
            )
        self.profile_conditioning = requested_profile_mode
        self.use_profile_conditioning = self.profile_conditioning != "none"
        self.use_profile_film = self.profile_conditioning in {"film", "film_qkv"}
        self.use_profile_qkv = self.profile_conditioning in {"qkv", "film_qkv"}
        self.profile_dim = int(profile_dim)
        self.profile_stats_dim = int(profile_stats_dim)
        self.profile_hidden_dim = int(profile_hidden_dim)
        self.profile_film_scale = float(profile_film_scale)
        self.profile_film_placement = str(profile_film_placement).lower().strip()
        if self.profile_film_placement in {"current", "token+pooled", "token_and_pooled"}:
            self.profile_film_placement = "token_pooled"
        if self.profile_film_placement not in {
            "token_pooled",
            "pooled_only",
            "late_token_only",
            "residual",
        }:
            raise ValueError(
                "profile_film_placement must be one of "
                "token_pooled/pooled_only/late_token_only/residual, "
                f"got {profile_film_placement!r}"
            )
        self.profile_film_residual_alpha = float(profile_film_residual_alpha)
        self.profile_qkv_scale = float(profile_qkv_scale)
        self.profile_qkv_layers = str(profile_qkv_layers)
        self.profile_qkv_residual = bool(profile_qkv_residual)
        self.profile_qkv_mode = str(profile_qkv_mode).lower().strip()
        if self.profile_qkv_mode not in {"static", "clsa"}:
            raise ValueError(f"profile_qkv_mode must be one of static/clsa, got {profile_qkv_mode!r}")
        self.profile_clsa_rank = int(profile_clsa_rank)
        self.profile_clsa_scale = float(profile_clsa_scale)
        self.profile_clsa_eta_max = float(profile_clsa_eta_max)
        self.profile_clsa_gate_init_bias = float(profile_clsa_gate_init_bias)
        self.profile_clsa_enable_fast_update = bool(profile_clsa_enable_fast_update)
        self.profile_clsa_loss_weight = float(profile_clsa_loss_weight)
        self._last_profile_clsa_loss_terms: List[torch.Tensor] = []
        self.decoder_mode = str(decoder_mode).lower().strip()
        if self.decoder_mode not in {"none", "gru", "self_attn", "cross_attn"}:
            raise ValueError(
                f"decoder_mode must be one of none/gru/self_attn/cross_attn, got {decoder_mode!r}"
            )
        self.decoder_layers = int(decoder_layers)
        self.rr_from = str(rr_from).lower().strip()
        if self.rr_from not in {"encoder", "decoder", "both"}:
            raise ValueError(f"rr_from must be one of encoder/decoder/both, got {rr_from!r}")
        self.rr_head_type = str(rr_head_type).lower().strip()
        print("rr_head_type: ", self.rr_head_type)
        if self.rr_head_type not in {"mlp", "token_tcn"}:
            raise ValueError(f"rr_head_type must be one of mlp/token_tcn, got {rr_head_type!r}")
        if self.rr_head_type == "token_tcn" and self.rr_from == "both":
            raise ValueError("rr_from='both' is not supported with rr_head_type='token_tcn'")
        self.rr_tcn_layers = int(rr_tcn_layers)
        self.rr_tcn_kernel_size = int(rr_tcn_kernel_size)
        self.rr_tcn_dropout = float(dropout if rr_tcn_dropout is None else rr_tcn_dropout)
        self._last_profile_attn_metrics: Dict[str, float] = {}
        self.use_tcn_token_mixer = bool(use_tcn_token_mixer)
        self.tcn_mixer_alpha = float(tcn_mixer_alpha)
        self.tcn_mixer_hidden = int(tcn_mixer_hidden)
        self.tcn_mixer_layers = int(tcn_mixer_layers)
        self.raw_tcn_mixer: Optional[ResidualIMUTemporalMixer]
        if self.use_tcn_token_mixer:
            self.raw_tcn_mixer = ResidualIMUTemporalMixer(
                channels=input_channels,
                hidden=self.tcn_mixer_hidden,
                layers=self.tcn_mixer_layers,
                alpha=self.tcn_mixer_alpha,
                dropout=dropout,
            )
        else:
            self.raw_tcn_mixer = None

        self.spec_proj: Optional[nn.Linear] = None
        self.pos = PositionalEncoding(d_model)
        enc_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = TransformerEncoder(enc_layer, num_layers=num_layers)
        # Keep the legacy GRU module/name so decoder_mode="gru" exactly preserves
        # the previous reconstruction path and old checkpoint keys.
        self.dec_rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.recon_decoder: Optional[ReconstructionDecoder]
        if self.decoder_mode in {"self_attn", "cross_attn"}:
            self.recon_decoder = ReconstructionDecoder(
                d_model=d_model,
                nhead=nhead,
                num_layers=self.decoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                mode=self.decoder_mode,
            )
        else:
            self.recon_decoder = None

        pressure_n_fft = min(self.n_fft, self.pred_len)
        self.pressure_freq_bins = pressure_n_fft // 2 + 1
        self.pressure_mag_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.pressure_freq_bins),
        )

        self.imu_token_mixer_mode = str(imu_token_mixer).lower().strip()
        if self.imu_token_mixer_mode not in {"none", "dwconv"}:
            raise ValueError(
                f"imu_token_mixer must be one of none/dwconv, got {imu_token_mixer!r}"
            )
        self.imu_token_mixer: Optional[DepthwiseTemporalTokenMixer]
        if self.imu_token_mixer_mode == "dwconv":
            self.imu_token_mixer = DepthwiseTemporalTokenMixer(d_model, dropout=dropout)
        else:
            self.imu_token_mixer = None

        if self.rr_head_type == "token_tcn":
            self.rr_head = TokenTCNRRHead(
                d_model=d_model,
                num_layers=self.rr_tcn_layers,
                kernel_size=self.rr_tcn_kernel_size,
                dropout=self.rr_tcn_dropout,
            )
        else:
            self.rr_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
        self.rr_fusion_head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.pressure_proj: Optional[nn.Linear] = None
        self.pressure_pos = PositionalEncoding(d_model)
        pressure_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.pressure_encoder = TransformerEncoder(
            pressure_layer, num_layers=max(1, num_layers // 2)
        )

        if self.use_profile_conditioning:
            if self.profile_stats_dim <= 0:
                raise ValueError(
                    "profile_stats_dim must be positive when profile conditioning is enabled"
                )
            self.profile_encoder: Optional[PatientProfileEncoder] = PatientProfileEncoder(
                in_dim=self.profile_stats_dim,
                profile_dim=self.profile_dim,
                hidden_dim=self.profile_hidden_dim,
            )
            self.profile_film_tokens: Optional[ProfileFiLM] = ProfileFiLM(
                self.profile_dim, d_model, self.profile_film_scale
            )
            self.profile_film_pooled: Optional[ProfileFiLM] = ProfileFiLM(
                self.profile_dim, d_model, self.profile_film_scale
            )
            self.profile_qkv_layer_indices = self._resolve_profile_qkv_layer_indices(num_layers)
            self.profile_qkv_conditioners = nn.ModuleDict(
                {
                    str(layer_idx): ProfileConditioner(
                        self.profile_dim,
                        d_model,
                        hidden_dim=self.profile_hidden_dim,
                    )
                    for layer_idx in self.profile_qkv_layer_indices
                }
            )
            if self.use_profile_qkv and self.profile_qkv_mode == "clsa":
                self.profile_clsa_qkv_adapters = nn.ModuleDict(
                    {
                        str(layer_idx): ProfileCLSAQKVAdapter(
                            d_model=d_model,
                            profile_dim=self.profile_dim,
                            rank=self.profile_clsa_rank,
                            scale=self.profile_clsa_scale,
                            eta_max=self.profile_clsa_eta_max,
                            hidden_dim=self.profile_hidden_dim,
                            gate_init_bias=self.profile_clsa_gate_init_bias,
                        )
                        for layer_idx in self.profile_qkv_layer_indices
                    }
                )
            else:
                self.profile_clsa_qkv_adapters = nn.ModuleDict()
        else:
            self.profile_encoder = None
            self.profile_film_tokens = None
            self.profile_film_pooled = None
            self.profile_qkv_layer_indices = []
            self.profile_qkv_conditioners = nn.ModuleDict()
            self.profile_clsa_qkv_adapters = nn.ModuleDict()

    def _resolve_profile_qkv_layer_indices(self, num_layers: int) -> List[int]:
        mode = str(self.profile_qkv_layers).lower().strip()
        if mode == "none":
            return []
        if mode == "last1":
            return [max(0, int(num_layers) - 1)]
        if mode == "last2":
            start = max(0, int(num_layers) - 2)
            return list(range(start, int(num_layers)))
        if mode == "all":
            return list(range(int(num_layers)))
        raise ValueError(f"Unsupported profile_qkv_layers={self.profile_qkv_layers!r}")

    def _expand_profile_vector(self, profile_vector: torch.Tensor, batch_size: int) -> torch.Tensor:
        if profile_vector.ndim == 1:
            profile_vector = profile_vector.unsqueeze(0)
        if profile_vector.size(0) == 1 and batch_size > 1:
            profile_vector = profile_vector.expand(batch_size, -1)
        if profile_vector.size(0) != batch_size:
            raise ValueError(
                f"profile_vector batch {profile_vector.size(0)} must match input batch {batch_size}"
            )
        return profile_vector

    @staticmethod
    def _mean_temporal_attention_distance(attn_weights: torch.Tensor) -> torch.Tensor:
        if attn_weights.ndim != 4:
            raise ValueError(f"Expected attn_weights (B,H,T,T), got {tuple(attn_weights.shape)}")
        t = int(attn_weights.size(-1))
        if t <= 1:
            return attn_weights.new_tensor(0.0)
        idx = torch.arange(t, device=attn_weights.device, dtype=attn_weights.dtype)
        dist = (idx[:, None] - idx[None, :]).abs()
        attn = attn_weights.mean(dim=1)
        return (attn * dist.unsqueeze(0)).sum(dim=(-1, -2)).mean()

    def _forward_encoder_layer(
        self,
        layer: TransformerEncoderLayer,
        src: torch.Tensor,
        profile_vector: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_distance = None
        conditioner = None
        if (
            self.use_profile_qkv
            and self.profile_qkv_mode != "clsa"
            and profile_vector is not None
            and layer_idx is not None
            and str(layer_idx) in self.profile_qkv_conditioners
        ):
            conditioner = self.profile_qkv_conditioners[str(layer_idx)]

        def _sa_block(x: torch.Tensor) -> torch.Tensor:
            q = x
            k = x
            v = x
            need_weights = conditioner is not None
            if conditioner is not None:
                cq, ck, cv = conditioner(profile_vector)
                scale = float(self.profile_qkv_scale)
                if bool(self.profile_qkv_residual):
                    base_attn, _base_weights = layer.self_attn(
                        x,
                        x,
                        x,
                        attn_mask=None,
                        key_padding_mask=None,
                        need_weights=False,
                        average_attn_weights=False,
                    )
                    q = q + cq.unsqueeze(1)
                    k = k + ck.unsqueeze(1)
                    v = v + cv.unsqueeze(1)
                else:
                    base_attn = None
                    q = q + scale * cq.unsqueeze(1)
                    k = k + scale * ck.unsqueeze(1)
                    v = v + scale * cv.unsqueeze(1)
            x_attn, attn_weights = layer.self_attn(
                q,
                k,
                v,
                attn_mask=None,
                key_padding_mask=None,
                need_weights=need_weights,
                average_attn_weights=False,
            )
            if attn_weights is not None:
                nonlocal attn_distance
                attn_distance = self._mean_temporal_attention_distance(attn_weights.detach())
            if conditioner is not None and bool(self.profile_qkv_residual):
                x_attn = base_attn + float(self.profile_qkv_scale) * (x_attn - base_attn)
            return layer.dropout1(x_attn)

        def _ff_block(x: torch.Tensor) -> torch.Tensor:
            x = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            return layer.dropout2(x)

        x = src
        if layer.norm_first:
            x = x + _sa_block(layer.norm1(x))
            x = x + _ff_block(layer.norm2(x))
        else:
            x = layer.norm1(x + _sa_block(x))
            x = layer.norm2(x + _ff_block(x))
        return x, attn_distance

    def _imu_stft_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected IMU tensor (B,T,C), got {tuple(x.shape)}")
        if x.size(1) < x.size(2):
            x = x.transpose(1, 2)

        b, _, c = x.shape
        if c != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, got {c}")
        if self.raw_tcn_mixer is not None:
            x = self.raw_tcn_mixer(x)

        x = x.transpose(1, 2)
        window = torch.hann_window(self.win_length, device=x.device)
        mags = []
        for ch in range(c):
            spec = torch.stft(
                x[:, ch],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                return_complex=True,
            )
            mags.append(spec.abs().unsqueeze(1))
        mag = torch.cat(mags, dim=1)
        b, c, f, tf = mag.shape
        feat = mag.reshape(b, c * f, tf).transpose(1, 2)

        if self.spec_proj is None:
            self.spec_proj = nn.Linear(c * f, self.d_model).to(x.device)
        return feat

    def _pressure_stft_tokens(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 3:
            y = y.squeeze(-1)
        if y.ndim != 2:
            raise ValueError(f"Expected pressure tensor (B,T) or (B,T,1), got {tuple(y.shape)}")

        t = y.size(1)
        n_fft = min(self.n_fft, t)
        win_length = min(self.win_length, t)
        hop_length = min(self.hop_length, max(1, win_length // 4))
        window = torch.hann_window(win_length, device=y.device)

        spec = torch.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        ).abs()
        return torch.log1p(spec).transpose(1, 2)

    def encode_pressure(self, y: torch.Tensor, target_tokens: Optional[int] = None) -> torch.Tensor:
        tokens = self._pressure_stft_tokens(y)
        if self.pressure_proj is None:
            self.pressure_proj = nn.Linear(tokens.size(-1), self.d_model).to(tokens.device)
        h = self.pressure_proj(tokens)
        h = self.pressure_encoder(self.pressure_pos(h))

        if target_tokens is not None and h.size(1) != target_tokens:
            h = F.interpolate(
                h.transpose(1, 2),
                size=target_tokens,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return h

    def pressure_stft_target(self, y: torch.Tensor, target_tokens: Optional[int] = None) -> torch.Tensor:
        target = self._pressure_stft_tokens(y)
        if target_tokens is not None and target.size(1) != target_tokens:
            target = F.interpolate(
                target.transpose(1, 2),
                size=target_tokens,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        if target.size(-1) != self.pressure_freq_bins:
            target = F.interpolate(
                target.transpose(1, 2),
                size=self.pressure_freq_bins,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return target

    def encode(
        self,
        x: torch.Tensor,
        profile_vector: Optional[torch.Tensor] = None,
        conditioning_mode: Optional[str] = None,
    ) -> torch.Tensor:
        tokens = self._imu_stft_tokens(x)
        h = self.spec_proj(tokens)
        if self.imu_token_mixer is not None:
            h = self.imu_token_mixer(h)
        h = self.pos(h)
        self._last_profile_attn_metrics = {}
        self._last_profile_clsa_loss_terms = []
        mode = str(conditioning_mode or self.profile_conditioning).lower().strip()
        if mode in {"qkv", "film_qkv"} and profile_vector is not None:
            profile_vector = self._expand_profile_vector(profile_vector, h.size(0))
            for layer_idx, layer in enumerate(self.encoder.layers):
                h, attn_distance = self._forward_encoder_layer(
                    layer,
                    h,
                    profile_vector=profile_vector,
                    layer_idx=layer_idx,
                )
                if (
                    self.use_profile_qkv
                    and self.profile_qkv_mode == "clsa"
                    and str(layer_idx) in self.profile_clsa_qkv_adapters
                ):
                    h, clsa_metrics = self.profile_clsa_qkv_adapters[str(layer_idx)](
                        h,
                        profile_vector,
                        enable_fast_update=bool(self.profile_clsa_enable_fast_update),
                    )
                    for name, value in clsa_metrics.items():
                        if name == "profile_clsa_loss":
                            self._last_profile_clsa_loss_terms.append(value)
                        self._last_profile_attn_metrics[f"{name}_layer_{layer_idx}"] = float(
                            value.detach().float().cpu()
                        )
                if attn_distance is not None:
                    self._last_profile_attn_metrics[f"attn_distance_layer_{layer_idx}"] = float(
                        attn_distance.detach().cpu()
                    )
            if self.encoder.norm is not None:
                h = self.encoder.norm(h)
        else:
            h = self.encoder(h)
        return h

    def last_profile_attention_metrics(self) -> Dict[str, float]:
        return dict(self._last_profile_attn_metrics)

    def last_profile_clsa_loss(self) -> Optional[torch.Tensor]:
        if not self._last_profile_clsa_loss_terms:
            return None
        return torch.stack([x.reshape(()) for x in self._last_profile_clsa_loss_terms]).mean()

    def decode_reconstruction(self, h: torch.Tensor) -> torch.Tensor:
        """Decode/refine encoder tokens before pressure-STFT reconstruction."""
        if self.decoder_mode == "none":
            return h
        if self.decoder_mode == "gru":
            h_dec, _ = self.dec_rnn(h)
            return h_dec
        if self.recon_decoder is None:
            raise RuntimeError(f"recon_decoder is missing for decoder_mode={self.decoder_mode!r}")
        return self.recon_decoder(h)

    def predict_rr_from_features(self, h: torch.Tensor, h_dec: torch.Tensor) -> torch.Tensor:
        """RR readout source for decoder ablations.

        encoder: legacy/default, unaffected by reconstruction decoder.
        decoder: pooled decoded reconstruction tokens.
        both:    pooled encoder and decoder tokens concatenated.
        """
        if self.rr_from == "encoder":
            if self.rr_head_type == "token_tcn":
                return self.rr_head(h).view(-1)
            return self.rr_head(h.mean(dim=1)).squeeze(-1)
        if self.rr_from == "decoder":
            if self.rr_head_type == "token_tcn":
                return self.rr_head(h_dec).view(-1)
            return self.rr_head(h_dec.mean(dim=1)).squeeze(-1)
        if self.rr_from == "both":
            if self.rr_head_type == "token_tcn":
                raise RuntimeError("rr_from='both' is not supported with rr_head_type='token_tcn'")
            pooled = torch.cat([h.mean(dim=1), h_dec.mean(dim=1)], dim=-1)
            return self.rr_fusion_head(pooled).squeeze(-1)
        raise RuntimeError(f"Unhandled rr_from={self.rr_from!r}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        h_dec = self.decode_reconstruction(h)
        pressure_logmag = F.softplus(self.pressure_mag_head(h_dec))
        rr = self.predict_rr_from_features(h, h_dec)
        return pressure_logmag, rr, h

    def forward_profile_conditioned(
        self,
        x: torch.Tensor,
        profile_stats: Optional[torch.Tensor] = None,
        profile_vector: Optional[torch.Tensor] = None,
        conditioning_mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_profile_conditioning:
            raise RuntimeError(
                "forward_profile_conditioned requires profile conditioning at model construction."
            )
        if self.profile_encoder is None:
            raise RuntimeError("Profile-conditioning modules are not initialized.")

        if profile_vector is None:
            if profile_stats is None:
                raise ValueError("profile_stats or profile_vector required")
            if profile_stats.ndim == 1:
                profile_stats = profile_stats.unsqueeze(0)
            if profile_stats.size(0) == 1 and x.size(0) > 1:
                profile_stats = profile_stats.expand(x.size(0), -1)
            profile_vector = self.profile_encoder(profile_stats)
        profile_vector = self._expand_profile_vector(profile_vector, x.size(0))
        mode = str(conditioning_mode or self.profile_conditioning).lower().strip()
        if mode in {"qkv_film", "shared_film_qkv", "shared_profile_qkv"}:
            mode = "film_qkv"
        if mode not in {"film", "qkv", "film_qkv"}:
            raise ValueError(f"Unsupported conditioning_mode={conditioning_mode!r}")

        if mode == "qkv":
            h_cond = self.encode(x, profile_vector=profile_vector, conditioning_mode=mode)
            h_dec = self.decode_reconstruction(h_cond)
        else:
            if self.profile_film_tokens is None or self.profile_film_pooled is None:
                raise RuntimeError("Profile FiLM modules are not initialized.")
            h = self.encode(
                x,
                profile_vector=profile_vector if mode == "film_qkv" else None,
                conditioning_mode="film_qkv" if mode == "film_qkv" else None,
            )
            placement = str(self.profile_film_placement)
            alpha = float(self.profile_film_residual_alpha)
            if placement == "pooled_only":
                h_cond = h
                h_dec = self.decode_reconstruction(h_cond)
            elif placement == "late_token_only":
                h_pre_dec = self.decode_reconstruction(h)
                h_dec = self.profile_film_tokens(h_pre_dec, profile_vector)
                h_cond = h_dec
            elif placement == "residual":
                h_film = self.profile_film_tokens(h, profile_vector)
                h_cond = h + alpha * (h_film - h)
                h_dec = self.decode_reconstruction(h_cond)
            else:
                h_cond = self.profile_film_tokens(h, profile_vector)
                h_dec = self.decode_reconstruction(h_cond)
        pressure_logmag = F.softplus(self.pressure_mag_head(h_dec))

        if mode in {"film", "film_qkv"} and self.rr_from == "encoder" and self.rr_head_type == "mlp":
            pooled = h_cond.mean(dim=1)
            if self.profile_film_placement in {"token_pooled", "pooled_only"}:
                pooled = self.profile_film_pooled(pooled, profile_vector)
            elif self.profile_film_placement == "residual":
                pooled_film = self.profile_film_pooled(pooled, profile_vector)
                pooled = pooled + float(self.profile_film_residual_alpha) * (pooled_film - pooled)
            rr = self.rr_head(pooled).squeeze(-1)
        else:
            rr = self.predict_rr_from_features(h_cond, h_dec)
        return pressure_logmag, rr, h_cond, profile_vector



def rr_targets_from_batch(pressure: torch.Tensor, br: Optional[torch.Tensor]) -> torch.Tensor:
    if br is not None and isinstance(br, torch.Tensor) and br.numel() >= pressure.size(0):
        return br.view(-1)[: pressure.size(0)].float()

    if pressure.ndim == 3:
        pressure = pressure.squeeze(-1)
    t = pressure.size(1)
    n_fft = min(256, t)
    win_length = min(256, t)
    hop_length = min(64, max(1, win_length // 4))
    window = torch.hann_window(win_length, device=pressure.device)
    spec = torch.stft(
        pressure,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    ).abs().mean(dim=-1)
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / BR_FS).to(pressure.device)
    mask = (freqs >= 0.05) & (freqs <= 0.75)
    if not mask.any():
        return pressure.new_zeros(pressure.size(0))
    local_idx = spec[:, mask].argmax(dim=1)
    return freqs[mask][local_idx] * 60.0


def pressure_stft_recon_loss(
    model: TinyIMU2PressureViT,
    pred_logmag: torch.Tensor,
    pressure: torch.Tensor,
    rr_pred: Optional[torch.Tensor],
    br: Optional[torch.Tensor],
    lambda_stft: float,
    lambda_rr: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    target_logmag = model.pressure_stft_target(pressure, target_tokens=pred_logmag.size(1))
    l_stft = F.l1_loss(pred_logmag, target_logmag)

    l_rr = pred_logmag.new_tensor(0.0)
    if rr_pred is not None and lambda_rr > 0:
        rr_true = rr_targets_from_batch(pressure, br)
        l_rr = F.smooth_l1_loss(rr_pred.view(-1), rr_true.view(-1))

    loss = lambda_stft * l_stft + lambda_rr * l_rr
    return loss, {"stft": float(l_stft.detach().cpu()), "rr": float(l_rr.detach().cpu())}


def augment_imu(x: torch.Tensor, noise_std: float = 0.03, gain_std: float = 0.10, shift_max: int = 24) -> torch.Tensor:
    y = x
    gain = torch.randn(y.size(0), 1, y.size(2), device=y.device, dtype=y.dtype) * gain_std + 1.0
    y = y * gain
    if shift_max > 0:
        shifts = torch.randint(-shift_max, shift_max + 1, (y.size(0),), device=y.device)
        out = []
        for i, s in enumerate(shifts.tolist()):
            out.append(torch.roll(y[i], shifts=s, dims=0))
        y = torch.stack(out, dim=0)
    sd = y.std(dim=1, keepdim=True).clamp_min(1e-6)
    y = y + torch.randn_like(y) * sd * noise_std
    return y


def token_contrastive_loss(h1: torch.Tensor, h2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    z1 = F.normalize(h1.reshape(-1, h1.size(-1)), dim=-1)
    z2 = F.normalize(h2.reshape(-1, h2.size(-1)), dim=-1)
    logits = (z1 @ z2.T) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def unpack_batch(
    batch: Iterable[torch.Tensor],
    device: str,
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


@dataclass
class EpochMetrics:
    loss: float
    stft: float
    rr: float
    contrast: float
    rr_stft_consistency: float
    profile_prior: float


def contrast_weight_for_epoch(args, epoch: int) -> float:
    warmup_epochs = max(0, int(args.contrast_warmup_epochs))
    ramp_end_epoch = max(warmup_epochs + 1, int(args.contrast_ramp_end_epoch))
    min_weight = float(args.lambda_contrast_min)
    max_weight = float(args.lambda_contrast)

    if epoch <= warmup_epochs or max_weight <= 0:
        return 0.0
    if epoch >= ramp_end_epoch:
        return max_weight
    progress = (epoch - warmup_epochs - 1) / max(1, ramp_end_epoch - warmup_epochs - 1)
    return min_weight + progress * (max_weight - min_weight)


def _set_model_profile_normalizer(
    model: nn.Module,
    source_profile_mean: Optional[torch.Tensor],
    source_profile_std: Optional[torch.Tensor],
) -> None:
    model.source_profile_mean = None if source_profile_mean is None else source_profile_mean.detach().clone()
    model.source_profile_std = None if source_profile_std is None else source_profile_std.detach().clone()


def _get_model_profile_normalizer(
    model: nn.Module,
    device: str | torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    mean = getattr(model, "source_profile_mean", None)
    std = getattr(model, "source_profile_std", None)
    if mean is None or std is None:
        return None, None
    return mean.to(device), std.to(device)


def _forward_with_optional_profile(
    model: nn.Module,
    imu: torch.Tensor,
    device: str | torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    pred_logmag, rr_pred, hidden = model(imu)
    if not bool(getattr(model, "use_profile_conditioning", getattr(model, "use_profile_film", False))):
        return pred_logmag, rr_pred, hidden, None, None

    z = pooled_features(hidden)
    rr_stft, stft_conf = estimate_rr_from_predicted_stft(
        pred_logmag,
        br_fs=float(BR_FS),
        return_confidence=True,
    )
    profile_stats = build_profile_stats(
        z.detach(),
        rr_pred.detach().reshape(-1),
        rr_stft.detach().reshape(-1),
        stft_conf.detach().reshape(-1),
    )
    source_profile_mean, source_profile_std = _get_model_profile_normalizer(model, device)
    if source_profile_mean is not None and source_profile_std is not None:
        profile_stats = normalize_profile_stats(profile_stats, source_profile_mean, source_profile_std)
    profile_batch = profile_stats.unsqueeze(0).expand(imu.size(0), -1)
    pred_logmag, rr_pred, hidden, profile_vector = model.forward_profile_conditioned(
        imu,
        profile_stats=profile_batch,
    )
    return pred_logmag, rr_pred, hidden, profile_stats, profile_vector


def train_one_epoch(model: nn.Module, loader, optimizer, device: str, args, lambda_contrast: float) -> EpochMetrics:
    model.train()
    totals = {
        "loss": [],
        "stft": [],
        "rr": [],
        "contrast": [],
        "rr_stft_consistency": [],
        "profile_prior": [],
    }
    for batch in loader:
        imu, pressure, _, br, _ = unpack_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        pred_logmag, rr_pred, _hidden, _profile_stats, profile_vector = _forward_with_optional_profile(
            model,
            imu,
            device,
        )
        loss, parts = pressure_stft_recon_loss(
            model,
            pred_logmag,
            pressure,
            rr_pred,
            br,
            lambda_stft=args.lambda_stft,
            lambda_rr=args.lambda_rr,
        )

        lc = pred_logmag.new_tensor(0.0)
        if lambda_contrast > 0:
            h_imu = model.encode(augment_imu(imu, shift_max=args.shift_max))
            h_pressure = model.encode_pressure(pressure, target_tokens=h_imu.size(1))
            lc = token_contrastive_loss(h_imu, h_pressure, temperature=args.temperature)
            loss = loss + lambda_contrast * lc

        l_rrspec = pred_logmag.new_tensor(0.0)
        if bool(getattr(model, "use_profile_conditioning", getattr(model, "use_profile_film", False))) and float(getattr(args, "lambda_rr_stft_consistency", 0.0)) > 0.0:
            l_rrspec = rr_stft_consistency_loss(pred_logmag, rr_pred)
            loss = loss + float(args.lambda_rr_stft_consistency) * l_rrspec

        l_profile_prior = pred_logmag.new_tensor(0.0)
        if profile_vector is not None and float(getattr(args, "lambda_profile_prior", 0.0)) > 0.0:
            l_profile_prior = (profile_vector.float() ** 2).mean()
            loss = loss + float(args.lambda_profile_prior) * l_profile_prior

        clsa_loss = model.last_profile_clsa_loss() if hasattr(model, "last_profile_clsa_loss") else None
        if clsa_loss is not None and float(getattr(args, "profile_clsa_loss_weight", 0.0)) > 0.0:
            loss = loss + float(getattr(args, "profile_clsa_loss_weight", 0.0)) * clsa_loss

        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        totals["loss"].append(float(loss.detach().cpu()))
        totals["stft"].append(parts["stft"])
        totals["rr"].append(parts["rr"])
        totals["contrast"].append(float(lc.detach().cpu()))
        totals["rr_stft_consistency"].append(float(l_rrspec.detach().cpu()))
        totals["profile_prior"].append(float(l_profile_prior.detach().cpu()))

    return EpochMetrics(**{k: float(np.mean(v)) if v else float("nan") for k, v in totals.items()})


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str, args, save_arrays: Optional[Path] = None) -> Dict[str, float]:
    model.eval()
    losses, parts_all = [], {"stft": [], "rr": [], "rr_stft_consistency": [], "profile_prior": []}
    pred_specs, true_specs = [], []
    rr_preds, rr_trues = [], []

    for batch in loader:
        imu, pressure, _, br, _ = unpack_batch(batch, device)
        pred_logmag, rr_pred, _, _profile_stats, profile_vector = _forward_with_optional_profile(
            model,
            imu,
            device,
        )
        loss, parts = pressure_stft_recon_loss(
            model,
            pred_logmag,
            pressure,
            rr_pred,
            br,
            lambda_stft=args.lambda_stft,
            lambda_rr=args.lambda_rr,
        )
        l_rrspec = pred_logmag.new_tensor(0.0)
        if bool(getattr(model, "use_profile_conditioning", getattr(model, "use_profile_film", False))) and float(getattr(args, "lambda_rr_stft_consistency", 0.0)) > 0.0:
            l_rrspec = rr_stft_consistency_loss(pred_logmag, rr_pred)
            loss = loss + float(args.lambda_rr_stft_consistency) * l_rrspec

        l_profile_prior = pred_logmag.new_tensor(0.0)
        if profile_vector is not None and float(getattr(args, "lambda_profile_prior", 0.0)) > 0.0:
            l_profile_prior = (profile_vector.float() ** 2).mean()
            loss = loss + float(args.lambda_profile_prior) * l_profile_prior

        losses.append(float(loss.cpu()))
        parts_all["stft"].append(parts["stft"])
        parts_all["rr"].append(parts["rr"])
        parts_all["rr_stft_consistency"].append(float(l_rrspec.detach().cpu()))
        parts_all["profile_prior"].append(float(l_profile_prior.detach().cpu()))

        true_logmag = model.pressure_stft_target(pressure, target_tokens=pred_logmag.size(1))
        pred_specs.append(pred_logmag.detach().cpu().numpy())
        true_specs.append(true_logmag.detach().cpu().numpy())

        rr_true = rr_targets_from_batch(pressure, br)
        rr_preds.append(rr_pred.detach().cpu().numpy().reshape(-1))
        rr_trues.append(rr_true.detach().cpu().numpy().reshape(-1))

    spec_true = np.concatenate(true_specs, axis=0)
    spec_pred = np.concatenate(pred_specs, axis=0)
    spec_err = spec_pred - spec_true
    rr_true = np.concatenate(rr_trues, axis=0)
    rr_pred = np.concatenate(rr_preds, axis=0)
    rr_err = rr_pred - rr_true

    spec_corr = float(np.corrcoef(spec_true.reshape(-1), spec_pred.reshape(-1))[0, 1]) if spec_true.size > 1 else float("nan")
    rr_corr = (
        float(np.corrcoef(rr_true.reshape(-1), rr_pred.reshape(-1))[0, 1])
        if rr_true.size > 1 and np.std(rr_true) > 1e-8 and np.std(rr_pred) > 1e-8
        else float("nan")
    )

    metrics = {
        "loss": float(np.mean(losses)),
        "stft": float(np.mean(parts_all["stft"])),
        "rr_loss": float(np.mean(parts_all["rr"])),
        "rr_stft_consistency": float(np.mean(parts_all["rr_stft_consistency"])),
        "profile_prior": float(np.mean(parts_all["profile_prior"])),
        "spec_mae": float(np.mean(np.abs(spec_err))),
        "spec_rmse": float(np.sqrt(np.mean(spec_err ** 2))),
        "spec_corr": spec_corr,
        "rr_mae": float(np.mean(np.abs(rr_err))),
        "rr_rmse": float(np.sqrt(np.mean(rr_err ** 2))),
        "rr_corr": rr_corr,
        "n_windows": int(spec_true.shape[0]),
    }

    if save_arrays is not None:
        save_arrays.mkdir(parents=True, exist_ok=True)
        np.save(save_arrays / "pressure_stft_true.npy", spec_true)
        np.save(save_arrays / "pressure_stft_pred.npy", spec_pred)
        np.save(save_arrays / "rr_true.npy", rr_true)
        np.save(save_arrays / "rr_pred.npy", rr_pred)

    return metrics


def pooled_features(hidden: torch.Tensor) -> torch.Tensor:
    return hidden.mean(dim=1)


@torch.no_grad()
def collect_source_tta_stats(model: nn.Module, loader, device: str, args) -> Dict[str, torch.Tensor]:
    model.eval()
    zs, rrs = [], []
    max_batches = int(args.tta_source_batches)
    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        imu, pressure, _, br, _ = unpack_batch(batch, device)
        _, _rr_pred, hidden = model(imu)
        z = pooled_features(hidden)
        rr_true = rr_targets_from_batch(pressure, br)
        zs.append(z.detach())
        rrs.append(rr_true.detach())

    if not zs:
        raise RuntimeError("Could not collect source TTA statistics: empty train loader.")

    z_all = torch.cat(zs, dim=0).float()
    rr = torch.cat(rrs, dim=0).float()
    mu = z_all.mean(dim=0)
    sd = z_all.std(dim=0, unbiased=False).clamp_min(1e-6)

    zc = z_all - mu
    rank = max(1, min(int(args.ssa_rank), zc.size(0) - 1, zc.size(1)))
    try:
        _, _, v = torch.pca_lowrank(zc, q=rank, center=False)
        basis = v[:, :rank]
    except Exception:
        _, _, vh = torch.linalg.svd(zc, full_matrices=False)
        basis = vh[:rank].T
    src_proj = zc @ basis
    src_proj_mu = src_proj.mean(dim=0)
    src_proj_sd = src_proj.std(dim=0, unbiased=False).clamp_min(1e-6)

    k = max(1, int(args.proto_k))
    order = torch.argsort(rr)
    chunks = torch.chunk(order, k)
    protos, centers = [], []
    for idx in chunks:
        if idx.numel() == 0:
            continue
        protos.append(z_all[idx].mean(dim=0))
        centers.append(rr[idx].mean())
    prototypes = torch.stack(protos, dim=0)
    rr_centers = torch.stack(centers, dim=0)

    return {
        "mu": mu.detach(),
        "sd": sd.detach(),
        "basis": basis.detach(),
        "src_proj_mu": src_proj_mu.detach(),
        "src_proj_sd": src_proj_sd.detach(),
        "prototypes": prototypes.detach(),
        "rr_centers": rr_centers.detach(),
    }


def rr_from_predicted_stft(pred_logmag: torch.Tensor, fs: float = BR_FS, min_hz: float = 0.05, max_hz: float = 0.75) -> torch.Tensor:
    del min_hz, max_hz
    return estimate_rr_from_predicted_stft(pred_logmag, br_fs=float(fs), return_confidence=False)


def ssa_alignment_loss(z: torch.Tensor, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    basis = stats["basis"].to(z.device)
    src_mu = stats["src_proj_mu"].to(z.device)
    src_sd = stats["src_proj_sd"].to(z.device)
    zc = z - stats["mu"].to(z.device)
    proj = zc @ basis
    mu = proj.mean(dim=0)
    sd = proj.std(dim=0, unbiased=False).clamp_min(1e-6)
    return F.mse_loss(mu, src_mu) + F.mse_loss(torch.log(sd), torch.log(src_sd))


def augmentation_consistency_loss(model: nn.Module, imu: torch.Tensor, pred_logmag: torch.Tensor, rr_pred: torch.Tensor, z: torch.Tensor, args) -> torch.Tensor:
    imu_aug = augment_imu(imu, shift_max=args.shift_max)
    pred_aug, rr_aug, h_aug = model(imu_aug)
    z_aug = pooled_features(h_aug)
    spec_loss = F.smooth_l1_loss(pred_aug, pred_logmag.detach())
    rr_loss = F.smooth_l1_loss(rr_aug.view(-1), rr_pred.detach().view(-1))
    z_loss = F.mse_loss(F.normalize(z_aug, dim=-1), F.normalize(z.detach(), dim=-1))
    return spec_loss + rr_loss + z_loss


def rr_stft_consistency_loss(pred_logmag: torch.Tensor, rr_pred: torch.Tensor) -> torch.Tensor:
    rr_spec = rr_from_predicted_stft(pred_logmag)
    return F.smooth_l1_loss(rr_pred.view(-1), rr_spec.detach().view(-1))


def gated_temporal_smoothness_loss(z: torch.Tensor, rr_pred: torch.Tensor, pred_logmag: torch.Tensor, gate_scale: float = 1.0) -> torch.Tensor:
    if z.size(0) < 2:
        return z.new_tensor(0.0)
    dz = (z[1:] - z[:-1]).pow(2).mean(dim=1).sqrt()
    gate = torch.exp(-dz.detach() / max(float(gate_scale), 1e-6)).clamp(0.0, 1.0)
    rr_step = F.smooth_l1_loss(rr_pred[1:], rr_pred[:-1], reduction="none")
    spec_step = (pred_logmag[1:] - pred_logmag[:-1]).abs().mean(dim=(1, 2))
    z_step = (F.normalize(z[1:], dim=-1) - F.normalize(z[:-1], dim=-1)).pow(2).mean(dim=1)
    return (gate * (rr_step + spec_step + z_step)).mean()


def prototype_alignment_loss(z: torch.Tensor, rr_pred: torch.Tensor, stats: Dict[str, torch.Tensor], temperature: float = 2.0) -> torch.Tensor:
    protos = stats["prototypes"].to(z.device)
    centers = stats["rr_centers"].to(z.device)
    if protos.numel() == 0:
        return z.new_tensor(0.0)
    dist_rr = (rr_pred.view(-1, 1) - centers.view(1, -1)).abs()
    weights = torch.softmax(-dist_rr / max(float(temperature), 1e-6), dim=1)
    target_proto = weights @ protos
    return F.mse_loss(F.normalize(z, dim=-1), F.normalize(target_proto.detach(), dim=-1))


def set_tta_trainable(model: nn.Module, mode: str = "norm_proj_rr") -> None:
    for p in model.parameters():
        p.requires_grad = False
    mode = str(mode).lower()
    for name, module in model.named_modules():
        allow = False
        if isinstance(module, nn.LayerNorm) and "norm" in mode:
            allow = True
        if "proj" in mode and any(k in name for k in ("spec_proj", "pressure_proj")):
            allow = True
        if "head" in mode and any(k in name for k in ("rr_head", "pressure_mag_head")):
            allow = True
        if "rr" in mode and "rr_head" in name:
            allow = True
        if allow:
            for p in module.parameters(recurse=False):
                p.requires_grad = True
    if "proj" in mode:
        for attr in ("spec_proj", "pressure_proj"):
            module = getattr(model, attr, None)
            if isinstance(module, nn.Module):
                for p in module.parameters():
                    p.requires_grad = True


def tta_loss_for_batch(
    model: nn.Module,
    imu: torch.Tensor,
    stats: Dict[str, torch.Tensor],
    args,
    prev_state: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    pred_logmag, rr_pred, hidden = model(imu)
    z = pooled_features(hidden)

    l_ssa = ssa_alignment_loss(z, stats) if args.lambda_tta_ssa > 0 else z.new_tensor(0.0)
    l_cons = augmentation_consistency_loss(model, imu, pred_logmag, rr_pred, z, args) if args.lambda_tta_cons > 0 else z.new_tensor(0.0)
    l_rrspec = rr_stft_consistency_loss(pred_logmag, rr_pred) if args.lambda_tta_rrspec > 0 else z.new_tensor(0.0)
    l_proto = prototype_alignment_loss(z, rr_pred, stats, temperature=args.proto_temperature) if args.lambda_tta_proto > 0 else z.new_tensor(0.0)
    l_smooth = z.new_tensor(0.0)
    if args.lambda_tta_smooth > 0:
        l_smooth = gated_temporal_smoothness_loss(z, rr_pred, pred_logmag, gate_scale=args.smooth_gate_scale)
        if prev_state is not None and prev_state.get("z") is not None:
            z_cat = torch.cat([prev_state["z"].to(z.device), z], dim=0)
            rr_cat = torch.cat([prev_state["rr"].to(z.device), rr_pred], dim=0)
            spec_cat = torch.cat([prev_state["spec"].to(z.device), pred_logmag], dim=0)
            l_smooth = 0.5 * (l_smooth + gated_temporal_smoothness_loss(z_cat, rr_cat, spec_cat, gate_scale=args.smooth_gate_scale))

    loss = (
        args.lambda_tta_ssa * l_ssa
        + args.lambda_tta_cons * l_cons
        + args.lambda_tta_rrspec * l_rrspec
        + args.lambda_tta_smooth * l_smooth
        + args.lambda_tta_proto * l_proto
    )
    parts = {
        "tta_loss": float(loss.detach().cpu()),
        "ssa": float(l_ssa.detach().cpu()),
        "cons": float(l_cons.detach().cpu()),
        "rrspec": float(l_rrspec.detach().cpu()),
        "smooth": float(l_smooth.detach().cpu()),
        "proto": float(l_proto.detach().cpu()),
    }
    next_state = {"z": z[-1:].detach(), "rr": rr_pred[-1:].detach(), "spec": pred_logmag[-1:].detach()}
    return loss, parts, next_state


def run_subject_tta(model: nn.Module, source_loader, target_loader, device: str, args, out_dir: Optional[Path] = None) -> Dict[str, float]:
    if str(args.tta).lower() == "none" or int(args.tta_epochs) <= 0:
        return {}

    active_tta_weight = (
        abs(float(args.lambda_tta_ssa))
        + abs(float(args.lambda_tta_cons))
        + abs(float(args.lambda_tta_rrspec))
        + abs(float(args.lambda_tta_smooth))
        + abs(float(args.lambda_tta_proto))
    )
    if active_tta_weight <= 0.0:
        print("[TTA] All TTA lambda weights are zero; skipping TTA.")
        return {}

    stats = collect_source_tta_stats(model, source_loader, device, args)
    set_tta_trainable(model, args.tta_adapt)
    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = int(sum(p.numel() for p in params))
    print(f"[TTA] Trainable parameters: {n_trainable} | adapt={args.tta_adapt}")
    if not params:
        print("[TTA] No trainable parameters selected; skipping TTA.")
        return {}
    opt = torch.optim.AdamW(params, lr=args.tta_lr, weight_decay=args.tta_weight_decay)

    rows = []
    for epoch in range(1, int(args.tta_epochs) + 1):
        model.train()
        prev_state = None
        totals: Dict[str, List[float]] = {"tta_loss": [], "ssa": [], "cons": [], "rrspec": [], "smooth": [], "proto": []}
        for batch in target_loader:
            imu, _, _, _, _ = unpack_batch(batch, device)
            for _ in range(max(1, int(args.tta_steps_per_batch))):
                opt.zero_grad(set_to_none=True)
                loss, parts, next_state = tta_loss_for_batch(model, imu, stats, args, prev_state=prev_state)
                if not torch.isfinite(loss):
                    continue
                if not loss.requires_grad:
                    print("[TTA] Loss has no grad_fn; skipping this update. Check nonzero --lambda-tta-* weights and --tta-adapt.")
                    continue
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                opt.step()
                prev_state = next_state
                for k in totals:
                    totals[k].append(parts[k])
        row = {f"tta_{k}": float(np.mean(v)) if v else float("nan") for k, v in totals.items()}
        row["tta_epoch"] = epoch
        rows.append(row)
        print(
            f"TTA epoch {epoch:03d} | loss {row['tta_tta_loss']:.4f} "
            f"ssa {row['tta_ssa']:.4f} cons {row['tta_cons']:.4f} "
            f"rrspec {row['tta_rrspec']:.4f} smooth {row['tta_smooth']:.4f} proto {row['tta_proto']:.4f}"
        )

    if out_dir is not None and rows:
        pd.DataFrame(rows).to_csv(out_dir / "tta_history.csv", index=False)
    return rows[-1] if rows else {}


def infer_n_channels(loader) -> int:
    batch = next(iter(loader))[0]
    if batch.ndim != 3:
        raise ValueError(f"Expected 3D IMU batch, got {tuple(batch.shape)}")
    return int(min(batch.shape[1], batch.shape[2]))


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args,
    epoch: int,
    best_val: float,
    hist: List[Dict[str, float]],
    extra_metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "epoch": int(epoch),
        "best_val": float(best_val),
        "history": hist,
    }
    if extra_metadata:
        payload.update(extra_metadata)
    return payload


def save_last_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args,
    epoch: int,
    best_val: float,
    hist: List[Dict[str, float]],
    extra_metadata: Optional[Dict[str, object]] = None,
) -> None:
    torch.save(_checkpoint_payload(model, optimizer, args, epoch, best_val, hist, extra_metadata=extra_metadata), path)


def load_resume_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, device: str) -> Tuple[int, float, List[Dict[str, float]]]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = int(ckpt.get("epoch", 0))
    best_val = float(ckpt.get("best_val", float("inf")))
    hist = list(ckpt.get("history", []))
    return epoch, best_val, hist


def read_summary_row(summary_path: Path, subject: str) -> Optional[Dict[str, float]]:
    if not summary_path.exists():
        return None
    df = pd.read_csv(summary_path)
    if "subject" not in df.columns:
        return None
    rows = df[df["subject"] == subject]
    if rows.empty:
        return None
    return rows.iloc[-1].to_dict()


def _profile_phase3_enabled(args) -> bool:
    mode = str(getattr(args, "profile_conditioning", "none")).lower().strip()
    return (
        bool(getattr(args, "use_profile_film", False))
        or bool(getattr(args, "use_profile_qkv", False))
        or bool(getattr(args, "use_profile_lora", False))
        or mode in {"film", "qkv", "film_qkv", "lora"}
    )


def _collect_phase3_profile_metadata(
    model: nn.Module,
    train_loader,
    device: str,
    args,
    *,
    apply_to_model: bool = False,
) -> Dict[str, object]:
    if not _profile_phase3_enabled(args):
        return {}

    max_batches = int(getattr(args, "profile_stats_max_batches", 0))
    max_batches_opt = None if max_batches <= 0 else max_batches
    normalizer = estimate_source_profile_normalizer(
        model,
        train_loader,
        device,
        max_batches=max_batches_opt,
        br_fs=float(BR_FS),
    )
    source_profile_stats = normalizer["source_profile_stats"].detach().cpu()
    source_profile_mean = normalizer["source_profile_mean"].detach().cpu()
    source_profile_std = normalizer["source_profile_std"].detach().cpu()
    source_profile_stats_norm = normalizer["source_profile_stats_norm"].detach().cpu()
    if bool(apply_to_model):
        _set_model_profile_normalizer(model, source_profile_mean, source_profile_std)
    latent_dim = int(getattr(model, "d_model", 0))
    split_stats = split_profile_stats(source_profile_stats, latent_dim)
    diag = profile_stats_diagnostics(
        source_profile_stats,
        source_profile_mean=source_profile_mean,
        source_profile_std=source_profile_std,
        profile_dim=int(getattr(model, "profile_dim", 0)),
    )
    diag.update(
        {
            "raw_profile_stats_rr_aux_mean_bpm": float(split_stats["rr_aux_mean"].item()),
            "raw_profile_stats_rr_stft_mean_bpm": float(split_stats["rr_stft_mean"].item()),
            "raw_profile_stats_rr_delta_mean_bpm": float(split_stats["rr_delta_mean"].item()),
            "raw_profile_stats_stft_confidence_mean": float(split_stats["stft_confidence_mean"].item()),
            "profile_stats_max_batches": int(max_batches if max_batches > 0 else 0),
            "profile_normalizer_applied_to_model": int(bool(apply_to_model)),
        }
    )
    return {
        "profile_metadata": {
            "source_profile_stats": source_profile_stats,
            "source_profile_mean": source_profile_mean,
            "source_profile_std": source_profile_std,
            "source_profile_stats_norm": source_profile_stats_norm,
            "profile_diagnostics": diag,
        }
    }


def read_history_rows(history_path: Path) -> List[Dict[str, float]]:
    if not history_path.exists():
        return []
    return pd.read_csv(history_path).to_dict("records")


def subject_outputs_complete(sbj_dir: Path) -> bool:
    required = [
        sbj_dir / "best_model.pt",
        sbj_dir / "history.csv",
        sbj_dir / "pressure_stft_true.npy",
        sbj_dir / "pressure_stft_pred.npy",
        sbj_dir / "rr_true.npy",
        sbj_dir / "rr_pred.npy",
    ]
    return all(path.exists() for path in required)


@torch.no_grad()
def frozen_embedding_from_batch(model: nn.Module, imu: torch.Tensor, args) -> torch.Tensor:
    pred_logmag, rr_pred, hidden = model(imu)

    mode = str(args.embed_pooling).lower()
    parts: List[torch.Tensor] = []

    if mode in {"mean", "mean_std", "mean_std_max", "rich"}:
        parts.append(hidden.mean(dim=1))
    if mode in {"mean_std", "mean_std_max", "rich"}:
        parts.append(hidden.std(dim=1, unbiased=False))
    if mode in {"mean_std_max", "rich"}:
        parts.append(hidden.max(dim=1).values)
    if mode == "max":
        parts.append(hidden.max(dim=1).values)
    if mode == "cls_last":
        parts.append(hidden[:, -1, :])
    if mode == "rich":
        parts.append(rr_pred.view(-1, 1))
        spec_mean = pred_logmag.mean(dim=(1, 2), keepdim=False).view(-1, 1)
        spec_std = pred_logmag.std(dim=(1, 2), unbiased=False, keepdim=False).view(-1, 1)
        spec_max = pred_logmag.amax(dim=(1, 2), keepdim=False).view(-1, 1)
        parts.extend([spec_mean, spec_std, spec_max])
        if bool(args.embed_stft_profile):
            parts.append(pred_logmag.mean(dim=1))

    if not parts:
        raise ValueError(f"Unsupported --embed-pooling={args.embed_pooling!r}")
    return torch.cat(parts, dim=1)


@torch.no_grad()
def collect_frozen_embeddings(model: nn.Module, loader, device: str, args) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    model.eval()
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    tlxs: List[np.ndarray] = []
    saw_tlx = False
    for batch in loader:
        imu, _, cond, _, tlx = unpack_batch(batch, device)
        z = frozen_embedding_from_batch(model, imu, args)
        xs.append(z.detach().cpu().numpy())
        ys.append(cond.detach().cpu().numpy().astype(int).reshape(-1))
        if tlx is not None:
            saw_tlx = True
            tlxs.append(tlx.detach().cpu().numpy().astype(np.float32).reshape(-1))
    if not xs:
        raise RuntimeError("No batches available for frozen-embedding collection.")
    tlx_out = np.concatenate(tlxs, axis=0) if saw_tlx and tlxs else None
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), tlx_out


def build_base_parser(default_subjects_list: List[str], default_out_dir: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=default_subjects_list, help="Subjects for LOSO")
    parser.add_argument("--data-str", default="imu_filt", choices=["imu_filt", "imu_ica"])
    parser.add_argument("--data-dir", default=SBJ_PROCESSED_DIR, help="Directory containing processed subject .pkl files")
    parser.add_argument("--data-group", default="mr", choices=["mr", "levels", "mr_levels"], help="Processed split to load; use mr for M/R pretraining or levels for L0-L3.")
    parser.add_argument("--include-tlx", action="store_true", help="Ask the latest dataloader for TLX values in pretraining loaders. Training ignores these unless downstream TLX probing is enabled.")
    parser.add_argument("--tlx-csv-path", default=TLX_CSV, help="Optional path to seated_tlx.csv for dataloader TLX mapping.")
    parser.add_argument("--mdl-dir", default=None, help="Model parent directory. Defaults to project dir")
    parser.add_argument("--out-dir", default=default_out_dir)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--use-profile-film", action="store_true", help="Enable profile-conditioned FiLM modules for alternate forward paths.")
    parser.add_argument("--use-profile-qkv", action="store_true", help="Enable profile-conditioned Q/K/V modulation inside selected transformer layers.")
    parser.add_argument("--shared-profile-qkv", action="store_true", help="Use one profile vector for both Profile-FiLM and residual QKV.")
    parser.add_argument("--use-profile-lora", action="store_true", help="Enable profile-conditioned low-rank residual adapters.")
    parser.add_argument("--profile-conditioning", default="none", choices=["none", "film", "qkv", "film_qkv", "lora"], help="Profile-conditioning path to use when profile modules are enabled.")
    parser.add_argument("--profile-dim", type=int, default=32)
    parser.add_argument("--profile-stats-dim", type=int, default=0)
    parser.add_argument("--profile-hidden-dim", type=int, default=128)
    parser.add_argument("--profile-film-scale", type=float, default=0.1)
    parser.add_argument(
        "--profile-film-placement",
        default="token_pooled",
        choices=["token_pooled", "pooled_only", "late_token_only", "residual"],
        help="Profile-FiLM placement: current token+pooled path, pooled-only, late-token-only, or residual-FiLM.",
    )
    parser.add_argument("--profile-film-residual-alpha", type=float, default=0.1)
    parser.add_argument("--profile-qkv-scale", type=float, default=0.1)
    parser.add_argument("--profile-qkv-layers", default="last1", choices=["none", "last1", "last2", "all"])
    parser.add_argument(
        "--profile-qkv-residual",
        action="store_true",
        help="Blend from unconditioned attention toward profile-QKV attention by --profile-qkv-scale.",
    )
    parser.add_argument("--profile-qkv-mode", default="static", choices=["static", "clsa"], help="Profile QKV path: static additive conditioner or embedded CLSA fast adapter.")
    parser.add_argument("--profile-clsa-rank", type=int, default=8)
    parser.add_argument("--profile-clsa-scale", type=float, default=0.01)
    parser.add_argument("--profile-clsa-eta-max", type=float, default=0.1)
    parser.add_argument("--profile-clsa-gate-init-bias", type=float, default=-2.0)
    parser.add_argument("--profile-clsa-enable-fast-update", type=int, default=1, help="Enable one-step CLSA fast update inside profile-QKV forward path.")
    parser.add_argument("--profile-clsa-loss-weight", type=float, default=0.0, help="Optional source-training regularizer on CLSA self-alignment loss. Default 0 keeps it only as fast-update objective.")
    parser.add_argument("--profile-lora-rank", type=int, default=8)
    parser.add_argument("--profile-lora-scale", type=float, default=0.05)
    parser.add_argument("--profile-stats-max-batches", type=int, default=50, help="Max source-train batches to use when estimating fold-level profile-stat normalization; 0 uses all.")
    parser.add_argument("--lambda-time", type=float, default=0.0, help="Deprecated; waveform reconstruction is disabled in this STFT/RR variant.")
    parser.add_argument("--lambda-stft", type=float, default=1.0, help="Weight for pressure log-STFT magnitude reconstruction.")
    parser.add_argument("--lambda-rr", type=float, default=0.1, help="Weight for auxiliary RR regression from pooled IMU features.")
    parser.add_argument("--lambda-rr-stft-consistency", type=float, default=0.05, help="Weight for RR-head vs STFT-derived RR consistency during profile-conditioned source training.")
    parser.add_argument("--lambda-profile-prior", type=float, default=0.01, help="Weight for the L2 prior on profile vectors during profile-conditioned source training.")
    parser.add_argument("--lambda-band", type=float, default=0.0, help="Deprecated; band-energy loss is disabled in this STFT/RR variant.")
    parser.add_argument("--lambda-contrast", type=float, default=0.05, help="Final weight for IMU-token <-> pressure-token contrastive loss after warmup/ramp")
    parser.add_argument("--lambda-contrast-min", type=float, default=0.0, help="Initial nonzero contrastive weight after warmup")
    parser.add_argument("--contrast-warmup-epochs", type=int, default=5, help="Number of initial epochs with contrastive loss disabled")
    parser.add_argument("--contrast-ramp-end-epoch", type=int, default=10, help="Epoch by which contrastive weight reaches --lambda-contrast")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--shift-max", type=int, default=24)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--pin-memory", type=int, default=0)
    parser.add_argument("--persistent-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume each subject from last_model.pt in --out-dir when available")

    parser.add_argument("--tta", default="none", choices=["none", "physio"], help="Run physiology-aware TTA after loading the best checkpoint.")
    parser.add_argument("--tta-epochs", type=int, default=0)
    parser.add_argument("--tta-steps-per-batch", type=int, default=1)
    parser.add_argument("--tta-lr", type=float, default=1e-5)
    parser.add_argument("--tta-weight-decay", type=float, default=0.0)
    parser.add_argument("--tta-adapt", default="norm_proj_rr", help="Small parameter set to adapt: e.g. norm, norm_proj, norm_proj_rr, norm_proj_head.")
    parser.add_argument("--tta-source-batches", type=int, default=0, help="Max source train batches for TTA stats/prototypes; 0 uses all.")
    parser.add_argument("--ssa-rank", type=int, default=32)
    parser.add_argument("--proto-k", type=int, default=6)
    parser.add_argument("--proto-temperature", type=float, default=2.0)
    parser.add_argument("--smooth-gate-scale", type=float, default=1.0)
    parser.add_argument("--lambda-tta-ssa", type=float, default=0.05)
    parser.add_argument("--lambda-tta-cons", type=float, default=0.10)
    parser.add_argument("--lambda-tta-rrspec", type=float, default=0.10)
    parser.add_argument("--lambda-tta-smooth", type=float, default=0.02)
    parser.add_argument("--lambda-tta-proto", type=float, default=0.05)

    parser.add_argument(
        "--decoder-mode",
        default="gru",
        choices=["none", "gru", "self_attn", "cross_attn"],
        help="Reconstruction decoder ablation. gru matches the current baseline.",
    )
    parser.add_argument(
        "--decoder-layers",
        type=int,
        default=1,
        help="Number of Transformer decoder/refiner layers for self_attn/cross_attn.",
    )
    parser.add_argument(
        "--rr-from",
        default="encoder",
        choices=["encoder", "decoder", "both"],
        help="Feature source for RR head. encoder matches the current baseline.",
    )
    parser.add_argument(
        "--imu-token-mixer",
        default="dwconv",
        choices=["none", "dwconv"],
        help="Optional depthwise temporal mixer over projected IMU STFT tokens before ViT tokenisation.",
    )
    parser.add_argument("--use-tcn-token-mixer", action="store_true", help="Enable a small residual raw-IMU temporal mixer before STFT tokenisation.")
    parser.add_argument("--tcn-mixer-alpha", type=float, default=0.05)
    parser.add_argument("--tcn-mixer-hidden", type=int, default=32)
    parser.add_argument("--tcn-mixer-layers", type=int, default=2)
    parser.add_argument(
        "--rr-head-type",
        default="mlp",
        choices=["mlp", "token_tcn"],
        help="RR readout head. mlp matches the current pooled baseline; token_tcn applies temporal convolutions over ViT tokens.",
    )
    parser.add_argument(
        "--rr-tcn-layers",
        type=int,
        default=2,
        help="Number of residual depthwise-separable token convolution blocks for --rr-head-type token_tcn.",
    )
    parser.add_argument(
        "--rr-tcn-kernel-size",
        type=int,
        default=3,
        help="Odd Conv1d kernel size over tokens for --rr-head-type token_tcn.",
    )
    parser.add_argument(
        "--rr-tcn-dropout",
        type=float,
        default=None,
        help="Dropout for --rr-head-type token_tcn. Defaults to the model dropout.",
    )
    return parser


PreEvalHook = Callable[[nn.Module, str, Any, Any, str, Any, Path], Dict[str, float]]
PostEvalHook = Callable[[nn.Module, str, List[str], Any, Any, str, Any, Path], List[Dict[str, Any]]]


def run_loocv_experiment(
    args,
    pre_eval_hooks: Optional[List[PreEvalHook]] = None,
    post_eval_hooks: Optional[List[PostEvalHook]] = None,
    config_mutator: Optional[Callable[[Any], None]] = None,
) -> Dict[str, pd.DataFrame]:
    pre_eval_hooks = pre_eval_hooks or []
    post_eval_hooks = post_eval_hooks or []
    if config_mutator is not None:
        config_mutator(args)

    set_seed(args.seed)
    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mdl_dir is None:
        args.mdl_dir = f"{M_DIR}/{args.data_str}/loocv"

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Subjects: {args.subjects}")
    print(f"Device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Model dir: {args.mdl_dir}")
    print(f"Output dir: {out_dir}")
    if bool(getattr(args, "eval_frozen_tlx", False)) or bool(getattr(args, "include_tlx", False)):
        print(f"TLX CSV: {args.tlx_csv_path}")

    rows: List[Dict[str, float]] = []
    extra_rows_by_name: Dict[str, List[Dict[str, Any]]] = {}

    for sbj in list(args.subjects):
        print(f"\n=== Held-out subject {sbj} ===")
        fold_seed = make_fold_seed(int(args.seed), str(sbj))
        set_seed(fold_seed)
        setattr(args, "base_seed", int(args.seed))
        setattr(args, "fold_seed", int(fold_seed))
        setattr(args, "calibration_seed", int(fold_seed))
        train_loader, val_loader, test_loader = build_loocv_loaders(
            sbj,
            args.subjects,
            args.data_str,
            val_split=getattr(args, "val_split", 0.25),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            data_dir=args.data_dir,
            mdl_dir=args.mdl_dir,
            autoencoder=None,
            data_group=args.data_group,
            include_tlx=bool(args.include_tlx),
            tlx_csv_path=args.tlx_csv_path,
            seed=fold_seed,
            num_workers=int(getattr(args, "num_workers", 0)),
            prefetch_factor=int(getattr(args, "prefetch_factor", 1)),
            pin_memory=bool(getattr(args, "pin_memory", 0)),
            persistent_workers=bool(getattr(args, "persistent_workers", 0)),
        )
        sbj_dir = out_dir / sbj
        sbj_dir.mkdir(parents=True, exist_ok=True)
        best_path = sbj_dir / "best_model.pt"
        last_path = sbj_dir / "last_model.pt"

        n_channels = infer_n_channels(train_loader)
        pred_len = int(round(20 * BR_FS))
        profile_stats_dim = int(getattr(args, "profile_stats_dim", 0))
        profile_conditioning = str(getattr(args, "profile_conditioning", "none")).lower().strip()
        if profile_conditioning == "none":
            if bool(getattr(args, "shared_profile_qkv", False)) or (
                bool(getattr(args, "use_profile_film", False))
                and bool(getattr(args, "use_profile_qkv", False))
            ):
                profile_conditioning = "film_qkv"
                setattr(args, "use_profile_film", True)
                setattr(args, "use_profile_qkv", True)
            elif bool(getattr(args, "use_profile_lora", False)):
                profile_conditioning = "lora"
            elif bool(getattr(args, "use_profile_qkv", False)):
                profile_conditioning = "qkv"
            elif bool(getattr(args, "use_profile_film", False)):
                profile_conditioning = "film"
        if profile_conditioning != "none" and profile_stats_dim <= 0:
            profile_stats_dim = infer_profile_stats_dim(int(args.d_model))
            setattr(args, "profile_stats_dim", profile_stats_dim)

        model = TinyIMU2PressureViT(
            input_channels=n_channels,
            d_model=args.d_model,
            pred_len=pred_len,
            nhead=args.heads,
            num_layers=args.layers,
            decoder_mode=str(getattr(args, "decoder_mode", "gru")),
            decoder_layers=int(getattr(args, "decoder_layers", 1)),
            rr_from=str(getattr(args, "rr_from", "encoder")),
            imu_token_mixer=str(getattr(args, "imu_token_mixer", "dwconv")),
            use_tcn_token_mixer=bool(getattr(args, "use_tcn_token_mixer", False)),
            tcn_mixer_alpha=float(getattr(args, "tcn_mixer_alpha", 0.05)),
            tcn_mixer_hidden=int(getattr(args, "tcn_mixer_hidden", 32)),
            tcn_mixer_layers=int(getattr(args, "tcn_mixer_layers", 2)),
            rr_head_type=str(getattr(args, "rr_head_type", "mlp")),
            rr_tcn_layers=int(getattr(args, "rr_tcn_layers", 2)),
            rr_tcn_kernel_size=int(getattr(args, "rr_tcn_kernel_size", 3)),
            rr_tcn_dropout=getattr(args, "rr_tcn_dropout", None),
            use_profile_film=bool(getattr(args, "use_profile_film", False)),
            use_profile_qkv=bool(getattr(args, "use_profile_qkv", False)),
            profile_conditioning=profile_conditioning,
            profile_dim=int(getattr(args, "profile_dim", 32)),
            profile_stats_dim=profile_stats_dim,
            profile_hidden_dim=int(getattr(args, "profile_hidden_dim", 128)),
            profile_film_scale=float(getattr(args, "profile_film_scale", 0.1)),
            profile_film_placement=str(getattr(args, "profile_film_placement", "token_pooled")),
            profile_film_residual_alpha=float(
                getattr(args, "profile_film_residual_alpha", 0.1)
            ),
            profile_qkv_scale=float(getattr(args, "profile_qkv_scale", 0.1)),
            profile_qkv_layers=str(
                getattr(args, "profile_qkv_layers", "last1")
            ),
            profile_qkv_residual=bool(getattr(args, "profile_qkv_residual", False)),
            profile_qkv_mode=str(getattr(args, "profile_qkv_mode", "static")),
            profile_clsa_rank=int(getattr(args, "profile_clsa_rank", 8)),
            profile_clsa_scale=float(getattr(args, "profile_clsa_scale", 0.01)),
            profile_clsa_eta_max=float(getattr(args, "profile_clsa_eta_max", 0.1)),
            profile_clsa_gate_init_bias=float(getattr(args, "profile_clsa_gate_init_bias", -2.0)),
            profile_clsa_enable_fast_update=bool(int(getattr(args, "profile_clsa_enable_fast_update", 1))),
            profile_clsa_loss_weight=float(getattr(args, "profile_clsa_loss_weight", 0.0)),
        ).to(device)


        with torch.no_grad():
            warm_batch = next(iter(train_loader))
            warm_imu, warm_pressure, _, _, _ = unpack_batch(warm_batch, device)
            warm_spec, _warm_rr, warm_h = model(warm_imu[:1])
            _ = model.pressure_stft_target(warm_pressure[:1], target_tokens=warm_spec.size(1))
            _ = model.encode_pressure(warm_pressure[:1], target_tokens=warm_h.size(1))

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        hist = []
        best_val = float("inf")
        start_epoch = 1

        if args.resume:
            if last_path.exists():
                last_epoch, best_val, hist = load_resume_checkpoint(last_path, model, opt, device)
                start_epoch = last_epoch + 1
                print(f"[RESUME] Loaded {last_path} at epoch {last_epoch}; target epochs={args.epochs}, best_val={best_val:.4f}")
            elif subject_outputs_complete(sbj_dir):
                hist = read_history_rows(sbj_dir / "history.csv")
                summary_row = read_summary_row(out_dir / "summary.csv", sbj)
                if summary_row is not None:
                    print(f"[RESUME] Complete outputs found for {sbj}; reusing summary row.")
                    rows.append(summary_row)
                    close_loaders(train_loader, val_loader, test_loader)
                    del train_loader, val_loader, test_loader, model
                    gc.collect()
                    if str(device).startswith("cuda") and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                print(f"[RESUME] Complete outputs found for {sbj}; summary row missing, rerunning test only.")
                start_epoch = args.epochs + 1

        if args.resume and subject_outputs_complete(sbj_dir) and start_epoch > args.epochs:
            summary_row = read_summary_row(out_dir / "summary.csv", sbj)
            if summary_row is not None:
                print(f"[RESUME] {sbj} already complete through epoch {start_epoch - 1}; reusing summary row.")
                rows.append(summary_row)
                close_loaders(train_loader, val_loader, test_loader)
                del train_loader, val_loader, test_loader, model
                gc.collect()
                if str(device).startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            print(f"[RESUME] {sbj} training complete; summary row missing, rerunning test only.")

        for epoch in range(start_epoch, args.epochs + 1):
            epoch_lambda_contrast = contrast_weight_for_epoch(args, epoch)
            tr = train_one_epoch(model, train_loader, opt, device, args, epoch_lambda_contrast)
            val = evaluate(model, val_loader, device, args)
            hist.append({
                "epoch": epoch,
                "lambda_contrast": epoch_lambda_contrast,
                **{f"train_{k}": v for k, v in asdict(tr).items()},
                **{f"val_{k}": v for k, v in val.items()},
            })
            print(
                f"epoch {epoch:03d} | lambda_contrast {epoch_lambda_contrast:.4f} | "
                f"train loss {tr.loss:.4f} stft {tr.stft:.4f} rr {tr.rr:.4f} "
                f"rrspec {tr.rr_stft_consistency:.4f} pprior {tr.profile_prior:.4f} con {tr.contrast:.4f} | "
                f"val loss {val['loss']:.4f} spec_mae {val['spec_mae']:.4f} "
                f"spec_corr {val['spec_corr']:.3f} rr_mae {val['rr_mae']:.3f} rr_corr {val['rr_corr']:.3f}"
            )
            if val["loss"] < best_val:
                best_val = val["loss"]
                torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "epoch": epoch, "val": val}, best_path)
            save_last_checkpoint(last_path, model, opt, args, epoch, best_val, hist)
            pd.DataFrame(hist).to_csv(sbj_dir / "history.csv", index=False)

        pd.DataFrame(hist).to_csv(sbj_dir / "history.csv", index=False)

        if not best_path.exists() and last_path.exists():
            print(f"[CKPT] No best_model.pt found for {sbj}; using last checkpoint for test.")
            ckpt = torch.load(last_path, map_location=device)
            torch.save(
                {
                    "model_state_dict": ckpt["model_state_dict"],
                    "args": ckpt.get("args", vars(args)),
                    "epoch": ckpt.get("epoch", args.epochs),
                    "val": {"loss": ckpt.get("best_val", float('nan'))},
                },
                best_path,
            )

        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        profile_ckpt_metadata = _collect_phase3_profile_metadata(
            model,
            train_loader,
            device,
            args,
            apply_to_model=False,
        )
        if profile_ckpt_metadata:
            best_payload = dict(ckpt)
            best_payload.update(profile_ckpt_metadata)
            torch.save(best_payload, best_path)
            if last_path.exists():
                last_ckpt = torch.load(last_path, map_location=device)
                last_payload = dict(last_ckpt)
                last_payload.update(profile_ckpt_metadata)
                torch.save(last_payload, last_path)
        pristine_post_eval_state = _state_dict_cpu_clone(model)
        test_pre = evaluate(model, test_loader, device, args, save_arrays=sbj_dir / "pre_tta")
        print(f"TEST_PRE_TTA {sbj}: {test_pre}")

        for hook in pre_eval_hooks:
            hook_metrics = hook(model, sbj, train_loader, test_loader, device, args, sbj_dir)
            if hook_metrics:
                name = hook_metrics.pop("__summary_name__", None)
                row = hook_metrics.pop("__summary_row__", None)
                if name is not None and row is not None:
                    extra_rows_by_name.setdefault(name, []).append(row)

        tta_last = {}
        test = test_pre
        if str(args.tta).lower() != "none" and int(args.tta_epochs) > 0:
            tta_last = run_subject_tta(model, train_loader, test_loader, device, args, out_dir=sbj_dir)
            test = evaluate(model, test_loader, device, args, save_arrays=sbj_dir / "post_tta")
            print(f"TEST_POST_TTA {sbj}: {test}")

        _ = evaluate(model, test_loader, device, args, save_arrays=sbj_dir)
        row = {
            "subject": sbj,
            "base_seed": int(getattr(args, "base_seed", args.seed)),
            "fold_seed": int(getattr(args, "fold_seed", fold_seed)),
            "calibration_seed": int(getattr(args, "calibration_seed", fold_seed)),
            **test,
        }
        if bool(getattr(model, "use_profile_lora", False)):
            row.update(
                {
                    "profile_lora_rank": int(getattr(model, "profile_lora_rank", 0)),
                    "profile_lora_scale": float(getattr(model, "profile_lora_scale", 0.0)),
                    "profile_lora_trainable_params": int(
                        sum(
                            p.numel()
                            for name, p in model.named_parameters()
                            if "profile_lora" in name and p.requires_grad
                        )
                    ),
                }
            )
        if profile_ckpt_metadata:
            row.update(profile_ckpt_metadata["profile_metadata"]["profile_diagnostics"])
        if str(args.tta).lower() != "none" and int(args.tta_epochs) > 0:
            row.update({f"pre_{k}": v for k, v in test_pre.items()})
            row.update(tta_last)
        rows.append(row)

        for hook in post_eval_hooks:
            _restore_model_state(model, pristine_post_eval_state, device)
            hook_params = inspect.signature(hook).parameters
            if len(hook_params) >= 9:
                hook_items = hook(model, sbj, list(args.subjects), train_loader, val_loader, test_loader, device, args, sbj_dir)
            else:
                hook_items = hook(model, sbj, list(args.subjects), train_loader, test_loader, device, args, sbj_dir)
            if not _state_dict_matches(model, pristine_post_eval_state):
                print(f"[HOOK_STATE] {getattr(hook, '__name__', 'post_eval_hook')} mutated model state for {sbj}; restoring best checkpoint state.")
                _restore_model_state(model, pristine_post_eval_state, device)
            for item in hook_items:
                name = item.pop("__summary_name__")
                extra_rows_by_name.setdefault(name, []).append(item)
        close_loaders(train_loader, val_loader, test_loader)
        del train_loader, val_loader, test_loader, model
        gc.collect()
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    print("\n=== Summary ===")
    print(df)
    print("\nMean metrics:")
    print(df.drop(columns=["subject"]).mean(numeric_only=True))

    outputs: Dict[str, pd.DataFrame] = {"summary": df}
    for name, row_list in extra_rows_by_name.items():
        extra_df = pd.DataFrame(row_list)
        extra_df.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"\n=== {name} ===")
        print(extra_df)
        drop_cols = [c for c in ("subject", "tag") if c in extra_df.columns]
        print(f"\n{name} mean metrics:")
        print(extra_df.drop(columns=drop_cols).mean(numeric_only=True))
        outputs[name] = extra_df
    return outputs
