#!/usr/bin/env python3
"""
Profile-conditioned IMU->RR/STFT with an added TCN Profile-FiLM MWL head
and 1-minute post-processing.

This is intentionally a separate entrypoint. It keeps the existing RR/STFT
training path unchanged, then adds a downstream mental-workload classifier
trained on source-subject L0-L3 windows and evaluated on the held-out subject.

Key points:
  - Backbone/source training is delegated to vit_pressure_crossmodal_stft_rr_core.
  - The MWL head is a small TCN over encoder tokens, with Profile-FiLM on both
    token and pooled features.
  - Evaluation can use the last-layer profile-QKV episodic TTT path from the
    latest DCT/QKV runner (e.g. profile_qkv_ttt_sample, last1 by default).
  - Window outputs are post-processed into non-overlapping 1-minute chunks by
    averaging class probabilities over consecutive windows.

Example smoke run:
  python vit_profile_tcn_mwl_qkv_ttt_1min.py \
    --subjects S13 S19 S22 S25 \
    --data-str imu_filt \
    --data-group mr \
    --mwl-train-data-group levels \
    --mwl-test-data-group levels \
    --mwl-task levels \
    --use-profile-qkv \
    --profile-conditioning qkv \
    --profile-qkv-layers last1 \
    --profile-qkv-scale 0.03 \
    --mwl-ttt-modes none profile_qkv_ttt_sample \
    --out-dir results/vit_profile_tcn_mwl_qkv_ttt_1min_smoke
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import types

# Some project files import ipdb unconditionally. Keep this standalone script
# runnable in environments where ipdb is not installed.
try:
    import ipdb  # noqa: F401
except Exception:
    shim = types.ModuleType("ipdb")
    shim.set_trace = lambda *args, **kwargs: None
    sys.modules.setdefault("ipdb", shim)
try:
    import pyxdf  # noqa: F401
except Exception:
    shim = types.ModuleType("pyxdf")
    shim.load_xdf = lambda *args, **kwargs: (_ for _ in ()).throw(ImportError("pyxdf is unavailable"))
    sys.modules.setdefault("pyxdf", shim)


try:
    import datapipeline  # noqa: F401
except Exception:
    shim = types.ModuleType("datapipeline")
    def _missing(*args, **kwargs):
        raise ImportError("datapipeline is unavailable")
    shim.load_and_snip = _missing
    shim.get_windowed_data = _missing
    shim.get_file_list = _missing
    shim.load_split_data = _missing
    shim.load_harness_data = _missing
    sys.modules.setdefault("datapipeline", shim)

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from config import SBJ_PROCESSED_DIR
from dataloader import LoadDataset
from vit_pressure_crossmodal_profile_encoder import ProfileFiLM, profile_stats_dim
from vit_pressure_crossmodal_stft_rr_core import (
    build_base_parser,
    default_subjects,
    pooled_features,
    run_loocv_experiment,
    unpack_batch,
)
from vit_pressure_crossmodal_stft_rr_rrprobe_tta_main import (
    FaithfulRRRegressor,
    TrainConfig,
    collect_rr_arrays,
    train_source_rr_regressor,
)
from vit_pressure_crossmodal_stft_rr_unsup_ladder_config_sweep import (
    _patch_loocv_generator_for_eval_subjects,
    _restore_state_dict,
    _set_rr_model_frozen,
    _state_dict_cpu_clone,
    _warm_lazy_modules,
    add_common_adaptation_args,
)
from vit_pressure_crossmodal_profile_ttt_episodic_dct_qkv import (
    TTT_PROFILE_MODES,
    _apply_qkv_effective_scale,
    _base_rr_probe_batch,
    _profile_conditioned_rr_probe,
    _profile_stats_from_unlabeled_batch,
    _ttt_loss_for_batch,
    evaluate_qkv_calibration_gate,
)
from imu_mwl_classify import (
    CLASS_LABEL_TO_ID,
    TASK_CHOICES,
    _groups_for_task,
    _infer_task_from_group,
    _load_classification_arrays,
    _target_classes,
)


IMU_ISSUES_MR = [17, 26, 30]
IMU_ISSUES_L = [15, 17, 21, 26, 28, 30]
SUBJECTS = default_subjects(IMU_ISSUES_MR, IMU_ISSUES_L)


MWL_TTT_ALL_MODES = {"none", *TTT_PROFILE_MODES.keys()}


def _parse_modes(text: str) -> List[str]:
    modes = [p.strip() for p in str(text).replace(",", " ").split() if p.strip()]
    if not modes:
        raise ValueError("No MWL TTT modes provided.")
    bad = [m for m in modes if m not in MWL_TTT_ALL_MODES]
    if bad:
        raise ValueError(f"Unsupported --mwl-ttt-modes entries: {bad}. Valid={sorted(MWL_TTT_ALL_MODES)}")
    return modes


def _parse_mwl_class_subset(text: str) -> Tuple[List[str], List[int]]:
    names = [part.strip().upper() for part in str(text or "").replace(" ", "").split(",") if part.strip()]
    if not names:
        return [], []
    bad = [name for name in names if name not in CLASS_LABEL_TO_ID]
    if bad:
        raise ValueError(f"Unsupported --mwl-class-subset labels {bad}. Valid={sorted(CLASS_LABEL_TO_ID)}")
    deduped = list(dict.fromkeys(names))
    return deduped, [int(CLASS_LABEL_TO_ID[name]) for name in deduped]


def _remap_mwl_subset(
    x: np.ndarray,
    y: np.ndarray,
    br: Optional[np.ndarray],
    cond: np.ndarray,
    original_ids: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    ids = [int(v) for v in original_ids]
    id_to_idx = {orig: idx for idx, orig in enumerate(ids)}
    cond = np.asarray(cond).astype(int).reshape(-1)
    mask = np.isin(cond, np.asarray(ids, dtype=int))
    if not bool(mask.any()):
        return x[:0], y[:0], None if br is None else br[:0], cond[:0]
    cond_remap = np.asarray([id_to_idx[int(v)] for v in cond[mask]], dtype=int)
    return x[mask], y[mask], None if br is None else br[mask], cond_remap


MWL_DIAGNOSTIC_TASKS = {
    "levels": {
        "name": "levels",
        "class_groups": [("L0", [2]), ("L1", [3]), ("L2", [4]), ("L3", [5])],
    },
    "binary_low_high": {
        "name": "binary_low_high",
        "class_groups": [("low_L0_L1", [2, 3]), ("high_L2_L3", [4, 5])],
    },
    "rest_vs_load": {
        "name": "rest_vs_load",
        "class_groups": [("R", [1]), ("load_L0_L3", [2, 3, 4, 5])],
    },
}


def _parse_csv_names(text: str) -> List[str]:
    return [part.strip() for part in str(text or "").replace(" ", "").split(",") if part.strip()]


def _parse_mwl_diagnostic_tasks(text: str) -> List[Dict[str, object]]:
    names = _parse_csv_names(text) or ["levels", "binary_low_high", "rest_vs_load"]
    bad = [name for name in names if name not in MWL_DIAGNOSTIC_TASKS]
    if bad:
        raise ValueError(f"Unsupported --mwl-diagnostic-tasks entries {bad}. Valid={sorted(MWL_DIAGNOSTIC_TASKS)}")
    return [MWL_DIAGNOSTIC_TASKS[name] for name in dict.fromkeys(names)]


def _parse_mwl_head_variants(text: str) -> List[str]:
    valid = {"A0", "A1", "A2", "linear", "mlp", "tcn"}
    names = _parse_csv_names(text) or ["A0", "A1", "A2"]
    bad = [name for name in names if name not in valid]
    if bad:
        raise ValueError(f"Unsupported --mwl-head-variants entries {bad}. Valid=A0,A1,A2")
    alias = {"linear": "A0", "mlp": "A1", "tcn": "A2"}
    return [alias.get(name, name) for name in dict.fromkeys(names)]


def _remap_mwl_class_groups(
    x: np.ndarray,
    y: np.ndarray,
    br: Optional[np.ndarray],
    cond: np.ndarray,
    class_groups: Sequence[Tuple[str, Sequence[int]]],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    id_to_idx: Dict[int, int] = {}
    for idx, (_name, ids) in enumerate(class_groups):
        for original_id in ids:
            id_to_idx[int(original_id)] = int(idx)
    cond = np.asarray(cond).astype(int).reshape(-1)
    mask = np.asarray([int(v) in id_to_idx for v in cond], dtype=bool)
    if not bool(mask.any()):
        return x[:0], y[:0], None if br is None else br[:0], cond[:0]
    cond_remap = np.asarray([id_to_idx[int(v)] for v in cond[mask]], dtype=int)
    return x[mask], y[mask], None if br is None else br[mask], cond_remap


def _build_mwl_classification_loaders(
    *,
    subject: str,
    subjects: List[str],
    data_str: str,
    batch_size: int,
    data_dir: str,
    train_data_group: str,
    test_data_group: str,
    task: str,
    include_levels_in_train: bool,
    class_subset: str,
    class_groups: Optional[Sequence[Tuple[str, Sequence[int]]]] = None,
    val_subjects: int = 0,
    seed: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader, Dict[str, object]]:
    if str(task) == "mr_levels":
        train_groups, test_groups = _groups_for_task(task, include_levels_in_train)
    else:
        train_groups, test_groups = [train_data_group], [test_data_group]
        if _infer_task_from_group(train_data_group) != _infer_task_from_group(test_data_group):
            raise ValueError(
                "Downstream classification requires matching train/test tasks unless task=mr_levels: "
                f"train_data_group={train_data_group}, test_data_group={test_data_group}."
            )

    if class_groups is not None:
        class_names = [str(name) for name, _ids in class_groups]
        class_original_ids = [list(map(int, ids)) for _name, ids in class_groups]
    else:
        class_names, flat_ids = _parse_mwl_class_subset(class_subset)
        if not flat_ids:
            flat_ids = list(_target_classes(task))
            inverse = {int(v): k for k, v in CLASS_LABEL_TO_ID.items()}
            class_names = [inverse.get(int(v), str(v)) for v in flat_ids]
        class_groups = [(name, [orig_id]) for name, orig_id in zip(class_names, flat_ids)]
        class_original_ids = [list(map(int, ids)) for _name, ids in class_groups]

    source_subjects = [sbj for sbj in subjects if sbj != subject]
    n_val_subjects = max(0, min(int(val_subjects), max(0, len(source_subjects) - 1)))
    rng = np.random.default_rng(int(seed))
    shuffled_source = list(source_subjects)
    rng.shuffle(shuffled_source)
    val_subject_list = sorted(shuffled_source[:n_val_subjects])
    train_subjects = sorted(shuffled_source[n_val_subjects:])

    x_train, y_train, br_train, cond_train = _load_classification_arrays(
        train_subjects, train_groups, data_str, data_dir, task
    )
    val_loader = None
    x_val = None
    y_val = None
    br_val = None
    cond_val = None
    if val_subject_list:
        x_val, y_val, br_val, cond_val = _load_classification_arrays(
            val_subject_list, train_groups, data_str, data_dir, task
        )
    x_test, y_test, br_test, cond_test = _load_classification_arrays(
        [subject], test_groups, data_str, data_dir, task
    )
    x_train, y_train, br_train, cond_train = _remap_mwl_class_groups(
        x_train, y_train, br_train, cond_train, class_groups
    )
    if x_val is not None:
        x_val, y_val, br_val, cond_val = _remap_mwl_class_groups(
            x_val, y_val, br_val, cond_val, class_groups
        )
    x_test, y_test, br_test, cond_test = _remap_mwl_class_groups(
        x_test, y_test, br_test, cond_test, class_groups
    )
    if x_train.shape[0] == 0:
        raise RuntimeError(
            f"No MWL training windows found for subset={class_names} "
            f"(held-out {subject}, train_groups={train_groups})."
        )
    if x_test.shape[0] == 0:
        raise RuntimeError(f"No MWL test windows found for subject {subject}, subset={class_names}.")

    train_loader = DataLoader(
        LoadDataset(x_train, y_train, cond_train, br_train, aug_ratio=0.0),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    if x_val is not None and x_val.shape[0] > 0:
        val_loader = DataLoader(
            LoadDataset(x_val, y_val, cond_val, br_val, aug_ratio=0.0),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
    test_loader = DataLoader(
        LoadDataset(x_test, y_test, cond_test, br_test, aug_ratio=0.0),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    meta: Dict[str, object] = {
        "class_names": class_names,
        "class_original_ids": class_original_ids,
        "class_id_map": {name: idx for idx, name in enumerate(class_names)},
        "train_groups": train_groups,
        "test_groups": test_groups,
        "train_subjects": train_subjects,
        "val_subjects": val_subject_list,
        "n_train_windows": int(x_train.shape[0]),
        "n_val_windows": int(0 if x_val is None else x_val.shape[0]),
        "n_test_windows": int(x_test.shape[0]),
        "uses_source_subject_validation": int(bool(val_subject_list)),
    }
    return train_loader, val_loader, test_loader, meta


class TCNBlock(nn.Module):
    """Residual same-length temporal convolution block."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = ((int(kernel_size) - 1) * int(dilation)) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation, padding=pad),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TCNProfileFiLMMWLHead(nn.Module):
    """TCN classifier head with profile-FiLM over token and pooled features.

    Input hidden tokens: (B, T_tokens, D)
    Profile vector:      (B, P) or (1, P)
    Output logits:       (B, num_classes)
    """

    def __init__(
        self,
        d_model: int,
        profile_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
        film_scale: float = 0.1,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.profile_dim = int(profile_dim)
        self.num_classes = int(num_classes)
        self.in_proj = nn.Conv1d(self.d_model, int(hidden_dim), kernel_size=1)
        blocks = []
        for i in range(int(num_layers)):
            blocks.append(TCNBlock(int(hidden_dim), kernel_size=kernel_size, dilation=2 ** i, dropout=dropout))
        self.tcn = nn.Sequential(*blocks)
        self.token_film = ProfileFiLM(self.profile_dim, int(hidden_dim), scale=float(film_scale))
        self.pooled_film = ProfileFiLM(self.profile_dim, int(hidden_dim), scale=float(film_scale))
        self.classifier = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim), self.num_classes),
        )

    def _expand_profile(self, profile: Optional[torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        if profile is None:
            return torch.zeros(batch_size, self.profile_dim, device=device)
        if profile.ndim == 1:
            profile = profile.unsqueeze(0)
        if profile.size(0) == 1 and batch_size > 1:
            profile = profile.expand(batch_size, -1)
        if profile.size(0) != batch_size:
            raise ValueError(f"Profile batch {profile.size(0)} does not match hidden batch {batch_size}.")
        return profile

    def forward(self, hidden: torch.Tensor, profile: Optional[torch.Tensor] = None) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(f"Expected hidden (B,T,D), got {tuple(hidden.shape)}")
        p = self._expand_profile(profile, hidden.size(0), hidden.device)
        h = self.in_proj(hidden.transpose(1, 2))      # (B,H,T)
        h = self.tcn(h).transpose(1, 2)               # (B,T,H)
        h = self.token_film(h, p)
        pooled = h.mean(dim=1)
        pooled = self.pooled_film(pooled, p)
        return self.classifier(pooled)


class PooledLinearMWLHead(nn.Module):
    """A0: mean-pooled hidden tokens followed by one linear classifier."""

    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.num_classes = int(num_classes)
        self.classifier = nn.Linear(int(d_model), self.num_classes)

    def forward(self, hidden: torch.Tensor, profile: Optional[torch.Tensor] = None) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(f"Expected hidden (B,T,D), got {tuple(hidden.shape)}")
        return self.classifier(hidden.mean(dim=1))


class PooledMLPMWLHead(nn.Module):
    """A1: mean-pooled hidden tokens followed by a small MLP."""

    def __init__(self, d_model: int, num_classes: int, hidden_dim: int = 32, dropout: float = 0.4):
        super().__init__()
        self.num_classes = int(num_classes)
        self.net = nn.Sequential(
            nn.LayerNorm(int(d_model)),
            nn.Linear(int(d_model), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.num_classes),
        )

    def forward(self, hidden: torch.Tensor, profile: Optional[torch.Tensor] = None) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(f"Expected hidden (B,T,D), got {tuple(hidden.shape)}")
        return self.net(hidden.mean(dim=1))


class TinyTCNMWLHead(nn.Module):
    """A2: tiny token TCN without profile-FiLM."""

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        hidden_dim: int = 32,
        num_layers: int = 1,
        kernel_size: int = 3,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.in_proj = nn.Conv1d(int(d_model), int(hidden_dim), kernel_size=1)
        self.tcn = nn.Sequential(
            *[
                TCNBlock(int(hidden_dim), kernel_size=kernel_size, dilation=2 ** i, dropout=dropout)
                for i in range(int(num_layers))
            ]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.num_classes),
        )

    def forward(self, hidden: torch.Tensor, profile: Optional[torch.Tensor] = None) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(f"Expected hidden (B,T,D), got {tuple(hidden.shape)}")
        h = self.in_proj(hidden.transpose(1, 2))
        h = self.tcn(h).transpose(1, 2)
        return self.classifier(h.mean(dim=1))


def _build_mwl_head(head_variant: str, args, num_classes: int) -> nn.Module:
    variant = str(head_variant).strip()
    if variant == "A0":
        return PooledLinearMWLHead(int(args.d_model), int(num_classes))
    if variant == "A1":
        return PooledMLPMWLHead(
            int(args.d_model),
            int(num_classes),
            hidden_dim=int(args.mwl_tcn_hidden_dim),
            dropout=float(args.mwl_dropout),
        )
    if variant == "A2":
        return TinyTCNMWLHead(
            int(args.d_model),
            int(num_classes),
            hidden_dim=int(args.mwl_tcn_hidden_dim),
            num_layers=int(args.mwl_tcn_layers),
            kernel_size=int(args.mwl_tcn_kernel_size),
            dropout=float(args.mwl_dropout),
        )
    raise ValueError(f"Unsupported MWL head variant {head_variant!r}")


@torch.no_grad()
def _forward_source_profile_hidden(model: nn.Module, imu: torch.Tensor, device: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return hidden/profile using the same source-time optional profile path."""
    pred_logmag, rr_aux, hidden = model(imu.to(device, non_blocking=True).float())
    if not bool(getattr(model, "use_profile_conditioning", getattr(model, "use_profile_film", False))):
        return hidden, None
    _stats_raw, stats_norm, _diag = _profile_stats_from_unlabeled_batch(model, imu, device)
    p = model.profile_encoder(stats_norm.unsqueeze(0)).detach()
    mode = str(getattr(model, "profile_conditioning", "none")).lower().strip()
    if mode not in {"film", "qkv"}:
        mode = "film" if bool(getattr(model, "use_profile_film", False)) else "qkv"
    _pred, _rr, hidden_cond, p_used = model.forward_profile_conditioned(
        imu.to(device, non_blocking=True).float(),
        profile_vector=p,
        conditioning_mode=mode,
    )
    return hidden_cond, p_used.detach()


def _train_rr_probe_for_ttt(model: nn.Module, train_loader, device: str, args) -> FaithfulRRRegressor:
    x_source, y_source, _ = collect_rr_arrays(
        model,
        train_loader,
        device,
        max_batches=int(getattr(args, "rr_probe_source_batches", 0)),
        include_kinematics=False,
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
    return rr_model


def _adapt_profile_and_get_hidden(
    model: nn.Module,
    rr_model: FaithfulRRRegressor,
    imu: torch.Tensor,
    mode: str,
    device: str,
    args,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
    """Apply one independent episodic TTT step/batch and return final hidden/profile.

    Target MWL labels are not used here; only the unlabeled IMU batch and the
    RR/STFT self-supervised losses from the QKV/FiLM TTT path are used.
    """
    mode = str(mode).lower().strip()
    if mode == "none":
        hidden, p = _forward_source_profile_hidden(model, imu, device)
        return hidden.detach(), p, {"ttt_loss": 0.0, "profile_delta_norm": 0.0}

    conditioning_mode = TTT_PROFILE_MODES[mode]
    if getattr(model, "profile_encoder", None) is None:
        raise RuntimeError("Profile TTT requested, but model.profile_encoder is missing.")

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    rr_model.eval()

    _raw, stats_norm, stats_diag = _profile_stats_from_unlabeled_batch(model, imu, device)
    with torch.no_grad():
        p0 = model.profile_encoder(stats_norm.unsqueeze(0)).detach()

    p_ttt = nn.Parameter(p0.detach().clone())
    opt = torch.optim.AdamW([p_ttt], lr=float(args.ttt_lr), weight_decay=float(args.ttt_weight_decay))
    last_parts: Dict[str, float] = {}
    for _ in range(1, int(args.ttt_inner_steps) + 1):
        opt.zero_grad(set_to_none=True)
        rr_probe, _z, rr_stft, rr_aux, conf = _profile_conditioned_rr_probe(
            model, rr_model, imu, p_ttt, device, conditioning_mode
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
        p_eval = p_ttt.detach()
        if p_eval.ndim == 1:
            p_eval = p_eval.unsqueeze(0)
        if p_eval.size(0) == 1 and imu.size(0) > 1:
            p_eval = p_eval.expand(imu.size(0), -1)
        p0_eval = p0.detach()
        if p0_eval.ndim == 1:
            p0_eval = p0_eval.unsqueeze(0)
        if p0_eval.size(0) == 1 and imu.size(0) > 1:
            p0_eval = p0_eval.expand(imu.size(0), -1)
        _pred_base, _rr_base, hidden_base, _p_base = model.forward_profile_conditioned(
            imu.to(device, non_blocking=True).float(),
            profile_vector=p0_eval,
            conditioning_mode=conditioning_mode,
        )
        _pred, _rr, hidden, p_used = model.forward_profile_conditioned(
            imu.to(device, non_blocking=True).float(),
            profile_vector=p_eval,
            conditioning_mode=conditioning_mode,
        )
    diag = dict(stats_diag)
    diag.update(last_parts)
    diag["profile_delta_norm"] = float((p_ttt.detach() - p0).norm(p=2).cpu())
    diag["hidden_delta_norm"] = float((hidden.detach() - hidden_base.detach()).norm(p=2).cpu())
    diag["hidden_delta_rms"] = float((hidden.detach() - hidden_base.detach()).pow(2).mean().sqrt().cpu())
    diag["profile_conditioning_mode"] = conditioning_mode
    return hidden.detach(), p_used.detach(), diag


def train_mwl_head(
    model: nn.Module,
    head: TCNProfileFiLMMWLHead,
    loader,
    val_loader,
    device: str,
    args,
) -> Dict[str, float]:
    """Train only the TCN Profile-FiLM MWL head on source-subject labels."""
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    head.train()
    head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=float(args.mwl_lr), weight_decay=float(args.mwl_weight_decay))

    def _score(split_loader, prefix: str) -> Dict[str, float]:
        head.eval()
        losses, preds, labels = [], [], []
        with torch.no_grad():
            for batch in split_loader:
                imu, _pressure, cond, _br, _tlx = unpack_batch(batch, device)
                y = cond.long().view(-1)
                hidden, profile = _forward_source_profile_hidden(model, imu, device)
                logits = head(hidden, profile)
                loss = F.cross_entropy(logits, y)
                losses.append(float(loss.detach().cpu()))
                preds.append(logits.detach().argmax(dim=1).cpu().numpy())
                labels.append(y.detach().cpu().numpy())
        yy = np.concatenate(labels) if labels else np.array([], dtype=int)
        pp = np.concatenate(preds) if preds else np.array([], dtype=int)
        return {
            f"{prefix}_loss": float(np.mean(losses)) if losses else float("nan"),
            f"{prefix}_acc": float(accuracy_score(yy, pp)) if yy.size else float("nan"),
            f"{prefix}_balanced_acc": float(balanced_accuracy_score(yy, pp)) if yy.size else float("nan"),
            f"{prefix}_f1_macro": float(f1_score(yy, pp, average="macro", zero_division=0)) if yy.size else float("nan"),
        }

    monitor = str(getattr(args, "mwl_monitor", "val_f1_macro"))
    maximize = not monitor.endswith("_loss")
    best_score = -float("inf") if maximize else float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history = []
    for epoch in range(1, int(args.mwl_epochs) + 1):
        head.train()
        losses, preds, labels = [], [], []
        for batch in loader:
            imu, _pressure, cond, _br, _tlx = unpack_batch(batch, device)
            y = cond.long().view(-1)
            with torch.no_grad():
                hidden, profile = _forward_source_profile_hidden(model, imu, device)
            logits = head(hidden, profile)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.mwl_grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), float(args.mwl_grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu()))
            preds.append(logits.detach().argmax(dim=1).cpu().numpy())
            labels.append(y.detach().cpu().numpy())
        yy = np.concatenate(labels) if labels else np.array([], dtype=int)
        pp = np.concatenate(preds) if preds else np.array([], dtype=int)
        row = {
            "epoch": int(epoch),
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "acc": float(accuracy_score(yy, pp)) if yy.size else float("nan"),
            "balanced_acc": float(balanced_accuracy_score(yy, pp)) if yy.size else float("nan"),
            "f1_macro": float(f1_score(yy, pp, average="macro", zero_division=0)) if yy.size else float("nan"),
        }
        if val_loader is not None:
            row.update(_score(val_loader, "val"))
        else:
            row.update(
                {
                    "val_loss": float("nan"),
                    "val_acc": float("nan"),
                    "val_balanced_acc": float("nan"),
                    "val_f1_macro": float("nan"),
                }
            )

        current = float(row.get(monitor, float("nan")))
        improved = False
        if math.isfinite(current):
            improved = current > best_score if maximize else current < best_score
        if improved:
            best_score = current
            best_epoch = int(epoch)
            stale_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            stale_epochs += 1
        history.append(row)
        if epoch == 1 or epoch == int(args.mwl_epochs) or epoch % max(1, int(args.mwl_log_every)) == 0:
            print(
                f"[MWL_HEAD] epoch={epoch} loss={row['loss']:.4f} acc={row['acc']:.4f} "
                f"f1={row['f1_macro']:.4f} val_loss={row['val_loss']:.4f} "
                f"val_f1={row['val_f1_macro']:.4f}"
            )
        if (
            bool(getattr(args, "mwl_early_stop", True))
            and val_loader is not None
            and stale_epochs >= int(getattr(args, "mwl_patience", 8))
        ):
            print(
                f"[MWL_HEAD] early_stop epoch={epoch} best_epoch={best_epoch} "
                f"{monitor}={best_score:.4f}"
            )
            break

    if best_state is not None:
        head.load_state_dict({k: v.to(device) for k, v in best_state.items()}, strict=True)

    last = history[-1] if history else {"loss": float("nan"), "acc": float("nan"), "f1_macro": float("nan")}
    out = dict(last)
    if best_epoch > 0:
        best_row = next((row for row in history if int(row["epoch"]) == best_epoch), {})
        out.update({f"best_{k}": v for k, v in best_row.items() if k != "epoch"})
    out["best_epoch"] = int(best_epoch)
    out["best_monitor"] = monitor
    out["best_monitor_value"] = float(best_score) if math.isfinite(best_score) else float("nan")
    out["epochs_ran"] = int(len(history))
    return out


@torch.no_grad()
def _predict_mwl_mode(
    model: nn.Module,
    head: nn.Module,
    rr_model: Optional[FaithfulRRRegressor],
    loader,
    mode: str,
    device: str,
    args,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    model.eval()
    head.eval()
    source_state = _state_dict_cpu_clone(model)

    probs_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    pred_all: List[np.ndarray] = []
    batch_rows: List[Dict[str, float]] = []
    for bi, batch in enumerate(loader):
        _restore_state_dict(model, source_state, device)
        for p in model.parameters():
            p.requires_grad = False
        imu, _pressure, cond, _br, _tlx = unpack_batch(batch, device)
        y = cond.long().view(-1).detach().cpu().numpy()

        if mode == "none":
            hidden, profile = _forward_source_profile_hidden(model, imu, device)
            diag = {"ttt_loss": 0.0, "profile_delta_norm": 0.0}
        else:
            if rr_model is None:
                raise RuntimeError("MWL profile TTT mode requires a trained RR probe.")
            hidden, profile, diag = _adapt_profile_and_get_hidden(model, rr_model, imu, mode, device, args)

        logits = head(hidden, profile)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        pred = probs.argmax(axis=1).astype(int)
        probs_all.append(probs)
        pred_all.append(pred)
        labels_all.append(y)
        batch_rows.append(
            {
                "batch_idx": int(bi),
                "mwl_mode": str(mode),
                "n_batch": int(len(y)),
                "batch_acc": float(accuracy_score(y, pred)) if len(y) else float("nan"),
                "mean_confidence": float(np.max(probs, axis=1).mean()) if probs.size else float("nan"),
                **{k: float(v) for k, v in diag.items() if isinstance(v, (int, float, np.floating))},
            }
        )

    _restore_state_dict(model, source_state, device)
    y_np = np.concatenate(labels_all, axis=0) if labels_all else np.array([], dtype=int)
    pred_np = np.concatenate(pred_all, axis=0) if pred_all else np.array([], dtype=int)
    prob_np = np.concatenate(probs_all, axis=0) if probs_all else np.zeros((0, head.num_classes), dtype=np.float32)
    return y_np, pred_np, prob_np, pd.DataFrame(batch_rows)


def _postprocess_1min(
    y_true: np.ndarray,
    prob: np.ndarray,
    *,
    seconds: float,
    window_shift_seconds: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    prob = np.asarray(prob, dtype=np.float32)
    if y_true.size == 0:
        return y_true, y_true, prob, pd.DataFrame()
    windows_per_chunk = max(1, int(round(float(seconds) / max(1e-6, float(window_shift_seconds)))))
    rows = []
    y_chunk, pred_chunk, prob_chunk = [], [], []
    for chunk_id, st in enumerate(range(0, y_true.size, windows_per_chunk)):
        en = min(y_true.size, st + windows_per_chunk)
        yy = y_true[st:en]
        pp = prob[st:en]
        p_mean = pp.mean(axis=0)
        # Majority ground-truth label for a 1-minute decision; ties pick the lower class id.
        counts = np.bincount(yy, minlength=prob.shape[1])
        y_maj = int(counts.argmax())
        pred = int(p_mean.argmax())
        y_chunk.append(y_maj)
        pred_chunk.append(pred)
        prob_chunk.append(p_mean)
        rows.append(
            {
                "chunk_id": int(chunk_id),
                "start_window": int(st),
                "end_window_exclusive": int(en),
                "n_windows": int(en - st),
                "true_majority": int(y_maj),
                "pred_1min": int(pred),
                "correct_1min": int(pred == y_maj),
                "confidence_1min": float(p_mean.max()),
            }
        )
    return (
        np.asarray(y_chunk, dtype=int),
        np.asarray(pred_chunk, dtype=int),
        np.vstack(prob_chunk).astype(np.float32),
        pd.DataFrame(rows),
    )


def _classification_metrics(y: np.ndarray, pred: np.ndarray, prefix: str) -> Dict[str, float]:
    y = np.asarray(y).astype(int).reshape(-1)
    pred = np.asarray(pred).astype(int).reshape(-1)
    if y.size == 0:
        return {f"{prefix}_acc": float("nan"), f"{prefix}_balanced_acc": float("nan"), f"{prefix}_f1_macro": float("nan"), f"{prefix}_n": 0}
    return {
        f"{prefix}_acc": float(accuracy_score(y, pred)),
        f"{prefix}_balanced_acc": float(balanced_accuracy_score(y, pred)),
        f"{prefix}_f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        f"{prefix}_n": int(y.size),
    }


def mwl_post_eval_hook(model, sbj: str, subjects: List[str], train_loader, test_loader, device: str, args, sbj_dir: Path):
    if not bool(getattr(args, "run_mwl_head", True)):
        return []
    _warm_lazy_modules(model, train_loader, device)
    full_subjects = list(getattr(args, "full_subjects_for_loso", None) or subjects)

    modes = _parse_modes(args.mwl_ttt_modes)
    has_qkv = any(m.startswith("profile_qkv_") for m in modes)
    rr_model = None
    if any(m != "none" for m in modes):
        rr_model = _train_rr_probe_for_ttt(model, train_loader, device, args)
    qkv_scale_diag = _apply_qkv_effective_scale(model, args)

    rows: List[Dict[str, float | str | int]] = []
    mwl_task = "mr_levels"
    task_specs = _parse_mwl_diagnostic_tasks(str(getattr(args, "mwl_diagnostic_tasks", "")))
    head_variants = _parse_mwl_head_variants(str(getattr(args, "mwl_head_variants", "")))

    for task_spec in task_specs:
        task_name = str(task_spec["name"])
        class_groups = task_spec["class_groups"]
        cls_train_loader, cls_val_loader, cls_test_loader, cls_meta = _build_mwl_classification_loaders(
            subject=sbj,
            subjects=full_subjects,
            data_str=args.data_str,
            batch_size=int(args.mwl_batch_size),
            data_dir=str(args.data_dir),
            train_data_group="mr_levels",
            test_data_group="mr_levels",
            task=mwl_task,
            include_levels_in_train=True,
            class_subset="",
            class_groups=class_groups,
            val_subjects=int(getattr(args, "mwl_val_subjects", 3)),
            seed=int(getattr(args, "seed", 0)),
        )
        class_names = [str(v) for v in cls_meta["class_names"]]
        num_classes = len(class_names)

        for head_variant in head_variants:
            head = _build_mwl_head(head_variant, args, num_classes).to(device)
            out_root = sbj_dir / "mwl_diagnostics" / task_name / head_variant
            out_root.mkdir(parents=True, exist_ok=True)
            train_last = train_mwl_head(model, head, cls_train_loader, cls_val_loader, device, args)
            torch.save(
                {
                    "head_state_dict": head.state_dict(),
                    "args": vars(args),
                    "subject": sbj,
                    "mwl_diagnostic_task": task_name,
                    "mwl_head_variant": head_variant,
                    "class_names": class_names,
                },
                out_root / "mwl_head.pt",
            )

            for requested_mode in modes:
                requested_mode = str(requested_mode).strip().lower()
                effective_mode = requested_mode
                gate_info: Dict[str, float | int | str] = {}
                if requested_mode.startswith("profile_qkv_") and rr_model is not None:
                    # The gate uses RR calibration windows only; it does not use MWL labels.
                    try:
                        from vit_pressure_crossmodal_stft_rr_unsup_ladder_config_sweep import _collect_tensor_windows
                        from vit_pressure_crossmodal_stft_rr_core import split_target_calibration_eval
                        x_target, y_target, _ = collect_rr_arrays(model, test_loader, device, max_batches=0, include_kinematics=False)
                        _x_cal, _y_cal, _, _x_eval, _y_eval, _, cal_idx = split_target_calibration_eval(
                            x_target, y_target, None, int(args.target_calibration_windows), seed=int(args.seed),
                            mode=str(args.target_calibration_mode), exclude_calibration_from_eval=bool(args.exclude_calibration_from_eval),
                        )
                        target_windows = _collect_tensor_windows(model, test_loader, device, max_windows=0)
                        cal_windows = target_windows.subset(cal_idx)
                        gate_info = evaluate_qkv_calibration_gate(model, rr_model, cal_windows, device, args)
                        if int(gate_info.get("qkv_cal_gate_pass", 1)) == 0 and str(args.profile_qkv_cal_gate_fallback).lower() in {"base", "none"}:
                            effective_mode = "none"
                    except Exception as e:
                        gate_info = {"qkv_cal_gate_error": str(e), "qkv_cal_gate_pass": 1}

                mode_dir = out_root / requested_mode
                mode_dir.mkdir(parents=True, exist_ok=True)
                y, pred, prob, batch_df = _predict_mwl_mode(model, head, rr_model, cls_test_loader, effective_mode, device, args)
                y_1m, pred_1m, prob_1m, chunks_df = _postprocess_1min(
                    y,
                    prob,
                    seconds=float(args.mwl_postprocess_seconds),
                    window_shift_seconds=float(args.mwl_window_shift_seconds),
                )

                win_metrics = _classification_metrics(y, pred, "mwl_window")
                min_metrics = _classification_metrics(y_1m, pred_1m, "mwl_1min")
                row: Dict[str, float | str | int] = {
                    "__summary_name__": "mwl_diagnostic_summary",
                    "subject": sbj,
                    "mwl_diagnostic_task": task_name,
                    "mwl_requested_mode": requested_mode,
                    "mwl_effective_mode": effective_mode,
                    "mwl_head_variant": head_variant,
                    "mwl_head": {"A0": "pooled_linear", "A1": "pooled_mlp", "A2": "tiny_tcn"}[head_variant],
                    "mwl_task": mwl_task,
                    "mwl_num_classes": int(num_classes),
                    "mwl_class_subset": ",".join(class_names),
                    "mwl_class_original_ids": json.dumps(list(cls_meta["class_original_ids"])),
                    "mwl_class_id_map": json.dumps(cls_meta["class_id_map"]),
                    "mwl_train_groups": " ".join(str(v) for v in cls_meta["train_groups"]),
                    "mwl_test_groups": " ".join(str(v) for v in cls_meta["test_groups"]),
                    "mwl_train_subjects": " ".join(str(v) for v in cls_meta["train_subjects"]),
                    "mwl_val_subjects": " ".join(str(v) for v in cls_meta["val_subjects"]),
                    "mwl_train_windows": int(cls_meta["n_train_windows"]),
                    "mwl_val_windows": int(cls_meta["n_val_windows"]),
                    "mwl_test_windows": int(cls_meta["n_test_windows"]),
                    "mwl_uses_source_subject_validation": int(cls_meta["uses_source_subject_validation"]),
                    "mwl_postprocess_seconds": float(args.mwl_postprocess_seconds),
                    "mwl_windows_per_chunk": int(round(float(args.mwl_postprocess_seconds) / max(1e-6, float(args.mwl_window_shift_seconds)))),
                    "mwl_train_loss_last": float(train_last.get("loss", float("nan"))),
                    "mwl_train_acc_last": float(train_last.get("acc", float("nan"))),
                    "mwl_train_f1_macro_last": float(train_last.get("f1_macro", float("nan"))),
                    "mwl_val_loss_last": float(train_last.get("val_loss", float("nan"))),
                    "mwl_val_acc_last": float(train_last.get("val_acc", float("nan"))),
                    "mwl_val_f1_macro_last": float(train_last.get("val_f1_macro", float("nan"))),
                    "mwl_best_epoch": int(train_last.get("best_epoch", 0)),
                    "mwl_epochs_ran": int(train_last.get("epochs_ran", 0)),
                    "mwl_best_monitor": str(train_last.get("best_monitor", "")),
                    "mwl_best_monitor_value": float(train_last.get("best_monitor_value", float("nan"))),
                    "mwl_best_val_f1_macro": float(train_last.get("best_val_f1_macro", float("nan"))),
                    "mwl_best_val_loss": float(train_last.get("best_val_loss", float("nan"))),
                    **win_metrics,
                    **min_metrics,
                    **qkv_scale_diag,
                    **gate_info,
                }
                rows.append(row)

                pred_df = pd.DataFrame({"window_idx": np.arange(y.size), "y_true": y, "y_pred": pred})
                for c, class_name in enumerate(class_names):
                    pred_df[f"prob_{class_name}"] = prob[:, c]
                pred_df.to_csv(mode_dir / "mwl_window_predictions.csv", index=False)
                chunks_df.to_csv(mode_dir / "mwl_1min_predictions.csv", index=False)
                batch_df.to_csv(mode_dir / "mwl_ttt_batches.csv", index=False)
                with open(mode_dir / "mwl_metrics.json", "w") as f:
                    json.dump({k: v for k, v in row.items() if not k.startswith("__")}, f, indent=2)
                print(
                    f"[MWL] subject={sbj} task={task_name} head={head_variant} mode={requested_mode} "
                    f"effective={effective_mode} window_acc={win_metrics['mwl_window_acc']:.4f} "
                    f"one_min_acc={min_metrics['mwl_1min_acc']:.4f}"
                )

    return rows


def add_mwl_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--eval-subjects",
        nargs="+",
        default=None,
        help=(
            "Held-out subject(s) to evaluate in this process while preserving "
            "the full --subjects cohort for LOSO source training."
        ),
    )
    parser.add_argument("--run-mwl-head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mwl-task", default="mr_levels", choices=list(TASK_CHOICES))
    parser.add_argument("--mwl-train-data-group", default="levels", choices=["mr", "levels", "mr_levels"])
    parser.add_argument("--mwl-test-data-group", default="levels", choices=["mr", "levels", "mr_levels"])
    parser.add_argument("--mwl-include-levels-in-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--mwl-class-subset",
        default="",
        help="Legacy comma-separated MWL labels to classify. Ignored when --mwl-diagnostic-tasks is set.",
    )
    parser.add_argument(
        "--mwl-diagnostic-tasks",
        default="levels,binary_low_high,rest_vs_load",
        help="Comma-separated diagnostics: levels,binary_low_high,rest_vs_load.",
    )
    parser.add_argument(
        "--mwl-head-variants",
        default="A0,A1,A2",
        help="Comma-separated simple heads: A0 pooled linear, A1 pooled MLP, A2 tiny no-profile TCN.",
    )
    parser.add_argument("--mwl-ttt-modes", default="none profile_qkv_ttt_sample")
    parser.add_argument("--mwl-batch-size", type=int, default=64)
    parser.add_argument("--mwl-epochs", type=int, default=100)
    parser.add_argument("--mwl-lr", type=float, default=3e-4)
    parser.add_argument("--mwl-weight-decay", type=float, default=1e-2)
    parser.add_argument("--mwl-grad-clip", type=float, default=1.0)
    parser.add_argument("--mwl-log-every", type=int, default=10)
    parser.add_argument("--mwl-tcn-hidden-dim", type=int, default=32)
    parser.add_argument("--mwl-tcn-layers", type=int, default=1)
    parser.add_argument("--mwl-tcn-kernel-size", type=int, default=3)
    parser.add_argument("--mwl-dropout", type=float, default=0.4)
    parser.add_argument("--mwl-profile-film-scale", type=float, default=0.1)
    parser.add_argument("--mwl-early-stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mwl-val-subjects", type=int, default=3)
    parser.add_argument("--mwl-patience", type=int, default=8)
    parser.add_argument("--mwl-monitor", default="val_f1_macro", choices=["val_f1_macro", "val_balanced_acc", "val_acc", "val_loss"])
    parser.add_argument("--mwl-postprocess-seconds", type=float, default=60.0)
    parser.add_argument("--mwl-window-shift-seconds", type=float, default=1.0)


def _add_argument_if_absent(parser: argparse.ArgumentParser, *name_or_flags, **kwargs):
    for flag in name_or_flags:
        if isinstance(flag, str) and flag.startswith("-") and flag in parser._option_string_actions:
            return None
    return parser.add_argument(*name_or_flags, **kwargs)


def add_ttt_args(parser: argparse.ArgumentParser) -> None:
    add = _add_argument_if_absent
    # These match the latest QKV/episodic TTT runner defaults.
    add(parser, "--ttt-batch-size", type=int, default=1)
    add(parser, "--ttt-inner-steps", type=int, default=5)
    add(parser, "--ttt-lr", type=float, default=3e-4)
    add(parser, "--ttt-weight-decay", type=float, default=0.0)
    add(parser, "--ttt-probe-stft-weight", type=float, default=0.05)
    add(parser, "--ttt-aux-stft-weight", type=float, default=0.05)
    add(parser, "--ttt-profile-prior-weight", type=float, default=0.01)
    add(parser, "--ttt-smoothness-weight", type=float, default=0.0)
    add(parser, "--ttt-stft-confidence-floor", type=float, default=0.0)
    add(parser, "--ttt-stft-confidence-power", type=float, default=2.0)
    add(parser, "--ttt-rr-disagreement-threshold", type=float, default=12.0)
    add(parser, "--ttt-rr-min-bpm", type=float, default=4.0)
    add(parser, "--ttt-rr-max-bpm", type=float, default=45.0)
    add(parser, "--ttt-max-temporal-jump-bpm", type=float, default=10.0)
    add(parser, "--profile-qkv-residual", action="store_true")
    add(parser, "--profile-qkv-alpha-init", type=float, default=1.0)
    add(parser, "--profile-qkv-alpha-max", type=float, default=1.0)
    add(parser, "--profile-qkv-calibration-gate", action="store_true")
    add(parser, "--profile-qkv-cal-gate-tolerance", type=float, default=0.05)
    add(parser, "--profile-qkv-cal-gate-metric", default="mae", choices=["mae", "rmse"])
    add(parser, "--profile-qkv-cal-gate-fallback", default="base", choices=["base", "none"])


def config_mutator(args) -> None:
    modes = _parse_modes(args.mwl_ttt_modes)
    has_film = any("profile_film_" in m for m in modes)
    has_qkv = any("profile_qkv_" in m for m in modes)
    if has_film and has_qkv:
        raise SystemExit("Do not mix profile_film_* and profile_qkv_* modes in one source checkpoint.")
    if has_qkv:
        args.use_profile_qkv = True
        args.use_profile_film = False
        args.profile_conditioning = "qkv"
        args.profile_qkv_layers = str(getattr(args, "profile_qkv_layers", "last1") or "last1")
    elif has_film:
        args.use_profile_film = True
        args.use_profile_qkv = False
        args.profile_conditioning = "film"
    if (has_film or has_qkv) and int(getattr(args, "profile_stats_dim", 0)) <= 0:
        args.profile_stats_dim = profile_stats_dim(int(args.d_model))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_base_parser(SUBJECTS, "vit_profile_tcn_mwl_qkv_ttt_1min")
    add_common_adaptation_args(parser)
    add_ttt_args(parser)
    add_mwl_args(parser)
    args = parser.parse_args(list(argv) if argv is not None else None)

    full_subjects = list(getattr(args, "subjects", []) or [])
    eval_subjects = list(getattr(args, "eval_subjects", None) or full_subjects)
    if len(full_subjects) < 2:
        raise SystemExit(
            "Need at least two subjects in --subjects because it defines the full LOSO source cohort. "
            "For subject-level scheduling, pass held-out subjects via --eval-subjects."
        )
    _patch_loocv_generator_for_eval_subjects(eval_subjects, full_subjects)
    args.full_subjects_for_loso = full_subjects
    args.subjects = eval_subjects

    run_loocv_experiment(args, post_eval_hooks=[mwl_post_eval_hook], config_mutator=config_mutator)


if __name__ == "__main__":
    main()
