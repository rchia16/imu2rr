#!/usr/bin/env python3
"""Build frozen F0+A1 MWL caches and train rest/low/high MWL probes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from config import M_DIR
from dataloader import load_data
from mwl_rest_low_high import (
    assert_no_subject_overlap,
    data_groups_for_rest_low_high,
    map_raw_conditions_to_rest_low_high,
    mapping_manifest,
    parse_list,
    split_source_subjects,
)
from rr_readout_phase_abc_utils import spectral_arrays_from_stft
from rr_readout_phase_abc_experiment import BinAffineDistribution
from rr_spectral_readout_experiment import model_from_checkpoint, subject_checkpoint_path
from run_rr_jbhi_f0_family import train_a1_readout
from vit_pressure_crossmodal_stft_rr_core import make_fold_seed


A1_VARIANTS: Dict[str, Tuple[str, str]] = {
    "a1_rr_only_logreg": ("rr_only", "logreg"),
    "a1_mean_logreg": ("a1_mean", "logreg"),
    "a1_distribution_logreg": ("a1_distribution", "logreg"),
    "a1_stats_logreg": ("a1_stats", "logreg"),
    "a1_latent_logreg": ("a1_latent", "logreg"),
    "a1_distribution_mlp": ("a1_distribution", "mlp"),
    "a1_latent_mlp": ("a1_latent", "mlp"),
    "a1_distribution_activity_mlp": ("a1_distribution_activity", "mlp"),
}

FEATURE_KEY_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "rr_only": ("rr_pred", "model_rr", "rr_probe", "rr", "rr_direct", "soft_spectral_rr", "hard_spectral_rr"),
    "a1_mean": ("a1_mean", "profile_mean", "feature_mean"),
    "a1_distribution": ("a1_distribution", "a1_probs", "spectral_distribution", "prob_distribution", "spectral_probability"),
    "a1_stats": ("a1_stats", "profile_stats", "spectral_stats"),
    "a1_latent": ("a1_latent", "latent", "embedding", "emb", "z", "pooled_hidden"),
    "a1_distribution_activity": ("a1_distribution_activity", "a1_dist_activity", "distribution_activity"),
}
SPECTRAL_STATS_KEYS = (
    "soft_spectral_rr",
    "hard_spectral_rr",
    "entropy",
    "max_probability",
    "top1_top2_gap",
    "peak_width_bpm",
    "spectral_variance",
    "hard_soft_gap_bpm",
    "spectral_hidden_disagreement_bpm",
    "respiratory_band_energy",
)
DEFAULT_SUBJECTS = "S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29"
DEFAULT_CHECKPOINT_ROOT = "/projects/BLVMob/imu-rr-seated/results/profile_clsa_qkv_fseries/runs/F0_static_shared"


@dataclass(frozen=True)
class CacheSplit:
    x: np.ndarray
    y: np.ndarray
    sample_id: np.ndarray
    raw_condition: np.ndarray
    group: np.ndarray
    raw_index: np.ndarray
    path: Path
    key: str


@dataclass(frozen=True)
class MWLFeatureArrays:
    x: np.ndarray
    pressure: np.ndarray
    br: np.ndarray
    y: np.ndarray
    sample_id: np.ndarray
    raw_condition: np.ndarray
    subject_id: np.ndarray
    group: np.ndarray
    raw_index: np.ndarray


class MWLWindowDataset(Dataset):
    def __init__(self, arrays: MWLFeatureArrays) -> None:
        x = arrays.x
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected windows shaped (N,T,C), got {arr.shape}.")
        n = int(arr.shape[0])
        self.imu = torch.from_numpy(arr)
        pressure = np.asarray(arrays.pressure, dtype=np.float32)
        if pressure.shape[0] != n:
            raise ValueError(f"Pressure rows {pressure.shape[0]} != IMU rows {n}.")
        self.pressure = torch.from_numpy(pressure)
        self.conds = torch.from_numpy(np.asarray(arrays.y, dtype=np.int64))
        self.br = torch.from_numpy(np.asarray(arrays.br, dtype=np.float32).reshape(-1))
        self.tlx = torch.zeros((n,), dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.imu.shape[0])

    def __getitem__(self, index: int):
        return self.imu[index], self.pressure[index], self.conds[index], self.br[index], self.tlx[index]


class MWLWindowDatasetWithMeta(Dataset):
    def __init__(self, arrays: MWLFeatureArrays) -> None:
        x = arrays.x
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected windows shaped (N,T,C), got {arr.shape}.")
        n = int(arr.shape[0])
        self.imu = torch.from_numpy(arr)
        pressure = np.asarray(arrays.pressure, dtype=np.float32)
        if pressure.shape[0] != n:
            raise ValueError(f"Pressure rows {pressure.shape[0]} != IMU rows {n}.")
        self.pressure = torch.from_numpy(pressure)
        self.conds = torch.from_numpy(np.asarray(arrays.y, dtype=np.int64))
        self.br = torch.from_numpy(np.asarray(arrays.br, dtype=np.float32).reshape(-1))
        self.tlx = torch.zeros((n,), dtype=torch.float32)
        self.subject_ids = np.asarray(arrays.subject_id, dtype=object).reshape(-1).astype(str)
        self.sample_ids = np.asarray(arrays.sample_id, dtype=object).reshape(-1).astype(str)

    def __len__(self) -> int:
        return int(self.imu.shape[0])

    def __getitem__(self, index: int):
        return (
            self.imu[index],
            self.pressure[index],
            self.conds[index],
            self.br[index],
            self.tlx[index],
            {"subject_id": str(self.subject_ids[index]), "subject_index": int(index), "sample_id": str(self.sample_ids[index])},
        )


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def config_hash(payload: Dict[str, object]) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def split_path(root: Path, seed: int, subject: str, split: str) -> Path:
    candidates = [
        root / f"seed_{seed:03d}" / subject / f"{split}.npz",
        root / f"seed_{seed:03d}" / subject / f"{split}_features.npz",
        root / f"seed_{seed:03d}" / subject / f"{split}_rest_low_high.npz",
        root / subject / f"seed_{seed:03d}" / f"{split}.npz",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing cached feature split for seed={seed} subject={subject} split={split}. Tried: {candidates}")


def cache_paths(root: Path, seed: int, subject: str) -> Dict[str, Path]:
    base = Path(root) / f"seed_{seed:03d}" / subject
    return {"train": base / "train.npz", "val": base / "val.npz", "test": base / "test.npz", "manifest": base / "cache_manifest.json"}


def load_subject_mwl_feature_arrays(
    subject: str,
    *,
    data_dir: str,
    data_str: str,
    pressure_str: str,
    data_groups: Sequence[str],
) -> MWLFeatureArrays:
    xs: List[np.ndarray] = []
    pressures: List[np.ndarray] = []
    brs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    sample_ids: List[np.ndarray] = []
    raw_conditions: List[np.ndarray] = []
    subject_ids: List[np.ndarray] = []
    groups: List[np.ndarray] = []
    raw_indices: List[np.ndarray] = []
    for group in data_groups:
        data = load_data(subject, data_dir=data_dir, data_group=group)
        if data_str not in data:
            raise KeyError(f"{subject} group={group} is missing data array {data_str!r}.")
        if pressure_str not in data:
            raise KeyError(f"{subject} group={group} is missing pressure array {pressure_str!r}.")
        if "br" not in data:
            raise KeyError(f"{subject} group={group} is missing RR labels under 'br'.")
        if "conds" not in data:
            raise KeyError(f"{subject} group={group} is missing raw condition labels under 'conds'.")
        x = np.asarray(data[data_str])
        pressure = np.asarray(data[pressure_str])
        br = np.asarray(data["br"], dtype=np.float32).reshape(-1)
        cond_raw_all = np.asarray(data["conds"], dtype=object).reshape(-1)
        n = int(x.shape[0])
        if pressure.shape[0] != n or br.shape[0] != n or cond_raw_all.shape[0] != n:
            raise ValueError(
                f"{subject} group={group}: mismatched rows x={x.shape[0]} pressure={pressure.shape[0]} "
                f"br={br.shape[0]} conds={cond_raw_all.shape[0]}."
            )
        y, keep, raw_norm = map_raw_conditions_to_rest_low_high(cond_raw_all)
        if not bool(keep.any()):
            continue
        raw_idx = np.flatnonzero(keep).astype(np.int64)
        xs.append(x[keep])
        pressures.append(pressure[keep])
        brs.append(br[keep])
        ys.append(y)
        sample_ids.append(np.asarray([f"{subject}:{group}:{int(i):06d}" for i in raw_idx], dtype=object))
        raw_conditions.append(raw_norm[keep])
        subject_ids.append(np.full(int(raw_idx.size), str(subject), dtype=object))
        groups.append(np.full(int(raw_idx.size), str(group), dtype=object))
        raw_indices.append(raw_idx)
    if not xs:
        raise RuntimeError(f"{subject}: no rest_low_high windows found for groups={list(data_groups)}.")
    return MWLFeatureArrays(
        x=np.concatenate(xs, axis=0),
        pressure=np.concatenate(pressures, axis=0),
        br=np.concatenate(brs, axis=0).astype(np.float32),
        y=np.concatenate(ys, axis=0).astype(np.int64),
        sample_id=np.concatenate(sample_ids, axis=0),
        raw_condition=np.concatenate(raw_conditions, axis=0),
        subject_id=np.concatenate(subject_ids, axis=0),
        group=np.concatenate(groups, axis=0),
        raw_index=np.concatenate(raw_indices, axis=0).astype(np.int64),
    )


def stack_mwl_feature_arrays(
    subjects: Sequence[str],
    *,
    data_dir: str,
    data_str: str,
    pressure_str: str,
    data_groups: Sequence[str],
) -> MWLFeatureArrays:
    parts = [
        load_subject_mwl_feature_arrays(
            subject,
            data_dir=data_dir,
            data_str=data_str,
            pressure_str=pressure_str,
            data_groups=data_groups,
        )
        for subject in subjects
    ]
    return MWLFeatureArrays(
        x=np.concatenate([p.x for p in parts], axis=0),
        pressure=np.concatenate([p.pressure for p in parts], axis=0),
        br=np.concatenate([p.br for p in parts], axis=0).astype(np.float32),
        y=np.concatenate([p.y for p in parts], axis=0).astype(np.int64),
        sample_id=np.concatenate([p.sample_id for p in parts], axis=0),
        raw_condition=np.concatenate([p.raw_condition for p in parts], axis=0),
        subject_id=np.concatenate([p.subject_id for p in parts], axis=0),
        group=np.concatenate([p.group for p in parts], axis=0),
        raw_index=np.concatenate([p.raw_index for p in parts], axis=0).astype(np.int64),
    )


def subset_mwl_feature_arrays(arrays: MWLFeatureArrays, indices: np.ndarray) -> MWLFeatureArrays:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    return MWLFeatureArrays(
        x=arrays.x[idx],
        pressure=arrays.pressure[idx],
        br=arrays.br[idx],
        y=arrays.y[idx],
        sample_id=arrays.sample_id[idx],
        raw_condition=arrays.raw_condition[idx],
        subject_id=arrays.subject_id[idx],
        group=arrays.group[idx],
        raw_index=arrays.raw_index[idx],
    )


def labels_from_npz(data: np.lib.npyio.NpzFile, path: Path) -> np.ndarray:
    for key in ("y", "label", "labels", "true_class", "class_id", "rest_low_high"):
        if key in data.files:
            y = np.asarray(data[key]).astype(np.int64).reshape(-1)
            if not set(np.unique(y).astype(int)).issubset({0, 1, 2}):
                raise ValueError(f"{path}: labels in {key!r} are not rest_low_high ids 0/1/2.")
            return y
    raise KeyError(f"{path}: missing labels. Expected one of y,label,labels,true_class,class_id,rest_low_high.")


def sample_ids_from_npz(data: np.lib.npyio.NpzFile, path: Path, subject: str, split: str, n: int) -> np.ndarray:
    for key in ("sample_id", "sample_ids", "window_id", "window_ids"):
        if key in data.files:
            out = np.asarray(data[key], dtype=object).reshape(-1).astype(str)
            if out.shape[0] != n:
                raise ValueError(f"{path}: {key} length {out.shape[0]} != labels length {n}.")
            return out
    return np.asarray([f"{subject}:{split}:{i:06d}" for i in range(int(n))], dtype=object)


def raw_conditions_from_npz(data: np.lib.npyio.NpzFile, n: int) -> np.ndarray:
    for key in ("raw_condition", "raw_conditions", "condition", "conditions"):
        if key in data.files:
            out = np.asarray(data[key], dtype=object).reshape(-1)
            if out.shape[0] == n:
                return out
    return np.asarray([""] * int(n), dtype=object)


def optional_array_from_npz(data: np.lib.npyio.NpzFile, key: str, n: int, default: object) -> np.ndarray:
    if key in data.files:
        out = np.asarray(data[key]).reshape(-1)
        if out.shape[0] == n:
            return out
    return np.asarray([default] * int(n), dtype=object)


def first_feature_key(data: np.lib.npyio.NpzFile, feature_set: str) -> str:
    if feature_set == "a1_stats":
        present = [key for key in SPECTRAL_STATS_KEYS if key in data.files]
        if present:
            return "+".join(present)
    for key in FEATURE_KEY_CANDIDATES[feature_set]:
        if key in data.files:
            return key
    raise KeyError(
        f"Missing tensor for feature_set={feature_set}. Tried keys={FEATURE_KEY_CANDIDATES[feature_set]}; available={data.files}"
    )


def flatten_features(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected features with batch dimension, got {arr.shape}.")
    return np.ascontiguousarray(arr, dtype=np.float32)


def feature_matrix_from_npz(data: np.lib.npyio.NpzFile, feature_set: str, key: str) -> np.ndarray:
    if "+" in key:
        cols = [flatten_features(np.asarray(data[part])) for part in key.split("+")]
        return np.concatenate(cols, axis=1).astype(np.float32, copy=False)
    return flatten_features(np.asarray(data[key]))


def load_cache_split(root: Path, seed: int, subject: str, split: str, feature_set: str) -> CacheSplit:
    path = split_path(root, seed, subject, split)
    with np.load(path, allow_pickle=True) as data:
        key = first_feature_key(data, feature_set)
        y = labels_from_npz(data, path)
        x = feature_matrix_from_npz(data, feature_set, key)
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"{path}: feature rows {x.shape[0]} != labels {y.shape[0]} for key={key}.")
        sample_id = sample_ids_from_npz(data, path, subject, split, y.shape[0])
        raw_condition = raw_conditions_from_npz(data, y.shape[0])
        group = optional_array_from_npz(data, "group", y.shape[0], "")
        raw_index = optional_array_from_npz(data, "raw_index", y.shape[0], -1).astype(int)
    return CacheSplit(x=x, y=y, sample_id=sample_id, raw_condition=raw_condition, group=group, raw_index=raw_index, path=path, key=key)


def fit_logreg(x_train: np.ndarray, y_train: np.ndarray):
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    multi_class="auto",
                    n_jobs=1,
                    random_state=0,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model


class SmallMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(in_dim)),
            nn.Linear(int(in_dim), int(hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


def predict_sklearn(model, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    prob = model.predict_proba(x).astype(np.float32)
    full = np.zeros((x.shape[0], 3), dtype=np.float32)
    classes = model.named_steps["clf"].classes_.astype(int)
    full[:, classes] = prob
    return full.argmax(axis=1).astype(int), full


def fit_mlp(train: CacheSplit, val: CacheSplit, args: argparse.Namespace, device: torch.device) -> SmallMLP:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train.x).astype(np.float32)
    x_val = scaler.transform(val.x).astype(np.float32)
    model = SmallMLP(x_train.shape[1], int(args.mlp_hidden_dim), float(args.dropout)).to(device)
    counts = np.bincount(train.y, minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / max(float(weights.mean()), 1e-6)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(train.y.astype(np.int64))),
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
    )
    best_state = None
    best_loss = float("inf")
    bad = 0
    xv = torch.from_numpy(x_val).to(device)
    yv = torch.from_numpy(val.y.astype(np.int64)).to(device)
    for _epoch in range(1, int(args.epochs) + 1):
        model.train()
        for batch_i, (xb, yb) in enumerate(loader):
            if int(args.max_train_batches) > 0 and batch_i >= int(args.max_train_batches):
                break
            loss = criterion(model(xb.to(device)), yb.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(xv), yv).detach().cpu())
        if val_loss < best_loss:
            best_loss = val_loss
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= int(args.patience):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model._a1_scaler_mean = scaler.mean_.astype(np.float32)  # type: ignore[attr-defined]
    model._a1_scaler_scale = scaler.scale_.astype(np.float32)  # type: ignore[attr-defined]
    return model


@torch.no_grad()
def predict_mlp(model: SmallMLP, x: np.ndarray, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    mean = getattr(model, "_a1_scaler_mean")
    scale = getattr(model, "_a1_scaler_scale")
    xx = ((x - mean) / np.where(scale < 1e-6, 1.0, scale)).astype(np.float32)
    logits = model(torch.from_numpy(xx).to(device))
    prob = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)
    return prob.argmax(axis=1).astype(int), prob


def simple_loader(arrays: MWLFeatureArrays, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        MWLWindowDataset(arrays),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(num_workers),
    )


def prime_loader(arrays: MWLFeatureArrays, batch_size: int) -> DataLoader:
    return DataLoader(
        MWLWindowDatasetWithMeta(arrays),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )


@torch.no_grad()
def collect_frozen_f0_cache(
    model: nn.Module,
    arrays: MWLFeatureArrays,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    batch_size = int(args.cached_feature_batch_size)
    max_batches = int(args.max_cache_batches)
    logits_rows: List[np.ndarray] = []
    probs_rows: List[np.ndarray] = []
    pooled_rows: List[np.ndarray] = []
    hidden_rows: List[np.ndarray] = []
    rr_direct_rows: List[np.ndarray] = []
    scalar_cols: Dict[str, List[np.ndarray]] = {key: [] for key in SPECTRAL_STATS_KEYS}
    true_classes: List[np.ndarray] = []
    sample_ids: List[np.ndarray] = []
    raw_conditions: List[np.ndarray] = []
    subject_ids: List[np.ndarray] = []
    groups: List[np.ndarray] = []
    raw_indices: List[np.ndarray] = []
    rr_true_rows: List[np.ndarray] = []
    bins_out: Optional[np.ndarray] = None
    for batch_i, st in enumerate(range(0, int(arrays.x.shape[0]), batch_size)):
        if max_batches > 0 and batch_i >= max_batches:
            break
        ed = min(st + batch_size, int(arrays.x.shape[0]))
        imu = torch.from_numpy(np.asarray(arrays.x[st:ed], dtype=np.float32)).to(device)
        pred_logmag, rr_direct, hidden = model(imu)
        hidden_np = hidden.detach().cpu().numpy().astype(np.float32)
        pooled_np = hidden.mean(dim=1).detach().cpu().numpy().astype(np.float32)
        rr_direct_np = rr_direct.detach().cpu().numpy().reshape(-1).astype(np.float32)
        spec = spectral_arrays_from_stft(pred_logmag.detach().cpu().numpy(), soft_temperature=0.1, hidden_rr=rr_direct_np)
        bins_out = spec["frequency_bins_bpm"] if bins_out is None else bins_out
        logits_rows.append(np.asarray(spec["spectral_logits"], dtype=np.float32))
        probs_rows.append(np.asarray(spec["spectral_probability"], dtype=np.float32))
        pooled_rows.append(pooled_np)
        hidden_rows.append(hidden_np)
        rr_direct_rows.append(rr_direct_np)
        for key in SPECTRAL_STATS_KEYS:
            scalar_cols[key].append(np.asarray(spec[key], dtype=np.float32).reshape(-1))
        true_classes.append(np.asarray(arrays.y[st:ed], dtype=np.int64))
        sample_ids.append(np.asarray(arrays.sample_id[st:ed], dtype=object))
        raw_conditions.append(np.asarray(arrays.raw_condition[st:ed], dtype=object))
        subject_ids.append(np.asarray(arrays.subject_id[st:ed], dtype=object))
        groups.append(np.asarray(arrays.group[st:ed], dtype=object))
        raw_indices.append(np.asarray(arrays.raw_index[st:ed], dtype=np.int64))
        rr_true_rows.append(np.asarray(arrays.br[st:ed], dtype=np.float32).reshape(-1))
    if bins_out is None:
        raise RuntimeError("No windows were available for cache generation.")
    probs = np.concatenate(probs_rows, axis=0).astype(np.float32)
    rr_direct_full = np.concatenate(rr_direct_rows, axis=0).astype(np.float32)
    stats_matrix = np.column_stack([np.concatenate(scalar_cols[key], axis=0) for key in SPECTRAL_STATS_KEYS]).astype(np.float32)
    return {
        "y": np.concatenate(true_classes, axis=0).astype(np.int64),
        "true_class": np.concatenate(true_classes, axis=0).astype(np.int64),
        "rest_low_high": np.concatenate(true_classes, axis=0).astype(np.int64),
        "rr_true": np.concatenate(rr_true_rows, axis=0).astype(np.float32),
        "sample_id": np.concatenate(sample_ids, axis=0),
        "raw_condition": np.concatenate(raw_conditions, axis=0),
        "condition": np.concatenate(raw_conditions, axis=0),
        "subject_id": np.concatenate(subject_ids, axis=0),
        "group": np.concatenate(groups, axis=0),
        "raw_index": np.concatenate(raw_indices, axis=0).astype(np.int64),
        "spectral_logits": np.concatenate(logits_rows, axis=0).astype(np.float32),
        "spectral_probability": probs,
        "f0_spectral_probability": probs,
        "frequency_bins_bpm": np.asarray(bins_out, dtype=np.float32),
        "pooled_hidden": np.concatenate(pooled_rows, axis=0).astype(np.float32),
        "attention_hidden": np.concatenate(hidden_rows, axis=0).astype(np.float32),
        "final_hidden_sequence": np.concatenate(hidden_rows, axis=0).astype(np.float32),
        "rr_direct": rr_direct_full,
        "rr_pred": rr_direct_full,
        "a1_latent": np.concatenate(pooled_rows, axis=0).astype(np.float32),
        "a1_stats": stats_matrix,
        **{key: np.concatenate(values, axis=0).astype(np.float32) for key, values in scalar_cols.items()},
    }


@torch.no_grad()
def apply_frozen_a1_readout(cache: Dict[str, np.ndarray], readout: nn.Module, device: torch.device) -> Dict[str, np.ndarray]:
    out = dict(cache)
    logits = torch.as_tensor(out["spectral_logits"], device=device).float()
    prob = readout(logits).detach().cpu().numpy().astype(np.float32)
    out["a1_calibrated_probability"] = prob
    out["a1_distribution"] = prob
    return out


def phase_abc_selected_a1_path(root: Path, seed: int, subject: str, method: str, require: bool = True) -> Path:
    selected_path = root / f"seed_{seed:03d}" / "phase_a_selected_checkpoints.json"
    key = f"{subject}:{method}"
    if selected_path.exists():
        selected = json.loads(selected_path.read_text())
        if key in selected and "checkpoint" in selected[key]:
            path = Path(selected[key]["checkpoint"])
            if path.exists():
                return path
            if require:
                raise FileNotFoundError(f"Selected Phase ABC checkpoint for {key} does not exist: {path}")
    fallback = root / f"seed_{seed:03d}" / "phase_a_checkpoints" / f"seed_{seed:03d}" / subject / method / "best.pt"
    if fallback.exists():
        return fallback
    if require:
        raise FileNotFoundError(
            f"Missing frozen Phase ABC A1 checkpoint for seed={seed} subject={subject} method={method}. "
            f"Tried {selected_path} key={key} and {fallback}."
        )
    return fallback


def load_frozen_a1_readout(path: Path, device: torch.device) -> Tuple[BinAffineDistribution, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state_dict")
    if not isinstance(state, dict):
        raise KeyError(f"{path}: missing model_state_dict.")
    state_keys = set(state)
    if state_keys != {"scale", "bias"}:
        raise RuntimeError(f"{path}: expected Phase ABC BinAffineDistribution keys {{'scale','bias'}}, got {sorted(state_keys)}.")
    n_bins = int(state["scale"].numel())
    readout = BinAffineDistribution(n_bins).to(device)
    readout.load_state_dict(state, strict=True)
    readout.eval()
    for p in readout.parameters():
        p.requires_grad = False
    trainable = int(sum(p.numel() for p in readout.parameters()))
    if trainable != 20:
        raise RuntimeError(f"{path}: expected established A1 20-parameter checkpoint, got {trainable} parameters.")
    return readout, ckpt


def write_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def build_cache_for_subject(seed: int, subject: str, all_subjects: Sequence[str], args: argparse.Namespace, device: torch.device) -> Dict[str, object]:
    fold_seed = make_fold_seed(int(seed), subject)
    set_seed(fold_seed)
    fold_args = argparse.Namespace(**{**vars(args), "seed": int(seed), "fold_seed": int(fold_seed)})
    family = str(args.representation_family)
    train_subjects, val_subjects = split_source_subjects(all_subjects, subject, int(args.val_subjects), fold_seed)
    assert_no_subject_overlap(train_subjects, val_subjects, subject)
    groups = data_groups_for_rest_low_high(args.data_group)
    train = stack_mwl_feature_arrays(
        train_subjects,
        data_dir=args.data_dir,
        data_str=args.data_str,
        pressure_str=args.pressure_str,
        data_groups=groups,
    )
    val = stack_mwl_feature_arrays(
        val_subjects,
        data_dir=args.data_dir,
        data_str=args.data_str,
        pressure_str=args.pressure_str,
        data_groups=groups,
    )
    test = stack_mwl_feature_arrays(
        [subject],
        data_dir=args.data_dir,
        data_str=args.data_str,
        pressure_str=args.pressure_str,
        data_groups=groups,
    )
    checkpoint_loader = prime_loader(train, batch_size=min(int(args.cached_feature_batch_size), max(1, int(train.y.size))))
    ckpt_path = subject_checkpoint_path(Path(args.checkpoint_root), subject, args.checkpoint_name)
    a1_path: Optional[Path] = None
    if family == "frozen_phase_abc_transfer":
        a1_path = phase_abc_selected_a1_path(
            Path(args.phase_abc_root),
            int(seed),
            subject,
            str(args.frozen_a1_method),
            require=bool(args.require_frozen_a1),
        )
    payload = {
        "subject": str(subject),
        "seed": int(seed),
        "fold_seed": int(fold_seed),
        "representation_family": family,
        "data_str": str(args.data_str),
        "pressure_str": str(args.pressure_str),
        "data_group": str(args.data_group),
        "train_subjects": list(train_subjects),
        "val_subjects": list(val_subjects),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_mtime_ns": ckpt_path.stat().st_mtime_ns if ckpt_path.exists() else None,
        "phase_abc_root": str(args.phase_abc_root),
        "frozen_a1_method": str(args.frozen_a1_method),
        "a1_checkpoint_path": "" if a1_path is None else str(a1_path),
        "a1_checkpoint_mtime_ns": None if a1_path is None or not a1_path.exists() else a1_path.stat().st_mtime_ns,
        "cached_feature_batch_size": int(args.cached_feature_batch_size),
        "max_cache_batches": int(args.max_cache_batches),
    }
    paths = cache_paths(Path(args.feature_cache_root), int(seed), subject)
    cfg_hash = config_hash(payload)
    if bool(args.resume) and all(paths[key].exists() for key in ("train", "val", "test", "manifest")):
        existing = json.loads(paths["manifest"].read_text())
        if existing.get("configuration_hash") == cfg_hash:
            return {"subject": subject, "seed": int(seed), "status": "reused", "configuration_hash": cfg_hash}
    model, ckpt, ckpt_args = model_from_checkpoint(ckpt_path, checkpoint_loader, device)
    for p in model.parameters():
        p.requires_grad = False
    train_cache = collect_frozen_f0_cache(model, train, args, device)
    val_cache = collect_frozen_f0_cache(model, val, args, device)
    test_cache = collect_frozen_f0_cache(model, test, args, device)
    a1_ckpt: Dict[str, Any] = {}
    a1_history: List[Dict[str, Any]] = []
    a1_updated = 0
    if family == "frozen_phase_abc_transfer":
        if a1_path is None:
            raise RuntimeError("Frozen Phase ABC transfer requires a resolved A1 checkpoint path.")
        readout, a1_ckpt = load_frozen_a1_readout(a1_path, device)
    elif family == "source_visible_rr_pss_pretrain":
        readout, a1_history = train_a1_readout(train_cache, val_cache, fold_args, str(device))
        a1_updated = int(sum(p.numel() for p in readout.parameters() if p.requires_grad))
        for p in readout.parameters():
            p.requires_grad = False
        readout.eval()
    else:
        raise ValueError(f"Unsupported representation family {family!r}.")
    cached = {
        "train": apply_frozen_a1_readout(train_cache, readout, device),
        "val": apply_frozen_a1_readout(val_cache, readout, device),
        "test": apply_frozen_a1_readout(test_cache, readout, device),
    }
    for split in ("train", "val", "test"):
        write_npz(paths[split], cached[split])
    manifest = {
        "subject": str(subject),
        "seed": int(seed),
        "fold_seed": int(fold_seed),
        "mapping": mapping_manifest(),
        "configuration_hash": cfg_hash,
        "representation_family": family,
        "transfer_mode": family,
        "f0_frozen_during_mwl_probe": True,
        "a1_frozen_during_mwl_probe": True,
        "f0_newly_trained_by_this_runner": False,
        "a1_newly_trained_by_this_runner": family == "source_visible_rr_pss_pretrain",
        "checkpoint_path": str(ckpt_path),
        "f0_checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "f0_checkpoint_sha256": sha256_file(ckpt_path),
        "checkpoint_epoch": ckpt.get("epoch", ""),
        "checkpoint_profile_conditioning": str(ckpt_args.get("profile_conditioning", "")),
        "a1_method": str(args.frozen_a1_method),
        "a1_checkpoint_path": "" if a1_path is None else str(a1_path),
        "a1_checkpoint_sha256": "" if a1_path is None else sha256_file(a1_path),
        "a1_checkpoint_method": str(a1_ckpt.get("method", "")),
        "a1_checkpoint_args": a1_ckpt.get("args", {}),
        "cache_files": {key: str(value) for key, value in paths.items() if key != "manifest"},
        "source_train_subjects": train_subjects,
        "source_val_subjects": val_subjects,
        "held_out_subject": str(subject),
        "readout_class": "BinAffineDistribution",
        "readout_trainable_parameter_count": 20,
        "a1_trainable_parameters_updated_in_mwl": 0,
        "a1_trainable_parameters_updated_before_mwl_probe": int(a1_updated),
        "a1_training_split": "none; loaded frozen Phase ABC checkpoint" if family == "frozen_phase_abc_transfer" else "source train RR labels; source validation for A1 selection",
        "mwl_probe_is_only_trainable_component": True,
        "a1_history": a1_history,
        "tensor_keys": {split: sorted(cached[split].keys()) for split in ("train", "val", "test")},
        "target_labels_in_cache_for_evaluation_only": True,
        "target_labels_used_for_profile": 0,
        "target_labels_used_for_a1_training": 0,
        "target_windows_used_for_a1_training": 0,
        "phase_abc_outputs_modified": False,
        "phase_abc_root": str(args.phase_abc_root),
        "train_windows": int(train.y.size),
        "val_windows": int(val.y.size),
        "test_windows": int(test.y.size),
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, default=str))
    return {"subject": subject, "seed": int(seed), "status": "rebuilt", "configuration_hash": cfg_hash}


def run_cache_stage(args: argparse.Namespace, device: torch.device) -> pd.DataFrame:
    cache_root = Path(args.feature_cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    all_subjects = parse_list(args.subjects)
    held_out_subjects = parse_list(args.held_out_subjects) if str(args.held_out_subjects).strip() else all_subjects
    unknown = sorted(set(held_out_subjects) - set(all_subjects))
    if unknown:
        raise SystemExit(f"--held-out-subjects contains subjects not present in --subjects: {unknown}")
    rows: List[Dict[str, object]] = []
    for seed in [int(v) for v in parse_list(args.seeds)]:
        for subject in held_out_subjects:
            print(f"[A1_CACHE] seed={seed:03d} held_out={subject}", flush=True)
            rows.append(build_cache_for_subject(seed, subject, all_subjects, args, device))
    df = pd.DataFrame(rows)
    if not df.empty:
        for seed in sorted(df["seed"].astype(int).unique()):
            df[df["seed"].astype(int) == int(seed)].to_csv(cache_root / f"cache_seed_{int(seed):03d}_status.csv", index=False)
    return df


def run_variant_for_subject(
    variant: str,
    subject: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Optional[pd.DataFrame], Dict[str, object]]:
    feature_set, classifier = A1_VARIANTS[variant]
    try:
        train = load_cache_split(Path(args.feature_cache_root), seed, subject, "train", feature_set)
        val = load_cache_split(Path(args.feature_cache_root), seed, subject, "val", feature_set)
        test = load_cache_split(Path(args.feature_cache_root), seed, subject, "test", feature_set)
    except Exception as exc:
        return None, {"variant": variant, "subject": subject, "seed": int(seed), "status": "missing_or_invalid", "error": str(exc)}
    if classifier == "logreg":
        model = fit_logreg(train.x, train.y)
        pred, prob = predict_sklearn(model, test.x)
    else:
        model = fit_mlp(train, val, args, device)
        pred, prob = predict_mlp(model, test.x, device)
    out_dir = Path(args.out_dir) / f"seed_{seed:03d}" / variant / subject
    out_dir.mkdir(parents=True, exist_ok=False)
    rows = pd.DataFrame(
        {
            "subject": subject,
            "seed": int(seed),
            "sample_id": test.sample_id.astype(str),
            "true_class": test.y.astype(int),
            "predicted_class": pred.astype(int),
            "prob_rest": prob[:, 0],
            "prob_low": prob[:, 1],
            "prob_high": prob[:, 2],
            "model_name": variant,
            "checkpoint_or_run_path": str(out_dir),
            "raw_condition": test.raw_condition.astype(str),
            "group": test.group.astype(str),
            "raw_index": test.raw_index.astype(int),
        }
    )
    rows.to_csv(out_dir / "mwl_predictions.csv", index=False)
    manifest = {
        "variant": variant,
        "feature_set": feature_set,
        "classifier": classifier,
        "seed": int(seed),
        "subject": subject,
        "mapping": mapping_manifest(),
        "source_cache_files": {"train": str(train.path), "val": str(val.path), "test": str(test.path)},
        "source_cache_sha256": {"train": sha256_file(train.path), "val": sha256_file(val.path), "test": sha256_file(test.path)},
        "tensor_keys": {"train": train.key, "val": val.key, "test": test.key},
        "trainable_scope": "downstream classifier only; cached F0+A1 tensors are read-only.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return rows, {**manifest, "status": "ok", "n_test": int(test.y.size)}


def run_probe_stage(args: argparse.Namespace, device: torch.device) -> pd.DataFrame:
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=False)
    all_subjects = parse_list(args.subjects)
    subjects = parse_list(args.held_out_subjects) if str(args.held_out_subjects).strip() else all_subjects
    unknown = sorted(set(subjects) - set(all_subjects))
    if unknown:
        raise SystemExit(f"--held-out-subjects contains subjects not present in --subjects: {unknown}")
    seeds = [int(v) for v in parse_list(args.seeds)]
    variants = parse_list(args.variants)
    bad = sorted(set(variants) - set(A1_VARIANTS))
    if bad:
        raise SystemExit(f"Unsupported A1 variants: {bad}. Valid={sorted(A1_VARIANTS)}")
    all_rows: List[pd.DataFrame] = []
    manifest_rows: List[Dict[str, object]] = []
    for seed in seeds:
        set_seed(seed)
        for variant in variants:
            for subject in subjects:
                rows, status = run_variant_for_subject(variant, subject, seed, args, device)
                manifest_rows.append(status)
                if rows is not None:
                    all_rows.append(rows)
    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(root / "mwl_predictions.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(root / "run_status.csv", index=False)
    (root / "mwl_a1_probe_manifest.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "mapping": mapping_manifest(),
                "supported_feature_sets": FEATURE_KEY_CANDIDATES,
                "variants": A1_VARIANTS,
                "git_commit": git_commit(),
                "hostname": platform.node(),
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "status_rows": manifest_rows,
            },
            indent=2,
            default=str,
        )
    )
    return pd.DataFrame(manifest_rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen F0+A1 rest/low/high MWL cache builder and probe runner.")
    parser.add_argument("--stage", choices=["cache", "probe", "all"], default="probe")
    parser.add_argument("--feature-cache-root", default="", help="Cache root with seed_000/Sxx/{train,val,test}.npz files.")
    parser.add_argument("--out-dir", default="results/a1_mwl_probe/rest_low_high")
    parser.add_argument("--subjects", default=DEFAULT_SUBJECTS)
    parser.add_argument("--held-out-subjects", default="", help="Optional subset of --subjects to run as LOSO held-out folds.")
    parser.add_argument("--seeds", default="0 1 2")
    parser.add_argument("--variants", default=" ".join(A1_VARIANTS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-train-batches", type=int, default=0, help="Smoke-test limiter; 0 means all batches.")
    parser.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--checkpoint-name", default="best_model.pt")
    parser.add_argument("--data-dir", default="/projects/BLVMob/imu-rr-seated/Data")
    parser.add_argument("--mdl-dir", default=M_DIR)
    parser.add_argument("--data-str", default="imu_filt")
    parser.add_argument("--pressure-str", default="pss_filt")
    parser.add_argument("--data-group", default="mr_levels")
    parser.add_argument(
        "--representation-family",
        choices=["frozen_phase_abc_transfer", "source_visible_rr_pss_pretrain"],
        default="frozen_phase_abc_transfer",
        help="Feature family for MWL probing. Source-visible mode trains A1 on source RR labels and expects --checkpoint-root to identify the F0 checkpoint family.",
    )
    parser.add_argument("--val-subjects", type=int, default=3)
    parser.add_argument("--cached-feature-batch-size", type=int, default=128)
    parser.add_argument("--feature-eval-batch-size", type=int, default=256)
    parser.add_argument("--profile-stats-max-batches", type=int, default=50)
    parser.add_argument("--target-calibration-windows", type=int, default=32)
    parser.add_argument("--target-calibration-mode", choices=["first", "random", "even"], default="first")
    parser.add_argument("--exclude-calibration-from-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-cache-batches", type=int, default=0, help="Smoke-test limiter for cache generation; 0 means all batches.")
    parser.add_argument(
        "--phase-abc-root",
        default="/projects/BLVMob/imu-rr-seated/results/rr_readout_phase_abc",
        help="Read-only root containing established Phase ABC F0+A1 outputs.",
    )
    parser.add_argument("--frozen-a1-method", default="A1_gaussian_kl")
    parser.add_argument("--require-frozen-a1", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("IMU_DATALOADER_WORKERS", "0")))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.stage in {"cache", "all"} and not args.feature_cache_root:
        args.feature_cache_root = str(Path(args.out_dir) / "cache")
    if args.stage == "probe" and not args.feature_cache_root:
        raise SystemExit("--feature-cache-root is required for --stage probe.")
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in str(args.device) else "cpu")
    if args.stage in {"cache", "all"}:
        run_cache_stage(args, device)
        print(f"[DONE] wrote cache root {args.feature_cache_root}", flush=True)
        if args.stage == "cache":
            return
        args = argparse.Namespace(**{**vars(args), "out_dir": str(Path(args.out_dir) / "probe_outputs")})
    run_probe_stage(args, device)
    print(f"[DONE] wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
