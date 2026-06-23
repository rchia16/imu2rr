#!/usr/bin/env python
"""
Minimal benchmark for MWL downstream classification from IMU/RR features.

Pretrained checkpoints may come from meditation/rest or PSS/RR runs, but MWL
classifiers train and test on L0-L3 labels so class ids remain comparable.

Modes:
  - raw_rr:   LDA on raw RR (BioHarness BR if present, else RR derived from PSS waveform)
  - model_rr: LDA on model-predicted RR (from a frozen checkpoint)
  - embed:    linear classifier on frozen model embeddings
  - imu_mwl:  direct IMU -> MWL classifier baseline (supervised; no pretrained runs)
  - imu_hr_mwl: direct IMU+BioHarness HR -> MWL classifier baseline
  - embed_rr: regression head on frozen embeddings to predict RR (choose embedding source via --embed-source)

Special-case:
  - normwear: always freezes the feature extractor and trains only the final linear layer.

The default backbone list matches analysis_shift_panels.py DEFAULT_MODELS.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score

from dataloader import load_data, make_dataset, LoadDataset
from config import SBJ_PROCESSED_DIR, M_DIR, BR_FS, ECG_FS, IMU_FS, IMU_ISSUES
from digitalsignalprocessing import get_max_freq, get_peaks
from utils import (
    RunInfo,
    find_run_yamls_by_model_layout,
    load_yaml,
    infer_method,
    infer_backbone,
    extract_run_timestamp,
    parse_run_prefix_flags,
    resolve_best_ckpt,
)

# Split-local label ids from preprocess.py label encoders:
#   mr:     M/R -> {0, 1}
#   levels: L0-L3 -> {0, 1, 2, 3}
COND_MR = (0, 1)
COND_LEVELS = (0, 1, 2, 3)
TASK_CHOICES = ("mr", "levels", "mr_levels")
CLASS_LABEL_TO_ID = {"M": 0, "R": 1, "L0": 2, "L1": 3, "L2": 4, "L3": 5}
ACTIVE_CLASS_SUBSET: Optional[Tuple[int, ...]] = None

# Mirrors analysis_shift_panels.py DEFAULT_MODELS, plus chronos2 and normwear appended.
# Keep normwear last (slowest / heaviest).
DEFAULT_BACKBONES = ["vit", "timesnet", "limu", "primus", "limu_bert_x", "unihar", "chronos2", "normwear"]

METHOD_ALIASES = {
    # Canonical method names used by run metadata/run folders.
    "flow": "flow",
    "flow+cmt": "flow_cmt",
    "flow+ssa": "flow_ssa",
    "flow+cmt+ssa": "flow_ssa_cmt",
    "flow+ssa+cmt": "flow_ssa_cmt",
    "ssa+cmt": "ssa_cmt",
    "cmt+ssa": "ssa_cmt",
    "cmt": "cmt",
    "ssa": "ssa",
    # no adaptation
    "baseline": "baseline",
    # br downstream, together with rr_downstream
    "baseline_pretrain": "baseline_pretrain", 
    # pretrain pss: together with pss_pretrain
    "pretrain": "pretrain",
}

# Matches extract_normwear_embeddings.py defaults (kept here to avoid importing that script).
DEFAULT_NORMWEAR_EMB_ROOT = "/projects/BLVMob/imu-rr-seated/Data/NormWear"

EMBED_SOURCES = ("rr_pretrain", "pss_pretrain", "rr_downstream")
INVALID_METHOD_TOKENS = {"embed", "raw_rr", "model_rr", "imu_mwl", "embed_rr"}


def _default_subjects() -> List[str]:
    return [f"S{str(i).zfill(2)}" for i in range(12, 31) if i not in IMU_ISSUES]


def _resolve_methods_arg(methods_arg: str) -> List[str]:
    if not methods_arg:
        return []
    return [m.strip() for m in methods_arg.split(",") if m.strip()]


def _method_to_key(method: str) -> str:
    m = method.strip().lower()
    return METHOD_ALIASES.get(m, m.replace("+", "_"))


def _validate_methods_for_mode(methods: List[str]) -> None:
    # Prevent common misuse: passing mode names (embed/raw_rr/...) as method keys.
    bad = [m for m in methods if m.strip().lower() in INVALID_METHOD_TOKENS]
    if bad:
        raise ValueError(
            "Invalid --methods value(s): "
            f"{bad}. These are mode names, not run methods. "
            "Use method keys like baseline, flow, flow_ssa, flow_cmt, flow_ssa_cmt."
        )


def _parse_class_subset(s: str) -> Optional[Tuple[int, ...]]:
    if not s:
        return None
    out = []
    for raw in s.split(","):
        label = raw.strip().upper()
        if not label:
            continue
        if label not in CLASS_LABEL_TO_ID:
            valid = ",".join(CLASS_LABEL_TO_ID)
            raise ValueError(f"Unsupported class label '{raw}'. Use comma-separated labels from: {valid}")
        out.append(CLASS_LABEL_TO_ID[label])
    if not out:
        return None
    return tuple(dict.fromkeys(out))


def _filter_split(
    x: np.ndarray,
    y: np.ndarray,
    br: Optional[np.ndarray],
    cond: np.ndarray,
    keep: Tuple[int, ...],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    mask = np.isin(cond.astype(int), np.asarray(keep, dtype=int))
    x2 = x[mask]
    y2 = y[mask]
    cond2 = cond[mask]
    br2 = None if br is None else br[mask]
    return x2, y2, br2, cond2


def _relabel_levels_to_zero_based(y: np.ndarray) -> np.ndarray:
    """
    Levels are already encoded as {0,1,2,3} by the split-local levels encoder.
    Keep this helper as a single normalization hook for cached legacy labels.
    """
    arr = np.asarray(y).astype(int, copy=True)
    if arr.size and arr.min() >= 2 and arr.max() <= 5:
        arr = arr - 2
    return arr


def _is_levels_group(data_group: str) -> bool:
    return str(data_group).lower() in {"level", "levels"}


def _target_name(task: str) -> str:
    if task == "mr_levels":
        return "M/R+L0-L3"
    return "L0-L3" if _is_levels_group(task) else "M/R"


def _canonical_class_labels(cond: np.ndarray, data_group: str, task: str) -> np.ndarray:
    arr = np.asarray(cond).astype(int, copy=True)
    if task == "mr_levels":
        if _is_levels_group(data_group):
            return _relabel_levels_to_zero_based(arr) + 2
        return arr
    if _is_levels_group(task):
        return _relabel_levels_to_zero_based(arr)
    return arr


def _target_classes(task: str) -> Tuple[int, ...]:
    if ACTIVE_CLASS_SUBSET is not None:
        return ACTIVE_CLASS_SUBSET
    if task == "mr_levels":
        return (0, 1, 2, 3, 4, 5)
    return COND_LEVELS if _is_levels_group(task) else COND_MR


def _infer_task_from_group(data_group: str) -> str:
    return "levels" if _is_levels_group(data_group) else "mr"


def _filter_classification_split(
    x: np.ndarray,
    y: np.ndarray,
    br: Optional[np.ndarray],
    cond: np.ndarray,
    data_group: str,
    task: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    cond_norm = _canonical_class_labels(cond, data_group, task)
    keep = np.asarray(_target_classes(task), dtype=int)
    mask = np.isin(cond_norm.astype(int), keep)
    x2 = x[mask]
    y2 = y[mask]
    br2 = None if br is None else br[mask]
    return x2, y2, br2, cond_norm[mask].astype(int)


def _groups_for_task(task: str, include_levels_in_train: bool) -> Tuple[List[str], List[str]]:
    if task == "levels":
        return ["levels"], ["levels"]
    if task == "mr":
        return ["mr"], ["mr"]
    if task == "mr_levels":
        train_groups = ["mr"]
        if include_levels_in_train:
            train_groups.append("levels")
        return train_groups, ["mr", "levels"]
    raise ValueError(f"Unknown downstream task '{task}'")


def _load_classification_arrays(
    subject_list: List[str],
    data_groups: List[str],
    data_str: str,
    data_dir: str,
    task: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    xs, psss, brs, conds = [], [], [], []
    saw_br = False
    for group in data_groups:
        data_list = [load_data(sbj, data_dir=data_dir, data_group=group) for sbj in subject_list]
        x, pss, br, cond = make_dataset(
            data_list, data_str, label_encoder_dir=data_dir, data_group=group
        )
        x, pss, br, cond = _filter_classification_split(x, pss, br, cond, group, task)
        if x.shape[0] == 0:
            continue
        xs.append(x)
        psss.append(pss)
        conds.append(cond)
        brs.append(br)
        saw_br = saw_br or br is not None

    if not xs:
        raise RuntimeError(f"No {_target_name(task)} windows found for groups={data_groups}.")

    x_all = np.concatenate(xs, axis=0)
    pss_all = np.concatenate(psss, axis=0)
    cond_all = np.concatenate(conds, axis=0).astype(int)
    br_all = None
    if saw_br:
        if any(br is None for br in brs):
            raise RuntimeError("Cannot combine splits when only some groups have BR arrays.")
        br_all = np.concatenate(brs, axis=0)
    return x_all, pss_all, br_all, cond_all


def _build_loso_loaders_mr_to_l(
    subject: str,
    subjects: List[str],
    data_str: str,
    batch_size: int,
    data_dir: str,
    train_data_group: str,
    test_data_group: str,
) -> Tuple[DataLoader, DataLoader]:
    """
    Train split: all other subjects, only M/R windows.
    Test split: held-out subject, only L0-L3 windows.
    """
    train_list = [
        load_data(sbj, data_dir=data_dir, data_group=train_data_group)
        for sbj in subjects if sbj != subject
    ]
    test_list = [
        load_data(subject, data_dir=data_dir, data_group=test_data_group)
    ]

    x_train, y_train, br_train, cond_train = make_dataset(
        train_list, data_str, label_encoder_dir=data_dir, 
        data_group=train_data_group
    )
    x_test, y_test, br_test, cond_test = make_dataset(
        test_list, data_str, label_encoder_dir=data_dir, 
        data_group=test_data_group
    )

    x_train, y_train, br_train, cond_train = _filter_split(
        x_train, y_train, br_train, cond_train, COND_MR)
    x_test, y_test, br_test, cond_test = _filter_split(
        x_test, y_test, br_test, cond_test, COND_LEVELS)

    if x_train.shape[0] == 0:
        raise RuntimeError(f"No M/R training windows found for LOSO train " \
                           "split (held-out {subject}).")
    if x_test.shape[0] == 0:
        raise RuntimeError(
            f"No L0-L3 test windows found for subject {subject}."
        )

    train_loader = DataLoader(
        LoadDataset(x_train, y_train, cond_train, br_train, aug_ratio=0.0),
        batch_size=batch_size,
        shuffle=True,
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
    return train_loader, test_loader


def _build_loso_loaders_l_to_l(
    subject: str,
    subjects: List[str],
    data_str: str,
    batch_size: int,
    data_dir: str,
    train_data_group: str,
    test_data_group: str,
    task: Optional[str] = None,
    include_levels_in_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Downstream classification split:
      Train source: all other subjects from the selected split.
      Test source: held-out subject from the selected split.

    The pretrained checkpoint can still come from M/R or PSS runs; only the
    downstream classifier labels must come from the same task on train/test.
    """
    task = task or _infer_task_from_group(train_data_group)
    if task == "mr_levels":
        train_groups, test_groups = _groups_for_task(task, include_levels_in_train)
    else:
        train_groups, test_groups = [train_data_group], [test_data_group]
        if _infer_task_from_group(train_data_group) != _infer_task_from_group(test_data_group):
            raise ValueError(
                "Downstream classification requires matching train/test tasks "
                "unless task=mr_levels: "
                f"train_data_group={train_data_group}, test_data_group={test_data_group}."
            )

    train_subjects = [sbj for sbj in subjects if sbj != subject]
    x_train, y_train, br_train, cond_train = _load_classification_arrays(
        train_subjects, train_groups, data_str, data_dir, task
    )
    x_test, y_test, br_test, cond_test = _load_classification_arrays(
        [subject], test_groups, data_str, data_dir, task
    )

    if x_train.shape[0] == 0:
        raise RuntimeError(
            f"No {_target_name(task)} training windows found for LOSO train split "
            f"(held-out {subject}, train_groups={train_groups})."
        )
    if x_test.shape[0] == 0:
        raise RuntimeError(f"No {_target_name(task)} test windows found for subject {subject}.")

    train_loader = DataLoader(
        LoadDataset(x_train, y_train, cond_train, br_train, aug_ratio=0.0),
        batch_size=batch_size,
        shuffle=True,
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
    return train_loader, test_loader


def _build_loso_imu_arrays_l_to_l(
    subject: str,
    subjects: List[str],
    data_str: str,
    data_dir: str,
    train_data_group: str,
    test_data_group: str,
    task: Optional[str] = None,
    include_levels_in_train: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Supervised IMU classifier baseline (no pretrained runs).
    """
    task = task or _infer_task_from_group(train_data_group)
    if task == "mr_levels":
        train_groups, test_groups = _groups_for_task(task, include_levels_in_train)
    else:
        train_groups, test_groups = [train_data_group], [test_data_group]
        if _infer_task_from_group(train_data_group) != _infer_task_from_group(test_data_group):
            raise ValueError(
                "Downstream classification requires matching train/test tasks "
                "unless task=mr_levels: "
                f"train_data_group={train_data_group}, test_data_group={test_data_group}."
            )

    train_subjects = [sbj for sbj in subjects if sbj != subject]
    x_train, _y_train, _br_train, cond_train = _load_classification_arrays(
        train_subjects, train_groups, data_str, data_dir, task
    )
    x_test, _y_test, _br_test, cond_test = _load_classification_arrays(
        [subject], test_groups, data_str, data_dir, task
    )
    y_train = cond_train.astype(int)
    y_test = cond_test.astype(int)

    if x_train.shape[0] == 0:
        raise RuntimeError(
            f"{subject}: No {_target_name(task)} training windows found for "
            f"LOSO train split (held-out {subject})."
        )
    if x_test.shape[0] == 0:
        raise RuntimeError(f"No {_target_name(task)} test windows found for subject {subject}.")
    return x_train, y_train, x_test, y_test


def _hr_simple_features(aux: np.ndarray) -> np.ndarray:
    """
    One BioHarness HR feature per window.

    Accepts either scalar/window HR arrays or windowed ECG arrays. ECG fallback
    estimates BPM from median peak spacing and leaves failed windows as NaN for
    train-median imputation in the fused LOSO helper.
    """
    A = np.asarray(aux)
    if A.ndim == 1:
        return A.reshape(-1, 1).astype(np.float32, copy=False)
    if A.ndim == 2 and A.shape[1] == 1:
        return A.astype(np.float32, copy=False)

    W = A.reshape(A.shape[0], -1)
    if W.shape[1] <= 120:
        return np.nanmean(W.astype(np.float32), axis=1).reshape(-1, 1)

    out = np.full((W.shape[0], 1), np.nan, dtype=np.float32)
    min_distance = max(1, int(0.3 * ECG_FS))
    for i, w in enumerate(W):
        w = np.asarray(w, dtype=np.float32)
        finite = np.isfinite(w)
        if finite.sum() < 3:
            continue
        wf = w[finite]
        wf = wf - np.nanmedian(wf)
        scale = float(np.nanstd(wf))
        if not np.isfinite(scale) or scale <= 0:
            continue
        peaks, _ = get_peaks(wf, distance=min_distance, height=0.5 * scale)
        if len(peaks) < 2:
            peaks, _ = get_peaks(-wf, distance=min_distance, height=0.5 * scale)
        if len(peaks) < 2:
            continue
        intervals = np.diff(peaks).astype(np.float32) / float(ECG_FS)
        intervals = intervals[(intervals > 0.3) & (intervals < 2.0)]
        if intervals.size == 0:
            continue
        out[i, 0] = np.float32(60.0 / np.median(intervals))
    return out


def _build_loso_imu_hr_arrays_l_to_l(
    subject: str,
    subjects: List[str],
    data_str: str,
    data_dir: str,
    train_data_group: str,
    test_data_group: str,
    task: Optional[str] = None,
    include_levels_in_train: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Supervised IMU+BioHarness HR classifier baseline (no pretrained runs).
    """
    task = task or _infer_task_from_group(train_data_group)
    if task == "mr_levels":
        train_groups, test_groups = _groups_for_task(task, include_levels_in_train)
    else:
        train_groups, test_groups = [train_data_group], [test_data_group]
        if _infer_task_from_group(train_data_group) != _infer_task_from_group(test_data_group):
            raise ValueError(
                "Downstream classification requires matching train/test tasks "
                "unless task=mr_levels: "
                f"train_data_group={train_data_group}, test_data_group={test_data_group}."
            )

    def _collect(subject_list: List[str], groups: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs, conds, auxs = [], [], []
        for group in groups:
            data_list = [load_data(sbj, data_dir=data_dir, data_group=group) for sbj in subject_list]
            x, y, br, cond = make_dataset(
                data_list, data_str, label_encoder_dir=data_dir, data_group=group
            )
            aux_key = "hr" if all("hr" in data for data in data_list) else "ecg_filt"
            if any(aux_key not in data for data in data_list):
                raise RuntimeError(
                    f"Missing BioHarness HR fallback '{aux_key}' for group={group}."
                )
            aux = np.concatenate([data[aux_key] for data in data_list], axis=0)
            if aux.shape[0] != x.shape[0]:
                raise RuntimeError(
                    f"BioHarness aux rows ({aux.shape[0]}) do not match IMU rows "
                    f"({x.shape[0]}) for group={group}."
                )

            cond_norm = _canonical_class_labels(cond, group, task)
            keep = np.asarray(_target_classes(task), dtype=int)
            mask = np.isin(cond_norm.astype(int), keep)
            if mask.any():
                xs.append(x[mask])
                conds.append(cond_norm[mask].astype(int))
                auxs.append(aux[mask])
            _ = y, br

        if not xs:
            raise RuntimeError(f"No {_target_name(task)} windows found for groups={groups}.")
        return (
            np.concatenate(xs, axis=0),
            np.concatenate(conds, axis=0).astype(int),
            np.concatenate(auxs, axis=0),
        )

    train_subjects = [sbj for sbj in subjects if sbj != subject]
    x_train, y_train, aux_train = _collect(train_subjects, train_groups)
    x_test, y_test, aux_test = _collect([subject], test_groups)

    X_train = np.concatenate([_imu_simple_features(x_train), _hr_simple_features(aux_train)], axis=1)
    X_test = np.concatenate([_imu_simple_features(x_test), _hr_simple_features(aux_test)], axis=1)

    med = np.nanmedian(X_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    for X in (X_train, X_test):
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(med, np.where(bad)[1])

    return X_train.astype(np.float32, copy=False), y_train, X_test.astype(np.float32, copy=False), y_test


def _imu_simple_features(x: np.ndarray) -> np.ndarray:
    """
    Minimal IMU features: per-channel mean and std over time.
    Accepts windows shaped (N,T,C) or (N,C,T).
    """
    X = np.asarray(x)
    if X.ndim != 3:
        raise RuntimeError(f"Expected IMU windows with 3 dims, got shape={X.shape}")
    # Heuristic: if axis 1 is small, assume (N,C,T) and swap.
    if X.shape[1] in (2, 3, 6, 9, 12) and X.shape[2] > X.shape[1]:
        X = np.transpose(X, (0, 2, 1))
    mu = X.mean(axis=1)
    sd = X.std(axis=1)
    return np.concatenate([mu, sd], axis=1).astype(np.float32, copy=False)


def _run_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=1,
        multi_class="auto",
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None
    return preds, probs


def _rr_target_from_loader(loader: DataLoader) -> np.ndarray:
    """
    Build an RR target in bpm from the loader.

    Preference:
      1) If BR is present and looks scalar -> treat as RR directly.
      2) If BR is present and looks like a waveform -> compute RR via max freq.
      3) Else fall back to PSS waveform -> compute RR via max freq.
    """
    ys = []
    for _xb, pss, _cond, br in loader:
        rr = None
        if br is not None:
            br_arr = br.detach().cpu().numpy()
            br_arr = np.asarray(br_arr)
            # If already a scalar per-window (B,) or (B,1), treat as RR directly.
            if br_arr.ndim <= 2 and br_arr.reshape(br_arr.shape[0], -1).shape[1] == 1:
                rr = br_arr.reshape(br_arr.shape[0], 1).astype(np.float32)
            else:
                # Assume waveform (B, T_br)
                rr = _rr_from_waveform(br_arr.reshape(br_arr.shape[0], -1)).reshape(-1, 1)
        if rr is None:
            rr = _rr_from_waveform(pss.detach().cpu().numpy()).reshape(-1, 1)
        ys.append(rr)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    return y


def _run_ridge_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    reg = Ridge(alpha=1.0, random_state=0)
    reg.fit(X_train, y_train.ravel())
    pred = reg.predict(X_test).astype(np.float32)
    return pred.reshape(-1, 1)


def _run_matches_embed_source(run: RunInfo, source: str) -> bool:
    cfg = run.cfg or {}
    extra = cfg.get("extra", {}) if isinstance(cfg.get("extra", {}), dict) else {}
    y_str = str(extra.get("y_str", "") or "").strip()
    task_pretrain = str(extra.get("task_pretrain", "") or "").strip()

    if source == "pss_pretrain":
        return y_str == "pss"
    if source == "rr_downstream":
        return y_str == "br" and task_pretrain == "pss"
    if source == "rr_pretrain":
        # Direct RR training (skip PSS): y_str=br and task_pretrain=none/empty.
        return y_str == "br" and (task_pretrain in {"", "none", "null"})
    raise ValueError(f"Unknown embed source '{source}'")


def _collect_stage_runs(
    root: Path,
    models: List[str],
    known_methods: List[str],
    data_str: str,
    embed_source: str,
    data_group: Optional[str] = None,
) -> Dict[Tuple[str, str, str], RunInfo]:
    """
    Stage-aware run discovery across vit/times layout.
    Returns latest run per (method, backbone, subject), filtered by embed_source stage.
    """
    out: Dict[Tuple[str, str, str], RunInfo] = {}
    if not root.exists():
        return out

    group_norm = None if data_group is None else str(data_group).lower()

    for run_yaml in find_run_yamls_by_model_layout(root, models):
        run_dir = run_yaml.parent
        cfg = load_yaml(run_yaml) or {}
        extra = cfg.get("extra", {}) if isinstance(cfg.get("extra", {}), dict) else {}

        subject = str(cfg.get("subject", "")).strip() or run_dir.parent.name
        method = infer_method(str(cfg.get("method", "")), run_dir, known_methods=known_methods)
        backbone = infer_backbone(str(cfg.get("backbone", "")), run_dir, known_models=models)
        run_data_str = str(cfg.get("data_str", "")).strip()
        run_group = str(extra.get("data_group", "")).strip().lower()

        if not subject or backbone not in models:
            continue
        if run_data_str and run_data_str != data_str:
            continue
        if group_norm is not None and run_group and run_group != group_norm:
            continue

        # Candidate run that passed backbone/data split filtering.
        candidate = RunInfo(
            run_dir=run_dir,
            run_yaml=run_yaml,
            run_ts=extract_run_timestamp(run_yaml),
            cfg=cfg,
            method=method,
            backbone=backbone,
            subject=subject,
        )
        if not _run_matches_embed_source(candidate, embed_source):
            continue

        key = (method, backbone, subject)
        prev = out.get(key)
        if prev is None or candidate.run_ts > prev.run_ts:
            out[key] = candidate

    return out


def _select_run_with_fallback(
    runs: Dict[Tuple[str, str, str], RunInfo],
    method_key: str,
    backbone: str,
    subject: str,
) -> Tuple[RunInfo, bool]:
    # Preferred order:
    # 1) exact requested method
    # 2) baseline for same subject/backbone
    # 3) newest stage-matching run for same subject/backbone
    exact = runs.get((method_key, backbone, subject))
    if exact is not None:
        return exact, False

    baseline = runs.get(("baseline", backbone, subject))
    if baseline is not None:
        return baseline, True

    cands = [
        r for (m, b, s), r in runs.items()
        if b == backbone and s == subject
    ]
    if cands:
        chosen = sorted(cands, key=lambda r: r.run_ts, reverse=True)[0]
        return chosen, True

    raise RuntimeError(
        f"No stage-matching run for subject={subject}, backbone={backbone}. "
        "Verify --root, --data-str, --embed-source, and data-group alignment."
    )


def _rr_from_waveform(pred_wave: np.ndarray) -> np.ndarray:
    return np.asarray([get_max_freq(w, fs=BR_FS) * 60.0 for w in pred_wave], dtype=np.float32)


def _collect_raw_rr(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    for _xb, pss, cond, br in loader:
        if br is not None:
            rr = br.view(br.size(0), -1)
        else:
            rr = _rr_from_waveform(pss.detach().cpu().numpy()).reshape(-1, 1)
        if torch.is_tensor(rr):
            arr = rr.detach().cpu().numpy()
        else:
            arr = np.asarray(rr)
        feats.append(arr)
        labels.append(cond.detach().cpu().numpy())
    X = np.concatenate(feats, axis=0)
    y = np.concatenate(labels, axis=0).astype(int)
    y = _relabel_levels_to_zero_based(y)
    return X, y


def _compute_rr_from_model_output(output) -> Optional[np.ndarray]:
    if isinstance(output, (tuple, list)):
        output = output[0]
    if torch.is_tensor(output):
        arr = output.detach().cpu().numpy()
    else:
        arr = np.asarray(output)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return _rr_from_waveform(arr)
    return None


def _load_model_from_run(
    run: RunInfo,
    device: str,
    warmup_batch: Optional[Tuple[torch.Tensor, ...]] = None,
    ckpt_prefix_override: Optional[str] = None,
    ckpt_path_override: Optional[Path] = None,
):
    cfg = run.cfg or {}
    extra = cfg.get("extra", {}) if isinstance(cfg.get("extra", {}), dict) else {}
    data_str = str(cfg.get("data_str", "imu_filt"))
    window_size = int(cfg.get("window_size", 0) or extra.get("window_size", 0) or 0)
    backbone = run.backbone

    ckpt_prefix = str(ckpt_prefix_override or extra.get("ckpt_prefix", "") or "pretrain")
    ckpt_path = Path(ckpt_path_override) if ckpt_path_override else resolve_best_ckpt(run.run_dir, ckpt_prefix=ckpt_prefix)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found for {run.run_dir}")

    flags = parse_run_prefix_flags(run.run_dir)

    if backbone in {"timesnet", "primus", "limu", "imu2clip", "limu_bert_x", "unihar", "normwear", "chronos2"}:
        from times_experiment import build_times_model

        n_channels = 6 if data_str == "imu_filt" else 2
        d_model = int(extra.get("d_model", 128))
        use_flow = bool(int(extra.get("flow", 0))) or bool(int(extra.get("cmt", 0)))
        encoder_init = str(extra.get("encoder_init", "external"))
        encoder_ckpt = extra.get("encoder_ckpt_path", None)

        model = build_times_model(
            backbone,
            window_size=window_size,
            n_channels=n_channels,
            d_model=d_model,
            device=device,
            use_flow=use_flow,
            encoder_init=encoder_init,
            encoder_ckpt=encoder_ckpt,
            extra=extra,
        ).to(device)

        state = torch.load(str(ckpt_path), map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    if backbone == "vit":
        from vit_experiment import IMU2ChestSpectroViT

        n_channels = 6 if data_str == "imu_filt" else 2
        pred_len = int(window_size / IMU_FS) * BR_FS if window_size else None

        def _get_cfg(name: str, default):
            v = extra.get(name, default)
            return default if v is None else v

        d_model = int(_get_cfg("d_model", 128))
        nhead = int(_get_cfg("nhead", 8))
        num_layers = int(_get_cfg("num_layers", 4))
        dim_feedforward = int(_get_cfg("dim_feedforward", 256))
        dropout = float(_get_cfg("dropout", 0.1))
        n_fft = int(_get_cfg("n_fft", 256))
        hop_length = int(_get_cfg("hop_length", 64))
        win_length = int(_get_cfg("win_length", 256))
        mae_decoder_layers = int(_get_cfg("mae_decoder_layers", 2))

        use_band_head = bool(int(_get_cfg("use_band_head", 1)))
        use_rr_head = bool(int(_get_cfg("use_rr_head", 1)))
        use_classifier = bool(int(_get_cfg("use_classifier", flags.get("cls", 0))))
        use_flow = bool(int(_get_cfg("use_flow", flags.get("flow", 0))))

        num_classes = None
        if use_classifier and warmup_batch is not None:
            try:
                conds = warmup_batch[2]
                if hasattr(conds, "max"):
                    num_classes = int(conds.max().item()) + 1
            except Exception:
                num_classes = None

        model = IMU2ChestSpectroViT(
            input_channels=n_channels,
            d_model=d_model,
            pred_len=(pred_len or 1),
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            use_band_head=use_band_head,
            use_classifier=use_classifier,
            num_classes=num_classes,
            use_flow=use_flow,
            flow_num_layers=4,
            mae_decoder_layers=mae_decoder_layers,
            use_rr_head=use_rr_head,
        ).to(device)

        # Warmup for lazy layers so projections exist before strict load.
        if warmup_batch is not None:
            try:
                warm_x = warmup_batch[0].to(device).float()
                model.eval()
                with torch.no_grad():
                    _ = model(warm_x)
            except Exception:
                pass

        state = torch.load(str(ckpt_path), map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    raise ValueError(f"Unsupported backbone '{backbone}' for {run.run_dir}")


def _collect_model_rr(
    run: RunInfo,
    loader: DataLoader,
    device: str,
    rr_source: str,
    ckpt_prefix_override: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    warmup_batch = next(iter(loader))
    model = _load_model_from_run(
        run, device=device, warmup_batch=warmup_batch, ckpt_prefix_override=ckpt_prefix_override
    )
    feats, labels = [], []

    with torch.no_grad():
        for xb, _pss, cond, _br in loader:
            xb = xb.to(device).float()
            rr_pred = None

            if rr_source in {"auto", "head"} and hasattr(model, "forward_rr"):
                try:
                    rr_pred = model.forward_rr(xb)
                except Exception:
                    rr_pred = None

            if rr_pred is None:
                output = model(xb)
                rr_pred = _compute_rr_from_model_output(output)
                if rr_pred is None:
                    raise RuntimeError("Unable to compute RR from model output")

            if torch.is_tensor(rr_pred):
                rr_pred = rr_pred.detach().cpu().numpy()
            rr_pred = np.asarray(rr_pred).reshape(-1, 1)
            feats.append(rr_pred)
            labels.append(cond.detach().cpu().numpy())

    X = np.concatenate(feats, axis=0)
    y = np.concatenate(labels, axis=0).astype(int)
    y = _relabel_levels_to_zero_based(y)
    return X, y


def _collect_embeddings(
    run: RunInfo,
    loader: DataLoader,
    device: str,
    ckpt_prefix_override: Optional[str] = None,
    ckpt_path_override: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    warmup_batch = next(iter(loader))
    model = _load_model_from_run(
        run,
        device=device,
        warmup_batch=warmup_batch,
        ckpt_prefix_override=ckpt_prefix_override,
        ckpt_path_override=ckpt_path_override,
    )
    feats, labels = [], []

    if run.backbone == "vit":
        from vit_experiment import get_hidden_mean

    # Explicitly freeze feature extractor params. We still extract embeddings under no_grad,
    # but this guarantees we never accidentally fine-tune the backbone (esp. for normwear).
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    with torch.no_grad():
        for xb, _pss, cond, _br in loader:
            xb = xb.to(device).float()
            if run.backbone == "vit":
                _y_hat, _band, _cls, hidden = model(xb)
                emb = get_hidden_mean(hidden)
            else:
                if hasattr(model, "pooled_features"):
                    emb = model.pooled_features(xb)
                elif hasattr(model, "forward_rr"):
                    _rr, hidden = model.forward_rr(xb, return_hidden=True)
                    emb = hidden
                else:
                    _y, hidden = model(xb, return_hidden=True)
                    emb = hidden
            feats.append(emb.detach().cpu().numpy())
            labels.append(cond.detach().cpu().numpy())

    X = np.concatenate(feats, axis=0)
    y = np.concatenate(labels, axis=0).astype(int)
    y = _relabel_levels_to_zero_based(y)
    return X, y


def _load_normwear_cached_subject(
    subject: str,
    emb_root: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load pooled NormWear embeddings previously produced by extract_normwear_embeddings.py.

    Expected files:
      {emb_root}/{subject}/pooled_fp32.npy
      {emb_root}/{subject}/conds.npy
    """
    sdir = emb_root / subject
    pooled = sdir / "pooled_fp32.npy"
    conds = sdir / "conds.npy"
    if not pooled.exists() or not conds.exists():
        raise FileNotFoundError(
            f"Missing NormWear cached embeddings for {subject}. "
            f"Expected {pooled} and {conds}."
        )
    X = np.load(pooled, mmap_mode="r")
    y = np.load(conds, mmap_mode="r")
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] != y.shape[0]:
        raise RuntimeError(f"{subject}: pooled_fp32.npy rows ({X.shape[0]}) != conds.npy rows ({y.shape[0]})")
    return X, y


def _normwear_loso_splits_l_to_l(
    subject: str,
    subjects: List[str],
    train_emb_root: Path,
    test_emb_root: Path,
    data_group: str,
    task: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train_list, y_train_list = [], []
    keep = np.asarray(_target_classes(task), dtype=int)
    for sbj in subjects:
        if sbj == subject:
            continue
        Xs, ys = _load_normwear_cached_subject(sbj, train_emb_root)
        ys = _canonical_class_labels(ys, data_group, task)
        mask = np.isin(ys, keep)
        if mask.any():
            X_train_list.append(Xs[mask])
            y_train_list.append(ys[mask].astype(int))

    X_test, y_test = _load_normwear_cached_subject(subject, test_emb_root)
    y_test = _canonical_class_labels(y_test, data_group, task)
    mask_t = np.isin(y_test, keep)
    X_test = X_test[mask_t]
    y_test = y_test[mask_t].astype(int)

    if not X_train_list:
        raise RuntimeError(f"No {_target_name(task)} cached windows found for NormWear LOSO train split (held-out {subject}).")
    if X_test.shape[0] == 0:
        raise RuntimeError(f"No {_target_name(task)} cached windows found for NormWear subject {subject}.")

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0).astype(int)
    return X_train, y_train, X_test, y_test.astype(int)


class _LinearClassifier(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _run_linear_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    device: str,
    epochs: int,
    lr: float,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    y_train = y_train.astype(int)
    y_test_dummy = np.zeros((X_test.shape[0],), dtype=int)

    in_dim = int(X_train.shape[1])
    num_classes = int(y_train.max()) + 1

    model = _LinearClassifier(in_dim, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)

    model.train()
    for _ in range(int(epochs)):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X_test).float().to(device)
        logits = model(xt)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=1)

    _ = y_test_dummy  # keep signature symmetric with other runners
    return preds, probs


def _run_lda(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    preds = lda.predict(X_test)
    probs = lda.predict_proba(X_test) if hasattr(lda, "predict_proba") else None
    return preds, probs


def main() -> None:
    p = argparse.ArgumentParser(description="IMU/RR -> condition classification LOSO benchmark.")
    p.add_argument("--mode", required=True, choices=["raw_rr", "model_rr", "embed", "imu_mwl", "imu_hr_mwl", "embed_rr"])
    p.add_argument("--methods", type=str, default="baseline,baseline_pretrain,ssa,cmt,ssa_cmt,flow,flow_ssa,flow_cmt,flow_ssa_cmt")
    p.add_argument("--backbone", type=str, default="vit", choices=DEFAULT_BACKBONES)
    p.add_argument("--embed-source", type=str, default="rr_downstream", choices=list(EMBED_SOURCES))
    p.add_argument("--data-str", type=str, default="imu_filt")
    p.add_argument("--train-data-group", type=str, default="mr", choices=["mr", "level", "levels"])
    p.add_argument("--test-data-group", type=str, default="levels", choices=["mr", "level", "levels"])
    # Used for supervised downstream classifier data. Checkpoint discovery
    # still uses --train-data-group to find the requested pretraining runs.
    p.add_argument("--downstream-data-group", type=str, default="levels", choices=["mr", "level", "levels"])
    p.add_argument("--downstream-task", type=str, default="", choices=["", *TASK_CHOICES])
    p.add_argument("--class-subset", type=str, default="", help="Comma-separated class labels to keep, e.g. M,R,L1,L3.")
    p.add_argument("--include-levels-in-train", action="store_true")
    p.add_argument("--root", type=str, default="")
    p.add_argument("--normwear-emb-root", type=str, default=DEFAULT_NORMWEAR_EMB_ROOT)
    p.add_argument("--normwear-train-emb-root", type=str, default="")
    p.add_argument("--normwear-test-emb-root", type=str, default="")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--rr-source", type=str, default="auto", choices=["auto", "head", "waveform"])
    p.add_argument("--clf-epochs", type=int, default=20)
    p.add_argument("--clf-lr", type=float, default=1e-3)
    p.add_argument("--clf-batch-size", type=int, default=256)
    args = p.parse_args()

    methods = _resolve_methods_arg(args.methods)
    _validate_methods_for_mode(methods)
    method_keys = sorted({_method_to_key(m) for m in methods})
    if not method_keys:
        raise ValueError("No methods specified.")
    global ACTIVE_CLASS_SUBSET
    ACTIVE_CLASS_SUBSET = _parse_class_subset(args.class_subset)
    downstream_task = args.downstream_task or _infer_task_from_group(args.downstream_data_group)
    if ACTIVE_CLASS_SUBSET is not None:
        downstream_task = "mr_levels"
        args.include_levels_in_train = True
        if args.mode == "embed" and args.backbone == "normwear":
            raise ValueError("--class-subset with normwear cached embeddings is not supported by this loader path.")

    root = Path(args.root) if args.root else Path(M_DIR) / args.data_str / "loocv"

    runs = {}
    if args.mode not in {"raw_rr", "imu_mwl", "imu_hr_mwl"} and not (args.mode in {"embed", "embed_rr"} and args.backbone == "normwear"):
        # Discovery is stage-driven (embed_source), not hard-gated by requested method.
        # methods still control reporting/eval grouping below.
        group_for_discovery = args.train_data_group if args.mode in {"embed", "embed_rr", "model_rr"} else None
        runs = _collect_stage_runs(
            root=root,
            models=[args.backbone],
            known_methods=method_keys,
            data_str=args.data_str,
            embed_source=args.embed_source,
            data_group=group_for_discovery,
        )
        if not runs:
            raise RuntimeError(f"No runs found under {root} for backbone={args.backbone} methods={method_keys}")

    subjects = _default_subjects()
    normwear_train_emb_root = Path(args.normwear_train_emb_root) if args.normwear_train_emb_root else Path(args.normwear_emb_root)
    normwear_test_emb_root = Path(args.normwear_test_emb_root) if args.normwear_test_emb_root else Path(args.normwear_emb_root)

    rows = []
    # Evaluate each requested method label across LOSO subjects.
    for method in methods:
        method_key = _method_to_key(method)
        all_true, all_pred = [], []

        for sbj in subjects:
            if args.mode == "imu_mwl":
                x_train, y_train, x_test, y_test = _build_loso_imu_arrays_l_to_l(
                    sbj,
                    subjects,
                    data_str=args.data_str,
                    data_dir=SBJ_PROCESSED_DIR,
                    train_data_group=args.downstream_data_group,
                    test_data_group=args.test_data_group,
                    task=downstream_task,
                    include_levels_in_train=args.include_levels_in_train,
                )
                X_train = _imu_simple_features(x_train)
                X_test = _imu_simple_features(x_test)
                preds, _probs = _run_logreg(X_train, y_train, X_test)
            elif args.mode == "imu_hr_mwl":
                X_train, y_train, X_test, y_test = _build_loso_imu_hr_arrays_l_to_l(
                    sbj,
                    subjects,
                    data_str=args.data_str,
                    data_dir=SBJ_PROCESSED_DIR,
                    train_data_group=args.downstream_data_group,
                    test_data_group=args.test_data_group,
                    task=downstream_task,
                    include_levels_in_train=args.include_levels_in_train,
                )
                preds, _probs = _run_logreg(X_train, y_train, X_test)
            elif args.mode == "raw_rr":
                train_loader, test_loader = _build_loso_loaders_l_to_l(
                    sbj,
                    subjects,
                    data_str=args.data_str,
                    batch_size=args.batch_size,
                    data_dir=SBJ_PROCESSED_DIR,
                    train_data_group=args.downstream_data_group,
                    test_data_group=args.test_data_group,
                    task=downstream_task,
                    include_levels_in_train=args.include_levels_in_train,
                )
                X_train, y_train = _collect_raw_rr(train_loader)
                X_test, y_test = _collect_raw_rr(test_loader)
                preds, _probs = _run_lda(X_train, y_train, X_test)
            else:
                if args.mode == "model_rr":
                    train_loader, test_loader = _build_loso_loaders_l_to_l(
                        sbj,
                        subjects,
                        data_str=args.data_str,
                        batch_size=args.batch_size,
                        data_dir=SBJ_PROCESSED_DIR,
                        train_data_group=args.downstream_data_group,
                        test_data_group=args.test_data_group,
                        task=downstream_task,
                        include_levels_in_train=args.include_levels_in_train,
                    )
                    # Model-RR baseline still needs a discovered pretrained run.
                    run, fallback_used = _select_run_with_fallback(
                        runs, method_key, args.backbone, sbj
                    )
                    if fallback_used:
                        print(
                            f"[RUN-FALLBACK] subject={sbj} requested_method={method_key} "
                            f"selected_method={run.method} run={run.run_dir}"
                        )
                    X_train, y_train = _collect_model_rr(run, train_loader, 
                                                         args.device,
                                                         args.rr_source)
                    X_test, y_test = _collect_model_rr(run, test_loader, 
                                                       args.device,
                                                       args.rr_source)
                    preds, _probs = _run_lda(X_train, y_train, X_test)
                elif args.mode == "embed":
                    if args.backbone == "normwear":
                        X_train, y_train, X_test, y_test = \
                                _normwear_loso_splits_l_to_l(
                                    sbj, subjects, normwear_train_emb_root, 
                                    normwear_test_emb_root,
                                    args.downstream_data_group,
                                    downstream_task,
                                )
                    else:
                        # Non-NormWear path: load run checkpoint and extract frozen embeddings.
                        train_loader, test_loader = _build_loso_loaders_l_to_l(
                            sbj,
                            subjects,
                            data_str=args.data_str,
                            batch_size=args.batch_size,
                            data_dir=SBJ_PROCESSED_DIR,
                            train_data_group=args.downstream_data_group,
                            test_data_group=args.test_data_group,
                            task=downstream_task,
                            include_levels_in_train=args.include_levels_in_train,
                        )
                        run, fallback_used = _select_run_with_fallback(runs, method_key, args.backbone, sbj)
                        if fallback_used:
                            print(
                                f"[RUN-FALLBACK] subject={sbj} requested_method={method_key} "
                                f"selected_method={run.method} run={run.run_dir}"
                            )
                        X_train, y_train = _collect_embeddings(
                            run, train_loader, args.device,
                            ckpt_prefix_override=("downstream" if args.embed_source == "rr_downstream" else "pretrain"),
                        )
                        X_test, y_test = _collect_embeddings(
                            run, test_loader, args.device,
                            ckpt_prefix_override=("downstream" if args.embed_source == "rr_downstream" else "pretrain"),
                        )
                    preds, _probs = _run_linear_classifier(
                        X_train, y_train, X_test,
                        device=args.device,
                        epochs=args.clf_epochs,
                        lr=args.clf_lr,
                        batch_size=args.clf_batch_size,
                    )
                else:
                    # embed_rr: train RR regressor on frozen embeddings from chosen source.
                    train_loader, test_loader = _build_loso_loaders_mr_to_l(
                        sbj,
                        subjects,
                        data_str=args.data_str,
                        batch_size=args.batch_size,
                        data_dir=SBJ_PROCESSED_DIR,
                        train_data_group=args.train_data_group,
                        test_data_group=args.test_data_group,
                    )
                    run, fallback_used = _select_run_with_fallback(runs, method_key, args.backbone, sbj)
                    if fallback_used:
                        print(
                            f"[RUN-FALLBACK] subject={sbj} requested_method={method_key} "
                            f"selected_method={run.method} run={run.run_dir}"
                        )
                    X_train, _ycond_tr = _collect_embeddings(
                        run, train_loader, args.device,
                        ckpt_prefix_override=("downstream" if args.embed_source == "rr_downstream" else "pretrain"),
                    )
                    X_test, _ycond_te = _collect_embeddings(
                        run, test_loader, args.device,
                        ckpt_prefix_override=("downstream" if args.embed_source == "rr_downstream" else "pretrain"),
                    )
                    y_train_rr = _rr_target_from_loader(train_loader)
                    y_test_rr = _rr_target_from_loader(test_loader)
                    preds = _run_ridge_regressor(X_train, y_train_rr, X_test)
                    y_test = y_test_rr

            all_true.append(y_test)
            all_pred.append(preds)

        if args.mode == "embed_rr":
            y_true = np.concatenate(all_true).astype(np.float32).reshape(-1)
            y_pred = np.concatenate(all_pred).astype(np.float32).reshape(-1)
            mae = float(np.mean(np.abs(y_true - y_pred)))
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            rows.append({
                "mode": args.mode,
                "method": method_key,
                "backbone": args.backbone,
                "mae": mae,
                "rmse": rmse,
                "n_test": int(y_true.shape[0]),
            })
        else:
            y_true = np.concatenate(all_true).astype(int)
            y_pred = np.concatenate(all_pred).astype(int)
            rows.append({
                "mode": args.mode,
                "method": method_key,
                "backbone": args.backbone,
                "acc": float(accuracy_score(y_true, y_pred)),
                "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
                "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
                "n_test": int(y_true.shape[0]),
            })

    # Minimal stdout report
    for r in rows:
        if r["mode"] == "embed_rr":
            print(
                f"{r['mode']}\t{r['backbone']}\t{r['method']}"
                f"\tmae={r['mae']:.4f}\trmse={r['rmse']:.4f}\tn={r['n_test']}"
            )
        else:
            print(
                f"{r['mode']}\t{r['backbone']}\t{r['method']}"
                f"\tacc={r['acc']:.4f}\tf1_macro={r['f1_macro']:.4f}"
                f"\tf1_weighted={r['f1_weighted']:.4f}\tn={r['n_test']}"
            )


if __name__ == "__main__":
    main()
