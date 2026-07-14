#!/usr/bin/env python3
"""
Profile-conditioned IMU->RR/STFT with focused 3-class MWL heads and
1-/2-minute post-processing.

This is intentionally a separate entrypoint. It keeps the existing RR/STFT
training path unchanged, then adds a downstream mental-workload classifier
trained on source-subject L0-L3 windows and evaluated on the held-out subject.

Key points:
  - Backbone/source training is delegated to vit_pressure_crossmodal_stft_rr_core.
  - The MWL head is a small TCN over encoder tokens, with Profile-FiLM on both
    token and pooled features.
  - The target task is rest_low_high:
      R -> 0, L0/L1 -> 1, L2/L3 -> 2, M excluded.
  - QKV TTT and HRV fusion are optional ablations, not the default path.
  - Window outputs are post-processed into non-overlapping 1-/2-minute chunks
    by averaging class probabilities over consecutive windows.

Example smoke run:
  python vit_profile_tcn_mwl_hierarchical_3class.py \
    --subjects S13 S19 S22 S25 \
    --data-str imu_filt \
    --data-group mr \
    --mwl-train-data-group levels \
    --mwl-test-data-group levels \
    --mwl-task rest_low_high \
    --variants flat_a0 flat_a2 hier_a2 hier_a2_smooth \
    --postprocess-seconds 60 120 \
    --mwl-ttt-modes none \
    --out-dir results/vit_profile_tcn_mwl_hierarchical_3class_smoke
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
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, recall_score
from torch.utils.data import DataLoader, TensorDataset

from config import ECG_FS, SBJ_PROCESSED_DIR
from dataloader import LoadDataset, load_data, make_dataset
from extract_ecg_hrv_features import ecg_window_features
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
    _canonical_class_labels,
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
    return [part.strip() for part in str(text or "").replace(",", " ").split() if part.strip()]


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


AUTONOMIC_KINDS = {
    "hr": ("hr", "HR", "bpm", "e4_hr", "hr_filt"),
    "ibi": ("ibi", "IBI", "rr_intervals"),
    "ppg": ("ppg", "bvp"),
    "ecg": ("ecg_filt", "ecg", "ECG"),
}

HRV_ABSOLUTE_FEATURES = [
    "autonomic_available",
    "hr_mean",
    "hr_median",
    "hr_std",
    "hr_min",
    "hr_max",
    "hr_slope",
    "hr_valid_ratio",
    "ibi_mean",
    "ibi_std",
    "sdnn",
    "rmssd",
    "pnn50",
    "ibi_valid_ratio",
    "hr_from_ibi_mean",
]
HRV_RELATIVE_BASE_FEATURES = [
    "hr_mean",
    "hr_median",
    "hr_std",
    "hr_min",
    "hr_max",
    "hr_slope",
    "ibi_mean",
    "ibi_std",
    "sdnn",
    "rmssd",
    "pnn50",
    "hr_from_ibi_mean",
]


def _parse_hrv_feature_modes(text: str) -> List[str]:
    valid = {"none", "hr_only", "hrv_basic", "hr_hrv_absolute", "hr_hrv_relative", "hr_hrv_absolute_plus_relative"}
    names = _parse_csv_names(text) or ["none", "hr_hrv_relative"]
    bad = [name for name in names if name not in valid]
    if bad:
        raise ValueError(f"Unsupported --hrv-feature-modes entries {bad}. Valid={sorted(valid)}")
    return list(dict.fromkeys(names))


def find_autonomic_source(subject_dict: dict, preferred: str = "auto") -> dict:
    """Return the best available autonomic source descriptor for a subject dict."""
    preferred = str(preferred or "auto").lower().strip()
    if preferred != "auto" and preferred in AUTONOMIC_KINDS:
        kinds = [preferred]
    else:
        kinds = ["hr", "ibi", "ppg", "ecg"]
    for kind in kinds:
        for key in AUTONOMIC_KINDS[kind]:
            if key in subject_dict and subject_dict[key] is not None:
                fs = float(ECG_FS) if kind == "ecg" else None
                return {"kind": kind, "key": key, "fs": fs, "values": subject_dict[key]}
    return {"kind": "none", "key": None, "fs": None, "values": None}


def _empty_autonomic_features() -> Dict[str, float]:
    return {name: 0.0 for name in HRV_ABSOLUTE_FEATURES}


def compute_hrv_features_for_window(
    source: dict,
    start_sec: float,
    end_sec: float,
    *,
    min_valid_ratio: float = 0.5,
) -> Dict[str, float]:
    """Compute one aligned HR/HRV feature row from a windowed source descriptor.

    For processed pickles, ECG is already windowed as ecg_filt[n_windows, n_samples].
    In that case start_sec is used as the integer window index.
    """
    out = _empty_autonomic_features()
    kind = str(source.get("kind", "none"))
    values = source.get("values", None)
    if values is None or kind == "none":
        return out
    try:
        arr = np.asarray(values)
    except Exception:
        return out
    out["autonomic_available"] = 1.0
    if kind == "ecg":
        idx = int(round(float(start_sec)))
        if arr.ndim == 2 and 0 <= idx < arr.shape[0]:
            win = arr[idx]
        else:
            fs = float(source.get("fs") or ECG_FS)
            st = max(0, int(round(float(start_sec) * fs)))
            en = min(arr.shape[0], int(round(float(end_sec) * fs)))
            win = arr[st:en]
        feat = ecg_window_features(
            np.asarray(win),
            fs=float(source.get("fs") or ECG_FS),
            min_hr_bpm=35.0,
            max_hr_bpm=220.0,
            min_valid_fraction=float(min_valid_ratio),
        )
        hr = float(feat.get("hr_bpm", float("nan")))
        rri = float(feat.get("rri_ms", float("nan")))
        if np.isfinite(hr):
            out.update({"hr_mean": hr, "hr_median": hr, "hr_min": hr, "hr_max": hr, "hr_valid_ratio": 1.0})
        if np.isfinite(rri):
            out.update(
                {
                    "ibi_mean": rri,
                    "ibi_std": float(feat.get("sdnn_ms", 0.0)),
                    "sdnn": float(feat.get("sdnn_ms", 0.0)),
                    "rmssd": float(feat.get("rmssd_ms", 0.0)),
                    "pnn50": float(feat.get("pnn50", 0.0)),
                    "ibi_valid_ratio": 1.0,
                    "hr_from_ibi_mean": float(60000.0 / rri) if rri > 0 else 0.0,
                }
            )
        return {k: (0.0 if not np.isfinite(v) else float(v)) for k, v in out.items()}
    return out


def _relative_hrv_matrix(features: np.ndarray, subject_ids: np.ndarray, names: Sequence[str]) -> np.ndarray:
    rel = np.zeros((features.shape[0], len(names)), dtype=np.float32)
    name_to_idx = {name: i for i, name in enumerate(HRV_ABSOLUTE_FEATURES)}
    for subject in np.unique(subject_ids.astype(str)):
        mask = subject_ids.astype(str) == str(subject)
        for j, name in enumerate(names):
            col = features[:, name_to_idx[name]].astype(float)
            vals = col[mask]
            finite = np.isfinite(vals)
            med = float(np.nanmedian(vals[finite])) if finite.any() else 0.0
            rel[mask, j] = np.nan_to_num(col[mask] - med, nan=0.0, posinf=0.0, neginf=0.0)
    return rel


def _select_hrv_features(
    absolute: np.ndarray,
    subject_ids: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, List[str]]:
    mode = str(mode).strip()
    name_to_idx = {name: i for i, name in enumerate(HRV_ABSOLUTE_FEATURES)}
    if mode == "none":
        return np.zeros((absolute.shape[0], 0), dtype=np.float32), []
    if mode == "hr_only":
        names = ["autonomic_available", "hr_mean", "hr_median", "hr_std", "hr_min", "hr_max", "hr_slope", "hr_valid_ratio"]
        return np.nan_to_num(absolute[:, [name_to_idx[n] for n in names]], nan=0.0).astype(np.float32), names
    if mode == "hrv_basic":
        names = ["autonomic_available", "sdnn", "rmssd", "pnn50", "ibi_valid_ratio"]
        return np.nan_to_num(absolute[:, [name_to_idx[n] for n in names]], nan=0.0).astype(np.float32), names
    if mode == "hr_hrv_absolute":
        return np.nan_to_num(absolute, nan=0.0).astype(np.float32), list(HRV_ABSOLUTE_FEATURES)
    rel = _relative_hrv_matrix(absolute, subject_ids, HRV_RELATIVE_BASE_FEATURES)
    rel_names = [f"{name}_relative" for name in HRV_RELATIVE_BASE_FEATURES]
    flags = np.nan_to_num(
        absolute[:, [name_to_idx["autonomic_available"], name_to_idx["hr_valid_ratio"], name_to_idx["ibi_valid_ratio"]]],
        nan=0.0,
    ).astype(np.float32)
    flag_names = ["autonomic_available", "hr_valid_ratio", "ibi_valid_ratio"]
    if mode == "hr_hrv_relative":
        return np.concatenate([flags, rel], axis=1), flag_names + rel_names
    if mode == "hr_hrv_absolute_plus_relative":
        return (
            np.concatenate([np.nan_to_num(absolute, nan=0.0).astype(np.float32), rel], axis=1),
            list(HRV_ABSOLUTE_FEATURES) + rel_names,
        )
    raise ValueError(f"Unsupported HRV feature mode {mode!r}")


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


def _remap_mwl_class_groups_with_extra(
    x: np.ndarray,
    y: np.ndarray,
    br: Optional[np.ndarray],
    cond: np.ndarray,
    class_groups: Sequence[Tuple[str, Sequence[int]]],
    *extras: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, List[Optional[np.ndarray]]]:
    id_to_idx: Dict[int, int] = {}
    for idx, (_name, ids) in enumerate(class_groups):
        for original_id in ids:
            id_to_idx[int(original_id)] = int(idx)
    cond = np.asarray(cond).astype(int).reshape(-1)
    mask = np.asarray([int(v) in id_to_idx for v in cond], dtype=bool)
    if not bool(mask.any()):
        empty_extras = [None if extra is None else extra[:0] for extra in extras]
        return x[:0], y[:0], None if br is None else br[:0], cond[:0], empty_extras
    cond_remap = np.asarray([id_to_idx[int(v)] for v in cond[mask]], dtype=int)
    filtered_extras = [None if extra is None else extra[mask] for extra in extras]
    return x[mask], y[mask], None if br is None else br[mask], cond_remap, filtered_extras


def _load_classification_arrays_with_hrv(
    subject_list: List[str],
    data_groups: List[str],
    data_str: str,
    data_dir: str,
    task: str,
    *,
    hrv_feature_mode: str,
    hrv_input_source: str,
    hrv_min_valid_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[str]]:
    xs, psss, brs, conds, hrv_abs_parts, subject_ids = [], [], [], [], [], []
    saw_br = False
    hrv_names: List[str] = []
    for group in data_groups:
        data_list = [load_data(sbj, data_dir=data_dir, data_group=group) for sbj in subject_list]
        out = make_dataset(
            data_list,
            data_str,
            label_encoder_dir=data_dir,
            data_group=group,
            include_subject_id=True,
        )
        x, pss, br, cond, sid = out[:5]
        cond_norm = _canonical_class_labels(cond, group, task)
        keep = np.asarray(_target_classes(task), dtype=int)
        mask = np.isin(cond_norm.astype(int), keep)
        if not bool(mask.any()):
            continue
        hrv_rows = []
        row_offset = 0
        for data in data_list:
            source = find_autonomic_source(data, preferred=hrv_input_source)
            n = int(np.asarray(data[data_str]).shape[0])
            for wi in range(n):
                hrv_rows.append(
                    compute_hrv_features_for_window(
                        source,
                        float(wi),
                        float(wi + 1),
                        min_valid_ratio=float(hrv_min_valid_ratio),
                    )
                )
            row_offset += n
        hrv_abs = np.asarray([[row[name] for name in HRV_ABSOLUTE_FEATURES] for row in hrv_rows], dtype=np.float32)
        xs.append(x[mask])
        psss.append(pss[mask])
        conds.append(cond_norm[mask].astype(int))
        hrv_abs_parts.append(hrv_abs[mask])
        subject_ids.append(np.asarray(sid, dtype=object)[mask])
        brs.append(None if br is None else br[mask])
        saw_br = saw_br or br is not None

    if not xs:
        raise RuntimeError(f"No {_target_name_for_mwl(task)} windows found for groups={data_groups}.")
    x_all = np.concatenate(xs, axis=0)
    pss_all = np.concatenate(psss, axis=0)
    cond_all = np.concatenate(conds, axis=0).astype(int)
    hrv_abs_all = np.concatenate(hrv_abs_parts, axis=0)
    sid_all = np.concatenate(subject_ids, axis=0)
    hrv_all, hrv_names = _select_hrv_features(hrv_abs_all, sid_all, hrv_feature_mode)
    br_all = None
    if saw_br:
        if any(br is None for br in brs):
            raise RuntimeError("Cannot combine splits when only some groups have BR arrays.")
        br_all = np.concatenate(brs, axis=0)
    return x_all, pss_all, br_all, cond_all, hrv_all, sid_all, hrv_names


def _target_name_for_mwl(task: str) -> str:
    if task == "mr_levels":
        return "M/R+L0-L3"
    return "L0-L3" if str(task).lower() in {"level", "levels"} else "M/R"


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
    hrv_feature_mode: str = "none",
    hrv_input_source: str = "auto",
    hrv_min_valid_ratio: float = 0.5,
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

    def _load_split(subject_list: List[str], groups: List[str]):
        return _load_classification_arrays_with_hrv(
            subject_list,
            groups,
            data_str,
            data_dir,
            task,
            hrv_feature_mode=hrv_feature_mode,
            hrv_input_source=hrv_input_source,
            hrv_min_valid_ratio=hrv_min_valid_ratio,
        )

    x_train, y_train, br_train, cond_train, hrv_train, sid_train, hrv_names = _load_split(train_subjects, train_groups)
    val_loader = None
    x_val = None
    y_val = None
    br_val = None
    cond_val = None
    hrv_val = None
    sid_val = None
    if val_subject_list:
        x_val, y_val, br_val, cond_val, hrv_val, sid_val, _ = _load_split(val_subject_list, train_groups)
    x_test, y_test, br_test, cond_test, hrv_test, sid_test, _ = _load_split([subject], test_groups)
    x_train, y_train, br_train, cond_train, train_extra = _remap_mwl_class_groups_with_extra(
        x_train, y_train, br_train, cond_train, class_groups, hrv_train, sid_train
    )
    hrv_train, sid_train = train_extra
    if x_val is not None:
        x_val, y_val, br_val, cond_val, val_extra = _remap_mwl_class_groups_with_extra(
            x_val, y_val, br_val, cond_val, class_groups, hrv_val, sid_val
        )
        hrv_val, sid_val = val_extra
    x_test, y_test, br_test, cond_test, test_extra = _remap_mwl_class_groups_with_extra(
        x_test, y_test, br_test, cond_test, class_groups, hrv_test, sid_test
    )
    hrv_test, sid_test = test_extra
    if x_train.shape[0] == 0:
        raise RuntimeError(
            f"No MWL training windows found for subset={class_names} "
            f"(held-out {subject}, train_groups={train_groups})."
        )
    if x_test.shape[0] == 0:
        raise RuntimeError(f"No MWL test windows found for subject {subject}, subset={class_names}.")

    def _tensor_ds(x, y, cond, br, hrv):
        br_arr = np.zeros((x.shape[0],), dtype=np.float32) if br is None else np.asarray(br, dtype=np.float32)
        return TensorDataset(
            torch.from_numpy(np.asarray(x)).float(),
            torch.from_numpy(np.asarray(y)).float(),
            torch.from_numpy(np.asarray(cond)).long(),
            torch.from_numpy(br_arr).float(),
            torch.from_numpy(np.asarray(hrv, dtype=np.float32)).float(),
        )

    train_loader = DataLoader(
        _tensor_ds(x_train, y_train, cond_train, br_train, hrv_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    if x_val is not None and x_val.shape[0] > 0:
        val_loader = DataLoader(
            _tensor_ds(x_val, y_val, cond_val, br_val, hrv_val),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
    test_loader = DataLoader(
        _tensor_ds(x_test, y_test, cond_test, br_test, hrv_test),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    def _counts_by_subject(split_name: str, sid: Optional[np.ndarray], labels: np.ndarray) -> List[Dict[str, object]]:
        if sid is None:
            sid = np.asarray([split_name] * int(labels.shape[0]), dtype=object)
        labels = np.asarray(labels).astype(int).reshape(-1)
        sid = np.asarray(sid, dtype=object).reshape(-1)
        out: List[Dict[str, object]] = []
        for one_subject in sorted(set(str(v) for v in sid)):
            mask = sid.astype(str) == str(one_subject)
            yy = labels[mask]
            counts = np.bincount(yy, minlength=max(3, len(class_names))).astype(int)
            row = {
                "subject": str(one_subject),
                "split": split_name,
                "n_rest": int(counts[0]) if counts.size > 0 else 0,
                "n_low": int(counts[1]) if counts.size > 1 else 0,
                "n_high": int(counts[2]) if counts.size > 2 else 0,
                "n_total": int(yy.size),
                "has_rest": int(counts[0] > 0) if counts.size > 0 else 0,
                "has_low": int(counts[1] > 0) if counts.size > 1 else 0,
                "has_high": int(counts[2] > 0) if counts.size > 2 else 0,
            }
            out.append(row)
        return out

    class_counts = []
    class_counts.extend(_counts_by_subject("source_train", sid_train, cond_train))
    if x_val is not None and cond_val is not None:
        class_counts.extend(_counts_by_subject("source_val", sid_val, cond_val))
    class_counts.extend(_counts_by_subject("target_test", sid_test, cond_test))

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
        "hrv_feature_mode": str(hrv_feature_mode),
        "hrv_feature_names": hrv_names,
        "hrv_dim": int(hrv_train.shape[1]),
        "hrv_available_train_ratio": float(np.mean(hrv_train[:, 0] > 0)) if hrv_train.shape[1] else 0.0,
        "hrv_available_test_ratio": float(np.mean(hrv_test[:, 0] > 0)) if hrv_test.shape[1] else 0.0,
        "class_counts_by_subject": class_counts,
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

    def features(self, hidden: torch.Tensor, profile: Optional[torch.Tensor] = None) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(f"Expected hidden (B,T,D), got {tuple(hidden.shape)}")
        return hidden.mean(dim=1)

    def forward(self, hidden: torch.Tensor, profile: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.classifier(self.features(hidden, profile))


class PooledMLPMWLHead(nn.Module):
    """A1: mean-pooled hidden tokens followed by a small MLP."""

    def __init__(self, d_model: int, num_classes: int, hidden_dim: int = 32, dropout: float = 0.4):
        super().__init__()
        self.num_classes = int(num_classes)
        self.feature_net = nn.Sequential(
            nn.LayerNorm(int(d_model)),
            nn.Linear(int(d_model), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.classifier = nn.Linear(int(hidden_dim), self.num_classes)

    def features(self, hidden: torch.Tensor, profile: Optional[torch.Tensor] = None) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(f"Expected hidden (B,T,D), got {tuple(hidden.shape)}")
        return self.feature_net(hidden.mean(dim=1))

    def forward(self, hidden: torch.Tensor, profile: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.classifier(self.features(hidden, profile))


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
        self.feature_norm = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Dropout(float(dropout)),
        )
        self.classifier = nn.Linear(int(hidden_dim), self.num_classes)

    def features(self, hidden: torch.Tensor, profile: Optional[torch.Tensor] = None) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(f"Expected hidden (B,T,D), got {tuple(hidden.shape)}")
        h = self.in_proj(hidden.transpose(1, 2))
        h = self.tcn(h).transpose(1, 2)
        return self.feature_norm(h.mean(dim=1))

    def forward(self, hidden: torch.Tensor, profile: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.classifier(self.features(hidden, profile))


class HRVFusionClassifier(nn.Module):
    def __init__(
        self,
        base_feature_dim: int,
        hrv_dim: int,
        num_classes: int,
        hrv_hidden_dim: int = 16,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hrv_branch = nn.Sequential(
            nn.LayerNorm(int(hrv_dim)),
            nn.Linear(int(hrv_dim), int(hrv_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hrv_hidden_dim), int(hrv_hidden_dim)),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(int(base_feature_dim) + int(hrv_hidden_dim)),
            nn.Dropout(float(dropout)),
            nn.Linear(int(base_feature_dim) + int(hrv_hidden_dim), int(num_classes)),
        )

    def forward(self, base_feature: torch.Tensor, hrv: torch.Tensor) -> torch.Tensor:
        z_hrv = self.hrv_branch(hrv.float())
        return self.classifier(torch.cat([base_feature, z_hrv], dim=1))


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


def _mwl_head_feature_dim(head_variant: str, args) -> int:
    variant = str(head_variant).strip()
    if variant == "A0":
        return int(args.d_model)
    if variant in {"A1", "A2"}:
        return int(args.mwl_tcn_hidden_dim)
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


def _unpack_mwl_batch(batch, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(batch) == 5:
        imu, pressure, cond, br, hrv = batch
    elif len(batch) == 4:
        imu, pressure, cond, br = batch
        hrv = torch.zeros((imu.size(0), 0), dtype=torch.float32)
    else:
        raise ValueError(f"Expected 4 or 5 tensors from MWL loader, got {len(batch)}")
    return (
        imu.float().to(device),
        pressure.float().to(device),
        cond.long().view(-1).to(device),
        br.float().to(device),
        hrv.float().to(device),
    )


def _mwl_logits(head: nn.Module, hidden: torch.Tensor, profile: Optional[torch.Tensor], hrv: torch.Tensor, fusion: Optional[nn.Module]) -> torch.Tensor:
    if fusion is None:
        return head(hidden, profile)
    base_feature = head.features(hidden, profile)
    return fusion(base_feature, hrv)


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
    head: nn.Module,
    fusion: Optional[nn.Module],
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
    params = list(head.parameters()) + ([] if fusion is None else list(fusion.parameters()))
    if fusion is not None:
        fusion.train()
        fusion.to(device)
    opt = torch.optim.AdamW(params, lr=float(args.mwl_lr), weight_decay=float(args.mwl_weight_decay))

    labels_for_weights = []
    for batch in loader:
        _imu, _pressure, cond, _br, _hrv = _unpack_mwl_batch(batch, device)
        labels_for_weights.append(cond.detach().cpu().numpy())
    labels_np = np.concatenate(labels_for_weights) if labels_for_weights else np.zeros((0,), dtype=int)
    n_classes = int(labels_np.max()) + 1 if labels_np.size else 2
    counts = np.bincount(labels_np.astype(int), minlength=n_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / max(float(weights.mean()), 1e-6)
    class_weight = torch.tensor(weights, dtype=torch.float32, device=device)

    def _score(split_loader, prefix: str) -> Dict[str, float]:
        head.eval()
        if fusion is not None:
            fusion.eval()
        losses, preds, labels = [], [], []
        with torch.no_grad():
            for batch in split_loader:
                imu, _pressure, y, _br, hrv = _unpack_mwl_batch(batch, device)
                hidden, profile = _forward_source_profile_hidden(model, imu, device)
                logits = _mwl_logits(head, hidden, profile, hrv, fusion)
                loss = F.cross_entropy(logits, y, weight=class_weight)
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
        if fusion is not None:
            fusion.train()
        losses, preds, labels = [], [], []
        for batch in loader:
            imu, _pressure, y, _br, hrv = _unpack_mwl_batch(batch, device)
            with torch.no_grad():
                hidden, profile = _forward_source_profile_hidden(model, imu, device)
            logits = _mwl_logits(head, hidden, profile, hrv, fusion)
            loss = F.cross_entropy(logits, y, weight=class_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.mwl_grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, float(args.mwl_grad_clip))
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
            best_state = {
                "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                "fusion": None if fusion is None else {k: v.detach().cpu().clone() for k, v in fusion.state_dict().items()},
            }
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
        head.load_state_dict({k: v.to(device) for k, v in best_state["head"].items()}, strict=True)
        if fusion is not None and best_state["fusion"] is not None:
            fusion.load_state_dict({k: v.to(device) for k, v in best_state["fusion"].items()}, strict=True)

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


def _predict_mwl_mode(
    model: nn.Module,
    head: nn.Module,
    fusion: Optional[nn.Module],
    rr_model: Optional[FaithfulRRRegressor],
    loader,
    mode: str,
    device: str,
    args,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    model.eval()
    head.eval()
    if fusion is not None:
        fusion.eval()
    source_state = _state_dict_cpu_clone(model)

    probs_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    pred_all: List[np.ndarray] = []
    batch_rows: List[Dict[str, float]] = []
    for bi, batch in enumerate(loader):
        _restore_state_dict(model, source_state, device)
        for p in model.parameters():
            p.requires_grad = False
        imu, _pressure, cond, _br, hrv = _unpack_mwl_batch(batch, device)
        y = cond.detach().cpu().numpy()
        hidden_base, profile_base = _forward_source_profile_hidden(model, imu, device)
        with torch.no_grad():
            logits_base = _mwl_logits(head, hidden_base, profile_base, hrv, fusion)
            probs_base = torch.softmax(logits_base, dim=1)

        if mode == "none":
            hidden, profile = hidden_base, profile_base
            diag = {"ttt_loss": 0.0, "profile_delta_norm": 0.0}
        else:
            if rr_model is None:
                raise RuntimeError("MWL profile TTT mode requires a trained RR probe.")
            hidden, profile, diag = _adapt_profile_and_get_hidden(model, rr_model, imu, mode, device, args)

        with torch.no_grad():
            logits = _mwl_logits(head, hidden, profile, hrv, fusion)
            probs_t = torch.softmax(logits, dim=1)
            probs = probs_t.detach().cpu().numpy()
        pred = probs.argmax(axis=1).astype(int)
        base_pred = probs_base.detach().argmax(dim=1).cpu().numpy().astype(int)
        probs_all.append(probs)
        pred_all.append(pred)
        labels_all.append(y)
        logit_delta = logits.detach() - logits_base.detach()
        prob_delta = probs_t.detach() - probs_base.detach()
        batch_rows.append(
            {
                "batch_idx": int(bi),
                "mwl_mode": str(mode),
                "n_batch": int(len(y)),
                "batch_acc": float(accuracy_score(y, pred)) if len(y) else float("nan"),
                "mean_confidence": float(np.max(probs, axis=1).mean()) if probs.size else float("nan"),
                "logit_delta_norm": float(logit_delta.norm(p=2).cpu()),
                "prob_delta_norm": float(prob_delta.norm(p=2).cpu()),
                "pred_changed_rate": float(np.mean(pred != base_pred)) if len(pred) else 0.0,
                "mean_confidence_none": float(probs_base.max(dim=1).values.mean().cpu()) if probs_base.numel() else float("nan"),
                "mean_confidence_ttt": float(np.max(probs, axis=1).mean()) if probs.size else float("nan"),
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
        return {
            f"{prefix}_acc": float("nan"),
            f"{prefix}_balanced_acc": float("nan"),
            f"{prefix}_f1_macro": float("nan"),
            f"{prefix}_high_recall": float("nan"),
            f"{prefix}_low_recall": float("nan"),
            f"{prefix}_n": 0,
        }
    recalls = recall_score(y, pred, labels=[0, 1], average=None, zero_division=0)
    return {
        f"{prefix}_acc": float(accuracy_score(y, pred)),
        f"{prefix}_balanced_acc": float(balanced_accuracy_score(y, pred)),
        f"{prefix}_f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        f"{prefix}_low_recall": float(recalls[0]) if len(recalls) > 0 else float("nan"),
        f"{prefix}_high_recall": float(recalls[1]) if len(recalls) > 1 else float("nan"),
        f"{prefix}_n": int(y.size),
    }


def _save_confusion_pair(
    out_dir: Path,
    *,
    subject: str,
    task: str,
    head: str,
    mode: str,
    hrv_mode: str,
    scope: str,
    y: np.ndarray,
    pred: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    base = f"{subject}__{task}__{head}__{mode}__{hrv_mode}__{scope}"
    pd.DataFrame(cm, index=["true_low", "true_high"], columns=["pred_low", "pred_high"]).to_csv(out_dir / f"{base}.csv")
    denom = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_norm = cm.astype(float) / denom
    pd.DataFrame(cm_norm, index=["true_low", "true_high"], columns=["pred_low", "pred_high"]).to_csv(out_dir / f"{base}_normalized.csv")


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
    hrv_feature_modes = _parse_hrv_feature_modes(str(getattr(args, "hrv_feature_modes", "none")))
    chunk_dir = Path(args.out_dir) / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir = Path(args.out_dir) / "confusion_matrices"
    window_pred_rows: List[Dict[str, object]] = []
    one_min_pred_rows: List[Dict[str, object]] = []
    ttt_diag_rows: List[Dict[str, object]] = []

    for task_spec in task_specs:
        task_name = str(task_spec["name"])
        class_groups = task_spec["class_groups"]
        for hrv_feature_mode in hrv_feature_modes:
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
                hrv_feature_mode=hrv_feature_mode,
                hrv_input_source=str(getattr(args, "hrv_input_source", "auto")),
                hrv_min_valid_ratio=float(getattr(args, "hrv_min_valid_ratio", 0.5)),
            )
            class_names = [str(v) for v in cls_meta["class_names"]]
            num_classes = len(class_names)
            use_fusion = bool(getattr(args, "use_hrv_fusion", False)) and hrv_feature_mode != "none" and int(cls_meta["hrv_dim"]) > 0

            for head_variant in head_variants:
                head = _build_mwl_head(head_variant, args, num_classes).to(device)
                fusion = None
                if use_fusion:
                    fusion = HRVFusionClassifier(
                        _mwl_head_feature_dim(head_variant, args),
                        int(cls_meta["hrv_dim"]),
                        num_classes,
                        hrv_hidden_dim=int(getattr(args, "hrv_hidden_dim", 16)),
                        dropout=float(getattr(args, "hrv_dropout", 0.3)),
                    ).to(device)
                out_root = sbj_dir / "mwl_diagnostics" / task_name / head_variant / hrv_feature_mode
                out_root.mkdir(parents=True, exist_ok=True)
                train_last = train_mwl_head(model, head, fusion, cls_train_loader, cls_val_loader, device, args)
                torch.save(
                    {
                        "head_state_dict": head.state_dict(),
                        "fusion_state_dict": None if fusion is None else fusion.state_dict(),
                        "args": vars(args),
                        "subject": sbj,
                        "mwl_diagnostic_task": task_name,
                        "mwl_head_variant": head_variant,
                        "hrv_feature_mode": hrv_feature_mode,
                        "hrv_feature_names": cls_meta["hrv_feature_names"],
                        "class_names": class_names,
                    },
                    out_root / "mwl_head.pt",
                )

                for requested_mode in modes:
                    requested_mode = str(requested_mode).strip().lower()
                    effective_mode = requested_mode
                    gate_info: Dict[str, float | int | str] = {}
                    if requested_mode.startswith("profile_qkv_") and rr_model is not None:
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
                    y, pred, prob, batch_df = _predict_mwl_mode(model, head, fusion, rr_model, cls_test_loader, effective_mode, device, args)
                    y_1m, pred_1m, prob_1m, chunks_df = _postprocess_1min(
                        y,
                        prob,
                        seconds=float(args.mwl_postprocess_seconds),
                        window_shift_seconds=float(args.mwl_window_shift_seconds),
                    )

                    if bool(getattr(args, "save_confusion_matrices", False)):
                        _save_confusion_pair(confusion_dir, subject=sbj, task=task_name, head=head_variant, mode=requested_mode, hrv_mode=hrv_feature_mode, scope="window", y=y, pred=pred)
                        _save_confusion_pair(confusion_dir, subject=sbj, task=task_name, head=head_variant, mode=requested_mode, hrv_mode=hrv_feature_mode, scope="one_min", y=y_1m, pred=pred_1m)

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
                        "hrv_feature_mode": hrv_feature_mode,
                        "use_hrv_fusion": int(use_fusion),
                        "hrv_dim": int(cls_meta["hrv_dim"]),
                        "hrv_feature_names": json.dumps(cls_meta["hrv_feature_names"]),
                        "hrv_available_train_ratio": float(cls_meta["hrv_available_train_ratio"]),
                        "hrv_available_test_ratio": float(cls_meta["hrv_available_test_ratio"]),
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
                        "mwl_val_balanced_acc_last": float(train_last.get("val_balanced_acc", float("nan"))),
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

                    pred_df = pd.DataFrame({"subject": sbj, "task": task_name, "head": head_variant, "mode": requested_mode, "hrv_feature_mode": hrv_feature_mode, "window_idx": np.arange(y.size), "y_true": y, "y_pred": pred})
                    for c, class_name in enumerate(class_names):
                        pred_df[f"prob_{class_name}"] = prob[:, c]
                    chunks_df = chunks_df.copy()
                    chunks_df.insert(0, "hrv_feature_mode", hrv_feature_mode)
                    chunks_df.insert(0, "mode", requested_mode)
                    chunks_df.insert(0, "head", head_variant)
                    chunks_df.insert(0, "task", task_name)
                    chunks_df.insert(0, "subject", sbj)
                    batch_df = batch_df.copy()
                    batch_df.insert(0, "hrv_feature_mode", hrv_feature_mode)
                    batch_df.insert(0, "mode", requested_mode)
                    batch_df.insert(0, "head", head_variant)
                    batch_df.insert(0, "task", task_name)
                    batch_df.insert(0, "subject", sbj)
                    pred_df.to_csv(mode_dir / "mwl_window_predictions.csv", index=False)
                    chunks_df.to_csv(mode_dir / "mwl_1min_predictions.csv", index=False)
                    batch_df.to_csv(mode_dir / "mwl_ttt_batches.csv", index=False)
                    window_pred_rows.extend(pred_df.to_dict(orient="records"))
                    one_min_pred_rows.extend(chunks_df.to_dict(orient="records"))
                    ttt_diag_rows.extend(batch_df.to_dict(orient="records"))
                    with open(mode_dir / "mwl_metrics.json", "w") as f:
                        json.dump({k: v for k, v in row.items() if not k.startswith("__")}, f, indent=2)
                    print(
                        f"[MWL] subject={sbj} task={task_name} head={head_variant} hrv={hrv_feature_mode} "
                        f"mode={requested_mode} effective={effective_mode} bal_acc={win_metrics['mwl_window_balanced_acc']:.4f} "
                        f"one_min_bal_acc={min_metrics['mwl_1min_balanced_acc']:.4f}"
                    )

    if rows:
        rows_df = pd.DataFrame([{k: v for k, v in row.items() if not str(k).startswith("__")} for row in rows])
        rows_df.to_csv(chunk_dir / f"{sbj}_mwl_diagnostic_summary.csv", index=False)
        rows_df.to_csv(chunk_dir / f"{sbj}_summary.csv", index=False)
    if window_pred_rows:
        pd.DataFrame(window_pred_rows).to_csv(chunk_dir / f"{sbj}_mwl_predictions_window.csv", index=False)
    if one_min_pred_rows:
        pd.DataFrame(one_min_pred_rows).to_csv(chunk_dir / f"{sbj}_mwl_predictions_one_min.csv", index=False)
    if ttt_diag_rows:
        pd.DataFrame(ttt_diag_rows).to_csv(chunk_dir / f"{sbj}_mwl_ttt_batches.csv", index=False)
    return rows


REST_LOW_HIGH_GROUPS = [("rest", [1]), ("low", [2, 3]), ("high", [4, 5])]
REST_LOW_HIGH_LABELS = (0, 1, 2)
REST_LOW_HIGH_NAMES = ["0_rest", "1_low", "2_high"]


def _parse_variants_3class(text: str) -> List[str]:
    valid = {"flat_a0", "flat_a2", "hier_a2", "hier_a2_smooth", "hier_a2_qkv_ttt", "hier_a2_hrv"}
    if isinstance(text, (list, tuple)):
        names = [str(v).strip() for v in text if str(v).strip()]
    else:
        names = _parse_csv_names(text)
    names = names or ["flat_a0", "flat_a2", "hier_a2", "hier_a2_smooth"]
    aliases = {
        "H0": "flat_a0",
        "H1": "flat_a2",
        "H2": "hier_a2",
        "H3": "hier_a2_smooth",
        "H4": "hier_a2_qkv_ttt",
        "H5": "hier_a2_hrv",
    }
    out = [aliases.get(name, aliases.get(name.upper(), name)) for name in names]
    bad = [name for name in out if name not in valid]
    if bad:
        raise ValueError(f"Unsupported --variants entries {bad}. Valid={sorted(valid)}")
    return list(dict.fromkeys(out))


def _parse_seconds_list(text_or_values) -> List[float]:
    if isinstance(text_or_values, (list, tuple)):
        parts = text_or_values
    else:
        parts = str(text_or_values or "").replace(",", " ").split()
    vals = [float(v) for v in parts if str(v).strip()]
    return vals or [60.0, 120.0]


def fixed_label_metrics(y_true, y_pred, labels=(0, 1, 2)) -> Dict[str, object]:
    labels = tuple(int(v) for v in labels)
    y = np.asarray(y_true).astype(int).reshape(-1)
    pred = np.asarray(y_pred).astype(int).reshape(-1)
    cm = confusion_matrix(y, pred, labels=list(labels))
    recalls = []
    for i, _label in enumerate(labels):
        denom = int(cm[i, :].sum())
        recalls.append(float(cm[i, i] / denom) if denom > 0 else 0.0)
    if y.size == 0:
        acc = float("nan")
        macro_f1 = float("nan")
    else:
        acc = float(accuracy_score(y, pred))
        macro_f1 = float(f1_score(y, pred, labels=list(labels), average="macro", zero_division=0))
    return {
        "accuracy": acc,
        "balanced_accuracy_fixed": float(np.mean(recalls)) if recalls else float("nan"),
        "macro_f1_fixed": macro_f1,
        "rest_recall": recalls[0] if len(recalls) > 0 else float("nan"),
        "low_recall": recalls[1] if len(recalls) > 1 else float("nan"),
        "high_recall": recalls[2] if len(recalls) > 2 else float("nan"),
        "confusion_matrix_3x3": cm,
    }


def _prefixed_fixed_metrics(y_true, y_pred, prefix: str) -> Dict[str, float]:
    m = fixed_label_metrics(y_true, y_pred, REST_LOW_HIGH_LABELS)
    return {
        f"{prefix}_accuracy": float(m["accuracy"]),
        f"{prefix}_balanced_accuracy_fixed": float(m["balanced_accuracy_fixed"]),
        f"{prefix}_macro_f1_fixed": float(m["macro_f1_fixed"]),
        f"{prefix}_rest_recall": float(m["rest_recall"]),
        f"{prefix}_low_recall": float(m["low_recall"]),
        f"{prefix}_high_recall": float(m["high_recall"]),
    }


class HierarchicalA2MWLHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_classes = 3
        self.feature_head = TinyTCNMWLHead(
            int(args.d_model),
            3,
            hidden_dim=int(args.mwl_tcn_hidden_dim),
            num_layers=int(args.mwl_tcn_layers),
            kernel_size=int(args.mwl_tcn_kernel_size),
            dropout=float(args.mwl_dropout),
        )
        dim = int(args.mwl_tcn_hidden_dim)
        self.rest_load_head = nn.Linear(dim, 2)
        self.low_high_head = nn.Linear(dim, 2)

    def features(self, hidden: torch.Tensor, profile: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.feature_head.features(hidden, profile)

    def forward(self, hidden: torch.Tensor, profile: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(hidden, profile)
        return self.rest_load_head(feat), self.low_high_head(feat)


class HierarchicalHRVFusionClassifier(nn.Module):
    def __init__(self, base_feature_dim: int, hrv_dim: int, hrv_hidden_dim: int = 16, dropout: float = 0.3):
        super().__init__()
        self.hrv_branch = nn.Sequential(
            nn.LayerNorm(int(hrv_dim)),
            nn.Linear(int(hrv_dim), int(hrv_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hrv_hidden_dim), int(hrv_hidden_dim)),
        )
        dim = int(base_feature_dim) + int(hrv_hidden_dim)
        self.rest_load_head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(float(dropout)), nn.Linear(dim, 2))
        self.low_high_head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(float(dropout)), nn.Linear(dim, 2))

    def forward(self, base_feature: torch.Tensor, hrv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([base_feature, self.hrv_branch(hrv.float())], dim=1)
        return self.rest_load_head(z), self.low_high_head(z)


def _hier_probs_from_logits(rest_load_logits: torch.Tensor, low_high_logits: torch.Tensor) -> torch.Tensor:
    p_rl = torch.softmax(rest_load_logits, dim=1)
    p_lh = torch.softmax(low_high_logits, dim=1)
    return torch.stack([p_rl[:, 0], p_rl[:, 1] * p_lh[:, 0], p_rl[:, 1] * p_lh[:, 1]], dim=1)


def _class_weights_np(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels).astype(int), minlength=int(n_classes)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / max(float(weights.mean()), 1e-6)
    return torch.tensor(weights, dtype=torch.float32)


def _train_3class_head(
    model: nn.Module,
    head: nn.Module,
    fusion: Optional[nn.Module],
    loader,
    val_loader,
    device: str,
    args,
    *,
    hierarchical: bool,
) -> Dict[str, float]:
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    head.to(device).train()
    if fusion is not None:
        fusion.to(device).train()
    params = list(head.parameters()) + ([] if fusion is None else list(fusion.parameters()))
    opt = torch.optim.AdamW(params, lr=float(args.mwl_lr), weight_decay=float(args.mwl_weight_decay))

    labels_np = []
    for batch in loader:
        _imu, _pressure, y, _br, _hrv = _unpack_mwl_batch(batch, device)
        labels_np.append(y.detach().cpu().numpy())
    y_train_np = np.concatenate(labels_np) if labels_np else np.zeros((0,), dtype=int)
    class_weight_3 = _class_weights_np(y_train_np, 3).to(device)
    rl_weight = _class_weights_np((y_train_np != 0).astype(int), 2).to(device)
    lh_train = y_train_np[y_train_np != 0] - 1
    lh_weight = _class_weights_np(lh_train, 2).to(device)

    def _forward_probs(hidden, profile, hrv):
        if hierarchical:
            feat = head.features(hidden, profile)
            if fusion is None:
                rest_load_logits, low_high_logits = head.rest_load_head(feat), head.low_high_head(feat)
            else:
                rest_load_logits, low_high_logits = fusion(feat, hrv)
            probs = _hier_probs_from_logits(rest_load_logits, low_high_logits)
            return probs, rest_load_logits, low_high_logits
        logits = _mwl_logits(head, hidden, profile, hrv, fusion)
        return torch.softmax(logits, dim=1), logits, None

    def _loss_from_outputs(y, probs, logits_a, logits_b):
        if not hierarchical:
            return F.cross_entropy(logits_a, y, weight=class_weight_3)
        y_rest_load = (y != 0).long()
        loss_rl = F.cross_entropy(logits_a, y_rest_load, weight=rl_weight)
        load_mask = y != 0
        if bool(load_mask.any()):
            y_low_high = (y[load_mask] - 1).long()
            loss_lh = F.cross_entropy(logits_b[load_mask], y_low_high, weight=lh_weight)
        else:
            loss_lh = torch.zeros((), device=device)
        return loss_rl + float(getattr(args, "lambda_low_high", 1.0)) * loss_lh

    @torch.no_grad()
    def _score(split_loader, prefix: str) -> Dict[str, float]:
        head.eval()
        if fusion is not None:
            fusion.eval()
        losses, preds, labels = [], [], []
        for batch in split_loader:
            imu, _pressure, y, _br, hrv = _unpack_mwl_batch(batch, device)
            hidden, profile = _forward_source_profile_hidden(model, imu, device)
            probs, logits_a, logits_b = _forward_probs(hidden, profile, hrv)
            losses.append(float(_loss_from_outputs(y, probs, logits_a, logits_b).detach().cpu()))
            preds.append(probs.argmax(dim=1).detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
        yy = np.concatenate(labels) if labels else np.array([], dtype=int)
        pp = np.concatenate(preds) if preds else np.array([], dtype=int)
        out = {f"{prefix}_loss": float(np.mean(losses)) if losses else float("nan")}
        out.update(_prefixed_fixed_metrics(yy, pp, prefix))
        return out

    monitor_alias = {
        "val_macro_f1": "val_macro_f1_fixed",
        "val_f1_macro": "val_macro_f1_fixed",
        "val_balanced_accuracy": "val_balanced_accuracy_fixed",
        "val_balanced_acc": "val_balanced_accuracy_fixed",
        "val_acc": "val_accuracy",
    }
    monitor = monitor_alias.get(str(getattr(args, "mwl_monitor", "val_macro_f1_fixed")), str(getattr(args, "mwl_monitor", "val_macro_f1_fixed")))
    maximize = not monitor.endswith("_loss")
    best_score = -float("inf") if maximize else float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history = []
    for epoch in range(1, int(args.mwl_epochs) + 1):
        head.train()
        if fusion is not None:
            fusion.train()
        losses, preds, labels = [], [], []
        for batch in loader:
            imu, _pressure, y, _br, hrv = _unpack_mwl_batch(batch, device)
            with torch.no_grad():
                hidden, profile = _forward_source_profile_hidden(model, imu, device)
            probs, logits_a, logits_b = _forward_probs(hidden, profile, hrv)
            loss = _loss_from_outputs(y, probs, logits_a, logits_b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.mwl_grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, float(args.mwl_grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu()))
            preds.append(probs.detach().argmax(dim=1).cpu().numpy())
            labels.append(y.detach().cpu().numpy())
        yy = np.concatenate(labels) if labels else np.array([], dtype=int)
        pp = np.concatenate(preds) if preds else np.array([], dtype=int)
        row = {"epoch": int(epoch), "loss": float(np.mean(losses)) if losses else float("nan")}
        row.update(_prefixed_fixed_metrics(yy, pp, "train"))
        if val_loader is not None:
            row.update(_score(val_loader, "val"))
        else:
            row.update({k: float("nan") for k in ["val_loss", "val_accuracy", "val_balanced_accuracy_fixed", "val_macro_f1_fixed"]})
        current = float(row.get(monitor, float("nan")))
        improved = math.isfinite(current) and (current > best_score if maximize else current < best_score)
        if improved:
            best_score = current
            best_epoch = int(epoch)
            stale_epochs = 0
            best_state = {
                "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                "fusion": None if fusion is None else {k: v.detach().cpu().clone() for k, v in fusion.state_dict().items()},
            }
        else:
            stale_epochs += 1
        history.append(row)
        if epoch == 1 or epoch == int(args.mwl_epochs) or epoch % max(1, int(args.mwl_log_every)) == 0:
            print(
                f"[MWL_3C] epoch={epoch} loss={row['loss']:.4f} "
                f"train_f1={row['train_macro_f1_fixed']:.4f} val_f1={row['val_macro_f1_fixed']:.4f}"
            )
        if bool(getattr(args, "mwl_early_stop", True)) and val_loader is not None and stale_epochs >= int(getattr(args, "mwl_patience", 8)):
            print(f"[MWL_3C] early_stop epoch={epoch} best_epoch={best_epoch} {monitor}={best_score:.4f}")
            break
    if best_state is not None:
        head.load_state_dict({k: v.to(device) for k, v in best_state["head"].items()}, strict=True)
        if fusion is not None and best_state["fusion"] is not None:
            fusion.load_state_dict({k: v.to(device) for k, v in best_state["fusion"].items()}, strict=True)
    last = dict(history[-1]) if history else {}
    best_row = next((row for row in history if int(row["epoch"]) == best_epoch), {})
    out = dict(last)
    out.update({f"best_{k}": v for k, v in best_row.items() if k != "epoch"})
    out["best_epoch"] = int(best_epoch)
    out["best_monitor"] = monitor
    out["best_monitor_value"] = float(best_score) if math.isfinite(best_score) else float("nan")
    out["epochs_ran"] = int(len(history))
    return out


def _predict_3class_mode(
    model: nn.Module,
    head: nn.Module,
    fusion: Optional[nn.Module],
    rr_model: Optional[FaithfulRRRegressor],
    loader,
    mode: str,
    device: str,
    args,
    *,
    hierarchical: bool,
    rest_threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, Dict[str, np.ndarray]]:
    model.eval()
    head.eval()
    if fusion is not None:
        fusion.eval()
    source_state = _state_dict_cpu_clone(model)
    probs_all, labels_all, pred_all = [], [], []
    rl_probs_all, lh_probs_all = [], []
    batch_rows = []
    for bi, batch in enumerate(loader):
        _restore_state_dict(model, source_state, device)
        for p in model.parameters():
            p.requires_grad = False
        imu, _pressure, y_t, _br, hrv = _unpack_mwl_batch(batch, device)
        y = y_t.detach().cpu().numpy()
        hidden_base, profile_base = _forward_source_profile_hidden(model, imu, device)
        with torch.no_grad():
            if hierarchical:
                feat_base = head.features(hidden_base, profile_base)
                if fusion is None:
                    rl_base, lh_base = head.rest_load_head(feat_base), head.low_high_head(feat_base)
                else:
                    rl_base, lh_base = fusion(feat_base, hrv)
                probs_base = _hier_probs_from_logits(rl_base, lh_base)
            else:
                logits_base = _mwl_logits(head, hidden_base, profile_base, hrv, fusion)
                probs_base = torch.softmax(logits_base, dim=1)
        if mode == "none":
            hidden, profile = hidden_base, profile_base
            diag = {"profile_delta_norm": 0.0, "hidden_delta_norm": 0.0, "hidden_delta_rms": 0.0}
        else:
            if rr_model is None:
                raise RuntimeError("MWL profile TTT mode requires a trained RR probe.")
            hidden, profile, diag = _adapt_profile_and_get_hidden(model, rr_model, imu, mode, device, args)
        with torch.no_grad():
            if hierarchical:
                feat = head.features(hidden, profile)
                if fusion is None:
                    rl_logits, lh_logits = head.rest_load_head(feat), head.low_high_head(feat)
                else:
                    rl_logits, lh_logits = fusion(feat, hrv)
                probs_t = _hier_probs_from_logits(rl_logits, lh_logits)
                rl_probs = torch.softmax(rl_logits, dim=1).detach().cpu().numpy()
                lh_probs = torch.softmax(lh_logits, dim=1).detach().cpu().numpy()
            else:
                logits = _mwl_logits(head, hidden, profile, hrv, fusion)
                probs_t = torch.softmax(logits, dim=1)
                rl_probs = np.full((int(y.shape[0]), 2), np.nan, dtype=np.float32)
                lh_probs = np.full((int(y.shape[0]), 2), np.nan, dtype=np.float32)
        probs = probs_t.detach().cpu().numpy()
        pred = probs.argmax(axis=1).astype(int)
        if rest_threshold is not None and hierarchical:
            load_pred = 1 + np.argmax(probs[:, 1:3], axis=1)
            pred = np.where(probs[:, 0] >= float(rest_threshold), 0, load_pred).astype(int)
        base_pred = probs_base.detach().argmax(dim=1).cpu().numpy().astype(int)
        probs_all.append(probs)
        pred_all.append(pred)
        labels_all.append(y)
        rl_probs_all.append(rl_probs)
        lh_probs_all.append(lh_probs)
        prob_delta = probs_t.detach() - probs_base.detach()
        batch_rows.append(
            {
                "batch_index": int(bi),
                "mode": str(mode),
                "n_batch": int(len(y)),
                "profile_delta_norm": float(diag.get("profile_delta_norm", 0.0)),
                "hidden_delta_norm": float(diag.get("hidden_delta_norm", 0.0)),
                "hidden_delta_rms": float(diag.get("hidden_delta_rms", 0.0)),
                "logit_delta_norm": float("nan"),
                "prob_delta_norm": float(prob_delta.norm(p=2).cpu()),
                "pred_changed_rate": float(np.mean(pred != base_pred)) if len(pred) else 0.0,
            }
        )
    _restore_state_dict(model, source_state, device)
    y_np = np.concatenate(labels_all, axis=0) if labels_all else np.array([], dtype=int)
    pred_np = np.concatenate(pred_all, axis=0) if pred_all else np.array([], dtype=int)
    prob_np = np.concatenate(probs_all, axis=0) if probs_all else np.zeros((0, 3), dtype=np.float32)
    aux = {
        "rest_load_probs": np.concatenate(rl_probs_all, axis=0) if rl_probs_all else np.zeros((0, 2), dtype=np.float32),
        "low_high_probs": np.concatenate(lh_probs_all, axis=0) if lh_probs_all else np.zeros((0, 2), dtype=np.float32),
    }
    return y_np, pred_np, prob_np, pd.DataFrame(batch_rows), aux


def _postprocess_chunks_3class(y_true: np.ndarray, prob: np.ndarray, *, seconds: float, window_shift_seconds: float):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    prob = np.asarray(prob, dtype=np.float32)
    windows_per_chunk = max(1, int(round(float(seconds) / max(1e-6, float(window_shift_seconds)))))
    rows, y_chunk, pred_chunk, prob_chunk = [], [], [], []
    for chunk_id, st in enumerate(range(0, y_true.size, windows_per_chunk)):
        en = min(y_true.size, st + windows_per_chunk)
        yy = y_true[st:en]
        pp = prob[st:en]
        p_mean = pp.mean(axis=0)
        counts = np.bincount(yy, minlength=3)
        tied = np.flatnonzero(counts == counts.max())
        if len(tied) > 1:
            y_maj = int(yy[len(yy) // 2])
            tie = 1
        else:
            y_maj = int(tied[0])
            tie = 0
        pred = int(p_mean.argmax())
        y_chunk.append(y_maj)
        pred_chunk.append(pred)
        prob_chunk.append(p_mean)
        rows.append(
            {
                "chunk_id": int(chunk_id),
                "start_window": int(st),
                "end_window": int(en - 1),
                "n_windows": int(en - st),
                "true_label": int(y_maj),
                "pred_label": int(pred),
                "tie_true_label": int(tie),
                "p_rest": float(p_mean[0]),
                "p_low": float(p_mean[1]),
                "p_high": float(p_mean[2]),
            }
        )
    return np.asarray(y_chunk, dtype=int), np.asarray(pred_chunk, dtype=int), np.vstack(prob_chunk).astype(np.float32) if prob_chunk else np.zeros((0, 3), dtype=np.float32), pd.DataFrame(rows)


def _save_confusion_3class(out_dir: Path, subject: str, variant: str, scope: str, y: np.ndarray, pred: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y, pred, labels=list(REST_LOW_HIGH_LABELS))
    pd.DataFrame(cm, index=REST_LOW_HIGH_NAMES, columns=REST_LOW_HIGH_NAMES).to_csv(out_dir / f"{subject}__{variant}__{scope}.csv")
    denom = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    pd.DataFrame(cm.astype(float) / denom, index=REST_LOW_HIGH_NAMES, columns=REST_LOW_HIGH_NAMES).to_csv(
        out_dir / f"{subject}__{variant}__{scope}_normalized.csv"
    )


def _rest_threshold_from_val(model, head, fusion, val_loader, device, args, hierarchical: bool) -> float:
    mode = str(getattr(args, "rest_threshold", "argmax")).lower().strip()
    if mode != "auto" or val_loader is None or not hierarchical:
        return float("nan")
    y, _pred, prob, _batch_df, _aux = _predict_3class_mode(
        model, head, fusion, None, val_loader, "none", device, args, hierarchical=hierarchical, rest_threshold=None
    )
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        load_pred = 1 + np.argmax(prob[:, 1:3], axis=1)
        pred = np.where(prob[:, 0] >= float(t), 0, load_pred).astype(int)
        score = float(fixed_label_metrics(y, pred)["macro_f1_fixed"])
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t


def mwl_post_eval_hook(model, sbj: str, subjects: List[str], train_loader, test_loader, device: str, args, sbj_dir: Path):
    if not bool(getattr(args, "run_mwl_head", True)):
        return []
    if str(getattr(args, "mwl_task", "rest_low_high")) != "rest_low_high":
        raise ValueError("vit_profile_tcn_mwl_hierarchical_3class.py only supports --mwl-task rest_low_high")
    _warm_lazy_modules(model, train_loader, device)
    full_subjects = list(getattr(args, "full_subjects_for_loso", None) or subjects)
    variants = _parse_variants_3class(getattr(args, "variants", ""))
    postprocess_seconds = _parse_seconds_list(getattr(args, "postprocess_seconds", getattr(args, "mwl_postprocess_seconds", "60 120")))
    modes = _parse_modes(args.mwl_ttt_modes)
    default_mode = "none"
    rr_model = None
    if any(("qkv" in v) for v in variants) or any(m != "none" for m in modes):
        if any(m != "none" for m in modes):
            rr_model = _train_rr_probe_for_ttt(model, train_loader, device, args)
    qkv_scale_diag = _apply_qkv_effective_scale(model, args)
    chunk_dir = Path(args.out_dir) / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir = Path(args.out_dir) / "confusion_matrices"
    rows, window_rows, one_min_rows, two_min_rows, ttt_rows, count_rows = [], [], [], [], [], []

    for variant in variants:
        hierarchical = variant.startswith("hier")
        head_variant = "A0" if variant == "flat_a0" else "A2"
        hrv_mode = "none"
        use_fusion_requested = False
        eval_modes = [default_mode]
        if variant == "hier_a2_qkv_ttt":
            eval_modes = [m for m in modes if m != "none"] or ["profile_qkv_ttt_sample"]
        if variant == "hier_a2_hrv":
            use_fusion_requested = bool(getattr(args, "use_hrv_fusion", False))
            hrv_modes = [m for m in _parse_hrv_feature_modes(str(getattr(args, "hrv_feature_modes", "none"))) if m != "none"]
            hrv_mode = hrv_modes[0] if hrv_modes else str(getattr(args, "hrv_feature_mode", "hr_hrv_relative"))

        cls_train_loader, cls_val_loader, cls_test_loader, cls_meta = _build_mwl_classification_loaders(
            subject=sbj,
            subjects=full_subjects,
            data_str=args.data_str,
            batch_size=int(args.mwl_batch_size),
            data_dir=str(args.data_dir),
            train_data_group="mr_levels",
            test_data_group="mr_levels",
            task="mr_levels",
            include_levels_in_train=True,
            class_subset="",
            class_groups=REST_LOW_HIGH_GROUPS,
            val_subjects=int(getattr(args, "mwl_val_subjects", 3)),
            seed=int(getattr(args, "seed", 0)),
            hrv_feature_mode=hrv_mode,
            hrv_input_source=str(getattr(args, "hrv_input_source", "auto")),
            hrv_min_valid_ratio=float(getattr(args, "hrv_min_valid_ratio", 0.5)),
        )
        count_rows.extend([{**r, "variant": variant} for r in cls_meta.get("class_counts_by_subject", [])])
        use_fusion = bool(use_fusion_requested and hrv_mode != "none" and int(cls_meta["hrv_dim"]) > 0)
        if hierarchical:
            head = HierarchicalA2MWLHead(args).to(device)
            fusion = None
            if use_fusion:
                fusion = HierarchicalHRVFusionClassifier(
                    int(args.mwl_tcn_hidden_dim),
                    int(cls_meta["hrv_dim"]),
                    hrv_hidden_dim=int(getattr(args, "hrv_hidden_dim", 16)),
                    dropout=float(getattr(args, "hrv_dropout", 0.3)),
                ).to(device)
        else:
            head = _build_mwl_head(head_variant, args, 3).to(device)
            fusion = None
            if use_fusion:
                fusion = HRVFusionClassifier(
                    _mwl_head_feature_dim(head_variant, args),
                    int(cls_meta["hrv_dim"]),
                    3,
                    hrv_hidden_dim=int(getattr(args, "hrv_hidden_dim", 16)),
                    dropout=float(getattr(args, "hrv_dropout", 0.3)),
                ).to(device)
        train_last = _train_3class_head(model, head, fusion, cls_train_loader, cls_val_loader, device, args, hierarchical=hierarchical)
        selected_threshold = _rest_threshold_from_val(model, head, fusion, cls_val_loader, device, args, hierarchical)
        threshold_for_pred = selected_threshold if math.isfinite(selected_threshold) else None

        for mode in eval_modes:
            effective_mode = str(mode)
            y, pred, prob, batch_df, aux = _predict_3class_mode(
                model, head, fusion, rr_model, cls_test_loader, effective_mode, device, args, hierarchical=hierarchical, rest_threshold=threshold_for_pred
            )
            if not set(np.unique(y).astype(int)).issubset({0, 1, 2}):
                raise RuntimeError(f"Unexpected labels after rest_low_high mapping: {sorted(set(np.unique(y).astype(int)))}")
            _save_confusion_3class(confusion_dir, sbj, variant, "window", y, pred)
            chunk_outputs = {}
            for seconds in postprocess_seconds:
                scope = "one_min" if int(round(seconds)) == 60 else "two_min" if int(round(seconds)) == 120 else f"{int(round(seconds))}s"
                y_c, pred_c, prob_c, chunk_df = _postprocess_chunks_3class(
                    y, prob, seconds=seconds, window_shift_seconds=float(args.mwl_window_shift_seconds)
                )
                _save_confusion_3class(confusion_dir, sbj, variant, scope, y_c, pred_c)
                chunk_outputs[scope] = (y_c, pred_c, prob_c, chunk_df, seconds)

            target_counts = np.bincount(y, minlength=3)
            row = {
                "__summary_name__": "mwl_diagnostic_summary",
                "subject": sbj,
                "variant": variant,
                "head": "hierarchical_a2" if hierarchical else {"A0": "flat_a0", "A2": "flat_a2"}[head_variant],
                "mode": effective_mode,
                "mwl_task": "rest_low_high",
                "hrv_feature_mode": hrv_mode,
                "use_hrv_fusion": int(use_fusion),
                "window_shift_seconds": float(args.mwl_window_shift_seconds),
                "postprocess_seconds": " ".join(str(int(s)) if float(s).is_integer() else str(s) for s in postprocess_seconds),
                "best_epoch": int(train_last.get("best_epoch", 0)),
                "best_val_accuracy": float(train_last.get("best_val_accuracy", float("nan"))),
                "best_val_balanced_accuracy_fixed": float(train_last.get("best_val_balanced_accuracy_fixed", float("nan"))),
                "best_val_macro_f1_fixed": float(train_last.get("best_val_macro_f1_fixed", float("nan"))),
                "train_accuracy": float(train_last.get("train_accuracy", float("nan"))),
                "train_macro_f1_fixed": float(train_last.get("train_macro_f1_fixed", float("nan"))),
                "selected_rest_threshold": float(selected_threshold) if math.isfinite(selected_threshold) else float("nan"),
                "target_has_rest": int(target_counts[0] > 0),
                "target_has_low": int(target_counts[1] > 0),
                "target_has_high": int(target_counts[2] > 0),
                "target_has_all_classes": int(np.all(target_counts[:3] > 0)),
                "mwl_train_subjects": " ".join(str(v) for v in cls_meta["train_subjects"]),
                "mwl_val_subjects": " ".join(str(v) for v in cls_meta["val_subjects"]),
                "mwl_uses_source_subject_validation": int(cls_meta["uses_source_subject_validation"]),
                **_prefixed_fixed_metrics(y, pred, "window"),
                **qkv_scale_diag,
            }
            for scope, (y_c, pred_c, _prob_c, _chunk_df, seconds) in chunk_outputs.items():
                prefix = scope
                row.update(_prefixed_fixed_metrics(y_c, pred_c, prefix))
                row[f"{scope}_windows_per_chunk"] = int(round(float(seconds) / max(1e-6, float(args.mwl_window_shift_seconds))))
            rows.append(row)

            rl = aux["rest_load_probs"]
            lh = aux["low_high_probs"]
            for i in range(y.size):
                window_rows.append(
                    {
                        "subject": sbj,
                        "variant": variant,
                        "head": row["head"],
                        "mode": effective_mode,
                        "sample_index": int(i),
                        "true_label": int(y[i]),
                        "pred_label": int(pred[i]),
                        "p_rest": float(prob[i, 0]),
                        "p_low": float(prob[i, 1]),
                        "p_high": float(prob[i, 2]),
                        "p_rest_load_rest": float(rl[i, 0]) if rl.size else float("nan"),
                        "p_rest_load_load": float(rl[i, 1]) if rl.size else float("nan"),
                        "p_low_high_low": float(lh[i, 0]) if lh.size else float("nan"),
                        "p_low_high_high": float(lh[i, 1]) if lh.size else float("nan"),
                        "condition_original": ["R", "low_L0_L1", "high_L2_L3"][int(y[i])],
                        "chunk_id_optional": int(i),
                        "hrv_feature_mode": hrv_mode,
                    }
                )
            for scope, (_y_c, _pred_c, _prob_c, chunk_df, _seconds) in chunk_outputs.items():
                chunk_df = chunk_df.copy()
                chunk_df.insert(0, "mode", effective_mode)
                chunk_df.insert(0, "head", row["head"])
                chunk_df.insert(0, "variant", variant)
                chunk_df.insert(0, "subject", sbj)
                chunk_df["hrv_feature_mode"] = hrv_mode
                if scope == "one_min":
                    one_min_rows.extend(chunk_df.to_dict(orient="records"))
                elif scope == "two_min":
                    two_min_rows.extend(chunk_df.to_dict(orient="records"))
            if not batch_df.empty:
                batch_df = batch_df.copy()
                batch_df.insert(0, "subject", sbj)
                batch_df.insert(1, "variant", variant)
                ttt_rows.extend(batch_df.to_dict(orient="records"))
            print(
                f"[MWL_3C] subject={sbj} variant={variant} mode={effective_mode} "
                f"win_f1={row['window_macro_f1_fixed']:.4f} two_min_f1={row.get('two_min_macro_f1_fixed', float('nan')):.4f}"
            )

    if rows:
        df = pd.DataFrame([{k: v for k, v in row.items() if not str(k).startswith("__")} for row in rows])
        df.to_csv(chunk_dir / f"{sbj}_summary.csv", index=False)
        df.to_csv(chunk_dir / f"{sbj}_mwl_diagnostic_summary.csv", index=False)
        present = df[df["target_has_all_classes"] == 1].copy()
        if not present.empty:
            present.to_csv(chunk_dir / f"{sbj}_summary_all_classes_present.csv", index=False)
    if count_rows:
        pd.DataFrame(count_rows).drop_duplicates().to_csv(chunk_dir / f"{sbj}_class_counts_by_subject.csv", index=False)
    if window_rows:
        pd.DataFrame(window_rows).to_csv(chunk_dir / f"{sbj}_mwl_predictions_window.csv", index=False)
    if one_min_rows:
        pd.DataFrame(one_min_rows).to_csv(chunk_dir / f"{sbj}_mwl_predictions_one_min.csv", index=False)
    if two_min_rows:
        pd.DataFrame(two_min_rows).to_csv(chunk_dir / f"{sbj}_mwl_predictions_two_min.csv", index=False)
    if ttt_rows:
        pd.DataFrame(ttt_rows).to_csv(chunk_dir / f"{sbj}_mwl_ttt_batches.csv", index=False)
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
    parser.add_argument("--mwl-task", default="rest_low_high", choices=sorted(set(TASK_CHOICES) | {"rest_low_high"}))
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
    parser.add_argument("--variants", nargs="+", default=["flat_a0", "flat_a2", "hier_a2", "hier_a2_smooth"])
    parser.add_argument("--mwl-ttt-modes", default="none")
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
    parser.add_argument(
        "--mwl-monitor",
        default="val_macro_f1_fixed",
        choices=[
            "val_macro_f1_fixed",
            "val_balanced_accuracy_fixed",
            "val_macro_f1",
            "val_f1_macro",
            "val_balanced_accuracy",
            "val_balanced_acc",
            "val_accuracy",
            "val_acc",
            "val_loss",
        ],
    )
    parser.add_argument("--lambda-low-high", type=float, default=1.0)
    parser.add_argument("--rest-threshold", default="argmax", help="Use 'auto' to tune rest threshold on source-validation subjects.")
    parser.add_argument("--postprocess-seconds", nargs="+", default=["60", "120"])
    parser.add_argument("--mwl-postprocess-seconds", type=float, default=60.0)
    parser.add_argument("--mwl-window-shift-seconds", type=float, default=30.0)
    parser.add_argument("--use-hrv-fusion", action="store_true")
    parser.add_argument("--hrv-feature-mode", choices=["none", "hr_only", "hrv_basic", "hr_hrv_absolute", "hr_hrv_relative", "hr_hrv_absolute_plus_relative"], default="none")
    parser.add_argument(
        "--hrv-feature-modes",
        default="none",
        help="Comma-separated HRV ablation modes. Overrides --hrv-feature-mode when set to multiple values.",
    )
    parser.add_argument("--hrv-hidden-dim", type=int, default=16)
    parser.add_argument("--hrv-dropout", type=float, default=0.3)
    parser.add_argument("--hrv-normalization", choices=["source", "target_relative", "source_plus_target_relative"], default="target_relative")
    parser.add_argument("--hrv-min-valid-ratio", type=float, default=0.5)
    parser.add_argument("--hrv-input-source", choices=["auto", "br", "hr", "ibi", "ppg", "ecg"], default="auto")
    parser.add_argument("--save-confusion-matrices", action="store_true")
    parser.add_argument("--save-ttt-diagnostics", action="store_true")


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
    hrv_modes = _parse_hrv_feature_modes(str(getattr(args, "hrv_feature_modes", "none")))
    if hrv_modes == ["none"] and str(getattr(args, "hrv_feature_mode", "none")) != "none":
        args.hrv_feature_modes = str(args.hrv_feature_mode)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_base_parser(SUBJECTS, "vit_profile_tcn_mwl_hierarchical_3class")
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
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    run_loocv_experiment(args, post_eval_hooks=[mwl_post_eval_hook], config_mutator=config_mutator)


if __name__ == "__main__":
    main()
