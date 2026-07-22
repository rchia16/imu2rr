#!/usr/bin/env python3
"""Shared rest/low/high MWL target mapping and LOSO array helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from dataloader import load_data


REST_LOW_HIGH_CLASS_NAMES = ("rest", "low", "high")
REST_LOW_HIGH_CLASS_IDS = (0, 1, 2)
REST_LOW_HIGH_RAW_TO_CLASS: Dict[str, int] = {
    "R": 0,
    "L0": 1,
    "L1": 1,
    "L2": 2,
    "L3": 2,
}
REST_LOW_HIGH_INCLUDED_RAW = ("R", "L0", "L1", "L2", "L3")
REST_LOW_HIGH_EXCLUDED_RAW = ("M", "meditation", "missing", "unknown", "outside_R_L0_L3")
REST_LOW_HIGH_GROUPS = (("rest", (1,)), ("low", (2, 3)), ("high", (4, 5)))


@dataclass(frozen=True)
class MWLSplitArrays:
    x: np.ndarray
    y: np.ndarray
    sample_id: np.ndarray
    raw_condition: np.ndarray
    subject_id: np.ndarray
    group: np.ndarray
    raw_index: np.ndarray


def mapping_manifest() -> Dict[str, object]:
    return {
        "target": "rest_low_high",
        "class_names": list(REST_LOW_HIGH_CLASS_NAMES),
        "class_ids": {name: idx for idx, name in enumerate(REST_LOW_HIGH_CLASS_NAMES)},
        "raw_label_mapping": dict(REST_LOW_HIGH_RAW_TO_CLASS),
        "included_raw_conditions": list(REST_LOW_HIGH_INCLUDED_RAW),
        "excluded_raw_conditions": list(REST_LOW_HIGH_EXCLUDED_RAW),
        "notes": "R->0/rest; L0,L1->1/low; L2,L3->2/high. M, missing, unknown, meditation and all other labels are excluded.",
    }


def parse_list(text: str | Sequence[str]) -> List[str]:
    if isinstance(text, (list, tuple)):
        out: List[str] = []
        for item in text:
            out.extend(parse_list(str(item)))
        return out
    return [part.strip() for part in str(text).replace(",", " ").split() if part.strip()]


def data_groups_for_rest_low_high(group: str) -> List[str]:
    normalized = str(group or "mr_levels").strip().lower()
    if normalized in {"mr_levels", "combined", "all"}:
        return ["mr", "levels"]
    if normalized in {"level", "levels"}:
        return ["levels"]
    if normalized == "mr":
        return ["mr"]
    raise ValueError(f"Unsupported MWL data group {group!r}; expected mr, levels, or mr_levels.")


def canonical_raw_condition(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode(errors="ignore")
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NONE", "NULL", "UNKNOWN"}:
        return None
    if text in {"REST", "R"}:
        return "R"
    if text in {"M", "MED", "MEDITATION"}:
        return "M"
    if text in {"L0", "L1", "L2", "L3"}:
        return text
    if text in {"0", "1", "2", "3"}:
        return f"L{text}"
    try:
        numeric = float(text)
        if numeric.is_integer() and int(numeric) in (0, 1, 2, 3):
            return f"L{int(numeric)}"
    except ValueError:
        pass
    return text


def map_raw_conditions_to_rest_low_high(raw_conditions: Iterable[object]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray([canonical_raw_condition(v) for v in raw_conditions], dtype=object)
    keep = np.asarray([str(v) in REST_LOW_HIGH_RAW_TO_CLASS for v in raw], dtype=bool)
    labels = np.asarray([REST_LOW_HIGH_RAW_TO_CLASS[str(v)] for v in raw[keep]], dtype=np.int64)
    return labels, keep, raw


def load_subject_rest_low_high_windows(
    subject: str,
    *,
    data_dir: str,
    data_str: str,
    data_groups: Sequence[str],
    strict: bool = True,
) -> MWLSplitArrays:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    sample_ids: List[np.ndarray] = []
    raw_conditions: List[np.ndarray] = []
    subject_ids: List[np.ndarray] = []
    groups: List[np.ndarray] = []
    raw_indices: List[np.ndarray] = []
    missing_groups: List[str] = []

    for group in data_groups:
        try:
            data = load_data(subject, data_dir=data_dir, data_group=group)
        except FileNotFoundError:
            missing_groups.append(str(group))
            continue
        if data_str not in data:
            raise KeyError(f"{subject} group={group} is missing data array {data_str!r}.")
        if "conds" not in data:
            raise KeyError(f"{subject} group={group} is missing raw condition labels under 'conds'.")
        x = np.asarray(data[data_str])
        cond_raw_all = np.asarray(data["conds"], dtype=object).reshape(-1)
        if x.shape[0] != cond_raw_all.shape[0]:
            raise ValueError(
                f"{subject} group={group}: {data_str} rows ({x.shape[0]}) do not match conds ({cond_raw_all.shape[0]})."
            )
        y, keep, raw_norm = map_raw_conditions_to_rest_low_high(cond_raw_all)
        if not bool(keep.any()):
            continue
        raw_idx = np.flatnonzero(keep).astype(np.int64)
        xs.append(x[keep])
        ys.append(y)
        sample_ids.append(np.asarray([f"{subject}:{group}:{int(i):06d}" for i in raw_idx], dtype=object))
        raw_conditions.append(raw_norm[keep])
        subject_ids.append(np.full(int(raw_idx.size), str(subject), dtype=object))
        groups.append(np.full(int(raw_idx.size), str(group), dtype=object))
        raw_indices.append(raw_idx)

    if missing_groups and strict:
        raise FileNotFoundError(f"{subject}: missing required data group(s): {missing_groups} under {Path(data_dir)}")
    if not xs:
        raise RuntimeError(f"{subject}: no rest_low_high windows found for groups={list(data_groups)}.")
    return MWLSplitArrays(
        x=np.concatenate(xs, axis=0),
        y=np.concatenate(ys, axis=0).astype(np.int64),
        sample_id=np.concatenate(sample_ids, axis=0),
        raw_condition=np.concatenate(raw_conditions, axis=0),
        subject_id=np.concatenate(subject_ids, axis=0),
        group=np.concatenate(groups, axis=0),
        raw_index=np.concatenate(raw_indices, axis=0).astype(np.int64),
    )


def split_source_subjects(subjects: Sequence[str], held_out: str, val_subjects: int, seed: int) -> Tuple[List[str], List[str]]:
    source = [str(s) for s in subjects if str(s) != str(held_out)]
    if str(held_out) not in {str(s) for s in subjects}:
        raise ValueError(f"Held-out subject {held_out!r} is not in the subject list.")
    n_val = max(0, min(int(val_subjects), max(0, len(source) - 1)))
    rng = np.random.default_rng(int(seed))
    shuffled = list(source)
    rng.shuffle(shuffled)
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])
    if str(held_out) in set(train) or str(held_out) in set(val):
        raise AssertionError("Held-out subject leaked into source train/validation split.")
    if not train:
        raise ValueError("Source train subject list is empty.")
    return train, val


def stack_subject_windows(
    subjects: Sequence[str],
    *,
    data_dir: str,
    data_str: str,
    data_groups: Sequence[str],
    strict: bool = True,
) -> MWLSplitArrays:
    parts = [
        load_subject_rest_low_high_windows(
            subject, data_dir=data_dir, data_str=data_str, data_groups=data_groups, strict=strict
        )
        for subject in subjects
    ]
    return MWLSplitArrays(
        x=np.concatenate([p.x for p in parts], axis=0),
        y=np.concatenate([p.y for p in parts], axis=0).astype(np.int64),
        sample_id=np.concatenate([p.sample_id for p in parts], axis=0),
        raw_condition=np.concatenate([p.raw_condition for p in parts], axis=0),
        subject_id=np.concatenate([p.subject_id for p in parts], axis=0),
        group=np.concatenate([p.group for p in parts], axis=0),
        raw_index=np.concatenate([p.raw_index for p in parts], axis=0).astype(np.int64),
    )


def assert_no_subject_overlap(train_subjects: Sequence[str], val_subjects: Sequence[str], test_subject: str) -> None:
    train = set(map(str, train_subjects))
    val = set(map(str, val_subjects))
    test = {str(test_subject)}
    if train & test or val & test:
        raise AssertionError(f"Held-out subject leaked into source splits: test={test_subject}")
    if train & val:
        raise AssertionError(f"Source train/validation overlap: {sorted(train & val)}")
