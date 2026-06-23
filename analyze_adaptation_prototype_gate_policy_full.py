#!/usr/bin/env python3
"""Prototype/OOD safety-gate analysis for fixed-alpha RR adaptation.

This script is the next-step experiment after the learned continuous alpha_hat
run.  It deliberately keeps the adaptation strengths hard-coded (for example
alpha=0.50 or 0.75) and asks a safer question:

    Does the incoming unlabeled target distribution belong to a source-learned
    region where fixed feature-mean adaptation is likely to help?

It reads the subject-level sweep table produced by the adaptation ladder
(`subject_rows.csv`) and evaluates leave-one-subject-out gates using only
strict no-label diagnostics as inputs.  True RR labels are used only to create
pseudo-target gate labels/gains on source subjects and to evaluate held-out
subjects.

Outputs:
  adaptation_mode_summary.csv
  adaptation_oracle_by_subject.csv
  adaptation_prototype_gate_by_subject.csv
  adaptation_prototype_gate_summary.csv
  adaptation_prototype_gate_features.json
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_CANDIDATES = [
    "none",
    "adapt_mean_alpha_050",
    "adapt_mean_alpha_075",
    "adapt_mean_alpha_100",
    "profile_film_init_only",
    "profile_film_unsup_sparc",
    "direct_stft_rr",
    "hybrid_probe_stft_conf",
]

DEFAULT_ALPHA_MODES = ["adapt_mean_alpha_050", "adapt_mean_alpha_075"]

# Columns that can be computed without target RR labels. The leakage filter below
# is intentionally conservative and is applied after this prefix allow-list.
ALLOWED_PREFIXES = (
    "feature_adapt_",
    "feature_align_",  # legacy alias; leaky diagnostics are filtered below
    "profile_stats_",
    "profile_vector_",
    "profile_gate_",
    "profile_delta_norm",
    "raw_profile_",
    "norm_profile_",
    "hybrid_",
    "unsup_",
    "readout_affine_",
    "rr_probe_n_",
    "eval_stft_confidence",
)

LEAKAGE_FILTER_VERSION = "strict_v4_prototype_gate_no_labels_no_residuals_no_overshoot"

LEAKY_SUBSTRINGS = (
    "mae",
    "rmse",
    "corr",
    "residual",
    "direction_dot",
    "direction_agree",
    "reduces_abs_error",
    "delta_mae",
    "overshoot",
    "oracle",
    "uses_target",
    "label",
    "y_true",
    "rr_true",
    "true_rr",
    "target_rr",
    "ground_truth",
    "base_pre",
    "candidate_eval",
    "eval_aux_rr_mae",
    "eval_stft_rr_mae",
)

LEAKY_NAMESPACES = (
    "eval_direction",
    "cal_direction",
    "profile_oracle",
)


@dataclass
class Standardizer:
    med: pd.Series
    mean: np.ndarray
    std: np.ndarray
    cols: List[str]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in self.cols})
        x = x.replace([np.inf, -np.inf], np.nan)
        x = x.fillna(self.med).to_numpy(dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return (x - self.mean.reshape(1, -1)) / self.std.reshape(1, -1)


def parse_space_list(text: str | Sequence[str]) -> List[str]:
    if isinstance(text, (list, tuple)):
        toks: List[str] = []
        for item in text:
            toks.extend(parse_space_list(str(item)))
        return toks
    return [t.strip() for t in str(text).replace(",", " ").split() if t.strip()]


def is_label_leaky_feature(name: str) -> bool:
    low = str(name).lower()
    if any(bad in low for bad in LEAKY_SUBSTRINGS):
        return True
    return any(low.startswith(ns) or ns in low for ns in LEAKY_NAMESPACES)


def candidate_no_label_feature_cols(df: pd.DataFrame) -> List[str]:
    return sorted({
        str(c)
        for c in df.columns
        if any(str(c).startswith(prefix) for prefix in ALLOWED_PREFIXES)
    })


def select_no_label_feature_cols(
    df: pd.DataFrame,
    *,
    min_non_na_frac: float,
) -> Tuple[List[str], List[str], List[str]]:
    selected: List[str] = []
    excluded_leaky: List[str] = []
    excluded_sparse: List[str] = []
    for name in candidate_no_label_feature_cols(df):
        if is_label_leaky_feature(name):
            excluded_leaky.append(name)
            continue
        values = pd.to_numeric(df[name], errors="coerce")
        if values.notna().mean() < float(min_non_na_frac):
            excluded_sparse.append(name)
            continue
        if values.nunique(dropna=True) <= 1:
            excluded_sparse.append(name)
            continue
        selected.append(name)
    offenders = sorted({c for c in selected if is_label_leaky_feature(c)})
    if offenders:
        raise RuntimeError("Leaky features passed strict filter: " + ", ".join(offenders))
    return sorted(selected), sorted(excluded_leaky), sorted(excluded_sparse)


def candidate_rows(df: pd.DataFrame, candidates: Sequence[str]) -> pd.DataFrame:
    required = {"subject", "mode", "rr_probe_pre_mae", "rr_probe_post_mae"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"subject_rows is missing required columns: {missing}")
    out = df[df["mode"].astype(str).isin(candidates)].copy()
    if out.empty:
        raise SystemExit("No subject rows found for requested candidates.")
    for c in ["rr_probe_pre_mae", "rr_probe_post_mae"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["rr_probe_pre_mae", "rr_probe_post_mae"]).copy()
    out["subject"] = out["subject"].astype(str)
    out["mode"] = out["mode"].astype(str)
    out["delta_mae"] = out["rr_probe_post_mae"] - out["rr_probe_pre_mae"]
    return out


def summarize_modes(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("mode", as_index=False)
        .agg(
            n_subjects=("subject", "nunique"),
            post_mae_mean=("rr_probe_post_mae", "mean"),
            delta_mae_mean=("delta_mae", "mean"),
            subjects_improved=("delta_mae", lambda x: int((x < 0).sum())),
            subjects_worse=("delta_mae", lambda x: int((x > 0).sum())),
        )
        .sort_values(["post_mae_mean", "delta_mae_mean", "mode"])
    )


def oracle_by_subject(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for subject, sub in df.groupby("subject", sort=True):
        best = sub.sort_values("rr_probe_post_mae").iloc[0]
        none = sub[sub["mode"] == "none"]
        none_mae = float(none["rr_probe_post_mae"].iloc[0]) if not none.empty else float(sub["rr_probe_pre_mae"].iloc[0])
        rows.append({
            "subject": subject,
            "oracle_mode": str(best["mode"]),
            "oracle_post_mae": float(best["rr_probe_post_mae"]),
            "none_post_mae": none_mae,
            "oracle_delta_vs_none": float(best["rr_probe_post_mae"] - none_mae),
        })
    return pd.DataFrame(rows)


def subject_mode_map(df: pd.DataFrame) -> Dict[Tuple[str, str], pd.Series]:
    rows: Dict[Tuple[str, str], pd.Series] = {}
    # If duplicate subject/mode rows remain, keep the last one as the summarizer does.
    for _, row in df.drop_duplicates(["subject", "mode"], keep="last").iterrows():
        rows[(str(row["subject"]), str(row["mode"]))] = row
    return rows


def fit_standardizer(train_rows: pd.DataFrame, feature_cols: Sequence[str]) -> Standardizer:
    x_df = pd.DataFrame({c: pd.to_numeric(train_rows[c], errors="coerce") for c in feature_cols})
    x_df = x_df.replace([np.inf, -np.inf], np.nan)
    med = x_df.median(axis=0, skipna=True).fillna(0.0)
    x = x_df.fillna(med).to_numpy(dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-6
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    std = np.maximum(std, 1e-6)
    return Standardizer(med=med, mean=mean, std=std, cols=list(feature_cols))


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    mu = x.mean(axis=0)
    sd = x.std(axis=0) + 1e-6
    sd = np.nan_to_num(sd, nan=1.0, posinf=1.0, neginf=1.0)
    sd = np.maximum(sd, 1e-6)
    xs = (x - mu.reshape(1, -1)) / sd.reshape(1, -1)
    xs = np.nan_to_num(xs, nan=0.0, posinf=0.0, neginf=0.0)
    design = np.column_stack([np.ones(xs.shape[0]), xs])
    reg = np.eye(design.shape[1]) * float(alpha)
    reg[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + reg, design.T @ y)
    return coef, mu, sd


def ridge_predict(model: Tuple[np.ndarray, np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    coef, mu, sd = model
    xx = np.asarray(x, dtype=np.float64)
    xx = np.nan_to_num(xx, nan=0.0, posinf=0.0, neginf=0.0)
    xs = (xx - mu.reshape(1, -1)) / sd.reshape(1, -1)
    xs = np.nan_to_num(xs, nan=0.0, posinf=0.0, neginf=0.0)
    design = np.column_stack([np.ones(xs.shape[0]), xs])
    return design @ coef


def euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.sqrt(np.maximum(((a[:, None, :] - b[None, :, :]) ** 2).mean(axis=2), 0.0))


def entropy_from_weights(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    w = w[np.isfinite(w) & (w > 0)]
    if w.size <= 1:
        return 0.0
    w = w / max(1e-12, w.sum())
    return float(-(w * np.log(w + 1e-12)).sum() / math.log(w.size))


def leaveone_nearest_distances(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] <= 1:
        return np.asarray([np.inf], dtype=np.float64)
    d = euclidean(x, x)
    np.fill_diagonal(d, np.inf)
    return np.min(d, axis=1)


def prototype_descriptor(
    *,
    x_train: np.ndarray,
    gain_train: np.ndarray,
    safe_train: np.ndarray,
    x_test: np.ndarray,
    subjects_train: Sequence[str],
    knn_k: int,
    ood_quantile: float,
) -> Dict[str, object]:
    """Build source-prototype/OOD descriptors for one held-out target."""
    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64).reshape(1, -1)
    gain_train = np.asarray(gain_train, dtype=np.float64).reshape(-1)
    safe_train = np.asarray(safe_train, dtype=bool).reshape(-1)
    subjects_train = list(subjects_train)
    if x_train.shape[0] == 0:
        return {
            "proto_nearest_dist": float("nan"),
            "proto_knn_safe_frac": float("nan"),
            "proto_soft_safe_prob": float("nan"),
            "proto_soft_gain": float("nan"),
            "proto_entropy": float("nan"),
            "proto_ood_score": float("nan"),
            "proto_ood_reject": 1,
            "proto_nearest_subject": "",
            "proto_features": np.zeros((1, 10), dtype=np.float64),
        }

    dist = euclidean(x_test, x_train).reshape(-1)
    order = np.argsort(dist)
    k = max(1, min(int(knn_k), dist.size))
    nn_idx = order[:k]
    nearest = int(order[0])
    nearest_dist = float(dist[nearest])
    second_dist = float(dist[order[1]]) if dist.size >= 2 else float("nan")
    margin = second_dist - nearest_dist if np.isfinite(second_dist) else float("nan")

    train_leaveone_nn = leaveone_nearest_distances(x_train)
    finite_train_nn = train_leaveone_nn[np.isfinite(train_leaveone_nn)]
    if finite_train_nn.size:
        ood_threshold = float(np.quantile(finite_train_nn, float(ood_quantile)))
    else:
        ood_threshold = float("inf")
    ood_reject = bool(np.isfinite(nearest_dist) and np.isfinite(ood_threshold) and nearest_dist > ood_threshold)

    temp_vals = dist[np.isfinite(dist)]
    temp = float(np.median(temp_vals)) if temp_vals.size else 1.0
    temp = max(temp, 1e-6)
    weights = np.exp(-dist / temp)
    weights = weights / max(1e-12, float(weights.sum()))

    safe_frac = float(np.mean(safe_train[nn_idx])) if nn_idx.size else 0.0
    knn_gain = float(np.mean(gain_train[nn_idx])) if nn_idx.size else 0.0
    soft_safe_prob = float(np.sum(weights * safe_train.astype(float)))
    soft_gain = float(np.sum(weights * gain_train))
    ent = entropy_from_weights(weights)

    safe_x = x_train[safe_train]
    harm_x = x_train[~safe_train]
    if safe_x.shape[0]:
        safe_centroid = safe_x.mean(axis=0, keepdims=True)
        dist_safe_centroid = float(euclidean(x_test, safe_centroid)[0, 0])
    else:
        dist_safe_centroid = float("nan")
    if harm_x.shape[0]:
        harm_centroid = harm_x.mean(axis=0, keepdims=True)
        dist_harm_centroid = float(euclidean(x_test, harm_centroid)[0, 0])
    else:
        dist_harm_centroid = float("nan")
    centroid_margin_safe = (
        dist_harm_centroid - dist_safe_centroid
        if np.isfinite(dist_harm_centroid) and np.isfinite(dist_safe_centroid)
        else float("nan")
    )
    global_rms_z = float(np.sqrt(np.mean(x_test * x_test)))

    proto_vec = np.asarray([
        nearest_dist,
        second_dist if np.isfinite(second_dist) else nearest_dist,
        margin if np.isfinite(margin) else 0.0,
        safe_frac,
        knn_gain,
        soft_safe_prob,
        soft_gain,
        ent,
        dist_safe_centroid if np.isfinite(dist_safe_centroid) else nearest_dist,
        dist_harm_centroid if np.isfinite(dist_harm_centroid) else nearest_dist,
        centroid_margin_safe if np.isfinite(centroid_margin_safe) else 0.0,
        global_rms_z,
        float(ood_threshold) if np.isfinite(ood_threshold) else 1e6,
        float(ood_reject),
    ], dtype=np.float64).reshape(1, -1)

    return {
        "proto_nearest_dist": nearest_dist,
        "proto_second_nearest_dist": second_dist,
        "proto_nearest_margin": margin,
        "proto_knn_safe_frac": safe_frac,
        "proto_knn_gain": knn_gain,
        "proto_soft_safe_prob": soft_safe_prob,
        "proto_soft_gain": soft_gain,
        "proto_entropy": ent,
        "proto_dist_safe_centroid": dist_safe_centroid,
        "proto_dist_harm_centroid": dist_harm_centroid,
        "proto_centroid_margin_safe": centroid_margin_safe,
        "proto_ood_score": global_rms_z,
        "proto_ood_threshold": float(ood_threshold) if np.isfinite(ood_threshold) else float("nan"),
        "proto_ood_reject": int(ood_reject),
        "proto_nearest_subject": subjects_train[nearest] if nearest < len(subjects_train) else "",
        "proto_features": proto_vec,
    }


def mode_post_mae(row_map: Mapping[Tuple[str, str], pd.Series], subject: str, mode: str) -> float:
    row = row_map.get((subject, mode))
    if row is None:
        return float("nan")
    return float(row["rr_probe_post_mae"])


def row_for_mode(row_map: Mapping[Tuple[str, str], pd.Series], subject: str, mode: str) -> Optional[pd.Series]:
    return row_map.get((subject, mode))


def build_policy_rows_for_alpha(
    *,
    df: pd.DataFrame,
    row_map: Mapping[Tuple[str, str], pd.Series],
    subjects: Sequence[str],
    alpha_mode: str,
    feature_cols: Sequence[str],
    args,
) -> List[Dict[str, object]]:
    rows_out: List[Dict[str, object]] = []
    for heldout in subjects:
        train_subjects = [s for s in subjects if s != heldout]
        train_rows_list = [row_for_mode(row_map, s, alpha_mode) for s in train_subjects]
        test_row = row_for_mode(row_map, heldout, alpha_mode)
        none_row = row_for_mode(row_map, heldout, "none")
        if test_row is None or none_row is None or any(r is None for r in train_rows_list):
            continue
        train_rows = pd.DataFrame(train_rows_list)
        test_rows = pd.DataFrame([test_row])

        # Source pseudo-target labels/gains.
        gain_train = []
        safe_train = []
        for s in train_subjects:
            none_mae = mode_post_mae(row_map, s, "none")
            alpha_mae = mode_post_mae(row_map, s, alpha_mode)
            gain = none_mae - alpha_mae
            gain_train.append(gain)
            safe_train.append(gain >= float(args.min_gain))
        gain_train_np = np.asarray(gain_train, dtype=np.float64)
        safe_train_np = np.asarray(safe_train, dtype=bool)

        stdzr = fit_standardizer(train_rows, feature_cols)
        x_train = stdzr.transform(train_rows)
        x_test = stdzr.transform(test_rows)

        proto_test = prototype_descriptor(
            x_train=x_train,
            gain_train=gain_train_np,
            safe_train=safe_train_np,
            x_test=x_test.reshape(-1),
            subjects_train=train_subjects,
            knn_k=int(args.knn_k),
            ood_quantile=float(args.ood_quantile),
        )

        # Train a second, simple gain regressor on no-label features plus source-prototype descriptors.
        proto_train_vecs: List[np.ndarray] = []
        for i, s in enumerate(train_subjects):
            sub_train_x = np.delete(x_train, i, axis=0)
            sub_gain = np.delete(gain_train_np, i, axis=0)
            sub_safe = np.delete(safe_train_np, i, axis=0)
            sub_subjects = [ss for j, ss in enumerate(train_subjects) if j != i]
            if sub_train_x.shape[0] == 0:
                # Degenerate fallback with zeros for tiny smoke tests.
                proto_train_vecs.append(np.zeros((1, 14), dtype=np.float64))
                continue
            desc = prototype_descriptor(
                x_train=sub_train_x,
                gain_train=sub_gain,
                safe_train=sub_safe,
                x_test=x_train[i],
                subjects_train=sub_subjects,
                knn_k=int(args.knn_k),
                ood_quantile=float(args.ood_quantile),
            )
            proto_train_vecs.append(np.asarray(desc["proto_features"], dtype=np.float64))
        proto_train = np.concatenate(proto_train_vecs, axis=0) if proto_train_vecs else np.zeros((x_train.shape[0], 14), dtype=np.float64)
        proto_test_x = np.asarray(proto_test["proto_features"], dtype=np.float64)

        x_train_aug = np.concatenate([x_train, proto_train], axis=1)
        x_test_aug = np.concatenate([x_test, proto_test_x], axis=1)
        ridge_model = ridge_fit(x_train_aug, gain_train_np, float(args.ridge_alpha))
        pred_gain_ridge = float(ridge_predict(ridge_model, x_test_aug)[0])

        none_mae = float(none_row["rr_probe_post_mae"])
        alpha_mae = float(test_row["rr_probe_post_mae"])
        actual_gain = none_mae - alpha_mae
        actual_safe = actual_gain >= float(args.min_gain)

        # Gate 1: pure source-prototype/KNN safety gate.
        proto_safe = (
            float(proto_test["proto_soft_safe_prob"]) >= float(args.safe_threshold)
            and float(proto_test["proto_soft_gain"]) >= float(args.min_gain)
        )
        if bool(args.reject_ood):
            proto_safe = proto_safe and not bool(proto_test["proto_ood_reject"])

        # Gate 2: ridge predicted-gain gate with OOD veto.
        ridge_safe = pred_gain_ridge >= float(args.min_gain)
        if bool(args.reject_ood):
            ridge_safe = ridge_safe and not bool(proto_test["proto_ood_reject"])

        for policy_name, passed, score in [
            (f"prototype_gate_{alpha_mode}", proto_safe, float(proto_test["proto_soft_gain"])),
            (f"prototype_ridge_gate_{alpha_mode}", ridge_safe, pred_gain_ridge),
        ]:
            selected_mode = alpha_mode if passed else "none"
            selected_mae = alpha_mae if passed else none_mae
            rows_out.append({
                "subject": heldout,
                "policy": policy_name,
                "alpha_mode": alpha_mode,
                "selected_mode": selected_mode,
                "selected_post_mae": selected_mae,
                "none_post_mae": none_mae,
                "alpha_post_mae": alpha_mae,
                "selected_delta_vs_none": selected_mae - none_mae,
                "actual_alpha_gain_vs_none": actual_gain,
                "actual_alpha_safe": int(actual_safe),
                "gate_pass": int(passed),
                "gate_score": float(score),
                "pred_gain_ridge": pred_gain_ridge,
                "proto_soft_gain": float(proto_test["proto_soft_gain"]),
                "proto_soft_safe_prob": float(proto_test["proto_soft_safe_prob"]),
                "proto_knn_safe_frac": float(proto_test["proto_knn_safe_frac"]),
                "proto_nearest_subject": str(proto_test["proto_nearest_subject"]),
                "proto_nearest_dist": float(proto_test["proto_nearest_dist"]),
                "proto_nearest_margin": float(proto_test["proto_nearest_margin"]),
                "proto_entropy": float(proto_test["proto_entropy"]),
                "proto_dist_safe_centroid": float(proto_test["proto_dist_safe_centroid"]),
                "proto_dist_harm_centroid": float(proto_test["proto_dist_harm_centroid"]),
                "proto_centroid_margin_safe": float(proto_test["proto_centroid_margin_safe"]),
                "proto_ood_score": float(proto_test["proto_ood_score"]),
                "proto_ood_threshold": float(proto_test["proto_ood_threshold"]),
                "proto_ood_reject": int(proto_test["proto_ood_reject"]),
                "min_gain": float(args.min_gain),
                "safe_threshold": float(args.safe_threshold),
                "reject_ood": int(bool(args.reject_ood)),
            })
    return rows_out


def add_best_alpha_policy(rows: pd.DataFrame, row_map: Mapping[Tuple[str, str], pd.Series], subjects: Sequence[str]) -> pd.DataFrame:
    out: List[Dict[str, object]] = []
    for gate_family in ["prototype_gate", "prototype_ridge_gate"]:
        for subject in subjects:
            sub = rows[(rows["subject"] == subject) & (rows["policy"].str.startswith(gate_family + "_adapt_mean_alpha_"))].copy()
            if sub.empty:
                continue
            none_mae = float(sub["none_post_mae"].iloc[0])
            # Choose the passing alpha with the best predicted gate score. If none pass, use none.
            passing = sub[sub["gate_pass"] == 1].copy()
            if passing.empty:
                selected_mode = "none"
                selected_mae = none_mae
                selected_score = float(sub["gate_score"].max()) if not sub.empty else float("nan")
                alpha_mode = "none"
            else:
                best = passing.sort_values(["gate_score", "alpha_mode"], ascending=[False, True]).iloc[0]
                selected_mode = str(best["alpha_mode"])
                selected_mae = float(best["alpha_post_mae"])
                selected_score = float(best["gate_score"])
                alpha_mode = str(best["alpha_mode"])
            out.append({
                "subject": subject,
                "policy": f"{gate_family}_best_alpha",
                "alpha_mode": alpha_mode,
                "selected_mode": selected_mode,
                "selected_post_mae": selected_mae,
                "none_post_mae": none_mae,
                "alpha_post_mae": selected_mae if selected_mode != "none" else float("nan"),
                "selected_delta_vs_none": selected_mae - none_mae,
                "actual_alpha_gain_vs_none": none_mae - selected_mae if selected_mode != "none" else 0.0,
                "actual_alpha_safe": int((none_mae - selected_mae) > 0.0),
                "gate_pass": int(selected_mode != "none"),
                "gate_score": selected_score,
            })
    return pd.concat([rows, pd.DataFrame(out)], axis=0, ignore_index=True, sort=False) if out else rows


def add_profile_fallback_policy(
    rows: pd.DataFrame,
    row_map: Mapping[Tuple[str, str], pd.Series],
    subjects: Sequence[str],
    profile_mode: str,
) -> pd.DataFrame:
    if not profile_mode:
        return rows
    out: List[Dict[str, object]] = []
    for base_policy in sorted(rows["policy"].dropna().astype(str).unique()):
        if not (base_policy.startswith("prototype_gate_best_alpha") or base_policy.startswith("prototype_ridge_gate_best_alpha")):
            continue
        for subject in subjects:
            base = rows[(rows["subject"] == subject) & (rows["policy"] == base_policy)]
            if base.empty:
                continue
            base_row = base.iloc[0]
            none_mae = float(base_row["none_post_mae"])
            profile_row = row_for_mode(row_map, subject, profile_mode)
            if profile_row is None:
                continue
            profile_mae = float(profile_row["rr_probe_post_mae"])
            # This fallback is intentionally conservative: use profile only when the
            # mean-adaptation gate rejected alpha and profile is available. The
            # decision to test it is analysis-side; true labels are not used to gate.
            if str(base_row["selected_mode"]) == "none":
                selected_mode = profile_mode
                selected_mae = profile_mae
                selected_kind = "profile_fallback"
            else:
                selected_mode = str(base_row["selected_mode"])
                selected_mae = float(base_row["selected_post_mae"])
                selected_kind = "alpha_gate"
            out.append({
                "subject": subject,
                "policy": f"{base_policy}_or_{profile_mode}",
                "alpha_mode": str(base_row.get("alpha_mode", "")),
                "selected_mode": selected_mode,
                "selected_post_mae": selected_mae,
                "none_post_mae": none_mae,
                "profile_post_mae": profile_mae,
                "selected_delta_vs_none": selected_mae - none_mae,
                "gate_pass": int(selected_kind == "alpha_gate"),
                "fallback_kind": selected_kind,
            })
    return pd.concat([rows, pd.DataFrame(out)], axis=0, ignore_index=True, sort=False) if out else rows


def summarize_policies(policy_rows: pd.DataFrame, mode_df: pd.DataFrame, oracle_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if policy_rows.empty:
        return pd.DataFrame()
    oracle_map = oracle_df.set_index("subject")["oracle_post_mae"].to_dict() if not oracle_df.empty else {}
    for policy, sub in policy_rows.groupby("policy", sort=True):
        sub = sub.copy()
        oracle_vals = np.asarray([oracle_map.get(s, np.nan) for s in sub["subject"].astype(str)], dtype=float)
        gate_pass = pd.to_numeric(sub.get("gate_pass", pd.Series(index=sub.index, dtype=float)), errors="coerce").fillna(0).astype(int)
        actual_safe = pd.to_numeric(sub.get("actual_alpha_safe", pd.Series(index=sub.index, dtype=float)), errors="coerce")
        actual_gain = pd.to_numeric(sub.get("actual_alpha_gain_vs_none", pd.Series(index=sub.index, dtype=float)), errors="coerce")
        has_safety_labels = bool(actual_safe.notna().any())
        if has_safety_labels:
            safe_int = actual_safe.fillna(0).astype(int)
            true_safe = int((safe_int == 1).sum())
            true_harmful = int((safe_int == 0).sum())
            false_accepts = int(((gate_pass == 1) & (safe_int == 0)).sum())
            false_rejects = int(((gate_pass == 0) & (safe_int == 1)).sum())
            true_accepts = int(((gate_pass == 1) & (safe_int == 1)).sum())
            true_rejects = int(((gate_pass == 0) & (safe_int == 0)).sum())
            harm_avoidance_rate = true_rejects / max(1, true_harmful)
            missed_opportunity_rate = false_rejects / max(1, true_safe)
            false_accept_mean_harm = float((-actual_gain[(gate_pass == 1) & (safe_int == 0)]).mean()) if false_accepts else 0.0
        else:
            true_safe = true_harmful = false_accepts = false_rejects = true_accepts = true_rejects = 0
            harm_avoidance_rate = missed_opportunity_rate = float("nan")
            false_accept_mean_harm = float("nan")
        rows.append({
            "policy": policy,
            "n_subjects": int(sub["subject"].nunique()),
            "post_mae_mean": float(sub["selected_post_mae"].mean()),
            "delta_vs_none_mean": float(sub["selected_delta_vs_none"].mean()),
            "subjects_improved_vs_none": int((sub["selected_delta_vs_none"] < 0).sum()),
            "subjects_worse_vs_none": int((sub["selected_delta_vs_none"] > 0).sum()),
            "gate_pass_rate": float(gate_pass.mean()),
            "oracle_regret_mean": float(np.nanmean(sub["selected_post_mae"].to_numpy(dtype=float) - oracle_vals)),
            "true_safe_alpha_subjects": true_safe,
            "true_harmful_alpha_subjects": true_harmful,
            "true_accepts": true_accepts,
            "false_accepts": false_accepts,
            "true_rejects": true_rejects,
            "false_rejects": false_rejects,
            "harm_avoidance_rate": float(harm_avoidance_rate),
            "missed_opportunity_rate": float(missed_opportunity_rate),
            "false_accept_mean_harm_bpm_mae": false_accept_mean_harm,
        })
    # Add fixed mode baselines into same summary for easy comparison.
    for _, row in mode_df.iterrows():
        rows.append({
            "policy": str(row["mode"]),
            "n_subjects": int(row["n_subjects"]),
            "post_mae_mean": float(row["post_mae_mean"]),
            "delta_vs_none_mean": float(row["delta_mae_mean"]),
            "subjects_improved_vs_none": int(row["subjects_improved"]),
            "subjects_worse_vs_none": int(row["subjects_worse"]),
            "gate_pass_rate": float("nan"),
            "oracle_regret_mean": float("nan"),
        })
    return pd.DataFrame(rows).sort_values(["post_mae_mean", "delta_vs_none_mean", "policy"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Prototype/OOD safety-gate analysis for fixed alpha adaptation.")
    ap.add_argument("--subject-rows", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--candidates", default=" ".join(DEFAULT_CANDIDATES))
    ap.add_argument("--alpha-modes", default=" ".join(DEFAULT_ALPHA_MODES))
    ap.add_argument("--profile-mode", default="profile_film_init_only")
    ap.add_argument("--include-profile-fallback", action="store_true", default=True)
    ap.add_argument("--no-profile-fallback", dest="include_profile_fallback", action="store_false")
    ap.add_argument("--min-gain", type=float, default=0.02, help="Minimum MAE improvement needed to call an alpha safe/helpful.")
    ap.add_argument("--safe-threshold", type=float, default=0.55, help="Prototype soft-safe probability threshold.")
    ap.add_argument("--knn-k", type=int, default=3)
    ap.add_argument("--ood-quantile", type=float, default=0.95)
    ap.add_argument("--reject-ood", action="store_true", default=True)
    ap.add_argument("--no-reject-ood", dest="reject_ood", action="store_false")
    ap.add_argument("--ridge-alpha", type=float, default=10.0)
    ap.add_argument("--min-non-na-frac", type=float, default=0.05)
    args = ap.parse_args()

    raw = pd.read_csv(args.subject_rows)
    candidates = parse_space_list(args.candidates)
    alpha_modes = parse_space_list(args.alpha_modes)
    needed = sorted(set(candidates + alpha_modes + ["none", args.profile_mode]))
    df = candidate_rows(raw, needed)
    row_map = subject_mode_map(df)
    subjects = sorted(df["subject"].astype(str).unique().tolist())

    feature_cols, excluded_leaky, excluded_sparse = select_no_label_feature_cols(
        df,
        min_non_na_frac=float(args.min_non_na_frac),
    )
    if not feature_cols:
        raise SystemExit("No strict no-label features survived filtering.")

    mode_summary = summarize_modes(df[df["mode"].isin(candidates)])
    oracle = oracle_by_subject(df[df["mode"].isin(candidates)])

    all_rows: List[Dict[str, object]] = []
    for alpha_mode in alpha_modes:
        all_rows.extend(build_policy_rows_for_alpha(
            df=df,
            row_map=row_map,
            subjects=subjects,
            alpha_mode=alpha_mode,
            feature_cols=feature_cols,
            args=args,
        ))
    policy_rows = pd.DataFrame(all_rows)
    if policy_rows.empty:
        raise SystemExit("No prototype-gate policy rows could be built; check alpha modes and subject rows.")
    policy_rows = add_best_alpha_policy(policy_rows, row_map, subjects)
    if bool(args.include_profile_fallback) and args.profile_mode:
        policy_rows = add_profile_fallback_policy(policy_rows, row_map, subjects, args.profile_mode)

    policy_summary = summarize_policies(policy_rows, mode_summary, oracle)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mode_summary.to_csv(args.out_dir / "adaptation_mode_summary.csv", index=False)
    oracle.to_csv(args.out_dir / "adaptation_oracle_by_subject.csv", index=False)
    policy_rows.to_csv(args.out_dir / "adaptation_prototype_gate_by_subject.csv", index=False)
    policy_summary.to_csv(args.out_dir / "adaptation_prototype_gate_summary.csv", index=False)
    with open(args.out_dir / "adaptation_prototype_gate_features.json", "w") as f:
        json.dump({
            "features": sorted(feature_cols),
            "excluded_leaky_features": sorted(excluded_leaky),
            "excluded_non_numeric_or_sparse_features": sorted(excluded_sparse),
            "leakage_filter_version": LEAKAGE_FILTER_VERSION,
            "strict_no_label": True,
            "candidates": sorted(candidates),
            "alpha_modes": sorted(alpha_modes),
            "profile_mode": str(args.profile_mode),
            "include_profile_fallback": bool(args.include_profile_fallback),
            "min_gain": float(args.min_gain),
            "safe_threshold": float(args.safe_threshold),
            "knn_k": int(args.knn_k),
            "ood_quantile": float(args.ood_quantile),
            "reject_ood": bool(args.reject_ood),
            "ridge_alpha": float(args.ridge_alpha),
        }, f, indent=2)

    with pd.option_context("display.max_columns", 120, "display.width", 240):
        print("\n=== Fixed mode summary ===")
        print(mode_summary)
        print("\n=== Prototype gate summary ===")
        print(policy_summary)
        print("\n=== Prototype gate selected modes ===")
        cols = ["subject", "policy", "selected_mode", "selected_delta_vs_none", "gate_pass", "proto_nearest_subject", "proto_soft_safe_prob", "proto_soft_gain", "pred_gain_ridge"]
        keep_cols = [c for c in cols if c in policy_rows.columns]
        print(policy_rows[keep_cols].sort_values(["policy", "subject"]).head(200))
        print(f"\n[INFO] Used {len(feature_cols)} strict no-label features; excluded {len(excluded_leaky)} leaky diagnostics.")
        print(f"[INFO] Wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
