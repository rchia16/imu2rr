#!/usr/bin/env python3
"""Strict no-label policy and learned continuous-alpha analysis for RR adaptation sweeps.

This script reads the subject_rows.csv produced by the adaptation sweep and runs two
analysis layers:

1) Strict no-label discrete policy, retained as a diagnostic baseline.
2) Learned continuous alpha_hat in [0, 1], trained by LOSO pseudo-targets from
   source-subject fixed-alpha curves.

Labels are allowed only as pseudo-target/meta-targets during source-subject
training and for final evaluation. They are never allowed as meta-input features
when --strict-no-label is active.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_CANDIDATES = [
    "none",
    "adapt_mean_alpha_025",
    "adapt_mean_alpha_050",
    "adapt_mean_alpha_075",
    "adapt_mean_alpha_100",
    "adapt_mean_profile_shrink",
    "profile_film_init_only",
    "profile_film_unsup_sparc",
    "direct_stft_rr",
    "hybrid_probe_stft_conf",
]

DEFAULT_ALPHA_GRID = {
    "none": 0.0,
    "adapt_mean_alpha_025": 0.25,
    "adapt_mean_alpha_050": 0.50,
    "adapt_mean_alpha_075": 0.75,
    "adapt_mean_alpha_100": 1.00,
}

ALPHA_SUMMARY_MODES = [
    "none",
    "feature_mean_align_alpha050",
    "feature_mean_align_alpha075",
    "feature_mean_align_alpha100",
    "feature_mean_align_profile_shrink",
    "adapt_mean_alpha_050",
    "adapt_mean_alpha_075",
    "adapt_mean_alpha_100",
    "learned_alpha_hat",
    "learned_alpha_hat_film_residual",
]

# Features that can be computed from unlabeled target windows/source priors.
ALLOWED_PREFIXES = (
    "feature_adapt_",
    "feature_align_",  # legacy alias; label-derived fields are filtered below
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

LEAKAGE_FILTER_VERSION = "strict_v3_alpha_hat_no_labels_no_residuals_no_overshoot"

# Conservative leakage filter. These fields can be excellent diagnostics but are
# unavailable for a truly unlabeled target subject.
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
    "post_mae",
    "pre_mae",
    "base_pre",
    "candidate_eval",
    "cal_mae",
    "cal_rmse",
    "cal_corr",
    "eval_aux_rr_mae",
    "eval_stft_rr_mae",
)

LEAKY_NAMESPACES = (
    "eval_direction",
    "cal_direction",
    "profile_oracle",
)


def _parse_candidates(text: str) -> List[str]:
    toks = [t.strip() for t in str(text).replace(",", " ").split() if t.strip()]
    return toks or list(DEFAULT_CANDIDATES)


def _parse_alpha_grid(text: str) -> Dict[str, float]:
    if not str(text).strip():
        return dict(DEFAULT_ALPHA_GRID)
    out: Dict[str, float] = {}
    for tok in str(text).replace(",", " ").split():
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            raise SystemExit(f"Bad --alpha-grid-modes token {tok!r}; expected mode:alpha")
        mode, val = tok.split(":", 1)
        out[mode.strip()] = float(val)
    if "none" not in out:
        out["none"] = 0.0
    return out


def is_label_leaky_feature(name: str) -> bool:
    low = str(name).lower()
    if any(bad in low for bad in LEAKY_SUBSTRINGS):
        return True
    return any(low.startswith(ns) or ns in low for ns in LEAKY_NAMESPACES)


def _candidate_rows(df: pd.DataFrame, candidates: Sequence[str]) -> pd.DataFrame:
    if "mode" not in df.columns or "subject" not in df.columns:
        raise SystemExit("subject_rows.csv must contain columns: mode, subject")
    out = df[df["mode"].astype(str).isin(candidates)].copy()
    if out.empty:
        raise SystemExit("No subject rows found for requested candidates.")
    for c in ["rr_probe_pre_mae", "rr_probe_post_mae"]:
        if c not in out.columns:
            raise SystemExit(f"subject_rows.csv missing required column {c!r}")
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["rr_probe_pre_mae", "rr_probe_post_mae"]).copy()
    out["delta_mae"] = out["rr_probe_post_mae"] - out["rr_probe_pre_mae"]
    out["subject"] = out["subject"].astype(str)
    out["mode"] = out["mode"].astype(str)
    return out


def _candidate_no_label_feature_cols(df: pd.DataFrame) -> List[str]:
    return sorted({
        str(col)
        for col in df.columns
        if any(str(col).startswith(prefix) for prefix in ALLOWED_PREFIXES)
    })


def _select_no_label_feature_cols(
    df: pd.DataFrame,
    *,
    strict_no_label: bool,
    allow_diagnostic_leakage: bool,
) -> Tuple[List[str], List[str], List[str]]:
    cols: List[str] = []
    excluded_leaky: List[str] = []
    excluded_non_numeric_or_sparse: List[str] = []
    enforce = bool(strict_no_label and not allow_diagnostic_leakage)
    for name in _candidate_no_label_feature_cols(df):
        if enforce and is_label_leaky_feature(name):
            excluded_leaky.append(name)
            continue
        values = pd.to_numeric(df[name], errors="coerce")
        if values.notna().mean() < 0.05:
            excluded_non_numeric_or_sparse.append(name)
            continue
        if values.nunique(dropna=True) <= 1:
            excluded_non_numeric_or_sparse.append(name)
            continue
        cols.append(name)
    return sorted(set(cols)), sorted(set(excluded_leaky)), sorted(set(excluded_non_numeric_or_sparse))


def _assert_strict_no_label_features(feature_cols: Sequence[str]) -> None:
    offenders = sorted({c for c in feature_cols if is_label_leaky_feature(c)})
    if offenders:
        raise RuntimeError(
            "Strict no-label feature selection leaked label-derived diagnostics: "
            + ", ".join(offenders)
        )


def _design_matrix(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    mode_levels: Optional[Sequence[str]] = None,
    medians: Optional[pd.Series] = None,
) -> Tuple[np.ndarray, pd.Series]:
    x_num = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in feature_cols})
    if medians is None:
        medians = x_num.median(axis=0, skipna=True).fillna(0.0)
    x_num = x_num.fillna(medians).to_numpy(dtype=np.float64)
    if mode_levels is None:
        return x_num, medians
    mode = df["mode"].astype(str).to_numpy()
    mode_map = {m: i for i, m in enumerate(mode_levels)}
    x_mode = np.zeros((len(df), len(mode_levels)), dtype=np.float64)
    for i, m in enumerate(mode):
        if m in mode_map:
            x_mode[i, mode_map[m]] = 1.0
    return np.concatenate([x_num, x_mode], axis=1), medians


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("Cannot fit ridge model on empty design matrix")
    mu = x.mean(axis=0)
    sd = x.std(axis=0) + 1e-6
    xs = (x - mu.reshape(1, -1)) / sd.reshape(1, -1)
    design = np.column_stack([np.ones(xs.shape[0]), xs])
    reg = np.eye(design.shape[1]) * float(alpha)
    reg[0, 0] = 0.0
    coef = np.linalg.pinv(design.T @ design + reg) @ design.T @ y
    return coef, mu, sd


def _ridge_predict(model: Tuple[np.ndarray, np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    coef, mu, sd = model
    xs = (x - mu.reshape(1, -1)) / sd.reshape(1, -1)
    design = np.column_stack([np.ones(xs.shape[0]), xs])
    return design @ coef


def _summarize_modes(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("mode", as_index=False).agg(
        n_subjects=("subject", "nunique"),
        post_mae_mean=("rr_probe_post_mae", "mean"),
        delta_mae_mean=("delta_mae", "mean"),
        subjects_improved=("delta_mae", lambda x: int((x < 0).sum())),
        subjects_worse=("delta_mae", lambda x: int((x > 0).sum())),
    ).sort_values(["post_mae_mean", "delta_mae_mean"])


def _oracle_by_subject(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject, sub in df.groupby("subject"):
        best = sub.sort_values("rr_probe_post_mae").iloc[0]
        none = sub[sub["mode"] == "none"]
        none_mae = float(none["rr_probe_post_mae"].iloc[0]) if not none.empty else float(sub["rr_probe_pre_mae"].iloc[0])
        rows.append({
            "subject": subject,
            "oracle_mode": best["mode"],
            "oracle_post_mae": float(best["rr_probe_post_mae"]),
            "none_post_mae": none_mae,
            "oracle_delta_vs_none": float(best["rr_probe_post_mae"] - none_mae),
        })
    return pd.DataFrame(rows)


def _loso_policy(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    ridge_alpha: float,
    safety_margin: float,
) -> pd.DataFrame:
    mode_levels = sorted(df["mode"].astype(str).unique().tolist())
    rows = []
    for heldout in sorted(df["subject"].astype(str).unique()):
        train = df[df["subject"].astype(str) != heldout].copy()
        test = df[df["subject"].astype(str) == heldout].copy()
        if train.empty or test.empty:
            continue
        x_train, med = _design_matrix(train, feature_cols, mode_levels)
        y_train = train["delta_mae"].to_numpy(dtype=np.float64)
        model = _ridge_fit(x_train, y_train, ridge_alpha)
        x_test, _ = _design_matrix(test, feature_cols, mode_levels, medians=med)
        test = test.copy()
        test["pred_delta_mae"] = _ridge_predict(model, x_test)

        best_pred = test.sort_values("pred_delta_mae").iloc[0]
        none_rows = test[test["mode"] == "none"]
        none = none_rows.iloc[0] if not none_rows.empty else test.iloc[0]
        if float(best_pred["pred_delta_mae"]) > -float(safety_margin) and not none_rows.empty:
            best_pred = none
        oracle = test.sort_values("rr_probe_post_mae").iloc[0]
        rows.append({
            "subject": heldout,
            "selected_mode": best_pred["mode"],
            "selected_pred_delta_mae": float(best_pred.get("pred_delta_mae", 0.0)),
            "selected_actual_delta_mae": float(best_pred["delta_mae"]),
            "selected_post_mae": float(best_pred["rr_probe_post_mae"]),
            "none_post_mae": float(none["rr_probe_post_mae"]),
            "selected_delta_vs_none": float(best_pred["rr_probe_post_mae"] - none["rr_probe_post_mae"]),
            "oracle_mode": oracle["mode"],
            "oracle_post_mae": float(oracle["rr_probe_post_mae"]),
            "oracle_delta_vs_none": float(oracle["rr_probe_post_mae"] - none["rr_probe_post_mae"]),
            "oracle_regret_mae": float(best_pred["rr_probe_post_mae"] - oracle["rr_probe_post_mae"]),
        })
    return pd.DataFrame(rows)


def _best_alpha_from_curve(curve: pd.DataFrame, alpha_grid: Mapping[str, float], method: str) -> Tuple[float, str, float]:
    rows = []
    for mode, alpha in alpha_grid.items():
        sub = curve[curve["mode"] == mode]
        if sub.empty:
            continue
        rows.append((float(alpha), str(mode), float(sub["rr_probe_post_mae"].iloc[0])))
    if not rows:
        return 0.0, "missing", float("nan")
    rows = sorted(rows, key=lambda t: t[0])
    best_alpha, best_mode, best_mae = min(rows, key=lambda t: t[2])
    if method == "grid_best" or len(rows) < 3:
        return float(best_alpha), best_mode, float(best_mae)
    x = np.asarray([r[0] for r in rows], dtype=np.float64)
    y = np.asarray([r[2] for r in rows], dtype=np.float64)
    try:
        a, b, c = np.polyfit(x, y, deg=2)
    except Exception:
        return float(best_alpha), best_mode, float(best_mae)
    if not np.isfinite(a) or a <= 1e-8:
        return float(best_alpha), best_mode, float(best_mae)
    alpha_star = float(np.clip(-b / (2.0 * a), 0.0, 1.0))
    pred_star = float(a * alpha_star * alpha_star + b * alpha_star + c)
    # Use the continuous optimum only when it is plausible and not a wild extrapolation.
    if np.isfinite(pred_star) and pred_star <= best_mae + 0.05:
        return alpha_star, "quadratic_safe", pred_star
    return float(best_alpha), best_mode, float(best_mae)


def _feature_rows_for_alpha(df: pd.DataFrame, feature_mode: str) -> pd.DataFrame:
    rows = df[df["mode"] == feature_mode].copy()
    if rows.empty:
        # Fall back to the strongest alpha mode available, then any non-none row.
        for mode in ["adapt_mean_alpha_100", "adapt_mean_alpha_075", "adapt_mean_alpha_050", "adapt_mean_alpha_025"]:
            rows = df[df["mode"] == mode].copy()
            if not rows.empty:
                break
    if rows.empty:
        rows = df[df["mode"] != "none"].copy()
    if rows.empty:
        raise SystemExit("Could not find feature rows for learned-alpha diagnostics.")
    return rows.drop_duplicates(subset=["subject"], keep="last")


def _alpha_training_table(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    alpha_grid: Mapping[str, float],
    feature_mode: str,
    target_method: str,
) -> pd.DataFrame:
    feature_rows = _feature_rows_for_alpha(df, feature_mode)
    targets = []
    for subject, sub in df.groupby("subject"):
        alpha_target, target_mode, target_mae = _best_alpha_from_curve(sub, alpha_grid, target_method)
        targets.append({
            "subject": str(subject),
            "alpha_target": float(alpha_target),
            "alpha_target_mode": target_mode,
            "alpha_target_post_mae_est": float(target_mae),
        })
    target_df = pd.DataFrame(targets)
    merged = feature_rows.merge(target_df, on="subject", how="inner")
    if merged.empty:
        raise SystemExit("No rows available for learned-alpha training.")
    return merged


def _find_prediction_file(sweep_root: Optional[Path], mode: str, subject: str) -> Optional[Path]:
    if sweep_root is None:
        return None
    root = Path(sweep_root)
    if not root.exists():
        return None
    candidates: List[Path] = []
    patterns = [
        f"{mode}/subjects/{subject}/**/*predictions_{subject}.csv",
        f"{mode}/**/*predictions_{subject}.csv",
        f"**/{mode}/subjects/{subject}/**/*predictions_{subject}.csv",
        f"**/{subject}/**/*predictions_{subject}.csv",
    ]
    for pat in patterns:
        candidates.extend(root.glob(pat))
    candidates = [p for p in candidates if p.is_file() and mode in str(p)]
    if not candidates:
        return None
    # Prefer exact mode path and smaller path depth.
    candidates = sorted(set(candidates), key=lambda p: (len(p.parts), str(p)))
    return candidates[0]


def _read_prediction(sweep_root: Optional[Path], mode: str, subject: str) -> Optional[pd.DataFrame]:
    path = _find_prediction_file(sweep_root, mode, subject)
    if path is None:
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "rr_true" not in df.columns:
        return None
    if "rr_pred_post" not in df.columns:
        return None
    out = df.copy()
    out["__prediction_path"] = str(path)
    return out


def _post_mae_for_mode(df: pd.DataFrame, subject: str, mode: str) -> float:
    row = df[(df["subject"] == subject) & (df["mode"] == mode)]
    if row.empty:
        return float("nan")
    return float(row["rr_probe_post_mae"].iloc[0])


def _none_mae_for_subject(df: pd.DataFrame, subject: str) -> float:
    val = _post_mae_for_mode(df, subject, "none")
    if np.isfinite(val):
        return float(val)
    sub = df[df["subject"] == subject]
    return float(sub["rr_probe_pre_mae"].iloc[0]) if not sub.empty else float("nan")


def _interp_prediction_from_grid(
    preds_by_alpha: Mapping[float, pd.DataFrame],
    alpha_hat: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, str]:
    available = sorted((float(a), p) for a, p in preds_by_alpha.items() if p is not None and "rr_pred_post" in p.columns)
    if len(available) < 2:
        return None, None, 0, "insufficient_prediction_grid"
    lengths = [len(p) for _, p in available]
    n = min(lengths)
    if n <= 0:
        return None, None, 0, "empty_prediction_grid"
    alphas = np.asarray([a for a, _ in available], dtype=np.float64)
    mat = np.stack([pd.to_numeric(p["rr_pred_post"].iloc[:n], errors="coerce").to_numpy(dtype=np.float64) for _, p in available], axis=1)
    rr_true = pd.to_numeric(available[0][1]["rr_true"].iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
    pred = np.asarray([np.interp(float(alpha_hat), alphas, mat[i]) for i in range(n)], dtype=np.float64)
    return rr_true, pred, 1, "prediction_grid_interpolation"


def _mae(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    n = min(y.size, pred.size)
    if n <= 0:
        return float("nan")
    keep = np.isfinite(y[:n]) & np.isfinite(pred[:n])
    if not np.any(keep):
        return float("nan")
    return float(np.mean(np.abs(y[:n][keep] - pred[:n][keep])))


def _interp_mae_curve(df: pd.DataFrame, subject: str, alpha_grid: Mapping[str, float], alpha_hat: float) -> float:
    pts = []
    for mode, alpha in alpha_grid.items():
        mae = _post_mae_for_mode(df, subject, mode)
        if np.isfinite(mae):
            pts.append((float(alpha), float(mae)))
    if len(pts) < 2:
        return float("nan")
    pts = sorted(pts)
    alphas = np.asarray([p[0] for p in pts], dtype=np.float64)
    maes = np.asarray([p[1] for p in pts], dtype=np.float64)
    return float(np.interp(float(alpha_hat), alphas, maes))


def _evaluate_learned_alpha(
    df: pd.DataFrame,
    alpha_table: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    ridge_alpha: float,
    alpha_grid: Mapping[str, float],
    sweep_root: Optional[Path],
    film_source_mode: str,
    film_residual_lambda: float,
) -> pd.DataFrame:
    rows = []
    subjects = sorted(alpha_table["subject"].astype(str).unique().tolist())
    for heldout in subjects:
        train = alpha_table[alpha_table["subject"].astype(str) != heldout].copy()
        test = alpha_table[alpha_table["subject"].astype(str) == heldout].copy()
        if train.empty or test.empty:
            continue
        x_train, med = _design_matrix(train, feature_cols, mode_levels=None)
        y_train = train["alpha_target"].to_numpy(dtype=np.float64)
        model = _ridge_fit(x_train, y_train, ridge_alpha)
        x_test, _ = _design_matrix(test, feature_cols, mode_levels=None, medians=med)
        alpha_raw = float(_ridge_predict(model, x_test)[0])
        alpha_hat = float(np.clip(alpha_raw, 0.0, 1.0))
        subject = str(heldout)
        none_mae = _none_mae_for_subject(df, subject)

        pred_map: Dict[float, pd.DataFrame] = {}
        for mode, alpha in alpha_grid.items():
            p = _read_prediction(sweep_root, mode, subject)
            if p is not None:
                pred_map[float(alpha)] = p
        rr_true, rr_alpha, used_predictions, eval_source = _interp_prediction_from_grid(pred_map, alpha_hat)
        if rr_true is not None and rr_alpha is not None:
            alpha_mae = _mae(rr_true, rr_alpha)
        else:
            alpha_mae = _interp_mae_curve(df, subject, alpha_grid, alpha_hat)
            used_predictions = 0
            eval_source = "mae_curve_interpolation"

        film_mae = float("nan")
        film_used = 0
        film_path = ""
        if rr_true is not None and rr_alpha is not None and film_source_mode:
            none_pred = _read_prediction(sweep_root, "none", subject)
            film_pred = _read_prediction(sweep_root, film_source_mode, subject)
            if none_pred is not None and film_pred is not None:
                n = min(len(rr_alpha), len(none_pred), len(film_pred), len(rr_true))
                if n > 0:
                    none_col = "rr_pred_post" if "rr_pred_post" in none_pred.columns else "rr_pred_pre"
                    film_col = "rr_pred_post"
                    base = pd.to_numeric(none_pred[none_col].iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
                    film = pd.to_numeric(film_pred[film_col].iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
                    pred_film = rr_alpha[:n] + float(film_residual_lambda) * (film - base)
                    film_mae = _mae(rr_true[:n], pred_film)
                    film_used = 1
                    film_path = str(film_pred.get("__prediction_path", pd.Series([""])).iloc[0]) if "__prediction_path" in film_pred else ""

        rows.append({
            "subject": subject,
            "alpha_target": float(test["alpha_target"].iloc[0]),
            "alpha_target_mode": str(test["alpha_target_mode"].iloc[0]),
            "alpha_target_post_mae_est": float(test["alpha_target_post_mae_est"].iloc[0]),
            "alpha_hat_raw": alpha_raw,
            "alpha_hat": alpha_hat,
            "none_post_mae": none_mae,
            "learned_alpha_post_mae": float(alpha_mae),
            "learned_alpha_delta_vs_none": float(alpha_mae - none_mae) if np.isfinite(alpha_mae) and np.isfinite(none_mae) else float("nan"),
            "learned_alpha_film_residual_post_mae": float(film_mae),
            "learned_alpha_film_residual_delta_vs_none": float(film_mae - none_mae) if np.isfinite(film_mae) and np.isfinite(none_mae) else float("nan"),
            "film_residual_lambda": float(film_residual_lambda),
            "film_source_mode": str(film_source_mode),
            "used_prediction_files": int(used_predictions),
            "used_film_prediction_files": int(film_used),
            "learned_alpha_eval_source": eval_source,
            "film_prediction_path": film_path,
        })
    return pd.DataFrame(rows)


def _learned_alpha_summary(
    df: pd.DataFrame,
    alpha_by_subject: pd.DataFrame,
    modes: Sequence[str],
) -> pd.DataFrame:
    rows = []
    subjects = sorted(df["subject"].astype(str).unique().tolist())
    for mode in modes:
        if mode == "learned_alpha_hat":
            vals = alpha_by_subject[["subject", "learned_alpha_post_mae", "learned_alpha_delta_vs_none"]].copy()
            vals = vals.rename(columns={"learned_alpha_post_mae": "post_mae", "learned_alpha_delta_vs_none": "delta_vs_none"})
        elif mode == "learned_alpha_hat_film_residual":
            vals = alpha_by_subject[["subject", "learned_alpha_film_residual_post_mae", "learned_alpha_film_residual_delta_vs_none"]].copy()
            vals = vals.rename(columns={"learned_alpha_film_residual_post_mae": "post_mae", "learned_alpha_film_residual_delta_vs_none": "delta_vs_none"})
        else:
            sub_rows = []
            for subject in subjects:
                post = _post_mae_for_mode(df, subject, mode)
                none = _none_mae_for_subject(df, subject)
                if np.isfinite(post):
                    sub_rows.append({"subject": subject, "post_mae": post, "delta_vs_none": post - none})
            vals = pd.DataFrame(sub_rows)
        if vals.empty:
            rows.append({
                "mode": mode,
                "n_subjects": 0,
                "post_mae_mean": float("nan"),
                "delta_vs_none_mean": float("nan"),
                "subjects_improved_vs_none": 0,
                "subjects_worse_vs_none": 0,
            })
            continue
        rows.append({
            "mode": mode,
            "n_subjects": int(vals["post_mae"].notna().sum()),
            "post_mae_mean": float(pd.to_numeric(vals["post_mae"], errors="coerce").mean()),
            "delta_vs_none_mean": float(pd.to_numeric(vals["delta_vs_none"], errors="coerce").mean()),
            "subjects_improved_vs_none": int((pd.to_numeric(vals["delta_vs_none"], errors="coerce") < 0).sum()),
            "subjects_worse_vs_none": int((pd.to_numeric(vals["delta_vs_none"], errors="coerce") > 0).sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-rows", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--sweep-root", default=None, type=Path, help="Optional sweep root used to find per-window prediction CSVs.")
    ap.add_argument("--candidates", default=" ".join(DEFAULT_CANDIDATES))
    ap.add_argument("--ridge-alpha", type=float, default=10.0)
    ap.add_argument("--safety-margin", type=float, default=0.02)
    ap.add_argument("--strict-no-label", dest="strict_no_label", action="store_true", default=True)
    ap.add_argument("--no-strict-no-label", dest="strict_no_label", action="store_false")
    ap.add_argument("--allow-diagnostic-leakage", action="store_true", default=False)
    ap.add_argument("--learn-alpha-policy", dest="learn_alpha_policy", action="store_true", default=True)
    ap.add_argument("--no-learn-alpha-policy", dest="learn_alpha_policy", action="store_false")
    ap.add_argument("--alpha-grid-modes", default="none:0 adapt_mean_alpha_025:0.25 adapt_mean_alpha_050:0.5 adapt_mean_alpha_075:0.75 adapt_mean_alpha_100:1.0")
    ap.add_argument("--alpha-target-method", default="quadratic_safe", choices=["grid_best", "quadratic_safe"])
    ap.add_argument("--learn-alpha-feature-mode", default="adapt_mean_alpha_100")
    ap.add_argument("--learn-alpha-film-source-mode", default="profile_film_init_only")
    ap.add_argument("--learn-alpha-film-residual-lambda", type=float, default=0.25)
    args = ap.parse_args()

    df_all = pd.read_csv(args.subject_rows)
    candidates = _parse_candidates(args.candidates)
    alpha_grid = _parse_alpha_grid(args.alpha_grid_modes)
    # Ensure fixed-alpha modes are retained for alpha training/evaluation even if
    # the candidate string was narrowed accidentally.
    for mode in alpha_grid:
        if mode not in candidates:
            candidates.append(mode)
    if args.learn_alpha_film_source_mode and args.learn_alpha_film_source_mode not in candidates:
        candidates.append(args.learn_alpha_film_source_mode)

    df = _candidate_rows(df_all, candidates)
    feature_cols, excluded_leaky, excluded_sparse = _select_no_label_feature_cols(
        df,
        strict_no_label=bool(args.strict_no_label),
        allow_diagnostic_leakage=bool(args.allow_diagnostic_leakage),
    )
    if args.strict_no_label and not args.allow_diagnostic_leakage:
        _assert_strict_no_label_features(feature_cols)
    if args.allow_diagnostic_leakage:
        print("[WARNING] --allow-diagnostic-leakage is analysis-only and not deployable.")

    mode_summary = _summarize_modes(df)
    oracle = _oracle_by_subject(df)
    selected = _loso_policy(df, feature_cols, ridge_alpha=args.ridge_alpha, safety_margin=args.safety_margin)
    selected["strict_no_label"] = int(bool(args.strict_no_label))
    selected["allow_diagnostic_leakage"] = int(bool(args.allow_diagnostic_leakage))

    alpha_by_subject = pd.DataFrame()
    alpha_summary = pd.DataFrame()
    if args.learn_alpha_policy:
        alpha_table = _alpha_training_table(
            df,
            feature_cols,
            alpha_grid=alpha_grid,
            feature_mode=str(args.learn_alpha_feature_mode),
            target_method=str(args.alpha_target_method),
        )
        alpha_by_subject = _evaluate_learned_alpha(
            df,
            alpha_table,
            feature_cols,
            ridge_alpha=float(args.ridge_alpha),
            alpha_grid=alpha_grid,
            sweep_root=args.sweep_root,
            film_source_mode=str(args.learn_alpha_film_source_mode),
            film_residual_lambda=float(args.learn_alpha_film_residual_lambda),
        )
        alpha_summary = _learned_alpha_summary(df, alpha_by_subject, ALPHA_SUMMARY_MODES)

    meta_summary = pd.DataFrame([{
        "n_subjects": int(selected.shape[0]),
        "n_no_label_features": int(len(feature_cols)),
        "n_excluded_leaky_features": int(len(excluded_leaky)),
        "n_excluded_non_numeric_or_sparse_features": int(len(excluded_sparse)),
        "strict_no_label": int(bool(args.strict_no_label)),
        "allow_diagnostic_leakage": int(bool(args.allow_diagnostic_leakage)),
        "leakage_filter_version": LEAKAGE_FILTER_VERSION,
        "ridge_alpha": float(args.ridge_alpha),
        "safety_margin": float(args.safety_margin),
        "policy_post_mae_mean": float(selected["selected_post_mae"].mean()) if not selected.empty else np.nan,
        "policy_delta_vs_none_mean": float(selected["selected_delta_vs_none"].mean()) if not selected.empty else np.nan,
        "policy_subjects_improved_vs_none": int((selected["selected_delta_vs_none"] < 0).sum()) if not selected.empty else 0,
        "policy_subjects_worse_vs_none": int((selected["selected_delta_vs_none"] > 0).sum()) if not selected.empty else 0,
        "oracle_post_mae_mean": float(oracle["oracle_post_mae"].mean()) if not oracle.empty else np.nan,
        "oracle_delta_vs_none_mean": float(oracle["oracle_delta_vs_none"].mean()) if not oracle.empty else np.nan,
        "policy_oracle_regret_mean": float(selected["oracle_regret_mae"].mean()) if not selected.empty else np.nan,
        "learn_alpha_policy": int(bool(args.learn_alpha_policy)),
        "learn_alpha_feature_mode": str(args.learn_alpha_feature_mode),
        "alpha_target_method": str(args.alpha_target_method),
        "learn_alpha_film_source_mode": str(args.learn_alpha_film_source_mode),
        "learn_alpha_film_residual_lambda": float(args.learn_alpha_film_residual_lambda),
        "learned_alpha_post_mae_mean": float(alpha_by_subject["learned_alpha_post_mae"].mean()) if not alpha_by_subject.empty else np.nan,
        "learned_alpha_delta_vs_none_mean": float(alpha_by_subject["learned_alpha_delta_vs_none"].mean()) if not alpha_by_subject.empty else np.nan,
        "learned_alpha_film_residual_post_mae_mean": float(alpha_by_subject["learned_alpha_film_residual_post_mae"].mean()) if not alpha_by_subject.empty else np.nan,
        "learned_alpha_film_residual_delta_vs_none_mean": float(alpha_by_subject["learned_alpha_film_residual_delta_vs_none"].mean()) if not alpha_by_subject.empty else np.nan,
        "learned_alpha_used_prediction_files_frac": float(alpha_by_subject["used_prediction_files"].mean()) if not alpha_by_subject.empty and "used_prediction_files" in alpha_by_subject else np.nan,
    }])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mode_summary.to_csv(args.out_dir / "adaptation_mode_summary.csv", index=False)
    oracle.to_csv(args.out_dir / "adaptation_oracle_by_subject.csv", index=False)
    selected.to_csv(args.out_dir / "adaptation_loso_policy_by_subject.csv", index=False)
    meta_summary.to_csv(args.out_dir / "adaptation_loso_policy_summary.csv", index=False)
    if args.learn_alpha_policy:
        alpha_by_subject.to_csv(args.out_dir / "adaptation_learned_alpha_by_subject.csv", index=False)
        alpha_summary.to_csv(args.out_dir / "adaptation_learned_alpha_summary.csv", index=False)
    with open(args.out_dir / "adaptation_no_label_features.json", "w") as f:
        json.dump({
            "features": sorted(feature_cols),
            "excluded_leaky_features": sorted(excluded_leaky),
            "excluded_non_numeric_or_sparse_features": sorted(excluded_sparse),
            "leakage_filter_version": LEAKAGE_FILTER_VERSION,
            "strict_no_label": bool(args.strict_no_label),
            "allow_diagnostic_leakage": bool(args.allow_diagnostic_leakage),
            "learn_alpha_policy": bool(args.learn_alpha_policy),
            "alpha_grid_modes": dict(sorted(alpha_grid.items())),
            "learn_alpha_feature_mode": str(args.learn_alpha_feature_mode),
            "learn_alpha_film_source_mode": str(args.learn_alpha_film_source_mode),
            "learn_alpha_film_residual_lambda": float(args.learn_alpha_film_residual_lambda),
            "candidates": sorted(candidates),
        }, f, indent=2)

    with pd.option_context("display.max_columns", 120, "display.width", 240):
        print("\n=== Adaptation mode summary ===")
        print(mode_summary)
        print("\n=== Strict no-label LOSO policy summary ===")
        print(meta_summary)
        print("\n=== Selected discrete modes ===")
        cols = ["subject", "selected_mode", "oracle_mode", "selected_delta_vs_none", "oracle_delta_vs_none", "oracle_regret_mae"]
        print(selected[cols] if not selected.empty else selected)
        if args.learn_alpha_policy:
            print("\n=== Learned alpha summary ===")
            print(alpha_summary)
            print("\n=== Learned alpha by subject ===")
            show = ["subject", "alpha_target", "alpha_hat", "learned_alpha_delta_vs_none", "learned_alpha_film_residual_delta_vs_none", "used_prediction_files", "learned_alpha_eval_source"]
            print(alpha_by_subject[show] if not alpha_by_subject.empty else alpha_by_subject)
        print(
            f"\n[INFO] Used {len(feature_cols)} strict no-label features; "
            f"excluded {len(excluded_leaky)} leaky diagnostics. "
            "See adaptation_no_label_features.json"
        )


if __name__ == "__main__":
    main()
