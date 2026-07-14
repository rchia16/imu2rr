from __future__ import annotations

import hashlib
import json
import os
import random
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from rr_spectral_readout_experiment import (
    BR_FS,
    DEFAULT_SUBJECTS,
    SoftSpectralRR,
    band_mean_spectrum,
    hard_spectral_rr,
    rr_metric_dict,
    spectral_entropy_from_stft,
)


CONFIDENCE_FEATURES = [
    "entropy",
    "max_probability",
    "top1_top2_gap",
    "peak_width_bpm",
    "spectral_variance",
    "hard_soft_gap_bpm",
    "spectral_hidden_disagreement_bpm",
    "respiratory_band_energy",
]

PHASE_A_METHODS = [
    "A0_soft_spectral",
    "A1_gaussian_kl",
    "A2_wasserstein",
    "A3_kl_rr_mae",
    "A4_hidden_only",
    "A5_spec_linear_residual",
    "A6_spec_hidden_conf_residual",
]

PHASE_B_METHODS = [
    "B0_always_spec",
    "B1_always_hybrid",
    "B2_disagreement_gate",
    "B3_soft_gate",
    "B4_oracle_gate",
]

PHASE_C_METHODS = [
    "C0_none",
    "C1_feature_mean",
    "C2_temperature",
    "C3_affine_readout",
    "C4_confidence_gated_affine",
    "C5_oracle_offset",
    "C6_oracle_affine",
]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=json_default)
    os.replace(tmp, path)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def stable_config_hash(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=json_default).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def run_manifest(args: Any) -> Dict[str, Any]:
    return {
        "argv": sys.argv,
        "args": vars(args),
        "git_commit": git_commit(),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(getattr(args, "device", "")),
    }


def trainable_parameter_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def gaussian_target_bpm(rr_true: torch.Tensor, bins_bpm: torch.Tensor, sigma_bpm: float) -> torch.Tensor:
    q = torch.exp(-0.5 * ((bins_bpm.view(1, -1) - rr_true.view(-1, 1)) / max(float(sigma_bpm), 1e-6)).pow(2))
    return q / q.sum(dim=1, keepdim=True).clamp_min(1e-8)


def expected_rr(prob: torch.Tensor, bins_bpm: torch.Tensor) -> torch.Tensor:
    return (prob * bins_bpm.view(1, -1)).sum(dim=1)


def spectral_arrays_from_stft(
    predicted_stft: np.ndarray,
    *,
    soft_temperature: float = 0.1,
    hidden_rr: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    stft = torch.as_tensor(predicted_stft, dtype=torch.float32)
    band, freqs_hz, _mask = band_mean_spectrum(stft, br_fs=BR_FS)
    bins_bpm = (60.0 * freqs_hz).detach().cpu().numpy().astype(np.float32)
    logits = band.detach().cpu().numpy().astype(np.float32)
    soft = SoftSpectralRR(BR_FS, temperature=float(soft_temperature))
    with torch.no_grad():
        prob_t, _ = soft.probabilities(stft)
        soft_rr = soft(stft)
        hard_rr, _conf, _peak = hard_spectral_rr(stft, br_fs=BR_FS)
        entropy = spectral_entropy_from_stft(stft, br_fs=BR_FS)
    prob = prob_t.detach().cpu().numpy().astype(np.float32)
    p = np.clip(prob, 1e-8, 1.0)
    order = np.sort(p, axis=1)
    max_prob = order[:, -1]
    top2 = order[:, -2] if order.shape[1] > 1 else np.zeros_like(max_prob)
    top1_top2_gap = max_prob - top2
    var = np.sum(p * (bins_bpm.reshape(1, -1) - np.sum(p * bins_bpm.reshape(1, -1), axis=1, keepdims=True)) ** 2, axis=1)
    peak_width = np.zeros(p.shape[0], dtype=np.float32)
    for i in range(p.shape[0]):
        half = 0.5 * np.max(p[i])
        idx = np.where(p[i] >= half)[0]
        peak_width[i] = float(bins_bpm[idx[-1]] - bins_bpm[idx[0]]) if idx.size else 0.0
    soft_np = soft_rr.detach().cpu().numpy().astype(np.float32)
    hard_np = hard_rr.detach().cpu().numpy().astype(np.float32)
    hidden_np = np.asarray(hidden_rr, dtype=np.float32).reshape(-1) if hidden_rr is not None else soft_np
    return {
        "spectral_logits": logits,
        "spectral_probability": prob,
        "frequency_bins_bpm": bins_bpm,
        "soft_spectral_rr": soft_np,
        "hard_spectral_rr": hard_np,
        "entropy": entropy.detach().cpu().numpy().astype(np.float32),
        "max_probability": max_prob.astype(np.float32),
        "top1_top2_gap": top1_top2_gap.astype(np.float32),
        "peak_width_bpm": peak_width.astype(np.float32),
        "spectral_variance": var.astype(np.float32),
        "hard_soft_gap_bpm": np.abs(hard_np - soft_np).astype(np.float32),
        "spectral_hidden_disagreement_bpm": np.abs(soft_np - hidden_np).astype(np.float32),
        "respiratory_band_energy": np.sum(np.clip(logits, 0.0, None), axis=1).astype(np.float32),
    }


def confidence_matrix(cache: Dict[str, np.ndarray]) -> np.ndarray:
    cols = []
    for name in CONFIDENCE_FEATURES:
        if name in cache:
            cols.append(np.asarray(cache[name], dtype=np.float32).reshape(-1, 1))
        else:
            cols.append(np.zeros((len(cache["soft_spectral_rr"]), 1), dtype=np.float32))
    return np.concatenate(cols, axis=1).astype(np.float32)


def standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32)


def subject_aggregated_mae(y_true: np.ndarray, y_pred: np.ndarray, subjects: Sequence[Any]) -> float:
    df = pd.DataFrame({"subject": list(map(str, subjects)), "err": np.abs(np.asarray(y_pred) - np.asarray(y_true))})
    if df.empty:
        return float("nan")
    return float(df.groupby("subject")["err"].mean().mean())


def per_subject_metrics(df: pd.DataFrame, method_col: str = "method") -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame()
    for (subject, seed, method), g in df.groupby(["subject", "seed", method_col], dropna=False):
        rows.append({"subject": subject, "seed": seed, "method": method, **rr_metric_dict(g["rr_true"].to_numpy(), g["rr_pred"].to_numpy())})
    return pd.DataFrame(rows)


def method_summary(per_subject: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if per_subject.empty:
        return pd.DataFrame()
    for method, g in per_subject.groupby("method"):
        vals = g["MAE"].to_numpy(dtype=float)
        q25, q75 = np.percentile(vals, [25, 75]) if vals.size else (np.nan, np.nan)
        rows.append(
            {
                "method": method,
                "mean_subject_seed_MAE": float(np.mean(vals)) if vals.size else float("nan"),
                "sd_subject_seed_MAE": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                "median_subject_seed_MAE": float(np.median(vals)) if vals.size else float("nan"),
                "iqr_subject_seed_MAE": float(q75 - q25) if vals.size else float("nan"),
                "RMSE": float(g["RMSE"].mean()),
                "bias": float(g["bias"].mean()),
                "Pearson_correlation": float(g["Pearson_correlation"].mean()),
                "number_of_windows": int(g["number_of_windows"].sum()),
            }
        )
    return pd.DataFrame(rows)


def assert_no_target_subject_in_source(held_out: str, cache: Dict[str, np.ndarray], split: str) -> None:
    key = "subject_id" if "subject_id" in cache else "source_subject_id"
    present = set(map(str, cache.get(key, [])))
    if held_out in present:
        raise RuntimeError(f"Leakage check failed: held-out subject {held_out} appears in {split} cache.")


def apply_smoke_overrides(args: Any) -> None:
    if not bool(getattr(args, "smoke", False)):
        return
    args.subjects = ["S12"]
    args.seeds = [0] if hasattr(args, "seeds") else getattr(args, "seeds", [0])
    args.seed = 0
    args.max_train_batches = 1
    args.max_val_batches = 1
    args.max_test_batches = 1
    args.max_epochs = min(int(args.max_epochs), 2)
    args.patience = min(int(args.patience), 1)
    args.bootstrap_resamples = 100
    args.num_workers = 0
    args.batch_size = min(int(args.batch_size), 8)
    args.cached_feature_batch_size = min(int(args.cached_feature_batch_size), 16)
