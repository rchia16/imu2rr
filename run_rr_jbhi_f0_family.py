#!/usr/bin/env python3
"""Run the seeded F0 RR family: full model plus three ablations.

This helper runs the cross-modal LOSO benchmark once per requested variant and
writes:
  - per-variant raw outputs under <out-dir>/<variant>/
  - a combined subject-row export at <out-dir>/subject_rows.csv
  - a combined summary table at <out-dir>/summary.csv
  - a manifest describing the exact variant settings
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from rr_readout_phase_abc_experiment import BinAffineDistribution
from rr_readout_phase_abc_utils import expected_rr, gaussian_target_bpm, spectral_arrays_from_stft
from rr_spectral_readout_experiment import rr_metric_dict
from vit_pressure_crossmodal_stft_rr_adaptation_alpha_hat_sweep import (
    TensorWindows,
    _args_for_rr_tta_mode,
    _collect_tensor_windows,
    _make_eval_indices,
    _profile_conditioning_mode_for_rr_mode,
    _profile_stats_from_loader,
    _profile_stats_from_windows,
    add_common_adaptation_args,
    feature_adaptive_evaluate,
)
from vit_pressure_crossmodal_stft_rr_core import (
    build_base_parser,
    make_fold_seed,
    run_loocv_experiment,
    set_seed,
    split_target_calibration_eval,
    unpack_batch,
)


DEFAULT_SUBJECTS = "S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29"
DEFAULT_VARIANTS = "full no_film no_tcn no_qkv"
CANONICAL_METHODS = {
    "native_rr_head": "F0 native RR head",
    "original_rr_readout": "F0 original RR readout",
    "a1_gaussian_kl": "F0 + A1 Gaussian-KL spectral readout",
}
ORIGINAL_READOUT_MODE = "tcn_profile_film_qkv_last1_0p01"


VARIANT_CONFIGS: Dict[str, Dict[str, object]] = {
    "full": {
        "use_profile_film": True,
        "use_profile_qkv": True,
        "shared_profile_qkv": True,
        "profile_conditioning": "film_qkv",
        "use_tcn_token_mixer": True,
        "tcn_mixer_alpha": 0.05,
        "profile_qkv_layers": "last1",
        "profile_qkv_scale": 0.01,
        "profile_qkv_residual": True,
        "profile_qkv_mode": "static",
        "profile_clsa_enable_fast_update": 0,
    },
    "no_film": {
        "use_profile_film": False,
        "use_profile_qkv": True,
        "shared_profile_qkv": True,
        "profile_conditioning": "qkv",
        "use_tcn_token_mixer": True,
        "tcn_mixer_alpha": 0.05,
        "profile_qkv_layers": "last1",
        "profile_qkv_scale": 0.01,
        "profile_qkv_residual": True,
        "profile_qkv_mode": "static",
        "profile_clsa_enable_fast_update": 0,
    },
    "no_tcn": {
        "use_profile_film": True,
        "use_profile_qkv": True,
        "shared_profile_qkv": True,
        "profile_conditioning": "film_qkv",
        "use_tcn_token_mixer": False,
        "tcn_mixer_alpha": 0.05,
        "profile_qkv_layers": "last1",
        "profile_qkv_scale": 0.01,
        "profile_qkv_residual": True,
        "profile_qkv_mode": "static",
        "profile_clsa_enable_fast_update": 0,
    },
    "no_qkv": {
        "use_profile_film": True,
        "use_profile_qkv": False,
        "shared_profile_qkv": False,
        "profile_conditioning": "film",
        "use_tcn_token_mixer": True,
        "tcn_mixer_alpha": 0.05,
        "profile_qkv_layers": "none",
        "profile_qkv_scale": 0.01,
        "profile_qkv_residual": False,
        "profile_qkv_mode": "static",
        "profile_clsa_enable_fast_update": 0,
    },
}


def parse_list(text: str) -> List[str]:
    return [t.strip() for t in str(text).replace(",", " ").split() if t.strip()]


def build_variant_args(args, variant: str):
    variant_args = copy.deepcopy(args)
    variant_args.out_dir = str(Path(args.out_dir) / variant)
    for key, value in VARIANT_CONFIGS[variant].items():
        setattr(variant_args, key, value)
    return variant_args


def checkpoint_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_array(values: np.ndarray) -> str:
    arr = np.asarray(values)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def index_json(values: np.ndarray) -> str:
    return json.dumps([int(x) for x in np.asarray(values, dtype=np.int64).reshape(-1).tolist()])


def parse_index_json(raw: object) -> np.ndarray:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.zeros((0,), dtype=np.int64)
    if isinstance(raw, np.ndarray):
        return raw.astype(np.int64, copy=False).reshape(-1)
    if isinstance(raw, (list, tuple)):
        return np.asarray(raw, dtype=np.int64).reshape(-1)
    text = str(raw).strip()
    if not text:
        return np.zeros((0,), dtype=np.int64)
    return np.asarray(json.loads(text), dtype=np.int64).reshape(-1)


def eval_indices_from_metrics(metrics: Dict[str, object]) -> np.ndarray:
    n_total = int(metrics.get("rr_probe_n_target_total", metrics.get("rr_probe_n_eval", 0)))
    cal_idx = parse_index_json(metrics.get("target_calibration_indices", "[]"))
    exclude = bool(int(metrics.get("eval_excludes_calibration", 0)))
    return _make_eval_indices(n_total, cal_idx, exclude)


def lineage_fields(cal_idx: np.ndarray, eval_idx: np.ndarray) -> Dict[str, object]:
    cal_idx = np.asarray(cal_idx, dtype=np.int64).reshape(-1)
    eval_idx = np.asarray(eval_idx, dtype=np.int64).reshape(-1)
    return {
        "calibration_indices": index_json(cal_idx),
        "evaluation_indices": index_json(eval_idx),
        "calibration_indices_hash": hash_array(cal_idx),
        "evaluation_indices_hash": hash_array(eval_idx),
        "calibration_evaluation_overlap_count": int(np.intersect1d(cal_idx, eval_idx).size),
        "n_calibration_windows": int(cal_idx.size),
        "n_evaluation_windows": int(eval_idx.size),
    }


def metric_sources(raw: str) -> List[str]:
    src = parse_list(raw)
    if src == ["all"] or "all" in src:
        return ["native_rr_head", "original_rr_readout", "a1_gaussian_kl"]
    invalid = [x for x in src if x not in CANONICAL_METHODS]
    if invalid:
        raise SystemExit(f"Unsupported metric source(s): {invalid}.")
    return src


def native_rows(summary: pd.DataFrame, seed: int, variant: str) -> pd.DataFrame:
    df = normalize_metrics(summary).copy()
    df["metric_source"] = "native_rr_head"
    df["benchmark_method"] = CANONICAL_METHODS["native_rr_head"]
    df["family"] = "f0_native"
    df["model"] = "crossmodal_rr"
    df["mode"] = variant
    df["variant"] = variant
    df["seed"] = int(seed)
    df["native_rr_mae"] = df["mae"]
    df["native_rr_rmse"] = df["rmse"]
    df["native_rr_corr"] = df["corr"]
    return df


def original_readout_hook(model, sbj: str, subjects, train_loader, test_loader, device: str, args, sbj_dir: Path):
    fold_seed = int(getattr(args, "fold_seed", make_fold_seed(int(args.seed), sbj)))
    set_seed(fold_seed)
    mode_args = _args_for_rr_tta_mode(args, ORIGINAL_READOUT_MODE)
    mode_args.seed = fold_seed
    mode_args.calibration_seed = fold_seed
    mode_args.profile_unsup_adapt_scope = "calibration"
    metrics = feature_adaptive_evaluate(
        model,
        train_loader,
        test_loader,
        sbj,
        device,
        mode_args,
        out_dir=sbj_dir / "original_rr_readout",
    )
    cal_idx = parse_index_json(metrics.get("target_calibration_indices", "[]"))
    eval_idx = eval_indices_from_metrics(metrics)
    best_path = sbj_dir / "best_model.pt"
    row = {
        "subject": sbj,
        "metric_source": "original_rr_readout",
        "benchmark_method": CANONICAL_METHODS["original_rr_readout"],
        "family": "f0",
        "model": "crossmodal_rr",
        "base_seed": int(getattr(args, "base_seed", args.seed)),
        "fold_seed": fold_seed,
        "calibration_seed": fold_seed,
        "f0_checkpoint_path": str(best_path),
        "f0_checkpoint_sha256": checkpoint_sha256(best_path) if best_path.exists() else "",
        "historical_runner_file": "vit_pressure_crossmodal_stft_rr_adaptation_alpha_hat_sweep.py",
        "historical_readout_source_file": "vit_pressure_crossmodal_stft_rr_rrprobe_tta_main.py",
        "historical_function_call_site": "feature_adaptive_evaluate/profile_vector_unsupervised_evaluate",
        "representation_tensor": "pooled transformer hidden features",
        "probe_class": "FaithfulRRRegressor",
        "probe_hyperparameters": json.dumps(
            {
                "epochs": int(args.rr_probe_epochs),
                "lr": float(args.rr_probe_lr),
                "weight_decay": float(args.rr_probe_weight_decay),
                "batch_size": int(args.rr_probe_batch_size),
                "train_adapter": bool(args.rr_probe_train_adapter),
            },
            sort_keys=True,
        ),
        "calibration_policy": str(args.target_calibration_mode),
        "profile_unsup_adapt_scope": str(mode_args.profile_unsup_adapt_scope),
        "target_labels_used_for_profile": 0,
        "target_labels_used_for_readout_adaptation": int(metrics.get("uses_target_rr_labels_for_adaptation", 0)),
        **lineage_fields(cal_idx, eval_idx),
        **metrics,
    }
    row["mae"] = float(metrics["rr_probe_post_mae"])
    row["rmse"] = float(metrics["rr_probe_post_rmse"])
    row["corr"] = float(metrics["rr_probe_post_corr"])
    return [{"__summary_name__": "original_rr_readout_summary", **row}]


def fixed_profile_context(model, test_loader, device: str, args, fold_seed: int) -> Dict[str, object]:
    mode_args = _args_for_rr_tta_mode(args, ORIGINAL_READOUT_MODE)
    mode_args.seed = int(fold_seed)
    mode_args.calibration_seed = int(fold_seed)
    mode_args.profile_unsup_adapt_scope = "calibration"
    target_windows = _collect_tensor_windows(model, test_loader, device, max_windows=0)
    n = int(target_windows.imu.size(0))
    dummy_x = np.zeros((n, 1), dtype=np.float32)
    y_target = target_windows.rr.detach().cpu().numpy().reshape(-1).astype(np.float32)
    _x_cal, _y_cal, _k_cal, _x_eval, _y_eval, _k_eval, cal_idx = split_target_calibration_eval(
        dummy_x,
        y_target,
        None,
        int(mode_args.target_calibration_windows),
        seed=int(fold_seed),
        mode=str(mode_args.target_calibration_mode),
        exclude_calibration_from_eval=bool(mode_args.exclude_calibration_from_eval),
    )
    eval_idx = _make_eval_indices(n, cal_idx, bool(mode_args.exclude_calibration_from_eval))
    cal_windows = target_windows.subset(cal_idx)
    profile_batch_size = int(getattr(mode_args, "feature_eval_batch_size", 256))
    _profile_raw, profile_norm = _profile_stats_from_windows(
        model,
        cal_windows,
        device,
        mode_args,
        batch_size=profile_batch_size,
    )
    with torch.no_grad():
        profile_vector = model.profile_encoder(profile_norm.unsqueeze(0)).detach()
    return {
        "mode_args": mode_args,
        "target_windows": target_windows,
        "eval_windows": target_windows.subset(eval_idx),
        "cal_idx": cal_idx,
        "eval_idx": eval_idx,
        "profile_vector": profile_vector,
        "conditioning_mode": _profile_conditioning_mode_for_rr_mode(ORIGINAL_READOUT_MODE),
    }


def source_profile_context(model, train_loader, device: str, args, fold_seed: int) -> Dict[str, object]:
    mode_args = _args_for_rr_tta_mode(args, ORIGINAL_READOUT_MODE)
    mode_args.seed = int(fold_seed)
    mode_args.calibration_seed = int(fold_seed)
    profile_raw, profile_norm = _profile_stats_from_loader(model, train_loader, device, mode_args)
    del profile_raw
    with torch.no_grad():
        profile_vector = model.profile_encoder(profile_norm.unsqueeze(0)).detach()
    return {
        "mode_args": mode_args,
        "profile_vector": profile_vector,
        "conditioning_mode": _profile_conditioning_mode_for_rr_mode(ORIGINAL_READOUT_MODE),
    }


@torch.no_grad()
def collect_profile_conditioned_spectral_cache_from_loader(
    model,
    loader,
    device: str,
    profile_vector: torch.Tensor,
    conditioning_mode: str,
) -> Dict[str, np.ndarray]:
    model.eval()
    logits: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    bins: Optional[np.ndarray] = None
    for batch in loader:
        imu, pressure, _cond, br, _tlx = unpack_batch(batch, device)
        p = profile_vector.to(device).expand(imu.size(0), -1)
        pred_logmag, _rr_pred, _hidden, _profile = model.forward_profile_conditioned(
            imu.float(),
            profile_vector=p,
            conditioning_mode=conditioning_mode,
        )
        spec = spectral_arrays_from_stft(pred_logmag.detach().cpu().numpy(), soft_temperature=0.1)
        logits.append(spec["spectral_logits"].astype(np.float32))
        if bins is None:
            bins = spec["frequency_bins_bpm"].astype(np.float32)
        rr_true = br.detach().cpu().numpy().reshape(-1).astype(np.float32)
        labels.append(rr_true)
    if bins is None:
        raise RuntimeError("No batches available for A1 spectral cache.")
    return {
        "spectral_logits": np.concatenate(logits, axis=0),
        "rr_true": np.concatenate(labels, axis=0),
        "frequency_bins_bpm": bins,
    }


@torch.no_grad()
def collect_profile_conditioned_spectral_cache_from_windows(
    model,
    windows: TensorWindows,
    device: str,
    profile_vector: torch.Tensor,
    conditioning_mode: str,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    model.eval()
    logits: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    bins: Optional[np.ndarray] = None
    n = int(windows.imu.size(0))
    for st in range(0, n, int(batch_size)):
        imu = windows.imu[st : st + int(batch_size)].to(device, non_blocking=True).float()
        p = profile_vector.to(device).expand(imu.size(0), -1)
        pred_logmag, _rr_pred, _hidden, _profile = model.forward_profile_conditioned(
            imu,
            profile_vector=p,
            conditioning_mode=conditioning_mode,
        )
        spec = spectral_arrays_from_stft(pred_logmag.detach().cpu().numpy(), soft_temperature=0.1)
        logits.append(spec["spectral_logits"].astype(np.float32))
        if bins is None:
            bins = spec["frequency_bins_bpm"].astype(np.float32)
        labels.append(windows.rr[st : st + int(batch_size)].detach().cpu().numpy().reshape(-1).astype(np.float32))
    if bins is None:
        raise RuntimeError("No windows available for A1 spectral cache.")
    return {
        "spectral_logits": np.concatenate(logits, axis=0),
        "rr_true": np.concatenate(labels, axis=0),
        "frequency_bins_bpm": bins,
    }


def train_a1_readout(
    train: Dict[str, np.ndarray],
    val: Dict[str, np.ndarray],
    args,
    device: str,
) -> Tuple[BinAffineDistribution, List[Dict[str, float]]]:
    set_seed(int(getattr(args, "fold_seed", args.seed)))
    model = BinAffineDistribution(train["spectral_logits"].shape[1]).to(device)
    n_trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    if n_trainable != 20:
        raise RuntimeError(f"Historical A1 should have 20 trainable scale/bias parameters, got {n_trainable}.")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x = torch.as_tensor(train["spectral_logits"], device=device).float()
    y = torch.as_tensor(train["rr_true"], device=device).float()
    bins = torch.as_tensor(train["frequency_bins_bpm"], device=device).float()
    vx = torch.as_tensor(val["spectral_logits"], device=device).float()
    vy = val["rr_true"].astype(np.float32)
    vbins = torch.as_tensor(val["frequency_bins_bpm"], device=device).float()
    best_state = copy.deepcopy(model.state_dict())
    best_mae = float("inf")
    wait = 0
    hist: List[Dict[str, float]] = []
    gen = torch.Generator(device=device).manual_seed(int(getattr(args, "fold_seed", args.seed)))
    for epoch in range(1, 31):
        model.train()
        perm = torch.randperm(x.size(0), generator=gen, device=device)
        losses = []
        for st in range(0, x.size(0), 128):
            idx = perm[st : st + 128]
            prob = model(x[idx])
            q = gaussian_target_bpm(y[idx], bins, 1.0)
            loss = -(q * torch.log(prob.clamp_min(1e-8))).sum(dim=1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            pred = expected_rr(model(vx), vbins).detach().cpu().numpy()
        val_mae = float(np.mean(np.abs(pred - vy)))
        hist.append({"epoch": float(epoch), "train_loss": float(np.mean(losses)), "val_mae": val_mae})
        if val_mae < best_mae:
            best_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= 5:
            break
    model.load_state_dict(best_state)
    return model, hist


def a1_gaussian_kl_hook(model, sbj: str, subjects, train_loader, val_loader, test_loader, device: str, args, sbj_dir: Path):
    fold_seed = int(getattr(args, "fold_seed", make_fold_seed(int(args.seed), sbj)))
    set_seed(fold_seed)
    for p in model.parameters():
        p.requires_grad = False
    source_context = source_profile_context(model, train_loader, device, args, fold_seed)
    target_context = fixed_profile_context(model, test_loader, device, args, fold_seed)
    source_profile_vector = source_context["profile_vector"]
    target_profile_vector = target_context["profile_vector"]
    source_conditioning_mode = str(source_context["conditioning_mode"])
    target_conditioning_mode = str(target_context["conditioning_mode"])
    train = collect_profile_conditioned_spectral_cache_from_loader(
        model,
        train_loader,
        device,
        source_profile_vector,  # type: ignore[arg-type]
        source_conditioning_mode,
    )
    val = collect_profile_conditioned_spectral_cache_from_loader(
        model,
        val_loader,
        device,
        source_profile_vector,  # type: ignore[arg-type]
        source_conditioning_mode,
    )
    test = collect_profile_conditioned_spectral_cache_from_windows(
        model,
        target_context["eval_windows"],  # type: ignore[arg-type]
        device,
        target_profile_vector,  # type: ignore[arg-type]
        target_conditioning_mode,
        batch_size=int(getattr(args, "feature_eval_batch_size", 256)),
    )
    readout, hist = train_a1_readout(train, val, args, device)
    trainable_count = int(sum(p.numel() for p in readout.parameters() if p.requires_grad))
    with torch.no_grad():
        prob = readout(torch.as_tensor(test["spectral_logits"], device=device).float())
        bins = torch.as_tensor(test["frequency_bins_bpm"], device=device).float()
        pred = expected_rr(prob, bins).detach().cpu().numpy().astype(np.float32)
    y = test["rr_true"].astype(np.float32)
    metrics = rr_metric_dict(y, pred)
    ckpt_dir = sbj_dir / "a1_gaussian_kl"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": readout.state_dict(),
            "method": "a1_gaussian_kl",
            "readout_class": "BinAffineDistribution",
            "trainable_parameter_count": trainable_count,
            "target_sigma_bpm": 1.0,
            "max_epochs": 30,
            "patience": 5,
            "history": hist,
            "frequency_bins_bpm": test["frequency_bins_bpm"],
            "calibration_indices": np.asarray(target_context["cal_idx"], dtype=np.int64),
            "evaluation_indices": np.asarray(target_context["eval_idx"], dtype=np.int64),
        },
        ckpt_dir / "best.pt",
    )
    pd.DataFrame(
        {
            "window_index_eval": np.asarray(target_context["eval_idx"], dtype=np.int64),
            "rr_true": y,
            "rr_pred": pred,
        }
    ).to_csv(ckpt_dir / f"a1_predictions_{sbj}.csv", index=False)
    lineage = lineage_fields(
        np.asarray(target_context["cal_idx"], dtype=np.int64),
        np.asarray(target_context["eval_idx"], dtype=np.int64),
    )
    (ckpt_dir / "lineage.json").write_text(
        json.dumps(
            {
                **lineage,
                "target_labels_used_for_profile": 0,
                "target_labels_used_for_a1_training": 0,
                "calibration_policy": str(target_context["mode_args"].target_calibration_mode),  # type: ignore[union-attr]
                "profile_unsup_adapt_scope": str(target_context["mode_args"].profile_unsup_adapt_scope),  # type: ignore[union-attr]
                "exclude_calibration_from_eval": bool(target_context["mode_args"].exclude_calibration_from_eval),  # type: ignore[union-attr]
                "source_profile_source": "source_train_loader",
                "target_profile_source": "held_out_target_calibration_windows",
                "source_profile_conditioning_mode": source_conditioning_mode,
                "target_profile_conditioning_mode": target_conditioning_mode,
                "profile_conditioning_mode": target_conditioning_mode,
                "spectral_cache_source": "model.forward_profile_conditioned",
                "readout_class": "BinAffineDistribution",
                "frequency_bins_hash": hash_array(test["frequency_bins_bpm"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    best_path = sbj_dir / "best_model.pt"
    row = {
        "subject": sbj,
        "metric_source": "a1_gaussian_kl",
        "benchmark_method": CANONICAL_METHODS["a1_gaussian_kl"],
        "family": "f0_a1",
        "model": "crossmodal_rr",
        "base_seed": int(getattr(args, "base_seed", args.seed)),
        "fold_seed": fold_seed,
        "calibration_seed": fold_seed,
        "f0_checkpoint_path": str(best_path),
        "f0_checkpoint_sha256": checkpoint_sha256(best_path) if best_path.exists() else "",
        "test_target_rr_hash": hash_array(y),
        "frequency_bins_hash": hash_array(test["frequency_bins_bpm"]),
        "spectral_cache_source": "model.forward_profile_conditioned",
        "source_profile_source": "source_train_loader",
        "target_profile_source": "held_out_target_calibration_windows",
        "source_profile_conditioning_mode": source_conditioning_mode,
        "target_profile_conditioning_mode": target_conditioning_mode,
        "profile_conditioning_mode": target_conditioning_mode,
        "readout_class": "BinAffineDistribution",
        "readout_state_keys": json.dumps(sorted(readout.state_dict().keys())),
        "trainable_parameter_count": trainable_count,
        "target_sigma_bpm": 1.0,
        "max_epochs": 30,
        "patience": 5,
        "checkpoint_selection": "source validation MAE",
        "calibration_policy": str(target_context["mode_args"].target_calibration_mode),  # type: ignore[union-attr]
        "profile_unsup_adapt_scope": str(target_context["mode_args"].profile_unsup_adapt_scope),  # type: ignore[union-attr]
        "target_labels_used_for_profile": 0,
        "target_labels_used_for_a1_training": 0,
        **lineage,
        "mae": float(metrics["MAE"]),
        "rmse": float(metrics["RMSE"]),
        "corr": float(metrics["Pearson_correlation"]),
        "number_of_windows": int(metrics["number_of_windows"]),
        **{f"a1_{k}": v for k, v in metrics.items()},
    }
    return [{"__summary_name__": "a1_gaussian_kl_summary", **row}]


def summarize_subject_rows(df: pd.DataFrame, *, family: str, mode: str, model: str, seed: int) -> Dict[str, object]:
    numeric_cols = [
        col
        for col in df.columns
        if col not in {"subject", "family", "model", "mode", "seed", "variant"}
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    row: Dict[str, object] = {
        "family": family,
        "model": model,
        "mode": mode,
        "variant": mode,
        "seed": int(seed),
        "n_subjects": int(df["subject"].nunique()),
    }
    for col in numeric_cols:
        row[f"{col}_mean"] = float(df[col].mean())
        row[f"{col}_std"] = float(df[col].std())
    return row


def normalize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {}
    for src, dst in (("rr_mae", "mae"), ("rr_rmse", "rmse"), ("rr_corr", "corr")):
        if src in out.columns and dst not in out.columns:
            rename_map[src] = dst
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def main() -> None:
    parser = build_base_parser(DEFAULT_SUBJECTS.split(), "rr_jbhi_f0_family")
    add_common_adaptation_args(parser)
    parser.set_defaults(
        decoder_mode="cross_attn",
        rr_head_type="mlp",
        epochs=20,
        batch_size=16,
        target_calibration_mode="random",
        exclude_calibration_from_eval=False,
        profile_clsa_enable_fast_update=0,
    )
    parser.add_argument("--variants", default=DEFAULT_VARIANTS, help="Space- or comma-separated F0 variants to run.")
    parser.add_argument(
        "--metric-source",
        choices=["native_rr_head", "original_rr_readout", "a1_gaussian_kl", "all"],
        default="original_rr_readout",
    )
    args = parser.parse_args()

    variants = parse_list(args.variants)
    invalid = [variant for variant in variants if variant not in VARIANT_CONFIGS]
    if invalid:
        raise SystemExit(f"Unsupported F0 variant(s): {invalid}. Valid variants: {sorted(VARIANT_CONFIGS)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_rows = []
    summary_rows = []
    variant_specs = []
    requested_sources = metric_sources(args.metric_source)

    for variant in variants:
        variant_args = build_variant_args(args, variant)
        variant_dir = Path(variant_args.out_dir)
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_specs.append({"variant": variant, "out_dir": str(variant_dir), "config": VARIANT_CONFIGS[variant]})

        post_hooks = []
        if "original_rr_readout" in requested_sources:
            post_hooks.append(original_readout_hook)
        if "a1_gaussian_kl" in requested_sources and variant == "full":
            post_hooks.append(a1_gaussian_kl_hook)

        outputs = run_loocv_experiment(variant_args, post_eval_hooks=post_hooks)
        source_frames: List[pd.DataFrame] = []
        if "native_rr_head" in requested_sources:
            source_frames.append(native_rows(outputs["summary"], int(args.seed), variant))
        if "original_rr_readout" in requested_sources and "original_rr_readout_summary" in outputs:
            original = outputs["original_rr_readout_summary"].copy()
            original["mode"] = variant
            original["variant"] = variant
            original["seed"] = int(args.seed)
            source_frames.append(original)
        if "a1_gaussian_kl" in requested_sources and "a1_gaussian_kl_summary" in outputs:
            a1 = outputs["a1_gaussian_kl_summary"].copy()
            a1["mode"] = variant
            a1["variant"] = variant
            a1["seed"] = int(args.seed)
            source_frames.append(a1)

        if not source_frames:
            continue
        subject_df = pd.concat(source_frames, ignore_index=True)
        combined_rows.append(subject_df)

        for metric_source, g in subject_df.groupby("metric_source", dropna=False):
            method = str(g["benchmark_method"].iloc[0]) if "benchmark_method" in g else str(metric_source)
            summary = summarize_subject_rows(
                g,
                family=str(g["family"].iloc[0]) if "family" in g else "f0",
                mode=variant,
                model="crossmodal_rr",
                seed=int(args.seed),
            )
            summary["metric_source"] = str(metric_source)
            summary["benchmark_method"] = method
            summary_rows.append(summary)

    combined_subject_rows = pd.concat(combined_rows, ignore_index=True) if combined_rows else pd.DataFrame()
    combined_summary = pd.DataFrame(summary_rows)

    combined_subject_rows.to_csv(out_dir / "subject_rows.csv", index=False)
    combined_summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "family": "f0",
                "seed": int(args.seed),
                "variants": variants,
                "variant_configs": variant_specs,
                "metric_sources": requested_sources,
                "canonical_method_names": CANONICAL_METHODS,
                "out_dir": str(out_dir),
                "subjects": args.subjects,
                "data_str": args.data_str,
                "data_dir": args.data_dir,
                "data_group": args.data_group,
                "mdl_dir": args.mdl_dir,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(combined_summary.to_string(index=False))
    print(f"[DONE] wrote {out_dir}")


if __name__ == "__main__":
    main()
