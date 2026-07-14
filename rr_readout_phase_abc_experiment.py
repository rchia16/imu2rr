from __future__ import annotations

import argparse
import copy
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import M_DIR, SBJ_PROCESSED_DIR
from rr_spectral_readout_experiment import (
    DEFAULT_SUBJECTS,
    BR_FS,
    SoftSpectralRR,
    collect_predictions,
    build_loaders_for_subject,
    model_from_checkpoint,
    rr_metric_dict,
    subject_checkpoint_path,
)
from vit_pressure_crossmodal_profile_encoder import estimate_rr_from_predicted_stft
from rr_readout_phase_abc_utils import (
    CONFIDENCE_FEATURES,
    PHASE_A_METHODS,
    PHASE_B_METHODS,
    PHASE_C_METHODS,
    apply_smoke_overrides,
    assert_no_target_subject_in_source,
    atomic_npz,
    confidence_matrix,
    expected_rr,
    gaussian_target_bpm,
    load_npz,
    method_summary,
    per_subject_metrics,
    run_manifest,
    set_all_seeds,
    spectral_arrays_from_stft,
    stable_config_hash,
    standardize_apply,
    standardize_fit,
    subject_aggregated_mae,
    trainable_parameter_count,
    write_csv,
    write_json,
)


SPECTRAL_A = {"A0_soft_spectral", "A1_gaussian_kl", "A2_wasserstein", "A3_kl_rr_mae"}
HYBRID_A = {"A4_hidden_only", "A5_spec_linear_residual", "A6_spec_hidden_conf_residual"}
CORRECTIVE_PHASE_A_METHODS = ["A0_soft_spectral", "A1_gaussian_kl", "A3_kl_rr_mae"]
CORRECTIVE_PHASE_B_METHODS = ["B0_always_spec", "B1_always_hidden", "B3_relative_advantage_gate", "B4_oracle_gate"]
CORRECTIVE_PHASE_C_METHODS = [
    "C0_none",
    "C1_feature_mean",
    "C3_affine_readout",
    "C4_confidence_gated_affine",
    "C5_oracle_offset",
    "C6_oracle_affine",
]


class BinAffineDistribution(nn.Module):
    def __init__(self, n_bins: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_bins))
        self.bias = nn.Parameter(torch.zeros(n_bins))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits * self.scale.view(1, -1) + self.bias.view(1, -1), dim=1)


class HiddenLinear(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.head = nn.Linear(d_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


class LinearResidual(nn.Module):
    def __init__(self, d_in: int, max_delta_bpm: float):
        super().__init__()
        self.max_delta_bpm = float(max_delta_bpm)
        self.head = nn.Linear(d_in, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_delta_bpm * torch.tanh(self.head(x).squeeze(-1))

    def forward(self, base: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return base.view(-1) + self.residual(x)


class SmallResidualMLP(nn.Module):
    def __init__(self, d_in: int, max_delta_bpm: float, dropout: float = 0.1):
        super().__init__()
        self.max_delta_bpm = float(max_delta_bpm)
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_delta_bpm * torch.tanh(self.net(x).squeeze(-1))

    def forward(self, base: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return base.view(-1) + self.residual(x)


class LogisticGate(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.linear = nn.Linear(d_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x).squeeze(-1))


class AdvantageRegressor(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.linear = nn.Linear(d_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


@dataclass
class AffineState:
    raw_a: torch.Tensor
    raw_b: torch.Tensor

    def values(self, a_range: Tuple[float, float], max_abs_bias: float) -> Tuple[torch.Tensor, torch.Tensor]:
        lo, hi = a_range
        a = lo + (hi - lo) * torch.sigmoid(self.raw_a)
        b = max_abs_bias * torch.tanh(self.raw_b)
        return a, b


def cache_paths(args: argparse.Namespace, subject: str) -> Dict[str, Path]:
    base = Path(args.out_dir) / "cache" / f"seed_{int(args.seed):03d}" / subject
    return {"train": base / "train.npz", "val": base / "val.npz", "test": base / "test.npz", "manifest": base / "cache_manifest.json"}


def cache_hash_payload(args: argparse.Namespace, subject: str, ckpt_path: Path) -> Dict[str, Any]:
    return {
        "subject": subject,
        "cache_schema": "phase_abc_v2_predicted_stft",
        "source_subjects": list(args.source_subjects),
        "seed": int(args.seed),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_mtime_ns": ckpt_path.stat().st_mtime_ns if ckpt_path.exists() else None,
        "data_str": args.data_str,
        "data_group": args.data_group,
        "val_split": float(args.val_split),
        "soft_temperature": float(args.soft_temperature),
        "batch_size": int(args.batch_size),
        "max_train_batches": int(args.max_train_batches),
        "max_val_batches": int(args.max_val_batches),
        "max_test_batches": int(args.max_test_batches),
    }


def collect_cache_split(
    args: argparse.Namespace,
    model: nn.Module,
    loader: Any,
    split: str,
) -> Dict[str, np.ndarray]:
    limit = {"train": args.max_train_batches, "val": args.max_val_batches, "test": args.max_test_batches}[split]
    df, arr = collect_predictions(
        model,
        loader,
        args.device,
        temperature=float(args.soft_temperature),
        max_batches=int(limit),
    )
    rr_direct = arr["rr_direct"].reshape(-1).astype(np.float32)
    spec = spectral_arrays_from_stft(arr["predicted_stft"], soft_temperature=float(args.soft_temperature), hidden_rr=rr_direct)
    n = len(rr_direct)
    out: Dict[str, np.ndarray] = {
        "rr_true": arr["rr_true"].reshape(-1).astype(np.float32),
        "rr_direct": rr_direct,
        "predicted_stft": arr["predicted_stft"].astype(np.float32),
        "pooled_hidden": arr["pooled_hidden"].astype(np.float32),
        "attention_hidden": arr["hidden_tokens"].astype(np.float32),
        "final_hidden_sequence": arr["hidden_tokens"].astype(np.float32),
        "window_index": df["window_index"].to_numpy(dtype=np.int64),
        "condition": np.zeros(n, dtype=np.int64),
        "subject_id": df["subject_id"].to_numpy(dtype=str),
    }
    out.update(spec)
    return out


def run_cache(args: argparse.Namespace) -> None:
    rows: List[Dict[str, Any]] = []
    for subject in args.subjects:
        paths = cache_paths(args, subject)
        loader_args = argparse.Namespace(**{**vars(args), "subjects": list(args.source_subjects)})
        train_loader, val_loader, test_loader = build_loaders_for_subject(loader_args, subject, include_subject_id=True, shuffle_train=False)
        ckpt_path = subject_checkpoint_path(Path(args.checkpoint_root), subject, args.checkpoint_name)
        payload = cache_hash_payload(args, subject, ckpt_path)
        cfg_hash = stable_config_hash(payload)
        if args.resume and all(paths[k].exists() for k in ("train", "val", "test", "manifest")):
            existing = json.loads(paths["manifest"].read_text())
            if existing.get("configuration_hash") == cfg_hash:
                rows.append({"subject": subject, "seed": int(args.seed), "status": "reused", "configuration_hash": cfg_hash})
                continue
        model, ckpt, _ckpt_args = model_from_checkpoint(ckpt_path, train_loader, args.device)
        for p in model.parameters():
            p.requires_grad = False
        for split, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
            data = collect_cache_split(args, model, loader, split)
            atomic_npz(paths[split], **data)
        manifest = {
            **run_manifest(args),
            "subject": subject,
            "source_checkpoint_path": str(ckpt_path),
            "source_checkpoint_epoch": ckpt.get("epoch", ""),
            "configuration_hash": cfg_hash,
            "cache_files": {k: str(v) for k, v in paths.items() if k != "manifest"},
            "target_labels_in_cache_for_evaluation_only": True,
            "trainable_parameter_count": 0,
        }
        write_json(paths["manifest"], manifest)
        rows.append({"subject": subject, "seed": int(args.seed), "status": "rebuilt", "configuration_hash": cfg_hash})
        close = getattr(train_loader, "_iterator", None)
        del close, model
        gc.collect()
    write_csv(Path(args.out_dir) / f"cache_seed_{int(args.seed):03d}_status.csv", pd.DataFrame(rows))


def ensure_cache(args: argparse.Namespace, subject: str) -> Dict[str, Dict[str, np.ndarray]]:
    paths = cache_paths(args, subject)
    if not all(paths[k].exists() for k in ("train", "val", "test")):
        run_cache_for_subject(args, subject)
    train = load_npz(paths["train"])
    val = load_npz(paths["val"])
    test = load_npz(paths["test"])
    if bool(getattr(args, "corrective", False)) and "predicted_stft" not in test:
        run_cache_for_subject(args, subject)
        train = load_npz(paths["train"])
        val = load_npz(paths["val"])
        test = load_npz(paths["test"])
    assert_no_target_subject_in_source(subject, train, "source train")
    assert_no_target_subject_in_source(subject, val, "source validation")
    return {"train": train, "val": val, "test": test}


def run_cache_for_subject(args: argparse.Namespace, subject: str) -> None:
    original = list(args.subjects)
    args.subjects = original
    run_cache(argparse.Namespace(**{**vars(args), "subjects": [subject]}))
    args.subjects = original


def train_distribution_method(
    method: str,
    train: Dict[str, np.ndarray],
    val: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Tuple[BinAffineDistribution, List[Dict[str, Any]]]:
    device = torch.device(args.device)
    model = BinAffineDistribution(train["spectral_logits"].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    x = torch.as_tensor(train["spectral_logits"], device=device).float()
    y = torch.as_tensor(train["rr_true"], device=device).float()
    bins = torch.as_tensor(train["frequency_bins_bpm"], device=device).float()
    vx = torch.as_tensor(val["spectral_logits"], device=device).float()
    vy_np = val["rr_true"].astype(np.float32)
    vbins = torch.as_tensor(val["frequency_bins_bpm"], device=device).float()
    best = copy.deepcopy(model.state_dict())
    best_mae = float("inf")
    wait = 0
    hist: List[Dict[str, Any]] = []
    gen = torch.Generator(device=device).manual_seed(int(args.seed))
    for epoch in range(1, int(args.max_epochs) + 1):
        model.train()
        perm = torch.randperm(x.size(0), generator=gen, device=device)
        losses = []
        for st in range(0, x.size(0), int(args.cached_feature_batch_size)):
            idx = perm[st : st + int(args.cached_feature_batch_size)]
            prob = model(x[idx])
            q = gaussian_target_bpm(y[idx], bins, float(args.target_sigma_bpm))
            if method == "A2_wasserstein":
                loss = torch.mean(torch.abs(torch.cumsum(prob, dim=1) - torch.cumsum(q, dim=1)))
            else:
                loss = -(q * torch.log(prob.clamp_min(1e-8))).sum(dim=1).mean()
                if method == "A3_kl_rr_mae":
                    loss = loss + F.l1_loss(expected_rr(prob, bins), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            pred = expected_rr(model(vx), vbins).detach().cpu().numpy()
        val_mae = subject_aggregated_mae(vy_np, pred, val["subject_id"])
        hist.append({"phase": "A", "method": method, "epoch": epoch, "train_loss": float(np.mean(losses)), "val_subject_aggregated_MAE": val_mae})
        if val_mae < best_mae:
            best_mae = val_mae
            best = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= int(args.patience):
            break
    model.load_state_dict(best)
    return model, hist


def train_regressor(
    method: str,
    train_x: np.ndarray,
    train_base: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_base: np.ndarray,
    val_y: np.ndarray,
    val_subjects: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[nn.Module, List[Dict[str, Any]]]:
    device = torch.device(args.device)
    if method == "A4_hidden_only":
        model: nn.Module = HiddenLinear(train_x.shape[1])
    elif method == "A5_spec_linear_residual":
        model = LinearResidual(train_x.shape[1], float(args.max_delta_bpm))
    else:
        model = SmallResidualMLP(train_x.shape[1], float(args.max_delta_bpm))
    model = model.to(device)
    x = torch.as_tensor(train_x, device=device).float()
    base = torch.as_tensor(train_base, device=device).float()
    y = torch.as_tensor(train_y, device=device).float()
    vx = torch.as_tensor(val_x, device=device).float()
    vbase = torch.as_tensor(val_base, device=device).float()
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    best = copy.deepcopy(model.state_dict())
    best_mae = float("inf")
    wait = 0
    hist: List[Dict[str, Any]] = []
    gen = torch.Generator(device=device).manual_seed(int(args.seed))
    for epoch in range(1, int(args.max_epochs) + 1):
        perm = torch.randperm(x.size(0), generator=gen, device=device)
        losses = []
        model.train()
        for st in range(0, x.size(0), int(args.cached_feature_batch_size)):
            idx = perm[st : st + int(args.cached_feature_batch_size)]
            if method == "A4_hidden_only":
                pred = model(x[idx])
                resid_penalty = torch.zeros((), device=device)
            else:
                pred = model(base[idx], x[idx])
                resid_penalty = model.residual(x[idx]).abs().mean()
            loss = F.smooth_l1_loss(pred, y[idx]) + float(args.residual_l1_weight) * resid_penalty
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            if method == "A4_hidden_only":
                pred_np = model(vx).detach().cpu().numpy()
            else:
                pred_np = model(vbase, vx).detach().cpu().numpy()
        val_mae = subject_aggregated_mae(val_y, pred_np, val_subjects)
        hist.append({"phase": "A", "method": method, "epoch": epoch, "train_loss": float(np.mean(losses)), "val_subject_aggregated_MAE": val_mae})
        if val_mae < best_mae:
            best_mae = val_mae
            best = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= int(args.patience):
            break
    model.load_state_dict(best)
    return model, hist


def eval_phase_a_method(
    method: str,
    model: Optional[nn.Module],
    cache: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    device = torch.device(args.device)
    base = cache["soft_spectral_rr"].reshape(-1).astype(np.float32)
    residual = np.zeros_like(base)
    if method == "A0_soft_spectral":
        return base, residual
    with torch.no_grad():
        if method in {"A1_gaussian_kl", "A2_wasserstein", "A3_kl_rr_mae"}:
            assert model is not None
            prob = model(torch.as_tensor(cache["spectral_logits"], device=device).float())
            bins = torch.as_tensor(cache["frequency_bins_bpm"], device=device).float()
            pred = expected_rr(prob, bins).detach().cpu().numpy().astype(np.float32)
            return pred, pred - base
        x = cache["pooled_hidden"].astype(np.float32)
        if method == "A6_spec_hidden_conf_residual":
            x = np.concatenate([x, confidence_matrix(cache)], axis=1)
        assert model is not None
        xt = torch.as_tensor(x, device=device).float()
        if method == "A4_hidden_only":
            pred = model(xt).detach().cpu().numpy().astype(np.float32)
            return pred, pred - base
        pred = model(torch.as_tensor(base, device=device).float(), xt).detach().cpu().numpy().astype(np.float32)
        return pred, pred - base


def phase_a_methods(args: argparse.Namespace) -> List[str]:
    if bool(getattr(args, "corrective", False)):
        return list(CORRECTIVE_PHASE_A_METHODS)
    methods = list(args.methods or PHASE_A_METHODS)
    if args.smoke:
        needed = {"A0_soft_spectral", "A1_gaussian_kl", "A4_hidden_only", "A6_spec_hidden_conf_residual"}
        methods = [m for m in PHASE_A_METHODS if m in needed]
    return [m for m in methods if m in PHASE_A_METHODS]


def soft_rr_from_cached_logits(cache: Dict[str, np.ndarray], temperature: float) -> np.ndarray:
    logits = np.asarray(cache["spectral_logits"], dtype=np.float32)
    bins = np.asarray(cache["frequency_bins_bpm"], dtype=np.float32).reshape(1, -1)
    scaled = logits / max(float(temperature), 1e-6)
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    prob = np.exp(scaled)
    prob = prob / np.clip(prob.sum(axis=1, keepdims=True), 1e-8, None)
    return np.sum(prob * bins, axis=1).astype(np.float32)


def previous_decoder_from_cached_logits(cache: Dict[str, np.ndarray]) -> np.ndarray:
    logits = np.asarray(cache["spectral_logits"], dtype=np.float32)
    bins = np.asarray(cache["frequency_bins_bpm"], dtype=np.float32)
    return bins[np.argmax(logits, axis=1)].astype(np.float32)


def write_a0_audit(args: argparse.Namespace) -> None:
    rows: List[Dict[str, Any]] = []
    for subject in args.subjects:
        cache = ensure_cache(args, subject)
        for split in ("val", "test"):
            data = cache[split]
            preds = {
                "A0_temperature_0p1_cached_expected": soft_rr_from_cached_logits(data, 0.1),
                "A0_temperature_1p0_cached_expected": soft_rr_from_cached_logits(data, 1.0),
                "A0_previous_decoder_cached": previous_decoder_from_cached_logits(data),
            }
            if "predicted_stft" in data:
                with torch.no_grad():
                    stft = torch.as_tensor(data["predicted_stft"], dtype=torch.float32, device=args.device)
                    preds["A0_previous_decoder_helper"] = estimate_rr_from_predicted_stft(stft, br_fs=BR_FS).detach().cpu().numpy().astype(np.float32)
                    preds["A0_softspectral_module_0p1"] = SoftSpectralRR(BR_FS, temperature=0.1)(stft).detach().cpu().numpy().astype(np.float32)
            for decoder, pred in preds.items():
                metric = rr_metric_dict(data["rr_true"], pred)
                rows.append(
                    {
                        "subject": subject,
                        "seed": int(args.seed),
                        "split": split,
                        "decoder": decoder,
                        "temperature": 0.1 if "0p1" in decoder else (1.0 if "1p0" in decoder else np.nan),
                        "cache_frequency_grid_hash": stable_config_hash({"frequency_bins_bpm": data["frequency_bins_bpm"]}),
                        "cache_spectral_tensor_shape": str(tuple(np.asarray(data["spectral_logits"]).shape)),
                        **metric,
                    }
                )
    write_csv(Path(args.out_dir) / "phase_a_a0_audit.csv", pd.DataFrame(rows))


def run_phase_a(args: argparse.Namespace) -> None:
    per_window: List[Dict[str, Any]] = []
    histories: List[Dict[str, Any]] = []
    selected: Dict[str, Any] = {}
    if bool(getattr(args, "corrective", False)) or bool(getattr(args, "audit_a0", False)):
        write_a0_audit(args)
    for subject in args.subjects:
        cache = ensure_cache(args, subject)
        train, val, test = cache["train"], cache["val"], cache["test"]
        models: Dict[str, nn.Module | None] = {"A0_soft_spectral": None}
        for method in phase_a_methods(args):
            if method == "A0_soft_spectral":
                continue
            if method in SPECTRAL_A:
                models[method], hist = train_distribution_method(method, train, val, args)
            else:
                train_x = train["pooled_hidden"]
                val_x = val["pooled_hidden"]
                if method == "A6_spec_hidden_conf_residual":
                    train_x = np.concatenate([train_x, confidence_matrix(train)], axis=1)
                    val_x = np.concatenate([val_x, confidence_matrix(val)], axis=1)
                models[method], hist = train_regressor(
                    method,
                    train_x,
                    train["soft_spectral_rr"],
                    train["rr_true"],
                    val_x,
                    val["soft_spectral_rr"],
                    val["rr_true"],
                    val["subject_id"],
                    args,
                )
            histories.extend(hist)
            ckpt = Path(args.out_dir) / "phase_a_checkpoints" / f"seed_{int(args.seed):03d}" / subject / method / "best.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": models[method].state_dict(), "method": method, "args": vars(args)}, ckpt)
            selected[f"{subject}:{method}"] = {"checkpoint": str(ckpt), "trainable_parameter_count": trainable_parameter_count(models[method])}
        for split, data in (("val", val), ("test", test)):
            for method in phase_a_methods(args):
                pred, residual = eval_phase_a_method(method, models.get(method), data, args)
                for i in range(len(pred)):
                    per_window.append(
                        {
                            "subject": subject,
                            "seed": int(args.seed),
                            "split": split,
                            "method": method,
                            "window_index": int(data["window_index"][i]),
                            "rr_true": float(data["rr_true"][i]),
                            "rr_pred": float(pred[i]),
                            "soft_spectral_rr": float(data["soft_spectral_rr"][i]),
                            "residual_bpm": float(residual[i]),
                            "entropy": float(data["entropy"][i]),
                            "top1_top2_gap": float(data["top1_top2_gap"][i]),
                        }
                    )
    out = Path(args.out_dir)
    win = pd.DataFrame(per_window)
    write_csv(out / "phase_a_per_window.csv", win)
    subj = per_subject_metrics(win[win["split"] == "test"])
    write_csv(out / "phase_a_per_subject.csv", subj)
    write_csv(out / "phase_a_summary.csv", method_summary(subj))
    write_csv(out / "phase_a_training_history.csv", pd.DataFrame(histories))
    write_json(out / "phase_a_selected_checkpoints.json", selected)


def select_phase_a_candidates(args: argparse.Namespace) -> Tuple[str, str]:
    df = pd.read_csv(Path(args.out_dir) / "phase_a_per_window.csv")
    val = df[df["split"] == "val"]
    by = val.groupby("method").apply(lambda g: np.mean(np.abs(g["rr_pred"] - g["rr_true"]))).sort_values()
    spectral = next((m for m in by.index if m in SPECTRAL_A), "A0_soft_spectral")
    hybrid = next((m for m in by.index if m in HYBRID_A), spectral)
    return str(spectral), str(hybrid)


def train_gate(x: np.ndarray, labels: np.ndarray, args: argparse.Namespace) -> Tuple[LogisticGate, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    mean, std = standardize_fit(x)
    xs = standardize_apply(x, mean, std)
    device = torch.device(args.device)
    model = LogisticGate(xs.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    xt = torch.as_tensor(xs, device=device).float()
    yt = torch.as_tensor(labels, device=device).float()
    hist = []
    for epoch in range(1, int(args.max_epochs) + 1):
        pred = model(xt)
        loss = F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), yt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        hist.append({"phase": "B", "method": "B3_soft_gate", "epoch": epoch, "train_loss": float(loss.detach().cpu())})
        if epoch >= int(args.patience):
            break
    return model, mean, std, hist


def auroc_score(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    pos = score[y == 1]
    neg = score[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    return float((pos.reshape(-1, 1) > neg.reshape(1, -1)).mean() + 0.5 * (pos.reshape(-1, 1) == neg.reshape(1, -1)).mean())


def train_advantage_gate(
    x: np.ndarray,
    target_advantage: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[AdvantageRegressor, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    mean, std = standardize_fit(x)
    xs = standardize_apply(x, mean, std)
    device = torch.device(args.device)
    model = AdvantageRegressor(xs.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    xt = torch.as_tensor(xs, device=device).float()
    yt = torch.as_tensor(target_advantage, device=device).float()
    hist: List[Dict[str, Any]] = []
    best = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    wait = 0
    for epoch in range(1, int(args.max_epochs) + 1):
        pred = model(xt)
        loss = F.smooth_l1_loss(pred, yt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        value = float(loss.detach().cpu())
        hist.append({"phase": "B", "method": "B3_relative_advantage_gate", "epoch": epoch, "train_loss": value})
        if value < best_loss:
            best_loss = value
            best = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= int(args.patience):
            break
    model.load_state_dict(best)
    return model, mean, std, hist


def run_phase_b_corrective(args: argparse.Namespace) -> None:
    if not (Path(args.out_dir) / "phase_a_per_window.csv").exists():
        run_phase_a(args)
    a = pd.read_csv(Path(args.out_dir) / "phase_a_per_window.csv")
    if "A1_gaussian_kl" not in set(a["method"].astype(str)):
        run_phase_a(args)
        a = pd.read_csv(Path(args.out_dir) / "phase_a_per_window.csv")
    rows: List[Dict[str, Any]] = []
    diag_rows: List[Dict[str, Any]] = []
    detect_rows: List[Dict[str, Any]] = []
    histories: List[Dict[str, Any]] = []
    manifest: Dict[str, Any] = {
        "mode": "corrective_relative_advantage",
        "spectral_method": "A1_gaussian_kl",
        "hidden_method": "frozen_rr_direct",
        "target": "abs(y_hat_spec - y) - abs(y_hat_hidden - y)",
        "gate_training_split": "source validation cache only",
    }
    for subject in args.subjects:
        cache = ensure_cache(args, subject)
        val_cache, test_cache = cache["val"], cache["test"]
        val_spec = a[(a.subject == subject) & (a.split == "val") & (a.method == "A1_gaussian_kl")].sort_values("window_index")
        test_spec = a[(a.subject == subject) & (a.split == "test") & (a.method == "A1_gaussian_kl")].sort_values("window_index")
        if val_spec.empty or test_spec.empty:
            continue
        val_hidden = val_cache["rr_direct"].reshape(-1).astype(np.float32)
        test_hidden = test_cache["rr_direct"].reshape(-1).astype(np.float32)
        val_spec_pred = val_spec["rr_pred"].to_numpy(dtype=np.float32)
        test_spec_pred = test_spec["rr_pred"].to_numpy(dtype=np.float32)
        val_y = val_spec["rr_true"].to_numpy(dtype=np.float32)
        test_y = test_spec["rr_true"].to_numpy(dtype=np.float32)
        target_adv = np.abs(val_spec_pred - val_y) - np.abs(val_hidden - val_y)
        gate_model, mean, std, hist = train_advantage_gate(confidence_matrix(val_cache), target_adv, args)
        histories.extend(hist)
        with torch.no_grad():
            pred_adv = gate_model(
                torch.as_tensor(standardize_apply(confidence_matrix(test_cache), mean, std), device=args.device).float()
            ).detach().cpu().numpy()
        use_hidden = pred_adv > 0.0
        preds = {
            "B0_always_spec": test_spec_pred,
            "B1_always_hidden": test_hidden,
            "B3_relative_advantage_gate": np.where(use_hidden, test_hidden, test_spec_pred),
            "B4_oracle_gate": np.where(np.abs(test_hidden - test_y) < np.abs(test_spec_pred - test_y), test_hidden, test_spec_pred),
        }
        for method in CORRECTIVE_PHASE_B_METHODS:
            pred = preds[method]
            for i, yp in enumerate(pred):
                rows.append(
                    {
                        "subject": subject,
                        "seed": int(args.seed),
                        "split": "test",
                        "method": method,
                        "window_index": int(test_spec["window_index"].iloc[i]),
                        "rr_true": float(test_y[i]),
                        "rr_pred": float(yp),
                        "predicted_relative_advantage": float(pred_adv[i]) if method == "B3_relative_advantage_gate" else np.nan,
                        "gate": float(use_hidden[i]) if method == "B3_relative_advantage_gate" else float(method in {"B1_always_hidden", "B4_oracle_gate"}),
                        "deployable": method != "B4_oracle_gate",
                        "uses_target_labels": method == "B4_oracle_gate",
                    }
                )
        for feat in CONFIDENCE_FEATURES:
            values = np.asarray(val_cache[feat], dtype=float)
            corr = float(pd.Series(values).corr(pd.Series(target_adv), method="spearman")) if len(values) > 1 else float("nan")
            detect_rows.append({"subject": subject, "seed": int(args.seed), "feature": feat, "spearman_relative_advantage": corr, "AUROC_hidden_better": auroc_score((target_adv > 0).astype(int), values)})
        for method, pred in preds.items():
            err = np.abs(pred - test_y)
            diag_rows.append(
                {
                    "subject": subject,
                    "seed": int(args.seed),
                    "method": method,
                    "gate_activation_rate": float(np.mean(use_hidden)),
                    "oracle_regret": float(np.mean(err - np.abs(preds["B4_oracle_gate"] - test_y))),
                    "MAE_gate_hidden_windows": float(np.mean(err[use_hidden])) if np.any(use_hidden) else float("nan"),
                    "MAE_gate_spectral_windows": float(np.mean(err[~use_hidden])) if np.any(~use_hidden) else float("nan"),
                }
            )
        manifest[subject] = {
            "gate_mean": mean,
            "gate_std": std,
            "source_val_target_advantage_mean": float(np.mean(target_adv)),
            "source_val_hidden_better_rate": float(np.mean(target_adv > 0.0)),
        }
    out = Path(args.out_dir)
    win = pd.DataFrame(rows)
    write_csv(out / "phase_b_per_window.csv", win)
    subj = per_subject_metrics(win)
    write_csv(out / "phase_b_per_subject.csv", subj)
    write_csv(out / "phase_b_summary.csv", method_summary(subj))
    write_csv(out / "phase_b_confidence_diagnostics.csv", pd.DataFrame(diag_rows))
    write_csv(out / "phase_b_error_detection_metrics.csv", pd.DataFrame(detect_rows))
    write_csv(out / "phase_b_training_history.csv", pd.DataFrame(histories))
    write_json(out / "phase_b_gate_manifest.json", manifest)


def run_phase_b(args: argparse.Namespace) -> None:
    if bool(getattr(args, "corrective", False)):
        run_phase_b_corrective(args)
        return
    if not (Path(args.out_dir) / "phase_a_per_window.csv").exists():
        run_phase_a(args)
    spectral_method, hybrid_method = select_phase_a_candidates(args)
    a = pd.read_csv(Path(args.out_dir) / "phase_a_per_window.csv")
    rows: List[Dict[str, Any]] = []
    diag_rows: List[Dict[str, Any]] = []
    detect_rows: List[Dict[str, Any]] = []
    manifest: Dict[str, Any] = {"spectral_method": spectral_method, "hybrid_method": hybrid_method}
    histories: List[Dict[str, Any]] = []
    for subject in args.subjects:
        cache = ensure_cache(args, subject)
        val_cache, test_cache = cache["val"], cache["test"]
        val_spec = a[(a.subject == subject) & (a.split == "val") & (a.method == spectral_method)].sort_values("window_index")
        val_hyb = a[(a.subject == subject) & (a.split == "val") & (a.method == hybrid_method)].sort_values("window_index")
        test_spec = a[(a.subject == subject) & (a.split == "test") & (a.method == spectral_method)].sort_values("window_index")
        test_hyb = a[(a.subject == subject) & (a.split == "test") & (a.method == hybrid_method)].sort_values("window_index")
        if val_spec.empty or val_hyb.empty or test_spec.empty or test_hyb.empty:
            continue
        val_err_spec = np.abs(val_spec["rr_pred"].to_numpy() - val_spec["rr_true"].to_numpy())
        val_err_hyb = np.abs(val_hyb["rr_pred"].to_numpy() - val_hyb["rr_true"].to_numpy())
        labels = (val_err_spec > float(args.high_error_threshold_bpm)).astype(np.float32)
        x_val = confidence_matrix(val_cache)
        gate, mean, std, hist = train_gate(x_val, labels, args)
        histories.extend(hist)
        with torch.no_grad():
            gate_val = gate(torch.as_tensor(standardize_apply(x_val, mean, std), device=args.device).float()).detach().cpu().numpy()
            gate_test = gate(torch.as_tensor(standardize_apply(confidence_matrix(test_cache), mean, std), device=args.device).float()).detach().cpu().numpy()
        disagreement_val = np.asarray(val_cache["spectral_hidden_disagreement_bpm"], dtype=float)
        thresholds = np.unique(np.quantile(disagreement_val, [0.25, 0.5, 0.75]))
        best_thr, best_mae = 0.0, float("inf")
        for thr in thresholds:
            pred = np.where(disagreement_val > thr, val_hyb["rr_pred"].to_numpy(), val_spec["rr_pred"].to_numpy())
            mae = float(np.mean(np.abs(pred - val_spec["rr_true"].to_numpy())))
            if mae < best_mae:
                best_mae, best_thr = mae, float(thr)
        preds = {
            "B0_always_spec": test_spec["rr_pred"].to_numpy(),
            "B1_always_hybrid": test_hyb["rr_pred"].to_numpy(),
            "B2_disagreement_gate": np.where(test_cache["spectral_hidden_disagreement_bpm"] > best_thr, test_hyb["rr_pred"].to_numpy(), test_spec["rr_pred"].to_numpy()),
            "B3_soft_gate": (1.0 - gate_test) * test_spec["rr_pred"].to_numpy() + gate_test * test_hyb["rr_pred"].to_numpy(),
            "B4_oracle_gate": np.where(
                np.abs(test_hyb["rr_pred"].to_numpy() - test_spec["rr_true"].to_numpy())
                < np.abs(test_spec["rr_pred"].to_numpy() - test_spec["rr_true"].to_numpy()),
                test_hyb["rr_pred"].to_numpy(),
                test_spec["rr_pred"].to_numpy(),
            ),
        }
        for method in (["B3_soft_gate"] if args.smoke else PHASE_B_METHODS):
            pred = preds[method]
            for i, yp in enumerate(pred):
                rows.append(
                    {
                        "subject": subject,
                        "seed": int(args.seed),
                        "split": "test",
                        "method": method,
                        "window_index": int(test_spec["window_index"].iloc[i]),
                        "rr_true": float(test_spec["rr_true"].iloc[i]),
                        "rr_pred": float(yp),
                        "gate": float(gate_test[i]) if method == "B3_soft_gate" else float(method in {"B1_always_hybrid", "B4_oracle_gate"}),
                        "deployable": method != "B4_oracle_gate",
                        "uses_target_labels": method == "B4_oracle_gate",
                    }
                )
        for feat in CONFIDENCE_FEATURES:
            values = np.asarray(val_cache[feat], dtype=float)
            corr = float(pd.Series(values).corr(pd.Series(val_err_spec), method="spearman")) if len(values) > 1 else float("nan")
            detect_rows.append({"subject": subject, "seed": int(args.seed), "feature": feat, "spearman_abs_error": corr, "AUROC": auroc_score(labels, values)})
        for name, pred in preds.items():
            if name not in {"B3_soft_gate", "B4_oracle_gate"} and args.smoke:
                continue
            err = np.abs(pred - test_spec["rr_true"].to_numpy())
            diag_rows.append(
                {
                    "subject": subject,
                    "seed": int(args.seed),
                    "method": name,
                    "gate_activation_rate": float(np.mean(gate_test)),
                    "oracle_regret": float(np.mean(err - np.abs(preds["B4_oracle_gate"] - test_spec["rr_true"].to_numpy()))),
                    "MAE_high_confidence": float(np.mean(err[gate_test <= np.median(gate_test)])),
                    "MAE_low_confidence": float(np.mean(err[gate_test > np.median(gate_test)])) if np.any(gate_test > np.median(gate_test)) else float("nan"),
                }
            )
        manifest[subject] = {"threshold": best_thr, "gate_mean": mean, "gate_std": std}
    out = Path(args.out_dir)
    win = pd.DataFrame(rows)
    write_csv(out / "phase_b_per_window.csv", win)
    subj = per_subject_metrics(win)
    write_csv(out / "phase_b_per_subject.csv", subj)
    write_csv(out / "phase_b_summary.csv", method_summary(subj))
    write_csv(out / "phase_b_confidence_diagnostics.csv", pd.DataFrame(diag_rows))
    write_csv(out / "phase_b_error_detection_metrics.csv", pd.DataFrame(detect_rows))
    write_csv(out / "phase_b_training_history.csv", pd.DataFrame(histories))
    write_json(out / "phase_b_gate_manifest.json", manifest)


def select_phase_c_base(args: argparse.Namespace) -> Tuple[str, pd.DataFrame]:
    if bool(getattr(args, "corrective", False)):
        if not (Path(args.out_dir) / "phase_a_per_window.csv").exists():
            run_phase_a(args)
        df = pd.read_csv(Path(args.out_dir) / "phase_a_per_window.csv")
        fixed = df[(df["split"] == "test") & (df["method"] == "A1_gaussian_kl")].copy()
        if fixed.empty:
            run_phase_a(args)
            df = pd.read_csv(Path(args.out_dir) / "phase_a_per_window.csv")
            fixed = df[(df["split"] == "test") & (df["method"] == "A1_gaussian_kl")].copy()
        return "phase_a_fixed_A1_gaussian_kl", fixed
    if (Path(args.out_dir) / "phase_b_per_window.csv").exists():
        df = pd.read_csv(Path(args.out_dir) / "phase_b_per_window.csv")
        deploy = df[df.get("deployable", True).astype(bool)]
        if not deploy.empty:
            return "phase_b", deploy
    if not (Path(args.out_dir) / "phase_a_per_window.csv").exists():
        run_phase_a(args)
    df = pd.read_csv(Path(args.out_dir) / "phase_a_per_window.csv")
    test = df[df["split"] == "test"]
    return "phase_a", test


def affine_calibrate_sequence(
    base: np.ndarray,
    hidden_ref: np.ndarray,
    gate: np.ndarray,
    window_index: np.ndarray,
    method: str,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    device = torch.device(args.device)
    state = AffineState(raw_a=torch.tensor(0.0, device=device, requires_grad=True), raw_b=torch.tensor(0.0, device=device, requires_grad=True))
    opt = torch.optim.AdamW([state.raw_a, state.raw_b], lr=float(args.adaptation_learning_rate))
    order = np.argsort(window_index)
    pred = np.zeros_like(base, dtype=np.float32)
    traj: List[Dict[str, Any]] = []
    prev: Optional[torch.Tensor] = None
    lo, hi = map(float, args.a_range)
    for step, idx in enumerate(order):
        a, b = state.values((lo, hi), float(args.max_abs_bias_bpm))
        correction = a * torch.tensor(float(base[idx]), device=device) + b
        if method == "C4_confidence_gated_affine":
            calibrated = (1.0 - float(gate[idx])) * torch.tensor(float(base[idx]), device=device) + float(gate[idx]) * correction
        else:
            calibrated = correction
        pred[idx] = float(calibrated.detach().cpu())
        target_ref = torch.tensor(float(hidden_ref[idx]), device=device)
        loss = float(args.lambda_agree) * torch.abs(calibrated - target_ref)
        if prev is not None:
            loss = loss + float(args.lambda_smooth) * torch.abs(calibrated - prev.detach())
        loss = loss + float(args.anchor_weight) * ((a - 1.0).pow(2) + (b / max(float(args.max_abs_bias_bpm), 1e-6)).pow(2))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        prev = calibrated.detach()
        traj.append({"step": step, "window_index": int(window_index[idx]), "method": method, "a": float(a.detach().cpu()), "b": float(b.detach().cpu()), "loss": float(loss.detach().cpu())})
        if step + 1 >= int(args.max_adaptation_steps):
            for rest in order[step + 1 :]:
                a, b = state.values((lo, hi), float(args.max_abs_bias_bpm))
                correction = a * torch.tensor(float(base[rest]), device=device) + b
                calibrated = correction if method != "C4_confidence_gated_affine" else (1.0 - float(gate[rest])) * torch.tensor(float(base[rest]), device=device) + float(gate[rest]) * correction
                pred[rest] = float(calibrated.detach().cpu())
            break
    return pred, traj


def run_phase_c(args: argparse.Namespace) -> None:
    if args.smoke and not (Path(args.out_dir) / "phase_b_per_window.csv").exists():
        run_phase_b(args)
    base_stage, base_df = select_phase_c_base(args)
    rows: List[Dict[str, Any]] = []
    traj_rows: List[Dict[str, Any]] = []
    oracle_rows: List[Dict[str, Any]] = []
    for subject in args.subjects:
        cache = ensure_cache(args, subject)
        test = cache["test"]
        sub = base_df[base_df["subject"] == subject].copy()
        if sub.empty:
            continue
        if bool(getattr(args, "corrective", False)):
            sub = sub.sort_values("window_index")
            base_method = "A1_gaussian_kl"
        elif "method" in sub:
            best = sub.groupby("method").apply(lambda g: np.mean(np.abs(g["rr_pred"] - g["rr_true"]))).sort_values().index[0]
            sub = sub[sub["method"] == best].sort_values("window_index")
            base_method = str(best)
        else:
            sub = sub.sort_values("window_index")
            base_method = base_stage
        base = sub["rr_pred"].to_numpy(dtype=np.float32)
        y = sub["rr_true"].to_numpy(dtype=np.float32)
        window_index = sub["window_index"].to_numpy(dtype=np.int64)
        hidden_ref = test["rr_direct"].reshape(-1).astype(np.float32)
        gate = sub["gate"].to_numpy(dtype=np.float32) if "gate" in sub.columns else confidence_matrix(test)[:, 0]
        if bool(getattr(args, "corrective", False)) and (Path(args.out_dir) / "phase_b_per_window.csv").exists():
            b = pd.read_csv(Path(args.out_dir) / "phase_b_per_window.csv")
            g = b[(b["subject"] == subject) & (b["method"] == "B3_relative_advantage_gate")].sort_values("window_index")
            if len(g) == len(sub):
                gate = g["gate"].to_numpy(dtype=np.float32)
        methods = ["C3_affine_readout"] if args.smoke else (CORRECTIVE_PHASE_C_METHODS if bool(getattr(args, "corrective", False)) else PHASE_C_METHODS)
        preds: Dict[str, np.ndarray] = {"C0_none": base}
        preds["C1_feature_mean"] = base + float(args.feature_mean_alpha) * (np.mean(hidden_ref) - np.mean(base))
        preds["C2_temperature"] = base
        for method in ("C3_affine_readout", "C4_confidence_gated_affine"):
            preds[method], traj = affine_calibrate_sequence(base, hidden_ref, gate, window_index, method, args)
            for r in traj:
                r.update({"subject": subject, "seed": int(args.seed)})
            traj_rows.extend(traj)
        offset = float(np.mean(y - base))
        preds["C5_oracle_offset"] = base + offset
        A = np.column_stack([base, np.ones_like(base)])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        preds["C6_oracle_affine"] = coef[0] * base + coef[1]
        for method in methods:
            pred = preds[method]
            uses_labels = method in {"C5_oracle_offset", "C6_oracle_affine"}
            for i, yp in enumerate(pred):
                rows.append(
                    {
                        "subject": subject,
                        "seed": int(args.seed),
                        "method": method,
                        "window_index": int(window_index[i]),
                        "rr_true": float(y[i]),
                        "rr_pred_pre_adaptation": float(base[i]),
                        "rr_pred": float(yp),
                        "base_method": base_method,
                        "deployable": not uses_labels,
                        "uses_target_labels": uses_labels,
                    }
                )
        oracle_rows.append({"subject": subject, "seed": int(args.seed), "base_method": base_method, "oracle_offset_MAE": rr_metric_dict(y, preds["C5_oracle_offset"])["MAE"], "oracle_affine_MAE": rr_metric_dict(y, preds["C6_oracle_affine"])["MAE"], "base_MAE": rr_metric_dict(y, base)["MAE"]})
    out = Path(args.out_dir)
    win = pd.DataFrame(rows)
    write_csv(out / "phase_c_per_window.csv", win)
    subj = per_subject_metrics(win)
    write_csv(out / "phase_c_per_subject.csv", subj)
    write_csv(out / "phase_c_summary.csv", method_summary(subj))
    write_csv(out / "phase_c_parameter_trajectories.csv", pd.DataFrame(traj_rows))
    write_csv(out / "phase_c_oracle_headroom.csv", pd.DataFrame(oracle_rows))
    write_json(out / "phase_c_manifest.json", {**run_manifest(args), "base_stage": base_stage, "deployable_target_labels_used_for_fit": False})


def parse_methods(raw: Optional[Sequence[str]]) -> Optional[List[str]]:
    if raw is None:
        return None
    out: List[str] = []
    for item in raw:
        for chunk in str(item).replace(",", " ").split():
            if chunk:
                out.append(chunk)
    return out or None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RR readout Phase A/B/C experiment over frozen spectral representations.")
    parser.add_argument("--phase", default="all", choices=["cache", "A", "B", "C", "analysis", "all"])
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--source-subjects", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-dir", default=SBJ_PROCESSED_DIR)
    parser.add_argument("--data-str", default="imu_filt", choices=["imu_filt", "imu_ica"])
    parser.add_argument("--data-group", default="mr", choices=["mr", "levels", "mr_levels"])
    parser.add_argument("--mdl-dir", default=M_DIR)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--checkpoint-name", default="best_model.pt")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cached-feature-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--pin-memory", type=int, default=0)
    parser.add_argument("--persistent-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--corrective", action="store_true", help="Run compact corrective A/B/C method set and fixed-A1 Phase C base.")
    parser.add_argument("--audit-a0", action="store_true", help="Write A0 decoder audit from identical cached spectral tensors.")
    parser.add_argument("--val-split", type=float, default=0.25)
    parser.add_argument("--soft-temperature", type=float, default=0.1)
    parser.add_argument("--target-sigma-bpm", type=float, default=1.0)
    parser.add_argument("--max-delta-bpm", type=float, default=2.0)
    parser.add_argument("--residual-l1-weight", type=float, default=0.01)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--high-error-threshold-bpm", type=float, default=3.0)
    parser.add_argument("--feature-mean-alpha", type=float, default=0.75)
    parser.add_argument("--a-init", type=float, default=1.0)
    parser.add_argument("--b-init", type=float, default=0.0)
    parser.add_argument("--temperature-init", type=float, default=1.0)
    parser.add_argument("--max-abs-bias-bpm", type=float, default=2.0)
    parser.add_argument("--a-range", nargs=2, type=float, default=[0.9, 1.1])
    parser.add_argument("--max-adaptation-steps", type=int, default=10)
    parser.add_argument("--adaptation-learning-rate", type=float, default=1e-3)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--lambda-agree", type=float, default=1.0)
    parser.add_argument("--lambda-aug", type=float, default=0.0)
    parser.add_argument("--lambda-smooth", type=float, default=0.1)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    args = parser.parse_args(argv)
    args.methods = parse_methods(args.methods)
    apply_smoke_overrides(args)
    if args.source_subjects is None:
        args.source_subjects = list(DEFAULT_SUBJECTS) if args.smoke and len(args.subjects) == 1 else list(args.subjects)
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    set_all_seeds(int(args.seed))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run_manifest.json", run_manifest(args))
    write_json(out / "config.json", vars(args))
    phases = ["cache", "A", "B", "C"] if args.phase == "all" else [args.phase]
    if "cache" in phases:
        run_cache(args)
    if "A" in phases:
        run_phase_a(args)
    if "B" in phases:
        run_phase_b(args)
    if "C" in phases:
        run_phase_c(args)
    if args.phase in {"analysis", "all"}:
        from rr_readout_phase_abc_analysis import run_analysis

        run_analysis(Path(args.out_dir), bootstrap_resamples=int(args.bootstrap_resamples), seed=int(args.seed))


if __name__ == "__main__":
    main()
