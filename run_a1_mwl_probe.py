#!/usr/bin/env python3
"""Train rest/low/high MWL probes on frozen cached F0+A1 tensors."""
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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mwl_rest_low_high import REST_LOW_HIGH_CLASS_NAMES, mapping_manifest, parse_list


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


@dataclass(frozen=True)
class CacheSplit:
    x: np.ndarray
    y: np.ndarray
    sample_id: np.ndarray
    raw_condition: np.ndarray
    path: Path
    key: str


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
    return CacheSplit(x=x, y=y, sample_id=sample_id, raw_condition=raw_condition, path=path, key=key)


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
        "source_cache_files": {
            "train": str(train.path),
            "val": str(val.path),
            "test": str(test.path),
        },
        "source_cache_sha256": {
            "train": sha256_file(train.path),
            "val": sha256_file(val.path),
            "test": sha256_file(test.path),
        },
        "tensor_keys": {"train": train.key, "val": val.key, "test": test.key},
        "trainable_scope": "downstream classifier only; cached F0+A1 tensors are read-only.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return rows, {**manifest, "status": "ok", "n_test": int(test.y.size)}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen F0+A1 rest/low/high MWL probe runner.")
    parser.add_argument("--feature-cache-root", required=True, help="Cache root with seed_000/Sxx/{train,val,test}.npz files.")
    parser.add_argument("--out-dir", default="results/a1_mwl_probe/rest_low_high")
    parser.add_argument("--subjects", default="S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29")
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
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=False)
    subjects = parse_list(args.subjects)
    seeds = [int(v) for v in parse_list(args.seeds)]
    variants = parse_list(args.variants)
    bad = sorted(set(variants) - set(A1_VARIANTS))
    if bad:
        raise SystemExit(f"Unsupported A1 variants: {bad}. Valid={sorted(A1_VARIANTS)}")
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in str(args.device) else "cpu")
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
    print(f"[DONE] wrote {root}", flush=True)


if __name__ == "__main__":
    main()
