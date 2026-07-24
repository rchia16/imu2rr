#!/usr/bin/env python3
"""LOSO rest/low/high MWL classifiers using existing direct-IMU architectures."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from mwl_rest_low_high import (
    REST_LOW_HIGH_CLASS_NAMES,
    assert_no_subject_overlap,
    data_groups_for_rest_low_high,
    mapping_manifest,
    parse_list,
    split_source_subjects,
    stack_subject_windows,
)
from rr_jbhi_models import RRForward, make_model


DEFAULT_SUBJECTS = "S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29"
DEFAULT_MODELS = "resnet1d cnn_gru tcn inceptiontime stft_cnn"


@dataclass
class FoldMeta:
    model: str
    subject: str
    seed: int
    fold_seed: int
    train_subjects: List[str]
    val_subjects: List[str]
    test_subject: str
    train_windows: int
    val_windows: int
    test_windows: int
    trainable_parameters: int
    best_epoch: int
    best_val_macro_f1: float
    best_val_balanced_accuracy: float
    epochs_ran: int


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.y.numel())

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class RRBackboneClassifier(nn.Module):
    """Classification wrapper around rr_jbhi_models.make_model encoders."""

    def __init__(self, model_name: str, in_ch: int, emb_dim: int, num_classes: int = 3) -> None:
        super().__init__()
        self.backbone = make_model(model_name, in_ch=in_ch, emb_dim=emb_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(int(emb_dim)),
            nn.Linear(int(emb_dim), int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: RRForward = self.backbone(x.float())
        return self.classifier(out.emb)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(False)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.benchmark = False


def make_fold_seed(seed: int, subject: str) -> int:
    return int(int(seed) * 1000 + int(str(subject).lstrip("S")))


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def ensure_channel_first(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected IMU windows shaped (N,T,C) or (N,C,T), got {arr.shape}.")
    if arr.shape[1] > arr.shape[2]:
        arr = np.transpose(arr, (0, 2, 1))
    return np.ascontiguousarray(arr, dtype=np.float32)


def fit_source_normalizer(x_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_cf = ensure_channel_first(x_train)
    mean = x_cf.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = x_cf.std(axis=(0, 2), keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def apply_normalizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x_cf = ensure_channel_first(x)
    return np.ascontiguousarray((x_cf - mean) / std, dtype=np.float32)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int, num_workers: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        WindowDataset(x, y),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        drop_last=False,
        num_workers=int(num_workers),
        worker_init_fn=worker_init_fn if int(num_workers) > 0 else None,
        generator=generator,
    )


def class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / max(float(weights.mean()), 1e-6)
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, preds = [], []
    for xb, _yb in loader:
        logits = model(xb.to(device))
        p = torch.softmax(logits, dim=1)
        probs.append(p.detach().cpu().numpy())
        preds.append(p.argmax(dim=1).detach().cpu().numpy())
    return np.concatenate(preds, axis=0).astype(int), np.concatenate(probs, axis=0).astype(np.float32)


@torch.no_grad()
def score_loader(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> Dict[str, float]:
    model.eval()
    labels, preds, losses = [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        losses.append(float(criterion(logits, yb).detach().cpu()))
        preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        labels.append(yb.detach().cpu().numpy())
    y = np.concatenate(labels, axis=0) if labels else np.zeros((0,), dtype=int)
    pred = np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=int)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": float(accuracy_score(y, pred)) if y.size else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if y.size else float("nan"),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)) if y.size else float("nan"),
    }


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_train: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    criterion = nn.CrossEntropyLoss(weight=class_weights(y_train).to(device))
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_score = -float("inf")
    best_epoch = 0
    bad = 0
    history: List[Dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses, labels, preds = [], [], []
        for batch_i, (xb, yb) in enumerate(train_loader):
            if int(args.max_train_batches) > 0 and batch_i >= int(args.max_train_batches):
                break
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            preds.append(logits.argmax(dim=1).detach().cpu().numpy())
            labels.append(yb.detach().cpu().numpy())
        yy = np.concatenate(labels, axis=0) if labels else np.zeros((0,), dtype=int)
        pp = np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=int)
        val = score_loader(model, val_loader, device, criterion)
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "train_macro_f1": float(f1_score(yy, pp, average="macro", zero_division=0)) if yy.size else float("nan"),
            "val_loss": float(val["loss"]),
            "val_macro_f1": float(val["macro_f1"]),
            "val_balanced_accuracy": float(val["balanced_accuracy"]),
        }
        history.append(row)
        current = float(row["val_macro_f1"])
        if np.isfinite(current) and current > best_score:
            best_score = current
            best_epoch = int(epoch)
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bool(args.early_stop) and bad >= int(args.patience):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    best_val = score_loader(model, val_loader, device, criterion)
    return {
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val["macro_f1"]),
        "best_val_balanced_accuracy": float(best_val["balanced_accuracy"]),
        "epochs_ran": int(len(history)),
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def config_hash(payload: Dict[str, object]) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def run_fold(model_name: str, subject: str, subjects: Sequence[str], args: argparse.Namespace, device: torch.device) -> Tuple[pd.DataFrame, FoldMeta]:
    fold_seed = make_fold_seed(int(args.seed), subject)
    set_seed(fold_seed)
    train_subjects, val_subjects = split_source_subjects(subjects, subject, int(args.val_subjects), fold_seed)
    assert_no_subject_overlap(train_subjects, val_subjects, subject)
    groups = data_groups_for_rest_low_high(args.data_group)
    train = stack_subject_windows(train_subjects, data_dir=args.data_dir, data_str=args.data_str, data_groups=groups)
    val = stack_subject_windows(val_subjects, data_dir=args.data_dir, data_str=args.data_str, data_groups=groups)
    test = stack_subject_windows([subject], data_dir=args.data_dir, data_str=args.data_str, data_groups=groups)
    if set(test.subject_id.astype(str)) & set(train.subject_id.astype(str)):
        raise AssertionError("Held-out target appeared in source training windows.")
    mean, std = fit_source_normalizer(train.x)
    x_train = apply_normalizer(train.x, mean, std)
    x_val = apply_normalizer(val.x, mean, std)
    x_test = apply_normalizer(test.x, mean, std)
    in_ch = int(x_train.shape[1])
    model = RRBackboneClassifier(model_name, in_ch=in_ch, emb_dim=int(args.emb_dim), num_classes=3)
    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    train_loader = make_loader(x_train, train.y, int(args.batch_size), True, fold_seed, int(args.num_workers))
    val_loader = make_loader(x_val, val.y, int(args.batch_size), False, fold_seed, int(args.num_workers))
    test_loader = make_loader(x_test, test.y, int(args.batch_size), False, fold_seed, int(args.num_workers))
    train_info = train_classifier(model, train_loader, val_loader, train.y, args, device)
    pred, prob = predict_loader(model, test_loader, device)
    out_dir = Path(args.out_dir) / model_name / subject
    out_dir.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "classifier_wrapper": "RRBackboneClassifier",
            "source_architecture": "rr_jbhi_models.make_model",
            "normalizer": {"mean": mean.reshape(-1).tolist(), "std": std.reshape(-1).tolist()},
            "fold_meta": {
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "test_subject": subject,
                "seed": int(args.seed),
                "fold_seed": int(fold_seed),
            },
        },
        out_dir / f"model_{model_name}_{subject}.pt",
    )
    pd.DataFrame(train_info["history"]).to_csv(out_dir / f"train_history_{model_name}_{subject}.csv", index=False)
    rows = pd.DataFrame(
        {
            "subject": str(subject),
            "seed": int(args.seed),
            "sample_id": test.sample_id.astype(str),
            "true_class": test.y.astype(int),
            "predicted_class": pred.astype(int),
            "prob_rest": prob[:, 0],
            "prob_low": prob[:, 1],
            "prob_high": prob[:, 2],
            "model_name": model_name,
            "checkpoint_or_run_path": str(out_dir),
            "raw_condition": test.raw_condition.astype(str),
            "group": test.group.astype(str),
            "raw_index": test.raw_index.astype(int),
            "source_train_subjects": " ".join(train_subjects),
            "source_val_subjects": " ".join(val_subjects),
        }
    )
    rows.to_csv(out_dir / "mwl_predictions.csv", index=False)
    meta = FoldMeta(
        model=model_name,
        subject=str(subject),
        seed=int(args.seed),
        fold_seed=int(fold_seed),
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subject=str(subject),
        train_windows=int(train.y.size),
        val_windows=int(val.y.size),
        test_windows=int(test.y.size),
        trainable_parameters=n_params,
        best_epoch=int(train_info["best_epoch"]),
        best_val_macro_f1=float(train_info["best_val_macro_f1"]),
        best_val_balanced_accuracy=float(train_info["best_val_balanced_accuracy"]),
        epochs_ran=int(train_info["epochs_ran"]),
    )
    (out_dir / "fold_manifest.json").write_text(json.dumps({**asdict(meta), "mapping": mapping_manifest()}, indent=2))
    del model, train_loader, val_loader, test_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, meta


def run_model(model_name: str, args: argparse.Namespace, device: torch.device) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subjects = list(getattr(args, "full_subjects", parse_list(args.subjects)))
    eval_subjects = parse_list(args.subjects)
    all_rows: List[pd.DataFrame] = []
    fold_meta: List[Dict[str, object]] = []
    model_root = Path(args.out_dir) / model_name
    model_root.mkdir(parents=True, exist_ok=False)
    for subject in eval_subjects:
        print(f"[MWL_DIRECT] model={model_name} held_out={subject}", flush=True)
        rows, meta = run_fold(model_name, subject, subjects, args, device)
        all_rows.append(rows)
        fold_meta.append(asdict(meta))
    pred_df = pd.concat(all_rows, ignore_index=True)
    pred_df.to_csv(model_root / "mwl_predictions.csv", index=False)
    meta_df = pd.DataFrame(fold_meta)
    meta_df.to_csv(model_root / "fold_rows.csv", index=False)
    return pred_df, meta_df


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct-IMU LOSO rest/low/high MWL benchmarks.")
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--subjects", default=DEFAULT_SUBJECTS)
    parser.add_argument("--data-str", default="imu_filt")
    parser.add_argument("--data-dir", default="/projects/BLVMob/imu-rr-seated/Data")
    parser.add_argument("--data-group", default="mr_levels", help="Use mr_levels to combine R from mr and L0-L3 from levels.")
    parser.add_argument("--out-dir", default="results/mwl_direct_imu_rest_low_high")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-subjects", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--early-stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-batches", type=int, default=0, help="Smoke-test limiter; 0 means all batches.")
    parser.add_argument("--eval-subjects", nargs="+", default=None, help="Optional held-out subjects to run from --subjects.")
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("IMU_DATALOADER_WORKERS", "0")))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=False)
    if args.eval_subjects:
        full_subjects = parse_list(args.subjects)
        eval_subjects = parse_list(args.eval_subjects)
        missing = sorted(set(eval_subjects) - set(full_subjects))
        if missing:
            raise SystemExit(f"--eval-subjects contains subjects not present in --subjects: {missing}")
        args.subjects = " ".join(eval_subjects)
        args.full_subjects = full_subjects
    else:
        args.full_subjects = parse_list(args.subjects)
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in str(args.device) else "cpu")
    manifest_config = {
        "args": vars(args),
        "mapping": mapping_manifest(),
        "source_model_file": "rr_jbhi_models.py",
        "source_runner_file": "run_rr_jbhi_baselines.py",
        "architecture_reuse": "Existing rr_jbhi_models encoders are instantiated unchanged via make_model; only a 3-class classifier wrapper/head is added.",
        "git_commit": git_commit(),
        "hostname": platform.node(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    manifest_config["config_hash"] = config_hash(manifest_config)
    all_pred, all_meta = [], []
    for model_name in parse_list(args.models):
        pred_df, meta_df = run_model(model_name, args, device)
        all_pred.append(pred_df)
        all_meta.append(meta_df)
    pd.concat(all_pred, ignore_index=True).to_csv(root / "mwl_predictions.csv", index=False)
    pd.concat(all_meta, ignore_index=True).to_csv(root / "fold_rows.csv", index=False)
    (root / "mwl_direct_imu_manifest.json").write_text(json.dumps(manifest_config, indent=2, default=str))
    print(f"[DONE] wrote {root}", flush=True)


if __name__ == "__main__":
    main()
