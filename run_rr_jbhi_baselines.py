#!/usr/bin/env python3
"""LOSO neural baseline suite for IMU respiration-rate estimation.

Representative baselines for a JBHI-style table:
  raw IMU ResNet1D, CNN-GRU, TCN, InceptionTime-style, IMU STFT-CNN,
  PatchTST-style, and a compact cross-modal pressure-reconstruction model.

Uses your existing dataloader.py / config.py style: LOOCV subjects, pressure
signal from dataset as cross-modal target, and br as RR target.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW

from dataloader import build_loocv_loaders
from evaluations import simple_regression_metrics
from rr_jbhi_models import make_model, RRForward


DEFAULT_SUBJECTS = "S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29"
DEFAULT_MODELS = "resnet1d cnn_gru tcn inceptiontime stft_cnn patchtst crossmodal_rr"


def parse_list(text: str) -> List[str]:
    return [t.strip() for t in str(text).replace(",", " ").split() if t.strip()]


def subjects_with_data(subjects: List[str], data_dir: str, data_group: str) -> List[str]:
    group = str(data_group or "").strip().lower()
    bases = [Path(data_dir)]
    if group and group not in {"none", "all", "root"}:
        bases = [Path(data_dir) / group, Path(data_dir)]

    out = []
    for subject in subjects:
        if any((base / f"{subject}.pkl").exists() for base in bases):
            out.append(subject)
        else:
            print(f"[DATA] skipping missing subject {subject} under {[str(base) for base in bases]}", flush=True)
    return out


def close_loaders(*loaders) -> None:
    for loader in loaders:
        iterator = getattr(loader, "_iterator", None)
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()
        if hasattr(loader, "_iterator"):
            loader._iterator = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_fold_seed(base_seed: int, subject: str) -> int:
    return int(int(base_seed) * 1000 + int(str(subject).lstrip("S")))


def unpack_batch(batch, device: torch.device):
    # dataloader returns x, pressure, condition, br [, tlx]
    if isinstance(batch, (list, tuple)) and len(batch) >= 4:
        x = batch[0].float().to(device)
        pressure = batch[1].float().to(device)
        rr = batch[3].float().to(device).view(-1)
        cond = batch[2].detach().cpu().numpy() if batch[2] is not None else None
        return x, pressure, rr, cond
    raise ValueError("Expected dataloader batch with at least (x, pressure, cond, br).")


def ensure_channel_first(x: torch.Tensor) -> torch.Tensor:
    # LoadDataset usually returns (B,C,T); preserve fallback for (B,T,C).
    if x.ndim == 3 and x.shape[1] > x.shape[2]:
        x = x.permute(0, 2, 1).contiguous()
    return x


def pressure_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred is None:
        return torch.tensor(0.0, device=target.device)
    if target.ndim > 2:
        target = target.squeeze()
    n = min(pred.shape[-1], target.shape[-1]) if target.ndim >= 2 else min(pred.shape[-1], target.numel())
    return F.smooth_l1_loss(pred[..., :n], target[..., :n])


def train_one(model: nn.Module, train_loader, val_loader, args, device: torch.device) -> Dict[str, float]:
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_val = float("inf")
    bad = 0
    history = []
    model.to(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            x, pressure, rr, _cond = unpack_batch(batch, device)
            x = ensure_channel_first(x)
            out: RRForward = model(x)
            rr_loss = F.smooth_l1_loss(out.pred, rr)
            rec_loss = pressure_loss(out.recon_pressure, pressure)
            loss = rr_loss + float(args.recon_weight) * rec_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        val = evaluate_loader(model, val_loader, device, alpha_mean=None)
        val_mae = float(val["mae"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_mae": val_mae})
        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_mae": best_val, "epochs_ran": len(history), "history": history}


@torch.no_grad()
def collect_outputs(model: nn.Module, loader, device: torch.device, alpha_mean=None) -> Dict[str, np.ndarray]:
    model.eval()
    preds, true, embs, conds = [], [], [], []
    for batch in loader:
        x, pressure, rr, cond = unpack_batch(batch, device)
        x = ensure_channel_first(x)
        out: RRForward = model(x)
        z = out.emb
        pred = out.pred
        if alpha_mean is not None:
            # NOTE: fixed feature-mean adaptation evaluated at embedding/head level.
            # This assumes the model has an MLPHead with `head(z)`. It is used only
            # for analysis/adaptation diagnostics, not during baseline training.
            z = z + float(alpha_mean["alpha"]) * (alpha_mean["source_mean"] - alpha_mean["target_mean"])
            pred = model.head(z)
        preds.append(pred.detach().cpu().numpy())
        true.append(rr.detach().cpu().numpy())
        embs.append(z.detach().cpu().numpy())
        if cond is not None:
            conds.append(cond)
    out = {
        "pred": np.concatenate(preds).reshape(-1),
        "true": np.concatenate(true).reshape(-1),
        "emb": np.concatenate(embs, axis=0),
    }
    if conds:
        out["cond"] = np.concatenate(conds).reshape(-1)
    return out


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader, device: torch.device, alpha_mean=None) -> Dict[str, float]:
    out = collect_outputs(model, loader, device, alpha_mean=alpha_mean)
    m = simple_regression_metrics(out["true"], out["pred"])
    return {"mae": float(m["mae"]), "rmse": float(m["rmse"]), "corr": float(m["corr"])}


def save_predictions(path: Path, subject: str, mode: str, out: Dict[str, np.ndarray]):
    df = pd.DataFrame({"subject": subject, "mode": mode, "rr_true": out["true"], "rr_pred": out["pred"]})
    if "cond" in out:
        df["cond"] = out["cond"]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_model(model_name: str, args, device: torch.device) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subjects = subjects_with_data(parse_list(args.subjects), args.data_dir, args.data_group)
    rows, pred_rows = [], []
    model_root = Path(args.out_dir) / model_name
    model_root.mkdir(parents=True, exist_ok=True)
    set_seed(int(args.seed))

    for subject in subjects:
        fold_seed = make_fold_seed(int(args.seed), subject)
        set_seed(fold_seed)
        train_loader, val_loader, test_loader = build_loocv_loaders(
            subject,
            subjects,
            args.data_str,
            val_split=args.val_split,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            data_dir=args.data_dir,
            mdl_dir=args.mdl_dir,
            data_group=args.data_group,
            seed=fold_seed,
            num_workers=int(args.num_workers),
            prefetch_factor=int(args.prefetch_factor),
            pin_memory=bool(args.pin_memory),
            persistent_workers=bool(args.persistent_workers),
        )
        print(f"\n[MODEL={model_name}] [LOSO={subject}]", flush=True)
        sample_x, *_ = next(iter(train_loader))
        x0 = ensure_channel_first(sample_x.float())
        in_ch = int(x0.shape[1])
        model = make_model(model_name, in_ch=in_ch, emb_dim=args.emb_dim)
        train_info = train_one(model, train_loader, val_loader, args, device)

        base_out = collect_outputs(model, test_loader, device)
        base_metrics = simple_regression_metrics(base_out["true"], base_out["pred"])
        fold_dir = model_root / subject
        fold_dir.mkdir(parents=True, exist_ok=True)
        save_predictions(fold_dir / f"predictions_{model_name}_{subject}_none.csv", subject, "none", base_out)
        np.savez_compressed(fold_dir / f"embeddings_{model_name}_{subject}_none.npz",
                            subject=subject, mode="none", emb=base_out["emb"], rr_true=base_out["true"], rr_pred=base_out["pred"])

        modes = [("none", base_out, base_metrics)]
        if args.eval_alpha075:
            train_out = collect_outputs(model, train_loader, device)
            source_mean = torch.from_numpy(train_out["emb"].mean(axis=0)).float().to(device).view(1, -1)
            target_mean = torch.from_numpy(base_out["emb"].mean(axis=0)).float().to(device).view(1, -1)
            alpha_cfg = {"alpha": 0.75, "source_mean": source_mean, "target_mean": target_mean}
            adapt_out = collect_outputs(model, test_loader, device, alpha_mean=alpha_cfg)
            adapt_metrics = simple_regression_metrics(adapt_out["true"], adapt_out["pred"])
            save_predictions(fold_dir / f"predictions_{model_name}_{subject}_alpha075.csv", subject, "alpha075", adapt_out)
            np.savez_compressed(fold_dir / f"embeddings_{model_name}_{subject}_alpha075.npz",
                                subject=subject, mode="alpha075", emb=adapt_out["emb"], rr_true=adapt_out["true"], rr_pred=adapt_out["pred"])
            modes.append(("alpha075", adapt_out, adapt_metrics))

        for mode, out, met in modes:
            rows.append({
                "model": model_name,
                "subject": subject,
                "mode": mode,
                "seed": int(args.seed),
                "base_seed": int(args.seed),
                "fold_seed": int(fold_seed),
                "mae": float(met["mae"]),
                "rmse": float(met["rmse"]),
                "corr": float(met["corr"]),
                "n": int(out["true"].shape[0]),
                "best_val_mae": float(train_info["best_val_mae"]),
                "epochs_ran": int(train_info["epochs_ran"]),
            })

        torch.save(model.state_dict(), fold_dir / f"model_{model_name}_{subject}.pt")
        with open(fold_dir / f"train_history_{model_name}_{subject}.json", "w") as f:
            json.dump(train_info["history"], f, indent=2)
        close_loaders(train_loader, val_loader, test_loader)
        del train_loader, val_loader, test_loader, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    subject_df = pd.DataFrame(rows)
    summary_df = subject_df.groupby(["model", "mode"], as_index=False).agg(
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        corr_mean=("corr", "mean"), corr_std=("corr", "std"),
        n_subjects=("subject", "nunique"),
    )
    summary_df["seed"] = int(args.seed)
    subject_df.to_csv(model_root / "subject_rows.csv", index=False)
    summary_df.to_csv(model_root / "summary.csv", index=False)
    return subject_df, summary_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=DEFAULT_MODELS)
    p.add_argument("--subjects", default=DEFAULT_SUBJECTS)
    p.add_argument("--data-str", default="imu_filt")
    p.add_argument("--data-dir", default="/projects/BLVMob/imu-rr-seated/Data")
    p.add_argument("--data-group", default="mr")
    p.add_argument("--mdl-dir", default="/projects/BLVMob/imu-rr-seated/models/imu_filt/loocv")
    p.add_argument("--out-dir", default="/projects/BLVMob/imu-rr-seated/results/jbhi_rr_baselines")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--val-split", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--recon-weight", type=float, default=0.15)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-alpha075", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num-workers", type=int, default=int(os.environ.get("IMU_DATALOADER_WORKERS", "0")))
    p.add_argument("--prefetch-factor", type=int, default=int(os.environ.get("IMU_DATALOADER_PREFETCH", "1")))
    p.add_argument("--pin-memory", type=int, default=int(os.environ.get("IMU_DATALOADER_PIN_MEMORY", "0")))
    p.add_argument("--persistent-workers", type=int, default=int(os.environ.get("IMU_DATALOADER_PERSISTENT_WORKERS", "0")))
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

    all_subject, all_summary = [], []
    for model_name in parse_list(args.models):
        subject_df, summary_df = run_model(model_name, args, device)
        all_subject.append(subject_df)
        all_summary.append(summary_df)
    pd.concat(all_subject, ignore_index=True).to_csv(Path(args.out_dir) / "subject_rows.csv", index=False)
    pd.concat(all_summary, ignore_index=True).sort_values("mae_mean").to_csv(Path(args.out_dir) / "summary.csv", index=False)
    print(f"[DONE] wrote {args.out_dir}/summary.csv")


if __name__ == "__main__":
    main()
