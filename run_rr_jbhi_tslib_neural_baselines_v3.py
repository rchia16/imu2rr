#!/usr/bin/env python3
"""Run source-faithful neural RR baselines under LOSO.

This runner deliberately excludes the main cross-modal method. The main method
is executed only through the full `vit_pressure_crossmodal_stft_rr_*` project
implementation and adaptation ladders.

No silent architectural fallbacks are used: PatchTST and TimesNet require their
upstream/project source modules to be importable. If missing, the run fails with
a clear setup error for that model.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ipdb
import yaml
import pywt

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW

from dataloader import loocv_generator
from rr_jbhi_tslib_source_models_v2 import (
    SOURCE_BASELINES_TSLIB as SOURCE_BASELINES, RRForward, make_model
)

DEFAULT_SUBJECTS = "S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29"
DEFAULT_MODELS = "resnet1d cnn_gru tcn inceptiontime patchtst_tslib timesnet_tslib"
IMU_DOWNSAMPLE_HZ = 30


def parse_list(text: str) -> List[str]:
    return [t.strip() for t in str(text).replace(",", " ").split() if t.strip()]


def subjects_with_data(subjects: List[str], data_dir: str, data_group: str) -> List[str]:
    from os.path import exists, join
    group = str(data_group or "").strip().lower()
    bases = [data_dir]
    if group and group not in {"none", "all", "root"}:
        bases = [join(data_dir, group), data_dir]
    out = []
    for s in subjects:
        if any(exists(join(b, f"{s}.pkl")) for b in bases):
            out.append(s)
        else:
            print(f"[DATA] skipping missing subject {s} under {bases}")
    return out

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_channel_first(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected IMU tensor [B,C,T] or [B,T,C], got {tuple(x.shape)}")
    # Project LoadDataset usually returns [B,C,T]. If second dimension is longer
    # than third, treat it as [B,T,C] and transpose.
    if x.shape[1] > x.shape[2]:
        x = x.transpose(1, 2).contiguous()
    return x.float()


def unpack_batch(batch, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray | None]:
    # Existing dataloader returns at least: imu, pressure, condition, br [, tlx]
    if not isinstance(batch, (list, tuple)) or len(batch) < 4:
        raise ValueError("Expected dataloader batch with at least (imu, pressure, cond, br)")
    x = ensure_channel_first(batch[0].to(device))
    rr = batch[3].float().to(device).view(-1)
    cond = None
    try:
        cond = batch[2].detach().cpu().numpy().reshape(-1)
    except Exception:
        cond = None
    return x, rr, cond


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    n = min(y_true.size, y_pred.size)
    y_true, y_pred = y_true[:n], y_pred[:n]
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[keep], y_pred[keep]
    if y_true.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "corr": float("nan"), "n": 0}
    err = y_pred - y_true
    corr = float("nan")
    if y_true.size >= 2 and np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "corr": corr,
        "n": int(y_true.size),
    }


def train_one(model: nn.Module, train_loader, val_loader, args, device: torch.device) -> Dict[str, object]:
    model.to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_val = float("inf")
    bad = 0
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses = []
        for batch in train_loader:
            x, rr, _ = unpack_batch(batch, device)
            out = model(x) # RRForward
            loss = F.smooth_l1_loss(out.pred, rr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val = evaluate(model, val_loader, device)
        val_mae = float(val["mae"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_mae": val_mae})
        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= int(args.patience):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_mae": best_val, "epochs_ran": len(history), "history": history}


@torch.no_grad()
def collect(model: nn.Module, loader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    preds, true, embs, conds = [], [], [], []
    for batch in loader:
        x, rr, cond = unpack_batch(batch, device)
        out: RRForward = model(x)
        preds.append(out.pred.detach().cpu().numpy().reshape(-1))
        true.append(rr.detach().cpu().numpy().reshape(-1))
        embs.append(out.emb.detach().cpu().numpy())
        if cond is not None:
            conds.append(cond)
    out = {"pred": np.concatenate(preds), "true": np.concatenate(true), "emb": np.concatenate(embs, axis=0)}
    if conds:
        out["cond"] = np.concatenate(conds).reshape(-1)
    return out


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> Dict[str, float]:
    out = collect(model, loader, device)
    return regression_metrics(out["true"], out["pred"])


def save_outputs(root: Path, subject: str, model_name: str, out: Dict[str, np.ndarray]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"subject": subject, "model": model_name, "mode": "source", "rr_true": out["true"], "rr_pred": out["pred"]})
    if "cond" in out:
        df["cond"] = out["cond"]
    df.to_csv(root / "predictions.csv", index=False)
    np.savez_compressed(root / "embeddings.npz", emb=out["emb"], rr_true=out["true"], rr_pred=out["pred"], subject=subject, mode=model_name)



def _load_yaml_or_json(path: Path) -> Dict[str, object]:
    try:
        text = path.read_text()
        if yaml is not None:
            obj = yaml.safe_load(text)
        else:
            obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _safe_float(x, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        val = float(x)
        return val if np.isfinite(val) else default
    except Exception:
        return default


def _tokenize_run_text(text: str) -> set[str]:
    return {tok for tok in str(text or "").lower().replace("-", "_").split("_") if tok}


def _run_method_matches(actual: str, requested: str, run_dir: Optional[Path] = None) -> bool:
    actual = str(actual or "").strip().lower()
    requested = str(requested or "").strip().lower()
    run_text = "" if run_dir is None else str(run_dir.name).lower()
    run_tokens = _tokenize_run_text(run_text)

    if requested in {"", "any", "*"}:
        return True

    if requested == "style":
        # Strict TimesNet import mode used for the paper package:
        # only import the plain raw-style TimesNet baseline, e.g.
        #   20260212-163747_S12_timesnet_raw_style_seed0
        # Exclude adaptation variants even if they also contain style, e.g.
        #   *_flow_style_*
        #   *_flow_ssa_style_*
        #   *_flow_cmt_style_*
        #   *_flow_ssa_cmt_style_*
        banned = {"flow", "ssa", "cmt"}
        return ("style" in run_tokens) and not bool(run_tokens & banned)

    if actual == requested:
        return True
    return False


def _read_timesnet_br_pickle(run_dir: Path) -> Optional[Dict[str, float]]:
    br_path = run_dir / "br.pkl"
    if not br_path.exists():
        return None
    try:
        with br_path.open("rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, tuple) and payload:
            overall = payload[0]
        else:
            overall = payload
        if not isinstance(overall, dict):
            return None
        return {
            "mae": _safe_float(overall.get("mae")),
            "rmse": _safe_float(overall.get("rmse")),
            "corr": _safe_float(overall.get("pearsonr_coeff")),
        }
    except Exception:
        return None


def _read_timesnet_metrics_csv(run_dir: Path) -> Optional[Dict[str, float]]:
    metrics_path = run_dir / "metrics_subject.csv"
    if not metrics_path.exists():
        return None
    try:
        df = pd.read_csv(metrics_path)
        if df.empty:
            return None
        row = df.iloc[-1]
        return {
            "mae": _safe_float(row.get("rr_mae")),
            "rmse": _safe_float(row.get("rr_rmse")),
            "corr": _safe_float(row.get("rr_pearson")),
        }
    except Exception:
        return None


def run_one_model(
    model_name: str, args, device: torch.device
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subjects = subjects_with_data(
        parse_list(args.subjects), args.data_dir, args.data_group
    )
    rows = []
    history_rows = []
    model_root = Path(args.out_dir) / model_name
    model_root.mkdir(parents=True, exist_ok=True)

    input_downsample_hz = None
    if args.timesnet_mode == 'downsample' and 'timesnet' in model_name:
        print("model name: ", model_name, end=' ')
        input_downsample_hz = IMU_DOWNSAMPLE_HZ

    for subject, train_loader, val_loader, test_loader in loocv_generator(
        subjects,
        args.data_str,
        val_split=args.val_split,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        data_dir=args.data_dir,
        mdl_dir=args.mdl_dir,
        data_group=args.data_group,
        input_downsample_hz=input_downsample_hz,
    ):
        print(
            f"[SOURCE-BASELINE] model={model_name} subject={subject}", 
            flush=True
        )
        sample = next(iter(train_loader))[0]
        x0 = ensure_channel_first(sample)
        in_ch = int(x0.shape[1])
        seq_len = int(x0.shape[2])
        seed_rows = []
        seed_preds = []

        ens_len = int(args.inception_ensemble) \
                if model_name == "inceptiontime" else 1

        for ens_idx in range(ens_len):
            set_seed(int(args.seed) + ens_idx)
            model = make_model(
                model_name, in_ch=in_ch, seq_len=seq_len, emb_dim=args.emb_dim
            )
            info = train_one(model, train_loader, val_loader, args, device)
            out = collect(model, test_loader, device)
            met = regression_metrics(out["true"], out["pred"])
            seed_rows.append({
                "subject": subject,
                "model": model_name,
                "ensemble_member": ens_idx,
                "mae": met["mae"],
                "rmse": met["rmse"],
                "corr": met["corr"],
                "n_windows": met["n"],
                "best_val_mae": float(info["best_val_mae"]),
                "epochs_ran": int(info["epochs_ran"]),
            })
            for h in info["history"]:
                history_rows.append({"subject": subject, "model": model_name, "ensemble_member": ens_idx, **h})
            seed_preds.append(out)

        if len(seed_preds) == 1:
            final = seed_preds[0]
        else:
            # Source-faithful InceptionTime ensemble: average predictions from independently initialised models.
            final = dict(seed_preds[0])
            final["pred"] = np.mean([p["pred"] for p in seed_preds], axis=0)
            final["emb"] = np.mean([p["emb"] for p in seed_preds], axis=0)
        final_met = regression_metrics(final["true"], final["pred"])
        rows.append({
            "subject": subject,
            "model": model_name,
            "mode": "source",
            "mae": final_met["mae"],
            "rmse": final_met["rmse"],
            "corr": final_met["corr"],
            "n_windows": final_met["n"],
            "ensemble_n": len(seed_preds),
        })
        save_outputs(model_root / "subjects" / subject, subject, model_name, final)
        pd.DataFrame(seed_rows).to_csv(model_root / "subjects" / subject / "ensemble_members.csv", index=False)

    subject_df = pd.DataFrame(rows)
    hist_df = pd.DataFrame(history_rows)
    summary = subject_df.groupby(["model", "mode"], as_index=False).agg(
        n_subjects=("subject", "nunique"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        corr_mean=("corr", "mean"), corr_std=("corr", "std"),
        n_windows=("n_windows", "sum"),
    )
    subject_df.to_csv(model_root / "subject_rows.csv", index=False)
    hist_df.to_csv(model_root / "train_history.csv", index=False)
    summary.to_csv(model_root / "summary.csv", index=False)
    return subject_df, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models", default=DEFAULT_MODELS,
        help=f"Space-separated from: {' '.join(SOURCE_BASELINES)}")
    ap.add_argument(
        "--patchtst-file", default="PatchTST.py",
        help="Path to the TSLib/PatchTST Model file supplied as PatchTST.py"
    )
    ap.add_argument(
        "--timesnet-file", default="TimesNet.py",
        help="Path to the TSLib TimesNet Model file supplied as TimesNet.py"
    )
    ap.add_argument(
        "--timesnet-mode", default="downsample",
        choices=[
            "train", "skip", "downsample"
        ],
        help="Special handling for timesnet_tslib. import_existing no longer "\
        "supported, downsample instead.")
    ap.add_argument("--subjects", default=DEFAULT_SUBJECTS)
    ap.add_argument("--data-str", default="imu_filt")
    ap.add_argument("--data-dir",
                    default="/projects/BLVMob/imu-rr-seated/Data")
    ap.add_argument("--data-group", default="mr")
    ap.add_argument("--mdl-dir",
                    default="/projects/BLVMob/imu-rr-seated/models")
    ap.add_argument(
        "--out-dir",
        default="/projects/BLVMob/imu-rr-seated/results/jbhi_source_baselines"
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--val-split", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--inception-ensemble", type=int, default=1,
        help="Set 5 for the full InceptionTime ensemble protocol."
    )
    args = ap.parse_args()

    os.environ.setdefault("PATCHTST_FILE", str(args.patchtst_file))
    os.environ.setdefault("TIMESNET_FILE", str(args.timesnet_file))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(
            vars(args), indent=2, sort_keys=True
        )
    )
    device = torch.device(
        args.device if torch.cuda.is_available() or \
        not str(args.device).startswith("cuda") else "cpu"
    )
    all_subject, all_summary = [], []
    for model_name in parse_list(args.models):
        if model_name == "timesnet_tslib" and \
                str(args.timesnet_mode).lower() == "skip":
            print(
                "[TIMESNET] skipping timesnet_tslib by --timesnet-mode=skip",
                flush=True
            )
            continue
        else:
            sub, summ = run_one_model(model_name, args, device)
        all_subject.append(sub)
        all_summary.append(summ)

    if not all_subject:
        raise RuntimeError(
            "No model results were produced. Check --models and "\
            "--timesnet-mode.")

    subject_all = pd.concat(all_subject, ignore_index=True)

    summary_all = pd.concat(
        all_summary, ignore_index=True
    ).sort_values("mae_mean")

    subject_all.to_csv(out_dir / "subject_rows.csv", index=False)
    summary_all.to_csv(out_dir / "summary.csv", index=False)
    print(summary_all.to_string(index=False))


if __name__ == "__main__":
    main()
