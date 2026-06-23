# experiment_recorder.py
from __future__ import annotations

import os
import json
import time
import glob
import shutil
import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple, List

import csv
import yaml
import torch
import pickle
import math

def _now() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def _safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None

def prediction_entropy_from_logits(logits: torch.Tensor) -> float:
    # logits: (B, C)
    p = torch.softmax(logits, dim=-1)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1).mean()
    return float(ent.detach().cpu().item())

def prediction_uncertainty_std(preds: List[torch.Tensor]) -> float:
    # preds: list of (B, ...) predictions from multiple views (e.g. style views)
    # returns mean std across batch
    if len(preds) <= 1:
        return 0.0
    P = torch.stack([p.detach() for p in preds], dim=0)  # (K,B,...)
    std = P.float().std(dim=0).mean()
    return float(std.detach().cpu().item())

def param_update_norm(model: torch.nn.Module, prev: Dict[str, torch.Tensor]) -> float:
    # L2 norm of Δθ over trainable params present in prev
    s = 0.0
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n not in prev:
            continue
        d = (p.detach() - prev[n]).float()
        s += float((d*d).sum().cpu().item())
    return float(math.sqrt(s))

def snapshot_trainable_params(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    snap = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            snap[n] = p.detach().clone()
    return snap

def get_method_from_prefix(prefix, start=2):
    print(prefix)
    prefix_split = prefix.split("_")
    shorthand = [
        prefix_split[i-1] for i, val in enumerate(prefix_split) 
        if i > start and val.isdigit() and int(val) == 1 and 
        prefix_split[i-1] != 'freeze'
    ]
    method = "_".join(shorthand)
    return method

@dataclass
class RunConfig:
    run_id: str
    subject: str
    method: str              # e.g. "baseline", "flow_ttt", "flow_ssa", "flow_ssa_cmt"
    backbone: str            # e.g. "vit", "timesnet", "primus", "limu"
    representation: str      # "raw" or "stft"
    seed: int
    device: str
    data_str: str
    window_size: int
    imu_fs: float
    br_fs: float
    extra: Dict[str, Any]

class ExperimentRecorder:
    """
    Lightweight file-based recorder.
    One instance per (subject, method, backbone) run.
    """
    def __init__(self, out_dir: str, run_cfg: RunConfig):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.cfg = run_cfg
        self.run_yaml = os.path.join(self.out_dir, "run.yaml")
        self.train_csv = os.path.join(self.out_dir, "train_log.csv")
        self.ttt_csv = os.path.join(self.out_dir, "ttt_steps.csv")
        self.metrics_csv = os.path.join(self.out_dir, "metrics_subject.csv")
        self.tb_dir = os.path.join(self.out_dir, "tb")
        self.tb_enabled = bool(
            isinstance(getattr(self.cfg, "extra", None), dict)
            and self.cfg.extra.get("tb_profile_enabled", False)
        )
        self.tb_writer = None
        if self.tb_enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as e:
                raise RuntimeError(
                    "TensorBoard profiling was enabled, but torch.utils.tensorboard "
                    "is unavailable. Install tensorboard or disable --tb-profile."
                ) from e
            self.tb_writer = SummaryWriter(log_dir=self.tb_dir)

        self.best_score = float("inf")
        self.best_epoch = -1
        self.last_epoch = 0
        self.ckpt_prefix = ""
        if isinstance(getattr(self.cfg, "extra", None), dict):
            self.ckpt_prefix = str(self.cfg.extra.get("ckpt_prefix", "") or "").strip()

        self.best_ckpt_path = self._ckpt_path("best", ckpt_prefix=self.ckpt_prefix)
        self.last_ckpt_path = self._ckpt_path("last", ckpt_prefix=self.ckpt_prefix)

        # Initialize csv headers once
        self._init_csv(self.train_csv, [
            "time", "epoch",
            "train_loss", "val_loss",
            "val_tes_mae", "val_tes_rmse", "val_rr_mae", "val_rr_rmse",
            "lr"
        ])
        self._init_csv(self.ttt_csv, [
            "time", "subject", "step",
            "flow_nll",
            "tes_mae", "tes_rmse",
            "rr_mae", "rr_rmse",
            "param_update_norm",
            "pred_entropy",
            "pred_uncertainty_std",
            "ssa_loss",
            "cmt_enabled",
            "shots"
        ])
        self._init_csv(self.metrics_csv, [
            "time", "subject", "method", "backbone", "representation",
            "tes_mae", "tes_rmse", "tes_pearson",
            "rr_mae", "rr_rmse", "rr_pearson",
            "best_epoch", "best_val_score"
        ])

        self.write_yaml()

    def _tb_add_scalar(self, tag: str, value: Optional[float], step: int) -> None:
        if self.tb_writer is None or value is None:
            return
        try:
            self.tb_writer.add_scalar(tag, float(value), int(step))
        except Exception:
            pass

    def flush(self) -> None:
        if self.tb_writer is not None:
            self.tb_writer.flush()

    def close(self) -> None:
        if self.tb_writer is not None:
            self.tb_writer.close()
            self.tb_writer = None

    def _ckpt_path(self, which: str, ckpt_prefix: str = "") -> str:
        """
        Build checkpoint path with optional stage prefix.
        Examples:
          ckpt_best.pt
          ckpt_pretrain_best.pt
          ckpt_downstream_last.pt
        """
        prefix = str(ckpt_prefix or "").strip()
        if prefix:
            fname = f"ckpt_{prefix}_{which}.pt"
        else:
            fname = f"ckpt_{which}.pt"
        return os.path.join(self.out_dir, fname)

    def set_ckpt_prefix(self, ckpt_prefix: str) -> None:
        """
        Set default checkpoint namespace for maybe_save_best/save_last.
        """
        self.ckpt_prefix = str(ckpt_prefix or "").strip()
        self.best_ckpt_path = self._ckpt_path("best", ckpt_prefix=self.ckpt_prefix)
        self.last_ckpt_path = self._ckpt_path("last", ckpt_prefix=self.ckpt_prefix)

    def checkpoint_path(self, which: str = "best", ckpt_prefix: str = "") -> str:
        """
        Public checkpoint path resolver.
        """
        prefix = self.ckpt_prefix if ckpt_prefix == "" else ckpt_prefix
        return self._ckpt_path(which, ckpt_prefix=prefix)

    def save_checkpoint(self, model: torch.nn.Module, which: str = "last",
                        ckpt_prefix: str = "") -> str:
        """
        Save a checkpoint to an explicit namespace without mutating best_score logic.
        """
        p = self.checkpoint_path(which=which, ckpt_prefix=ckpt_prefix)
        torch.save(model.state_dict(), p)
        return p

    def _init_csv(self, path: str, header: List[str]):
        if os.path.exists(path):
            return
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)

    def _save_state(self):
        """
        Persist minimal run state so we can resume / skip training safely.
        """
        state = {
            "best_score": float(self.best_score),
            "best_epoch": int(self.best_epoch),
            "last_epoch": int(self.last_epoch),
        }
        with open(os.path.join(self.out_dir, "state.json"), "w") as f:
            json.dump(state, f, indent=2, sort_keys=True)

    def _infer_last_epoch_from_log(self) -> int:
        if not os.path.exists(self.train_csv):
            return 0
        try:
            with open(self.train_csv, "r", newline="") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                return 0
            return int(rows[-1].get("epoch", 0) or 0)
        except Exception:
            return 0

    def _load_state(self):
        """
        Load persisted run state if it exists.
        """
        path = os.path.join(self.out_dir, "state.json")
        if not os.path.exists(path):
            self.last_epoch = self._infer_last_epoch_from_log()
            return
        with open(path, "r") as f:
            state = json.load(f)
        self.best_score = float(state.get("best_score", self.best_score))
        self.best_epoch = int(state.get("best_epoch", self.best_epoch))
        self.last_epoch = int(state.get("last_epoch", self._infer_last_epoch_from_log()) or 0)

    def write_yaml(self):
        with open(self.run_yaml, "w") as f:
            yaml.safe_dump(asdict(self.cfg), f, sort_keys=False)

    def log_epoch(self, *, epoch: int,
                  train_loss: Optional[float]=None,
                  val_loss: Optional[float]=None,
                  val_tes: Optional[Dict[str, float]]=None,
                  val_rr: Optional[Dict[str, float]]=None,
                  lr: Optional[float]=None):
        row = [
            _now(), epoch,
            _safe_float(train_loss), _safe_float(val_loss),
            _safe_float((val_tes or {}).get("mae")), _safe_float((val_tes or {}).get("rmse")),
            _safe_float((val_rr or {}).get("mae")),  _safe_float((val_rr or {}).get("rmse")),
            _safe_float(lr),
        ]
        with open(self.train_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)
        self._tb_add_scalar("loss/train", _safe_float(train_loss), epoch)
        self._tb_add_scalar("loss/val", _safe_float(val_loss), epoch)
        self._tb_add_scalar("metrics/val_tes_mae", _safe_float((val_tes or {}).get("mae")), epoch)
        self._tb_add_scalar("metrics/val_tes_rmse", _safe_float((val_tes or {}).get("rmse")), epoch)
        self._tb_add_scalar("metrics/val_rr_mae", _safe_float((val_rr or {}).get("mae")), epoch)
        self._tb_add_scalar("metrics/val_rr_rmse", _safe_float((val_rr or {}).get("rmse")), epoch)
        self._tb_add_scalar("optim/lr", _safe_float(lr), epoch)
        self.last_epoch = max(int(self.last_epoch), int(epoch))
        self._save_state()

    def maybe_save_best(self, model: torch.nn.Module, score: float, epoch: int,
                        complimentary_models:dict={}):
        # score: whatever you define as "val score" (usually val TES MAE)
        score = float(score)

        if score < self.best_score:
            self.best_score = score
            self.best_epoch = int(epoch)

            self.save_checkpoint(model, which="best")

            if len(complimentary_models) > 0:
                for mdl_str, mdl in complimentary_models.items():
                    ckpt_path = os.path.join(self.out_dir, f"{mdl_str}.pt")
                    torch.save(mdl.state_dict(), ckpt_path)

            self._save_state()
            return True

        return False

    def save_last(self, model: torch.nn.Module, epoch: Optional[int] = None):
        if epoch is not None:
            self.last_epoch = max(int(self.last_epoch), int(epoch))
        self.save_checkpoint(model, which="last")
        self._save_state()

    def save_last_training(
        self,
        model: torch.nn.Module,
        *,
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        self.last_epoch = max(int(self.last_epoch), int(epoch))
        payload: Dict[str, Any] = {
            "epoch": int(epoch),
            "model_state": model.state_dict(),
        }
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()
        if scaler is not None:
            try:
                payload["scaler_state"] = scaler.state_dict()
            except Exception:
                pass
        if extra_state:
            payload.update(extra_state)
        path = self.checkpoint_path(which="last")
        torch.save(payload, path)
        self._save_state()
        return path

    def log_ttt_step(self, *, subject: str, step: int,
                     flow_nll: Optional[float]=None,
                     tes: Optional[Dict[str, float]]=None,
                     rr: Optional[Dict[str, float]]=None,
                     param_norm: Optional[float]=None,
                     pred_entropy: Optional[float]=None,
                     pred_uncertainty_std: Optional[float]=None,
                     ssa_loss: Optional[float]=None,
                     cmt_enabled: bool=False,
                     shots: Optional[int]=None):
        row = [
            _now(), subject, step,
            _safe_float(flow_nll),
            _safe_float((tes or {}).get("mae")), _safe_float((tes or {}).get("rmse")),
            _safe_float((rr or {}).get("mae")),  _safe_float((rr or {}).get("rmse")),
            _safe_float(param_norm),
            _safe_float(pred_entropy),
            _safe_float(pred_uncertainty_std),
            _safe_float(ssa_loss),
            int(cmt_enabled),
            shots if shots is not None else "",
        ]
        with open(self.ttt_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)
        self._tb_add_scalar("ttt/flow_nll", _safe_float(flow_nll), step)
        self._tb_add_scalar("ttt/tes_mae", _safe_float((tes or {}).get("mae")), step)
        self._tb_add_scalar("ttt/tes_rmse", _safe_float((tes or {}).get("rmse")), step)
        self._tb_add_scalar("ttt/rr_mae", _safe_float((rr or {}).get("mae")), step)
        self._tb_add_scalar("ttt/rr_rmse", _safe_float((rr or {}).get("rmse")), step)
        self._tb_add_scalar("ttt/param_update_norm", _safe_float(param_norm), step)
        self._tb_add_scalar("ttt/pred_entropy", _safe_float(pred_entropy), step)
        self._tb_add_scalar("ttt/pred_uncertainty_std", _safe_float(pred_uncertainty_std), step)
        self._tb_add_scalar("ttt/ssa_loss", _safe_float(ssa_loss), step)

    def log_train_step_timing(
        self,
        *,
        global_step: int,
        epoch: int,
        step: int,
        timings: Dict[str, float],
    ) -> None:
        if self.tb_writer is None:
            return
        for key, value in timings.items():
            self._tb_add_scalar(f"timing/train/{key}", _safe_float(value), global_step)
        self._tb_add_scalar("timing/train/epoch", float(epoch), global_step)
        self._tb_add_scalar("timing/train/step_in_epoch", float(step), global_step)

    def log_subject_metrics(self, *,
                            subject: str,
                            tes: Optional[Dict[str, float]]=None,
                            rr: Optional[Dict[str, float]]=None,
                            tes_pearson: Optional[float]=None,
                            rr_pearson: Optional[float]=None):
        row = [
            _now(), subject, self.cfg.method, self.cfg.backbone, self.cfg.representation,
            _safe_float((tes or {}).get("mae")), _safe_float((tes or {}).get("rmse")), _safe_float(tes_pearson),
            _safe_float((rr or {}).get("mae")),  _safe_float((rr or {}).get("rmse")),  _safe_float(rr_pearson),
            self.best_epoch, self.best_score
        ]
        with open(self.metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def dump_pickle(self, name: str, obj: Any):
        with open(os.path.join(self.out_dir, name), "wb") as f:
            pickle.dump(obj, f)

    def update_extra(self, **kwargs: Any) -> None:
        """
        Merge fields into cfg.extra and persist run.yaml.
        """
        if not isinstance(self.cfg.extra, dict):
            self.cfg.extra = {}
        changed = False
        for k, v in kwargs.items():
            if self.cfg.extra.get(k) != v:
                self.cfg.extra[k] = v
                changed = True
        if changed:
            self.write_yaml()

def _now_compact() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def _normalized_cfg_dict(cfg_like: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(cfg_like)
    d.pop("run_id", None)
    d.pop("cfg_signature", None)
    d.pop("created_at", None)
    d.pop("run_dir", None)
    d.pop("overwritten", None)

    extra = d.get("extra", {})
    if isinstance(extra, dict):
        extra = dict(extra)
        # These affect rerun behavior but should not define whether a run is resumable.
        for k in ("epochs", "overwrite", "overwrite_limu_ssl"):
            extra.pop(k, None)
        # These are derived bookkeeping fields filled in after run creation.
        for k in (
            "loaded_pretrained_ckpt",
            "loaded_pretrained_ckpt_path",
            "loaded_pretrained_run_dir",
            "loaded_pretrained_run_id",
            "encoder_pretrained_ckpt",
            "encoder_pretrained_ckpt_path",
            "encoder_pretrained_run_dir",
            "encoder_pretrained_run_id",
        ):
            extra.pop(k, None)
        d["extra"] = extra

    return d

def make_cfg_signature(run_cfg: "RunConfig") -> str:
    """
    Stable signature for a configuration so we can detect if an equivalent run already exists.
    Excludes run_id (and any other volatile fields you choose).
    """
    d = _normalized_cfg_dict(asdict(run_cfg))

    # Normalize dict ordering
    payload = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def default_run_id(run_cfg: "RunConfig") -> str:
    """
    Human-readable run_id. You can customize this.
    """
    # Keep it short but informative; include timestamp to avoid collisions.
    # Subject/method/backbone/representation/seed are already in cfg.
    return f"{_now_compact()}_{run_cfg.subject}_{run_cfg.backbone}_{run_cfg.representation}_{run_cfg.method}_seed{run_cfg.seed}"

def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def _write_yaml(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def _clear_run_dir(run_dir: str) -> None:
    """
    Delete known artifacts so reruns don't mix logs/checkpoints.
    Leaves run.yaml intact (will be rewritten).
    """
    patterns = [
        "ckpt_best.pt",
        "ckpt_last.pt",
        "ckpt_*_best.pt",
        "ckpt_*_last.pt",
        "train_log.csv",
        "ttt_steps.csv",
        "metrics_subject.csv",
        "ssa.pkl",
        "cmt.pkl",
        "state.json",
        "tb",
        "*.npz",
        "*.pkl",
    ]
    for pat in patterns:
        for p in glob.glob(os.path.join(run_dir, pat)):
            try:
                os.remove(p)
            except IsADirectoryError:
                shutil.rmtree(p, ignore_errors=True)

def _load_state(run_dir: str) -> Dict[str, Any]:
    p = os.path.join(run_dir, "state.json")
    if not os.path.exists(p):
        return {}
    with open(p, "r") as f:
        return json.load(f)

def _save_state(run_dir: str, state: Dict[str, Any]) -> None:
    p = os.path.join(run_dir, "state.json")
    with open(p, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)

def prepare_or_resume_recorder(
    base_dir: str,
    run_cfg: "RunConfig",
    overwrite: bool = False,
) -> Tuple["ExperimentRecorder", bool]:
    """
    Detect if an equivalent configuration already exists under base_dir.

    - If match found and overwrite=False:
        - Load existing run (same directory)
        - Return should_train=False so caller can skip training/adaptation
    - If match found and overwrite=True:
        - Clear artifacts and return should_train=True (rerun)
    - If no match found:
        - Generate a new run_id + directory
        - Return should_train=True

    Returns:
        recorder, should_train
    """
    os.makedirs(base_dir, exist_ok=True)

    sig = make_cfg_signature(run_cfg)

    # Search for existing run dirs: we assume each run has run.yaml
    candidates = []
    for entry in os.listdir(base_dir):
        run_dir = os.path.join(base_dir, entry)
        if not os.path.isdir(run_dir):
            continue
        yml = os.path.join(run_dir, "run.yaml")
        if os.path.exists(yml):
            candidates.append(run_dir)

    matched_dir = None
    for run_dir in candidates:
        meta = _read_yaml(os.path.join(run_dir, "run.yaml"))
        try:
            payload = json.dumps(
                _normalized_cfg_dict(meta),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            existing_sig = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        except Exception:
            existing_sig = None

        if existing_sig == sig:
            matched_dir = run_dir
            break

    if matched_dir is None:
        # New run
        if not run_cfg.run_id:
            run_cfg.run_id = default_run_id(run_cfg)
        run_dir = os.path.join(base_dir, run_cfg.run_id)
        os.makedirs(run_dir, exist_ok=True)

        rec = ExperimentRecorder(out_dir=run_dir, run_cfg=run_cfg)
        # Rewrite YAML with signature metadata
        meta = asdict(run_cfg)
        meta["cfg_signature"] = sig
        meta["created_at"] = _now_compact()
        meta["run_dir"] = run_dir
        _write_yaml(os.path.join(run_dir, "run.yaml"), meta)

        _save_state(
            run_dir,
            {"best_score": rec.best_score, "best_epoch": rec.best_epoch, "last_epoch": rec.last_epoch},
        )
        return rec, True

    # Existing run found
    run_dir = matched_dir

    if overwrite:
        _clear_run_dir(run_dir)

        # Use existing directory but new run_id for clarity, OR keep old.
        # Requirement says: "skip unless overwrite enabled"; overwrite means rerun.
        # We'll keep directory name stable to truly "overwrite".
        if not run_cfg.run_id:
            # Keep directory name as-is; but store the current run_id in yaml anyway
            run_cfg.run_id = os.path.basename(run_dir)

        rec = ExperimentRecorder(out_dir=run_dir, run_cfg=run_cfg)
        meta = asdict(run_cfg)
        meta["cfg_signature"] = sig
        meta["created_at"] = _now_compact()
        meta["run_dir"] = run_dir
        meta["overwritten"] = True
        _write_yaml(os.path.join(run_dir, "run.yaml"), meta)

        _save_state(
            run_dir,
            {"best_score": rec.best_score, "best_epoch": rec.best_epoch, "last_epoch": rec.last_epoch},
        )
        return rec, True

    # Resume (skip training)
    # Load stored run_cfg from YAML so the recorder reflects the stored run exactly
    stored = _read_yaml(os.path.join(run_dir, "run.yaml"))
    # If YAML is directly RunConfig dump, use those fields; else adapt.
    # Fill from stored when present, otherwise keep provided.
    for k, v in stored.items():
        if hasattr(run_cfg, k):
            setattr(run_cfg, k, v)

    if not run_cfg.run_id:
        run_cfg.run_id = os.path.basename(run_dir)

    rec = ExperimentRecorder(out_dir=run_dir, run_cfg=run_cfg)

    # Restore best epoch/score if available
    state = _load_state(run_dir)
    if "best_score" in state:
        rec.best_score = float(state["best_score"])
    if "best_epoch" in state:
        rec.best_epoch = int(state["best_epoch"])
    if "last_epoch" in state:
        rec.last_epoch = int(state["last_epoch"])
    else:
        rec.last_epoch = rec._infer_last_epoch_from_log()

    return rec, False



# ---- OPTIONAL: minimal hook inside ExperimentRecorder to persist best_epoch/score ----
# Add these 2 lines inside your existing ExperimentRecorder.maybe_save_best and save_last:

# In ExperimentRecorder.maybe_save_best(...):
#     _save_state(self.out_dir, {"best_score": self.best_score, "best_epoch": self.best_epoch})

# In ExperimentRecorder.save_last(...):
#     _save_state(self.out_dir, {"best_score": self.best_score, "best_epoch": self.best_epoch})
