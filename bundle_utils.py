import json
import time
from os.path import join, exists, basename, dirname, abspath
from typing import Any, Dict, Optional

import torch
from torch.nn.parameter import UninitializedParameter

try:
    from torch.nn.parameter import UninitializedBuffer
    _UNINIT_TYPES = (UninitializedParameter, UninitializedBuffer)
except Exception:
    _UNINIT_TYPES = (UninitializedParameter,)


def resolve_base_ckpt(out_dir: str, ckpt_prefix: str = "pretrain") -> Optional[str]:
    """
    Resolve the canonical base checkpoint path inside a run directory.
    Prefers ckpt_{prefix}_best.pt, falls back to ckpt_{prefix}_last.pt.
    Returns None if neither exists.
    """
    best_path = join(out_dir, f"ckpt_{ckpt_prefix}_best.pt")
    if exists(best_path):
        return best_path
    last_path = join(out_dir, f"ckpt_{ckpt_prefix}_last.pt")
    if exists(last_path):
        return last_path
    return None


def write_bundle(out_dir: str, bundle: Dict[str, Any]) -> str:
    """
    Write a minimal bundle.json into the run directory.
    """
    payload = dict(bundle)
    payload.setdefault("bundle_version", "1.0")
    payload.setdefault("timestamp", time.strftime("%Y%m%d-%H%M%S"))
    path = join(out_dir, "bundle.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def _base_ckpt_name(path: Optional[str]) -> Optional[str]:
    return basename(path) if path is not None else None

def resolve_pretrain_reference(
    *,
    ckpt_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    ckpt_prefix: str = "pretrain",
) -> Dict[str, Optional[str]]:
    """
    Resolve a canonical pretrained checkpoint reference for bundle/run metadata.
    """
    path = ckpt_path
    if path is None and out_dir is not None:
        path = resolve_base_ckpt(out_dir, ckpt_prefix=ckpt_prefix)
    if path is None or not exists(path):
        return {
            "pretrained_ckpt": None,
            "pretrained_ckpt_path": None,
            "pretrained_run_dir": None,
            "pretrained_run_id": None,
        }
    ckpt_abs = abspath(path)
    run_dir = dirname(ckpt_abs)
    return {
        "pretrained_ckpt": basename(ckpt_abs),
        "pretrained_ckpt_path": ckpt_abs,
        "pretrained_run_dir": run_dir,
        "pretrained_run_id": basename(run_dir),
    }


def load_base_checkpoint(
    model: torch.nn.Module,
    *,
    ckpt_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    ckpt_prefix: str = "pretrain",
    device: str = "cpu",
    strict: bool = True,
) -> Optional[str]:
    """
    Load a base checkpoint into `model`.
    If ckpt_path is provided, it is used directly.
    Otherwise resolves via out_dir + ckpt_prefix.
    Returns the resolved path if loaded, else None.
    """
    path = ckpt_path
    if path is None and out_dir is not None:
        path = resolve_base_ckpt(out_dir, ckpt_prefix=ckpt_prefix)
    if path is None or not exists(path):
        return None
    print(f"[CKPT] Loading base checkpoint: {path}")
    state = torch.load(path, map_location=device)
    # Handle Lightning-style checkpoints gracefully.
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    # Drop uninitialized (lazy) params from checkpoint to avoid load_state_dict errors.
    if isinstance(state, dict):
        removed = [k for k, v in state.items() if isinstance(v, _UNINIT_TYPES)]
        if removed:
            print(f"[CKPT] Dropping {len(removed)} uninitialized params: {removed[:5]}{'...' if len(removed) > 5 else ''}")
            state = {k: v for k, v in state.items() if k not in removed}
            strict = False

    model.load_state_dict(state, strict=strict)
    return path
