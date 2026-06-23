#!/usr/bin/env python3
"""Source-faithful JBHI RR baseline models with TSLib PatchTST/TimesNet wrappers.

This self-contained module defines the project-native neural comparators
(ResNet1D, CNN-GRU, TCN, InceptionTime) and directly wraps the provided
Time-Series-Library style PatchTST.py and TimesNet.py files.

IMPORTANT:
  - The main cross-modal model is intentionally not defined here. It must be run
    via vit_pressure_crossmodal_stft_rr_core.py and the adaptation ladder.
  - PatchTST/TimesNet internals are not simplified. These wrappers instantiate
    the uploaded/source Model classes and only adapt their output interface to
    scalar RR regression.
  - The TSLib layers/ package must be present. If it is unavailable, import
    fails loudly; no fallback model is used.
"""
from __future__ import annotations

import math
import os
import sys
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RRForward:
    pred: torch.Tensor
    emb: torch.Tensor
    aux: Optional[Dict[str, torch.Tensor]] = None


class RRHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Project-native strong baselines
# ---------------------------------------------------------------------------
class ResBlock1D(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int = 7, stride: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv1d(c_out, c_out, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(c_out),
        )
        self.shortcut = nn.Identity()
        if c_in != c_out or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm1d(c_out),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.net(x) + self.shortcut(x), inplace=True)


class ResNet1DRR(nn.Module):
    """Strong raw-IMU ResNet1D baseline; project-native, not an upstream claim."""
    def __init__(self, in_ch: int = 6, width: int = 64, emb_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_ch, width, 7, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            ResBlock1D(width, width),
            ResBlock1D(width, width * 2, stride=2),
            ResBlock1D(width * 2, width * 2),
            ResBlock1D(width * 2, width * 4, stride=2),
            ResBlock1D(width * 4, width * 4),
        )
        self.proj = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(width * 4, emb_dim), nn.GELU())
        self.head = RRHead(emb_dim)

    def forward(self, x: torch.Tensor) -> RRForward:
        h = self.encoder(x.float())
        z = self.proj(h)
        return RRForward(pred=self.head(z), emb=z)


class CNNGRURR(nn.Module):
    """CNN-GRU raw-IMU baseline; project-native temporal neural comparator."""
    def __init__(self, in_ch: int = 6, width: int = 64, emb_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, width, 7, padding=3, bias=False), nn.BatchNorm1d(width), nn.ReLU(inplace=True),
            nn.Conv1d(width, width, 5, padding=2, bias=False), nn.BatchNorm1d(width), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(width, width * 2, 5, padding=2, bias=False), nn.BatchNorm1d(width * 2), nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(width * 2, emb_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim), nn.GELU())
        self.head = RRHead(emb_dim)

    def forward(self, x: torch.Tensor) -> RRForward:
        h = self.conv(x.float()).transpose(1, 2)
        y, _ = self.gru(h)
        z = self.proj(y.mean(dim=1))
        return RRForward(pred=self.head(z), emb=z)


class Chomp1d(nn.Module):
    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = int(chomp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp] if self.chomp > 0 else x


class TemporalBlock(nn.Module):
    """TCN block following Bai et al. causal padded residual structure."""
    def __init__(self, c_in: int, c_out: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(c_out, c_out, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.net(x) + self.downsample(x), inplace=True)


class TCNRR(nn.Module):
    """Temporal convolutional network baseline; standard causal dilated TCN."""
    def __init__(self, in_ch: int = 6, width: int = 64, emb_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        layers = []
        c = in_ch
        for d in [1, 2, 4, 8, 16, 32]:
            layers.append(TemporalBlock(c, width, kernel_size=5, dilation=d, dropout=dropout))
            c = width
        self.encoder = nn.Sequential(*layers)
        self.proj = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(width, emb_dim), nn.GELU())
        self.head = RRHead(emb_dim)

    def forward(self, x: torch.Tensor) -> RRForward:
        h = self.encoder(x.float())
        z = self.proj(h)
        return RRForward(pred=self.head(z), emb=z)


# ---------------------------------------------------------------------------
# InceptionTime faithful PyTorch port
# ---------------------------------------------------------------------------
class InceptionModule(nn.Module):
    """InceptionTime module faithful to Fawaz et al. / official repo structure.

    NOTE: This is a PyTorch port of the published module: bottleneck 1x1 conv,
    parallel large kernels, max-pool branch, concat, BN, ReLU. The official
    repo is TensorFlow/Keras; weights are trained here under the same LOSO
    protocol rather than imported.
    """
    def __init__(self, c_in: int, nb_filters: int = 32, kernel_size: int = 40, bottleneck: int = 32):
        super().__init__()
        kernel_sizes = [kernel_size // (2 ** i) for i in range(3)]
        kernel_sizes = [k if k % 2 == 1 else k - 1 for k in kernel_sizes]
        self.use_bottleneck = c_in > 1
        self.bottleneck = nn.Conv1d(c_in, bottleneck, 1, bias=False) if self.use_bottleneck else nn.Identity()
        b_ch = bottleneck if self.use_bottleneck else c_in
        self.conv_list = nn.ModuleList([
            nn.Conv1d(b_ch, nb_filters, k, padding=k // 2, bias=False) for k in kernel_sizes
        ])
        self.maxconvpool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(c_in, nb_filters, 1, bias=False),
        )
        self.bn = nn.BatchNorm1d(nb_filters * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x)
        ys = [conv(z) for conv in self.conv_list]
        ys.append(self.maxconvpool(x))
        return F.relu(self.bn(torch.cat(ys, dim=1)), inplace=True)


class InceptionBlock(nn.Module):
    """Six-module InceptionTime block with residual every third module."""
    def __init__(self, c_in: int, depth: int = 6, nb_filters: int = 32, kernel_size: int = 40):
        super().__init__()
        self.modules_list = nn.ModuleList()
        self.shortcuts = nn.ModuleDict()
        c = c_in
        out_c = nb_filters * 4
        for d in range(depth):
            self.modules_list.append(InceptionModule(c, nb_filters=nb_filters, kernel_size=kernel_size))
            if d % 3 == 2:
                self.shortcuts[str(d)] = nn.Sequential(nn.Conv1d(c_in if d == 2 else out_c, out_c, 1, bias=False), nn.BatchNorm1d(out_c))
            c = out_c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        for i, mod in enumerate(self.modules_list):
            x = mod(x)
            if i % 3 == 2:
                x = F.relu(x + self.shortcuts[str(i)](res), inplace=True)
                res = x
        return x


class InceptionTimeRR(nn.Module):
    """Source-faithful single InceptionTime model for regression.

    NOTE: The official InceptionTime paper often ensembles 5 models. Use
    --inception-ensemble 5 in the runner for that source-faithful ensemble.
    """
    def __init__(self, in_ch: int = 6, nb_filters: int = 32, emb_dim: int = 128):
        super().__init__()
        self.encoder = InceptionBlock(in_ch, depth=6, nb_filters=nb_filters, kernel_size=40)
        self.proj = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(nb_filters * 4, emb_dim), nn.GELU())
        self.head = RRHead(emb_dim)

    def forward(self, x: torch.Tensor) -> RRForward:
        h = self.encoder(x.float())
        z = self.proj(h)
        return RRForward(pred=self.head(z), emb=z)




# ---------------------------------------------------------------------------
# TSLib PatchTST / TimesNet direct source wrappers
# ---------------------------------------------------------------------------
SOURCE_BASELINES_TSLIB = [
    "resnet1d",
    "cnn_gru",
    "tcn",
    "inceptiontime",
    "patchtst_tslib",
    "timesnet_tslib",
]


def _load_module_from_file(env_name: str, default_file: str, module_alias: str):
    """Load a TSLib model module from a concrete file path.

    NOTE: This intentionally does not vendor the `layers/` package. The official
    TSLib/PatchTST modules import `layers.*`; if that package is not available in
    the repo or PYTHONPATH, the ImportError is the correct behaviour.
    """
    path_text = os.environ.get(env_name, "").strip() or default_file
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise ImportError(
            f"{env_name} was not found at {path}. Copy {default_file} into the repo root "
            f"or set {env_name}=/absolute/path/to/{default_file}. No simplified fallback is used."
        )
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(module_alias, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as exc:
        raise ImportError(
            f"Failed to import {path}. This usually means the official TSLib/PatchTST `layers/` "
            f"package is not on PYTHONPATH. No simplified fallback is used. Original error: {exc}"
        ) from exc
    if not hasattr(module, "Model"):
        raise ImportError(f"{path} does not define a Model class")
    return module


class _ProjectionEmbeddingHook:
    """Capture the tensor entering a source model's final projection layer."""

    def __init__(self):
        self.value: Optional[torch.Tensor] = None

    def __call__(self, _module, inputs, _output):
        if inputs and torch.is_tensor(inputs[0]):
            x = inputs[0]
            self.value = x.reshape(x.size(0), -1)


class PatchTSTTSLibRR(nn.Module):
    """Thin RR wrapper around the provided/official PatchTST Model.

    The uploaded PatchTST implementation exposes the official task-dispatch
    interface and a classification branch. For scalar RR regression, we use the
    classification branch with `num_class=1`; this preserves the patch embedding,
    encoder, flatten/dropout/projection path of the source model. The scalar
    output is trained with a regression loss by the runner.

    NOTE: This is a task-head adaptation, not an architectural simplification.
    The PatchTST internals are the source Model class from `PatchTST.py`.
    """

    def __init__(
        self,
        in_ch: int,
        seq_len: int,
        emb_dim: int = 128,
        patch_len: int = 16,
        stride: int = 8,
        e_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        module = _load_module_from_file("PATCHTST_FILE", "PatchTST.py", "_jbhi_patchtst_tslib_source")
        cfg = SimpleNamespace(
            task_name="classification",
            seq_len=int(seq_len),
            pred_len=1,
            enc_in=int(in_ch),
            d_model=int(emb_dim),
            d_ff=int(emb_dim) * 4,
            e_layers=int(e_layers),
            n_heads=int(n_heads),
            factor=3,
            dropout=float(dropout),
            activation="gelu",
            num_class=1,
        )
        self.model = module.Model(cfg, patch_len=int(patch_len), stride=int(stride))
        self._hook = _ProjectionEmbeddingHook()
        self.model.projection.register_forward_hook(self._hook)

    def forward(self, x: torch.Tensor) -> RRForward:
        # Source PatchTST Model expects x_enc as [B, T, C]. Project loaders give [B, C, T].
        x_enc = x.float().transpose(1, 2).contiguous()
        y = self.model(x_enc, None, None, None)
        pred = y.reshape(y.size(0), -1)[:, 0]
        emb = self._hook.value
        if emb is None:
            # This should not happen unless the upstream projection path changes.
            raise RuntimeError("PatchTST projection hook did not capture embeddings")
        return RRForward(pred=pred, emb=emb)


class TimesNetTSLibRR(nn.Module):
    """Thin RR wrapper around the provided/official TSLib TimesNet Model.

    The uploaded TimesNet implementation includes a `task_name='regression'`
    branch that outputs `[B, pred_len, c_out]`. For scalar RR, we set
    `pred_len=1` and `c_out=1`, then train that scalar with the runner's RR loss.

    NOTE: The TimesNet internals are the source Model class from `TimesNet.py`;
    this wrapper only supplies the TSLib config namespace and reshapes outputs.
    """

    def __init__(
        self,
        in_ch: int,
        seq_len: int,
        emb_dim: int = 128,
        e_layers: int = 2,
        top_k: int = 5,
        num_kernels: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        module = _load_module_from_file("TIMESNET_FILE", "TimesNet.py", "_jbhi_timesnet_tslib_source")
        cfg = SimpleNamespace(
            task_name="regression",
            seq_len=int(seq_len),
            label_len=0,
            pred_len=1,
            top_k=int(top_k),
            d_model=int(emb_dim),
            d_ff=int(emb_dim) * 4,
            num_kernels=int(num_kernels),
            e_layers=int(e_layers),
            enc_in=int(in_ch),
            c_out=1,
            embed="timeF",
            freq="s",
            dropout=float(dropout),
        )
        self.model = module.Model(cfg)
        self._hook = _ProjectionEmbeddingHook()
        self.model.projection.register_forward_hook(self._hook)

    def forward(self, x: torch.Tensor) -> RRForward:
        # TSLib TimesNet expects x_enc as [B, T, C]. Project loaders give [B, C, T].
        x_enc = x.float().transpose(1, 2).contiguous()
        y = self.model(x_enc, None, None, None)
        pred = y.reshape(y.size(0), -1)[:, 0]
        emb = self._hook.value
        if emb is None:
            raise RuntimeError("TimesNet projection hook did not capture embeddings")
        return RRForward(pred=pred, emb=emb)


def make_model(name: str, in_ch: int, seq_len: int, emb_dim: int = 128) -> nn.Module:
    name = str(name).lower().strip()
    if name == "resnet1d":
        return ResNet1DRR(in_ch=in_ch, emb_dim=emb_dim)
    if name == "cnn_gru":
        return CNNGRURR(in_ch=in_ch, emb_dim=emb_dim)
    if name == "tcn":
        return TCNRR(in_ch=in_ch, emb_dim=emb_dim)
    if name == "inceptiontime":
        return InceptionTimeRR(in_ch=in_ch, emb_dim=emb_dim)
    if name in {"patchtst_tslib", "patchtst_official"}:
        return PatchTSTTSLibRR(in_ch=in_ch, seq_len=seq_len, emb_dim=emb_dim)
    if name in {"timesnet_tslib", "timesnet_official"}:
        return TimesNetTSLibRR(in_ch=in_ch, seq_len=seq_len, emb_dim=emb_dim)
    raise ValueError(f"Unknown TSLib/source baseline {name!r}")
