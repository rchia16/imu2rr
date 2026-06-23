#!/usr/bin/env python3
"""Neural RR baselines and cross-modal model components for JBHI-style experiments.

The goal is to keep a compact, readable set of representative baselines:
  - ResNet1D raw IMU -> RR
  - CNN-GRU raw IMU -> RR
  - TCN raw IMU -> RR
  - InceptionTime-style raw IMU -> RR
  - STFT-CNN IMU spectrogram -> RR
  - PatchTST-style patch Transformer -> RR
  - CrossModalRRNet: IMU encoder with pressure waveform reconstruction + RR head

References / implementation notes:
  - InceptionTime: official TensorFlow repo https://github.com/hfawaz/InceptionTime
    and paper Fawaz et al. 2019. NOTE: this is a compact PyTorch adaptation,
    not the full 5-model ensemble.
  - PatchTST: official repo https://github.com/PatchTST/PatchTST. NOTE: this
    uses the key patch-token Transformer idea with channel mixing simplified for
    RR regression.
  - TimesNet: official TSlib repo https://github.com/thuml/Time-Series-Library.
    NOTE: TimesNet is handled as an optional external baseline in the runner if
    your existing times_experiment.py/TSlib setup is available; this file avoids
    vendoring external code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Common output container
# -----------------------------------------------------------------------------
@dataclass
class RRForward:
    pred: torch.Tensor
    emb: torch.Tensor
    recon_pressure: Optional[torch.Tensor] = None
    aux: Optional[Dict[str, torch.Tensor]] = None


class GlobalAvgPool1d(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=-1)


class MLPHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


# -----------------------------------------------------------------------------
# ResNet1D baseline
# -----------------------------------------------------------------------------
class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7, stride: int = 1):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = nn.Identity()
        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.conv(x) + self.skip(x))


class ResNet1DRR(nn.Module):
    def __init__(self, in_ch: int = 6, width: int = 64, emb_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_ch, width, 7, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
            ResBlock1D(width, width),
            ResBlock1D(width, width * 2, stride=2),
            ResBlock1D(width * 2, width * 2),
            ResBlock1D(width * 2, width * 4, stride=2),
            ResBlock1D(width * 4, width * 4),
        )
        self.proj = nn.Sequential(GlobalAvgPool1d(), nn.Linear(width * 4, emb_dim), nn.GELU())
        self.head = MLPHead(emb_dim)

    def forward(self, x: torch.Tensor) -> RRForward:
        h = self.encoder(x.float())
        z = self.proj(h)
        return RRForward(pred=self.head(z), emb=z)


# -----------------------------------------------------------------------------
# CNN-GRU baseline
# -----------------------------------------------------------------------------
class CNNGRURR(nn.Module):
    def __init__(self, in_ch: int = 6, width: int = 64, emb_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, width, 7, padding=3), nn.BatchNorm1d(width), nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2), nn.BatchNorm1d(width), nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(width, width * 2, 5, padding=2), nn.BatchNorm1d(width * 2), nn.GELU(),
        )
        self.gru = nn.GRU(width * 2, emb_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim), nn.GELU())
        self.head = MLPHead(emb_dim)

    def forward(self, x: torch.Tensor) -> RRForward:
        h = self.conv(x.float()).transpose(1, 2)
        y, _ = self.gru(h)
        z = self.proj(y.mean(dim=1))
        return RRForward(pred=self.head(z), emb=z)


# -----------------------------------------------------------------------------
# TCN baseline
# -----------------------------------------------------------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = int(chomp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp] if self.chomp > 0 else x


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation),
            Chomp1d(pad), nn.BatchNorm1d(out_ch), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation),
            Chomp1d(pad), nn.BatchNorm1d(out_ch), nn.GELU(), nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.net(x) + self.skip(x))


class TCNRR(nn.Module):
    def __init__(self, in_ch: int = 6, width: int = 64, emb_dim: int = 128):
        super().__init__()
        layers = []
        ch = in_ch
        for d in [1, 2, 4, 8, 16]:
            layers.append(TCNBlock(ch, width, dilation=d))
            ch = width
        self.encoder = nn.Sequential(*layers)
        self.proj = nn.Sequential(GlobalAvgPool1d(), nn.Linear(width, emb_dim), nn.GELU())
        self.head = MLPHead(emb_dim)

    def forward(self, x: torch.Tensor) -> RRForward:
        h = self.encoder(x.float())
        z = self.proj(h)
        return RRForward(pred=self.head(z), emb=z)


# -----------------------------------------------------------------------------
# InceptionTime-style baseline
# -----------------------------------------------------------------------------
class InceptionBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 32, bottleneck: int = 32):
        super().__init__()
        mid = bottleneck if in_ch > 1 else in_ch
        self.bottleneck = nn.Conv1d(in_ch, mid, 1, bias=False)
        self.branches = nn.ModuleList([
            nn.Conv1d(mid, out_ch, k, padding=k // 2, bias=False) for k in [9, 19, 39]
        ])
        self.pool_branch = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(in_ch, out_ch, 1, bias=False))
        self.bn = nn.BatchNorm1d(out_ch * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xb = self.bottleneck(x)
        ys = [b(xb) for b in self.branches] + [self.pool_branch(x)]
        return F.gelu(self.bn(torch.cat(ys, dim=1)))


class InceptionTimeRR(nn.Module):
    def __init__(self, in_ch: int = 6, blocks: int = 6, out_ch: int = 32, emb_dim: int = 128):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(blocks):
            layers.append(InceptionBlock1D(ch, out_ch=out_ch))
            ch = out_ch * 4
        self.encoder = nn.Sequential(*layers)
        self.proj = nn.Sequential(GlobalAvgPool1d(), nn.Linear(ch, emb_dim), nn.GELU())
        self.head = MLPHead(emb_dim)

    def forward(self, x: torch.Tensor) -> RRForward:
        h = self.encoder(x.float())
        z = self.proj(h)
        return RRForward(pred=self.head(z), emb=z)


# -----------------------------------------------------------------------------
# STFT-CNN baseline
# -----------------------------------------------------------------------------
class STFTCNNRR(nn.Module):
    """IMU STFT spectrogram CNN -> RR.

    NOTE: Uses torch.stft internally for a direct spectrogram baseline. This is
    a representative neural counterpart to classical dominant-frequency RR.
    """
    def __init__(self, in_ch: int = 6, emb_dim: int = 128, n_fft: int = 128, hop: int = 32):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(128, emb_dim), nn.GELU())
        self.head = MLPHead(emb_dim)

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        xx = x.reshape(b * c, t)
        win = torch.hann_window(self.n_fft, device=x.device)
        z = torch.stft(xx, n_fft=self.n_fft, hop_length=self.hop, window=win, return_complex=True)
        mag = torch.log1p(z.abs())
        return mag.reshape(b, c, mag.shape[-2], mag.shape[-1])

    def forward(self, x: torch.Tensor) -> RRForward:
        spec = self._stft(x.float())
        z = self.proj(self.conv(spec))
        return RRForward(pred=self.head(z), emb=z)


# -----------------------------------------------------------------------------
# PatchTST-style baseline
# -----------------------------------------------------------------------------
class PatchTSTRR(nn.Module):
    """Compact PatchTST-style patch Transformer for RR regression.

    NOTE: Captures the PatchTST idea of segmenting time series into patch tokens.
    The original channel-independence design is simplified here for multichannel
    IMU RR regression to keep the baseline readable and aligned with this repo.
    """
    def __init__(self, in_ch: int = 6, emb_dim: int = 128, patch_len: int = 32, stride: int = 16, depth: int = 3, heads: int = 4):
        super().__init__()
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.patch_proj = nn.Linear(in_ch * patch_len, emb_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=heads, dim_feedforward=emb_dim * 4,
            dropout=0.1, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.cls = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.head = MLPHead(emb_dim)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def _patches(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T) -> (B,N,C*patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.flatten(2)

    def forward(self, x: torch.Tensor) -> RRForward:
        p = self._patches(x.float())
        tok = self.patch_proj(p)
        cls = self.cls.expand(tok.shape[0], -1, -1)
        h = self.encoder(torch.cat([cls, tok], dim=1))
        z = h[:, 0]
        return RRForward(pred=self.head(z), emb=z)


# -----------------------------------------------------------------------------
# Cross-modal respiration-grounded model
# -----------------------------------------------------------------------------
class CrossModalRRNet(nn.Module):
    """IMU -> latent RR plus pressure waveform reconstruction.

    NOTE: This is a compact, journal-baseline implementation of the core idea in
    the project: learn an IMU representation grounded by pressure/respiration.
    It reconstructs pressure waveform directly for stability and simplicity. If
    you need exact STFT patch contrastive reconstruction, use your full
    vit_pressure_crossmodal_* scripts; this class provides the readable main
    ablation baseline for the JBHI table.
    """
    def __init__(self, in_ch: int = 6, width: int = 64, emb_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, width, 7, padding=3), nn.BatchNorm1d(width), nn.GELU(),
            ResBlock1D(width, width),
            ResBlock1D(width, width * 2, stride=2),
            ResBlock1D(width * 2, width * 4, stride=2),
        )
        self.pool = GlobalAvgPool1d()
        self.proj = nn.Sequential(nn.Linear(width * 4, emb_dim), nn.GELU())
        self.head = MLPHead(emb_dim)
        self.pressure_decoder = nn.Sequential(
            nn.ConvTranspose1d(width * 4, width * 2, 4, stride=2, padding=1), nn.BatchNorm1d(width * 2), nn.GELU(),
            nn.ConvTranspose1d(width * 2, width, 4, stride=2, padding=1), nn.BatchNorm1d(width), nn.GELU(),
            nn.Conv1d(width, 1, 7, padding=3),
        )

    def forward(self, x: torch.Tensor) -> RRForward:
        h = self.stem(x.float())
        z = self.proj(self.pool(h))
        recon = self.pressure_decoder(h).squeeze(1)
        # Decoder length can differ by one/two samples after stride ops.
        recon = recon[..., : x.shape[-1]]
        return RRForward(pred=self.head(z), emb=z, recon_pressure=recon)


def make_model(name: str, in_ch: int = 6, emb_dim: int = 128) -> nn.Module:
    name = str(name).lower().strip()
    if name == "resnet1d":
        return ResNet1DRR(in_ch=in_ch, emb_dim=emb_dim)
    if name == "cnn_gru":
        return CNNGRURR(in_ch=in_ch, emb_dim=emb_dim)
    if name == "tcn":
        return TCNRR(in_ch=in_ch, emb_dim=emb_dim)
    if name == "inceptiontime":
        return InceptionTimeRR(in_ch=in_ch, emb_dim=emb_dim)
    if name == "stft_cnn":
        return STFTCNNRR(in_ch=in_ch, emb_dim=emb_dim)
    if name == "patchtst":
        return PatchTSTRR(in_ch=in_ch, emb_dim=emb_dim)
    if name in {"crossmodal", "crossmodal_rr"}:
        return CrossModalRRNet(in_ch=in_ch, emb_dim=emb_dim)
    raise ValueError(f"Unknown model {name!r}")
