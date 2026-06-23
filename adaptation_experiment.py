import ipdb
import sys
import os
from os import makedirs
from os.path import exists, join
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Optional, Tuple
import pickle

from pretrain import get_evals
from config import IMU_FS, BR_FS, SBJ_PROCESSED_DIR, M_DIR, SEAT_DATA_DIR
from dataloader import loocv_generator, make_fewshot_loader_from_test
from digitalsignalprocessing import get_max_freq
from experiment_recorder import (
    ExperimentRecorder, snapshot_trainable_params, param_update_norm,
    prediction_uncertainty_std
)

PRIMUS_DIR  = os.environ.get("PRIMUS_DIR",  "./pretrained-imu-encoders")
LIMUB_DIR   = os.environ.get("LIMUB_DIR",   "./LIMU-BERT-Public")

for p in [PRIMUS_DIR, LIMUB_DIR]:
    if p and p not in sys.path:
        sys.path.insert(0, p)

_timesnet_freeze_kwargs = {
    'proj_keywords': ("spec_proj",),
    'unfreeze_proj': True,
    'unfreeze_pytorch_layernorm':True,
    'unfreeze_custom_layernorm':False,  # not needed
    'unfreeze_other_norms':False,       # TimesNet usually LN only
    'hard_freeze_keywords':("flow", "clf_head", "predict_linear", "projection"),
}

_limu_freeze_kwargs = {
        'proj_keywords':("spec_proj",),       # keep strict; avoid "proj" (attention has proj_q/k/v)
        'unfreeze_proj':True,
        'unfreeze_pytorch_layernorm':False,  # LIMU uses custom LN
        'unfreeze_custom_layernorm':True,    # <-- enables gamma/beta
        'unfreeze_other_norms':False,
        'hard_freeze_keywords':("flow", "clf_head"),
}

_primus_freeze_kwargs = {
    'proj_keywords': ("spec_proj", "projection", "adapter", "adapters", 
                     "bottleneck"),
    'unfreeze_proj': True,
    'unfreeze_pytorch_layernorm': True,
    'unfreeze_custom_layernorm': False,
    'unfreeze_other_norms': True,
    'hard_freeze_keywords': (
        "text_encoder", "video_encoder", "audio_encoder",
        "mmcl_loss", "ssl_loss",
        "flow", "clf_head",
        "head",  # keep downstream head fixed for flow-TTT stability
    ),
}
_imu2clip_freeze_kwargs = {
    'proj_keywords': ('projection', 'proj', 'adapter', 'adapters', 'bottleneck'),
    'unfreeze_proj': True,
    'unfreeze_pytorch_layernorm': True,
    'unfreeze_custom_layernorm': False,
    'unfreeze_other_norms': True,
    'hard_freeze_keywords': ('flow', 'clf_head', 'head'),
}

_limubx_freeze_kwargs = {
    'proj_keywords': ('projection', 'adapter', 'adapters', 'bottleneck'),
    'unfreeze_proj': True,
    'unfreeze_pytorch_layernorm': True,
    'unfreeze_custom_layernorm': True,
    'unfreeze_other_norms': False,
    'hard_freeze_keywords': ('flow', 'clf_head', 'head'),
}

_unihar_freeze_kwargs = {
    'proj_keywords': ('projection', 'adapter', 'adapters', 'bottleneck'),
    'unfreeze_proj': True,
    'unfreeze_pytorch_layernorm': True,
    'unfreeze_custom_layernorm': True,
    'unfreeze_other_norms': False,
    'hard_freeze_keywords': ('flow', 'clf_head', 'head'),
}

_normwear_freeze_kwargs = {
    'proj_keywords': ('projection', 'proj', 'adapter', 'adapters', 'bottleneck'),
    'unfreeze_proj': True,
    'unfreeze_pytorch_layernorm': True,
    'unfreeze_custom_layernorm': True,
    'unfreeze_other_norms': True,
    'hard_freeze_keywords': ('flow', 'clf_head', 'head'),
}

mdl_freeze_kwargs = {'timesnet': _timesnet_freeze_kwargs,
                     'limu': _limu_freeze_kwargs,
                     'primus': _primus_freeze_kwargs,
                     'imu2clip': _imu2clip_freeze_kwargs,
                     'limu_bert_x': _limubx_freeze_kwargs,
                     'unihar': _unihar_freeze_kwargs,
                     'normwear': _normwear_freeze_kwargs,
}

# ============================================================
# Flow bits (same idea as vit_experiment.py SimpleRealNVPFlow)
# ============================================================
class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden_dim=128, mask_even=True):
        super().__init__()
        mask = torch.zeros(dim)
        mask[::2] = 1.0 if mask_even else 0.0
        mask[1::2] = 0.0 if mask_even else 1.0
        self.register_buffer("mask", mask)

        self.net_s = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim)
        )
        self.net_t = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim)
        )

    def forward(self, z, reverse=False):
        m = self.mask
        z_masked = z * m
        s = torch.tanh(self.net_s(z_masked))
        t = self.net_t(z_masked)
        if not reverse:
            z_trans = z_masked + (1 - m) * (z * torch.exp(s) + t)
            log_det = ((1 - m) * s).sum(dim=1)
        else:
            z_trans = z_masked + (1 - m) * ((z - t) * torch.exp(-s))
            log_det = -((1 - m) * s).sum(dim=1)
        return z_trans, log_det

class SimpleRealNVPFlow(nn.Module):
    def __init__(self, dim, num_layers=4, hidden_dim=128):
        super().__init__()
        # alternate masks
        layers = []
        for i in range(num_layers):
            layers.append(AffineCoupling(dim, hidden_dim, mask_even=(i % 2 == 0)))
        self.layers = nn.ModuleList(layers)

    def forward(self, z):
        """
        z: (B, D)
        returns:
            u: (B, D)
            log_det: (B,)
        """
        log_det_total = torch.zeros(z.size(0), device=z.device)
        u = z
        for layer in self.layers:
            u, log_det = layer(u, reverse=False)
            log_det_total += log_det
        return u, log_det_total

    def log_prob(self, z):
        """
        log p(z) under standard normal base + flow
        """
        u, log_det = self.forward(z)
        # log N(0, I) up to constant
        log_pu = -0.5 * (u ** 2).sum(dim=1)
        log_pz = log_pu + log_det
        return log_pz

def flow_to_u(flow, z):
    """z -> u for your SimpleRealNVPFlow"""
    u, _ = flow.forward(z)
    return u

def flow_from_u(flow, u):
    """u -> z for your SimpleRealNVPFlow (invert coupling stack)"""
    z = u
    # run layers in reverse with reverse=True
    for layer in reversed(flow.layers):
        z, _ = layer(z, reverse=True)
    return z

# ============================================================
# Attention pooling over time
# ============================================================
class AttnPool1D(nn.Module):
    """x: (B,T,D) -> (B,D)"""
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        scores = torch.einsum("btd,d->bt", x, self.q)
        w = F.softmax(scores, dim=1)
        w = self.drop(w)
        return torch.einsum("btd,bt->bd", x, w)


# ============================================================
# Freeze policy: "LayerNorm + proj" style (like vit_experiment.py)
# ============================================================

def freeze_all_but_norm_and_proj(
    model: nn.Module,
    # modules to *optionally* adapt (kept conservative by default)
    proj_keywords: Tuple[str, ...] = ("spec_proj", "projection", "proj_head",
                                      "adapter", "adapters", "bottleneck"),
    unfreeze_proj: bool = True,

    # norms to adapt
    unfreeze_pytorch_layernorm: bool = True,
    unfreeze_custom_layernorm: bool = True,   # LIMU-BERT: gamma/beta
    unfreeze_other_norms: bool = True,        # BatchNorm/GroupNorm (PRIMUS sometimes)

    # always hard-freeze these subtrees by name (override per model below)
    hard_freeze_keywords: Tuple[str, ...] = (),
) -> None:
    """
    Freezes everything, then unfreezes:
      - nn.LayerNorm (weight/bias)
      - custom LayerNorm gamma/beta (LIMU-BERT)
      - BatchNorm/GroupNorm (optional; useful for PRIMUS encoders)
      - small proj/adapters (optional; name-based)

    Finally, hard-freezes any module subtree whose module_name contains hard_freeze_keywords.
    """
    def _set_direct_params(m: nn.Module, flag: bool) -> None:
        for p in m.parameters(recurse=False):
            p.requires_grad = flag

    def _freeze_subtree(m: nn.Module) -> None:
        for p in m.parameters(recurse=True):
            p.requires_grad = False

    # 0) Freeze all
    for p in model.parameters():
        p.requires_grad = False

    other_norm_layers = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)
    # 1) Unfreeze norms + optional proj/adapters
    for module_name, m in model.named_modules():
        if unfreeze_pytorch_layernorm and isinstance(m, nn.LayerNorm):
            _set_direct_params(m, True)

        if unfreeze_other_norms and isinstance(m, other_norm_layers):
            _set_direct_params(m, True)

        if unfreeze_custom_layernorm:
            gamma = getattr(m, "gamma", None)
            beta = getattr(m, "beta", None)
            if isinstance(gamma, torch.nn.Parameter):
                gamma.requires_grad = True
            if isinstance(beta, torch.nn.Parameter):
                beta.requires_grad = True

        proj_check = any(k in module_name for k in proj_keywords)
        if unfreeze_proj and proj_keywords and proj_check:
            _set_direct_params(m, True)

    # 2) Hard-freeze “never adapt” subtrees (overrides accidental unfreeze)
    if hard_freeze_keywords:
        for module_name, m in model.named_modules():
            if any(k in module_name for k in hard_freeze_keywords):
                _freeze_subtree(m)


def trainable_params(model: nn.Module) -> Iterable[nn.Parameter]:
    return (p for p in model.parameters() if p.requires_grad)

def _fallback_unfreeze_norm_like_params(model: nn.Module) -> int:
    """
    Best-effort fallback for backbones whose norm/proj module names differ from
    our model-specific freeze presets. Keeps flow/heads frozen.
    """
    blocked = ("flow", "clf_head", "_cmt_rr_head", "head")
    unfrozen = 0
    for name, p in model.named_parameters():
        lname = name.lower()
        if any(k in lname for k in blocked):
            continue
        if ("norm" in lname) or (".ln" in lname) or ("layernorm" in lname):
            p.requires_grad = True
            unfrozen += int(p.numel())
    return unfrozen


def assert_flow_frozen(flow: Optional[nn.Module]) -> None:
    if flow is None:
        return
    for p in flow.parameters():
        assert p.requires_grad is False, "Flow parameters must be frozen for flow-TTT."

def freeze_all_but_layernorm_and_proj(model):
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            for p in m.parameters():
                p.requires_grad = True
    if hasattr(model, 'spec_proj') and model.spec_proj is not None:
        for p in model.spec_proj.parameters():
            p.requires_grad = True

# ---------------------------------------------------------------------
# Spectral loss and band-energy target (time-domain chest input)
# ---------------------------------------------------------------------
def spectral_loss(
    y_hat,
    y_true,
    n_fft=256,
    hop_length=64,
    win_length=256,
    power=1.0,
):
    """
    y_hat, y_true: (B, T)
    Returns L1 loss between magnitude STFTs.
    """
    # Sanitize inputs (if anything upstream produced NaN/Inf)
    y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=0.0, neginf=0.0)
    y_true = torch.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)

    window = torch.hann_window(win_length, device=y_hat.device)

    # torch.stft expects (..., T)
    Y_true = torch.stft(
        y_true,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=False,
    )
    Y_hat = torch.stft(
        y_hat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=False,
    )

    # Magnitude with epsilon to avoid weird 0**(negative) or Inf-Inf issues downstream
    mag_true = torch.sqrt((Y_true ** 2).sum(dim=-1) + 1e-8)  # (B, F, T_frames)
    mag_hat = torch.sqrt((Y_hat ** 2).sum(dim=-1) + 1e-8)

    loss = F.l1_loss(mag_hat, mag_true)

    if torch.isnan(loss):
        raise RuntimeError("NaN detected in spectral_loss")

    return loss

def compute_band_energy_targets(
    y_true,
    fs,
    n_fft=256,
    hop_length=64,
    win_length=256,
    band=(0.1, 0.6),  # Hz
):
    """
    y_true: (B, T)
    returns e_true: (B,) band energy in given frequency band
    """
    y_true = torch.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)

    window = torch.hann_window(win_length, device=y_true.device)
    Y_true = torch.stft(
        y_true,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=False,
    )  # (B, F, T_frames, 2)

    # magnitude squared
    mag2 = (Y_true ** 2).sum(dim=-1)  # (B, F, T_frames)
    mag2 = torch.nan_to_num(mag2, nan=0.0, posinf=0.0, neginf=0.0)

    # Frequency axis in Hz
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / fs).to(y_true.device)  # (F_rfft,)

    f_min, f_max = band
    band_mask = (freqs >= f_min) & (freqs <= f_max)

    # If band_mask is all False, avoid empty indexing
    if not band_mask.any():
        return mag2.new_zeros(mag2.shape[0])

    # Apply mask and sum energy over freq+time
    mag2_band = mag2[:, band_mask, :]  # (B, F_band, T_frames)
    e_true = mag2_band.sum(dim=(1, 2))  # (B,)

    e_true = torch.nan_to_num(e_true, nan=0.0, posinf=0.0, neginf=0.0)
    return e_true

def get_hidden_mean(hidden):
    """
    Convert encoder hidden features to per-sample latent vectors (B, D).

    Accepts:
      - (B, T, D): mean-pool across time
      - (B, D): already pooled, returned as-is
      - (D,): single sample pooled feature, promoted to (1, D)
    """
    if hidden.ndim == 3:
        return hidden.mean(dim=1)
    if hidden.ndim == 2:
        return hidden
    if hidden.ndim == 1:
        return hidden.unsqueeze(0)
    raise ValueError(f"Unexpected hidden shape: {tuple(hidden.shape)}")


def get_model_sequence_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "sequence_features"):
        hidden = model.sequence_features(x)
        return hidden if hidden.ndim == 3 else hidden.unsqueeze(1)
    if hasattr(model, "forward_features"):
        hidden = model.forward_features(x)
        return hidden if hidden.ndim == 3 else hidden.unsqueeze(1)
    try:
        _, hidden = model(x, return_hidden=True)
    except Exception:
        out = model(x)
        if isinstance(out, (tuple, list)) and len(out) >= 4 and torch.is_tensor(out[-1]):
            hidden = out[-1]
        elif isinstance(out, (tuple, list)) and len(out) >= 2 and torch.is_tensor(out[-1]):
            hidden = out[-1]
        else:
            raise RuntimeError("Could not infer sequence features from model output")
    return hidden if hidden.ndim == 3 else hidden.unsqueeze(1)


def get_model_pooled_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "pooled_features"):
        return model.pooled_features(x)
    return get_hidden_mean(get_model_sequence_features(model, x))

# ---------------------------------------------------------------------
# Unsupervised TTT objective (classifier-based)
#   - entropy minimisation (Tent-style)
#   - consistency between original & augmented views
# ---------------------------------------------------------------------
def _classification_entropy(logits):
    """
    logits: (B, C)
    Returns mean prediction entropy H(p) = -sum p log p
    """
    p = F.softmax(logits, dim=-1)
    log_p = F.log_softmax(logits, dim=-1)
    ent = -(p * log_p).sum(dim=-1).mean()
    return ent


def _consistency_kl(logits1, logits2):
    """
    Symmetric KL between two predictive distributions.
    logits1, logits2: (B, C)
    """
    log_p1 = F.log_softmax(logits1, dim=-1)
    p1 = F.softmax(logits1, dim=-1)

    log_p2 = F.log_softmax(logits2, dim=-1)
    p2 = F.softmax(logits2, dim=-1)

    kl1 = F.kl_div(log_p1, p2, reduction='batchmean')
    kl2 = F.kl_div(log_p2, p1, reduction='batchmean')
    return 0.5 * (kl1 + kl2)


def _simple_imu_augment(x):
    """
    Very lightweight augmentation for TTT.
    x: (B, T, 6)
    """
    # if noise_std <= 0.0:
    #     return x
    # noise = noise_std * torch.randn_like(x)
    # return x + noise

    x_aug = x.permute(0, 2, 1).detach().clone()
    scale = torch.normal(
        mean=2.0,
        std=1.1,
        size=(x_aug.shape[0], x_aug.shape[2]),
        device=x_aug.device,
        dtype=x_aug.dtype,
    ).unsqueeze(1)
    x_aug = x_aug * scale
    x_aug = spectral_augment(
        x_aug,
        fs=IMU_FS,           # 120 Hz
        max_phase=np.pi/3,   # phase jitter up to ±90°
        max_shift_hz=0.7,    # small ±0.5 Hz frequency shift
        dim=-1,
        p_phase=0.5,
        p_shift=0.5,
    )
    x_aug = x_aug.permute(0, 2, 1)
    return x_aug

def _waveform_corr_loss(y_hat, y_prior, eps=1e-8):
    """
    1 - mean Pearson correlation between predicted and prior waveforms.
    y_hat, y_prior: (B, T)
    """
    # center
    y_hat = y_hat - y_hat.mean(dim=1, keepdim=True)
    y_prior = y_prior - y_prior.mean(dim=1, keepdim=True)

    num = (y_hat * y_prior).sum(dim=1)  # (B,)
    den = torch.sqrt(
        (y_hat ** 2).sum(dim=1) * (y_prior ** 2).sum(dim=1) + eps
    )
    r = num / (den + eps)               # (B,)
    return 1.0 - r.mean()

def style_shift_imu(
    x,  # (B,T,C)
    gain_range=(0.7, 1.3),
    shift_max=24,          # samples @120Hz
    drift_max=0.12,        # * std
    noise_max=0.05,        # * std
    leak_max=0.05,
):
    B, T, C = x.shape
    device = x.device
    dtype = x.dtype

    y = x

    # time shift per channel
    if shift_max > 0:
        shifts = torch.randint(-shift_max, shift_max + 1, (B, C), device=device)
        t = torch.arange(T, device=device).view(1, T, 1)     # (1,T,1)
        s = shifts.view(B, 1, C)                             # (B,1,C)
        idx = (t - s) % T                                    # (B,T,C)
        y = torch.gather(y, dim=1, index=idx.long())

    # per-channel gain
    g = torch.empty((B, 1, C), device=device, dtype=dtype).uniform_(*gain_range)
    y = y * g

    # stats per channel
    std = y.std(dim=1, keepdim=True).clamp_min(1e-6)         # (B,1,C)

    # baseline drift: smoothed random walk
    if drift_max > 0:
        drift_strength = torch.empty((B, 1, C), device=device, dtype=dtype).uniform_(0.0, drift_max) * std
        rw = torch.randn((B, T, C), device=device, dtype=dtype).cumsum(dim=1)
        rw = rw / rw.std(dim=1, keepdim=True).clamp_min(1e-6)
        # heavy smoothing with avgpool
        rw = F.avg_pool1d(rw.transpose(1,2), kernel_size=101, stride=1, padding=50).transpose(1,2)
        rw = rw / rw.std(dim=1, keepdim=True).clamp_min(1e-6)
        y = y + rw * drift_strength

    # colored noise: low-passed white noise
    if noise_max > 0:
        noise_strength = torch.empty((B, 1, C), device=device, dtype=dtype).uniform_(0.0, noise_max) * std
        n = torch.randn((B, T, C), device=device, dtype=dtype)
        n = F.avg_pool1d(n.transpose(1,2), kernel_size=33, stride=1, padding=16).transpose(1,2)
        n = n / n.std(dim=1, keepdim=True).clamp_min(1e-6)
        y = y + n * noise_strength

    # tiny cross-channel leakage
    if C > 1 and leak_max > 0:
        leak = torch.empty((B, C, C), device=device, dtype=dtype).uniform_(0.0, leak_max)
        leak = leak * (1.0 - torch.eye(C, device=device, dtype=dtype).view(1, C, C))
        leak = leak / leak.sum(dim=-1, keepdim=True).clamp_min(1.0)
        y = y + torch.einsum("bcc,btc->btc", leak, y)

    return y


def make_style_views(x, K=4, **kwargs):
    # view0 is identity
    views = [x]
    for _ in range(K - 1):
        views.append(style_shift_imu(x, **kwargs))
    return torch.stack(views, dim=0)  # (K,B,T,C)


def emb_consistency_loss(h_means):  # list of (B,D)
    # anchor at view0
    e0 = F.normalize(h_means[0], dim=-1)
    loss = 0.0
    for k in range(1, len(h_means)):
        ek = F.normalize(h_means[k], dim=-1)
        loss = loss + F.mse_loss(ek, e0)
    return loss / max(1, (len(h_means) - 1))


def _build_rr_prior_waveform(
    cls_logits,
    T_chest,
    med_bpm_range=(4.0, 7.0),
    rest_bpm_range=(12.0, 20.0),
    med_class_idx=1,
    rest_class_idx=0,
    fs=BR_FS,
):
    """
    Build an expected chest waveform prior from classifier logits as a
    probabilistic combination of class-specific prototype waves.

    cls_logits: (B, C)
    T_chest:    int, number of chest samples
    Returns:
        y_prior: (B, T_chest) waveform prior

    Assumes:
        - 'meditation' class is at med_class_idx
        - 'rest' class is at rest_class_idx
        - Meditation RR in [4,7] bpm
        - Rest RR in [12,20] bpm

    We construct:
        y_prior = p_med * y_med + p_rest * y_rest
    where y_med, y_rest are fixed prototype waves and p_* are classifier
    probabilities (renormalised on med+rest).
    """
    B, C = cls_logits.shape

    # soft class probabilities (no gradients into classifier)
    probs = F.softmax(cls_logits.detach(), dim=-1)  # (B, C)

    # if fewer than 2 classes, we can't build med/rest prior
    if C < 2:
        # default to a flat zero prior
        return cls_logits.new_zeros(B, T_chest)

    # probabilities for meditation and rest
    p_med = probs[:, med_class_idx]   # (B,)
    p_rest = probs[:, rest_class_idx] # (B,)

    # renormalise on med+rest in case of extra classes
    p_sum = (p_med + p_rest).clamp(min=1e-6)
    p_med = p_med / p_sum
    p_rest = p_rest / p_sum

    # centre BPMs for each class
    med_bpm_mean = 0.5 * (med_bpm_range[0] + med_bpm_range[1])    # ~5.5 bpm
    rest_bpm_mean = 0.5 * (rest_bpm_range[0] + rest_bpm_range[1]) # ~16 bpm

    med_freq_hz = med_bpm_mean / 60.0
    rest_freq_hz = rest_bpm_mean / 60.0

    # time axis in seconds
    t = torch.arange(
        T_chest,
        device=cls_logits.device,
        dtype=torch.float32,
    ) / float(fs)                           # (T,)
    t = t.unsqueeze(0)                      # (1, T)

    # class-specific prototype waves (shared across batch)
    y_med_proto = torch.sin(2.0 * math.pi * med_freq_hz * t)   # (1, T)
    y_rest_proto = torch.sin(2.0 * math.pi * rest_freq_hz * t) # (1, T)

    # broadcast prototypes to batch
    y_med = y_med_proto.expand(B, -1)       # (B, T)
    y_rest = y_rest_proto.expand(B, -1)     # (B, T)

    # probabilistic mixture based on classifier outputs
    y_prior = p_med.unsqueeze(1) * y_med + p_rest.unsqueeze(1) * y_rest  # (B, T_chest)

    return y_prior


@torch.no_grad()
def compute_source_ic_and_rr(model, dataloader, device, max_batches=None):
    model.eval()
    U_list, y_list = [], []

    for bi, batch in enumerate(dataloader):
        # Adjust to your dataloader tuple shape:
        imu, pss, cond, rr = batch
        imu = imu.float().to(device)
        rr = rr.float().to(device)

        z = get_model_pooled_features(model, imu)              # (B, D)
        u = flow_to_u(model.flow, z)          # (B, D)

        U_list.append(u.detach().cpu())
        y_list.append(rr.detach().cpu())

        if (max_batches is not None) and (bi + 1 >= max_batches):
            break

    U = torch.cat(U_list, dim=0)               # (N, D)
    y = torch.cat(y_list, dim=0).view(-1)      # (N,)
    return U, y


def pick_causal_ic_dims(U, y, top_k=16, ridge=1e-3):
    """
    Simple ridge regression to score IC dims by |weight|.
    """
    # Standardize
    Uc = (U - U.mean(0, keepdim=True)) / (U.std(0, keepdim=True).clamp_min(1e-6))
    yc = y - y.mean()

    # Ridge closed form: w = (U^T U + λI)^{-1} U^T y
    D = Uc.shape[1]
    A = Uc.T @ Uc + ridge * torch.eye(D)
    b = Uc.T @ yc
    w = torch.linalg.solve(A, b)               # (D,)

    idx = torch.topk(w.abs(), k=min(top_k, D)).indices
    idx = idx.sort().values
    return idx, w

@torch.no_grad()
def estimate_target_nuisance_stats(model, fewshot_loader, device, causal_idx):
    model.eval()
    U_list = []

    for batch in fewshot_loader:
        imu, rr, *_ = batch
        imu = imu.float().to(device)

        z = get_model_pooled_features(model, imu)
        u = flow_to_u(model.flow, z)        # (B, D)
        U_list.append(u.detach())

    U_t = torch.cat(U_list, dim=0)             # (K, D)

    D = U_t.shape[1]
    mask = torch.ones(D, dtype=torch.bool, device=U_t.device)
    mask[causal_idx.to(U_t.device)] = False
    U_nuis = U_t[:, mask]                      # (K, Dn)

    mu = U_nuis.mean(0)
    std = U_nuis.std(0).clamp_min(1e-6)
    return mask, mu, std

def synthesize_cmt_pairs(U_src, y_src, nuisance_mask, mu_t, std_t, n_aug=4096, seed=0):
    """
    U_src: (N, D) source ICs
    y_src: (N,) source labels
    nuisance_mask: (D,) bool mask True for nuisance dims
    mu_t, std_t: (Dn,) nuisance stats from target few-shot
    """
    g = torch.Generator().manual_seed(seed)

    N, D = U_src.shape
    idx = torch.randint(low=0, high=N, size=(n_aug,), generator=g)

    u_base = U_src[idx].clone()                # (n_aug, D)
    y_aug = y_src[idx].clone()                 # (n_aug,)

    Dn = nuisance_mask.sum().item()
    noise = torch.randn(n_aug, Dn, generator=g) * std_t.unsqueeze(0) + mu_t.unsqueeze(0)

    u_base[:, nuisance_mask] = noise
    return u_base, y_aug



def test_time_train_unsupervised(
    model,
    test_dataloader,
    device,
    ttt_lr=1e-4,
    steps_per_batch=1,
    lambda_entropy=1.0,
    lambda_consistency=1.0,
):
    """
    True unsupervised TTT:
      - no chest or label usage
      - adapts model using classifier head only
      - objective = entropy minimisation + consistency on augmented views

    IMPORTANT: expects model(imu) -> (y_hat_frames, band_pred, cls_logits)
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=ttt_lr)

    for imu, _, _, _ in test_dataloader:
        imu = imu.float().to(device)     # (B, T_imu, 6)

        for _ in range(steps_per_batch):
            optimizer.zero_grad()

            # original view
            _, _, logits, _ = model(imu)
            if logits is None:
                # no classifier head: nothing to adapt on
                return

            # augmented view
            imu_aug = _simple_imu_augment(imu)
            _, _, logits_aug, _ = model(imu_aug)

            loss_ent = _classification_entropy(logits)
            loss_cons = _consistency_kl(logits, logits_aug)

            loss = lambda_entropy * loss_ent + lambda_consistency * loss_cons

            if torch.isnan(loss) or torch.isinf(loss):
                print("[TTT] NaN/Inf encountered, stopping TTT for this subject.")
                return

            loss.backward()
            optimizer.step()

# ---------------------------------------------------------------------
# TTTFlow: unsupervised TTT using flow likelihood on h_mean
# ---------------------------------------------------------------------
def collect_trainable_params(*modules):
    seen = set()
    params = []
    for m in modules:
        if m is None:
            continue
        for p in m.parameters():
            if p.requires_grad and id(p) not in seen:
                params.append(p)
                seen.add(id(p))
    return params

def _ae_canonicalize(x_btc: torch.Tensor, autoencoder) -> torch.Tensor:
    """
    x_btc: (B, T, C)
    returns: (B, T, C) after AE reconstruction
    """
    # AE expects (B, C, T)
    x_bct = x_btc.permute(0, 2, 1).contiguous()
    y_bct = autoencoder(x_bct)
    y_btc = y_bct.permute(0, 2, 1).contiguous()
    return y_btc

def test_time_train_flow(
    model,
    test_dataloader,
    device,
    ttt_lr=1e-4,
    steps_per_batch=1,
    temporal_encoder=None,
    use_rr_prior=False,
    rr_lambda=1e-3,
    med_class_idx=1,
    rest_class_idx=0,
    use_style_views=True,
    style_k=4,
    beta_cons=0.1,
    autoencoder=None,
    ae_nll_lambda=1.0,
    ae_align_lambda=0.1,
    ae_detach_target=True,
    adapt_autoencoder=False,
    ssa_stats:dict=None,
    ssa_lambda=0.0,
    fewshot_loader=None,
    use_cmt=False,
    cmt_top_k=16,
    cmt_n_aug=4096,
    cmt_steps=50,
    cmt_lr=1e-4,
    cmt_lambda=1.0,
    log_every:int=10,
    recorder: Optional[ExperimentRecorder]=None,
    sbj:str="",
    step_offset:int=None,
    prefer_rr_head=True,
    mdl_str: str = "timesnet",
    low_mem: bool = False,
    use_amp: bool = True,
):
    """
    Unsupervised Test-Time Training with normalizing flow (TTTFlow).

    - Uses flow log-likelihood on:
        * h_mean  (spectral / ViT view)
        * h_time  (temporal encoder view, if provided)
      with a simple average of their log-probs.
    - Optionally adds a classifier-informed waveform prior loss:
        * build expected chest waveform from classifier probs using
          meditation RR in [4,7] bpm and rest RR in [12,20] bpm.
        * encourage predicted chest waveform to correlate with this prior.
    - Expects model.flow to be a SimpleRealNVPFlow on d_model-dim features.
    - Flow and classifier parameters are frozen; backbone (and temporal encoder)
      adapt.
    """
    if model.flow is None:
        print("[TTTFlow] No flow in model; skipping.")
        return

    # Freeze flow/classifier and adapt only model-appropriate norm/proj params.
    freeze_kwargs = mdl_freeze_kwargs.get(str(mdl_str).lower(), _timesnet_freeze_kwargs)
    freeze_all_but_norm_and_proj(model, **freeze_kwargs)
    for p in model.flow.parameters(): p.requires_grad = False
    if hasattr(model, 'clf_head') and model.clf_head is not None:
        for p in model.clf_head.parameters(): p.requires_grad = False

    if autoencoder is not None:
        autoencoder = autoencoder.to(device)

    params = collect_trainable_params(model)
    if len(params) == 0:
        fallback_count = _fallback_unfreeze_norm_like_params(model)
        params = collect_trainable_params(model)
        print(
            f"[TTTFlow] trainable-params fallback engaged: "
            f"norm_like_params={fallback_count}, groups={len(params)}"
        )

    if temporal_encoder is not None:
        params = collect_trainable_params(model, temporal_encoder)

    if (autoencoder is not None) and adapt_autoencoder:
        params = collect_trainable_params(model, temporal_encoder, autoencoder)

    if len(params) == 0:
        raise RuntimeError(
            "[TTTFlow] No trainable parameters available after freeze policy; "
            "cannot run gradient-based TTT."
        )

    optimizer = torch.optim.Adam(params, lr=ttt_lr)

    # step counter for logging
    global_step = int(step_offset or 0)

    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and "cuda" in device)
    for imu, chest, conds, br in test_dataloader:
        imu = imu.float().to(device)
        chest = chest.float().to(device)
        br = br.float().to(device) if br is not None else None

        # infer expected chest length from IMU length and sampling rates
        B, T_imu, _ = imu.shape
        T_chest = int(round(T_imu * BR_FS / IMU_FS))

        for _ in range(steps_per_batch):
            # snapshot training parameters
            prev = snapshot_trainable_params(model)

            optimizer.zero_grad()

            views = make_style_views(
                imu, K=style_k, shift_max=24, leak_max=0.05
            ) if use_style_views else imu.unsqueeze(0)

            loss = torch.tensor(0.0, device=device)

            cls_logits = None
            y_hat_ks = [] if not low_mem else None
            h_means = [] if not low_mem else None
            nlls_spec = [] if not low_mem else None
            nlls_ae = [] if not low_mem else None
            align_losses = [] if not low_mem else None
            ssa_losses = [] if ((ssa_stats is not None and ssa_lambda > 0.0) and (not low_mem)) else None

            flow_nll_acc = 0.0 if low_mem else None
            for k in range(views.size(0)):
                x_k = views[k]  # (B, T, C)
                nll_ae = None
                align = None

                # --- spec / raw view ---
                with torch.cuda.amp.autocast(
                    enabled=use_amp and "cuda" in device):
                    cls_k = None
                    out_k = model(x_k)
                    if isinstance(out_k, (tuple, list)):
                        y_hat_k = out_k[0]
                        if len(out_k) >= 3:
                            cls_k = out_k[2]
                    else:
                        y_hat_k = out_k
                z_spec = get_model_pooled_features(model, x_k)              # (B, d_model)

                if y_hat_ks is not None:
                    y_hat_ks.append(y_hat_k)

                if ssa_losses is not None:
                    # directly from ViT model outputs
                    ssa_losses.append(ssa_moment_loss(z_spec, ssa_stats))


                if h_means is not None:
                    h_means.append(z_spec)
                nll_spec = (-model.flow.log_prob(z_spec)).mean()
                if nlls_spec is not None:
                    nlls_spec.append(nll_spec)
                if flow_nll_acc is not None:
                    flow_nll_acc += float(nll_spec.detach().cpu().item())

                # Keep canonical outputs for RR prior below
                if k == 0:
                    y_hat_frames_spec = y_hat_k
                    if use_rr_prior:
                        cls_logits = cls_k

                # --- AE-canonicalized view (optional) ---
                if autoencoder is not None:
                    with torch.set_grad_enabled(adapt_autoencoder):
                        x_ae = _ae_canonicalize(x_k, autoencoder)

                    with torch.cuda.amp.autocast(
                        enabled=use_amp and "cuda" in device):
                        out_ae = model(x_ae)
                        y_hat_ae = out_ae[0] if isinstance(out_ae, (tuple, list)) else out_ae

                    z_ae = get_model_pooled_features(model, x_ae)           # (B, d_model)

                    nll_ae = (-model.flow.log_prob(z_ae)).mean()
                    if nlls_ae is not None:
                        nlls_ae.append(nll_ae)

                    # alignment: bridge spec latent toward AE-view latent
                    target = z_ae.detach() if ae_detach_target else z_ae
                    align = F.mse_loss(z_spec, target)
                    if align_losses is not None:
                        align_losses.append(align)

                if low_mem:
                    loss_k = nll_spec
                    if autoencoder is not None and nll_ae is not None:
                        loss_k = loss_k + ae_nll_lambda * nll_ae
                        loss_k = loss_k + ae_align_lambda * align
                    if use_rr_prior and (k == 0) and (cls_logits is not None):
                        if y_hat_frames_spec.size(1) != T_chest:
                            y_hat = F.interpolate(
                                y_hat_frames_spec.unsqueeze(1),
                                size=T_chest,
                                mode="linear",
                                align_corners=False,
                            ).squeeze(1)
                        else:
                            y_hat = y_hat_frames_spec
                        y_prior = _build_rr_prior_waveform(
                            cls_logits,
                            T_chest,
                            med_bpm_range=(4.0, 7.0),
                            rest_bpm_range=(12.0, 20.0),
                            med_class_idx=med_class_idx,
                            rest_class_idx=rest_class_idx,
                            fs=BR_FS,
                        )
                        prior_loss = _waveform_corr_loss(y_hat, y_prior)
                        loss_k = loss_k + rr_lambda * prior_loss
                    # average over views for comparable scale
                    loss_k = loss_k / views.size(0)
                    if not loss_k.requires_grad:
                        raise RuntimeError(
                            "[TTTFlow] Detached low_mem loss: no gradient path from "
                            "loss to any trainable parameter."
                        )
                    scaler.scale(loss_k).backward()

            if not low_mem:
                # --- aggregate losses ---
                flow_loss = torch.stack(nlls_spec).mean()
                loss += flow_loss

                if autoencoder is not None and len(nlls_ae) > 0:
                    loss = loss + ae_nll_lambda * torch.stack(nlls_ae).mean()
                    loss = loss + ae_align_lambda * torch.stack(align_losses).mean()

                # your existing embedding consistency over style views
                if use_style_views and style_k > 1:
                    loss = loss + beta_cons * emb_consistency_loss(h_means)

                if ssa_losses is not None and len(ssa_losses) > 0:
                    loss = loss + ssa_lambda * torch.stack(ssa_losses).mean()

                # RR prior term: only if classifier head exists and we enabled it
                if use_rr_prior and (cls_logits is not None):
                    # decode spectral chest prediction to chest time grid
                    if y_hat_frames_spec.size(1) != T_chest:
                        y_hat = F.interpolate(
                            y_hat_frames_spec.unsqueeze(1),
                            size=T_chest,
                            mode="linear",
                            align_corners=False,
                        ).squeeze(1)
                    else:
                        y_hat = y_hat_frames_spec

                    # build classifier-based waveform prior
                    y_prior = _build_rr_prior_waveform(
                        cls_logits,
                        T_chest,
                        med_bpm_range=(4.0, 7.0),
                        rest_bpm_range=(12.0, 20.0),
                        med_class_idx=med_class_idx,
                        rest_class_idx=rest_class_idx,
                        fs=BR_FS,
                    )

                    prior_loss = _waveform_corr_loss(y_hat, y_prior)
                    loss = loss + rr_lambda * prior_loss

            if torch.isnan(loss) or torch.isinf(loss):
                print("[TTTFlow] NaN/Inf encountered, stopping TTTFlow for this subject.")
                return
            if not low_mem:
                if not loss.requires_grad:
                    raise RuntimeError(
                        "[TTTFlow] Detached aggregated loss: no gradient path from "
                        "loss to any trainable parameter."
                    )
                scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # step-wise logging
            param_norm = param_update_norm(model, prev)
            flow_nll = float(flow_loss.detach().cpu().item()) if not low_mem else (flow_nll_acc / max(1, views.size(0)))
            pred_uncertainty = prediction_uncertainty_std(y_hat_ks) if y_hat_ks is not None else None
            pred_entropy = prediction_entropy_from_logits(cls_logits) \
                    if cls_logits is not None else None
            ssa_loss_val = None
            if ssa_losses is not None and len(ssa_losses) > 0:
                ssa_loss_val = float(torch.stack(ssa_losses).mean().detach().cpu().item())

            # compute quick TES metrics on current batch (view0)
            with torch.no_grad():
                # y_hat_frames_spec is on STFT grid; interpolate to chest grid for TES metrics
                if y_hat_frames_spec.size(1) != chest.size(1):
                    y_hat_chest = F.interpolate(
                        y_hat_frames_spec.unsqueeze(1),
                        size=chest.size(1),
                        mode="linear",
                        align_corners=False,
                    ).squeeze(1)
                else:
                    y_hat_chest = y_hat_frames_spec
                tes_mae = float((y_hat_chest - chest).abs().mean().detach().cpu().item())
                tes_rmse = float(torch.sqrt(((y_hat_chest - chest) ** 2).mean()).detach().cpu().item())

            rr_metrics = None
            if (log_every is not None) and (log_every > 0) and (global_step % log_every == 0):
                # RR metrics: prefer RR head (same model) if available; fallback to RR-from-chest
                with torch.no_grad():
                    rr_true = None
                    if br is not None and br.numel() == B:
                        rr_true = br.detach().cpu().numpy()
                    else:
                        # derive "true RR" from chest labels
                        y_true_np = chest.detach().cpu().numpy()
                        rr_true = np.array([get_max_freq(win, fs=BR_FS) * 60.0 for win in y_true_np], dtype=np.float32)

                    rr_pred = None
                    if prefer_rr_head and hasattr(model, "rr_head") and (model.rr_head is not None):
                        # direct RR from pooled latent (scalar)
                        rr_pred_t = model.forward_rr(imu)  # (B,)
                        rr_pred = rr_pred_t.detach().cpu().numpy()
                    else:
                        # fallback: derive RR from predicted chest
                        y_pred_np = y_hat_chest.detach().cpu().numpy()
                        rr_pred = np.array([get_max_freq(win, fs=BR_FS) * 60.0 for win in y_pred_np], dtype=np.float32)

                    rr_mae = float(np.mean(np.abs(rr_pred - rr_true)))
                    rr_rmse = float(np.sqrt(np.mean((rr_pred - rr_true) ** 2)))
                    rr_metrics = {"mae": rr_mae, "rmse": rr_rmse}

            if recorder is not None:
                recorder.log_ttt_step(
                    subject=sbj,
                    step=global_step,
                    flow_nll=flow_nll,
                    tes={"mae": tes_mae, "rmse": tes_rmse},
                    rr=rr_metrics,
                    param_norm=param_norm,
                    pred_entropy=pred_entropy,
                    pred_uncertainty_std=pred_uncertainty,
                    ssa_loss=ssa_loss_val,
                    cmt_enabled=bool(use_cmt),
                    shots=None,
                )

            global_step += 1


def compute_flow_loglik_stats(model, dataloader, device):
    """
    Compute mean/std of flow log-likelihood on h_mean for a given dataloader.

    Returns:
        stats: dict with keys 'mean', 'std', 'n' or None if no data / no flow.
    """
    if model.flow is None:
        return None

    model.eval()
    all_logp = []

    with torch.no_grad():
        for imu, _, _, _ in dataloader:
            imu = imu.float().to(device)
            # forward to get pooled features
            h_mean = get_model_pooled_features(model, imu)
            log_pz = model.flow.log_prob(h_mean)  # (B,)
            all_logp.append(log_pz.detach().cpu().numpy())

    if len(all_logp) == 0:
        return None

    logp = np.concatenate(all_logp, axis=0)
    stats = {
        "mean": float(logp.mean()),
        "std": float(logp.std()),
        "n": int(logp.shape[0]),
    }
    return stats

def compute_ssa_stats(
    model,
    dataloader,
    device,
    r: int = 32,
    max_batches: int | None = None,
):
    """
    Compute Significant-Subspace Alignment (SSA) reference stats from SOURCE (train) data.

    Returns a dict with:
      - mu:   (D,) mean feature
      - U:    (D, r) top-r PCA basis (orthonormal columns)
      - m_a:  (r,) mean of projected coeffs
      - v_a:  (r,) var  of projected coeffs
      - r:    int
    """
    if dataloader is None:
        return None

    model.eval()
    feats = []

    with torch.no_grad():
        for bi, (imu, _, _, _) in enumerate(dataloader):
            imu = imu.float().to(device)
            z = get_model_pooled_features(model, imu)                # (B, D)
            feats.append(z.detach().float().cpu())
            if (max_batches is not None) and (bi + 1 >= max_batches):
                break

    if len(feats) == 0:
        return None

    X = torch.cat(feats, dim=0)                        # (N, D)
    mu = X.mean(dim=0, keepdim=True)                   # (1, D)
    Xc = X - mu                                        # (N, D)

    # PCA via low-rank SVD
    # U: (D, r), S: (r,), V: (N, r)  (torch returns V as V, not V^T)
    # We'll use V basis for projection.
    q = min(r, Xc.shape[0], Xc.shape[1])
    U, S, V = torch.pca_lowrank(Xc, q=q, center=False)
    V = V[:, :q].contiguous()                          # (D, q)

    # Project source features into subspace and compute moments
    A = (Xc @ V)                                       # (N, q)
    m_a = A.mean(dim=0)                                # (q,)
    v_a = A.var(dim=0, unbiased=False).clamp_min(1e-8) # (q,)

    stats = {
        "mu":  mu.squeeze(0),                          # (D,)
        "V":   V,                                      # (D, q)
        "m_a": m_a,                                    # (q,)
        "v_a": v_a,                                    # (q,)
        "r":   int(q),
    }
    return stats


def ssa_moment_loss(z: torch.Tensor, ssa_stats: dict):
    """
    z: (B, D) test batch features
    ssa_stats: dict from compute_ssa_stats
    Loss = MSE(mean_proj, mean_src) + MSE(var_proj, var_src) in top-r PCA subspace
    """
    mu = ssa_stats["mu"].to(z.device, z.dtype)          # (D,)
    V  = ssa_stats["V"].to(z.device, z.dtype)           # (D, r)
    m0 = ssa_stats["m_a"].to(z.device, z.dtype)         # (r,)
    v0 = ssa_stats["v_a"].to(z.device, z.dtype)         # (r,)

    zc = z - mu.unsqueeze(0)                            # (B, D)
    a  = zc @ V                                         # (B, r)

    m = a.mean(dim=0)                                   # (r,)
    v = a.var(dim=0, unbiased=False).clamp_min(1e-8)    # (r,)

    return F.mse_loss(m, m0) + F.mse_loss(v, v0)


def flow_aligned_multiview_loss(
    z_spec,
    z_time,
    flow: SimpleRealNVPFlow,
    lambda_match: float = 1.0,
):
    """
    Flow-aligned multiview loss:

      L = 0.5 * [ NLL(z_spec) + NLL(z_time) ] + lambda_match * ||z_spec - z_time||^2

    where NLL(z) = -log p_flow(z).

    z_spec: (B, D) spectral ViT pooled features (h_mean)
    z_time: (B, D) temporal encoder features
    flow:   normalizing flow defined on R^D
    """
    # log p for each view
    log_p_spec = flow.log_prob(z_spec)   # (B,)
    log_p_time = flow.log_prob(z_time)   # (B,)

    # average negative log-likelihood for both
    nll_spec = -log_p_spec.mean()
    nll_time = -log_p_time.mean()
    nll = 0.5 * (nll_spec + nll_time)

    # match the two views in feature space
    match = F.mse_loss(z_spec, z_time)

    return nll + lambda_match * match

def train_cmt_rr_head(
    model,
    train_dataloader,
    fewshot_loader,
    device,
    *,
    top_k_ic: int = 16,
    n_aug: int = 4096,
    steps: int = 50,
    lr: float = 1e-4,
    lambda_cmt: float = 1.0,
    max_src_batches: int = 50,
    ridge: float = 1e-3,
    seed: int = 0,
    attach_to_model: bool = True,
):
    """
    Few-shot domain adaptation by causal mechanism transfer (feature-level CMT)
    implemented on top of your existing ViT latent + RealNVP flow.

    Trains a small RR head on synthetic target-like latents generated by:
      1) Fit IC-space 'causal' dims on source (train fold)
      2) Estimate target nuisance stats from few-shot target
      3) Swap nuisance components into source ICs
      4) Map ICs back to latent z via inverse flow
      5) Train a small head z -> RR

    Returns:
      rr_head (nn.Module) or None
    """
    if (fewshot_loader is None) or (model.flow is None):
        return None

    model.eval()

    # ---- 1) source ICs + labels
    U_src, y_src = compute_source_ic_and_rr(
        model, train_dataloader, device, max_batches=max_src_batches
    )  # U_src: (N,D) on CPU, y_src: (N,) CPU

    causal_idx, _ = pick_causal_ic_dims(
        U_src, y_src, top_k=top_k_ic, ridge=ridge
    )

    # ---- 2) target nuisance stats from few-shot
    nuisance_mask, mu_t, std_t = estimate_target_nuisance_stats(
        model, fewshot_loader, device, causal_idx
    )
    # move stats to CPU for synth (keeps it simple)
    nuisance_mask_cpu = nuisance_mask.detach().cpu()
    mu_t_cpu = mu_t.detach().cpu()
    std_t_cpu = std_t.detach().cpu()

    # ---- 3) synth IC/label pairs
    u_aug, y_aug = synthesize_cmt_pairs(
        U_src, y_src, nuisance_mask_cpu, mu_t_cpu, std_t_cpu,
        n_aug=n_aug, seed=seed
    )
    u_aug = u_aug.to(device)
    y_aug = y_aug.to(device)

    # ---- 4) map ICs back to z using inverse flow
    with torch.no_grad():
        z_aug = flow_from_u(model.flow, u_aug)  # (n_aug, D)

    # ---- 5) train head on z_aug -> RR
    rr_head = torch.nn.Linear(z_aug.shape[1], 1).to(device)
    rr_head.train()

    opt = torch.optim.Adam(rr_head.parameters(), lr=lr)
    loss_curve = []

    for _ in range(steps):
        pred = rr_head(z_aug).view(-1)
        loss = torch.nn.functional.mse_loss(pred, y_aug) * lambda_cmt
        loss_curve.append(float(loss.detach().cpu().item()))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    if attach_to_model:
        model._cmt_rr_head = rr_head

    cmt_log = {
        "causal_idx": causal_idx.detach().cpu(),
        "nuisance_mask": nuisance_mask_cpu,
        "mu_t": mu_t_cpu,
        "std_t": std_t_cpu,
        "loss_curve": loss_curve,
        "top_k_ic": int(top_k_ic),
        "n_aug": int(n_aug),
        "steps": int(steps),
        "lr": float(lr),
        "lambda_cmt": float(lambda_cmt),
    }

    return rr_head, cmt_log



def get_labels_and_preds(model, train_dataloader, test_dataloader):
    x_train_ex, y_train_ex = [], []
    x_test_ex,  y_test_ex  = [], []
    cond_test_all = []
    cond_train_all = []

    model.eval()
    with torch.no_grad():
        for imu, chest, conds, br in train_dataloader:
            imu = imu.float().to(device)
            chest = chest.float().to(device)

            try:
                y_hat_frames_spec, _, _, hidden = model(imu)
            except:
                y_hat_frames_spec, hidden = model(imu, return_hidden=True)

            if hidden.ndim > 2:
                h_mean = get_hidden_mean(hidden)
            else:
                h_mean = hidden

            x_train_ex.append(imu.cpu().numpy())
            y_train_ex.append(chest.cpu().numpy())
            cond_train_all.append(conds.cpu().numpy())
            # store only chest + prediction shapes for plotting
        x_train_ex = np.concatenate(x_train_ex, axis=0)
        y_train_ex = np.concatenate(y_train_ex, axis=0)

    # collect for plotting + eval (test)
    with torch.no_grad():
        for imu, chest, conds, br in test_dataloader:
            imu = imu.float().to(device)
            chest = chest.float().to(device)

            try:
                y_hat_frames_spec, _, _, hidden = model(imu)
            except:
                y_hat_frames, hidden = model(imu, return_hidden=True)

            if hidden.ndim > 2:
                h_mean = get_hidden_mean(hidden)
            else:
                h_mean = hidden

            # ---- CMT RR prediction (optional) ----
            if hasattr(model, "_cmt_rr_head"):
                rr_pred_cmt = model._cmt_rr_head(h_mean).view(-1)  # (B,)
            else:
                rr_pred_cmt = None

            x_test_ex.append(imu.cpu().numpy())
            y_test_ex.append(chest.cpu().numpy())

            y_test_all.append(chest.cpu().numpy())
            cond_test_all.append(conds.cpu().numpy())

            if rr_pred_cmt is not None:
                rr_preds_cmt.append(rr_pred_cmt.cpu())

    x_test_ex = np.concatenate(x_test_ex, axis=0)
    y_test_ex = np.concatenate(y_test_ex, axis=0)
    y_test_all = np.concatenate(y_test_all, axis=0)
    y_hat_test_all = np.concatenate(y_hat_test_all, axis=0)
    cond_test_all = np.concatenate(cond_test_all, axis=0)
    cond_train_all = np.concatenate(cond_train_all, axis=0)

    out = {'train_lbl': y_train_ex,
           'train_pred': y_train_ex,
           'train_cond': cond_train_all,
           'test_lbl': y_test_all,
           'test_pred': y_hat_test_all,
           'test_cond': cond_test_all}

    return out

def get_pretrained_model(mdl_str, cfg, d_model=128, device='cuda:0',
                         n_channels=6, window_size=2400,
                         limu_pretrain_ckpt=None):
    from times_experiment import build_times_model
    encoder_ckpt = limu_pretrain_ckpt if mdl_str == 'limu' else None
    return build_times_model(
        mdl_str,
        window_size=window_size,
        n_channels=n_channels,
        d_model=d_model,
        device=device,
        use_flow=False,
        encoder_init='external',
        encoder_ckpt=encoder_ckpt,
        extra={},
    )



def main():
    from times_experiment import make_cfg
    imu_issues = [17, 26, 30]
    subjects = [
        'S' + str(i).zfill(2)
        for i in range(12, 31)
        if i not in imu_issues
    ]
    strategy = 'times'
    data_str = 'imu_filt'
    device = 'cuda:1'
    debug = False
    freeze = True
    overwrite = False
    epochs = 100 if not debug else 2
    patch_size = int((120/18)*36)

    n_channels = 6 if data_str == 'imu_filt' else 2

    lr = 3e-4
    downstream_lr = 1e-6

    window_size = 20 * IMU_FS
    pred_len = int(window_size / IMU_FS) * BR_FS

    mdl_str = 'timesnet'
    y_str = 'br' # or 'br'

    sbj_processed_dir = SBJ_PROCESSED_DIR
    m_dir = M_DIR

    results_dir = join(sbj_processed_dir, mdl_str)
    plt_dir = join('plots', mdl_str)
    makedirs(results_dir, exist_ok=True)
    makedirs(plt_dir, exist_ok=True)

    prefix = None

    models = ['timesnet', 'primus', 'limu']

    m_dir = join(m_dir, data_str)
    if m_dir == SEAT_DATA_DIR:
        model_parent_directory = sep.join(m_dir.split(sep)[:-1]+['loocv'])
    else:
        model_parent_directory = join(m_dir, 'loocv')

    generator = loocv_generator(subjects,
                                data_str,
                                shuffle=False,
                                data_dir=sbj_processed_dir,
                                mdl_dir=model_parent_directory,
                                autoencoder=None)

    USE_FLOW = True
    TTT_FLOW_LR = 3e-5
    TTT_FLOW_STEPS_PER_BATCH = 1
    flow_num_layers = 4
    flow_dim_ff = 256

    FREEZE = False

    USE_SSA = False
    SSA_R = 32
    SSA_MAX_BATCHES = 50
    SSA_LAMBDA = 0.05

    USE_CMT = False
    CMT_K_SHOT = 8
    CMT_TOP_K_IC = 16
    CMT_N_AUG = 4096
    CMT_STEPS = 50
    CMT_LR = 1e-4

    for sbj, train_dataloader, val_dataloader, test_dataloader in generator:
        print(f"\n=== Subject {sbj} ===")
        pretrained_prefix = f'{sbj}_{data_str}_'

        prefix = f'{sbj}_'\
                f'flow_{int(USE_FLOW)}_'\
                f'freeze_{int(FREEZE)}_'\
                f'ssa_{int(USE_SSA)}_'\
                f'cmt_{int(USE_CMT)}_'

        # print(prefix)
        mdl_dir = join(model_parent_directory, sbj)

        # Set checkpoints and reference directories
        times_mdl_dir = join(mdl_dir, 'timesnet')
        times_ckpt = join(times_mdl_dir, pretrained_prefix+'ckp_last.pt')

        primus_mdl_dir = join(mdl_dir, 'primus')
        primus_ckpt = join(primus_mdl_dir, pretrained_prefix+'ckp_last.pt')

        limu_mdl_dir = join(mdl_dir, 'limu-bert')
        limu_ckpt = join(limu_mdl_dir, pretrained_prefix+'ckp_last.pt')
        limu_pretrain_ckpt = join(
            limu_mdl_dir, f"{sbj}_{data_str}_"+'limu_ssl_last.pt'
        )

        ckpt_fnames = {'timesnet': times_ckpt,
                       'primus': primus_ckpt,
                       'limu': limu_ckpt}

        for fname in ckpt_fnames.values():
            assert exists(fname), \
                    "Pretrain ckpt not complete for {}".format(fname)

        for mdl_str in models:

            if mdl_str == 'primus':
                d_model = 512
            else:
                d_model = 128

            cfg = make_cfg(window_size=window_size, n_channels=n_channels,
                           d_model=d_model, pred_len=pred_len)

            # A sensible starting point for patch models on IMU
            cfg.patch_len = patch_size
            cfg.stride = patch_size

            model = get_pretrained_model(mdl_str, cfg, d_model=d_model,
                                         device=device, n_channels=n_channels,
                                         window_size=window_size,
                                         limu_pretrain_ckpt=limu_pretrain_ckpt)
            ckpt_fname = ckpt_fnames[mdl_str]

            if ckpt_fname is not None and exists(ckpt_fname) and not overwrite:
                ckpt = torch.load(ckpt_fname, map_location=device)
                model.load_state_dict(ckpt)

            if USE_FLOW or USE_CMT:
                model.flow = SimpleRealNVPFlow(
                    d_model, num_layers=flow_num_layers, hidden_dim=flow_dim_ff
                )

            ssa_stats = None
            ssa_lambda = 0.0

            model = model.to(device)

            # TODO : test
            if USE_SSA:
                ssa_stats = compute_ssa_stats(
                    model,
                    train_dataloader,
                    device,
                    r=SSA_R,
                    max_batches=SSA_MAX_BATCHES,
                )
                ssa_lambda = SSA_LAMBDA

            # CMT (optional): build fewshot loader from test subject
            fewshot_loader = None
            if USE_CMT:
                fewshot_loader, _ = make_fewshot_loader_from_test(test_dataloader, k=CMT_K_SHOT, seed=0)

                # Train and attach a CMT RR head (few-shot supervised)
                train_cmt_rr_head(
                    model=model,
                    train_dataloader=train_dataloader,
                    fewshot_loader=fewshot_loader,
                    device=device,
                    top_k_ic=CMT_TOP_K_IC,
                    n_aug=CMT_N_AUG,
                    steps=CMT_STEPS,
                    lr=CMT_LR,
                    lambda_cmt=1.0,
                    max_src_batches=SSA_MAX_BATCHES,
                    seed=0,
                    attach_to_model=True,
                )

            if USE_FLOW:
                test_time_train_flow(
                    model,
                    test_dataloader,
                    device,
                    ttt_lr=TTT_FLOW_LR,
                    steps_per_batch=TTT_FLOW_STEPS_PER_BATCH,
                    ssa_stats=ssa_stats,
                    ssa_lambda=ssa_lambda,
                    mdl_str=mdl_str,
                )

            regressor = AdaptableIMURR(model, d_model, mdl_str=mdl_str)
            regressor = regressor.to(device)
            criterion = nn.L1Loss()

            if freeze:
                for name, p in regressor.named_parameters():
                    if name in ['encoder', 'pool']:
                        p.requires_grad = False
                    else:
                        p.requires_grad = True

                regressor_param_list = [p for p in regressor.parameters() if p.requires_grad]
                downstream_optimizer = torch.optim.Adam(regressor_param_list,
                                                        lr=downstream_lr)
            else:
                downstream_optimizer = torch.optim.Adam(regressor.parameters(), lr=downstream_lr)

            for epoch in range(epochs):
                regressor.train()
                downstream_train_losses = []
                total_loss = 0

                for imu, chest, conds, br in train_dataloader:
                    imu = imu.float().to(device)
                    br = br.float().to(device)
                    pss = torch.Tensor([get_max_freq(win, fs=BR_FS)*60 for win in
                                        chest]).float().to(device)

                    pred, _ = regressor(imu)
                    loss = criterion(pred, pss)
                    loss.backward()
                    downstream_optimizer.step()

                    total_loss += loss

                avg_loss = total_loss / len(train_dataloader)
                downstream_train_losses.append(avg_loss)

                total_loss = 0
                with torch.no_grad():
                    regressor.eval()
                    downstream_val_losses = []
                    for imu, chest, conds, br in val_dataloader:
                        imu = imu.float().to(device)
                        br = br.float().to(device)
                        pss = torch.Tensor([get_max_freq(win, fs=BR_FS)*60 for win in
                                            chest]).float().to(device)

                        pred, _ = regressor(imu)
                        loss = criterion(pred, pss)

                        total_loss += loss

                avg_loss = total_loss / len(val_dataloader)
                downstream_val_losses.append(avg_loss)
                print("{}\t{}".format(downstream_train_losses[-1], 
                                      downstream_val_losses[-1]))

            test_predictions = []
            br_test = []
            cond_test = []
            for imu, chest, cond, br in test_dataloader:
                y_hat, _ = regressor(imu.float().to(device))
                pss = np.array([get_max_freq(win, fs=BR_FS)*60 for win in chest])

                br_test.append(pss)
                test_predictions.append(y_hat.detach().cpu().numpy())
                cond_test.append(cond)

            predictions = np.concatenate(test_predictions, axis=0)
            
            br_test = np.concatenate(br_test, axis=0)
            cond_test = np.concatenate(cond_test, axis=0)

            downstream_test_evals = get_evals(
                br_test, predictions.squeeze(), cond_test)
            print("Downstream task eval: ", downstream_test_evals)

            if not debug:
                # write out the lbls and preds
                eval_file = ckpt_fname.split('ckp_last.pt')[0] + 'eval.pkl'
                if overwrite or exists(eval_file):
                    with open(eval_file, 'wb') as f:
                        pickle.dump(downstream_test_evals, f)

                labels_file = ckpt_fname.split('ckp_last.pt')[0] + 'labels.pkl'
                if overwrite or exists(labels_file):
                    with open(labels_file, 'wb') as f:
                        pickle.dump(br_test, f)

                preds_file = ckpt_fname.split('ckp_last.pt')[0] + 'preds.pkl'
                if overwrite or exists(preds_file):
                    with open(preds_file, 'wb') as f:
                        pickle.dump(predictions, f)


if __name__ == '__main__':
    main()
