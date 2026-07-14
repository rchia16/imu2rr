import math

import numpy as np
import torch

from rr_spectral_readout_experiment import (
    BoundedResidualRR,
    FinalTokenProfileFiLM,
    SoftSpectralRR,
    Stage2Model,
    Stage3ProfileModel,
    gaussian_peak_target,
    hard_spectral_rr,
    peak_distribution_loss,
)
from vit_pressure_crossmodal_stft_rr_core import TinyIMU2PressureViT


def test_soft_spectral_rr_shape_and_bounds():
    pred = torch.rand(4, 8, 129, requires_grad=True)
    readout = SoftSpectralRR(br_fs=18.0, temperature=0.1)
    rr = readout(pred)
    assert rr.shape == (4,)
    assert torch.all(rr >= 3.0)
    assert torch.all(rr <= 45.0)


def test_peak_target_shape():
    pred = torch.rand(3, 7, 129)
    readout = SoftSpectralRR(br_fs=18.0)
    probs, freqs = readout.probabilities(pred)
    target = gaussian_peak_target(torch.tensor([12.0, 18.0, 24.0]), freqs, sigma_hz=0.03)
    assert target.shape == probs.shape
    assert torch.allclose(target.sum(dim=1), torch.ones(3), atol=1e-5)


def test_soft_rr_backpropagates_into_spectrum():
    pred = torch.rand(2, 6, 129, requires_grad=True)
    rr = SoftSpectralRR(br_fs=18.0)(pred)
    loss = rr.sum()
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_peak_loss_backpropagates():
    pred = torch.rand(2, 6, 129, requires_grad=True)
    rr_true = torch.tensor([12.0, 20.0])
    loss = peak_distribution_loss(pred, rr_true, SoftSpectralRR(br_fs=18.0), sigma_hz=0.03)
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0


def test_bounded_residual_identity_and_shape():
    soft = torch.tensor([10.0, 20.0, 30.0])
    z = torch.randn(3, 16)
    residual = BoundedResidualRR(d_in=16, epsilon_bpm=1.0)
    out = residual(soft, z)
    assert out.shape == (3,)
    assert torch.allclose(out, soft, atol=1e-6)


def test_film_identity_broadcast_and_gradients():
    film = FinalTokenProfileFiLM(profile_dim=10, d_model=16, scale=0.03)
    h = torch.randn(4, 6, 16)
    profile = torch.randn(4, 10)
    out = film(h, profile)
    assert out.shape == h.shape
    assert torch.allclose(out, h, atol=1e-6)
    loss = out.pow(2).mean()
    loss.backward()
    grads = [p.grad for p in film.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


def test_true_spectrum_hard_rr_plausible_and_deterministic():
    stft = torch.zeros(2, 5, 129)
    # With br_fs=18 and n_fft=256, bin 3 is about 0.2109 Hz, or 12.66 BPM.
    stft[:, :, 3] = 10.0
    rr1, conf1, peak1 = hard_spectral_rr(stft, br_fs=18.0)
    rr2, conf2, peak2 = hard_spectral_rr(stft, br_fs=18.0)
    assert torch.all(rr1 >= 3.0)
    assert torch.all(rr1 <= 45.0)
    assert torch.allclose(rr1, rr2)
    assert torch.allclose(conf1, conf2)
    assert torch.allclose(peak1, peak2)


def test_no_nan_inf_in_soft_readout():
    pred = torch.zeros(3, 4, 129)
    rr = SoftSpectralRR(br_fs=18.0, temperature=0.1)(pred)
    assert torch.isfinite(rr).all()


def test_stage3_profile_model_exposes_stage2_base_for_shared_eval_helpers():
    base = TinyIMU2PressureViT(
        input_channels=2,
        d_model=16,
        pred_len=64,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )
    stage2 = Stage2Model(base, SoftSpectralRR(br_fs=18.0))
    stage3 = Stage3ProfileModel(stage2, profile_dim=5, film_scale=0.03)

    assert stage3.base is base
    pressure = torch.randn(2, 64)
    true = stage3.base.pressure_stft_target(pressure, target_tokens=1)
    assert true.shape[0] == 2
