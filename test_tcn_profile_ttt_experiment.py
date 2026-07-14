import inspect
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rr_jbhi_models import TCNRR
from tcn_profile_ttt_experiment import (
    AdaptationGateOutput,
    MetaSubjectEpisode,
    TTTAdapter,
    TTTConfig,
    TTT_SUBJECT_DEFAULTS,
    ProfileAffineRR,
    ProfileAdaptationGate,
    ProfileConditionedTCN,
    ProfileFiLM,
    ProfileGatedMeanAlignment,
    SubjectAwareDataset,
    WrappedTCN,
    adapt_on_unlabelled_prefix,
    aggregate_and_write,
    assert_subject_pure,
    build_meta_episodes,
    build_meta_subject_episode,
    compute_ttt_losses,
    deterministic_profile_permutation,
    evaluate_conditioned,
    fit_profile_normalizer,
    latent_stats_from_support,
    mean_alignment_alpha,
    meta_query_step,
    parse_args,
    raw_imu_features,
    selected_profile_indices,
    set_seed,
    snapshot_state_dict,
    valid_temporal_pairs,
)


def test_wrapped_tcn_reproduction():
    set_seed(0)
    base = TCNRR(in_ch=6, width=8, emb_dim=16)
    wrapped = WrappedTCN(base)
    base.eval()
    wrapped.eval()
    x = torch.randn(4, 6, 128)
    with torch.no_grad():
        expected = base(x).pred
        got = wrapped(x).prediction
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-6)


def test_identity_initialisation():
    set_seed(1)
    base = WrappedTCN(TCNRR(in_ch=6, width=8, emb_dim=16)).eval()
    model = ProfileConditionedTCN(base, profile_dim=10, use_film=True, use_affine=True, film_scale=0.03, affine_gain_bound=0.03, affine_bias_bound_bpm=1.0).eval()
    x = torch.randn(3, 6, 128)
    p = torch.randn(3, 10)
    with torch.no_grad():
        out0 = base(x).prediction
        out = model(x, p)
    assert torch.allclose(out.prediction, out0, atol=1e-6, rtol=1e-6)
    assert torch.allclose(out.film_gamma, torch.ones_like(out.film_gamma), atol=1e-6)
    assert torch.allclose(out.film_beta, torch.zeros_like(out.film_beta), atol=1e-6)
    assert torch.allclose(out.affine_gain, torch.ones_like(out.affine_gain), atol=1e-6)
    assert torch.allclose(out.affine_bias, torch.zeros_like(out.affine_bias), atol=1e-6)


def test_bounded_film_output():
    film = ProfileFiLM(profile_dim=4, hidden_dim=5, scale=0.03)
    with torch.no_grad():
        film.gamma.weight.fill_(100.0)
        film.beta.weight.fill_(-100.0)
    z = torch.randn(2, 5)
    p = torch.ones(2, 4)
    out, gamma, beta = film(z, p)
    assert out.shape == z.shape
    assert torch.max(torch.abs(gamma - 1.0)) <= 0.030001
    assert torch.max(torch.abs(beta)) <= 0.030001


def test_bounded_affine_output():
    aff = ProfileAffineRR(profile_dim=4, gain_bound=0.03, bias_bound_bpm=1.0)
    with torch.no_grad():
        aff.gain.weight.fill_(100.0)
        aff.bias.weight.fill_(-100.0)
    pred = torch.tensor([10.0, 20.0])
    p = torch.ones(2, 4)
    out, gain, bias = aff(pred, p)
    assert out.shape == pred.shape
    assert torch.max(torch.abs(gain - 1.0)) <= 0.030001
    assert torch.max(torch.abs(bias)) <= 1.000001


def test_deterministic_profile_window_selection():
    assert selected_profile_indices(20, 8, seed=0) == list(range(8))
    assert selected_profile_indices(20, 8, seed=3, chronological=False) == selected_profile_indices(20, 8, seed=3, chronological=False)


def test_subject_pure_profile_construction():
    assert_subject_pure(["S12", "S12"], "S12")
    try:
        assert_subject_pure(["S12", "S13"], "S12")
    except AssertionError:
        return
    raise AssertionError("mixed-subject profile did not fail")


def test_no_target_label_leakage_in_normalizer():
    profiles = {"S12": np.ones(4), "S13": np.ones(4) * 2, "S14": np.ones(4) * 3}
    mean, std = fit_profile_normalizer(profiles, ["S13", "S14"], "S12")
    assert np.allclose(mean, np.ones(4) * 2.5)
    assert np.all(std > 0)


def test_correct_output_shapes():
    base = WrappedTCN(TCNRR(in_ch=6, width=8, emb_dim=16)).eval()
    model = ProfileConditionedTCN(base, profile_dim=7, use_film=True, use_affine=True, film_scale=0.03, affine_gain_bound=0.03, affine_bias_bound_bpm=1.0).eval()
    x = torch.randn(5, 6, 128)
    p = torch.randn(5, 7)
    out = model(x, p)
    assert out.prediction.shape == (5,)
    assert out.hidden_tokens.ndim == 3
    assert out.pooled_hidden.shape == (5, 16)


def test_no_nans_or_infinities():
    x = np.random.default_rng(0).normal(size=(8, 128, 6))
    feats, names = raw_imu_features(x)
    assert len(feats) == len(names)
    assert np.isfinite(feats).all()


def test_same_seed_repeatability():
    subjects = ["S12", "S13", "S14", "S15"]
    assert deterministic_profile_permutation(subjects, 7) == deterministic_profile_permutation(subjects, 7)
    ds1 = SubjectAwareDataset(np.zeros((3, 128, 6)), np.ones(3), np.zeros(3), ["S12"] * 3, [0, 1, 2], aug_ratio=0.0)
    ds2 = SubjectAwareDataset(np.zeros((3, 128, 6)), np.ones(3), np.zeros(3), ["S12"] * 3, [0, 1, 2], aug_ratio=0.0)
    assert [ds1[i]["window_index"] for i in range(3)] == [ds2[i]["window_index"] for i in range(3)]


def _support(batch_size=4):
    return {
        "imu": torch.randn(batch_size, 6, 128),
        "subject_id": ["S12"] * batch_size,
        "window_index": np.arange(batch_size),
        "condition": np.zeros(batch_size),
        "sequence_index": np.arange(batch_size),
    }


def _adapter(group="film_affine", use_film=True, use_affine=True):
    set_seed(3)
    base = WrappedTCN(TCNRR(in_ch=6, width=8, emb_dim=16)).eval()
    static = ProfileConditionedTCN(base, profile_dim=5, use_film=use_film, use_affine=use_affine, film_scale=0.03, affine_gain_bound=0.03, affine_bias_bound_bpm=1.0).eval()
    profile = torch.randn(1, 5)
    return TTTAdapter(static, profile, group)


def test_ttt_api_cannot_accept_target_labels():
    sig = inspect.signature(adapt_on_unlabelled_prefix)
    assert "rr" not in sig.parameters
    adapter = _adapter("affine_only", use_film=False, use_affine=True)
    support = _support()
    support["rr"] = torch.ones(4)
    try:
        adapt_on_unlabelled_prefix(adapter, support, TTTConfig(ttt_parameter_group="affine_only"))
    except AssertionError:
        return
    raise AssertionError("TTT accepted target labels")


def test_temporal_pairs_valid_and_invalid():
    pairs = valid_temporal_pairs(["S12", "S12", "S12"], [0, 1, 2], [0, 1, 2], [0, 0, 1])
    assert pairs == [(0, 1)]
    assert valid_temporal_pairs(["S12", "S13"], [0, 1], [0, 1], [0, 0]) == []


def test_ttt_steps_zero_reproduces_static_model():
    adapter = _adapter("film_affine")
    support = _support()
    with torch.no_grad():
        static = adapter.static_model(support["imu"], adapter.profile.expand(support["imu"].shape[0], -1)).prediction
    result = adapt_on_unlabelled_prefix(adapter, support, TTTConfig(ttt_steps=0, ttt_parameter_group="film_affine", seed=4))
    with torch.no_grad():
        adapted = adapter(support["imu"]).prediction
    assert torch.allclose(static, adapted, atol=1e-6, rtol=1e-6)
    assert result.diagnostics["ttt_delta_norm"] == 0.0


def test_affine_only_ttt_changes_only_affine_deltas_and_freezes_tcn():
    adapter = _adapter("affine_only", use_film=False, use_affine=True)
    before = snapshot_state_dict(adapter.static_model)
    result = adapt_on_unlabelled_prefix(adapter, _support(), TTTConfig(ttt_steps=1, ttt_lr=1e-3, ttt_parameter_group="affine_only", seed=5))
    after = snapshot_state_dict(adapter.static_model)
    assert set(adapter.trainable_ttt_parameter_names()) == {"delta_gain", "delta_bias"}
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert result.state.delta_gamma is None
    assert result.state.delta_beta is None
    assert result.state.delta_gain is not None
    assert result.state.delta_bias is not None


def test_film_only_ttt_changes_only_film_deltas():
    adapter = _adapter("film_only")
    result = adapt_on_unlabelled_prefix(adapter, _support(), TTTConfig(ttt_steps=1, ttt_lr=1e-3, ttt_parameter_group="film_only", seed=6))
    assert set(adapter.trainable_ttt_parameter_names()) == {"delta_gamma", "delta_beta"}
    assert result.state.delta_gamma is not None
    assert result.state.delta_beta is not None
    assert result.state.delta_gain is None
    assert result.state.delta_bias is None


def test_film_affine_ttt_permitted_deltas_only():
    adapter = _adapter("film_affine")
    names = set(adapter.trainable_ttt_parameter_names())
    assert names == {"delta_gamma", "delta_beta", "delta_gain", "delta_bias"}


def test_one_step_no_change_when_all_loss_weights_zero():
    adapter = _adapter("film_affine")
    result = adapt_on_unlabelled_prefix(adapter, _support(), TTTConfig(ttt_steps=1, ttt_lr=1e-3, ttt_parameter_group="film_affine", lambda_ttt_aug=0.0, lambda_ttt_temp=0.0, lambda_ttt_spec=0.0, lambda_ttt_anchor=0.0, seed=7))
    assert result.diagnostics["ttt_delta_norm"] == 0.0


def test_repeated_same_seed_adaptation_is_repeatable():
    set_seed(10)
    support = _support()
    a1 = _adapter("film_affine")
    a2 = _adapter("film_affine")
    cfg = TTTConfig(ttt_steps=1, ttt_lr=1e-3, ttt_parameter_group="film_affine", seed=9)
    r1 = adapt_on_unlabelled_prefix(a1, support, cfg)
    r2 = adapt_on_unlabelled_prefix(a2, support, cfg)
    assert np.isclose(r1.diagnostics["ttt_delta_norm"], r2.diagnostics["ttt_delta_norm"])


def test_ttt_losses_finite_and_weighted_sum():
    adapter = _adapter("film_affine")
    cfg = TTTConfig(lambda_ttt_aug=1.2, lambda_ttt_temp=0.3, lambda_ttt_spec=0.0, lambda_ttt_anchor=0.7, seed=8)
    total, diag = compute_ttt_losses(adapter, _support(), cfg, seed=8)
    expected = 1.2 * diag["ttt_aug_loss"] + 0.3 * diag["ttt_temp_loss"] + 0.7 * diag["ttt_anchor_loss"]
    assert torch.isfinite(total)
    assert np.isclose(float(total.detach()), expected, rtol=1e-5, atol=1e-6)


def test_spectral_loss_confidence_weighted():
    adapter = _adapter("film_affine")
    support = _support()
    support["spectrum_logits"] = torch.tensor([[5.0, 0.0, -1.0, -2.0]] * 4)
    support["spectrum_freqs"] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    cfg = TTTConfig(ttt_use_spectrum=True, seed=8)
    _total, diag = compute_ttt_losses(adapter, support, cfg, seed=8)
    assert diag["ttt_spec_loss"] >= 0.0
    assert 0.0 <= diag["spectral_confidence_mean"] <= 1.0


def test_anchor_loss_zero_at_initialisation():
    adapter = _adapter("film_affine")
    _total, diag = compute_ttt_losses(adapter, _support(), TTTConfig(seed=8), seed=8)
    assert diag["ttt_anchor_loss"] == 0.0


def test_ttt_bounds_and_projection():
    adapter = _adapter("film_affine")
    result = adapt_on_unlabelled_prefix(adapter, _support(), TTTConfig(ttt_steps=1, ttt_lr=1.0, ttt_parameter_group="film_affine", ttt_max_delta_norm=1e-4, seed=12))
    assert result.diagnostics["ttt_delta_norm"] <= 1.0001e-4
    x = _support()["imu"]
    out = adapter(x)
    assert torch.max(torch.abs(out.film_gamma - 1.0)) <= 0.030001
    assert torch.max(torch.abs(out.film_beta)) <= 0.030001
    assert torch.max(torch.abs(out.affine_gain - 1.0)) <= 0.030001
    assert torch.max(torch.abs(out.affine_bias)) <= 1.000001


def test_rejection_is_recorded_and_falls_back():
    adapter = _adapter("film_affine")
    result = adapt_on_unlabelled_prefix(adapter, _support(), TTTConfig(ttt_steps=1, ttt_lr=1.0, ttt_parameter_group="film_affine", ttt_max_prediction_jump_bpm=0.0, seed=13))
    assert result.diagnostics["update_was_rejected"]
    assert result.diagnostics["ttt_delta_norm"] == 0.0


def test_phase12_subject_defaults_include_required_ttt_and_gate_fields():
    required = {
        "gate_value",
        "effective_ttt_lr",
        "ttt_loss_pre",
        "ttt_loss_post",
        "ttt_delta_norm",
    }
    assert required.issubset(TTT_SUBJECT_DEFAULTS)


def test_window_prediction_rows_include_required_phase12_columns():
    base = WrappedTCN(TCNRR(in_ch=6, width=8, emb_dim=16)).eval()
    ds = SubjectAwareDataset(
        np.zeros((2, 128, 6), dtype=np.float32),
        np.ones(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        ["S12", "S12"],
        [0, 1],
        aug_ratio=0.0,
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    out = evaluate_conditioned(base, loader, {"S12": np.zeros(1, dtype=np.float32)}, torch.device("cpu"))
    assert out["rows"]
    required = {
        "sequence_index",
        "rr_true",
        "rr_pred",
        "rr_pred_plain",
        "prediction_delta",
        "is_profile_window",
    }
    assert required.issubset(out["rows"][0])


def test_phase12_aggregate_writes_expected_audit_files():
    args = parse_args(["--mode", "t4_ttt_affine", "--subjects", "S12", "S13", "--eval-subjects", "S12", "--device", "cpu"])
    subject_row = {
        "seed": 0,
        "subject": "S12",
        "method": "C0_plain_tcn",
        "n_windows": 2,
        "mae": 1.0,
        "rmse": 1.0,
        "rr_corr": 1.0,
        "bias": 0.0,
        "median_ae": 1.0,
        "p95_ae": 1.0,
        "profile_windows": 1,
        "profile_norm": 0.0,
        "profile_distance": 0.0,
        "film_gamma_delta_norm": 0.0,
        "film_beta_norm": 0.0,
        "hidden_delta_rms": 0.0,
        "affine_gain": 1.0,
        "affine_bias": 0.0,
        "prediction_delta_mean": 0.0,
        "prediction_delta_std": 0.0,
        **TTT_SUBJECT_DEFAULTS,
    }
    pred_row = {
        "seed": 0,
        "subject": "S12",
        "method": "C0_plain_tcn",
        "window_index": 0,
        "condition": 0.0,
        "sequence_index": 0,
        "rr_true": 12.0,
        "rr_pred": 12.0,
        "rr_pred_plain": 12.0,
        "prediction_delta": 0.0,
        "is_profile_window": False,
    }
    with TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        aggregate_and_write(
            args,
            run_dir,
            [{
                "subject_rows": [subject_row],
                "window_predictions": [pred_row],
                "training_history": [],
                "profile_vectors": [],
                "profile_diagnostics": [],
                "leakage_audits": [],
            }],
        )
        for name in [
            "summary.csv",
            "per_seed_summary.csv",
            "subject_rows.csv",
            "window_predictions.csv",
            "ttt_diagnostics.csv",
            "gate_diagnostics.csv",
            "oracle_calibration.csv",
            "control_comparisons.csv",
            "manifest.json",
            "implementation_audit.json",
            "leakage_audit.json",
            "config.json",
        ]:
            assert (run_dir / name).exists(), name
        subject_cols = set(pd.read_csv(run_dir / "subject_rows.csv").columns)
        assert {"gate_value", "effective_ttt_lr", "ttt_loss_pre", "ttt_loss_post", "ttt_delta_norm"}.issubset(subject_cols)
        pred_cols = set(pd.read_csv(run_dir / "window_predictions.csv").columns)
        assert {"sequence_index", "rr_pred_plain", "prediction_delta", "is_profile_window"}.issubset(pred_cols)


def _synthetic_arrays():
    rng = np.random.default_rng(2)
    xs, rr, cond, subjects, widx = [], [], [], [], []
    for subj_i, subj in enumerate(["S12", "S13", "S14"]):
        for i in range(8):
            xs.append(rng.normal(size=(128, 6)).astype(np.float32) + subj_i * 0.01)
            rr.append(12.0 + subj_i + i * 0.05)
            cond.append(float(i % 2))
            subjects.append(subj)
            widx.append(i)
    return {
        "x": np.stack(xs),
        "rr": np.asarray(rr, dtype=np.float32),
        "cond": np.asarray(cond, dtype=np.float32),
        "subject_id": np.asarray(subjects, dtype=object),
        "window_index": np.asarray(widx, dtype=np.int64),
    }


def test_meta_episode_excludes_real_target_and_splits_are_disjoint():
    arrays = _synthetic_arrays()
    profiles = {s: np.ones(5, dtype=np.float32) * i for i, s in enumerate(["S12", "S13", "S14"])}
    args = parse_args(["--mode", "t6_meta_gated_ttt", "--subjects", "S12", "S13", "S14", "--eval-subjects", "S12", "--meta-support-windows", "3", "--meta-query-windows", "2", "--device", "cpu"])
    episodes = build_meta_episodes(arrays, ["S13", "S14"], "S12", profiles, args, torch.device("cpu"))
    assert {e.pseudo_target_subject for e in episodes} == {"S13", "S14"}
    for episode in episodes:
        assert set(episode.support_indices.cpu().tolist()).isdisjoint(set(episode.query_indices.cpu().tolist()))
        assert "rr" not in episode.support_batch
        assert "rr" in episode.query_batch


def test_meta_episode_selection_is_seed_reproducible():
    arrays = _synthetic_arrays()
    profiles = {s: np.ones(5, dtype=np.float32) * i for i, s in enumerate(["S12", "S13", "S14"])}
    args = parse_args(["--mode", "t6_meta_gated_ttt", "--subjects", "S12", "S13", "S14", "--eval-subjects", "S12", "--meta-support-windows", "3", "--meta-query-windows", "2", "--device", "cpu"])
    e1 = build_meta_episodes(arrays, ["S13", "S14"], "S12", profiles, args, torch.device("cpu"))
    e2 = build_meta_episodes(arrays, ["S13", "S14"], "S12", profiles, args, torch.device("cpu"))
    assert [e.support_indices.tolist() for e in e1] == [e.support_indices.tolist() for e in e2]
    assert [e.query_indices.tolist() for e in e1] == [e.query_indices.tolist() for e in e2]


def test_profile_adaptation_gate_bounds_and_fixed_controls():
    gate = ProfileAdaptationGate(input_dim=8, hidden_dim=4, lr_min=0.1, lr_max=0.5, delta_min=0.2, delta_max=0.8, alpha_max=0.5, include_alignment=True)
    out = gate(torch.randn(3, 8))
    assert isinstance(out, AdaptationGateOutput)
    assert torch.all((out.gate >= 0.0) & (out.gate <= 1.0))
    assert torch.all((out.lr_multiplier >= 0.1) & (out.lr_multiplier <= 0.5))
    assert torch.all((out.max_delta_multiplier >= 0.2) & (out.max_delta_multiplier <= 0.8))
    assert out.mean_alignment_alpha is not None
    assert torch.all((out.mean_alignment_alpha >= 0.0) & (out.mean_alignment_alpha <= 0.5))


def test_meta_query_step_updates_gate_and_lr_heads_but_not_tcn():
    arrays = _synthetic_arrays()
    profile = {"S13": np.ones(5, dtype=np.float32)}
    episode = build_meta_subject_episode("S13", arrays, profile, support_windows=3, query_windows=2, seed=0, device=torch.device("cpu"))
    static = ProfileConditionedTCN(WrappedTCN(TCNRR(in_ch=6, width=8, emb_dim=16)), profile_dim=5, use_film=True, use_affine=True, film_scale=0.03, affine_gain_bound=0.03, affine_bias_bound_bpm=1.0)
    gate = ProfileAdaptationGate(input_dim=8, hidden_dim=8)
    args = parse_args(["--mode", "t6_meta_gated_ttt", "--subjects", "S12", "S13", "--eval-subjects", "S12", "--device", "cpu"])
    before = snapshot_state_dict(static.wrapped)
    loss, _diag = meta_query_step(static, gate, episode, profile, args, mode="first_order")
    loss.backward()
    gate_grad = sum(float((p.grad.detach() ** 2).sum()) for p in gate.gate_head.parameters() if p.grad is not None)
    lr_grad = sum(float((p.grad.detach() ** 2).sum()) for p in gate.lr_head.parameters() if p.grad is not None)
    film_grad = sum(float((p.grad.detach() ** 2).sum()) for p in static.film.parameters() if p.grad is not None)
    assert gate_grad > 0.0
    assert lr_grad > 0.0
    assert film_grad > 0.0
    after = snapshot_state_dict(static.wrapped)
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_meta_query_step_second_order_executes_on_synthetic_batch():
    arrays = _synthetic_arrays()
    profile = {"S13": np.zeros(5, dtype=np.float32)}
    episode = build_meta_subject_episode("S13", arrays, profile, support_windows=3, query_windows=2, seed=0, device=torch.device("cpu"))
    static = ProfileConditionedTCN(WrappedTCN(TCNRR(in_ch=6, width=8, emb_dim=16)), profile_dim=5, use_film=True, use_affine=True, film_scale=0.03, affine_gain_bound=0.03, affine_bias_bound_bpm=1.0)
    gate = ProfileAdaptationGate(input_dim=8, hidden_dim=8)
    args = parse_args(["--mode", "t6_meta_gated_ttt", "--subjects", "S12", "S13", "--eval-subjects", "S12", "--meta-mode", "full_second_order", "--device", "cpu"])
    loss, diag = meta_query_step(static, gate, episode, profile, args, mode="full_second_order")
    assert torch.isfinite(loss)
    assert diag["query_windows"] == 2


def test_mean_alignment_alpha_bounds_and_zero_equivalence():
    aligner = ProfileGatedMeanAlignment(profile_dim=5, alpha_max=0.5)
    profile = np.ones(5, dtype=np.float32)
    alpha = mean_alignment_alpha(aligner, profile, torch.device("cpu"), mode="profile_gated", fixed_alpha=None, seed=0)
    assert 0.0 <= alpha <= 0.5
    assert alpha < 0.01
    pooled = torch.randn(3, 16)
    source = torch.randn(16)
    target = torch.randn(16)
    out0, a0 = aligner(pooled, torch.ones(3, 5), source, target, fixed_alpha=0.0)
    assert torch.allclose(out0, pooled)
    assert torch.allclose(a0, torch.zeros_like(a0))


def test_latent_target_stats_use_support_without_rr():
    base = WrappedTCN(TCNRR(in_ch=6, width=8, emb_dim=16)).eval()
    support = _support(batch_size=3)
    mean, std, count = latent_stats_from_support(base, support)
    assert count == 3
    assert mean.shape == std.shape
    assert "rr" not in support
