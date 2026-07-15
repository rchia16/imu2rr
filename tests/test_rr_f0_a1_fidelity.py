import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rr_readout_phase_abc_experiment import BinAffineDistribution
import run_rr_fixed_a1_benchmark as benchmark
import run_rr_jbhi_f0_family as f0_family
from vit_pressure_crossmodal_stft_rr_core import split_target_calibration_eval


def test_historical_a1_is_direct_per_bin_affine() -> None:
    readout = BinAffineDistribution(10)
    assert sorted(readout.state_dict().keys()) == ["bias", "scale"]
    assert sum(p.numel() for p in readout.parameters() if p.requires_grad) == 20


def test_f0_a1_uses_historical_bin_affine_readout() -> None:
    source = inspect.getsource(f0_family.train_a1_readout)
    assert "BinAffineDistribution" in source
    assert "BasisDistributionReadout" not in source


def test_f0_a1_collects_profile_conditioned_spectrum() -> None:
    loader_source = inspect.getsource(f0_family.collect_profile_conditioned_spectral_cache_from_loader)
    windows_source = inspect.getsource(f0_family.collect_profile_conditioned_spectral_cache_from_windows)
    assert "forward_profile_conditioned" in loader_source
    assert "forward_profile_conditioned" in windows_source
    assert "model(imu.float())" not in loader_source
    assert "model(imu)" not in windows_source


def test_standalone_a1_stage_is_disabled(tmp_path) -> None:
    with pytest.raises(SystemExit, match="Standalone A1 is disabled"):
        benchmark.main(["--root", str(tmp_path), "--stages", "a1", "--dry-run"])


def test_benchmark_defaults_use_five_fixed_seeds() -> None:
    assert benchmark.parse_list(benchmark.DEFAULT_SEEDS) == ["0", "1", "2", "3", "4"]


def test_primary_reference_is_original_readout() -> None:
    assert benchmark.PRIMARY_REFERENCE == "F0 original RR readout"


def test_metric_columns_exports_corr_aliases() -> None:
    df = pd.DataFrame({"Pearson_correlation": [0.1], "rr_corr": [0.2]})
    out = benchmark.metric_columns(df)
    assert out.loc[0, "corr"] == pytest.approx(0.1)


def test_load_f0_rows_rejects_unknown_metric_source(tmp_path) -> None:
    path = tmp_path / "subject_rows.csv"
    pd.DataFrame(
        {
            "subject": ["S12"],
            "mode": ["full"],
            "metric_source": ["not_a_method"],
            "mae": [1.0],
            "rmse": [1.0],
            "corr": [0.0],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unsupported metric_source"):
        benchmark.load_f0_rows(path, seed=0)


def _f0_lineage_rows(*, a1_hash: str = "ckpt", a1_eval: str = "eval") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject": "S12",
                "mode": "full",
                "metric_source": "original_rr_readout",
                "benchmark_method": "F0 original RR readout",
                "family": "f0",
                "model": "crossmodal_rr",
                "f0_checkpoint_sha256": "ckpt",
                "evaluation_indices_hash": "eval",
                "mae": 1.0,
                "rmse": 1.0,
                "corr": 0.0,
            },
            {
                "subject": "S12",
                "mode": "full",
                "metric_source": "a1_gaussian_kl",
                "benchmark_method": "F0 + A1 Gaussian-KL spectral readout",
                "family": "f0_a1",
                "model": "crossmodal_rr",
                "f0_checkpoint_sha256": a1_hash,
                "evaluation_indices_hash": a1_eval,
                "mae": 1.1,
                "rmse": 1.1,
                "corr": 0.0,
            },
        ]
    )


def test_shared_checkpoint_and_eval_index_lineage_is_enforced(tmp_path) -> None:
    rows = _f0_lineage_rows()
    benchmark.validate_integrated_f0_lineage(rows, path=tmp_path / "subject_rows.csv", seed=0)
    with pytest.raises(ValueError, match="f0_checkpoint_sha256 mismatch"):
        benchmark.validate_integrated_f0_lineage(
            _f0_lineage_rows(a1_hash="different"),
            path=tmp_path / "subject_rows.csv",
            seed=0,
        )
    with pytest.raises(ValueError, match="evaluation_indices_hash mismatch"):
        benchmark.validate_integrated_f0_lineage(
            _f0_lineage_rows(a1_eval="different"),
            path=tmp_path / "subject_rows.csv",
            seed=0,
        )


def test_missing_original_or_a1_rows_fail(tmp_path) -> None:
    rows = _f0_lineage_rows()
    with pytest.raises(ValueError, match="missing required metric_source=original_rr_readout"):
        benchmark.validate_integrated_f0_lineage(
            rows[rows["metric_source"] != "original_rr_readout"],
            path=tmp_path / "subject_rows.csv",
            seed=0,
        )
    with pytest.raises(ValueError, match="missing required metric_source=a1_gaussian_kl"):
        benchmark.validate_integrated_f0_lineage(
            rows[rows["metric_source"] != "a1_gaussian_kl"],
            path=tmp_path / "subject_rows.csv",
            seed=0,
        )


def test_calibration_split_is_deterministic() -> None:
    x = np.arange(20, dtype=np.float32).reshape(-1, 1)
    y = np.arange(20, dtype=np.float32)
    first = split_target_calibration_eval(x, y, None, 5, seed=12, mode="random", exclude_calibration_from_eval=False)[-1]
    second = split_target_calibration_eval(x, y, None, 5, seed=12, mode="random", exclude_calibration_from_eval=False)[-1]
    other = split_target_calibration_eval(x, y, None, 5, seed=13, mode="random", exclude_calibration_from_eval=False)[-1]
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)


def test_profile_metadata_collection_does_not_apply_normalizer_by_default() -> None:
    source = inspect.getsource(f0_family.run_loocv_experiment)
    assert "apply_to_model=False" in source


def test_profile_stats_from_windows_uses_same_normalizer_path() -> None:
    source = inspect.getsource(f0_family._profile_stats_from_windows)
    assert "_normalize_profile_stats_for_model(model, stats_raw, device)" in source


def test_fixed_profile_reuse_for_a1_target_cache() -> None:
    source = inspect.getsource(f0_family.a1_gaussian_kl_hook)
    assert source.count("fixed_profile_context(") == 1
    assert "target_profile_vector" in source
    assert "collect_profile_conditioned_spectral_cache_from_windows" in source


def test_profile_conditioned_cache_is_batch_partition_invariant(monkeypatch) -> None:
    class FakeModel(torch.nn.Module):
        def forward_profile_conditioned(self, imu, profile_vector, conditioning_mode):
            del conditioning_mode
            bsz = imu.size(0)
            pred = imu[:, :10].reshape(bsz, 2, 5) + profile_vector[:, :1].reshape(bsz, 1, 1)
            return pred, torch.zeros(bsz), torch.zeros(bsz, 1), profile_vector

    def fake_spectral_arrays(predicted_stft, soft_temperature, hidden_rr=None):
        del soft_temperature, hidden_rr
        arr = np.asarray(predicted_stft, dtype=np.float32)
        return {
            "spectral_logits": arr.reshape(arr.shape[0], -1),
            "frequency_bins_bpm": np.arange(arr.reshape(arr.shape[0], -1).shape[1], dtype=np.float32),
        }

    monkeypatch.setattr(f0_family, "spectral_arrays_from_stft", fake_spectral_arrays)
    windows = f0_family.TensorWindows(
        imu=torch.arange(40, dtype=torch.float32).reshape(4, 10),
        pressure=torch.zeros(4, 1),
        rr=torch.arange(4, dtype=torch.float32),
    )
    profile = torch.ones(1, 1)
    first = f0_family.collect_profile_conditioned_spectral_cache_from_windows(
        FakeModel(),
        windows,
        "cpu",
        profile,
        "film_qkv",
        batch_size=1,
    )
    second = f0_family.collect_profile_conditioned_spectral_cache_from_windows(
        FakeModel(),
        windows,
        "cpu",
        profile,
        "film_qkv",
        batch_size=3,
    )
    np.testing.assert_allclose(first["spectral_logits"], second["spectral_logits"])
    np.testing.assert_array_equal(first["rr_true"], second["rr_true"])
    np.testing.assert_array_equal(first["frequency_bins_bpm"], second["frequency_bins_bpm"])
