# Faithful F0/A1 Implementation Report

## Historical F0 Readout Mapping

- historical runner file: `vit_pressure_crossmodal_stft_rr_adaptation_alpha_hat_sweep.py`
- historical readout source file: `vit_pressure_crossmodal_stft_rr_rrprobe_tta_main.py`
- historical evaluation function: `profile_vector_unsupervised_evaluate`
- historical function call site: `feature_adaptive_evaluate` via the `RR_CONFIG_SWEEP` path
- representation tensor supplied to readout: pooled transformer hidden features
- profile-construction function: `_profile_stats_from_windows`
- probe class: `FaithfulRRRegressor`
- probe hyperparameters: `rr_probe_epochs=100`, `rr_probe_lr=1e-3`, `rr_probe_weight_decay=1e-4`, `rr_probe_batch_size=256`, `rr_probe_train_adapter=False`
- returned metric keys used for the primary deterministic endpoint: `rr_probe_post_mae`, `rr_probe_post_rmse`, `rr_probe_post_corr`
- calibration-window policy: `target_calibration_windows=32`, `target_calibration_mode=random`, seeded with the fold seed
- calibration/evaluation overlap policy: calibration windows are included in scored evaluation for historical equivalence
- target RR labels used for profile/readout adaptation: no; target labels are used only for evaluation diagnostics

## Benchmark Endpoint Mapping

- `native_rr_head`: diagnostic native F0 RR-head output, reported as `F0 native RR head`
- `original_rr_readout`: primary deterministic historical readout, reported as `F0 original RR readout`
- `a1_gaussian_kl`: 20-parameter spectral distribution readout from the same F0 checkpoint, reported as `F0 + A1 Gaussian-KL spectral readout`

## Correctness Guards

- Source-profile normalizer metadata is saved but not applied to the model for faithful held-out evaluation.
- F0 uses 20 source-training epochs, batch size 16, 100 RR-probe epochs, and 32 seeded-random target calibration windows.
- Benchmark paired tests use `F0 original RR readout` as the prespecified reference.
