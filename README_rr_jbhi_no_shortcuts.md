# JBHI RR no-shortcuts experiment package

This file set replaces the earlier compact scaffold. It separates **source-faithful neural baselines** from the **full project cross-modal method**.

## Principle

Do not claim a simplified architecture is the main method or an official baseline.

- The main cross-modal model is run through the full project implementation:
  - `vit_pressure_crossmodal_stft_rr_core.py`
  - `vit_pressure_crossmodal_stft_rr_adaptation_alpha_hat_sweep.py`
  - `vit_pressure_crossmodal_stft_rr_adaptation_prototype_gate_sweep_full.py`

- The prototype/OOD gate is run through the full analyzer:
  - `analyze_adaptation_prototype_gate_policy_full.py`

- Continuous learned alpha is run through the full analyzer:
  - `analyze_adaptation_alpha_hat_policy_full.py`

- Prototype-gate and alpha-hat summarizers are full copies, not wrappers:
  - `summarize_rr_adaptation_prototype_gate_sweep_full.py`
  - `summarize_rr_adaptation_alpha_hat_sweep.py`

## New files

```text
rr_jbhi_source_baseline_models.py
run_rr_jbhi_source_neural_baselines.py
run_rr_jbhi_no_shortcuts_main_suite.sh
run_adaptation_alpha_hat_tests_full.sh
run_adaptation_prototype_gate_tests_full.sh
vit_pressure_crossmodal_stft_rr_adaptation_prototype_gate_sweep_full.py
summarize_rr_adaptation_prototype_gate_sweep_full.py
analyze_adaptation_alpha_hat_policy_full.py
analyze_adaptation_prototype_gate_policy_full.py
README_rr_jbhi_no_shortcuts.md
```

## Source-faithful baseline policy

`rr_jbhi_source_baseline_models.py` contains:

```text
resnet1d
cnn_gru
tcn
inceptiontime
patchtst_official
timesnet_official
```

Important:

- `resnet1d`, `cnn_gru`, and `tcn` are project-native strong neural baselines.
- `inceptiontime` is a faithful PyTorch port of the official InceptionTime module/block design. Use `--inception-ensemble 5` for the full ensemble-style protocol.
- `patchtst_official` requires `patchtst_backbone.py` from the official/project PatchTST implementation. There is **no fallback**.
- `timesnet_official` requires the official Time-Series-Library path. Set `TIMESNET_REPO=/path/to/Time-Series-Library`. There is **no fallback**.

## Main cross-modal method

The main method is **not** implemented in the baseline file. It is run only through the full cross-modal project scripts, which include:

```text
IMU STFT token encoder
pressure-STFT reconstruction
RR probe/readout
Profile-FiLM
fixed feature-mean alpha modes
Profile-FiLM/SPARC controls
prototype/OOD gate
continuous alpha_hat policy
```

## Full suite

```bash
bash run_rr_jbhi_no_shortcuts_main_suite.sh
```

Common override:

```bash
ROOT=/projects/BLVMob/imu-rr-seated/results/jbhi_no_shortcuts_v1 \
BASELINE_MODELS="resnet1d cnn_gru tcn inceptiontime" \
BASELINE_EPOCHS=80 \
BATCH_SIZE=128 \
DEVICE=cuda:0 \
bash run_rr_jbhi_no_shortcuts_main_suite.sh
```

To include official external baselines:

```bash
BASELINE_MODELS="resnet1d cnn_gru tcn inceptiontime patchtst_official timesnet_official" \
TIMESNET_REPO=/path/to/Time-Series-Library \
bash run_rr_jbhi_no_shortcuts_main_suite.sh
```

If PatchTST or TimesNet source code is missing, the run fails loudly. That is intentional.

## Alpha-hat only

```bash
bash run_adaptation_alpha_hat_tests_full.sh
```

Outputs:

```text
adaptation_alpha_hat_policy/adaptation_learned_alpha_by_subject.csv
adaptation_alpha_hat_policy/adaptation_learned_alpha_summary.csv
adaptation_alpha_hat_policy/adaptation_no_label_features.json
```

## Prototype/OOD gate only

```bash
bash run_adaptation_prototype_gate_tests_full.sh
```

Outputs:

```text
adaptation_prototype_gate_policy/adaptation_prototype_gate_by_subject.csv
adaptation_prototype_gate_policy/adaptation_prototype_gate_summary.csv
adaptation_prototype_gate_policy/adaptation_prototype_gate_features.json
```

The full prototype summary includes additional safety diagnostics:

```text
true_accepts
false_accepts
true_rejects
false_rejects
harm_avoidance_rate
missed_opportunity_rate
false_accept_mean_harm_bpm_mae
oracle_regret_mean
```

## Leakage policy

Both alpha-hat and prototype-gate analyzers use strict no-label feature filters. Label-derived diagnostics such as MAE/RMSE/correlation/residual/direction/overshoot/oracle/true RR fields are excluded from selected policy features and only allowed as pseudo-targets/evaluation labels.

## What to report

Recommended primary table:

```text
source neural baselines:
  resnet1d
  cnn_gru
  tcn
  inceptiontime
  patchtst_official, if source installed
  timesnet_official, if source installed

full project cross-modal method:
  none
  adapt_mean_alpha_075
  profile_film_init_only
  profile_film_unsup_sparc
  learned_alpha_hat
  prototype_ridge_gate_adapt_mean_alpha_075
```

Do not report the old compact `crossmodal_rr` scaffold as the main method.
