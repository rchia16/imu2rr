# Codex implementation instructions: RR readout Phases A, B, and C

## Objective

Extend the existing RR spectral-readout investigation into three controlled phases:

- **Phase A:** static probabilistic and hybrid readouts;
- **Phase B:** confidence diagnostics and confidence-gated readout selection;
- **Phase C:** lightweight, no-label test-time calibration plus clearly separated oracle diagnostics.

The goal is to determine whether the remaining held-out-subject error is caused mainly by readout/calibration rather than by a deficient cross-modal respiratory representation.

Implement the experiment framework and run only minimal smoke checks. **Do not launch the full 15-subject, 3-seed experiment from the Codex session.**

---

## Repository context and files to inspect

Work in the current IMU2RR repository. Start with only these files unless an imported definition must be traced:

1. `rr_spectral_readout_experiment.py`
2. `rr_spectral_readout_analysis.py`
3. `run_rr_spectral_readout_experiment.sh`
4. the current F0 model implementation and its checkpoint-loading helper

Do not recursively inspect the whole repository. Use targeted searches such as:

```bash
grep -n "def .*stage\|class .*Readout\|argparse\|checkpoint" rr_spectral_readout_experiment.py
```

The existing runner already uses per-stage/per-seed logs, multiple GPUs, `--resume`, and the following default experiment context:

```text
subjects: S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29
seeds: 0 1 2
checkpoint root: /projects/BLVMob/imu-rr-seated/results/profile_clsa_qkv_fseries/runs/F0_static_shared
```

Preserve backward compatibility with the current spectral-readout outputs and analysis.

---

## Required new files

Prefer new files rather than turning the existing experiment script into an unreviewable monolith:

```text
rr_readout_phase_abc_experiment.py
rr_readout_phase_abc_analysis.py
run_rr_readout_phase_abc.sh
run_rr_readout_phase_abc_smoke.sh
```

Small shared helpers may be added to:

```text
rr_readout_phase_abc_utils.py
```

Do not rename or delete the existing spectral-readout scripts.

---

# Core experimental rules

## 1. Preserve LOSO separation

For held-out subject `Sxx`:

- fit readout parameters and gates using source-subject training data only;
- select hyperparameters/checkpoints using source-subject validation data only;
- evaluate exactly once on the held-out subject test data;
- never use target labels for deployable Phase B or Phase C methods;
- label all target-label methods with `oracle_` in filenames, tables, and method IDs.

## 2. Freeze the representation initially

For Phases A and B, freeze:

- TCN front end;
- STFT tokenizer/decoder used to construct the respiratory spectrum;
- cross-modal encoder;
- Profile-FiLM and static shared QKV components already contained in the source checkpoint.

Only the newly introduced readout, residual, confidence model, or gate may train.

For Phase C, adapt only the explicitly named low-dimensional calibration parameters. Never update the encoder, TCN, FiLM, QKV, or STFT backbone.

## 3. Cache frozen outputs once

Add a cache stage that writes frozen source-train, source-validation, and target-test representations for each `(subject, seed)`:

```text
cache/seed_000/S12/train.npz
cache/seed_000/S12/val.npz
cache/seed_000/S12/test.npz
cache/seed_000/S12/cache_manifest.json
```

Each cache should contain, where available:

```text
rr_true
rr_direct
spectral_logits or spectral_probability
frequency_bins_bpm
soft_spectral_rr
hard_spectral_rr
pooled_hidden
attention_hidden or final hidden sequence
window_index
condition
```

Also precompute spectral confidence features:

```text
entropy
max_probability
top1_top2_gap
peak_width_bpm
spectral_variance
hard_soft_gap_bpm
spectral_hidden_disagreement_bpm
respiratory_band_energy
```

Requirements:

- include the source checkpoint path and a stable configuration hash in `cache_manifest.json`;
- invalidate/rebuild the cache if the checkpoint/configuration hash changes;
- use atomic writes (`temporary file -> rename`);
- all later methods must reuse the cache rather than rerunning the cross-modal model;
- do not pool target test windows into source fitting.

---

# Phase A: static probabilistic and hybrid readouts

Implement the following **initial, bounded experiment set only**. Do not add a large hyperparameter sweep.

## Phase A method registry

| Method ID | Description |
|---|---|
| `A0_soft_spectral` | Existing expected-frequency decoder; no trainable parameters. |
| `A1_gaussian_kl` | Frequency-bin distribution trained with KL divergence to a Gaussian soft target. |
| `A2_wasserstein` | Ordered frequency distribution trained with a 1D CDF/Wasserstein-style loss. |
| `A3_kl_rr_mae` | `A1` loss plus MAE on the expected RR. |
| `A4_hidden_only` | Frozen pooled-hidden linear or ridge RR probe. |
| `A5_spec_linear_residual` | Soft spectral RR plus a bounded linear residual from pooled hidden state. |
| `A6_spec_hidden_conf_residual` | Soft spectral RR plus a bounded small MLP residual using hidden and confidence features. |

Keep the existing direct RR, hard spectral, and best prior frozen probe as reference rows in analysis, but do not retrain them unless needed.

## Probabilistic target

For true RR `y` and frequency-bin centre `r_k`, construct:

```text
q_k proportional to exp(-(r_k - y)^2 / (2 sigma^2))
```

Use one fixed default:

```text
sigma = 1.0 BPM
```

Expose `--target-sigma-bpm`, but do not smoke-test or sweep multiple sigma values. Hyperparameter changes can be performed later by the user.

Decode distributions using expected frequency:

```text
rr_pred = sum_k probability_k * frequency_bin_bpm_k
```

## Wasserstein loss

Use the mean absolute difference between target and predicted cumulative distributions:

```text
loss_w1 = mean(abs(cumsum(p) - cumsum(q)))
```

This is sufficient for the first experiment; do not add an external optimal-transport dependency.

## Residual readouts

Use:

```text
rr_pred = soft_spectral_rr + max_delta_bpm * tanh(raw_delta)
```

Defaults:

```text
max_delta_bpm = 2.0
residual_l1_weight = 0.01
```

For `A6`, use a deliberately small network, for example:

```text
LayerNorm -> Linear(input_dim, 32) -> GELU -> Dropout(0.1) -> Linear(32, 1)
```

Do not exceed hidden size 64 or two trainable linear layers.

## Training defaults

Use modest defaults and early stopping:

```text
max_epochs = 30
patience = 5
batch_size = 128 for cached features
optimizer = AdamW
learning_rate = 1e-3
weight_decay = 1e-4
grad_clip = 1.0
selection metric = source-validation subject-aggregated MAE
```

Do not select checkpoints using target performance.

## Required Phase A outputs

```text
phase_a_per_window.csv
phase_a_per_subject.csv
phase_a_summary.csv
phase_a_training_history.csv
phase_a_selected_checkpoints.json
```

Per-window output must include at least:

```text
subject, seed, split, method, window_index, rr_true, rr_pred,
soft_spectral_rr, residual_bpm, entropy, top1_top2_gap
```

---

# Phase B: confidence diagnostics and gating

Phase B must consume completed Phase A predictions and cached features. It must not rerun the encoder.

## Phase B diagnostic features

Assess whether the following source-validation features predict absolute RR error:

```text
entropy
max_probability
top1_top2_gap
peak_width_bpm
spectral_variance
hard_soft_gap_bpm
spectral_hidden_disagreement_bpm
respiratory_band_energy
```

Fit a small source-trained error-risk model using source data only. Use logistic regression or a one-layer MLP. Prefer logistic regression unless it is technically incompatible.

Define the default high-error label as:

```text
absolute_error > 3.0 BPM
```

Expose this as `--high-error-threshold-bpm`; do not sweep it in smoke tests.

## Phase B policies

| Method ID | Description |
|---|---|
| `B0_always_spec` | Always use the best Phase A spectral/probabilistic readout. |
| `B1_always_hybrid` | Always use the best Phase A hybrid readout. |
| `B2_disagreement_gate` | Use hybrid only when source-selected spectral/hidden disagreement exceeds a threshold. |
| `B3_soft_gate` | Continuous interpolation using a source-trained confidence gate. |
| `B4_oracle_gate` | Target-label upper bound; choose lower-error prediction per target window. Diagnostic only. |

Use source-validation MAE to identify the Phase A spectral candidate and hybrid candidate. Record their exact method IDs in the Phase B manifest.

For `B3`:

```text
rr_pred = (1 - gate) * rr_spectral + gate * rr_hybrid
```

where `gate` is in `[0, 1]`.

The gate should use only confidence features available at deployment. It must not receive `rr_true`, target subject ID, or target error.

## Required controls

Add two inexpensive diagnostic controls to the analysis, without training additional neural models:

```text
random_gate_matched_rate
shuffled_confidence_gate
```

These may be generated during analysis using deterministic seeds. They do not need separate training jobs.

## Required Phase B outputs

```text
phase_b_confidence_diagnostics.csv
phase_b_error_detection_metrics.csv
phase_b_per_window.csv
phase_b_per_subject.csv
phase_b_summary.csv
phase_b_gate_manifest.json
```

Diagnostics should include:

```text
Spearman correlation with absolute error
AUROC for >3 BPM error
error by confidence quartile
gate activation rate
oracle regret
MAE on high-confidence windows
MAE on low-confidence windows
```

Do not fail the experiment if AUROC is undefined for a tiny smoke subset; write `NaN` plus a warning.

---

# Phase C: lightweight no-label calibration

Phase C should start from the best static Phase A or Phase B deployable readout selected only from source validation.

## Deployable Phase C methods

| Method ID | Description |
|---|---|
| `C0_none` | No target adaptation. |
| `C1_feature_mean` | Existing-style frozen feature mean alignment with fixed alpha. |
| `C2_temperature` | Adapt one spectral temperature scalar. |
| `C3_affine_readout` | Adapt scalar `a` and `b` in `a * rr_base + b`. |
| `C4_confidence_gated_affine` | Apply the affine correction proportionally to the Phase B uncertainty gate. |

Use these fixed defaults:

```text
feature_mean_alpha = 0.75
a_init = 1.0
b_init = 0.0
temperature_init = 1.0
max_abs_bias_bpm = 2.0
a_range = [0.9, 1.1]
max_adaptation_steps = 10
adaptation_learning_rate = 1e-3
anchor_weight = 1.0
```

Parameterise bounded values safely, for example with `tanh`/sigmoid transforms rather than clipping after every step.

## No-label objective

Use only terms available without target RR labels:

```text
agreement between calibrated spectral prediction and frozen hidden probe
augmentation consistency if cached paired augmentations are available
temporal smoothness between chronologically adjacent windows
strong anchor to source/default calibration parameters
```

A suitable form is:

```text
loss =
    lambda_agree * abs(rr_calibrated - rr_hidden_frozen)
  + lambda_aug * abs(rr_calibrated(x) - rr_calibrated(aug_x))
  + lambda_smooth * abs(rr_t - rr_previous)
  + lambda_anchor * parameter_distance_from_initial
```

Defaults:

```text
lambda_agree = 1.0
lambda_aug = 0.0 unless paired augmentations already exist in cache
lambda_smooth = 0.1
lambda_anchor = 1.0
```

Do not introduce a new augmentation pipeline solely for Phase C in this implementation. Leave `lambda_aug=0` when no paired cache exists.

## Chronological adaptation

For deployable target adaptation:

- sort target windows by `window_index`;
- reset calibration parameters at the start of each held-out subject;
- use the current/past unlabeled windows only;
- never use future target windows to alter earlier predictions;
- save both pre-adaptation and post-adaptation predictions;
- save final parameter values and per-step parameter trajectories.

## Oracle diagnostics

Implement separately and mark clearly:

| Method ID | Description |
|---|---|
| `C5_oracle_offset` | Best target-label additive offset per held-out subject. |
| `C6_oracle_affine` | Best target-label scale and offset per held-out subject. |

Oracle rows must include:

```text
deployable = false
uses_target_labels = true
```

## Required Phase C outputs

```text
phase_c_per_window.csv
phase_c_per_subject.csv
phase_c_summary.csv
phase_c_parameter_trajectories.csv
phase_c_oracle_headroom.csv
phase_c_manifest.json
```

---

# Shared analysis and statistical reporting

Extend the existing subject-level analysis pattern rather than pooling windows for inference.

For each method report:

```text
mean subject-seed MAE
SD subject-seed MAE
median and IQR
RMSE
bias
Pearson correlation where valid
number of held-out subjects improved/worsened
seed-wise mean and SD
```

For primary paired comparisons, calculate differences at `(subject, seed)` level and use paired bootstrap confidence intervals.

Primary comparisons:

```text
A1_gaussian_kl vs A0_soft_spectral
A2_wasserstein vs A0_soft_spectral
A3_kl_rr_mae vs A0_soft_spectral
A5_spec_linear_residual vs A0_soft_spectral
A6_spec_hidden_conf_residual vs A0_soft_spectral
A6_spec_hidden_conf_residual vs A4_hidden_only
B3_soft_gate vs B0_always_spec
B3_soft_gate vs B1_always_hybrid
C1_feature_mean vs C0_none
C2_temperature vs C0_none
C3_affine_readout vs C0_none
C4_confidence_gated_affine vs C0_none
```

Use 10,000 bootstrap resamples for full analysis and only 100 in smoke mode.

Add a final compact table:

```text
phase_abc_final_method_comparison.csv
phase_abc_paired_comparisons.csv
phase_abc_subject_winners.csv
phase_abc_seed_summary.csv
```

---

# Minimal smoke-test policy

The smoke path exists only to catch syntax, shape, gradient, checkpoint, cache, and serialization failures. It must not attempt to establish scientific performance.

## Smoke configuration

`--smoke` must force:

```text
subjects = S12
seeds = 0
max_train_batches = 1
max_val_batches = 1
max_test_batches = 1
max_epochs = 2
patience = 1
bootstrap_resamples = 100
num_workers = 0
batch_size <= 8 before caching
cached_feature_batch_size <= 16
```

## Do not smoke every variant

Run only these representative paths:

```text
Phase A: A1_gaussian_kl and A6_spec_hidden_conf_residual
Phase B: B3_soft_gate
Phase C: C3_affine_readout
```

Baselines required by those paths may be calculated in-memory, but do not launch separate smoke jobs for every baseline or ablation.

## Maximum checks Codex should perform

Perform exactly these checks unless one fails:

1. Python syntax compilation for new/modified Python files:

   ```bash
   python -m py_compile rr_readout_phase_abc_*.py
   ```

2. Shell syntax for the two new runners:

   ```bash
   bash -n run_rr_readout_phase_abc.sh run_rr_readout_phase_abc_smoke.sh
   ```

3. One end-to-end smoke command covering the four representative methods.

Do not run a separate unit-test matrix, full seed, full subject, or full phase sweep. Do not run lint/type-check tools unless a syntax or import error requires them.

If the end-to-end smoke fails, make one targeted correction and rerun only the failed phase/method. Do not restart every completed smoke stage.

---

# Runner, resume, and failure logging

## Main runner requirements

`run_rr_readout_phase_abc.sh` must support environment overrides consistent with the existing runner:

```text
SEEDS
PHASES
METHODS
CUDA_DEVICES
OUT_ROOT
CHECKPOINT_ROOT
DATA_DIR
MDL_DIR
SUBJECTS
BATCH_SIZE
NUM_WORKERS
RESUME_FLAG
PYTHON_BIN
```

Recommended defaults:

```text
SEEDS="0 1 2"
PHASES="cache A B C analysis"
SUBJECTS="S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29"
OUT_ROOT="/projects/BLVMob/imu-rr-seated/results/rr_readout_phase_abc"
```

Do not make a failed job erase successful outputs. Parent runner behaviour should be:

- continue independent jobs after one job fails;
- collect exit codes;
- return non-zero at the end if any job failed;
- print only a compact failure summary to the terminal;
- preserve full logs on disk.

## Log layout

Write separate stdout and stderr logs:

```text
logs/phase_A_A1_gaussian_kl_S12_seed_000.out.log
logs/phase_A_A1_gaussian_kl_S12_seed_000.err.log
```

Also write:

```text
run_manifest.tsv
job_status.tsv
failed_jobs.tsv
```

`failed_jobs.tsv` must contain:

```text
timestamp
phase
method
subject
seed
gpu
exit_code
command
stdout_log
stderr_log
```

Enable:

```bash
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
```

Do not enable `CUDA_LAUNCH_BLOCKING=1` for normal runs. Support it through:

```bash
DEBUG_SYNC=1
```

for targeted reruns.

## Automatic rerun script

Generate an executable script after the run:

```text
rerun_failed_phase_abc.sh
```

It should contain one safely quoted command per row of `failed_jobs.tsv`, preserving the original phase, method, subject, seed, output root, and checkpoint root.

When `DEBUG_SYNC=1`, add:

```bash
CUDA_LAUNCH_BLOCKING=1
TORCH_SHOW_CPP_STACKTRACES=1
```

Do not automatically rerun failures in a loop. The user wants logs and explicit rerun commands for debugging.

## Resume markers

For each successful `(phase, method, subject, seed)` job, write an atomic marker:

```text
status/phase_A/A1_gaussian_kl/S12/seed_000.success.json
```

The marker must contain:

```text
configuration hash
checkpoint hash or path
expected output paths
completion timestamp
```

`--resume` skips only when:

- the marker exists;
- the configuration hash matches;
- every expected output exists and is non-empty.

Also support:

```text
--force
--rerun-failed <failed_jobs.tsv>
```

---

# Command-line interface

The Python experiment should support at least:

```text
--phase {cache,A,B,C,analysis}
--methods ...
--subjects ...
--seed
--checkpoint-root
--out-dir
--data-dir
--mdl-dir
--device
--batch-size
--num-workers
--resume
--force
--smoke
--max-train-batches
--max-val-batches
--max-test-batches
--max-epochs
--bootstrap-resamples
```

A single smoke command should be possible, for example:

```bash
bash run_rr_readout_phase_abc_smoke.sh
```

A full run should be possible later with:

```bash
SEEDS="0 1 2" \
CUDA_DEVICES="0 1" \
PHASES="cache A B C analysis" \
bash run_rr_readout_phase_abc.sh
```

A targeted failure rerun should be possible with:

```bash
DEBUG_SYNC=1 bash rerun_failed_phase_abc.sh
```

---

# Implementation quality requirements

1. Use deterministic NumPy/PyTorch seeds.
2. Preserve subject and chronological window identifiers through all caches and outputs.
3. Validate tensor/array shapes at cache creation boundaries, not repeatedly inside every batch.
4. Avoid large in-memory concatenations when streaming to disk is straightforward.
5. Avoid new dependencies unless already installed.
6. Keep method definitions in a registry/config structure rather than a long chain of duplicated code.
7. Include `deployable` and `uses_target_labels` columns in all Phase B/C summaries.
8. Warn, rather than crash, when a smoke subset cannot produce a correlation or AUROC.
9. Use `torch.load(..., weights_only=True)` when compatible with the existing checkpoint payload; otherwise retain compatibility and add a short comment explaining why `weights_only=False` is required.
10. Do not alter the original F0 checkpoints.

---

# Non-goals for this implementation

Do not add:

- new encoder or transformer layers;
- new Profile-FiLM/QKV variants;
- a broad sigma, loss-weight, architecture, or threshold sweep;
- target-label model selection for deployable methods;
- full LOSO execution during the Codex session;
- publication figures;
- unrelated refactors of the IMU2RR repository.

---

# Acceptance criteria

The implementation is complete when:

1. all requested files are created;
2. cache reuse works and is protected by a configuration hash;
3. Phase A, B, and C method registries contain exactly the requested initial variants;
4. deployable and oracle methods are clearly separated;
5. the three syntax checks and one minimal end-to-end smoke run pass, or unresolved errors are captured in `.err.log` and `failed_jobs.tsv` with a generated rerun command;
6. analysis writes subject-level and paired-comparison tables without window-pooled inference;
7. the main runner can later execute all 15 subjects and three seeds with resume support;
8. Codex does not run the full experiment.

---

# Codex response discipline

Keep the final Codex response compact to reduce token use. Do not paste source files or full logs. Report only:

```text
files created/modified
one-sentence architecture summary
syntax-check result
smoke result
paths to stdout/stderr logs
path to failed_jobs.tsv and rerun_failed_phase_abc.sh if failures remain
exact full-run command
```

When a smoke error occurs, inspect only the last 80 lines of the relevant stderr log first:

```bash
tail -n 80 <error-log>
```

Use targeted `grep` or a small line range next. Do not paste or read entire multi-thousand-line logs unless the traceback is otherwise incomplete.
