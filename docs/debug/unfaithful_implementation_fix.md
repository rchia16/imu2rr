# Codex implementation instructions: restore faithful F0, fix benchmark bugs, and add fixed A1

## Objective

Re-implement the exact historical F0 source-training, subject-profile, and original RR-readout pipeline; correct the confirmed profile-normalisation, metric-export, seeding, and ablation bugs; preserve the historical validation and checkpoint-selection behaviour for equivalence; and add **F0 + A1 Gaussian-target KL** as a separate spectral-distribution readout from the same corrected F0 checkpoint.

The final benchmark must contain three clearly separated F0 outputs:

1. **F0 native RR head** — diagnostic only.
2. **F0 original RR readout** — prespecified primary deterministic method and historical minimum-MAE model.
3. **F0 + A1 Gaussian-KL spectral readout** — fixed source-trained probabilistic spectral variant.

Do not substitute one endpoint for another.

The previous implementation review and identified failure modes remain the technical basis for this work. 

---

# 1. Inspect and identify the exact historical code path first

Before modifying any implementation, locate the exact historical code path that produced approximately:

```text
F0 post MAE: 2.698 BPM
F0 post correlation: 0.407
```

Trace the implementation from the historical runner and logs that emitted:

```text
RR_CONFIG_SWEEP
rr_probe_pre_mae
rr_probe_post_mae
rr_probe_post_corr
```

Document in the implementation report:

```text
historical runner file
historical readout source file
historical evaluation function
historical function call site
representation tensor supplied to the readout
profile-construction function
probe class
probe hyperparameters
returned metric keys
calibration-window policy
calibration/evaluation overlap policy
whether target RR labels were used
```

Do not choose a function based only on its filename.

If the historical implementation is embedded inside a script rather than exposed as a reusable function, extract it into one function without changing:

* tensor operations;
* pooling;
* standardisation;
* target scaling;
* optimisation;
* checkpoint selection;
* clipping;
* metric computation.

Do not recreate the historical readout from a written description when the numerical implementation already exists.

---

# 2. Canonical method definitions

Use the following names consistently across CSVs, manifests, logs, tables, and statistical comparisons.

## Diagnostic method

```text
F0 native RR head
```

Internal identifier:

```text
native_rr_head
```

This is the core model’s native or auxiliary RR prediction.

It must not be reported as the historical F0 result.

## Primary deterministic method

```text
F0 original RR readout
```

Internal identifier:

```text
original_rr_readout
```

This must use the exact historical downstream readout and fixed target-subject profile that produced the previous F0 headline result.

If the historical fields are confirmed to be:

```text
rr_probe_post_mae
rr_probe_post_rmse
rr_probe_post_corr
```

then map those values to the primary benchmark columns.

Do not make this mapping until the historical code-path inspection confirms equivalence.

## Probabilistic spectral variant

```text
F0 + A1 Gaussian-KL spectral readout
```

Internal identifier:

```text
a1_gaussian_kl
```

A1 must branch from the same corrected F0 checkpoint and use F0’s reconstructed respiratory spectral logits.

---

# 3. Faithful F0 configuration

The F0 reproduction must use:

```text
source-training epochs:             20
source-training batch size:         16
RR-probe epochs:                    100
target calibration windows:         32
calibration selection:              seeded random
TCN token mixer:                    enabled
TCN mixer alpha:                    0.05
Profile-FiLM:                       enabled
Profile-QKV:                        enabled
shared Profile-QKV:                 enabled
Profile-QKV layers:                 final transformer layer only
Profile-QKV scale:                  0.01
Profile-QKV residual:               enabled
CLSA:                               disabled
fast QKV adaptation:                disabled
feature-mean alignment:             disabled
test-time parameter updates:        disabled
```

Do not silently inherit baseline defaults such as:

```text
60 epochs
batch size 128
```

Do not change model dimensions, loss weights, target units, data windows, architecture, or historical optimiser settings.

---

# 4. Fix the test-only profile-normalisation bug

## File

```text
vit_pressure_crossmodal_stft_rr_core.py
```

## Function

```python
_collect_phase3_profile_metadata
```

## Problem

The model is trained and validated with raw profile statistics, but a source-profile normaliser is currently estimated and installed after checkpoint loading, immediately before held-out evaluation.

That creates:

```text
training: raw profile statistics
testing:  z-normalised profile statistics
```

The faithful reproduction must not change the profile encoder’s input representation only at test time.

## Required change

Update the signature:

```python
def _collect_phase3_profile_metadata(
    model,
    train_loader,
    device,
    args,
    *,
    apply_to_model: bool = False,
):
```

Continue calculating and returning:

```text
source_profile_stats
source_profile_mean
source_profile_std
source_profile_stats_norm
profile diagnostics
```

Only call:

```python
_set_model_profile_normalizer(...)
```

when:

```python
apply_to_model is True
```

Add to diagnostics:

```python
"profile_normalizer_applied_to_model": int(apply_to_model)
```

## Call site

In `run_loocv_experiment`, call:

```python
profile_ckpt_metadata = _collect_phase3_profile_metadata(
    model,
    train_loader,
    device,
    args,
    apply_to_model=False,
)
```

The normaliser may be saved as metadata, but it must not affect faithful held-out evaluation.

Do not modify `_forward_with_optional_profile` to normalise unconditionally.

Do not introduce normalised-profile training in this task.

---

# 5. Restore one fixed target-subject profile

The current implementation must not generate a new profile from every held-out test minibatch.

For each held-out subject:

1. deterministically select 32 calibration windows;
2. calculate the target-subject profile once;
3. freeze the profile;
4. reuse the same profile for all evaluation batches;
5. perform no target-specific parameter updates.

Required conceptual flow:

```python
calibration_indices = select_calibration_windows(
    subject_data,
    n_windows=32,
    seed=fold_seed,
)

with torch.no_grad():
    model.eval()
    fixed_subject_profile = build_subject_profile(
        model=model,
        calibration_windows=calibration_indices,
    )

for test_batch in test_loader:
    predictions = evaluate_with_fixed_profile(
        model=model,
        batch=test_batch,
        subject_profile=fixed_subject_profile,
    )
```

The prediction for a window must not depend on:

* the other windows in its test minibatch;
* test batch size;
* test batch order;
* whether it appears in the final smaller batch.

## Calibration policy

Inspect and reproduce the historical policy exactly:

* whether calibration windows were excluded from scored evaluation;
* whether calibration and evaluation windows overlapped;
* whether the method was transductive or inductive;
* whether target labels were used.

Save:

```text
n_subject_windows_total
calibration_indices
evaluation_indices
calibration_indices_hash
evaluation_indices_hash
calibration_evaluation_overlap_count
calibration_policy
```

Do not use target RR labels for profile construction unless the historical implementation did so.

If the historical implementation used target labels, do not label it deployable. Mark it explicitly as:

```text
oracle
```

or:

```text
supervised_target_calibration
```

The intended deployable policy is:

```text
source features + source RR labels
    → train RR readout

target calibration IMU/model outputs only
    → construct fixed profile

target evaluation
    → frozen profile + frozen source-trained readout
```

---

# 6. Restore the original F0 RR readout

Use the exact historical readout implementation identified in Section 1.

The original readout must run from the same restored core checkpoint used for all other F0 branches.

Required fold sequence:

```python
train_core_f0_model()
restore_best_f0_checkpoint()
model.eval()

native_metrics = evaluate_native_rr_head(...)

original_metrics = evaluate_historical_original_rr_readout(
    model=model,
    train_loader=train_loader,
    validation_loader=validation_loader,
    calibration_windows=calibration_windows,
    test_loader=test_loader,
    fixed_subject_profile=fixed_subject_profile,
)
```

Do not:

* train the F0 backbone twice;
* initialise a second backbone for the original readout;
* select a different F0 checkpoint;
* use the native RR-head metrics as the original-readout metrics.

Report separately:

```text
native_rr_head
rr_probe_pre
rr_probe_profile_post
stft_rr
aux_rr
```

The exact historical names may be retained internally, but the benchmark output must clearly map them to their roles.

---

# 7. Add the fixed A1 spectral branch

Reuse the existing fixed A1 implementation rather than recreating it.

Expected source may include:

```text
rr_readout_phase_abc_experiment.py
```

Locate the exact class and training function before integrating it.

## A1 architecture

The fixed A1 readout must have:

```text
10 basis scale coefficients
10 basis bias coefficients
20 trainable parameters total
```

Expected form:

```python
BasisDistributionReadout(
    n_bins=n_bins,
    n_basis=10,
)
```

A1 calibrates F0 spectral logits:

```python
log_scale = basis @ log_scale_coef
bias = basis @ bias_coef

calibrated_logits = (
    spectral_logits * torch.exp(log_scale)
    + bias
)

probabilities = torch.softmax(
    calibrated_logits,
    dim=-1,
)
```

## Gaussian-soft target

Use:

```text
sigma = 1.0 BPM
```

The target distribution is:

[
q_k \propto
\exp\left(
-\frac{1}{2}
\left(\frac{f_k-y}{1.0}\right)^2
\right)
]

The implemented cross-entropy:

```python
loss = -(target_distribution * log_probabilities).sum(
    dim=-1
).mean()
```

is equivalent to minimising:

```text
KL(q || p)
```

up to the fixed entropy of the target distribution.

## Point estimate

Use the expected respiratory rate:

```python
predicted_rr = (
    probabilities * rr_frequency_bins_bpm
).sum(dim=-1)
```

## Source-only training policy

For each held-out fold:

1. restore the same corrected F0 checkpoint;
2. freeze every F0 parameter;
3. cache source-train spectral logits and source RR labels;
4. cache source-validation spectral logits and labels;
5. cache held-out target-test spectral logits;
6. train only the 20 A1 parameters on source-train data;
7. select A1 checkpoint using source-validation MAE;
8. freeze A1;
9. evaluate on target-test data;
10. perform no target-label fitting;
11. perform no target-specific parameter update.

A1 must not load a separately trained F0 backbone.

---

# 8. Enforce shared F0 checkpoint and test windows

For every seed and held-out subject, the following must be identical between:

```text
F0 original RR readout
F0 + A1 Gaussian-KL spectral readout
```

Required shared values:

```text
F0 checkpoint path
F0 checkpoint SHA-256
F0 model configuration
seed
fold seed
held-out subject
test indices
test-indices hash
target RR values
target RR units
profile-normalisation policy
source/validation split
```

Only the final readout may differ.

Conceptual structure:

```text
Corrected faithful F0 checkpoint
│
├── native RR head
│     diagnostic only
│
├── original F0 RR readout
│     primary deterministic method
│
└── reconstructed spectral logits
      └── fixed A1 Gaussian-KL readout
            probabilistic spectral variant
```

---

# 9. Update the F0 runner

## File

```text
run_rr_jbhi_f0_family.py
```

Add:

```python
parser.add_argument(
    "--metric-source",
    choices=[
        "native_rr_head",
        "original_rr_readout",
        "a1_gaussian_kl",
        "all",
    ],
    default="original_rr_readout",
)
```

Recommended behaviour:

### `native_rr_head`

Run only the native diagnostic endpoint.

### `original_rr_readout`

Run the faithful historical deterministic endpoint.

### `a1_gaussian_kl`

Run A1 from the corrected F0 checkpoint.

### `all`

Run all three branches from one restored checkpoint.

Label subject rows:

```text
metric_source
benchmark_method
```

Canonical mappings:

```python
{
    "native_rr_head": "F0 native RR head",
    "original_rr_readout": "F0 original RR readout",
    "a1_gaussian_kl": "F0 + A1 Gaussian-KL spectral readout",
}
```

---

# 10. Restore separate F0 and baseline defaults

## File

```text
run_rr_fixed_a1_benchmark.py
```

Add or retain:

```python
parser.add_argument(
    "--batch-size",
    type=int,
    default=128,
    help="Batch size for conventional baselines.",
)

parser.add_argument(
    "--baseline-epochs",
    type=int,
    default=60,
)

parser.add_argument(
    "--f0-batch-size",
    type=int,
    default=16,
)

parser.add_argument(
    "--f0-epochs",
    type=int,
    default=20,
)

parser.add_argument(
    "--f0-rr-probe-epochs",
    type=int,
    default=100,
)

parser.add_argument(
    "--f0-calibration-windows",
    type=int,
    default=32,
)
```

The F0 command must pass:

```text
--epochs 20
--batch-size 16
--rr-probe-epochs 100
--calibration-windows 32
--metric-source all
```

The baseline command must continue using:

```text
--baseline-epochs 60
--batch-size 128
```

Do not pass baseline batch size into F0.

---

# 11. Correct baseline correlation export

## File

```text
evaluations.py
```

## Function

```python
simple_regression_metrics
```

Return a canonical correlation field while preserving backward compatibility:

```python
def simple_regression_metrics(y_true, y_pred):
    metrics = Evaluation(
        y_true,
        y_pred,
    ).get_evals()

    corr = float(metrics["pearsonr_coeff"])

    return {
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "corr": corr,
        "pearsonr": corr,
    }
```

## File

```text
run_rr_jbhi_baselines.py
```

Use strict access:

```python
return {
    "mae": float(metrics["mae"]),
    "rmse": float(metrics["rmse"]),
    "corr": float(metrics["corr"]),
}
```

Replace all uses of:

```python
metrics.get("corr", np.nan)
```

with:

```python
metrics["corr"]
```

A missing metric must raise an error.

---

# 12. Correct F0 ablations

## File

```text
run_rr_jbhi_f0_family.py
```

## Full F0

```python
"full": {
    "use_profile_film": True,
    "use_profile_qkv": True,
    "shared_profile_qkv": True,
    "profile_conditioning": "film_qkv",
    "use_tcn_token_mixer": True,
    "tcn_mixer_alpha": 0.05,
    "profile_qkv_layers": "last1",
    "profile_qkv_scale": 0.01,
    "profile_qkv_residual": True,
}
```

## No-FiLM

Remove FiLM only:

```python
"no_film": {
    "use_profile_film": False,
    "use_profile_qkv": True,
    "shared_profile_qkv": True,
    "profile_conditioning": "qkv",
    "use_tcn_token_mixer": True,
    "tcn_mixer_alpha": 0.05,
    "profile_qkv_layers": "last1",
    "profile_qkv_scale": 0.01,
    "profile_qkv_residual": True,
}
```

## No-QKV

Remove QKV only:

```python
"no_qkv": {
    "use_profile_film": True,
    "use_profile_qkv": False,
    "shared_profile_qkv": False,
    "profile_conditioning": "film",
    "use_tcn_token_mixer": True,
    "tcn_mixer_alpha": 0.05,
    "profile_qkv_layers": "none",
    "profile_qkv_scale": 0.01,
    "profile_qkv_residual": False,
}
```

Each ablation must change one mechanism only.

A1 should initially be evaluated only on full F0 unless a separate A1 ablation study is explicitly requested.

---

# 13. Deterministic fold seeding

Implement a stable fold-seed helper.

Do not use Python’s built-in `hash()`.

Recommended:

```python
def make_fold_seed(
    base_seed: int,
    subject: str,
) -> int:
    subject_number = int(
        subject.lstrip("S")
    )
    return int(
        base_seed * 1000
        + subject_number
    )
```

Apply the fold seed immediately before:

* F0 model construction;
* source dataloader shuffling;
* calibration-window sampling;
* original RR-readout initialisation;
* A1 initialisation;
* baseline model construction.

Save:

```text
base_seed
fold_seed
calibration_seed
```

Later folds must not inherit an RNG state determined by earlier fold execution order.

---

# 14. Correct the alpha CLI flag

## File

```text
run_rr_jbhi_baselines.py
```

Replace:

```python
parser.add_argument(
    "--eval-alpha075",
    action="store_true",
    default=True,
)
```

with:

```python
parser.add_argument(
    "--eval-alpha075",
    action=argparse.BooleanOptionalAction,
    default=True,
)
```

Support:

```text
--eval-alpha075
--no-eval-alpha075
```

---

# 15. Benchmark reference and comparisons

## File

```text
run_rr_fixed_a1_benchmark.py
```

Set:

```python
PRIMARY_REFERENCE = (
    "F0 original RR readout"
)
```

Do not use A1 as the hard-coded statistical reference.

Report paired differences as:

```text
comparison MAE − F0 original RR readout MAE
```

For A1:

```text
A1 MAE − F0 original RR-readout MAE
```

Interpretation:

```text
negative: A1 has lower MAE
positive: original F0 has lower MAE
```

F0 is the prespecified primary deterministic method.

Only call it the final minimum-MAE method if the corrected multi-seed benchmark confirms that it retains the lowest aggregate MAE.

---

# 16. Result loading

## File

```text
run_rr_fixed_a1_benchmark.py
```

Update `load_f0_rows` so it returns separate rows.

## Native row

```python
native = df[
    df["metric_source"]
    == "native_rr_head"
].copy()

native["family"] = "f0_native"
native["benchmark_method"] = (
    "F0 native RR head"
)

native["mae"] = native["native_rr_mae"]
native["rmse"] = native["native_rr_rmse"]
native["corr"] = native["native_rr_corr"]
```

## Original-readout row

```python
original = df[
    df["metric_source"]
    == "original_rr_readout"
].copy()

original["family"] = "f0"
original["benchmark_method"] = (
    "F0 original RR readout"
)
```

Map the exact confirmed historical fields:

```python
original["mae"] = original[
    "rr_probe_post_mae"
]
original["rmse"] = original[
    "rr_probe_post_rmse"
]
original["corr"] = original[
    "rr_probe_post_corr"
]
```

Use different field names if the code-path inspection identifies different historical outputs.

## A1 row

```python
a1 = df[
    df["metric_source"]
    == "a1_gaussian_kl"
].copy()

a1["family"] = "f0_a1"
a1["benchmark_method"] = (
    "F0 + A1 Gaussian-KL spectral readout"
)

a1["mae"] = a1["a1_mae"]
a1["rmse"] = a1["a1_rmse"]
a1["corr"] = a1["a1_corr"]
```

Allowed metric sources:

```python
allowed = {
    "native_rr_head",
    "original_rr_readout",
    "a1_gaussian_kl",
}
```

Raise `ValueError` for any unexpected source.

Do not aggregate native, original, and A1 results into a single F0 row.

---

# 17. Subject-level output columns

Write, at minimum:

```text
seed
fold_seed
subject
variant
metric_source
benchmark_method

f0_checkpoint_path
f0_checkpoint_sha256
test_indices_hash

native_rr_mae
native_rr_rmse
native_rr_corr

rr_probe_pre_mae
rr_probe_pre_rmse
rr_probe_pre_corr

rr_probe_post_mae
rr_probe_post_rmse
rr_probe_post_corr

stft_rr_mae
stft_rr_rmse
stft_rr_corr

aux_rr_mae
aux_rr_rmse
aux_rr_corr

a1_mae
a1_rmse
a1_corr
a1_cross_entropy
a1_entropy_mean
a1_true_bin_probability_mean

n_subject_windows_total
n_calibration_windows
calibration_seed
calibration_indices_hash
evaluation_indices_hash
calibration_evaluation_overlap_count
calibration_policy

profile_normalizer_applied_to_model
test_time_parameter_updates
target_labels_used_for_profile
target_labels_used_for_a1
a1_target_parameters_updated
```

---

# 18. Manifest requirements

For every seed and fold, record:

```text
implementation_policy
git_commit
seed
fold_seed
subject

epochs
batch_size
RR-probe epochs
number of calibration windows
calibration selection policy
calibration/evaluation overlap policy

TCN mixer enabled
TCN mixer alpha
Profile-FiLM enabled
Profile-QKV enabled
shared Profile-QKV enabled
Profile-QKV layers
Profile-QKV scale
Profile-QKV residual

profile normaliser applied to model
test-time parameter updates
validation split policy
checkpoint-selection metric

metric source
primary method
primary statistical reference

F0 checkpoint path
F0 checkpoint SHA-256
test-indices hash

A1 enabled
A1 number of basis functions
A1 trainable parameter count
A1 Gaussian sigma in BPM
A1 loss
A1 point estimator
A1 checkpoint-selection split
A1 checkpoint-selection metric
A1 target labels used
A1 target parameters updated
```

Recommended values:

```text
implementation_policy =
faithful_historical_f0_with_fixed_a1

primary_method =
F0 original RR readout

primary_statistical_reference =
F0 original RR readout

A1 number of basis functions = 10
A1 trainable parameter count = 20
A1 Gaussian sigma = 1.0 BPM
A1 loss =
gaussian_target_cross_entropy_equivalent_to_kl_q_p
A1 point estimator = expected_rr
A1 checkpoint-selection split = source_validation
A1 checkpoint-selection metric = mae
A1 target labels used = false
A1 target parameters updated = false
```

---

# 19. Policy validators

Add small validation helpers.

## Original F0 policy

```python
def validate_original_f0_policy(
    manifest: dict,
) -> None:
    expected = {
        "epochs": 20,
        "batch_size": 16,
        "rr_probe_epochs": 100,
        "n_calibration_windows": 32,
        "tcn_mixer_alpha": 0.05,
        "profile_film": True,
        "profile_qkv": True,
        "shared_profile_qkv": True,
        "profile_qkv_layers": "last1",
        "profile_qkv_scale": 0.01,
        "profile_normalizer_applied_to_model": False,
        "test_time_parameter_updates": False,
        "metric_source": "original_rr_readout",
    }

    mismatches = {
        key: (
            manifest.get(key),
            expected_value,
        )
        for key, expected_value
        in expected.items()
        if manifest.get(key)
        != expected_value
    }

    if mismatches:
        raise ValueError(
            "Run does not satisfy original F0 policy: "
            f"{mismatches}"
        )
```

## A1 policy

```python
def validate_a1_policy(
    manifest: dict,
) -> None:
    expected = {
        "metric_source": "a1_gaussian_kl",
        "a1_n_basis": 10,
        "a1_n_parameters": 20,
        "a1_gaussian_sigma_bpm": 1.0,
        "a1_point_estimator": "expected_rr",
        "a1_checkpoint_selection_split": (
            "source_validation"
        ),
        "a1_checkpoint_selection_metric": "mae",
        "a1_target_labels_used": False,
        "a1_target_parameters_updated": False,
    }

    mismatches = {
        key: (
            manifest.get(key),
            expected_value,
        )
        for key, expected_value
        in expected.items()
        if manifest.get(key)
        != expected_value
    }

    if mismatches:
        raise ValueError(
            "Run does not satisfy fixed A1 policy: "
            f"{mismatches}"
        )
```

---

# 20. Unit tests

Create:

```text
tests/test_rr_faithful_f0_a1.py
```

These tests must not:

* load the research dataset;
* train the F0 model;
* require CUDA;
* execute a subprocess;
* run a smoke experiment;
* use the benchmark dry-run mode.

Use small tensors, synthetic arrays, temporary CSVs, and monkeypatching.

## Test 1: metadata collection is non-mutating

Verify that:

```python
_collect_phase3_profile_metadata(
    ...,
    apply_to_model=False,
)
```

returns metadata but does not set:

```text
model.source_profile_mean
model.source_profile_std
```

## Test 2: explicit normaliser application remains possible

Verify that:

```python
apply_to_model=True
```

sets the normaliser.

## Test 3: LOSO orchestration passes `False`

Monkeypatch `_collect_phase3_profile_metadata` and verify the post-checkpoint evaluation path calls it with:

```python
apply_to_model=False
```

## Test 4: canonical correlation export

Using perfectly correlated arrays, assert:

```python
metrics["corr"] == pytest.approx(1.0)
metrics["pearsonr"] == pytest.approx(1.0)
```

## Test 5: F0 and baseline defaults remain separate

Inspect generated command lists without launching them.

Assert F0 receives:

```text
--epochs 20
--batch-size 16
--rr-probe-epochs 100
--calibration-windows 32
```

Assert baselines receive:

```text
--baseline-epochs 60
--batch-size 128
```

## Test 6: ablations are one-factor changes

Assert `no_film` preserves all QKV settings.

Assert `no_qkv` preserves FiLM and TCN settings.

## Test 7: deterministic calibration selection

For the same:

```text
seed
subject
available windows
```

assert identical selected indices.

For a different seed, assert different indices when enough windows are available.

## Test 8: fixed profile is built once

Use a dummy profile builder and two dummy test batches.

Assert:

```python
profile_builder.call_count == 1
```

## Test 9: predictions are invariant to batch partition

Evaluate identical synthetic samples as:

* one full batch;
* two smaller batches;
* reordered batches.

Restore sample order and assert:

```python
np.testing.assert_allclose(
    full_batch_predictions,
    partitioned_predictions,
    rtol=1e-6,
    atol=1e-6,
)
```

## Test 10: original endpoint is not replaced by native head

Create synthetic results:

```text
native MAE = 4.0
pre-probe MAE = 3.0
post-probe MAE = 2.7
```

Assert:

```text
F0 original RR readout MAE = 2.7
F0 native RR head MAE = 4.0
```

## Test 11: A1 contains exactly 20 trainable parameters

```python
readout = BasisDistributionReadout(
    n_bins=64,
    n_basis=10,
)

n_trainable = sum(
    parameter.numel()
    for parameter in readout.parameters()
    if parameter.requires_grad
)

assert n_trainable == 20
```

## Test 12: Gaussian target properties

Verify:

* probabilities sum to one;
* the peak bin is nearest the target RR;
* a target centred on a frequency bin gives a symmetric distribution;
* sigma equals 1.0 BPM.

## Test 13: A1 loss is Gaussian-target cross-entropy

Verify the implemented loss equals:

```python
-(
    target_distribution
    * log_probabilities
).sum(dim=-1).mean()
```

## Test 14: A1 point estimate is the expectation

Assert:

```python
predicted_rr == (
    probabilities * rr_bins
).sum(dim=-1)
```

## Test 15: original F0 and A1 share the same checkpoint

Create synthetic rows and assert:

```python
original_row["f0_checkpoint_sha256"] == (
    a1_row["f0_checkpoint_sha256"]
)
```

## Test 16: original F0 and A1 use the same test windows

Assert:

```python
original_row["test_indices_hash"] == (
    a1_row["test_indices_hash"]
)
```

## Test 17: A1 uses source labels only

Assert manifest values:

```text
a1_target_labels_used = false
a1_target_parameters_updated = false
a1_checkpoint_selection_split = source_validation
```

## Test 18: primary reference is original F0

Assert:

```python
PRIMARY_REFERENCE == (
    "F0 original RR readout"
)
```

## Test 19: unsupported metric sources fail

Provide:

```text
metric_source = unknown_readout
```

and assert `ValueError`.

## Test command

```bash
pytest -q tests/test_rr_faithful_f0_a1.py
```

---

# 21. Validation and checkpoint-selection policy

The historical implementation used:

* random window-level source validation;
* composite-loss checkpoint selection.

These may be methodologically imperfect, especially with overlapping windows.

For the faithful historical reproduction:

```text
preserve the historical source-validation split
preserve the historical F0 checkpoint-selection rule
record both in the manifest
do not silently improve them
```

Do not change them while claiming equivalence to the prior F0 result.

A later improved experiment may use:

* subject-grouped validation;
* session-grouped validation;
* temporally blocked validation;
* checkpoint selection by RR MAE.

That must be implemented under a separate method or experiment name.

A1 checkpoint selection remains:

```text
source-validation MAE
```

because that is part of the fixed A1 definition.

---

# 22. Staged verification after implementation

Do not immediately run all five seeds.

## Stage A: existing-checkpoint diagnostic

Using an existing seed-0 F0 checkpoint, compare:

| Test | Profile statistics | Profile construction    | Readout             |
| ---- | ------------------ | ----------------------- | ------------------- |
| A    | Normalised         | Current minibatch       | Native head         |
| B    | Raw                | Current minibatch       | Native head         |
| C    | Raw                | Fixed 32-window profile | Native head         |
| D    | Raw                | Fixed 32-window profile | Original F0 readout |

Run initially on:

```text
S12
S16
S24
```

This diagnostic separates:

* test-only normalisation;
* minibatch-dependent profiling;
* native-head weakness;
* original-readout behaviour.

This is not a smoke test. It is a targeted checkpoint re-evaluation.

## Stage B: exact seed-0 reproduction

Run the exact historical F0 configuration for seed 0.

Report separately:

```text
native_rr_head
rr_probe_pre
rr_probe_profile_post
stft_rr
aux_rr
A1 Gaussian-KL
```

Compare subject-level metrics and predictions against the saved historical seed-0 outputs.

## Stage C: equivalence gate

Do not run seeds 1–4 until seed 0 satisfies approximately:

```text
aggregate MAE difference ≤ 0.10 BPM
aggregate correlation difference ≤ 0.03
per-subject prediction correlation ≥ 0.99
```

Use stricter tolerances when exact deterministic reproduction is expected.

If equivalence fails, stop and report:

```text
subject-level differences
configuration differences
checkpoint differences
calibration-index differences
prediction-hash differences
```

## Stage D: multi-seed benchmark

Only after passing the gate:

```text
run seeds 0–4
run conventional baselines
run F0 original RR readout
run F0 + A1
run requested clean ablations
```

---

# 23. Required files

For every seed, save:

```text
seed_XXX/f0/subject_summary.csv
seed_XXX/f0/summary.csv
seed_XXX/f0/manifest.json

seed_XXX/a1/subject_summary.csv
seed_XXX/a1/summary.csv
seed_XXX/a1/manifest.json

seed_XXX/baselines/subject_summary.csv
seed_XXX/baselines/summary.csv
seed_XXX/baselines/manifest.json
```

Alternatively, F0 and A1 may share a directory if:

* rows are separated by `metric_source`;
* files do not overwrite each other;
* checkpoint lineage is explicit.

---

# 24. Acceptance criteria

The implementation is complete only when:

1. The exact historical F0 readout function has been identified and reused.
2. Profile metadata collection does not alter held-out evaluation by default.
3. F0 training uses 20 epochs and batch size 16.
4. The original RR probe uses the historical complete configuration.
5. Exactly 32 deterministic target calibration windows are selected.
6. One fixed profile is reused across all target test batches.
7. Predictions are invariant to test batch partition and order.
8. F0 native, pre-probe, post-probe, STFT, auxiliary, and A1 outputs remain separate.
9. The historical post-profile readout is the primary F0 endpoint.
10. The native RR head is diagnostic only.
11. Baseline correlations are numeric and never silently replaced with `NaN`.
12. No-FiLM and no-QKV are genuine one-factor ablations.
13. Fold seeds are independent of execution order.
14. A1 contains exactly 20 trainable parameters.
15. A1 uses Gaussian-soft targets with sigma 1.0 BPM.
16. A1 returns expected RR from its probability distribution.
17. A1 uses source labels only.
18. A1 performs no target-specific parameter update.
19. F0 original and A1 share the same F0 checkpoint hash.
20. F0 original and A1 share the same test-index hash.
21. The primary statistical reference is F0 original RR readout.
22. Historical validation and checkpoint-selection behaviour are preserved and documented.
23. Unit tests pass without dataset access, model training, CUDA, subprocesses, dry runs, or smoke runs.
24. Seed 0 passes the equivalence gate before seeds 1–4 are launched.

---

# 25. Do not do the following

Do not:

* redesign the F0 architecture;
* change loss weights;
* change transformer or TCN dimensions;
* introduce normalised-profile training;
* change source validation splitting in the faithful reproduction;
* change F0 checkpoint selection in the faithful reproduction;
* apply CLSA;
* apply fast QKV test-time adaptation;
* apply feature-mean alignment;
* fit F0 or A1 using held-out RR labels;
* retrain the F0 backbone separately for A1;
* replace the original F0 readout with the native head;
* make A1 the hard-coded primary reference;
* run seeds 1–4 before seed-0 equivalence;
* refactor unrelated project code.

---

# 26. Final implementation report

After completing the code changes and unit tests, report:

```text
files changed
functions changed
historical F0 function identified
historical F0 configuration confirmed
profile-normalisation fix
fixed-profile implementation
baseline metric fix
ablation corrections
A1 integration
tests added
test results
remaining limitations
commands required for Stage A
commands required for Stage B
```

Do not launch the full multi-seed benchmark as part of the implementation task.

