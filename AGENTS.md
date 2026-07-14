# AGENTS.md

## Project

This repository contains experiments for predicting respiratory rate and mental
workload from wearable IMU signals.

The respiratory-rate experiments use leave-one-subject-out evaluation and must
preserve strict subject separation.

## Important files

- `tcnet_experiment.py`
  - Existing plain TCN respiratory-rate baseline.
  - Treat its architecture, preprocessing and LOSO split as the baseline source
    of truth.

- `dataloader.py`
  - Dataset loading, subject selection and window construction.

- `augmentations.py`
  - Existing signal augmentations.

- `evaluations.py`
  - Shared evaluation metrics.

- `experiment_recorder.py`
  - Experiment configuration, manifests and output recording.

- `adaptation_utils.py`
  - Existing adaptation utilities.

- `config.py`
  - Shared paths and configuration defaults.

## General implementation rules

1. Prefer standalone experiment files rather than modifying existing validated
   experiments.
2. Reuse existing dataset, model and evaluation code through imports.
3. Do not duplicate the TCN architecture unless wrapping it is unavoidable.
4. Do not silently alter:
   - subject lists;
   - LOSO splits;
   - preprocessing;
   - signal sampling rates;
   - window lengths;
   - target definitions;
   - metrics.
5. New experiment outputs must be written to a new timestamped directory.
6. Never overwrite existing checkpoints or result folders.
7. Preserve backward compatibility with existing scripts.

## Reproducibility

Every experiment must record:

- random seed;
- subject list;
- held-out subject;
- command-line arguments;
- git commit;
- hostname;
- Python version;
- PyTorch version;
- CUDA version;
- trainable parameter count.

Seed Python, NumPy, PyTorch, CUDA and DataLoader workers.

## Subject leakage rules

The held-out subject must not be used for:

- source training;
- source validation;
- hyperparameter selection;
- checkpoint selection;
- feature normalisation;
- profile normalisation;
- adaptation-gate training;
- target-profile model fitting.

Target labels may only be accessed after target predictions have been finalised.

Add runtime assertions for these rules when implementing profiling, adaptation
or test-time training.

## Evaluation rules

The held-out subject is the unit of generalisation.

Report metrics per subject and seed before aggregation.

Required RR metrics:

- MAE;
- RMSE;
- Pearson correlation;
- bias;
- median absolute error;
- 95th percentile absolute error;
- number of windows.

Do not use individual windows as independent samples for inferential
statistics.

## Coding conventions

- Use type annotations for new public functions.
- Use dataclasses for structured model outputs and adaptation state.
- Avoid broad exception handling.
- Fail clearly when required checkpoint metadata is missing.
- Keep configurable values in argparse rather than hard-coding them.
- Use existing project naming conventions where possible.
- Add concise comments for leakage-sensitive or mathematically non-obvious code.

## Testing

Before a full run:

1. run a single-subject, single-seed smoke test;
2. use no more than two epochs;
3. limit the number of batches;
4. verify all expected files are generated;
5. verify no target-label leakage;
6. verify deterministic repeatability.

Do not launch full multi-subject or multi-seed experiments unless explicitly
requested.

## Existing-file protection

Do not modify the following files for standalone profiling or TTT experiments
unless explicitly instructed:

- `tcnet_experiment.py`
- `dataloader.py`
- `evaluations.py`
- `augmentations.py`

Prefer wrappers, subclasses and helper modules.

## Current experiment specification

For the TCN subject-profile and test-time-training experiment, follow:

`docs/experiments/tcn_profile_ttt_spec.md`

Implement it in phases:

1. static profiling controls T0–T3;
2. one-step TTT T4–T5;
3. meta-gated TTT T6;
4. optional profile-gated mean alignment T7.

Do not proceed to the next phase until the current phase passes its smoke tests
and leakage checks.
