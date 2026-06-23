# JBHI source baselines with uploaded TSLib PatchTST and TimesNet

This patch adds new files rather than modifying the earlier no-shortcuts files.
It directly wraps the uploaded `PatchTST.py` and `TimesNet.py` source-style
modules.

## New files

- `rr_jbhi_tslib_source_models.py`
- `run_rr_jbhi_tslib_neural_baselines.py`
- `run_rr_jbhi_tslib_main_suite.sh`

## Required source files

Copy these into the repo root, or pass absolute paths:

```bash
PatchTST.py
TimesNet.py
```

Both files depend on the TSLib `layers/` package. The wrapper intentionally does
not vendor or reimplement those layers. If `layers/` is unavailable, the run will
fail loudly rather than silently using a simplified model.

## Model names

```text
resnet1d
cnn_gru
tcn
inceptiontime
patchtst_tslib
timesnet_tslib
```

Aliases are also accepted:

```text
patchtst_official
timesnet_official
```

## PatchTST adaptation

The provided PatchTST model exposes the original task-dispatch interface and a
classification branch. For scalar RR regression, the wrapper sets:

```text
task_name = classification
num_class = 1
```

The model's scalar output is trained with SmoothL1 regression loss by the runner.
This is a task-head adaptation, not a replacement of PatchTST internals.

## TimesNet adaptation

The provided TimesNet model includes a `task_name='regression'` branch. For RR,
the wrapper sets:

```text
task_name = regression
pred_len = 1
c_out = 1
```

and trains the scalar output against RR.

## Run

```bash
PATCHTST_FILE=/absolute/path/to/PatchTST.py \
TIMESNET_FILE=/absolute/path/to/TimesNet.py \
ROOT=/projects/BLVMob/imu-rr-seated/results/jbhi_tslib_source_v1 \
bash run_rr_jbhi_tslib_main_suite.sh
```

For a quick native-only run:

```bash
MODELS="resnet1d cnn_gru tcn inceptiontime" \
bash run_rr_jbhi_tslib_main_suite.sh
```

For TSLib baselines:

```bash
MODELS="patchtst_tslib timesnet_tslib" \
PATCHTST_FILE=PatchTST.py \
TIMESNET_FILE=TimesNet.py \
bash run_rr_jbhi_tslib_main_suite.sh
```

## What this does not do

This does not run the main cross-modal method. The main method must still be run
through the full `vit_pressure_crossmodal_stft_rr_*` adaptation ladder scripts.
