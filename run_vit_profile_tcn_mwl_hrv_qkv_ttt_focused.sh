#!/usr/bin/env bash
set -euo pipefail

# Focused binary MWL HR/HRV late-fusion diagnostic.
#
# Debug:
#   DEBUG=1 SUBJECTS="S13 S19 S22 S25" DEVICES="cuda:0" \
#     bash run_vit_profile_tcn_mwl_hrv_qkv_ttt_focused.sh
#
# Full:
#   DEVICES="cuda:0 cuda:1" bash run_vit_profile_tcn_mwl_hrv_qkv_ttt_focused.sh

EXPERIMENT_SET="${EXPERIMENT_SET:-binary_hrv_focus}"
TIMESTAMP_SUFFIX="${TIMESTAMP_SUFFIX:-$(date -u +%Y%m%dT%H%M%SZ)}"
TASKS="${TASKS:-binary_low_high}"
HEADS="${HEADS:-A0 A2}"
MWL_TTT_MODES="${MWL_TTT_MODES:-none profile_qkv_ttt_sample}"
HRV_FEATURE_MODES="${HRV_FEATURE_MODES:-none hr_hrv_relative}"

MWL_TCN_HIDDEN_DIM="${MWL_TCN_HIDDEN_DIM:-32}"
MWL_TCN_LAYERS="${MWL_TCN_LAYERS:-1}"
MWL_DROPOUT="${MWL_DROPOUT:-0.4}"
MWL_WEIGHT_DECAY="${MWL_WEIGHT_DECAY:-1e-2}"
MWL_LR="${MWL_LR:-3e-4}"
MWL_EPOCHS="${MWL_EPOCHS:-100}"
MWL_VAL_SUBJECTS="${MWL_VAL_SUBJECTS:-3}"
MWL_PATIENCE="${MWL_PATIENCE:-8}"
MWL_MONITOR="${MWL_MONITOR:-val_f1_macro}"
PROFILE_QKV_LAYERS="${PROFILE_QKV_LAYERS:-last1}"
PROFILE_QKV_SCALE="${PROFILE_QKV_SCALE:-0.03}"
SAVE_CONFUSION_MATRICES="${SAVE_CONFUSION_MATRICES:-1}"
SAVE_TTT_DIAGNOSTICS="${SAVE_TTT_DIAGNOSTICS:-1}"
USE_HRV_FUSION="${USE_HRV_FUSION:-1}"
HRV_INPUT_SOURCE="${HRV_INPUT_SOURCE:-auto}"
MWL_POSTPROCESS_SECONDS="${MWL_POSTPROCESS_SECONDS:-60}"
MWL_WINDOW_SHIFT_SECONDS="${MWL_WINDOW_SHIFT_SECONDS:-30}"

export EXPERIMENT_SET
export TIMESTAMP_SUFFIX
export MWL_DIAGNOSTIC_TASKS="$TASKS"
export MWL_HEAD_VARIANTS="$HEADS"
export MWL_TTT_MODES
export HRV_FEATURE_MODES
export MWL_TCN_HIDDEN_DIM
export MWL_TCN_LAYERS
export MWL_DROPOUT
export MWL_WEIGHT_DECAY
export MWL_LR
export MWL_EPOCHS
export MWL_VAL_SUBJECTS
export MWL_PATIENCE
export MWL_MONITOR
export PROFILE_QKV_LAYERS
export PROFILE_QKV_SCALE
export SAVE_CONFUSION_MATRICES
export SAVE_TTT_DIAGNOSTICS
export USE_HRV_FUSION
export HRV_INPUT_SOURCE
export MWL_POSTPROCESS_SECONDS
export MWL_WINDOW_SHIFT_SECONDS

bash run_vit_profile_tcn_mwl_qkv_ttt_1min.sh

if [ -z "${RESULT_DIR:-}" ]; then
  ROOT_DIR="${ROOT_DIR:-/projects/BLVMob/imu-rr-seated}"
  TIMESTAMP_SUFFIX="${TIMESTAMP_SUFFIX:-$(date -u +%Y%m%dT%H%M%SZ)}"
  RESULT_DIR_BASE="${RESULT_DIR_BASE:-${ROOT_DIR}/results/vit_profile_tcn_mwl_qkv_ttt_1min}"
  RESULT_DIR="${RESULT_DIR_BASE}/${EXPERIMENT_SET}_${TIMESTAMP_SUFFIX}"
fi

RESULT_DIR="$RESULT_DIR" python - <<'PY'
import os
from pathlib import Path
import pandas as pd

root = Path(os.environ["RESULT_DIR"])
chunks = root / "chunks"
if not chunks.exists():
    raise SystemExit(0)
for name in ["summary", "mwl_diagnostic_summary", "mwl_predictions_window", "mwl_predictions_one_min", "mwl_ttt_batches"]:
    files = sorted(chunks.glob(f"*_{name}.csv"))
    if files:
        pd.concat([pd.read_csv(f) for f in files], ignore_index=True).to_csv(root / f"{name}.csv", index=False)
PY

echo "[FOCUSED_DONE] result_dir=$RESULT_DIR"
