#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU LOSO scheduler for the separate MWL experiment:
#   IMU -> RR/STFT profile-conditioned source model
#   + TCN Profile-FiLM mental-workload classification head
#   + episodic last-layer profile-QKV TTT
#   + non-overlapping 1-minute post-processing of MWL probabilities
#
# Default smoke/full use:
#   DEBUG=1 DEVICES="cuda:0" SUBJECTS="S13 S19 S22 S25" \
#     bash run_vit_profile_tcn_mwl_qkv_ttt_1min.sh
#
#   DEVICES="cuda:0 cuda:1" EXPERIMENT_SET=qkv_sample \
#     bash run_vit_profile_tcn_mwl_qkv_ttt_1min.sh
#
# Main presets:
#   EXPERIMENT_SET=qkv_sample      -> none + profile_qkv_ttt_sample
#   EXPERIMENT_SET=qkv_batch       -> none + profile_qkv_ttt_batch
#   EXPERIMENT_SET=qkv_compare     -> none + sample + batch
#   EXPERIMENT_SET=qkv_cal_gated   -> none + sample, with QKV calibration gate
#   EXPERIMENT_SET=film_sample     -> none + profile_film_ttt_sample
#   EXPERIMENT_SET=film_batch      -> none + profile_film_ttt_batch
#   EXPERIMENT_SET=film_compare    -> none + sample + batch

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="${SCRIPT:-vit_profile_tcn_mwl_qkv_ttt_1min.py}"
ROOT_DIR="${ROOT_DIR:-/projects/BLVMob/imu-rr-seated}"
RESULT_DIR_USER="${RESULT_DIR:-}"
TIMESTAMP_SUFFIX="${TIMESTAMP_SUFFIX:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_DIR_BASE="${RESULT_DIR_BASE:-${ROOT_DIR}/results/vit_profile_tcn_mwl_qkv_ttt_1min}"

# shellcheck disable=SC2206
DEVICES=(${DEVICES:-cuda:0 cuda:1})
DATA_STR="${DATA_STR:-imu_filt}"
DATA_GROUP="${DATA_GROUP:-mr}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/Data}"
MDL_DIR="${MDL_DIR:-${ROOT_DIR}/models/${DATA_STR}/loocv}"
SEED="${SEED:-0}"
DEBUG="${DEBUG:-0}"
EXPERIMENT_SET="${EXPERIMENT_SET:-qkv_sample}"

# Source RR/STFT model training defaults.
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
D_MODEL="${D_MODEL:-128}"
LAYERS="${LAYERS:-2}"
HEADS="${HEADS:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
RESUME="${RESUME:-1}"

# Source losses.
LAMBDA_STFT="${LAMBDA_STFT:-1.0}"
LAMBDA_RR="${LAMBDA_RR:-0.1}"
LAMBDA_RR_STFT_CONSISTENCY="${LAMBDA_RR_STFT_CONSISTENCY:-0.05}"
LAMBDA_PROFILE_PRIOR="${LAMBDA_PROFILE_PRIOR:-0.01}"
LAMBDA_CONTRAST="${LAMBDA_CONTRAST:-0.05}"
CONTRAST_WARMUP_EPOCHS="${CONTRAST_WARMUP_EPOCHS:-5}"
CONTRAST_RAMP_END_EPOCH="${CONTRAST_RAMP_END_EPOCH:-10}"

# Profile-conditioning defaults. Last-layer QKV is intentional here.
PROFILE_DIM="${PROFILE_DIM:-32}"
PROFILE_HIDDEN_DIM="${PROFILE_HIDDEN_DIM:-128}"
PROFILE_FILM_SCALE="${PROFILE_FILM_SCALE:-0.1}"
PROFILE_QKV_LAYERS="${PROFILE_QKV_LAYERS:-last1}"
PROFILE_QKV_SCALE="${PROFILE_QKV_SCALE:-0.03}"
PROFILE_STATS_MAX_BATCHES="${PROFILE_STATS_MAX_BATCHES:-50}"

# RR probe used only for unsupervised QKV/FiLM episodic TTT losses.
RR_PROBE_EPOCHS="${RR_PROBE_EPOCHS:-100}"
RR_PROBE_LR="${RR_PROBE_LR:-1e-3}"
RR_PROBE_WEIGHT_DECAY="${RR_PROBE_WEIGHT_DECAY:-1e-4}"
RR_PROBE_BATCH_SIZE="${RR_PROBE_BATCH_SIZE:-256}"
RR_PROBE_SOURCE_BATCHES="${RR_PROBE_SOURCE_BATCHES:-0}"
RR_PROBE_TRAIN_ADAPTER="${RR_PROBE_TRAIN_ADAPTER:-1}"

# Episodic TTT hyperparameters.
TTT_BATCH_SIZE="${TTT_BATCH_SIZE:-1}"
TTT_INNER_STEPS="${TTT_INNER_STEPS:-5}"
TTT_LR="${TTT_LR:-3e-4}"
TTT_WEIGHT_DECAY="${TTT_WEIGHT_DECAY:-0.0}"
TTT_PROBE_STFT_WEIGHT="${TTT_PROBE_STFT_WEIGHT:-0.05}"
TTT_AUX_STFT_WEIGHT="${TTT_AUX_STFT_WEIGHT:-0.05}"
TTT_PROFILE_PRIOR_WEIGHT="${TTT_PROFILE_PRIOR_WEIGHT:-0.01}"
TTT_SMOOTHNESS_WEIGHT="${TTT_SMOOTHNESS_WEIGHT:-0.0}"
TTT_STFT_CONFIDENCE_FLOOR="${TTT_STFT_CONFIDENCE_FLOOR:-0.0}"
TTT_STFT_CONFIDENCE_POWER="${TTT_STFT_CONFIDENCE_POWER:-2.0}"
TTT_RR_DISAGREEMENT_THRESHOLD="${TTT_RR_DISAGREEMENT_THRESHOLD:-12.0}"
TTT_RR_MIN_BPM="${TTT_RR_MIN_BPM:-4.0}"
TTT_RR_MAX_BPM="${TTT_RR_MAX_BPM:-45.0}"
TTT_MAX_TEMPORAL_JUMP_BPM="${TTT_MAX_TEMPORAL_JUMP_BPM:-10.0}"

# QKV safety controls.
PROFILE_QKV_RESIDUAL="${PROFILE_QKV_RESIDUAL:-0}"
PROFILE_QKV_ALPHA_INIT="${PROFILE_QKV_ALPHA_INIT:-1.0}"
PROFILE_QKV_ALPHA_MAX="${PROFILE_QKV_ALPHA_MAX:-1.0}"
PROFILE_QKV_CALIBRATION_GATE="${PROFILE_QKV_CALIBRATION_GATE:-0}"
PROFILE_QKV_CAL_GATE_TOLERANCE="${PROFILE_QKV_CAL_GATE_TOLERANCE:-0.05}"
PROFILE_QKV_CAL_GATE_METRIC="${PROFILE_QKV_CAL_GATE_METRIC:-mae}"
PROFILE_QKV_CAL_GATE_FALLBACK="${PROFILE_QKV_CAL_GATE_FALLBACK:-base}"

# Calibration/eval prefix controls inherited from common adaptation args.
TARGET_CALIBRATION_WINDOWS="${TARGET_CALIBRATION_WINDOWS:-32}"
TARGET_CALIBRATION_MODE="${TARGET_CALIBRATION_MODE:-first}"
EXCLUDE_CALIBRATION_FROM_EVAL="${EXCLUDE_CALIBRATION_FROM_EVAL:-1}"

# MWL downstream classification defaults.
MWL_TASK="${MWL_TASK:-mr_levels}"
MWL_TRAIN_DATA_GROUP="${MWL_TRAIN_DATA_GROUP:-mr_levels}"
MWL_TEST_DATA_GROUP="${MWL_TEST_DATA_GROUP:-mr_levels}"
MWL_INCLUDE_LEVELS_IN_TRAIN="${MWL_INCLUDE_LEVELS_IN_TRAIN:-1}"
MWL_CLASS_SUBSET="${MWL_CLASS_SUBSET:-}"
MWL_DIAGNOSTIC_TASKS="${MWL_DIAGNOSTIC_TASKS:-levels,binary_low_high,rest_vs_load}"
MWL_HEAD_VARIANTS="${MWL_HEAD_VARIANTS:-A0,A1,A2}"
MWL_BATCH_SIZE="${MWL_BATCH_SIZE:-64}"
MWL_EPOCHS="${MWL_EPOCHS:-100}"
MWL_LR="${MWL_LR:-3e-4}"
MWL_WEIGHT_DECAY="${MWL_WEIGHT_DECAY:-1e-2}"
MWL_TCN_HIDDEN_DIM="${MWL_TCN_HIDDEN_DIM:-32}"
MWL_TCN_LAYERS="${MWL_TCN_LAYERS:-1}"
MWL_TCN_KERNEL_SIZE="${MWL_TCN_KERNEL_SIZE:-3}"
MWL_DROPOUT="${MWL_DROPOUT:-0.4}"
MWL_PROFILE_FILM_SCALE="${MWL_PROFILE_FILM_SCALE:-0.1}"
MWL_EARLY_STOP="${MWL_EARLY_STOP:-1}"
MWL_VAL_SUBJECTS="${MWL_VAL_SUBJECTS:-3}"
MWL_PATIENCE="${MWL_PATIENCE:-8}"
MWL_MONITOR="${MWL_MONITOR:-val_f1_macro}"
MWL_POSTPROCESS_SECONDS="${MWL_POSTPROCESS_SECONDS:-60}"
MWL_WINDOW_SHIFT_SECONDS="${MWL_WINDOW_SHIFT_SECONDS:-1}"

# Full LOSO source cohort used in the current profile encoder experiments.
# shellcheck disable=SC2206
FULL_SUBJECTS=(${FULL_SUBJECTS:-S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29})

if [ -n "${SUBJECTS:-}" ]; then
  # shellcheck disable=SC2206
  RUN_SUBJECTS=(${SUBJECTS})
elif [ "$DEBUG" = "1" ]; then
  RUN_SUBJECTS=(S13 S19 S22 S25)
else
  RUN_SUBJECTS=("${FULL_SUBJECTS[@]}")
fi

case "$EXPERIMENT_SET" in
  qkv_sample)
    MWL_TTT_MODES="${MWL_TTT_MODES:-none profile_qkv_ttt_sample}"
    ;;
  qkv_batch)
    MWL_TTT_MODES="${MWL_TTT_MODES:-none profile_qkv_ttt_batch}"
    TTT_BATCH_SIZE="${TTT_BATCH_SIZE:-8}"
    ;;
  qkv_compare)
    MWL_TTT_MODES="${MWL_TTT_MODES:-none profile_qkv_ttt_sample profile_qkv_ttt_batch}"
    ;;
  qkv_cal_gated)
    MWL_TTT_MODES="${MWL_TTT_MODES:-none profile_qkv_ttt_sample}"
    PROFILE_QKV_CALIBRATION_GATE=1
    ;;
  film_sample)
    MWL_TTT_MODES="${MWL_TTT_MODES:-none profile_film_ttt_sample}"
    ;;
  film_batch)
    MWL_TTT_MODES="${MWL_TTT_MODES:-none profile_film_ttt_batch}"
    TTT_BATCH_SIZE="${TTT_BATCH_SIZE:-8}"
    ;;
  film_compare)
    MWL_TTT_MODES="${MWL_TTT_MODES:-none profile_film_ttt_sample profile_film_ttt_batch}"
    ;;
  *)
    echo "[ERROR] Unknown EXPERIMENT_SET=${EXPERIMENT_SET}" >&2
    exit 2
    ;;
esac

if [ -n "$RESULT_DIR_USER" ]; then
  RESULT_DIR="$RESULT_DIR_USER"
else
  RESULT_DIR="${RESULT_DIR_BASE}/${EXPERIMENT_SET}_${TIMESTAMP_SUFFIX}"
fi
LOG_ROOT="${RESULT_DIR}/logs"
mkdir -p "$LOG_ROOT" "$RESULT_DIR"

if [ ! -f "$SCRIPT" ]; then
  echo "[ERROR] Script not found: $SCRIPT" >&2
  echo "[HINT] Run from the repo root, copy the script there, or set SCRIPT=/path/to/vit_profile_tcn_mwl_qkv_ttt_1min.py" >&2
  exit 1
fi

GPU_PIDS=()
GPU_LABELS=()
for _ in "${DEVICES[@]}"; do
  GPU_PIDS+=("")
  GPU_LABELS+=("")
done

cleanup_running_jobs() {
  local gpu_index
  for gpu_index in "${!GPU_PIDS[@]}"; do
    local pid="${GPU_PIDS[$gpu_index]}"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo "[CLEANUP] stopping ${GPU_LABELS[$gpu_index]:-unknown} pid=$pid"
      pkill -TERM -P "$pid" 2>/dev/null || true
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
}
trap 'echo; echo "[INTERRUPT] terminating spawned runs"; cleanup_running_jobs; exit 130' INT TERM

sanitize_device() {
  echo "$1" | tr ':/' '__'
}

bool_flag() {
  local value="$1"
  local true_flag="$2"
  local false_flag="${3:-}"
  if [ "$value" = "1" ] || [ "$value" = "true" ] || [ "$value" = "TRUE" ]; then
    echo "$true_flag"
  elif [ -n "$false_flag" ]; then
    echo "$false_flag"
  fi
}

build_common_args() {
  local subject="$1"
  local device="$2"
  CMD_ARGS=(
    "$SCRIPT"
    --subjects "${FULL_SUBJECTS[@]}"
    --eval-subjects "$subject"
    --data-str "$DATA_STR"
    --data-dir "$DATA_DIR"
    --data-group "$DATA_GROUP"
    --mdl-dir "$MDL_DIR"
    --out-dir "$RESULT_DIR"
    --device "$device"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --lr "$LR"
    --weight-decay "$WEIGHT_DECAY"
    --d-model "$D_MODEL"
    --layers "$LAYERS"
    --heads "$HEADS"
    --num-workers "$NUM_WORKERS"
    --seed "$SEED"
    --lambda-stft "$LAMBDA_STFT"
    --lambda-rr "$LAMBDA_RR"
    --lambda-rr-stft-consistency "$LAMBDA_RR_STFT_CONSISTENCY"
    --lambda-profile-prior "$LAMBDA_PROFILE_PRIOR"
    --lambda-contrast "$LAMBDA_CONTRAST"
    --contrast-warmup-epochs "$CONTRAST_WARMUP_EPOCHS"
    --contrast-ramp-end-epoch "$CONTRAST_RAMP_END_EPOCH"
    --profile-dim "$PROFILE_DIM"
    --profile-hidden-dim "$PROFILE_HIDDEN_DIM"
    --profile-film-scale "$PROFILE_FILM_SCALE"
    --profile-qkv-scale "$PROFILE_QKV_SCALE"
    --profile-qkv-layers "$PROFILE_QKV_LAYERS"
    --profile-stats-max-batches "$PROFILE_STATS_MAX_BATCHES"
    --rr-probe-epochs "$RR_PROBE_EPOCHS"
    --rr-probe-lr "$RR_PROBE_LR"
    --rr-probe-weight-decay "$RR_PROBE_WEIGHT_DECAY"
    --rr-probe-batch-size "$RR_PROBE_BATCH_SIZE"
    --rr-probe-source-batches "$RR_PROBE_SOURCE_BATCHES"
    --ttt-batch-size "$TTT_BATCH_SIZE"
    --ttt-inner-steps "$TTT_INNER_STEPS"
    --ttt-lr "$TTT_LR"
    --ttt-weight-decay "$TTT_WEIGHT_DECAY"
    --ttt-probe-stft-weight "$TTT_PROBE_STFT_WEIGHT"
    --ttt-aux-stft-weight "$TTT_AUX_STFT_WEIGHT"
    --ttt-profile-prior-weight "$TTT_PROFILE_PRIOR_WEIGHT"
    --ttt-smoothness-weight "$TTT_SMOOTHNESS_WEIGHT"
    --ttt-stft-confidence-floor "$TTT_STFT_CONFIDENCE_FLOOR"
    --ttt-stft-confidence-power "$TTT_STFT_CONFIDENCE_POWER"
    --ttt-rr-disagreement-threshold "$TTT_RR_DISAGREEMENT_THRESHOLD"
    --ttt-rr-min-bpm "$TTT_RR_MIN_BPM"
    --ttt-rr-max-bpm "$TTT_RR_MAX_BPM"
    --ttt-max-temporal-jump-bpm "$TTT_MAX_TEMPORAL_JUMP_BPM"
    --profile-qkv-alpha-init "$PROFILE_QKV_ALPHA_INIT"
    --profile-qkv-alpha-max "$PROFILE_QKV_ALPHA_MAX"
    --profile-qkv-cal-gate-tolerance "$PROFILE_QKV_CAL_GATE_TOLERANCE"
    --profile-qkv-cal-gate-metric "$PROFILE_QKV_CAL_GATE_METRIC"
    --profile-qkv-cal-gate-fallback "$PROFILE_QKV_CAL_GATE_FALLBACK"
    --target-calibration-windows "$TARGET_CALIBRATION_WINDOWS"
    --target-calibration-mode "$TARGET_CALIBRATION_MODE"
    --mwl-task "$MWL_TASK"
    --mwl-train-data-group "$MWL_TRAIN_DATA_GROUP"
    --mwl-test-data-group "$MWL_TEST_DATA_GROUP"
    --mwl-class-subset "$MWL_CLASS_SUBSET"
    --mwl-diagnostic-tasks "$MWL_DIAGNOSTIC_TASKS"
    --mwl-head-variants "$MWL_HEAD_VARIANTS"
    --mwl-ttt-modes "$MWL_TTT_MODES"
    --mwl-batch-size "$MWL_BATCH_SIZE"
    --mwl-epochs "$MWL_EPOCHS"
    --mwl-lr "$MWL_LR"
    --mwl-weight-decay "$MWL_WEIGHT_DECAY"
    --mwl-tcn-hidden-dim "$MWL_TCN_HIDDEN_DIM"
    --mwl-tcn-layers "$MWL_TCN_LAYERS"
    --mwl-tcn-kernel-size "$MWL_TCN_KERNEL_SIZE"
    --mwl-dropout "$MWL_DROPOUT"
    --mwl-profile-film-scale "$MWL_PROFILE_FILM_SCALE"
    --mwl-val-subjects "$MWL_VAL_SUBJECTS"
    --mwl-patience "$MWL_PATIENCE"
    --mwl-monitor "$MWL_MONITOR"
    --mwl-postprocess-seconds "$MWL_POSTPROCESS_SECONDS"
    --mwl-window-shift-seconds "$MWL_WINDOW_SHIFT_SECONDS"
  )

  if [ "$RESUME" = "1" ]; then CMD_ARGS+=(--resume); fi
  if [ "$RR_PROBE_TRAIN_ADAPTER" = "1" ]; then CMD_ARGS+=(--rr-probe-train-adapter); fi
  if [ "$PROFILE_QKV_RESIDUAL" = "1" ]; then CMD_ARGS+=(--profile-qkv-residual); fi
  if [ "$PROFILE_QKV_CALIBRATION_GATE" = "1" ]; then CMD_ARGS+=(--profile-qkv-calibration-gate); fi
  if [ "$EXCLUDE_CALIBRATION_FROM_EVAL" = "1" ]; then CMD_ARGS+=(--exclude-calibration-from-eval); fi
  if [ "$MWL_INCLUDE_LEVELS_IN_TRAIN" = "1" ]; then CMD_ARGS+=(--mwl-include-levels-in-train); else CMD_ARGS+=(--no-mwl-include-levels-in-train); fi
  if [ "$MWL_EARLY_STOP" = "1" ]; then CMD_ARGS+=(--mwl-early-stop); else CMD_ARGS+=(--no-mwl-early-stop); fi

  case "$MWL_TTT_MODES" in
    *profile_qkv*) CMD_ARGS+=(--use-profile-qkv --profile-conditioning qkv) ;;
    *profile_film*) CMD_ARGS+=(--use-profile-film --profile-conditioning film) ;;
  esac
}

print_command() {
  printf '%q ' "$PYTHON_BIN" "${CMD_ARGS[@]}"
  printf '\n'
}

run_subject_on_device() {
  local subject="$1"
  local device="$2"
  local logfile="$LOG_ROOT/${subject}__$(sanitize_device "$device").log"
  build_common_args "$subject" "$device"
  echo "[RUN] subject=$subject device=$device log=$logfile"
  {
    echo "timestamp_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "subject=$subject"
    echo "device=$device"
    echo "experiment_set=$EXPERIMENT_SET"
    echo "mwl_ttt_modes=$MWL_TTT_MODES"
    echo "result_dir=$RESULT_DIR"
    echo "script=$SCRIPT"
    echo
    echo "[COMMAND]"
    print_command
    echo
    "$PYTHON_BIN" "${CMD_ARGS[@]}"
    echo
    echo "timestamp_end=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$logfile" 2>&1
}

write_manifest() {
  local manifest="$RESULT_DIR/manifest.txt"
  {
    echo "timestamp_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "result_dir=$RESULT_DIR"
    echo "script=$SCRIPT"
    echo "experiment_set=$EXPERIMENT_SET"
    echo "devices=${DEVICES[*]}"
    echo "subjects=${RUN_SUBJECTS[*]}"
    echo "data_str=$DATA_STR"
    echo "data_group=$DATA_GROUP"
    echo "mwl_task=$MWL_TASK"
    echo "mwl_train_data_group=$MWL_TRAIN_DATA_GROUP"
    echo "mwl_test_data_group=$MWL_TEST_DATA_GROUP"
    echo "mwl_class_subset=$MWL_CLASS_SUBSET"
    echo "mwl_diagnostic_tasks=$MWL_DIAGNOSTIC_TASKS"
    echo "mwl_head_variants=$MWL_HEAD_VARIANTS"
    echo "mwl_ttt_modes=$MWL_TTT_MODES"
    echo "mwl_early_stop=$MWL_EARLY_STOP"
    echo "mwl_val_subjects=$MWL_VAL_SUBJECTS"
    echo "mwl_patience=$MWL_PATIENCE"
    echo "mwl_monitor=$MWL_MONITOR"
    echo "profile_qkv_layers=$PROFILE_QKV_LAYERS"
    echo "profile_qkv_scale=$PROFILE_QKV_SCALE"
    echo "mwl_postprocess_seconds=$MWL_POSTPROCESS_SECONDS"
    echo "mwl_window_shift_seconds=$MWL_WINDOW_SHIFT_SECONDS"
  } > "$manifest"
}

write_manifest

echo "[START] result_dir=$RESULT_DIR"
echo "[START] subjects=${RUN_SUBJECTS[*]}"
echo "[START] devices=${DEVICES[*]}"
echo "[START] mwl_ttt_modes=$MWL_TTT_MODES"

NEXT_SUBJECT_INDEX=0
FAILED=0
while [ "$NEXT_SUBJECT_INDEX" -lt "${#RUN_SUBJECTS[@]}" ]; do
  for gpu_index in "${!DEVICES[@]}"; do
    pid="${GPU_PIDS[$gpu_index]}"
    if [ -n "$pid" ]; then
      if kill -0 "$pid" 2>/dev/null; then
        continue
      fi
      if ! wait "$pid"; then
        echo "[ERROR] ${GPU_LABELS[$gpu_index]} failed" >&2
        FAILED=1
      fi
      GPU_PIDS[$gpu_index]=""
      GPU_LABELS[$gpu_index]=""
    fi

    if [ "$NEXT_SUBJECT_INDEX" -lt "${#RUN_SUBJECTS[@]}" ]; then
      subject="${RUN_SUBJECTS[$NEXT_SUBJECT_INDEX]}"
      device="${DEVICES[$gpu_index]}"
      run_subject_on_device "$subject" "$device" &
      GPU_PIDS[$gpu_index]="$!"
      GPU_LABELS[$gpu_index]="$subject on $device"
      NEXT_SUBJECT_INDEX=$((NEXT_SUBJECT_INDEX + 1))
    fi
  done
  sleep 5
done

for gpu_index in "${!GPU_PIDS[@]}"; do
  pid="${GPU_PIDS[$gpu_index]}"
  if [ -n "$pid" ]; then
    if ! wait "$pid"; then
      echo "[ERROR] ${GPU_LABELS[$gpu_index]} failed" >&2
      FAILED=1
    fi
  fi
done

if [ "$FAILED" -ne 0 ]; then
  echo "[DONE_WITH_ERRORS] result_dir=$RESULT_DIR" >&2
  exit 1
fi

echo "[DONE] result_dir=$RESULT_DIR"
echo "[DONE] logs=$LOG_ROOT"
