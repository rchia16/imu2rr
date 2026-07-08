#!/usr/bin/env bash
set -euo pipefail

# Subject-level multi-GPU scheduler for episodic profile TTT.
#
# This script is for the strict TTT protocol:
#   for each eval sample or small eval batch:
#     restore source-trained model state
#     initialize a fresh profile vector from only that unlabeled input
#     optimize self-supervised TTT loss on that input
#     predict that same input
#     discard the adapted profile vector before the next independent input
#
# It intentionally does NOT run calibration-prefix transductive adaptation.
# CAL_WINDOWS is kept only to exclude the first N target windows from evaluation.
#
# Recommended smoke test:
#   DEBUG=1 EXPERIMENT_SET=ttt_qkv_sample PROFILE_QKV_SCALE=0.03 \
#     DEVICES="cuda:0" SUBJECTS="S13 S19 S22 S25" bash run_profile_encoder_rr_ttt.sh
#
# Strict sample-level FiLM TTT:
#   EXPERIMENT_SET=ttt_film_sample DEVICES="cuda:0 cuda:1" bash run_profile_encoder_rr_ttt.sh
#
# Strict sample-level QKV TTT:
#   EXPERIMENT_SET=ttt_qkv_sample PROFILE_QKV_SCALE=0.03 \
#     DEVICES="cuda:0 cuda:1" bash run_profile_encoder_rr_ttt.sh
#
# Small-batch episodic QKV TTT:
#   EXPERIMENT_SET=ttt_qkv_batch TTT_BATCH_SIZE=8 PROFILE_QKV_SCALE=0.03 \
#     DEVICES="cuda:0 cuda:1" bash run_profile_encoder_rr_ttt.sh

PYTHON_BIN="${PYTHON_BIN:-python}"

# New episodic TTT script.
SCRIPT="${SCRIPT:-vit_pressure_crossmodal_profile_ttt_episodic.py}"

ROOT_DIR="${ROOT_DIR:-/projects/BLVMob/imu-rr-seated}"
RESULT_DIR_USER="${RESULT_DIR:-}"
TIMESTAMP_SUFFIX="${TIMESTAMP_SUFFIX:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_DIR_BASE="${RESULT_DIR_BASE:-${ROOT_DIR}/results/profile_encoder_rr_ttt}"

# shellcheck disable=SC2206
DEVICES=(${DEVICES:-cuda:0 cuda:1})

DATA_STR="${DATA_STR:-imu_filt}"
DATA_GROUP="${DATA_GROUP:-mr}"
CAL_WINDOWS="${CAL_WINDOWS:-32}"
SEED="${SEED:-0}"
DEBUG="${DEBUG:-0}"
EXPERIMENT_SET="${EXPERIMENT_SET:-ttt_film_sample}"

PROFILE_DIM="${PROFILE_DIM:-32}"
PROFILE_HIDDEN_DIM="${PROFILE_HIDDEN_DIM:-128}"
PROFILE_FILM_SCALE="${PROFILE_FILM_SCALE:-0.1}"
PROFILE_QKV_LAYERS="${PROFILE_QKV_LAYERS:-last1}"
PROFILE_STATS_MAX_BATCHES="${PROFILE_STATS_MAX_BATCHES:-50}"

# QKV should default small for episodic TTT unless explicitly overridden.
case "${EXPERIMENT_SET}" in
  ttt_qkv_sample|ttt_qkv_batch|ttt_qkv_compare)
    PROFILE_QKV_SCALE="${PROFILE_QKV_SCALE:-0.03}"
    ;;
  *)
    PROFILE_QKV_SCALE="${PROFILE_QKV_SCALE:-0.1}"
    ;;
esac

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

# Full LOSO source cohort.
# shellcheck disable=SC2206
FULL_SUBJECTS=(${FULL_SUBJECTS:-S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29})

TTT_FILM_SAMPLE_MODES=(
  none
  profile_film_ttt_sample
)

TTT_FILM_BATCH_MODES=(
  none
  profile_film_ttt_batch
)

TTT_QKV_SAMPLE_MODES=(
  none
  profile_qkv_ttt_sample
)

TTT_QKV_BATCH_MODES=(
  none
  profile_qkv_ttt_batch
)

TTT_FILM_COMPARE_MODES=(
  none
  profile_film_ttt_sample
  profile_film_ttt_batch
)

TTT_QKV_COMPARE_MODES=(
  none
  profile_qkv_ttt_sample
  profile_qkv_ttt_batch
)

select_mode_set() {
  local preset="$1"
  case "${preset}" in
    ttt_film_sample)
      RUN_MODES=("${TTT_FILM_SAMPLE_MODES[@]}")
      ;;
    ttt_film_batch)
      RUN_MODES=("${TTT_FILM_BATCH_MODES[@]}")
      ;;
    ttt_qkv_sample)
      RUN_MODES=("${TTT_QKV_SAMPLE_MODES[@]}")
      ;;
    ttt_qkv_batch)
      RUN_MODES=("${TTT_QKV_BATCH_MODES[@]}")
      ;;
    ttt_film_compare)
      RUN_MODES=("${TTT_FILM_COMPARE_MODES[@]}")
      ;;
    ttt_qkv_compare)
      RUN_MODES=("${TTT_QKV_COMPARE_MODES[@]}")
      ;;
    *)
      echo "[ERROR] unsupported EXPERIMENT_SET=${preset}" >&2
      echo "[ERROR] valid values: ttt_film_sample ttt_film_batch ttt_qkv_sample ttt_qkv_batch ttt_film_compare ttt_qkv_compare" >&2
      exit 1
      ;;
  esac
}

if [ "${DEBUG}" = "1" ]; then
  EPOCHS="${EPOCHS:-10}"
  RR_PROBE_EPOCHS="${RR_PROBE_EPOCHS:-30}"
else
  EPOCHS="${EPOCHS:-20}"
  RR_PROBE_EPOCHS="${RR_PROBE_EPOCHS:-100}"
fi

if [ -n "${MODES:-}" ]; then
  # shellcheck disable=SC2206
  RUN_MODES=(${MODES})
else
  select_mode_set "${EXPERIMENT_SET}"
fi

if [ "$#" -gt 0 ]; then
  echo "[WARN] ignoring positional subject arguments: $*"
  echo "[WARN] use SUBJECTS=... to run a subset; defaulting to the full LOSO cohort"
fi

if [ -n "${SUBJECTS:-}" ]; then
  # shellcheck disable=SC2206
  EVAL_SUBJECTS=(${SUBJECTS})
else
  EVAL_SUBJECTS=("${FULL_SUBJECTS[@]}")
fi

sanitize_token() {
  echo "$1" | tr ':/ .,' '_____' | tr -cd 'A-Za-z0-9_.=-'
}

sanitize_device() {
  echo "$1" | tr ':/' '__'
}

EXPERIMENT_TAG="${EXPERIMENT_SET}"
case "${EXPERIMENT_SET}" in
  ttt_qkv_sample|ttt_qkv_batch|ttt_qkv_compare)
    EXPERIMENT_TAG="${EXPERIMENT_TAG}_scale$(sanitize_token "${PROFILE_QKV_SCALE}")_layers$(sanitize_token "${PROFILE_QKV_LAYERS}")"
    ;;
esac

EXPERIMENT_TAG="${EXPERIMENT_TAG}_bs$(sanitize_token "${TTT_BATCH_SIZE}")_steps$(sanitize_token "${TTT_INNER_STEPS}")_lr$(sanitize_token "${TTT_LR}")"

if [ -z "${RESULT_DIR_USER}" ]; then
  if [ "${DEBUG}" = "1" ]; then
    RESULT_DIR_BASE="${ROOT_DIR}/results/profile_encoder_rr_ttt_debug"
  fi
  RESULT_DIR="${RESULT_DIR_BASE}_${EXPERIMENT_TAG}_${TIMESTAMP_SUFFIX}"
else
  RESULT_DIR="${RESULT_DIR_USER}"
fi

validate_ttt_mode_family() {
  local has_film=0
  local has_qkv=0
  local mode
  for mode in "${RUN_MODES[@]}"; do
    case "${mode}" in
      profile_film_*) has_film=1 ;;
      profile_qkv_*) has_qkv=1 ;;
    esac
  done

  if [ "${has_film}" = "1" ] && [ "${has_qkv}" = "1" ]; then
    echo "[ERROR] RUN_MODES mixes profile_film_* and profile_qkv_* modes." >&2
    echo "[ERROR] Run FiLM and QKV TTT experiments in separate jobs." >&2
    exit 1
  fi
}

validate_entrypoint() {
  if [ ! -f "${SCRIPT}" ]; then
    echo "[ERROR] episodic TTT entrypoint not found: ${SCRIPT}" >&2
    echo "[ERROR] Save the new script as vit_pressure_crossmodal_profile_ttt_episodic.py or set SCRIPT=..." >&2
    exit 1
  fi

  if ! grep -q "episodic" "${SCRIPT}"; then
    echo "[WARN] ${SCRIPT} does not contain the word 'episodic'; verify this is the TTT script." >&2
  fi

  if ! grep -q "ttt-modes" "${SCRIPT}"; then
    echo "[ERROR] ${SCRIPT} does not appear to expose --ttt-modes." >&2
    exit 1
  fi
}

HELP_TEXT=""
load_help_text() {
  set +e
  HELP_TEXT="$("${PYTHON_BIN}" "${SCRIPT}" --help 2>&1)"
  local status=$?
  set -e
  if [ "${status}" -ne 0 ]; then
    echo "[ERROR] could not read ${SCRIPT} --help; status=${status}" >&2
    echo "${HELP_TEXT}" >&2
    exit 1
  fi
}

supports_flag() {
  local flag="$1"
  grep -q -- "${flag}" <<< "${HELP_TEXT}"
}

require_flag() {
  local flag="$1"
  if ! supports_flag "${flag}"; then
    echo "[ERROR] ${SCRIPT} does not support required flag ${flag}" >&2
    exit 1
  fi
}

COMMON_ARGS=(
  --data-str "${DATA_STR}"
  --data-dir "${ROOT_DIR}/Data/"
  --data-group "${DATA_GROUP}"
  --mdl-dir "${ROOT_DIR}/models/${DATA_STR}/loocv"

  --epochs "${EPOCHS}"
  --batch-size 64
  --lr 3e-4
  --weight-decay 1e-4
  --d-model 128
  --layers 2
  --heads 8
  --lambda-stft 1.0
  --lambda-rr 0.01
  --lambda-contrast 0.1
  --contrast-warmup-epochs 5
  --contrast-ramp-end-epoch 10
  --seed "${SEED}"
  --tta none
  --tta-epochs 0

  --rr-probe-epochs "${RR_PROBE_EPOCHS}"
  --rr-probe-lr 1e-3
  --rr-probe-weight-decay 1e-4
  --rr-probe-batch-size 256

  # Calibration windows are NOT used for transductive adaptation here.
  # They are only excluded from evaluation by the episodic TTT script.
  --target-calibration-windows "${CAL_WINDOWS}"
  --target-calibration-mode first
  --exclude-calibration-from-eval

  --profile-dim "${PROFILE_DIM}"
  --profile-hidden-dim "${PROFILE_HIDDEN_DIM}"
  --profile-film-scale "${PROFILE_FILM_SCALE}"
  --profile-qkv-scale "${PROFILE_QKV_SCALE}"
  --profile-qkv-layers "${PROFILE_QKV_LAYERS}"
  --profile-stats-max-batches "${PROFILE_STATS_MAX_BATCHES}"

  --ttt-batch-size "${TTT_BATCH_SIZE}"
  --ttt-inner-steps "${TTT_INNER_STEPS}"
  --ttt-lr "${TTT_LR}"
  --ttt-weight-decay "${TTT_WEIGHT_DECAY}"

  --ttt-probe-stft-weight "${TTT_PROBE_STFT_WEIGHT}"
  --ttt-aux-stft-weight "${TTT_AUX_STFT_WEIGHT}"
  --ttt-profile-prior-weight "${TTT_PROFILE_PRIOR_WEIGHT}"
  --ttt-smoothness-weight "${TTT_SMOOTHNESS_WEIGHT}"

  --ttt-stft-confidence-floor "${TTT_STFT_CONFIDENCE_FLOOR}"
  --ttt-stft-confidence-power "${TTT_STFT_CONFIDENCE_POWER}"
  --ttt-rr-disagreement-threshold "${TTT_RR_DISAGREEMENT_THRESHOLD}"
  --ttt-rr-min-bpm "${TTT_RR_MIN_BPM}"
  --ttt-rr-max-bpm "${TTT_RR_MAX_BPM}"
  --ttt-max-temporal-jump-bpm "${TTT_MAX_TEMPORAL_JUMP_BPM}"
)

validate_entrypoint
validate_ttt_mode_family
load_help_text

require_flag "--ttt-modes"
require_flag "--ttt-batch-size"
require_flag "--ttt-inner-steps"
require_flag "--ttt-lr"
require_flag "--ttt-probe-stft-weight"
require_flag "--ttt-aux-stft-weight"
require_flag "--ttt-profile-prior-weight"

subject_log_path() {
  local subject="$1"
  local device="$2"
  local safe_device
  safe_device="$(sanitize_device "${device}")"
  echo "${RESULT_DIR}/logs/${subject}__${safe_device}.log"
}

write_run_manifest() {
  mkdir -p "${RESULT_DIR}/logs"
  {
    echo "timestamp_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "experiment_set=${EXPERIMENT_SET}"
    echo "experiment_tag=${EXPERIMENT_TAG}"
    echo "result_dir=${RESULT_DIR}"
    echo "script=${SCRIPT}"
    echo "debug=${DEBUG}"
    echo "devices=${DEVICES[*]}"
    echo "full_subjects=${FULL_SUBJECTS[*]}"
    echo "eval_subjects=${EVAL_SUBJECTS[*]}"
    echo "modes=${RUN_MODES[*]}"
    echo "epochs=${EPOCHS}"
    echo "rr_probe_epochs=${RR_PROBE_EPOCHS}"
    echo "seed=${SEED}"
    echo "data_str=${DATA_STR}"
    echo "data_group=${DATA_GROUP}"
    echo "cal_windows_excluded_from_eval=${CAL_WINDOWS}"
    echo "protocol=episodic_reset_each_sample_or_batch"
    echo "uses_target_rr_labels_for_adaptation=0"
    echo "uses_future_windows=0"
    echo "carries_state_between_batches=0"
    echo "restores_source_state_each_batch=1"
    echo "adapted_object=profile_vector_only"
    echo "profile_dim=${PROFILE_DIM}"
    echo "profile_hidden_dim=${PROFILE_HIDDEN_DIM}"
    echo "profile_film_scale=${PROFILE_FILM_SCALE}"
    echo "profile_qkv_scale=${PROFILE_QKV_SCALE}"
    echo "profile_qkv_layers=${PROFILE_QKV_LAYERS}"
    echo "profile_stats_max_batches=${PROFILE_STATS_MAX_BATCHES}"
    echo "ttt_batch_size=${TTT_BATCH_SIZE}"
    echo "ttt_inner_steps=${TTT_INNER_STEPS}"
    echo "ttt_lr=${TTT_LR}"
    echo "ttt_weight_decay=${TTT_WEIGHT_DECAY}"
    echo "ttt_probe_stft_weight=${TTT_PROBE_STFT_WEIGHT}"
    echo "ttt_aux_stft_weight=${TTT_AUX_STFT_WEIGHT}"
    echo "ttt_profile_prior_weight=${TTT_PROFILE_PRIOR_WEIGHT}"
    echo "ttt_smoothness_weight=${TTT_SMOOTHNESS_WEIGHT}"
    echo "ttt_stft_confidence_floor=${TTT_STFT_CONFIDENCE_FLOOR}"
    echo "ttt_stft_confidence_power=${TTT_STFT_CONFIDENCE_POWER}"
    echo "ttt_rr_disagreement_threshold=${TTT_RR_DISAGREEMENT_THRESHOLD}"
    echo "ttt_rr_min_bpm=${TTT_RR_MIN_BPM}"
    echo "ttt_rr_max_bpm=${TTT_RR_MAX_BPM}"
    echo "ttt_max_temporal_jump_bpm=${TTT_MAX_TEMPORAL_JUMP_BPM}"
  } | tee "${RESULT_DIR}/logs/run_manifest.env"
}

GPU_PIDS=()
GPU_LABELS=()
GPU_LOGS=()

for _ in "${DEVICES[@]}"; do
  GPU_PIDS+=("")
  GPU_LABELS+=("")
  GPU_LOGS+=("")
done

FAILED=0
CLEANUP_DONE=0

cleanup_running_jobs() {
  if [ "${CLEANUP_DONE}" -ne 0 ]; then
    return 0
  fi
  CLEANUP_DONE=1

  local i pid
  for i in "${!GPU_PIDS[@]}"; do
    pid="${GPU_PIDS[$i]}"
    if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
      echo "[CLEANUP] stopping ${GPU_LABELS[$i]} pid=${pid}"
      pkill -TERM -P "${pid}" 2>/dev/null || true
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done

  sleep 2

  for i in "${!GPU_PIDS[@]}"; do
    pid="${GPU_PIDS[$i]}"
    if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
      pkill -KILL -P "${pid}" 2>/dev/null || true
      kill -KILL "${pid}" 2>/dev/null || true
    fi
    GPU_PIDS[$i]=""
    GPU_LABELS[$i]=""
    GPU_LOGS[$i]=""
  done
}

handle_interrupt() {
  echo
  echo "[INTERRUPT] terminating running subject jobs"
  cleanup_running_jobs
  exit 130
}

trap handle_interrupt INT TERM

reap_completed_job() {
  local finished_pid status i label log

  set +e
  wait -n -p finished_pid 2>/dev/null
  status=$?
  set -e

  if [ -z "${finished_pid:-}" ]; then
    return 1
  fi

  for i in "${!GPU_PIDS[@]}"; do
    if [ "${GPU_PIDS[$i]}" = "${finished_pid}" ]; then
      label="${GPU_LABELS[$i]}"
      log="${GPU_LOGS[$i]}"

      if [ "${status}" -eq 0 ]; then
        echo "[DONE] ${label} log=${log}" >&2
      else
        echo "[FAIL] ${label} status=${status} log=${log}" >&2
        FAILED=1
      fi

      GPU_PIDS[$i]=""
      GPU_LABELS[$i]=""
      GPU_LOGS[$i]=""
      return 0
    fi
  done

  echo "[WARN] reaped unexpected pid=${finished_pid} status=${status}" >&2
  return 0
}

wait_for_free_gpu() {
  local i
  while true; do
    if [ "${FAILED}" -ne 0 ]; then
      return 1
    fi

    for i in "${!GPU_PIDS[@]}"; do
      if [ -z "${GPU_PIDS[$i]}" ]; then
        echo "${i}"
        return 0
      fi
    done

    if ! reap_completed_job; then
      echo "[ABORT] could not reap a completed subject job"
      return 1
    fi
  done
}

launch_subject() {
  local gpu_idx="$1"
  local subject="$2"
  local device="${DEVICES[$gpu_idx]}"
  local run_id="${subject}"
  local out_dir="${RESULT_DIR}/subjects/${subject}"
  local log_file
  log_file="$(subject_log_path "${subject}" "${device}")"

  mkdir -p "${out_dir}" "${RESULT_DIR}/logs"

  local cmd=(
    "${PYTHON_BIN}" "${SCRIPT}"
    "${COMMON_ARGS[@]}"
    --device "${device}"
    --subjects "${FULL_SUBJECTS[@]}"
    --eval-subjects "${subject}"
    --ttt-modes "${RUN_MODES[*]}"
    --out-dir "${out_dir}"
    --sweep-root "${RESULT_DIR}"
    --sweep-run-id "${run_id}"
  )

  echo "[RUN] subject=${subject} device=${device}"
  echo "[LOG] ${log_file}"

  {
    echo "============================================================"
    echo "timestamp_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "subject=${subject}"
    echo "device=${device}"
    echo "experiment_set=${EXPERIMENT_SET}"
    echo "experiment_tag=${EXPERIMENT_TAG}"
    echo "debug=${DEBUG}"
    echo "epochs=${EPOCHS}"
    echo "rr_probe_epochs=${RR_PROBE_EPOCHS}"
    echo "modes=${RUN_MODES[*]}"
    echo "full_subjects=${FULL_SUBJECTS[*]}"
    echo "result_dir=${RESULT_DIR}"
    echo "script=${SCRIPT}"
    echo "protocol=episodic_reset_each_sample_or_batch"
    echo "uses_target_rr_labels_for_adaptation=0"
    echo "uses_future_windows=0"
    echo "carries_state_between_batches=0"
    echo "restores_source_state_each_batch=1"
    echo "cal_windows_excluded_from_eval=${CAL_WINDOWS}"
    echo "profile_film_scale=${PROFILE_FILM_SCALE}"
    echo "profile_qkv_scale=${PROFILE_QKV_SCALE}"
    echo "profile_qkv_layers=${PROFILE_QKV_LAYERS}"
    echo "ttt_batch_size=${TTT_BATCH_SIZE}"
    echo "ttt_inner_steps=${TTT_INNER_STEPS}"
    echo "ttt_lr=${TTT_LR}"
    echo "ttt_probe_stft_weight=${TTT_PROBE_STFT_WEIGHT}"
    echo "ttt_aux_stft_weight=${TTT_AUX_STFT_WEIGHT}"
    echo "ttt_profile_prior_weight=${TTT_PROFILE_PRIOR_WEIGHT}"
    echo "ttt_smoothness_weight=${TTT_SMOOTHNESS_WEIGHT}"
    echo "ttt_stft_confidence_floor=${TTT_STFT_CONFIDENCE_FLOOR}"
    echo "ttt_stft_confidence_power=${TTT_STFT_CONFIDENCE_POWER}"
    echo "python_command=${cmd[*]}"
    echo "============================================================"
    echo
  } | tee "${log_file}"

  (
    set -euo pipefail
    "${cmd[@]}"
  ) 2>&1 | tee -a "${log_file}" &

  GPU_PIDS[$gpu_idx]="$!"
  GPU_LABELS[$gpu_idx]="subject=${subject} device=${device}"
  GPU_LOGS[$gpu_idx]="${log_file}"
}

echo "============================================================"
echo "Profile encoder episodic TTT subject-level sweep"
echo "DEBUG=${DEBUG}"
echo "EXPERIMENT_SET=${EXPERIMENT_SET}"
echo "EXPERIMENT_TAG=${EXPERIMENT_TAG}"
echo "TIMESTAMP_SUFFIX=${TIMESTAMP_SUFFIX}"
echo "RESULT_DIR=${RESULT_DIR}"
echo "DEVICES=${DEVICES[*]}"
echo "FULL_SUBJECTS=${FULL_SUBJECTS[*]}"
echo "EVAL_SUBJECTS=${EVAL_SUBJECTS[*]}"
echo "MODES=${RUN_MODES[*]}"
echo "EPOCHS=${EPOCHS}"
echo "RR_PROBE_EPOCHS=${RR_PROBE_EPOCHS}"
echo "PROTOCOL=episodic_reset_each_sample_or_batch"
echo "USES_TARGET_RR_LABELS_FOR_ADAPTATION=0"
echo "USES_FUTURE_WINDOWS=0"
echo "CARRIES_STATE_BETWEEN_BATCHES=0"
echo "RESTORES_SOURCE_STATE_EACH_BATCH=1"
echo "CAL_WINDOWS_EXCLUDED_FROM_EVAL=${CAL_WINDOWS}"
echo "PROFILE_FILM_SCALE=${PROFILE_FILM_SCALE}"
echo "PROFILE_QKV_SCALE=${PROFILE_QKV_SCALE}"
echo "PROFILE_QKV_LAYERS=${PROFILE_QKV_LAYERS}"
echo "TTT_BATCH_SIZE=${TTT_BATCH_SIZE}"
echo "TTT_INNER_STEPS=${TTT_INNER_STEPS}"
echo "TTT_LR=${TTT_LR}"
echo "TTT_PROBE_STFT_WEIGHT=${TTT_PROBE_STFT_WEIGHT}"
echo "TTT_AUX_STFT_WEIGHT=${TTT_AUX_STFT_WEIGHT}"
echo "TTT_PROFILE_PRIOR_WEIGHT=${TTT_PROFILE_PRIOR_WEIGHT}"
echo "TTT_SMOOTHNESS_WEIGHT=${TTT_SMOOTHNESS_WEIGHT}"
echo "TTT_STFT_CONFIDENCE_FLOOR=${TTT_STFT_CONFIDENCE_FLOOR}"
echo "TTT_STFT_CONFIDENCE_POWER=${TTT_STFT_CONFIDENCE_POWER}"
echo "Logs: ${RESULT_DIR}/logs/"
echo "============================================================"

mkdir -p "${RESULT_DIR}/logs"
write_run_manifest

for subject in "${EVAL_SUBJECTS[@]}"; do
  gpu_idx="$(wait_for_free_gpu)" || {
    echo "[ABORT] stopping launch because an earlier subject failed"
    cleanup_running_jobs
    exit 1
  }
  launch_subject "${gpu_idx}" "${subject}"
done

while true; do
  if [ "${FAILED}" -ne 0 ]; then
    echo "[ABORT] at least one subject failed; terminating remaining jobs"
    cleanup_running_jobs
    exit 1
  fi

  running=0
  for pid in "${GPU_PIDS[@]}"; do
    if [ -n "${pid}" ]; then
      running=1
      break
    fi
  done

  if [ "${running}" -eq 0 ]; then
    break
  fi

  if ! reap_completed_job; then
    echo "[ABORT] could not reap a completed subject job"
    cleanup_running_jobs
    exit 1
  fi
done

# The episodic TTT Python script writes these directly at the sweep root.
COMPARISON_CSV="${RESULT_DIR}/episodic_ttt_comparison.csv"
SUBJECT_CSV="${RESULT_DIR}/episodic_ttt_subject_rows.csv"

{
  echo "============================================================"
  echo "timestamp_summary=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "experiment_set=${EXPERIMENT_SET}"
  echo "experiment_tag=${EXPERIMENT_TAG}"
  echo "result_dir=${RESULT_DIR}"
  echo "modes=${RUN_MODES[*]}"
  echo "protocol=episodic_reset_each_sample_or_batch"
  echo "profile_qkv_scale=${PROFILE_QKV_SCALE}"
  echo "ttt_batch_size=${TTT_BATCH_SIZE}"
  echo "ttt_inner_steps=${TTT_INNER_STEPS}"
  echo "ttt_lr=${TTT_LR}"
  echo "comparison_csv=${COMPARISON_CSV}"
  echo "subject_csv=${SUBJECT_CSV}"
  echo "============================================================"
} | tee "${RESULT_DIR}/logs/summary.log"

if [ -f "${COMPARISON_CSV}" ]; then
  echo "[SUMMARY] comparison already written by ${SCRIPT}: ${COMPARISON_CSV}" | tee -a "${RESULT_DIR}/logs/summary.log"
else
  echo "[WARN] expected comparison CSV not found: ${COMPARISON_CSV}" | tee -a "${RESULT_DIR}/logs/summary.log"
fi

if [ -f "${SUBJECT_CSV}" ]; then
  echo "[SUMMARY] subject rows already written by ${SCRIPT}: ${SUBJECT_CSV}" | tee -a "${RESULT_DIR}/logs/summary.log"
else
  echo "[WARN] expected subject CSV not found: ${SUBJECT_CSV}" | tee -a "${RESULT_DIR}/logs/summary.log"
fi

echo "[DONE] All subject jobs finished."
echo "Comparison: ${COMPARISON_CSV}"
echo "Subject rows: ${SUBJECT_CSV}"
echo "Manifest: ${RESULT_DIR}/logs/run_manifest.env"
echo "Logs: ${RESULT_DIR}/logs/"
