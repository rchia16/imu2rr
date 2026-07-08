#!/usr/bin/env bash
set -euo pipefail

# Focused E-series ablation for shared-profile FiLM-QKV p_t adaptation without
# STFT-derived RR pseudo-targets.

RUN_ID="${RUN_ID:-shared_film_qkv_no_stft_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-/projects/BLVMob/imu-rr-seated/results/${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

LADDER="${LADDER:-vit_pressure_crossmodal_stft_rr_adaptation_alpha_hat_sweep.py}"
SUMMARIZER="${SUMMARIZER:-summarize_rr_adaptation_alpha_hat_sweep.py}"

DATA_STR="${DATA_STR:-imu_filt}"
DATA_DIR="${DATA_DIR:-/projects/BLVMob/imu-rr-seated/Data}"
DATA_GROUP="${DATA_GROUP:-mr}"
MDL_DIR="${MDL_DIR:-/projects/BLVMob/imu-rr-seated/models/imu_filt/loocv}"
DEVICE="${DEVICE:-cuda:0}"
DEVICES="${DEVICES:-${DEVICE}}"

SUBJECTS_WAS_SET="${SUBJECTS+x}"
EVAL_SUBJECTS_WAS_SET="${EVAL_SUBJECTS+x}"
EPOCHS_WAS_SET="${EPOCHS+x}"
BATCH_SIZE_WAS_SET="${BATCH_SIZE+x}"
RR_PROBE_EPOCHS_WAS_SET="${RR_PROBE_EPOCHS+x}"
UNSUP_FEATURE_EPOCHS_WAS_SET="${UNSUP_FEATURE_EPOCHS+x}"
TARGET_CALIBRATION_WINDOWS_WAS_SET="${TARGET_CALIBRATION_WINDOWS+x}"
FEATURE_SOURCE_ANCHOR_WINDOWS_WAS_SET="${FEATURE_SOURCE_ANCHOR_WINDOWS+x}"

SUBJECTS="${SUBJECTS:-S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29}"
EVAL_SUBJECTS="${EVAL_SUBJECTS:-${SUBJECTS}}"
MODES="${MODES:-tcn_profile_film_qkv_last1_0p01 tcn_profile_film_qkv_last1_0p01_pt_no_stft tcn_profile_film_qkv_last1_0p01_pt_aux_only tcn_profile_film_qkv_last1_0p01_pt_reg_only tcn_profile_film_qkv_last1_0p01_pt_no_stft_budget}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
RR_PROBE_EPOCHS="${RR_PROBE_EPOCHS:-100}"
UNSUP_FEATURE_EPOCHS="${UNSUP_FEATURE_EPOCHS:-10}"
RR_HEAD_TYPE="${RR_HEAD_TYPE:-mlp}"
DECODER_MODE="${DECODER_MODE:-cross_attn}"
TARGET_CALIBRATION_WINDOWS="${TARGET_CALIBRATION_WINDOWS:-32}"
TARGET_CALIBRATION_MODE="${TARGET_CALIBRATION_MODE:-random}"
FEATURE_SOURCE_ANCHOR_WINDOWS="${FEATURE_SOURCE_ANCHOR_WINDOWS:-512}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"
DEBUG="${DEBUG:-0}"

if [[ "${DEBUG}" == "1" ]]; then
  if [[ -z "${SUBJECTS_WAS_SET}" ]]; then SUBJECTS="S12 S13 S14"; fi
  if [[ -z "${EVAL_SUBJECTS_WAS_SET}" ]]; then EVAL_SUBJECTS="S12"; fi
  if [[ -z "${EPOCHS_WAS_SET}" ]]; then EPOCHS="1"; fi
  if [[ -z "${BATCH_SIZE_WAS_SET}" ]]; then BATCH_SIZE="4"; fi
  if [[ -z "${RR_PROBE_EPOCHS_WAS_SET}" ]]; then RR_PROBE_EPOCHS="1"; fi
  if [[ -z "${UNSUP_FEATURE_EPOCHS_WAS_SET}" ]]; then UNSUP_FEATURE_EPOCHS="1"; fi
  if [[ -z "${TARGET_CALIBRATION_WINDOWS_WAS_SET}" ]]; then TARGET_CALIBRATION_WINDOWS="4"; fi
  if [[ -z "${FEATURE_SOURCE_ANCHOR_WINDOWS_WAS_SET}" ]]; then FEATURE_SOURCE_ANCHOR_WINDOWS="8"; fi
fi

export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

mkdir -p "${OUT_DIR}/logs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

run_chunk() {
  local chunk_device="$1"
  local chunk_run_id="$2"
  local chunk_log_file="$3"
  shift 3
  local chunk_eval_subjects=("$@")
  local skip_args=()
  if [[ "${SKIP_COMPLETED}" == "1" ]]; then
    skip_args+=(--skip-completed)
  fi

  "${PYTHON_BIN}" -u "${LADDER}" \
    --subjects ${SUBJECTS} \
    --eval-subjects "${chunk_eval_subjects[@]}" \
    --data-str "${DATA_STR}" \
    --data-dir "${DATA_DIR}" \
    --data-group "${DATA_GROUP}" \
    --mdl-dir "${MDL_DIR}" \
    --out-dir "${OUT_DIR}" \
    --sweep-root "${OUT_DIR}" \
    --sweep-run-id "${chunk_run_id}" \
    --device "${chunk_device}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --rr-probe-epochs "${RR_PROBE_EPOCHS}" \
    --unsup-feature-epochs "${UNSUP_FEATURE_EPOCHS}" \
    --decoder-mode "${DECODER_MODE}" \
    --rr-head-type "${RR_HEAD_TYPE}" \
    --use-profile-film \
    --use-profile-qkv \
    --shared-profile-qkv \
    --profile-conditioning film_qkv \
    --profile-qkv-layers last1 \
    --profile-qkv-scale 0.01 \
    --profile-qkv-residual \
    --use-tcn-token-mixer \
    --tcn-mixer-alpha 0.05 \
    --rr-tta-modes "${MODES}" \
    --use-unsup-mode-defaults \
    --target-calibration-windows "${TARGET_CALIBRATION_WINDOWS}" \
    --target-calibration-mode "${TARGET_CALIBRATION_MODE}" \
    --feature-source-anchor-windows "${FEATURE_SOURCE_ANCHOR_WINDOWS}" \
    --profile-unsup-adapt-scope calibration \
    --include-calibration-in-eval \
    --adaptation-use-calibration-only \
    "${skip_args[@]}" \
    2>&1 | tee "${chunk_log_file}"
}

devices=(${DEVICES})
eval_subject_array=(${EVAL_SUBJECTS})

echo "[RUN] shared FiLM-QKV no-STFT ablation"
echo "[RUN] subjects: ${SUBJECTS}"
echo "[RUN] eval subjects: ${EVAL_SUBJECTS}"
echo "[RUN] modes: ${MODES}"
echo "[RUN] devices: ${DEVICES}"
echo "[RUN] out: ${OUT_DIR}"

if (( ${#devices[@]} > 1 && ${#eval_subject_array[@]} > 1 )); then
  pids=()
  for slot in "${!devices[@]}"; do
    chunk_subjects=()
    for idx in "${!eval_subject_array[@]}"; do
      if (( idx % ${#devices[@]} == slot )); then
        chunk_subjects+=("${eval_subject_array[$idx]}")
      fi
    done
    if (( ${#chunk_subjects[@]} == 0 )); then
      continue
    fi
    chunk_run_id="${RUN_ID}_gpu${slot}"
    chunk_log_file="${OUT_DIR}/logs/${chunk_run_id}_${STAMP}.log"
    echo "[RUN] ${devices[$slot]} -> ${chunk_subjects[*]}"
    run_chunk "${devices[$slot]}" "${chunk_run_id}" "${chunk_log_file}" "${chunk_subjects[@]}" &
    pids+=("$!")
  done

  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if (( status != 0 )); then
    exit "${status}"
  fi
else
  run_chunk "${DEVICE}" "${RUN_ID}" "${OUT_DIR}/logs/${RUN_ID}_${STAMP}.log" "${eval_subject_array[@]}"
fi

"${PYTHON_BIN}" -u "${SUMMARIZER}" \
  --root "${OUT_DIR}" \
  --out-csv "${OUT_DIR}/summary.csv" \
  --combined-subject-csv "${OUT_DIR}/subject_rows.csv" \
  2>&1 | tee "${OUT_DIR}/logs/summary_${STAMP}.log"

echo "[DONE] shared FiLM-QKV no-STFT ablation root: ${OUT_DIR}"
echo "[DONE] summary: ${OUT_DIR}/summary.csv"
