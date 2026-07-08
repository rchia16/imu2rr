#!/usr/bin/env bash
set -euo pipefail

# Focused F-series for embedded profile-conditioned CLSA-QKV fast adaptation.
# Each variant trains/loads its own source checkpoint because static-QKV and
# CLSA-QKV have different source-time forward paths.

RUN_ID="${RUN_ID:-profile_clsa_qkv_fseries_$(date -u +%Y%m%dT%H%M%SZ)}"
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
TARGET_CALIBRATION_WINDOWS_WAS_SET="${TARGET_CALIBRATION_WINDOWS+x}"
FEATURE_SOURCE_ANCHOR_WINDOWS_WAS_SET="${FEATURE_SOURCE_ANCHOR_WINDOWS+x}"

SUBJECTS="${SUBJECTS:-S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29}"
EVAL_SUBJECTS="${EVAL_SUBJECTS:-${SUBJECTS}}"
VARIANTS="${VARIANTS:-F0_static_shared F1_clsa_no_film F2_film_clsa F3_film_clsa_no_fast F4_film_clsa_rank4}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
RR_PROBE_EPOCHS="${RR_PROBE_EPOCHS:-100}"
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
  if [[ -z "${TARGET_CALIBRATION_WINDOWS_WAS_SET}" ]]; then TARGET_CALIBRATION_WINDOWS="4"; fi
  if [[ -z "${FEATURE_SOURCE_ANCHOR_WINDOWS_WAS_SET}" ]]; then FEATURE_SOURCE_ANCHOR_WINDOWS="8"; fi
fi

export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/runs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MANIFEST="${OUT_DIR}/manifest.tsv"
if [[ ! -f "${MANIFEST}" ]]; then
  printf "variant\tmode\tdescription\trun_dir\textra_args\n" > "${MANIFEST}"
fi

contains_variant() {
  local wanted="$1"
  local item
  for item in ${VARIANTS}; do
    if [[ "${item}" == "${wanted}" ]]; then
      return 0
    fi
  done
  return 1
}

run_variant() {
  local variant="$1"
  local mode="$2"
  local description="$3"
  shift 3
  local extra_args=("$@")
  local run_dir="${OUT_DIR}/runs/${variant}"
  mkdir -p "${run_dir}"
  printf "%s\t%s\t%s\t%s\t%s\n" "${variant}" "${mode}" "${description}" "${run_dir}" "${extra_args[*]}" >> "${MANIFEST}"

  local devices=(${DEVICES})
  local eval_subject_array=(${EVAL_SUBJECTS})
  local skip_args=()
  if [[ "${SKIP_COMPLETED}" == "1" ]]; then
    skip_args+=(--skip-completed)
  fi

  run_chunk() {
    local chunk_device="$1"
    local chunk_run_id="$2"
    local chunk_log_file="$3"
    shift 3
    local chunk_eval_subjects=("$@")

    "${PYTHON_BIN}" -u "${LADDER}" \
      --subjects ${SUBJECTS} \
      --eval-subjects "${chunk_eval_subjects[@]}" \
      --data-str "${DATA_STR}" \
      --data-dir "${DATA_DIR}" \
      --data-group "${DATA_GROUP}" \
      --mdl-dir "${MDL_DIR}" \
      --out-dir "${run_dir}" \
      --sweep-root "${run_dir}" \
      --sweep-run-id "${chunk_run_id}" \
      --device "${chunk_device}" \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --rr-probe-epochs "${RR_PROBE_EPOCHS}" \
      --decoder-mode "${DECODER_MODE}" \
      --rr-head-type "${RR_HEAD_TYPE}" \
      --rr-tta-modes "${mode}" \
      --use-unsup-mode-defaults \
      --target-calibration-windows "${TARGET_CALIBRATION_WINDOWS}" \
      --target-calibration-mode "${TARGET_CALIBRATION_MODE}" \
      --feature-source-anchor-windows "${FEATURE_SOURCE_ANCHOR_WINDOWS}" \
      --profile-unsup-adapt-scope calibration \
      --include-calibration-in-eval \
      --adaptation-use-calibration-only \
      "${extra_args[@]}" \
      "${skip_args[@]}" \
      2>&1 | tee "${chunk_log_file}"
  }

  echo "[RUN] ${variant}: ${description}"
  echo "[RUN] mode=${mode} out=${run_dir}"

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
      chunk_run_id="${variant}_gpu${slot}"
      chunk_log_file="${OUT_DIR}/logs/${chunk_run_id}_${STAMP}.log"
      echo "[RUN] ${variant}: ${devices[$slot]} -> ${chunk_subjects[*]}"
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
    run_chunk "${DEVICE}" "${variant}" "${OUT_DIR}/logs/${variant}_${STAMP}.log" "${eval_subject_array[@]}"
  fi

  "${PYTHON_BIN}" -u "${SUMMARIZER}" \
    --root "${run_dir}" \
    --out-csv "${run_dir}/summary.csv" \
    --combined-subject-csv "${run_dir}/subject_rows.csv" \
    2>&1 | tee "${OUT_DIR}/logs/${variant}_summary_${STAMP}.log"
}

common_shared_args=(
  --use-tcn-token-mixer
  --tcn-mixer-alpha 0.05
  --profile-qkv-layers last1
  --profile-qkv-scale 0.01
  --profile-qkv-residual
)

if contains_variant F0_static_shared; then
  run_variant F0_static_shared \
    tcn_profile_film_qkv_last1_0p01 \
    "F0: static TCN + shared Profile-FiLM/QKV last1 scale 0.01" \
    --use-profile-film --use-profile-qkv --shared-profile-qkv --profile-conditioning film_qkv \
    "${common_shared_args[@]}"
fi

if contains_variant F1_clsa_no_film; then
  run_variant F1_clsa_no_film \
    tcn_clsa_qkv_last1_no_film \
    "F1: TCN + profile-initialized CLSA-QKV last1, no FiLM" \
    --use-profile-qkv --profile-conditioning qkv --profile-qkv-mode clsa \
    --profile-clsa-rank 8 --profile-clsa-scale 0.01 --profile-clsa-eta-max 0.1 \
    --profile-clsa-enable-fast-update 1 --profile-clsa-gate-init-bias -2.0 \
    "${common_shared_args[@]}"
fi

if contains_variant F2_film_clsa; then
  run_variant F2_film_clsa \
    tcn_profile_film_clsa_qkv_last1 \
    "F2: TCN + Profile-FiLM + embedded CLSA-QKV last1" \
    --use-profile-film --use-profile-qkv --shared-profile-qkv --profile-conditioning film_qkv --profile-qkv-mode clsa \
    --profile-clsa-rank 8 --profile-clsa-scale 0.01 --profile-clsa-eta-max 0.1 \
    --profile-clsa-enable-fast-update 1 --profile-clsa-gate-init-bias -2.0 \
    "${common_shared_args[@]}"
fi

if contains_variant F3_film_clsa_no_fast; then
  run_variant F3_film_clsa_no_fast \
    tcn_profile_film_clsa_qkv_last1_no_fast_update \
    "F3: TCN + Profile-FiLM + CLSA-QKV adapter with fast update disabled" \
    --use-profile-film --use-profile-qkv --shared-profile-qkv --profile-conditioning film_qkv --profile-qkv-mode clsa \
    --profile-clsa-rank 8 --profile-clsa-scale 0.01 --profile-clsa-eta-max 0.1 \
    --profile-clsa-enable-fast-update 0 --profile-clsa-gate-init-bias -2.0 \
    "${common_shared_args[@]}"
fi

if contains_variant F4_film_clsa_rank4; then
  run_variant F4_film_clsa_rank4 \
    tcn_profile_film_clsa_qkv_last1_rank4 \
    "F4: TCN + Profile-FiLM + embedded CLSA-QKV last1 rank 4" \
    --use-profile-film --use-profile-qkv --shared-profile-qkv --profile-conditioning film_qkv --profile-qkv-mode clsa \
    --profile-clsa-rank 4 --profile-clsa-scale 0.01 --profile-clsa-eta-max 0.1 \
    --profile-clsa-enable-fast-update 1 --profile-clsa-gate-init-bias -2.0 \
    "${common_shared_args[@]}"
fi

echo "[DONE] Profile CLSA-QKV F-series root: ${OUT_DIR}"
echo "[DONE] Manifest: ${MANIFEST}"
