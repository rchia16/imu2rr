#!/usr/bin/env bash
set -euo pipefail

# Explorative staged evaluation for the depthwise token mixer, profile FiLM,
# profile/readout TTT capacity, and QKV profile conditioning.
#
# Stages:
#   A: TCN mixer ablation
#   B: safer FiLM placement ablation
#   C: TTT capacity ladder
#   D: QKV profile conditioning, intended only if FiLM plateaus
#   E: TCN + shared-profile FiLM-QKV last1 scale 0.01 modes
#
# The Python API supports the staged variants below. RUN_UNWIRED_VARIANTS
# remains as a guard for future local rows that intentionally point at flags
# not yet wired into the model or adaptation ladder.

RUN_ID="${RUN_ID:-tcn_profile_ttt_explore_$(date -u +%Y%m%dT%H%M%SZ)}"
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

SUBJECTS="${SUBJECTS:-S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29}"
EVAL_SUBJECTS="${EVAL_SUBJECTS:-${SUBJECTS}}"
STAGES="${STAGES:-A B C D E}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
RR_PROBE_EPOCHS="${RR_PROBE_EPOCHS:-100}"
RR_HEAD_TYPE="${RR_HEAD_TYPE:-mlp}"
DECODER_MODE="${DECODER_MODE:-cross_attn}"
TARGET_CALIBRATION_WINDOWS="${TARGET_CALIBRATION_WINDOWS:-32}"
TARGET_CALIBRATION_MODE="${TARGET_CALIBRATION_MODE:-random}"
PROFILE_FILM_SCALE="${PROFILE_FILM_SCALE:-0.1}"

# Leave at 0 unless adding future local rows before wiring their flags.
RUN_UNWIRED_VARIANTS="${RUN_UNWIRED_VARIANTS:-0}"

export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/runs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MANIFEST="${OUT_DIR}/manifest.tsv"
if [[ ! -f "${MANIFEST}" ]]; then
  printf "stage\tvariant\tstatus\tdescription\trun_dir\trr_tta_modes\textra_args\n" > "${MANIFEST}"
fi

has_stage() {
  local wanted="$1"
  local stage
  for stage in ${STAGES}; do
    if [[ "${stage}" == "${wanted}" ]]; then
      return 0
    fi
  done
  return 1
}

append_manifest() {
  local stage="$1"
  local variant="$2"
  local status="$3"
  local description="$4"
  local run_dir="$5"
  local tta_modes="$6"
  local extra_args="$7"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${stage}" "${variant}" "${status}" "${description}" "${run_dir}" "${tta_modes}" "${extra_args}" \
    >> "${MANIFEST}"
}

run_variant() {
  local stage="$1"
  local variant="$2"
  local description="$3"
  local supported="$4"
  local imu_token_mixer="$5"
  local profile_kind="$6"
  local tta_modes="$7"
  shift 7
  local extra_args=("$@")

  local run_dir="${OUT_DIR}/runs/${variant}"
  local log_file="${OUT_DIR}/logs/${variant}_${STAMP}.log"
  local summary_log="${OUT_DIR}/logs/${variant}_summary_${STAMP}.log"
  local profile_args=()

  if [[ "${supported}" != "1" && "${RUN_UNWIRED_VARIANTS}" != "1" ]]; then
    echo "[SKIP] ${variant}: ${description}"
    append_manifest "${stage}" "${variant}" "SKIP_UNWIRED" "${description}" "${run_dir}" "${tta_modes}" "${extra_args[*]}"
    return 0
  fi

  case "${profile_kind}" in
    none)
      profile_args+=(--profile-conditioning none)
      ;;
    film)
      profile_args+=(--use-profile-film --profile-conditioning film --profile-film-scale "${PROFILE_FILM_SCALE}")
      ;;
    qkv)
      profile_args+=(--use-profile-qkv --profile-conditioning qkv)
      ;;
    shared)
      profile_args+=(
        --use-profile-film
        --use-profile-qkv
        --shared-profile-qkv
        --profile-conditioning film_qkv
        --profile-qkv-layers last1
        --profile-qkv-scale 0.01
        --profile-qkv-residual
        --use-tcn-token-mixer
        --tcn-mixer-alpha 0.05
      )
      ;;
    *)
      echo "[ERROR] Unknown profile_kind=${profile_kind}" >&2
      exit 2
      ;;
  esac

  mkdir -p "${run_dir}"
  append_manifest "${stage}" "${variant}" "RUN" "${description}" "${run_dir}" "${tta_modes}" "${extra_args[*]}"
  echo "[RUN] ${variant}: ${description}"

  run_variant_chunk() {
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
    --imu-token-mixer "${imu_token_mixer}" \
    --rr-tta-modes "${tta_modes}" \
    --use-unsup-mode-defaults \
    --target-calibration-windows "${TARGET_CALIBRATION_WINDOWS}" \
    --target-calibration-mode "${TARGET_CALIBRATION_MODE}" \
    --profile-unsup-adapt-scope calibration \
    --include-calibration-in-eval \
    --adaptation-use-calibration-only \
    "${profile_args[@]}" \
    "${extra_args[@]}" \
    2>&1 | tee "${chunk_log_file}"
  }

  local devices=(${DEVICES})
  local eval_subject_array=(${EVAL_SUBJECTS})
  if (( ${#devices[@]} > 1 && ${#eval_subject_array[@]} > 1 )); then
    local pids=()
    local slot
    for slot in "${!devices[@]}"; do
      local chunk_subjects=()
      local idx
      for idx in "${!eval_subject_array[@]}"; do
        if (( idx % ${#devices[@]} == slot )); then
          chunk_subjects+=("${eval_subject_array[$idx]}")
        fi
      done
      if (( ${#chunk_subjects[@]} == 0 )); then
        continue
      fi
      local chunk_run_id="${variant}_gpu${slot}"
      local chunk_log_file="${OUT_DIR}/logs/${chunk_run_id}_${STAMP}.log"
      echo "[RUN] ${variant}: ${devices[$slot]} -> ${chunk_subjects[*]}"
      run_variant_chunk "${devices[$slot]}" "${chunk_run_id}" "${chunk_log_file}" "${chunk_subjects[@]}" &
      pids+=("$!")
    done
    local status=0
    local pid
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        status=1
      fi
    done
    if (( status != 0 )); then
      return "${status}"
    fi
  else
    run_variant_chunk "${DEVICE}" "${variant}" "${log_file}" "${eval_subject_array[@]}"
  fi

  "${PYTHON_BIN}" -u "${SUMMARIZER}" \
    --root "${run_dir}" \
    --out-csv "${run_dir}/summary.csv" \
    --combined-subject-csv "${run_dir}/subject_rows.csv" \
    2>&1 | tee "${summary_log}"
}

# if has_stage A; then
#   run_variant A A0_no_tcn_no_profile \
#     "A0: no TCN, no profile" \
#     1 none none "none"
#   run_variant A A1_tcn_no_profile \
#     "A1: TCN, no profile" \
#     1 dwconv none "none"
#   run_variant A A2_tcn_profile_film_init_only \
#     "A2: TCN + profile_film_init_only" \
#     1 dwconv film "profile_film_init_only"
#   run_variant A A3_tcn_profile_film_unsup_sparc \
#     "A3: TCN + profile_film_unsup_sparc" \
#     1 dwconv film "profile_film_unsup_sparc"
# fi

if has_stage B; then
  # run_variant B B0_current_token_and_pooled_film \
  #   "B0: token FiLM + pooled FiLM current" \
  #   1 dwconv film "profile_film_init_only profile_film_unsup_sparc"
  run_variant B B1_pooled_film_only \
    "B1: pooled FiLM only" \
    1 dwconv film "profile_film_init_only profile_film_unsup_sparc" \
    --profile-film-placement pooled_only
  run_variant B B2_late_token_film_only \
    "B2: late-token FiLM only" \
    1 dwconv film "profile_film_init_only profile_film_unsup_sparc" \
    --profile-film-placement late_token_only
  run_variant B B3_residual_film_alpha_0p1 \
    "B3: residual FiLM with alpha=0.1" \
    1 dwconv film "profile_film_init_only profile_film_unsup_sparc" \
    --profile-film-placement residual --profile-film-residual-alpha 0.1
  run_variant B B4_residual_film_alpha_0p3 \
    "B4: residual FiLM with alpha=0.3" \
    1 dwconv film "profile_film_init_only profile_film_unsup_sparc" \
    --profile-film-placement residual --profile-film-residual-alpha 0.3
fi

if has_stage C; then
  # run_variant C C0_no_ttt \
  #   "C0: no TTT" \
  #   1 dwconv film "profile_film_init_only"
  # run_variant C C1_adapt_pt_only \
  #   "C1: adapt p_t only" \
  #   1 dwconv film "profile_film_unsup_stft_consistency"
  # run_variant C C2_adapt_pt_readout_affine \
  #   "C2: adapt p_t + readout affine" \
  #   1 dwconv film "profile_film_unsup_readout_affine"
  # run_variant C C3_adapt_pt_readout_affine_conf_stft \
  #   "C3: adapt p_t + readout affine + confidence-weighted STFT" \
  #   1 dwconv film "profile_film_unsup_sparc" \
  #   --unsup-stft-confidence-floor 0.1 --unsup-stft-confidence-power 1.0
  run_variant C C4_episodic_batch_pt_only \
    "C4: episodic batch TTT p_t only" \
    1 dwconv film "profile_film_unsup_stft_consistency" \
    --profile-unsup-episodic-batch
  run_variant C C5_episodic_batch_pt_readout_affine \
    "C5: episodic batch TTT p_t + readout affine" \
    1 dwconv film "profile_film_unsup_readout_affine" \
    --profile-unsup-episodic-batch --unsup-readout-affine
fi

if has_stage D; then
  # run_variant D D0_qkv_last1_scale_0p01 \
  #   "D0: QKV last1 scale 0.01" \
  #   1 dwconv qkv "profile_qkv_unsup_stft_consistency" \
  #   --profile-qkv-layers last1 --profile-qkv-scale 0.01
  # run_variant D D1_qkv_last1_scale_0p03 \
  #   "D1: QKV last1 scale 0.03" \
  #   1 dwconv qkv "profile_qkv_unsup_stft_consistency" \
  #   --profile-qkv-layers last1 --profile-qkv-scale 0.03
  run_variant D D2_residual_qkv_last1_scale_0p03 \
    "D2: residual QKV last1 scale 0.03" \
    1 dwconv qkv "profile_qkv_unsup_stft_consistency" \
    --profile-qkv-layers last1 --profile-qkv-scale 0.03 --profile-qkv-residual
  # run_variant D D3_qkv_attention_prior \
  #   "D3: residual QKV + attention prior" \
  #   1 dwconv qkv "profile_qkv_unsup_stft_smooth_prior_attn" \
  #   --profile-qkv-layers last1 --profile-qkv-scale 0.03 --profile-qkv-residual --unsup-attention-profile-weight 0.005
fi

if has_stage E; then
  run_variant E E0_tcn_shared_film_qkv_init \
    "new: TCN + shared-profile FiLM-QKV last1 scale 0.01" \
    1 dwconv shared "tcn_profile_film_qkv_last1_0p01"
  run_variant E E1_tcn_shared_film_qkv_sparc_pt \
    "new: TCN + shared-profile FiLM-QKV + p_t-only SPARC" \
    1 dwconv shared "tcn_profile_film_qkv_last1_0p01_sparc_pt"
  run_variant E E2_tcn_shared_film_qkv_sparc_pt_budget \
    "new: TCN + shared-profile FiLM-QKV + p_t-only SPARC + safety budget" \
    1 dwconv shared "tcn_profile_film_qkv_last1_0p01_sparc_pt_budget"
fi

echo "[DONE] Staged exploration root: ${OUT_DIR}"
echo "[DONE] Manifest: ${MANIFEST}"
