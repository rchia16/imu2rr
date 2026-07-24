#!/usr/bin/env bash
set -uo pipefail

export PYTHONFAULTHANDLER=1

PYTHON_BIN="${PYTHON_BIN:-python}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="${ROOT:-/projects/BLVMob/imu-rr-seated/results/a1_mwl_probe/rest_low_high/screening_${TIMESTAMP}}"
DIRECT_ROOT="${DIRECT_ROOT:-${ROOT}/direct_imu}"
A1_ROOT="${A1_ROOT:-${ROOT}/a1_probe}"
A1_CACHE_ROOT="${A1_CACHE_ROOT:-${A1_ROOT}/cache}"
ONE_MIN_ROOT="${ONE_MIN_ROOT:-${ROOT}/one_minute}"
COMBINED_ROOT="${COMBINED_ROOT:-${ROOT}/combined_benchmark}"

SUBJECTS="${SUBJECTS:-S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29}"
SEEDS="${SEEDS:-0 1 2}"
STAGES="${STAGES:-cache direct a1 one_min aggregate}"
CUDA_DEVICES="${CUDA_DEVICES:-0 1}"
DIRECT_MODELS="${DIRECT_MODELS:-resnet1d cnn_gru tcn inceptiontime stft_cnn}"
A1_VARIANTS="${A1_VARIANTS:-a1_rr_only_logreg a1_mean_logreg a1_distribution_logreg a1_stats_logreg a1_latent_logreg a1_distribution_mlp a1_latent_mlp a1_distribution_activity_mlp}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/projects/BLVMob/imu-rr-seated/results/profile_clsa_qkv_fseries/runs/F0_static_shared}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-best_model.pt}"
DATA_STR="${DATA_STR:-imu_filt}"
PRESSURE_STR="${PRESSURE_STR:-pss_filt}"
DATA_DIR="${DATA_DIR:-/projects/BLVMob/imu-rr-seated/Data}"
DATA_GROUP="${DATA_GROUP:-mr_levels}"
MDL_DIR="${MDL_DIR:-/projects/BLVMob/imu-rr-seated/models}"
PHASE_ABC_ROOT="${PHASE_ABC_ROOT:-/projects/BLVMob/imu-rr-seated/results/rr_readout_phase_abc}"
FROZEN_A1_METHOD="${FROZEN_A1_METHOD:-A1_gaussian_kl}"
REPRESENTATION_FAMILY="${REPRESENTATION_FAMILY:-frozen_phase_abc_transfer}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-128}"
VAL_SUBJECTS="${VAL_SUBJECTS:-3}"
PATIENCE="${PATIENCE:-12}"
NUM_WORKERS="${NUM_WORKERS:-0}"
CACHED_FEATURE_BATCH_SIZE="${CACHED_FEATURE_BATCH_SIZE:-128}"
FEATURE_EVAL_BATCH_SIZE="${FEATURE_EVAL_BATCH_SIZE:-256}"
PROFILE_STATS_MAX_BATCHES="${PROFILE_STATS_MAX_BATCHES:-50}"
TARGET_CALIBRATION_WINDOWS="${TARGET_CALIBRATION_WINDOWS:-32}"
TARGET_CALIBRATION_MODE="${TARGET_CALIBRATION_MODE:-first}"
MAX_CACHE_BATCHES="${MAX_CACHE_BATCHES:-0}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
PERMUTATION_RESAMPLES="${PERMUTATION_RESAMPLES:-10000}"
BOOTSTRAP_RESAMPLES="${BOOTSTRAP_RESAMPLES:-10000}"
MWL_POSTPROCESS_SECONDS="${MWL_POSTPROCESS_SECONDS:-60}"
MWL_WINDOW_SHIFT_SECONDS="${MWL_WINDOW_SHIFT_SECONDS:-20}"
MWL_DROP_INCOMPLETE_CHUNKS="${MWL_DROP_INCOMPLETE_CHUNKS:-0}"

mkdir -p "${ROOT}/logs" "${DIRECT_ROOT}" "${A1_ROOT}" "${A1_CACHE_ROOT}" "${ONE_MIN_ROOT}"

status_file="${ROOT}/job_status.tsv"
failed_file="${ROOT}/failed_jobs.tsv"
manifest="${ROOT}/screening_manifest.tsv"

{
  printf 'key\tvalue\n'
  printf 'commit\t'
  git rev-parse HEAD 2>/dev/null || printf 'unknown\n'
  printf 'hostname\t%s\n' "$(hostname)"
  printf 'python\t%s\n' "$(${PYTHON_BIN} -c 'import sys; print(sys.version.replace("\n", " "))')"
  printf 'torch\t'
  ${PYTHON_BIN} -c 'import torch; print(torch.__version__)' 2>/dev/null || printf 'unavailable\n'
  printf 'cuda\t'
  ${PYTHON_BIN} -c 'import torch; print(torch.version.cuda)' 2>/dev/null || printf 'unavailable\n'
  printf 'stages\t%s\n' "${STAGES}"
  printf 'seeds\t%s\n' "${SEEDS}"
  printf 'subjects\t%s\n' "${SUBJECTS}"
  printf 'cuda_devices\t%s\n' "${CUDA_DEVICES}"
  printf 'checkpoint_root\t%s\n' "${CHECKPOINT_ROOT}"
  printf 'phase_abc_root\t%s\n' "${PHASE_ABC_ROOT}"
  printf 'frozen_a1_method\t%s\n' "${FROZEN_A1_METHOD}"
  printf 'representation_family\t%s\n' "${REPRESENTATION_FAMILY}"
  printf 'a1_cache_root\t%s\n' "${A1_CACHE_ROOT}"
  printf 'one_minute_root\t%s\n' "${ONE_MIN_ROOT}"
  printf 'mwl_postprocess_seconds\t%s\n' "${MWL_POSTPROCESS_SECONDS}"
  printf 'mwl_window_shift_seconds\t%s\n' "${MWL_WINDOW_SHIFT_SECONDS}"
} > "${manifest}"

printf 'timestamp\tstage\tseed\tgpu\texit_code\tcommand\tstdout_log\tstderr_log\n' > "${status_file}"
printf 'timestamp\tstage\tseed\tgpu\texit_code\tcommand\tstdout_log\tstderr_log\n' > "${failed_file}"

read -r -a gpu_array <<< "${CUDA_DEVICES}"
if [[ ${#gpu_array[@]} -eq 0 ]]; then
  gpu_array=(0)
fi

contains_stage() {
  local wanted="$1"
  for stage in ${STAGES}; do
    if [[ "${stage}" == "${wanted}" ]]; then
      return 0
    fi
  done
  return 1
}

run_seed_job() {
  local stage="$1"
  local seed="$2"
  local gpu="$3"
  local stdout_log="${ROOT}/logs/${stage}_seed_$(printf '%03d' "${seed}").out.log"
  local stderr_log="${ROOT}/logs/${stage}_seed_$(printf '%03d' "${seed}").err.log"
  local cmd=()

  if [[ "${stage}" == "direct" ]]; then
    cmd=(
      "${PYTHON_BIN}" -u run_mwl_direct_imu_benchmarks.py
      --models "${DIRECT_MODELS}"
      --subjects "${SUBJECTS}"
      --data-str "${DATA_STR}"
      --data-dir "${DATA_DIR}"
      --data-group "${DATA_GROUP}"
      --out-dir "${DIRECT_ROOT}/seed_$(printf '%03d' "${seed}")"
      --epochs "${EPOCHS}"
      --batch-size "${BATCH_SIZE}"
      --val-subjects "${VAL_SUBJECTS}"
      --patience "${PATIENCE}"
      --seed "${seed}"
      --device cuda:0
      --num-workers "${NUM_WORKERS}"
    )
    if [[ "${MAX_TRAIN_BATCHES}" != "0" ]]; then
      cmd+=(--max-train-batches "${MAX_TRAIN_BATCHES}")
    fi
  elif [[ "${stage}" == "cache" ]]; then
    cmd=(
      "${PYTHON_BIN}" -u run_a1_mwl_probe.py
      --stage cache
      --feature-cache-root "${A1_CACHE_ROOT}"
      --out-dir "${A1_ROOT}"
      --subjects "${SUBJECTS}"
      --seeds "${seed}"
      --checkpoint-root "${CHECKPOINT_ROOT}"
      --checkpoint-name "${CHECKPOINT_NAME}"
      --data-dir "${DATA_DIR}"
      --mdl-dir "${MDL_DIR}"
      --data-str "${DATA_STR}"
      --pressure-str "${PRESSURE_STR}"
      --data-group "${DATA_GROUP}"
      --representation-family "${REPRESENTATION_FAMILY}"
      --val-subjects "${VAL_SUBJECTS}"
      --cached-feature-batch-size "${CACHED_FEATURE_BATCH_SIZE}"
      --feature-eval-batch-size "${FEATURE_EVAL_BATCH_SIZE}"
      --profile-stats-max-batches "${PROFILE_STATS_MAX_BATCHES}"
      --target-calibration-windows "${TARGET_CALIBRATION_WINDOWS}"
      --target-calibration-mode "${TARGET_CALIBRATION_MODE}"
      --exclude-calibration-from-eval
      --max-cache-batches "${MAX_CACHE_BATCHES}"
      --phase-abc-root "${PHASE_ABC_ROOT}"
      --frozen-a1-method "${FROZEN_A1_METHOD}"
      --require-frozen-a1
      --device cuda:0
      --num-workers "${NUM_WORKERS}"
      --resume
    )
  elif [[ "${stage}" == "a1" ]]; then
    cmd=(
      "${PYTHON_BIN}" -u run_a1_mwl_probe.py
      --stage probe
      --feature-cache-root "${A1_CACHE_ROOT}"
      --out-dir "${A1_ROOT}/job_seed_$(printf '%03d' "${seed}")"
      --subjects "${SUBJECTS}"
      --seeds "${seed}"
      --variants "${A1_VARIANTS}"
      --epochs "${EPOCHS}"
      --patience "${PATIENCE}"
      --batch-size "${BATCH_SIZE}"
      --device cuda:0
      --max-train-batches "${MAX_TRAIN_BATCHES}"
    )
  else
    echo "unsupported stage=${stage}" >&2
    return 1
  fi

  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${cmd[@]}"
  ) > "${stdout_log}" 2> "${stderr_log}"
  local ec=$?
  local now
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local cmd_text
  cmd_text="$(printf '%q ' "${cmd[@]}")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${now}" "${stage}" "${seed}" "${gpu}" "${ec}" "${cmd_text}" "${stdout_log}" "${stderr_log}" >> "${status_file}"
  if [[ "${ec}" -ne 0 ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${now}" "${stage}" "${seed}" "${gpu}" "${ec}" "${cmd_text}" "${stdout_log}" "${stderr_log}" >> "${failed_file}"
  fi
  return "${ec}"
}

run_parallel_stage() {
  local stage="$1"
  local pids=()
  local idx=0
  local failures=0
  for seed in ${SEEDS}; do
    local gpu="${gpu_array[$((idx % ${#gpu_array[@]}))]}"
    run_seed_job "${stage}" "${seed}" "${gpu}" &
    pids+=($!)
    idx=$((idx + 1))
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failures=$((failures + 1))
    fi
  done
  return "${failures}"
}

if contains_stage "cache"; then
  run_parallel_stage "cache" || {
    echo "Cache stage failed. See ${failed_file} and ${ROOT}/logs." >&2
    exit 1
  }
fi

if contains_stage "direct"; then
  run_parallel_stage "direct" || {
    echo "Direct-IMU stage failed. See ${failed_file} and ${ROOT}/logs." >&2
    exit 1
  }
fi

if contains_stage "a1"; then
  run_parallel_stage "a1" || {
    echo "A1 probe stage failed. See ${failed_file} and ${ROOT}/logs." >&2
    exit 1
  }
fi

if contains_stage "one_min"; then
  one_min_args=()
  if [[ "${MWL_DROP_INCOMPLETE_CHUNKS}" == "1" ]]; then
    one_min_args+=(--drop-incomplete-chunks)
  else
    one_min_args+=(--no-drop-incomplete-chunks)
  fi
  "${PYTHON_BIN}" -u postprocess_mwl_one_minute.py \
    --prediction-roots "${DIRECT_ROOT}" "${A1_ROOT}" \
    --out-dir "${ONE_MIN_ROOT}" \
    --subjects "${SUBJECTS}" \
    --seeds "${SEEDS}" \
    --seconds "${MWL_POSTPROCESS_SECONDS}" \
    --window-shift-seconds "${MWL_WINDOW_SHIFT_SECONDS}" \
    "${one_min_args[@]}"
fi

if contains_stage "aggregate"; then
  "${PYTHON_BIN}" -u aggregate_mwl_rest_low_high_benchmark.py \
    --prediction-roots "${DIRECT_ROOT}" "${A1_ROOT}" "${ONE_MIN_ROOT}" \
    --out-dir "${COMBINED_ROOT}" \
    --subjects "${SUBJECTS}" \
    --seeds "${SEEDS}" \
    --permutation-resamples "${PERMUTATION_RESAMPLES}" \
    --bootstrap-resamples "${BOOTSTRAP_RESAMPLES}"
fi

echo "[DONE] combined benchmark: ${COMBINED_ROOT}"
