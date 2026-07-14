#!/usr/bin/env bash
set -uo pipefail

SEEDS="${SEEDS:-0 1 2}"
STAGES="${STAGES:-audit frozen_readouts rr_finetune profile_film}"
CUDA_DEVICES="${CUDA_DEVICES:-0 1}"
OUT_ROOT="${OUT_ROOT:-/projects/BLVMob/imu-rr-seated/results/rr_spectral_readout}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/projects/BLVMob/imu-rr-seated/results/profile_clsa_qkv_fseries/runs/F0_static_shared}"
DATA_DIR="${DATA_DIR:-/projects/BLVMob/imu-rr-seated/Data}"
MDL_DIR="${MDL_DIR:-/projects/BLVMob/imu-rr-seated/models}"
SUBJECTS="${SUBJECTS:-S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
RESUME_FLAG="${RESUME_FLAG:---resume}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${OUT_ROOT}/logs"

manifest="${OUT_ROOT}/run_manifest.tsv"
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
  printf 'seeds\t%s\n' "${SEEDS}"
  printf 'stages\t%s\n' "${STAGES}"
  printf 'cuda_devices\t%s\n' "${CUDA_DEVICES}"
  printf 'checkpoint_root\t%s\n' "${CHECKPOINT_ROOT}"
} > "${manifest}"

read -r -a gpu_array <<< "${CUDA_DEVICES}"
if [[ ${#gpu_array[@]} -eq 0 ]]; then
  gpu_array=(0)
fi

pids=()
statuses=()
job_index=0

for seed in ${SEEDS}; do
  for stage in ${STAGES}; do
    gpu="${gpu_array[$((job_index % ${#gpu_array[@]}))]}"
    out_dir="${OUT_ROOT}/seed_$(printf '%03d' "${seed}")"
    log="${OUT_ROOT}/logs/${stage}_seed_$(printf '%03d' "${seed}").log"
    mkdir -p "${out_dir}"
    (
      set -euo pipefail
      export CUDA_VISIBLE_DEVICES="${gpu}"
      ${PYTHON_BIN} rr_spectral_readout_experiment.py \
        --stage "${stage}" \
        --subjects ${SUBJECTS} \
        --checkpoint-root "${CHECKPOINT_ROOT}" \
        --out-dir "${out_dir}" \
        --data-dir "${DATA_DIR}" \
        --mdl-dir "${MDL_DIR}" \
        --seed "${seed}" \
        --device cuda:0 \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        ${RESUME_FLAG}
    ) > "${log}" 2>&1 &
    pids+=("$!")
    job_index=$((job_index + 1))

    if [[ ${#pids[@]} -ge ${#gpu_array[@]} ]]; then
      for pid in "${pids[@]}"; do
        if wait "${pid}"; then
          statuses+=(0)
        else
          statuses+=("$?")
        fi
      done
      pids=()
    fi
  done
done

for pid in "${pids[@]}"; do
  if wait "${pid}"; then
    statuses+=(0)
  else
    statuses+=("$?")
  fi
done

failed=0
for status in "${statuses[@]}"; do
  if [[ "${status}" -ne 0 ]]; then
    failed="${status}"
    break
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "At least one RR spectral readout job failed. See ${OUT_ROOT}/logs." >&2
  exit "${failed}"
fi

for seed in ${SEEDS}; do
  out_dir="${OUT_ROOT}/seed_$(printf '%03d' "${seed}")"
  ${PYTHON_BIN} rr_spectral_readout_analysis.py \
    --out-dir "${out_dir}" \
    --seed "${seed}" \
    > "${OUT_ROOT}/logs/analysis_seed_$(printf '%03d' "${seed}").log" 2>&1
done

echo "RR spectral readout jobs completed under ${OUT_ROOT}."
