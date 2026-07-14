#!/usr/bin/env bash
set -uo pipefail

export PYTHONFAULTHANDLER=1

SEEDS="${SEEDS:-0 1 2}"
PHASES="${PHASES:-cache A B C analysis}"
METHODS="${METHODS:-}"
CORRECTIVE="${CORRECTIVE:-0}"
CUDA_DEVICES="${CUDA_DEVICES:-0 1}"
OUT_ROOT="${OUT_ROOT:-/projects/BLVMob/imu-rr-seated/results/rr_readout_phase_abc}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/projects/BLVMob/imu-rr-seated/results/profile_clsa_qkv_fseries/runs/F0_static_shared}"
DATA_DIR="${DATA_DIR:-/projects/BLVMob/imu-rr-seated/Data}"
MDL_DIR="${MDL_DIR:-/projects/BLVMob/imu-rr-seated/models}"
SUBJECTS="${SUBJECTS:-S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
RESUME_FLAG="${RESUME_FLAG:---resume}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Compact corrective experiment mode:
#   CORRECTIVE=1 bash run_rr_readout_phase_abc.sh
#
# This reuses the normal cache/output plumbing but narrows the scientific matrix:
#   - A0 decoder audit on identical cached spectra/frequency grid;
#   - Phase A trainable methods: A1 Gaussian-KL and A3 KL+RR-MAE only;
#   - Phase B relative-advantage gate: abs(spec-y) - abs(hidden-y);
#   - Phase C fixed A1 base, with no held-out-test method selection.
if [[ "${CORRECTIVE}" == "1" ]]; then
  PHASES="${PHASES:-cache A B C analysis}"
  METHODS="${METHODS:-}"
fi

mkdir -p "${OUT_ROOT}/logs"

manifest="${OUT_ROOT}/run_manifest.tsv"
status_file="${OUT_ROOT}/job_status.tsv"
failed_file="${OUT_ROOT}/failed_jobs.tsv"

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
  printf 'phases\t%s\n' "${PHASES}"
  printf 'methods\t%s\n' "${METHODS}"
  printf 'corrective\t%s\n' "${CORRECTIVE}"
  printf 'cuda_devices\t%s\n' "${CUDA_DEVICES}"
  printf 'checkpoint_root\t%s\n' "${CHECKPOINT_ROOT}"
} > "${manifest}"

printf 'timestamp\tphase\tmethod\tsubject\tseed\tgpu\texit_code\tcommand\tstdout_log\tstderr_log\n' > "${status_file}"
printf 'timestamp\tphase\tmethod\tsubject\tseed\tgpu\texit_code\tcommand\tstdout_log\tstderr_log\n' > "${failed_file}"

read -r -a gpu_array <<< "${CUDA_DEVICES}"
if [[ ${#gpu_array[@]} -eq 0 ]]; then
  gpu_array=(0)
fi

run_job() {
  local phase="$1"
  local method="$2"
  local subject="$3"
  local seed="$4"
  local gpu="$5"
  local seed_dir="${OUT_ROOT}/seed_$(printf '%03d' "${seed}")"
  local method_label="${method:-all}"
  method_label="${method_label// /_}"
  local label="phase_${phase}_${method_label}_${subject:-all}_seed_$(printf '%03d' "${seed}")"
  local stdout_log="${OUT_ROOT}/logs/${label}.out.log"
  local stderr_log="${OUT_ROOT}/logs/${label}.err.log"
  mkdir -p "${seed_dir}"
  local method_args=()
  if [[ -n "${method}" ]]; then
    method_args=(--methods "${method}")
  fi
  local subject_args=()
  if [[ -n "${subject}" ]]; then
    subject_args=(--subjects "${subject}")
  else
    subject_args=(--subjects ${SUBJECTS})
  fi
  local cmd=(
    "${PYTHON_BIN}" rr_readout_phase_abc_experiment.py
    --phase "${phase}"
    "${subject_args[@]}"
    --checkpoint-root "${CHECKPOINT_ROOT}"
    --out-dir "${seed_dir}"
    --data-dir "${DATA_DIR}"
    --mdl-dir "${MDL_DIR}"
    --seed "${seed}"
    --device cuda:0
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    "${method_args[@]}"
  )
  if [[ "${CORRECTIVE}" == "1" ]]; then
    cmd+=(--corrective --audit-a0)
  fi
  if [[ -n "${RESUME_FLAG}" ]]; then
    cmd+=(${RESUME_FLAG})
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
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${now}" "${phase}" "${method:-all}" "${subject:-all}" "${seed}" "${gpu}" "${ec}" "${cmd_text}" "${stdout_log}" "${stderr_log}" >> "${status_file}"
  if [[ "${ec}" -ne 0 ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${now}" "${phase}" "${method:-all}" "${subject:-all}" "${seed}" "${gpu}" "${ec}" "${cmd_text}" "${stdout_log}" "${stderr_log}" >> "${failed_file}"
  fi
  return "${ec}"
}

job_index=0
failures=0

for seed in ${SEEDS}; do
  for phase in ${PHASES}; do
    gpu="${gpu_array[$((job_index % ${#gpu_array[@]}))]}"
    if [[ "${phase}" == "analysis" ]]; then
      run_job "analysis" "" "" "${seed}" "${gpu}" || failures=$((failures + 1))
      job_index=$((job_index + 1))
      continue
    fi
    run_job "${phase}" "${METHODS}" "" "${seed}" "${gpu}" || failures=$((failures + 1))
    job_index=$((job_index + 1))
  done
done

if [[ "${failures}" -ne 0 ]]; then
  echo "${failures} RR readout Phase ABC job(s) failed. See ${failed_file} and ${OUT_ROOT}/logs." >&2
  exit 1
fi

echo "RR readout Phase ABC jobs completed under ${OUT_ROOT}."
