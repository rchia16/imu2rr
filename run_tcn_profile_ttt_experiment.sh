#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-/projects/BLVMob/imu-rr-seated/results/tcn_profile_ttt}"
DATA_DIR="${DATA_DIR:-/projects/BLVMob/imu-rr-seated/Data}"
DATA_GROUP="${DATA_GROUP:-mr}"
DATA_STR="${DATA_STR:-imu_filt}"
CUDA_DEVICES="${CUDA_DEVICES:-0 1}"
MODES="${MODES:-t0_plain t1_profile_affine t2_profile_film t3_profile_film_affine t4_ttt_affine t5_ttt_film_affine t6_meta_gated_ttt t7_profile_mean_alignment}"
SUBJECTS="${SUBJECTS:-S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29}"
SEEDS="${SEEDS:-0 1 2}"
RESUME="${RESUME:-0}"
STAMP="${STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${OUT_ROOT}/full_${STAMP}}"

if [[ -e "${RUN_ROOT}" && "${RESUME}" != "1" ]]; then
  echo "Refusing to overwrite existing run root: ${RUN_ROOT}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/logs"
git rev-parse HEAD > "${RUN_ROOT}/git_commit.txt" 2>/dev/null || true
env | sort > "${RUN_ROOT}/environment.txt"

IFS=' ' read -r -a GPU_LIST <<< "${CUDA_DEVICES}"
job_index=0
pids=()

for seed in ${SEEDS}; do
  gpu="${GPU_LIST[$((job_index % ${#GPU_LIST[@]}))]}"
  log="${RUN_ROOT}/logs/seed_${seed}.log"
  cmd=(
    "${PYTHON_BIN}" tcn_profile_ttt_experiment.py
    --mode ${MODES}
    --subjects ${SUBJECTS}
    --seed "${seed}"
    --device "cuda:${gpu}"
    --data-dir "${DATA_DIR}"
    --data-group "${DATA_GROUP}"
    --data-str "${DATA_STR}"
    --out-dir "${RUN_ROOT}"
  )
  if [[ "${RESUME}" == "1" ]]; then
    cmd+=(--resume)
  fi
  cmd+=("$@")
  echo "[RUN] seed=${seed} gpu=${gpu} log=${log}"
  "${cmd[@]}" > "${log}" 2>&1 &
  pids+=("$!")
  job_index=$((job_index + 1))
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" != "0" ]]; then
  echo "At least one experiment job failed; analysis skipped." >&2
  exit "${status}"
fi

for run_dir in "${RUN_ROOT}"/phase*_*_seed*; do
  [[ -d "${run_dir}" ]] || continue
  "${PYTHON_BIN}" tcn_profile_ttt_analysis.py --run-dir "${run_dir}" > "${RUN_ROOT}/logs/analysis_$(basename "${run_dir}").log" 2>&1
done

echo "[DONE] run root: ${RUN_ROOT}"
