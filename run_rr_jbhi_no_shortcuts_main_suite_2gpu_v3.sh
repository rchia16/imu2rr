#!/usr/bin/env bash
set -euo pipefail

# Full no-shortcuts JBHI suite, two-GPU version.
#
# Runs:
#   1. TSLib/source neural baselines split across two GPUs.
#   2. Full alpha-hat adaptation split by held-out subject across two GPUs.
#   3. Full prototype/OOD gate adaptation split by held-out subject across two GPUs.
#   4. Combined tables.
# ROOT=/projects/BLVMob/imu-rr-seated/results/jbhi_no_shortcuts_2gpu_v3_20260623T031658Z \
# SKIP_COMPLETED=1 \
# bash run_rr_jbhi_no_shortcuts_main_suite_2gpu_v3.sh

#
# NOTE:
#   This script intentionally calls the *_2gpu_v2 runners. It does not call the
#   deprecated run_rr_jbhi_source_neural_baselines.py path.

RUN_ID="${RUN_ID:-jbhi_no_shortcuts_2gpu_v3_$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="${ROOT:-/projects/BLVMob/imu-rr-seated/results/${RUN_ID}}"
DATA_DIR="${DATA_DIR:-/projects/BLVMob/imu-rr-seated/Data}"
DATA_GROUP="${DATA_GROUP:-mr}"
DATA_STR="${DATA_STR:-imu_filt}"
MDL_DIR="${MDL_DIR:-/projects/BLVMob/imu-rr-seated/models/imu_filt/loocv}"
GPUS="${GPUS:-0 1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"

# Baseline runner settings.
BASELINE_EPOCHS="${BASELINE_EPOCHS:-80}"
BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-128}"
PATCHTST_FILE="${PATCHTST_FILE:-PatchTST.py}"
TIMESNET_FILE="${TIMESNET_FILE:-TimesNet.py}"
# downsample only: justifiable since RR info survives downsampling
# (30Hz >> 0.3)
TIMESNET_MODE="${TIMESNET_MODE:-downsample}"
REUSE_EXISTING="${REUSE_EXISTING:-auto}"
REUSE_EXISTING_ROOTS="${REUSE_EXISTING_ROOTS:-/projects/BLVMob/imu-rr-seated/results}"
REUSE_EXISTING_MODE="${REUSE_EXISTING_MODE:-source}"
MODELS_GPU0="${MODELS_GPU0:-patchtst_tslib resnet1d tcn cnn_gru inceptiontime}"
MODELS_GPU1="${MODELS_GPU1:-timesnet_tslib}"

# Adaptation runner settings.
SUBJECTS="${SUBJECTS:-S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29}"
EVAL_SUBJECTS="${EVAL_SUBJECTS:-${SUBJECTS}}"
ADAPT_EPOCHS="${ADAPT_EPOCHS:-20}"
ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE:-16}"
RR_PROBE_EPOCHS="${RR_PROBE_EPOCHS:-100}"
RR_HEAD_TYPE="${RR_HEAD_TYPE:-mlp}"

export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

mkdir -p "${ROOT}/logs"

echo "[1/4] Running TSLib/source neural baselines on two GPUs"
RUN_ID="source_neural_baselines" \
ROOT="${ROOT}" \
DATA_DIR="${DATA_DIR}" \
DATA_GROUP="${DATA_GROUP}" \
DATA_STR="${DATA_STR}" \
GPUS="${GPUS}" \
EPOCHS="${BASELINE_EPOCHS}" \
BATCH_SIZE="${BASELINE_BATCH_SIZE}" \
PATCHTST_FILE="${PATCHTST_FILE}" \
TIMESNET_FILE="${TIMESNET_FILE}" \
TIMESNET_MODE="${TIMESNET_MODE}" \
REUSE_EXISTING="${REUSE_EXISTING}" \
REUSE_EXISTING_ROOTS="${REUSE_EXISTING_ROOTS}" \
REUSE_EXISTING_MODE="${REUSE_EXISTING_MODE}" \
MODELS_GPU0="${MODELS_GPU0}" \
MODELS_GPU1="${MODELS_GPU1}" \
SKIP_COMPLETED="${SKIP_COMPLETED}" \
# bash run_rr_jbhi_tslib_main_suite_2gpu_v3.sh \
#   2>&1 | tee "${ROOT}/logs/source_neural_baselines_2gpu.log"

echo "[2/4] Running full alpha-hat adaptation on two GPUs"
OUT_DIR="${ROOT}/alpha_hat_full" \
DATA_DIR="${DATA_DIR}" \
DATA_GROUP="${DATA_GROUP}" \
DATA_STR="${DATA_STR}" \
MDL_DIR="${MDL_DIR}" \
GPUS="${GPUS}" \
SUBJECTS="${SUBJECTS}" \
EVAL_SUBJECTS="${EVAL_SUBJECTS}" \
EPOCHS="${ADAPT_EPOCHS}" \
BATCH_SIZE="${ADAPT_BATCH_SIZE}" \
RR_PROBE_EPOCHS="${RR_PROBE_EPOCHS}" \
SKIP_COMPLETED="${SKIP_COMPLETED}" \
bash run_adaptation_alpha_hat_tests_full_2gpu_v2.sh \
  2>&1 | tee "${ROOT}/logs/alpha_hat_full_2gpu.log"

echo "[3/4] Running full prototype/OOD gate adaptation on two GPUs"
OUT_DIR="${ROOT}/prototype_gate_full" \
DATA_DIR="${DATA_DIR}" \
DATA_GROUP="${DATA_GROUP}" \
DATA_STR="${DATA_STR}" \
MDL_DIR="${MDL_DIR}" \
GPUS="${GPUS}" \
SUBJECTS="${SUBJECTS}" \
EVAL_SUBJECTS="${EVAL_SUBJECTS}" \
EPOCHS="${ADAPT_EPOCHS}" \
BATCH_SIZE="${ADAPT_BATCH_SIZE}" \
RR_PROBE_EPOCHS="${RR_PROBE_EPOCHS}" \
SKIP_COMPLETED="${SKIP_COMPLETED}" \
RR_HEAD_TYPE="${RR_HEAD_TYPE}" \
bash run_adaptation_prototype_gate_tests_full_2gpu_v2.sh \
  2>&1 | tee "${ROOT}/logs/prototype_gate_full_2gpu.log"

echo "[4/4] Building combined tables where available"
python -u make_rr_jbhi_tables.py \
  --baseline-root "${ROOT}/source_neural_baselines" \
  --adaptation-root "${ROOT}" \
  --out-dir "${ROOT}/tables" \
  2>&1 | tee "${ROOT}/logs/tables.log" || true

echo "[DONE] ${ROOT}"
