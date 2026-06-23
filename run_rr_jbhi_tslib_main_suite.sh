#!/usr/bin/env bash
set -euo pipefail

# JBHI no-shortcuts source baseline runner with direct TSLib PatchTST/TimesNet files.
# This runner does not run the main cross-modal model; use the adaptation/full
# ladder scripts for the main method.

RUN_ID="${RUN_ID:-jbhi_tslib_$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="${ROOT:-/projects/BLVMob/imu-rr-seated/results/${RUN_ID}}"
DATA_DIR="${DATA_DIR:-/projects/BLVMob/imu-rr-seated/Data}"
DATA_GROUP="${DATA_GROUP:-mr}"
DATA_STR="${DATA_STR:-imu_filt}"
DEVICE="${DEVICE:-cuda:0}"
MODELS="${MODELS:-resnet1d cnn_gru tcn inceptiontime patchtst_tslib timesnet_tslib}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-128}"
PATCHTST_FILE="${PATCHTST_FILE:-PatchTST.py}"
TIMESNET_FILE="${TIMESNET_FILE:-TimesNet.py}"

export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PATCHTST_FILE
export TIMESNET_FILE

mkdir -p "${ROOT}/logs"

python -u run_rr_jbhi_tslib_neural_baselines.py \
  --models "${MODELS}" \
  --data-str "${DATA_STR}" \
  --data-dir "${DATA_DIR}" \
  --data-group "${DATA_GROUP}" \
  --out-dir "${ROOT}/source_neural_baselines" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --patchtst-file "${PATCHTST_FILE}" \
  --timesnet-file "${TIMESNET_FILE}" \
  2>&1 | tee "${ROOT}/logs/source_neural_baselines_tslib.log"

echo "[DONE] ${ROOT}"
