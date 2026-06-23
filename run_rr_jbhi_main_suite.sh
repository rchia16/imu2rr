#!/usr/bin/env bash
set -euo pipefail

# JBHI-ready RR baseline + main adaptation experiment runner.
# 1) Train/evaluate representative neural baselines.
# 2) Run embedding/t-SNE diagnostics.
# 3) Optionally run the exact prototype-gated alpha adaptation experiment if the
#    prototype gate scripts are in the repo root.
# 4) Build combined result tables.

RUN_ID="${RUN_ID:-jbhi_rr_$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="${ROOT:-/projects/BLVMob/imu-rr-seated/results/${RUN_ID}}"
DATA_DIR="${DATA_DIR:-/projects/BLVMob/imu-rr-seated/Data}"
DATA_GROUP="${DATA_GROUP:-mr}"
DATA_STR="${DATA_STR:-imu_filt}"
DEVICE="${DEVICE:-cuda:0}"
MODELS="${MODELS:-resnet1d cnn_gru tcn inceptiontime stft_cnn patchtst crossmodal_rr}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-128}"

export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

mkdir -p "${ROOT}/logs"

python -u run_rr_jbhi_baselines.py \
  --models "${MODELS}" \
  --data-str "${DATA_STR}" \
  --data-dir "${DATA_DIR}" \
  --data-group "${DATA_GROUP}" \
  --out-dir "${ROOT}/baselines" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  2>&1 | tee "${ROOT}/logs/baselines.log"

python -u analyze_rr_jbhi_embeddings.py \
  --root "${ROOT}/baselines" \
  --out-dir "${ROOT}/embedding_diagnostics" \
  2>&1 | tee "${ROOT}/logs/embedding_diagnostics.log"

if [[ "${RUN_PROTOTYPE_GATE:-1}" != "0" && -f run_adaptation_prototype_gate_tests.sh ]]; then
  OUT_DIR="${ROOT}/prototype_gate" \
  bash run_adaptation_prototype_gate_tests.sh \
  2>&1 | tee "${ROOT}/logs/prototype_gate.log"
else
  echo "[NOTE] Skipping prototype gate run. Set RUN_PROTOTYPE_GATE=1 and copy prototype gate scripts to repo root to enable."
fi

python -u make_rr_jbhi_tables.py \
  --baseline-root "${ROOT}/baselines" \
  --adaptation-root "${ROOT}" \
  --out-dir "${ROOT}/tables" \
  2>&1 | tee "${ROOT}/logs/tables.log"

echo "[DONE] ${ROOT}"
