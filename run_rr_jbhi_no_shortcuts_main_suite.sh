#!/usr/bin/env bash
set -euo pipefail

# No-shortcuts JBHI suite.
# - Source-faithful neural baselines are run separately.
# - The main cross-modal method is NOT reimplemented here; it is run through the
#   full project adaptation ladder and full alpha/prototype policy analyzers.

RUN_ID="${RUN_ID:-jbhi_no_shortcuts_$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="${ROOT:-/projects/BLVMob/imu-rr-seated/results/${RUN_ID}}"
DATA_DIR="${DATA_DIR:-/projects/BLVMob/imu-rr-seated/Data}"
DATA_GROUP="${DATA_GROUP:-mr}"
DATA_STR="${DATA_STR:-imu_filt}"
DEVICE="${DEVICE:-cuda:0}"
BASELINE_MODELS="${BASELINE_MODELS:-resnet1d cnn_gru tcn inceptiontime}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-128}"

export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

mkdir -p "${ROOT}/logs"

echo "[1/4] Running source-faithful neural baselines"
python -u run_rr_jbhi_source_neural_baselines.py \
  --models "${BASELINE_MODELS}" \
  --data-str "${DATA_STR}" \
  --data-dir "${DATA_DIR}" \
  --data-group "${DATA_GROUP}" \
  --out-dir "${ROOT}/source_neural_baselines" \
  --epochs "${BASELINE_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  2>&1 | tee "${ROOT}/logs/source_neural_baselines.log"

echo "[2/4] Running full exact cross-modal adaptation ladder + alpha_hat policy"
OUT_DIR="${ROOT}/alpha_hat_full" \
LADDER="vit_pressure_crossmodal_stft_rr_adaptation_alpha_hat_sweep.py" \
SUMMARIZER="summarize_rr_adaptation_alpha_hat_sweep.py" \
POLICY="analyze_adaptation_alpha_hat_policy_full.py" \
bash run_adaptation_alpha_hat_tests_full.sh \
  2>&1 | tee "${ROOT}/logs/alpha_hat_full.log"

echo "[3/4] Running full exact cross-modal adaptation ladder + prototype/OOD gate"
OUT_DIR="${ROOT}/prototype_gate_full" \
LADDER="vit_pressure_crossmodal_stft_rr_adaptation_prototype_gate_sweep_full.py" \
SUMMARIZER="summarize_rr_adaptation_prototype_gate_sweep_full.py" \
POLICY="analyze_adaptation_prototype_gate_policy_full.py" \
bash run_adaptation_prototype_gate_tests_full.sh \
  2>&1 | tee "${ROOT}/logs/prototype_gate_full.log"

echo "[4/4] Building combined tables where available"
python -u make_rr_jbhi_tables.py \
  --baseline-root "${ROOT}/source_neural_baselines" \
  --adaptation-root "${ROOT}" \
  --out-dir "${ROOT}/tables" \
  2>&1 | tee "${ROOT}/logs/tables.log" || true

echo "[DONE] ${ROOT}"
