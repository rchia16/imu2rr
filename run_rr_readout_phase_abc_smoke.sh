#!/usr/bin/env bash
set -euo pipefail

export PYTHONFAULTHANDLER=1

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-/tmp/rr_readout_phase_abc_smoke}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/projects/BLVMob/imu-rr-seated/results/profile_clsa_qkv_fseries/runs/F0_static_shared}"
DATA_DIR="${DATA_DIR:-/projects/BLVMob/imu-rr-seated/Data}"
MDL_DIR="${MDL_DIR:-/projects/BLVMob/imu-rr-seated/models}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "${OUT_ROOT}/logs"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

"${PYTHON_BIN}" rr_readout_phase_abc_experiment.py \
  --phase all \
  --smoke \
  --subjects S12 \
  --seed 0 \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --out-dir "${OUT_ROOT}/seed_000" \
  --data-dir "${DATA_DIR}" \
  --mdl-dir "${MDL_DIR}" \
  --device "${DEVICE}" \
  --batch-size 8 \
  --cached-feature-batch-size 16 \
  --num-workers 0 \
  --resume \
  > "${OUT_ROOT}/logs/smoke.out.log" \
  2> "${OUT_ROOT}/logs/smoke.err.log"

echo "RR readout Phase ABC smoke completed under ${OUT_ROOT}/seed_000."
