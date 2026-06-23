#!/usr/bin/env bash
set -euo pipefail

# Full no-shortcut prototype/OOD safety-gate adaptation runner.
# Runs the full adaptation ladder and then the full prototype gate analyzer.

RUN_ID="${RUN_ID:-sparc_adaptation_prototype_gate_full_random32}"
OUT_DIR="${OUT_DIR:-/projects/BLVMob/imu-rr-seated/results/${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

LADDER="${LADDER:-vit_pressure_crossmodal_stft_rr_adaptation_prototype_gate_sweep_full.py}"
SUMMARIZER="${SUMMARIZER:-summarize_rr_adaptation_prototype_gate_sweep_full.py}"
POLICY="${POLICY:-analyze_adaptation_prototype_gate_policy_full.py}"

TTA_MODES="${TTA_MODES:-none adapt_mean_alpha_050 adapt_mean_alpha_075 adapt_mean_alpha_100 profile_film_init_only profile_film_unsup_sparc direct_stft_rr hybrid_probe_stft_conf}"

export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

mkdir -p "${OUT_DIR}/logs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

"${PYTHON_BIN}" -u "${LADDER}" \
  --subjects S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29 \
  --eval-subjects S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29 \
  --data-str imu_filt \
  --data-dir /projects/BLVMob/imu-rr-seated/Data \
  --data-group mr \
  --mdl-dir /projects/BLVMob/imu-rr-seated/models/imu_filt/loocv \
  --out-dir "${OUT_DIR}" \
  --sweep-root "${OUT_DIR}" \
  --sweep-run-id prototype_gate_full_random32 \
  --device cuda:0 \
  --epochs 20 \
  --rr-probe-epochs 100 \
  --use-profile-film \
  --profile-conditioning film \
  --profile-film-scale 0.1 \
  --decoder-mode cross_attn \
  --rr-tta-modes "${TTA_MODES}" \
  --use-unsup-mode-defaults \
  --target-calibration-windows 32 \
  --target-calibration-mode random \
  --profile-unsup-adapt-scope calibration \
  --exclude-calibration-from-eval \
  --adaptation-use-calibration-only \
  2>&1 | tee "${OUT_DIR}/logs/prototype_gate_train_${STAMP}.log"

"${PYTHON_BIN}" -u "${SUMMARIZER}" \
  --root "${OUT_DIR}" \
  --out-csv "${OUT_DIR}/summary.csv" \
  --combined-subject-csv "${OUT_DIR}/subject_rows.csv" \
  2>&1 | tee "${OUT_DIR}/logs/prototype_gate_summary_${STAMP}.log"

"${PYTHON_BIN}" -u "${POLICY}" \
  --subject-rows "${OUT_DIR}/subject_rows.csv" \
  --out-dir "${OUT_DIR}/adaptation_prototype_gate_policy" \
  --candidates "${TTA_MODES}" \
  --alpha-modes "adapt_mean_alpha_050 adapt_mean_alpha_075" \
  --profile-mode profile_film_init_only \
  --include-profile-fallback \
  --min-gain 0.02 \
  --safe-threshold 0.55 \
  --knn-k 3 \
  --ood-quantile 0.95 \
  --reject-ood \
  --ridge-alpha 10.0 \
  2>&1 | tee "${OUT_DIR}/logs/prototype_gate_policy_${STAMP}.log"

echo "[DONE] Full prototype gate policy: ${OUT_DIR}/adaptation_prototype_gate_policy"
