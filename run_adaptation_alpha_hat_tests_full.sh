#!/usr/bin/env bash
set -euo pipefail

# Full no-shortcut learned-alpha adaptation runner.
# Uses the full same-checkpoint adaptation ladder and full learned alpha policy.

RUN_ID="${RUN_ID:-sparc_adaptation_alpha_hat_full_random32}"
OUT_DIR="${OUT_DIR:-/projects/BLVMob/imu-rr-seated/results/${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

LADDER="${LADDER:-vit_pressure_crossmodal_stft_rr_adaptation_alpha_hat_sweep.py}"
SUMMARIZER="${SUMMARIZER:-summarize_rr_adaptation_alpha_hat_sweep.py}"
POLICY="${POLICY:-analyze_adaptation_alpha_hat_policy_full.py}"

EXPERIMENT_SET="${EXPERIMENT_SET:-alpha_hat}"
case "${EXPERIMENT_SET}" in
  alpha_hat)
    TTA_MODES="${TTA_MODES:-none adapt_mean_alpha_025 adapt_mean_alpha_050 adapt_mean_alpha_075 adapt_mean_alpha_100 profile_film_init_only profile_film_unsup_sparc direct_stft_rr hybrid_probe_stft_conf}"
    ALPHA_GRID_MODES="${ALPHA_GRID_MODES:-none:0 adapt_mean_alpha_025:0.25 adapt_mean_alpha_050:0.5 adapt_mean_alpha_075:0.75 adapt_mean_alpha_100:1.0}"
    LEARN_ALPHA_FEATURE_MODE="${LEARN_ALPHA_FEATURE_MODE:-adapt_mean_alpha_100}"
    ;;
  feature_mean_align)
    TTA_MODES="${TTA_MODES:-none feature_mean_align_alpha050 feature_mean_align_alpha075 feature_mean_align_alpha100 feature_mean_align_profile_shrink}"
    ALPHA_GRID_MODES="${ALPHA_GRID_MODES:-none:0 feature_mean_align_alpha050:0.5 feature_mean_align_alpha075:0.75 feature_mean_align_alpha100:1.0}"
    LEARN_ALPHA_FEATURE_MODE="${LEARN_ALPHA_FEATURE_MODE:-feature_mean_align_alpha100}"
    ;;
  *)
    echo "[ERROR] Unknown EXPERIMENT_SET=${EXPERIMENT_SET}" >&2
    exit 2
    ;;
esac

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
  --sweep-run-id alpha_hat_full_random32 \
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
  2>&1 | tee "${OUT_DIR}/logs/alpha_hat_train_${STAMP}.log"

"${PYTHON_BIN}" -u "${SUMMARIZER}" \
  --root "${OUT_DIR}" \
  --out-csv "${OUT_DIR}/summary.csv" \
  --combined-subject-csv "${OUT_DIR}/subject_rows.csv" \
  2>&1 | tee "${OUT_DIR}/logs/alpha_hat_summary_${STAMP}.log"

"${PYTHON_BIN}" -u "${POLICY}" \
  --subject-rows "${OUT_DIR}/subject_rows.csv" \
  --sweep-root "${OUT_DIR}" \
  --out-dir "${OUT_DIR}/adaptation_alpha_hat_policy" \
  --candidates "${TTA_MODES}" \
  --strict-no-label \
  --learn-alpha-policy \
  --alpha-grid-modes "${ALPHA_GRID_MODES}" \
  --alpha-target-method quadratic_safe \
  --learn-alpha-feature-mode "${LEARN_ALPHA_FEATURE_MODE}" \
  --learn-alpha-film-source-mode profile_film_init_only \
  --learn-alpha-film-residual-lambda 0.25 \
  2>&1 | tee "${OUT_DIR}/logs/alpha_hat_policy_${STAMP}.log"

echo "[DONE] Full alpha-hat policy: ${OUT_DIR}/adaptation_alpha_hat_policy"
