#!/usr/bin/env bash
set -euo pipefail

# Two-GPU full no-shortcut learned-alpha adaptation runner.
#
# Strategy:
#   - keep the source-subject pool identical on both workers via --subjects;
#   - split only --eval-subjects across two GPUs;
#   - write each worker into an isolated sweep root;
#   - summarize recursively over both worker roots;
#   - run the learned-alpha policy once on the merged subject_rows.csv.
#
# NOTE:
#   Each worker sees CUDA_VISIBLE_DEVICES=<physical_gpu> and the Python process
#   uses --device cuda:0. This avoids hard-coding physical cuda indices inside
#   Python and matches common cluster practice.

RUN_ID="${RUN_ID:-sparc_adaptation_alpha_hat_full_2gpu_v2_random32}"
OUT_DIR="${OUT_DIR:-/projects/BLVMob/imu-rr-seated/results/${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

LADDER="${LADDER:-vit_pressure_crossmodal_stft_rr_adaptation_alpha_hat_sweep.py}"
SUMMARIZER="${SUMMARIZER:-summarize_rr_adaptation_alpha_hat_sweep.py}"
POLICY="${POLICY:-analyze_adaptation_alpha_hat_policy_full.py}"

DATA_STR="${DATA_STR:-imu_filt}"
DATA_DIR="${DATA_DIR:-/projects/BLVMob/imu-rr-seated/Data}"
DATA_GROUP="${DATA_GROUP:-mr}"
MDL_DIR="${MDL_DIR:-/projects/BLVMob/imu-rr-seated/models/imu_filt/loocv}"
EPOCHS="${EPOCHS:-20}"
RR_PROBE_EPOCHS="${RR_PROBE_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"

SUBJECTS="${SUBJECTS:-S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29}"
EVAL_SUBJECTS="${EVAL_SUBJECTS:-${SUBJECTS}}"
TTA_MODES="${TTA_MODES:-none adapt_mean_alpha_025 adapt_mean_alpha_050 adapt_mean_alpha_075 adapt_mean_alpha_100 profile_film_init_only profile_film_unsup_sparc direct_stft_rr hybrid_probe_stft_conf}"

GPUS="${GPUS:-0 1}"
read -r GPU0 GPU1 _ <<< "${GPUS}"
if [[ -z "${GPU0:-}" || -z "${GPU1:-}" ]]; then
  echo "[ERROR] GPUS must contain at least two GPU ids, e.g. GPUS='0 1'" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/workers"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

# Split eval subjects alternately across the two workers unless explicitly set.
if [[ -n "${EVAL_SUBJECTS_GPU0:-}" && -n "${EVAL_SUBJECTS_GPU1:-}" ]]; then
  SPLIT0="${EVAL_SUBJECTS_GPU0}"
  SPLIT1="${EVAL_SUBJECTS_GPU1}"
else
  read -r -a EVAL_ARRAY <<< "${EVAL_SUBJECTS}"
  split0=()
  split1=()
  for i in "${!EVAL_ARRAY[@]}"; do
    if (( i % 2 == 0 )); then
      split0+=("${EVAL_ARRAY[$i]}")
    else
      split1+=("${EVAL_ARRAY[$i]}")
    fi
  done
  SPLIT0="${split0[*]}"
  SPLIT1="${split1[*]}"
fi

if [[ -z "${SPLIT0}" || -z "${SPLIT1}" ]]; then
  echo "[ERROR] Empty split. SPLIT0='${SPLIT0}' SPLIT1='${SPLIT1}'" >&2
  exit 2
fi

echo "[SPLIT] GPU ${GPU0}: ${SPLIT0}"
echo "[SPLIT] GPU ${GPU1}: ${SPLIT1}"

run_worker() {
  local gpu="$1"
  local eval_subjects="$2"
  local worker_name="$3"
  local worker_root="${OUT_DIR}/workers/${worker_name}"
  local log_file="${OUT_DIR}/logs/alpha_hat_train_${worker_name}_${STAMP}.log"
  mkdir -p "${worker_root}"
  echo "[START] alpha_hat ${worker_name} gpu=${gpu} eval_subjects=${eval_subjects}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u "${LADDER}" \
    --subjects ${SUBJECTS} \
    --eval-subjects ${eval_subjects} \
    --data-str "${DATA_STR}" \
    --data-dir "${DATA_DIR}" \
    --data-group "${DATA_GROUP}" \
    --mdl-dir "${MDL_DIR}" \
    --out-dir "${worker_root}" \
    --sweep-root "${worker_root}" \
    --sweep-run-id "alpha_hat_full_2gpu_v2_${worker_name}" \
    --device cuda:0 \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --rr-probe-epochs "${RR_PROBE_EPOCHS}" \
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
    2>&1 | tee "${log_file}"
}

set +e
run_worker "${GPU0}" "${SPLIT0}" "gpu${GPU0}" &
pid0=$!
run_worker "${GPU1}" "${SPLIT1}" "gpu${GPU1}" &
pid1=$!
wait "${pid0}"
status0=$?
wait "${pid1}"
status1=$?
set -e

if [[ "${status0}" -ne 0 || "${status1}" -ne 0 ]]; then
  echo "[ERROR] alpha_hat workers failed: gpu${GPU0}=${status0}, gpu${GPU1}=${status1}" >&2
  exit 1
fi

"${PYTHON_BIN}" -u "${SUMMARIZER}" \
  --root "${OUT_DIR}/workers" \
  --out-csv "${OUT_DIR}/summary.csv" \
  --combined-subject-csv "${OUT_DIR}/subject_rows.csv" \
  2>&1 | tee "${OUT_DIR}/logs/alpha_hat_summary_${STAMP}.log"

"${PYTHON_BIN}" -u "${POLICY}" \
  --subject-rows "${OUT_DIR}/subject_rows.csv" \
  --sweep-root "${OUT_DIR}/workers" \
  --out-dir "${OUT_DIR}/adaptation_alpha_hat_policy" \
  --candidates "${TTA_MODES}" \
  --strict-no-label \
  --learn-alpha-policy \
  --alpha-grid-modes "none:0 adapt_mean_alpha_025:0.25 adapt_mean_alpha_050:0.5 adapt_mean_alpha_075:0.75 adapt_mean_alpha_100:1.0" \
  --alpha-target-method quadratic_safe \
  --learn-alpha-feature-mode adapt_mean_alpha_100 \
  --learn-alpha-film-source-mode profile_film_init_only \
  --learn-alpha-film-residual-lambda 0.25 \
  2>&1 | tee "${OUT_DIR}/logs/alpha_hat_policy_${STAMP}.log"

echo "[DONE] Full 2-GPU alpha-hat policy: ${OUT_DIR}/adaptation_alpha_hat_policy"
