#!/usr/bin/env bash
set -euo pipefail

# Two-GPU JBHI/TSLib source-baseline runner.
#
# Purpose:
#   Run independent neural baseline jobs concurrently on two GPUs without
#   writing into the same output directory at the same time.
#
# NOTE:
#   This script only parallelizes the source neural baseline stage. The full
#   cross-modal adaptation/alpha/prototype-gate scripts train/evaluate large
#   same-checkpoint sweeps and should be run separately unless you explicitly
#   enable RUN_ADAPTATION_AFTER_BASELINES=1 below.
#
# NOTE:
#   This v3 runner calls run_rr_jbhi_tslib_neural_baselines_v3.py, which imports
#   rr_jbhi_tslib_source_models_v2.py and can import existing TimesNet results. This avoids depending on the deprecated
#   rr_jbhi_source_baseline_models.py scaffold.

RUN_ID="${RUN_ID:-jbhi_tslib_2gpu_$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="${ROOT:-/projects/BLVMob/imu-rr-seated/results/${RUN_ID}}"
DATA_DIR="${DATA_DIR:-/projects/BLVMob/imu-rr-seated/Data}"
DATA_GROUP="${DATA_GROUP:-mr}"
DATA_STR="${DATA_STR:-imu_filt}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-128}"
PATCHTST_FILE="${PATCHTST_FILE:-PatchTST.py}"
TIMESNET_FILE="${TIMESNET_FILE:-TimesNet.py}"
TIMESNET_MODE="${TIMESNET_MODE:-downsample}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"

# GPU ids visible on this machine. Override for a different pair, e.g. GPUS="1 3".
GPUS="${GPUS:-0 1}"
read -r GPU0 GPU1 _ <<< "${GPUS}"
if [[ -z "${GPU0:-}" || -z "${GPU1:-}" ]]; then
  echo "[ERROR] GPUS must contain at least two GPU ids, e.g. GPUS='0 1'" >&2
  exit 2
fi

# Default split keeps the heaviest TSLib models separated.
MODELS_GPU0="${MODELS_GPU0:-resnet1d tcn patchtst_tslib}"
MODELS_GPU1="${MODELS_GPU1:-cnn_gru inceptiontime timesnet_tslib}"

# Set this to 1 only if you want the full adaptation scripts to run after the
# neural baselines. They are not run concurrently here to avoid checkpoint and
# output contention.
RUN_ADAPTATION_AFTER_BASELINES="${RUN_ADAPTATION_AFTER_BASELINES:-0}"
ADAPT_DEVICE="${ADAPT_DEVICE:-cuda:${GPU0}}"

export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PATCHTST_FILE
export TIMESNET_FILE

mkdir -p "${ROOT}/logs"

BASE_ROOT="${ROOT}/source_neural_baselines"
WORKER0_ROOT="${BASE_ROOT}/worker_gpu${GPU0}"
WORKER1_ROOT="${BASE_ROOT}/worker_gpu${GPU1}"
MERGED_ROOT="${BASE_ROOT}/merged"
mkdir -p "${WORKER0_ROOT}" "${WORKER1_ROOT}" "${MERGED_ROOT}"

run_worker() {
  local gpu="$1"
  local models="$2"
  local out_dir="$3"
  local log_file="$4"
  local skip_args=()
  if [[ "${SKIP_COMPLETED}" == "1" ]]; then
    skip_args+=(--skip-existing)
  fi
  echo "[START] gpu=${gpu} models=${models} out=${out_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u run_rr_jbhi_tslib_neural_baselines_v3.py \
    --models "${models}" \
    --data-str "${DATA_STR}" \
    --data-dir "${DATA_DIR}" \
    --data-group "${DATA_GROUP}" \
    --out-dir "${out_dir}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --device "cuda:0" \
    --patchtst-file "${PATCHTST_FILE}" \
    --timesnet-file "${TIMESNET_FILE}" \
    --timesnet-mode "${TIMESNET_MODE}" \
    "${skip_args[@]}" \
    2>&1 | tee "${log_file}"
}

set +e
run_worker "${GPU0}" "${MODELS_GPU0}" "${WORKER0_ROOT}" "${ROOT}/logs/source_neural_baselines_gpu${GPU0}.log" &
pid0=$!
run_worker "${GPU1}" "${MODELS_GPU1}" "${WORKER1_ROOT}" "${ROOT}/logs/source_neural_baselines_gpu${GPU1}.log" &
pid1=$!

wait "${pid0}"
status0=$?
wait "${pid1}"
status1=$?
set -e

if [[ "${status0}" -ne 0 || "${status1}" -ne 0 ]]; then
  echo "[ERROR] One or more GPU workers failed: gpu${GPU0}=${status0}, gpu${GPU1}=${status1}" >&2
  exit 1
fi

python - <<'PY' "${WORKER0_ROOT}" "${WORKER1_ROOT}" "${MERGED_ROOT}"
from pathlib import Path
import json
import shutil
import sys
import pandas as pd

worker_roots = [Path(sys.argv[1]), Path(sys.argv[2])]
merged = Path(sys.argv[3])
merged.mkdir(parents=True, exist_ok=True)

def rewrite_path_value(value, src_root, dst_root):
    if pd.isna(value):
        return value
    text = str(value)
    if not text:
        return text
    src = str(src_root.resolve())
    dst = str(dst_root.resolve())
    parts = text.split()
    if not parts:
        return text
    rewritten = []
    changed = False
    for part in parts:
        if part.startswith(src):
            rewritten.append(dst + part[len(src):])
            changed = True
        else:
            rewritten.append(part)
    return " ".join(rewritten) if changed else text

def rewrite_checkpoint_columns(df, src_root, dst_root):
    if 'checkpoint_path' in df.columns:
        df = df.copy()
        df['checkpoint_path'] = df['checkpoint_path'].map(
            lambda value: rewrite_path_value(value, src_root, dst_root)
        )
    return df

def rewrite_csv_checkpoint_columns(path, src_root, dst_root):
    if not path.exists():
        return
    df = pd.read_csv(path)
    out = rewrite_checkpoint_columns(df, src_root, dst_root)
    if out is not df:
        out.to_csv(path, index=False)

def rewrite_manifest_payload(obj, src_root, dst_root):
    if isinstance(obj, dict):
        return {
            key: rewrite_manifest_payload(value, src_root, dst_root)
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [rewrite_manifest_payload(value, src_root, dst_root) for value in obj]
    if isinstance(obj, str):
        return rewrite_path_value(obj, src_root, dst_root)
    return obj

def rewrite_checkpoint_manifests(root, src_root, dst_root):
    for path in root.glob('subjects/*/checkpoint_manifest.json'):
        with path.open() as f:
            payload = json.load(f)
        payload = rewrite_manifest_payload(payload, src_root, dst_root)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))

subject_frames = []
summary_frames = []
for root in worker_roots:
    sr = root / 'subject_rows.csv'
    sm = root / 'summary.csv'
    if sr.exists():
        subject_frames.append(rewrite_checkpoint_columns(pd.read_csv(sr), root, merged))
    if sm.exists():
        summary_frames.append(pd.read_csv(sm))
    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue
        if model_dir.name.startswith('worker_') or model_dir.name == 'merged':
            continue
        dest = merged / model_dir.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(model_dir, dest)
        rewrite_csv_checkpoint_columns(dest / 'subject_rows.csv', root, merged)
        for members_csv in dest.glob('subjects/*/ensemble_members.csv'):
            rewrite_csv_checkpoint_columns(members_csv, root, merged)
        rewrite_checkpoint_manifests(dest, root, merged)

if subject_frames:
    pd.concat(subject_frames, ignore_index=True).to_csv(merged / 'subject_rows.csv', index=False)
if summary_frames:
    summary = pd.concat(summary_frames, ignore_index=True)
    if 'mae_mean' in summary.columns:
        summary = summary.sort_values('mae_mean')
    summary.to_csv(merged / 'summary.csv', index=False)
    print(summary.to_string(index=False))
print(f'[MERGED] {merged}')
PY

# Convenience copy: keep merged summary at the original source_neural_baselines level.
cp "${MERGED_ROOT}/subject_rows.csv" "${BASE_ROOT}/subject_rows.csv"
cp "${MERGED_ROOT}/summary.csv" "${BASE_ROOT}/summary.csv"

if [[ "${RUN_ADAPTATION_AFTER_BASELINES}" == "1" ]]; then
  echo "[ADAPTATION] Running full 2-GPU prototype/OOD gate adaptation under ${ROOT}"
  OUT_DIR="${ROOT}/adaptation_prototype_gate_full_2gpu_v2" \
    GPUS="${GPUS}" \
    DATA_DIR="${DATA_DIR}" \
    DATA_GROUP="${DATA_GROUP}" \
    DATA_STR="${DATA_STR}" \
    bash run_adaptation_prototype_gate_tests_full_2gpu_v2.sh \
    2>&1 | tee "${ROOT}/logs/adaptation_prototype_gate_2gpu_v2.log"
fi

echo "[DONE] ${ROOT}"
echo "[SUMMARY] ${BASE_ROOT}/summary.csv"
