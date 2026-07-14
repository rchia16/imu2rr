#!/usr/bin/env bash
set -euo pipefail

# Seeded RR benchmark runner.
# - Runs the current RR baseline suite for seeds 0..9 by default.
# - Runs the F0 family (full + single-factor ablations) for the same seeds.
# - Aggregates per-seed outputs into a combined subject export and cross-seed
#   summary tables.

RUN_ID="${RUN_ID:-jbhi_rr_seeded_$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="${ROOT:-/projects/BLVMob/imu-rr-seated/results/${RUN_ID}}"
DATA_DIR="${DATA_DIR:-/projects/BLVMob/imu-rr-seated/Data}"
DATA_GROUP="${DATA_GROUP:-mr}"
DATA_STR="${DATA_STR:-imu_filt}"
MDL_DIR="${MDL_DIR:-/projects/BLVMob/imu-rr-seated/models/imu_filt/loocv}"
DEVICE="${DEVICE:-cuda:0}"
SUBJECTS="${SUBJECTS:-S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29}"
BASELINE_MODELS="${BASELINE_MODELS:-resnet1d cnn_gru tcn inceptiontime stft_cnn patchtst crossmodal_rr}"
# SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
SEEDS="${SEEDS:-0 1 2}"

BASELINE_EPOCHS="${BASELINE_EPOCHS:-60}"
BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-128}"
BASELINE_LR="${BASELINE_LR:-1e-3}"
BASELINE_WEIGHT_DECAY="${BASELINE_WEIGHT_DECAY:-1e-4}"
BASELINE_EMB_DIM="${BASELINE_EMB_DIM:-128}"

IMU_DATALOADER_WORKERS="${IMU_DATALOADER_WORKERS:-0}"
IMU_DATALOADER_PREFETCH="${IMU_DATALOADER_PREFETCH:-1}"
IMU_DATALOADER_PIN_MEMORY="${IMU_DATALOADER_PIN_MEMORY:-0}"
IMU_DATALOADER_PERSISTENT_WORKERS="${IMU_DATALOADER_PERSISTENT_WORKERS:-0}"

F0_EPOCHS="${F0_EPOCHS:-${BASELINE_EPOCHS}}"
F0_BATCH_SIZE="${F0_BATCH_SIZE:-${BASELINE_BATCH_SIZE}}"
F0_LR="${F0_LR:-${BASELINE_LR}}"
F0_WEIGHT_DECAY="${F0_WEIGHT_DECAY:-${BASELINE_WEIGHT_DECAY}}"
F0_EMB_DIM="${F0_EMB_DIM:-${BASELINE_EMB_DIM}}"
F0_DECODER_MODE="${F0_DECODER_MODE:-cross_attn}"
F0_RR_HEAD_TYPE="${F0_RR_HEAD_TYPE:-mlp}"

PYTHON_BIN="${PYTHON_BIN:-python}"

export RUN_ID ROOT DATA_DIR DATA_GROUP DATA_STR MDL_DIR DEVICE SUBJECTS BASELINE_MODELS SEEDS
export BASELINE_EPOCHS BASELINE_BATCH_SIZE BASELINE_LR BASELINE_WEIGHT_DECAY BASELINE_EMB_DIM
export F0_EPOCHS F0_BATCH_SIZE F0_LR F0_WEIGHT_DECAY F0_EMB_DIM F0_DECODER_MODE F0_RR_HEAD_TYPE
export IMU_DATALOADER_WORKERS IMU_DATALOADER_PREFETCH IMU_DATALOADER_PIN_MEMORY IMU_DATALOADER_PERSISTENT_WORKERS

export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TORCH_DATALOADER_SHARING_STRATEGY="${TORCH_DATALOADER_SHARING_STRATEGY:-file_system}"

mkdir -p "${ROOT}/logs"

echo "[RUN] ${RUN_ID}"
echo "[RUN] root=${ROOT}"
echo "[RUN] seeds=${SEEDS}"
echo "[RUN] subjects=${SUBJECTS}"
echo "[RUN] dataloader_workers=${IMU_DATALOADER_WORKERS} prefetch=${IMU_DATALOADER_PREFETCH} pin_memory=${IMU_DATALOADER_PIN_MEMORY} persistent=${IMU_DATALOADER_PERSISTENT_WORKERS}"

# shellcheck disable=SC2206
SEED_ARRAY=(${SEEDS})

for seed in "${SEED_ARRAY[@]}"; do
  printf -v seed_tag "seed_%03d" "${seed}"
  seed_root="${ROOT}/${seed_tag}"
  seed_logs="${seed_root}/logs"
  baseline_root="${seed_root}/baselines"
  f0_root="${seed_root}/f0"

  mkdir -p "${seed_logs}" "${baseline_root}" "${f0_root}"

  echo "[SEED ${seed}] baselines -> ${baseline_root}"
  "${PYTHON_BIN}" -u run_rr_jbhi_baselines.py \
    --models "${BASELINE_MODELS}" \
    --subjects "${SUBJECTS}" \
    --data-str "${DATA_STR}" \
    --data-dir "${DATA_DIR}" \
    --data-group "${DATA_GROUP}" \
    --mdl-dir "${MDL_DIR}" \
    --out-dir "${baseline_root}" \
    --epochs "${BASELINE_EPOCHS}" \
    --batch-size "${BASELINE_BATCH_SIZE}" \
    --lr "${BASELINE_LR}" \
    --weight-decay "${BASELINE_WEIGHT_DECAY}" \
    --emb-dim "${BASELINE_EMB_DIM}" \
    --num-workers "${IMU_DATALOADER_WORKERS}" \
    --prefetch-factor "${IMU_DATALOADER_PREFETCH}" \
    --pin-memory "${IMU_DATALOADER_PIN_MEMORY}" \
    --persistent-workers "${IMU_DATALOADER_PERSISTENT_WORKERS}" \
    --seed "${seed}" \
    --device "${DEVICE}" \
    2>&1 | tee "${seed_logs}/baselines.log"

  echo "[SEED ${seed}] f0 family -> ${f0_root}"
  "${PYTHON_BIN}" -u run_rr_jbhi_f0_family.py \
    --subjects ${SUBJECTS} \
    --data-str "${DATA_STR}" \
    --data-dir "${DATA_DIR}" \
    --data-group "${DATA_GROUP}" \
    --mdl-dir "${MDL_DIR}" \
    --out-dir "${f0_root}" \
    --epochs "${F0_EPOCHS}" \
    --batch-size "${F0_BATCH_SIZE}" \
    --lr "${F0_LR}" \
    --weight-decay "${F0_WEIGHT_DECAY}" \
    --d-model "${F0_EMB_DIM}" \
    --decoder-mode "${F0_DECODER_MODE}" \
    --rr-head-type "${F0_RR_HEAD_TYPE}" \
    --num-workers "${IMU_DATALOADER_WORKERS}" \
    --prefetch-factor "${IMU_DATALOADER_PREFETCH}" \
    --pin-memory "${IMU_DATALOADER_PIN_MEMORY}" \
    --persistent-workers "${IMU_DATALOADER_PERSISTENT_WORKERS}" \
    --seed "${seed}" \
    --device "${DEVICE}" \
    2>&1 | tee "${seed_logs}/f0.log"
done

"${PYTHON_BIN}" - "${ROOT}" <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd


def _read_csv(path: Path, *, family: str, seed: int, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    if "family" not in df.columns:
        df["family"] = family
    if "seed" not in df.columns:
        df["seed"] = int(seed)
    if "source" not in df.columns:
        df["source"] = source
    return df


def _load_subject_rows(path: Path, *, family: str, seed: int, source: str) -> pd.DataFrame:
    df = _read_csv(path, family=family, seed=seed, source=source)
    for src, dst in (("rr_mae", "mae"), ("rr_rmse", "rmse"), ("rr_corr", "corr")):
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
    return df


def _cross_seed_summary(subject_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, model, mode), group in subject_rows.groupby(["family", "model", "mode"], dropna=False):
        row = {
            "family": family,
            "model": model,
            "mode": mode,
            "n_rows": int(len(group)),
            "n_subjects": int(group["subject"].nunique()) if "subject" in group.columns else 0,
            "n_seeds": int(group["seed"].nunique()) if "seed" in group.columns else 0,
        }
        for metric in ("mae", "rmse", "corr"):
            if metric in group.columns:
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_std"] = float(group[metric].std())
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["family", "model", "mode"]).reset_index(drop=True)
    return out


root = Path(sys.argv[1])
seed_dirs = sorted(p for p in root.glob("seed_*") if p.is_dir())
per_seed_tables = []
subject_tables = []

for seed_dir in seed_dirs:
    try:
        seed = int(seed_dir.name.split("_")[-1])
    except ValueError:
        continue

    baseline_root = seed_dir / "baselines"
    f0_root = seed_dir / "f0"

    baseline_summary = _read_csv(baseline_root / "summary.csv", family="baseline", seed=seed, source="baseline")
    baseline_summary["suite"] = "baseline"
    baseline_summary["variant"] = baseline_summary["mode"]
    per_seed_tables.append(baseline_summary)

    baseline_subjects = _load_subject_rows(baseline_root / "subject_rows.csv", family="baseline", seed=seed, source="baseline")
    baseline_subjects["suite"] = "baseline"
    baseline_subjects["variant"] = baseline_subjects["mode"]
    subject_tables.append(baseline_subjects)

    f0_summary = _read_csv(f0_root / "summary.csv", family="f0", seed=seed, source="f0")
    f0_summary["suite"] = "f0"
    per_seed_tables.append(f0_summary)

    f0_subjects = _load_subject_rows(f0_root / "subject_rows.csv", family="f0", seed=seed, source="f0")
    f0_subjects["suite"] = "f0"
    subject_tables.append(f0_subjects)

per_seed_summary = pd.concat(per_seed_tables, ignore_index=True) if per_seed_tables else pd.DataFrame()
subject_rows = pd.concat(subject_tables, ignore_index=True) if subject_tables else pd.DataFrame()
cross_seed_summary = _cross_seed_summary(subject_rows) if not subject_rows.empty else pd.DataFrame()

per_seed_summary.to_csv(root / "per_seed_summary.csv", index=False)
subject_rows.to_csv(root / "subject_rows.csv", index=False)
cross_seed_summary.to_csv(root / "summary.csv", index=False)

manifest = {
    "run_id": os.environ.get("RUN_ID", ""),
    "root": str(root),
    "seeds": [int(p.name.split("_")[-1]) for p in seed_dirs],
    "baseline_models": os.environ.get("BASELINE_MODELS", ""),
    "subjects": os.environ.get("SUBJECTS", ""),
    "data_str": os.environ.get("DATA_STR", ""),
    "data_dir": os.environ.get("DATA_DIR", ""),
    "data_group": os.environ.get("DATA_GROUP", ""),
    "mdl_dir": os.environ.get("MDL_DIR", ""),
    "f0_decoder_mode": os.environ.get("F0_DECODER_MODE", ""),
    "f0_rr_head_type": os.environ.get("F0_RR_HEAD_TYPE", ""),
    "files": {
        "per_seed_summary": str(root / "per_seed_summary.csv"),
        "subject_rows": str(root / "subject_rows.csv"),
        "summary": str(root / "summary.csv"),
    },
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

print(cross_seed_summary.to_string(index=False) if not cross_seed_summary.empty else "[WARN] no summary rows found")
print(f"[DONE] wrote {root}")
PY

echo "[DONE] ${ROOT}"
