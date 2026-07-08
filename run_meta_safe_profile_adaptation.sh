#!/usr/bin/env bash
set -euo pipefail

SUBJECTS_STR="${SUBJECTS:-S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29}"
EVAL_SUBJECTS_STR="${EVAL_SUBJECTS:-${SUBJECTS_STR}}"
DEVICES_STR="${DEVICES:-cuda:0}"
if [[ "${DEBUG:-0}" == "1" ]]; then
  OUT_DIR="${OUT_DIR:-results/meta_safe_profile_adaptation_debug}"
else
  OUT_DIR="${OUT_DIR:-results/meta_safe_profile_adaptation}"
fi
DATA_STR="${DATA_STR:-imu_filt}"
DATA_GROUP="${DATA_GROUP:-mr}"

read -r -a SUBJECT_ARR <<< "${SUBJECTS_STR}"
read -r -a EVAL_SUBJECT_ARR <<< "${EVAL_SUBJECTS_STR}"
read -r -a DEVICE_ARR <<< "${DEVICES_STR}"

if [[ "${DEBUG:-0}" == "1" ]]; then
  EPOCHS="${EPOCHS:-1}"
  BATCH_SIZE="${BATCH_SIZE:-8}"
  META_BATCHES="${META_BATCHES:-1}"
else
  EPOCHS="${EPOCHS:-20}"
  BATCH_SIZE="${BATCH_SIZE:-64}"
  META_BATCHES="${META_BATCHES:-4}"
fi
NUM_WORKERS="${NUM_WORKERS:-0}"
EXTRA_ARGS=()

if [[ "${DEBUG:-0}" == "1" ]]; then
  EXTRA_ARGS+=(
    --d-model 32
    --layers 1
    --heads 4
    --decoder-mode none
    --lambda-contrast 0.0
    --train-aug-ratio 0.0
    --max-train-batches 1
    --max-eval-batches 1
    --profile-stats-max-batches 1
  )
fi

COMMON_ARGS=(
  --subjects "${SUBJECT_ARR[@]}"
  --data-str "${DATA_STR}"
  --data-group "${DATA_GROUP}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --meta-every 1
  --meta-batches "${META_BATCHES}"
  --correction-modes none readout_affine profile_film qkv_last1_small
  --profile-film-scale 0.1
  --profile-qkv-scale 0.01
  --profile-qkv-layers last1
  --lambda-meta 1.0
  --lambda-harm 1.0
  --lambda-moment 0.01
  --lambda-update 0.01
  --safety-mode fallback
  --subject-balanced
  --use-moment-loss
  --use-meta-no-harm
  "${EXTRA_ARGS[@]}"
  "$@"
)

merge_chunk_outputs() {
  local root="$1"
  local chunks_dir="$2"
  python - "$root" "$chunks_dir" <<'PY'
from pathlib import Path
import json
import shutil
import sys

import pandas as pd

root = Path(sys.argv[1])
chunks_dir = Path(sys.argv[2])
root.mkdir(parents=True, exist_ok=True)
chunk_dirs = sorted(p for p in chunks_dir.glob("gpu_*") if p.is_dir())

def read_csvs(name):
    frames = []
    for d in chunk_dirs:
        path = d / name
        if path.exists() and path.stat().st_size > 0:
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

first_config = next((d / "config.json" for d in chunk_dirs if (d / "config.json").exists()), None)
if first_config is not None:
    shutil.copyfile(first_config, root / "config.json")

meta = read_csvs("meta_train_log.csv")
meta.to_csv(root / "meta_train_log.csv", index=False)

subject = read_csvs("subject_metrics.csv")
if subject.empty:
    subject.to_csv(root / "subject_metrics.csv", index=False)
    pd.DataFrame().to_csv(root / "summary.csv", index=False)
    pd.DataFrame().to_csv(root / "harm_summary.csv", index=False)
    pd.DataFrame().to_csv(root / "correction_budget_summary.csv", index=False)
    raise SystemExit(0)

subject = subject.sort_values(["subject", "mode"]).reset_index(drop=True)
if "delta_vs_none" not in subject.columns or subject["delta_vs_none"].isna().any():
    none = subject[subject["mode"] == "none"].set_index("subject")["post_mae"].to_dict()
    subject["delta_vs_none"] = [
        float(row["post_mae"]) - float(none.get(row["subject"], row["post_mae"]))
        for _, row in subject.iterrows()
    ]
subject["improved_vs_none"] = subject["delta_vs_none"] < -1e-8
subject["worse_vs_none"] = subject["delta_vs_none"] > 1e-8
subject.to_csv(root / "subject_metrics.csv", index=False)

summary_rows = []
harm_rows = []
budget_rows = []
for mode, g in subject.groupby("mode", sort=False):
    d = g["delta_vs_none"].astype(float)
    summary_rows.append({
        "mode": mode,
        "n_subjects": int(g["subject"].nunique()),
        "post_mae_mean": float(g["post_mae"].mean()),
        "post_mae_std": float(g["post_mae"].std(ddof=0)),
        "delta_vs_none_mean": float(g["delta_vs_none"].mean()),
        "subjects_improved_vs_none": int(g["improved_vs_none"].sum()),
        "subjects_worse_vs_none": int(g["worse_vs_none"].sum()),
        "worst_subject_degradation": float(d.max()),
        "fallback_rate_mean": float(g["safety_fallback_rate"].mean()),
    })
    harm_rows.append({
        "mode": mode,
        "subjects_harmed": int(g["worse_vs_none"].sum()),
        "subjects_improved": int(g["improved_vs_none"].sum()),
        "mean_degradation_positive_only": float(d[d > 0].mean()) if bool((d > 0).any()) else 0.0,
        "worst_subject_degradation": float(d.max()),
    })
    budget_rows.append({
        "mode": mode,
        "mean_abs_rr_shift": float(g["mean_abs_rr_shift"].mean()),
        "max_abs_rr_shift": float(g["max_abs_rr_shift"].max()),
        "profile_norm_mean": float(g["profile_norm"].mean()),
        "update_norm_mean": float(g["update_norm"].mean()),
        "fallback_rate_mean": float(g["safety_fallback_rate"].mean()),
    })

pd.DataFrame(summary_rows).to_csv(root / "summary.csv", index=False)
pd.DataFrame(harm_rows).to_csv(root / "harm_summary.csv", index=False)
pd.DataFrame(budget_rows).to_csv(root / "correction_budget_summary.csv", index=False)
PY
}

if [[ "${#DEVICE_ARR[@]}" -le 1 ]]; then
  DEVICE="${DEVICE_ARR[0]}"
  python meta_safe_profile_adaptation.py \
    "${COMMON_ARGS[@]}" \
    --eval-subjects "${EVAL_SUBJECT_ARR[@]}" \
    --out-dir "${OUT_DIR}" \
    --device "${DEVICE}"
else
  RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
  CHUNKS_DIR="${OUT_DIR}/chunks/${RUN_ID}"
  mkdir -p "${CHUNKS_DIR}"
  pids=()
  for gpu_idx in "${!DEVICE_ARR[@]}"; do
    device="${DEVICE_ARR[$gpu_idx]}"
    eval_chunk=()
    for subj_idx in "${!EVAL_SUBJECT_ARR[@]}"; do
      if (( subj_idx % ${#DEVICE_ARR[@]} == gpu_idx )); then
        eval_chunk+=("${EVAL_SUBJECT_ARR[$subj_idx]}")
      fi
    done
    if [[ "${#eval_chunk[@]}" -eq 0 ]]; then
      continue
    fi
    chunk_dir="${CHUNKS_DIR}/gpu_${gpu_idx}"
    echo "[RUN] device=${device} eval_subjects=${eval_chunk[*]} out_dir=${chunk_dir}"
    python meta_safe_profile_adaptation.py \
      "${COMMON_ARGS[@]}" \
      --eval-subjects "${eval_chunk[@]}" \
      --out-dir "${chunk_dir}" \
      --device "${device}" &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
  merge_chunk_outputs "${OUT_DIR}" "${CHUNKS_DIR}"
  echo "[DONE] merged multi-GPU outputs into ${OUT_DIR}"
fi
