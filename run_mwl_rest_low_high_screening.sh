#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="${ROOT:-results/a1_mwl_probe/rest_low_high/screening_${TIMESTAMP}}"
DIRECT_ROOT="${DIRECT_ROOT:-${ROOT}/direct_imu}"
A1_ROOT="${A1_ROOT:-${ROOT}/a1_probe}"
COMBINED_ROOT="${COMBINED_ROOT:-${ROOT}/combined_benchmark}"
A1_FEATURE_CACHE_ROOT="${A1_FEATURE_CACHE_ROOT:?Set A1_FEATURE_CACHE_ROOT to frozen F0+A1 cache root.}"

SUBJECTS="${SUBJECTS:-S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29}"
SEEDS="${SEEDS:-0 1 2}"
DIRECT_MODELS="${DIRECT_MODELS:-resnet1d cnn_gru tcn inceptiontime stft_cnn}"
A1_VARIANTS="${A1_VARIANTS:-a1_rr_only_logreg a1_mean_logreg a1_distribution_logreg a1_latent_logreg a1_distribution_mlp a1_latent_mlp a1_distribution_activity_mlp}"
DATA_STR="${DATA_STR:-imu_filt}"
DATA_DIR="${DATA_DIR:-/projects/BLVMob/imu-rr-seated/Data}"
DATA_GROUP="${DATA_GROUP:-mr_levels}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-128}"
VAL_SUBJECTS="${VAL_SUBJECTS:-3}"
PATIENCE="${PATIENCE:-12}"

mkdir -p "$ROOT"

echo "[SCREENING] root=${ROOT}"
echo "[SCREENING] seeds=${SEEDS}"
echo "[SCREENING] subjects=${SUBJECTS}"

for seed in $SEEDS; do
  seed_direct_root="${DIRECT_ROOT}/seed_$(printf '%03d' "$seed")"
  "$PYTHON_BIN" -u run_mwl_direct_imu_benchmarks.py \
    --models "$DIRECT_MODELS" \
    --subjects "$SUBJECTS" \
    --data-str "$DATA_STR" \
    --data-dir "$DATA_DIR" \
    --data-group "$DATA_GROUP" \
    --out-dir "$seed_direct_root" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --val-subjects "$VAL_SUBJECTS" \
    --patience "$PATIENCE" \
    --seed "$seed" \
    --device "$DEVICE"
done

"$PYTHON_BIN" -u run_a1_mwl_probe.py \
  --feature-cache-root "$A1_FEATURE_CACHE_ROOT" \
  --out-dir "$A1_ROOT" \
  --subjects "$SUBJECTS" \
  --seeds "$SEEDS" \
  --variants "$A1_VARIANTS" \
  --device "$DEVICE"

"$PYTHON_BIN" -u aggregate_mwl_rest_low_high_benchmark.py \
  --prediction-roots "$DIRECT_ROOT" "$A1_ROOT" \
  --out-dir "$COMBINED_ROOT" \
  --subjects "$SUBJECTS" \
  --seeds "$SEEDS"

echo "[DONE] combined benchmark: ${COMBINED_ROOT}"
