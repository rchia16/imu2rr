# JBHI RR baseline and main experiment scripts

This script set is intended for a clean JBHI-style comparison of wearable IMU
respiration-rate estimation under unseen-subject shift.

## Files

- `rr_jbhi_models.py`
  - Representative neural baselines:
    - `resnet1d`
    - `cnn_gru`
    - `tcn`
    - `inceptiontime`
    - `stft_cnn`
    - `patchtst`
    - `crossmodal_rr`
  - The implementations are compact PyTorch adaptations designed to be readable
    and consistent with this project. `NOTE` comments mark where they simplify
    official architectures.

- `run_rr_jbhi_baselines.py`
  - LOSO training/evaluation harness using your existing `dataloader.py`.
  - Saves per-subject predictions, embeddings, fold histories, subject rows,
    and summary CSVs.
  - Supports the embedding-level fixed `alpha075` mean-shift evaluation.

- `analyze_rr_jbhi_embeddings.py`
  - Loads saved embeddings.
  - Writes t-SNE plots, prototype distance tables, and adaptation-shift tables.

- `run_adaptation_prototype_gate_tests.sh`
  - Previously generated main test-time adaptation experiment.
  - Implements the option requested earlier:

```text
hard-coded alpha_075 + learned prototype/OOD safety gate + fallback to none
```

- `analyze_adaptation_prototype_gate_policy.py`
  - Strict no-label prototype/OOD safety-gate analyzer.
  - Uses source-subject prototypes to decide whether fixed alpha adaptation is
    safe.

- `make_rr_jbhi_tables.py`
  - Combines baseline and adaptation outputs into manuscript tables.

- `run_rr_jbhi_main_suite.sh`
  - Runs the whole suite.

## References / baseline mapping

- InceptionTime:
  - Official repo: https://github.com/hfawaz/InceptionTime
  - NOTE: `InceptionTimeRR` is a compact PyTorch adaptation, not the full
    5-member ensemble.

- PatchTST:
  - Official repo: https://github.com/PatchTST/PatchTST
  - NOTE: `PatchTSTRR` preserves the patch-token Transformer idea but simplifies
    channel independence for multichannel IMU RR regression.

- TimesNet:
  - Official TSLib repo: https://github.com/thuml/Time-Series-Library
  - NOTE: use your existing `times_experiment.py` for the exact TimesNet/TSLib
    path. This suite keeps TimesNet as an external baseline to avoid copying
    upstream TSLib code.

- TS-TCC:
  - Official repo: https://github.com/emadeldeen24/TS-TCC
  - NOTE: not included as a default supervised baseline here. It is a good
    optional representation-learning comparison if you want one additional
    self-supervised encoder.

## Run full suite

Copy all scripts into the repo root, then run:

```bash
bash run_rr_jbhi_main_suite.sh
```

Common overrides:

```bash
RUN_ID="jbhi_rr_baselines_v1" \
MODELS="resnet1d cnn_gru tcn inceptiontime stft_cnn patchtst crossmodal_rr" \
EPOCHS=60 \
BATCH_SIZE=128 \
DEVICE=cuda:0 \
bash run_rr_jbhi_main_suite.sh
```

## Run only baselines

```bash
python run_rr_jbhi_baselines.py \
  --models "resnet1d cnn_gru tcn inceptiontime stft_cnn patchtst crossmodal_rr" \
  --data-str imu_filt \
  --data-dir /projects/BLVMob/imu-rr-seated/Data \
  --data-group mr \
  --out-dir /projects/BLVMob/imu-rr-seated/results/jbhi_rr_baselines \
  --epochs 60 \
  --device cuda:0
```

## Run embedding diagnostics

```bash
python analyze_rr_jbhi_embeddings.py \
  --root /projects/BLVMob/imu-rr-seated/results/jbhi_rr_baselines \
  --out-dir /projects/BLVMob/imu-rr-seated/results/jbhi_rr_baselines/embedding_diagnostics
```

## Run prototype gate only

```bash
bash run_adaptation_prototype_gate_tests.sh
```

or, from an existing adaptation `subject_rows.csv`:

```bash
python analyze_adaptation_prototype_gate_policy.py \
  --subject-rows /path/to/subject_rows.csv \
  --out-dir /path/to/adaptation_prototype_gate_policy \
  --alpha-modes "adapt_mean_alpha_050 adapt_mean_alpha_075" \
  --profile-mode profile_film_init_only
```

## What to report in the paper

Recommended main table:

```text
Classical baselines:
  Hernandez/Rodiger
  STFT dominant frequency
  handcrafted ridge/SVR

Neural baselines:
  ResNet1D
  CNN-GRU
  TCN
  InceptionTime-style
  STFT-CNN
  PatchTST-style
  TimesNet/NormWear from existing scripts, if available

Main method / ablations:
  crossmodal_rr
  crossmodal_rr + alpha075
  prototype_ridge_gate_adapt_mean_alpha_075
  profile_film_init_only / SPARC from exact adaptation sweep
```

Recommended diagnostics:

```text
subject-level MAE table
prototype/OOD gate confusion table
harm avoidance vs missed opportunity
embedding t-SNE by subject
embedding t-SNE by RR error
adaptation shift diagnostics
```
