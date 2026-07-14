# TCN Subject Profiling and Safe Test-Time Training Experiment

## 1. Purpose

Implement a standalone experiment that tests whether the existing plain TCN respiratory-rate model can be improved using:

1. static subject profiling;
2. bounded Profile-FiLM;
3. profile-conditioned RR gain and bias correction;
4. one-step unsupervised test-time training;
5. profile-gated adaptation;
6. pseudo-held-out source-subject meta-training;
7. optional profile-gated feature-mean alignment.

The experiment must preserve the plain TCN as the primary predictor. Profiling and test-time training must operate only as small, bounded corrections around the source-trained TCN.

The experiment must not replace or silently modify the validated baseline.

---

## 2. Scientific motivation

The plain TCN currently provides strong and comparatively seed-stable respiratory-rate performance.

The working hypothesis is that some held-out-subject errors arise from low-dimensional shifts rather than a failure of the temporal representation itself.

Potential sources of shift include:

* headset orientation;
* sensor placement;
* sensor coupling;
* subject movement amplitude;
* subject breathing amplitude;
* latent feature mean and scale;
* RR prediction gain;
* RR prediction bias.

The proposed model is:

[
h_{1:T}=\operatorname{TCN}*{\theta}(x*{1:T}),
]

[
z=\operatorname{Pool}(h_{1:T}),
]

[
z'=\gamma(p_s)\odot z+\beta(p_s),
]

[
\hat r_0=f_{\mathrm{RR}}(z'),
]

[
\hat r=a(p_s)\hat r_0+b(p_s),
]

where (p_s) is an unlabelled subject profile.

All profile-dependent corrections must be bounded and identity-initialised.

The experiment must test whether:

* static profiling improves the TCN;
* Profile-FiLM is more useful than direct profile concatenation;
* subject information is genuinely useful;
* one-step TTT improves over static profiling;
* a learned gate can suppress harmful adaptation;
* held-out errors are consistent with gain, bias, feature-mean or feature-scale shifts.

---

## 3. Files to create

Create the following new files:

```text
tcn_profile_ttt_experiment.py
tcn_profile_ttt_analysis.py
run_tcn_profile_ttt_experiment.sh
test_tcn_profile_ttt_experiment.py
```

Do not modify the existing validated TCN baseline unless explicitly required.

Prefer imports, wrappers and subclasses using:

```text
tcnet_experiment.py
dataloader.py
augmentations.py
evaluations.py
experiment_recorder.py
adaptation_utils.py
utils.py
config.py
```

The new implementation must not overwrite existing checkpoints or result directories.

---

## 4. Implementation phases

Implement the experiment in four phases.

### Phase 1: static profiling

Implement:

```text
T0 plain TCN
T1 TCN + profile-conditioned affine RR correction
T2 TCN + bounded Profile-FiLM
T3 TCN + Profile-FiLM + affine RR correction
```

Do not proceed to Phase 2 until Phase 1 passes:

* baseline reproduction tests;
* identity-initialisation tests;
* leakage checks;
* one-subject smoke tests;
* shuffled-profile controls.

### Phase 2: safe one-step TTT

Implement:

```text
T4 affine-only TTT
T5 FiLM + affine TTT
```

Do not proceed to Phase 3 until:

* only permitted parameters receive gradients;
* adaptation remains bounded;
* the TTT objective decreases;
* predictions do not collapse;
* target labels are absent from adaptation.

### Phase 3: profile-gated meta-trained TTT

Implement:

```text
T6 pseudo-held-out meta-trained profile-gated TTT
```

Do not proceed to Phase 4 until:

* pseudo-target episodes are leakage-safe;
* support and query windows are disjoint;
* query labels are used only during source meta-training;
* the learned gate is not trained on the real held-out target.

### Phase 4: optional mean alignment

Implement:

```text
T7 profile-gated feature-mean alignment
```

This is optional and must remain disabled by default.

---

## 5. Standard LOSO subjects

Use:

```text
S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29
```

Support overriding the list through:

```bash
--subjects
```

Default preliminary seeds:

```text
0 1 2
```

Support:

```bash
--seed
```

Do not change subject exclusions or LOSO partitioning silently.

---

## 6. Command-line interface

The primary script must support:

```bash
--mode
--subjects
--seed
--device
--data-dir
--model-dir
--out-dir
--batch-size
--num-workers
--epochs
--patience
--learning-rate
--weight-decay
--resume
```

Valid modes:

```text
t0_plain
t1_profile_affine
t2_profile_film
t3_profile_film_affine
t4_ttt_affine
t5_ttt_film_affine
t6_meta_gated_ttt
t7_profile_mean_alignment
all
```

Add a dry-run or smoke-test option:

```bash
--batch-limit
```

All configurable experimental values must be exposed through `argparse`.

---

## 7. Reproducibility

Set:

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

Seed:

* DataLoader generators;
* DataLoader workers;
* deterministic profile-window selection;
* bootstrap analysis;
* shuffled-profile permutations;
* pseudo-target episode selection.

Record:

```text
seed
subject list
held-out subject
command-line arguments
git commit
hostname
Python version
PyTorch version
CUDA version
trainable parameter count
```

Do not claim exact determinism where CUDA operations remain nondeterministic. Record deterministic settings in the manifest.

---

## 8. Base TCN wrapper

Wrap the existing TCN so that it returns:

```python
@dataclass
class TCNForward:
    prediction: torch.Tensor
    hidden_tokens: torch.Tensor
    pooled_hidden: torch.Tensor
```

Required shapes:

```text
prediction:     [batch]
hidden_tokens:  [batch, time, hidden_dim]
pooled_hidden:  [batch, hidden_dim]
```

The wrapper must use the existing TCN encoder and RR head.

Add a reproduction test confirming:

[
\hat r_{\mathrm{wrapped}}
\approx
\hat r_{\mathrm{original}}
]

for the same model state and input batch.

Use a strict numerical tolerance.

---

## 9. Subject-aware data interface

Every sample used by this experiment must expose:

```text
imu
rr
subject_id
window_index
condition
sequence_index
```

When the existing dataset does not return these fields, create a standalone wrapper.

Do not infer subject identity from the input signal.

The implementation must support:

* subject-balanced source sampling;
* deterministic chronological ordering;
* valid adjacent-window identification;
* grouping profile statistics by subject.

Profile statistics must never be calculated by averaging a mixed-subject batch without first separating subjects.

---

## 10. Subject profile construction

### 10.1 Profile windows

Build each subject profile using the first (K) unlabelled windows.

Support:

```bash
--profile-windows 16
--profile-windows 32
--profile-windows 64
```

Default:

```text
32
```

Use chronological windows when ordering is available.

Otherwise use deterministic seed-controlled selection.

For the real held-out subject:

1. use the first (K) windows as an unlabelled profile prefix;
2. do not access their RR labels while constructing the profile;
3. exclude them from the primary post-profile evaluation;
4. report an all-window secondary analysis separately;
5. record the selected indices.

### 10.2 Raw IMU profile features

Calculate per-channel:

```text
mean
standard deviation
median
median absolute deviation
root mean square
signal energy
dominant frequency
spectral entropy
low-frequency energy
mid-frequency energy
high-frequency energy
```

Also calculate:

```text
accelerometer norm statistics
gyroscope norm statistics
cross-axis correlations
temporal variability
```

### 10.3 Latent TCN profile features

Using the unconditioned TCN, calculate:

```text
mean pooled-hidden vector
standard deviation pooled-hidden vector
mean hidden-token norm
hidden temporal variability
prediction mean
prediction standard deviation
augmentation disagreement
```

Optional auxiliary spectral confidence features may include:

```text
spectral entropy
top-one/top-two margin
peak sharpness
```

### 10.4 Prohibited profile features

Do not include:

```text
true target RR
true target pressure
target prediction error
oracle gain
oracle bias
target labels
target validation metrics
```

Do not initially include differences between multiple predicted RR heads unless explicitly enabled as an ablation.

### 10.5 Profile normalisation

Fit profile normalisation using source-subject profiles only.

Apply the same fixed source-derived normaliser to:

* source training profiles;
* source validation profiles;
* held-out target profiles.

Save:

```text
profile_mean.npy
profile_std.npy
profile_schema.json
source_profile_window_indices.csv
target_profile_window_indices.csv
```

Add an assertion confirming that the held-out subject did not contribute to the normaliser.

---

## 11. T1: profile-conditioned affine RR correction

Implement:

```python
class ProfileAffineRR(nn.Module):
    ...
```

Predict bounded gain and bias:

[
a(p)=1+\epsilon_a\tanh(g_a(p)),
]

[
b(p)=\epsilon_b\tanh(g_b(p)).
]

Apply:

[
\hat r=a(p)\hat r_0+b(p).
]

Support:

```bash
--affine-gain-bound 0.01
--affine-gain-bound 0.03
--affine-gain-bound 0.05

--affine-bias-bound-bpm 0.5
--affine-bias-bound-bpm 1.0
--affine-bias-bound-bpm 2.0
```

Initialise:

```text
gain = 1
bias = 0
```

At initialisation:

[
\hat r=\hat r_0.
]

Add affine identity regularisation:

[
\mathcal L_{\mathrm{affine}}
============================

(a-1)^2+b^2.
]

Log per subject:

```text
affine_gain
affine_bias
prediction_delta_mean
prediction_delta_std
```

---

## 12. T2: bounded Profile-FiLM

Implement:

```python
class BoundedProfileFiLM(nn.Module):
    ...
```

Use:

[
\gamma(p)=1+\epsilon_\gamma\tanh(g_\gamma(p)),
]

[
\beta(p)=\epsilon_\beta\tanh(g_\beta(p)).
]

Apply:

[
z'=\gamma(p)\odot z+\beta(p).
]

Primary placement:

```text
after temporal pooling and before the RR head
```

Optional placement:

```text
after the final TCN block and before pooling
```

Support:

```bash
--film-scale 0.01
--film-scale 0.03
--film-scale 0.05

--film-placement pooled
--film-placement final_tokens
```

Initialise:

```text
gamma = 1
beta = 0
```

Add identity regularisation:

[
\mathcal L_{\mathrm{FiLM}}
==========================

|\gamma-1|_2^2+|\beta|_2^2.
]

Log:

```text
gamma_delta_norm
beta_norm
hidden_delta_rms
prediction_delta
```

---

## 13. T3: Profile-FiLM and affine correction

Combine T1 and T2:

[
z'=\gamma(p)\odot z+\beta(p),
]

[
\hat r_0=f_{\mathrm{RR}}(z'),
]

[
\hat r=a(p)\hat r_0+b(p).
]

The two corrections must remain independently bounded.

Add separate diagnostics for:

* FiLM-only prediction;
* affine-only correction;
* final combined prediction.

Do not add QKV correction or backbone adaptation.

---

## 14. Auxiliary respiratory-spectrum head

Implement an optional source-trained auxiliary head:

```python
class TCNRespiratorySpectrumHead(nn.Module):
    ...
```

The head predicts a probability distribution over respiratory frequencies:

```text
0.05 Hz to 0.75 Hz
```

Estimate RR using:

[
\hat r_{\mathrm{spec}}
======================

60\sum_k p_kf_k.
]

During source training, use a Gaussian target centred at:

[
f^*=\frac{r}{60}.
]

Define:

[
q_k
\propto
\exp
\left[
-\frac{(f_k-f^*)^2}{2\sigma^2}
\right].
]

Use peak-distribution loss:

[
\mathcal L_{\mathrm{peak}}
==========================

-\sum_kq_k\log p_k.
]

Support:

```bash
--use-spectrum-head
--lambda-spectrum
--peak-sigma-hz
```

Calculate detached spectral confidence using:

```text
top-one/top-two margin
spectral entropy
peak sharpness
```

The auxiliary head is an adaptation anchor, not the primary prediction endpoint.

---

## 15. TTT adaptation state

Define:

```python
@dataclass
class TTTState:
    delta_gamma: torch.Tensor
    delta_beta: torch.Tensor
    delta_gain: torch.Tensor
    delta_bias: torch.Tensor
```

Supported parameter groups:

```text
affine_only
film_only
film_affine
```

Default:

```text
film_affine
```

Do not permit TTT to update:

```text
TCN convolutional kernels
early TCN layers
profile encoder
source normalisation
auxiliary spectrum backbone
unrelated model parameters
```

Log all trainable adaptation parameter names and counts.

---

## 16. Unsupervised TTT objective

Use:

[
\mathcal L_{\mathrm{TTT}}
=========================

\lambda_{\mathrm{aug}}\mathcal L_{\mathrm{aug}}
+
\lambda_{\mathrm{temp}}\mathcal L_{\mathrm{temp}}
+
\lambda_{\mathrm{spec}}\mathcal L_{\mathrm{spec}}
+
\lambda_{\mathrm{anchor}}\mathcal L_{\mathrm{anchor}}.
]

### 16.1 Augmentation consistency

For an RR-preserving augmentation (\mathcal A):

[
\mathcal L_{\mathrm{aug}}
=========================

\operatorname{SmoothL1}
\left(
\hat r(x),
\hat r(\mathcal A(x))
\right).
]

Allowed augmentations:

```text
small Gaussian noise
small amplitude scaling
small 3D axis rotation
short temporal masking
small channel dropout
```

Do not use:

```text
time stretching
strong time warping
resampling that changes respiratory frequency
large temporal shifts
```

### 16.2 Temporal consistency

For valid adjacent windows:

[
\mathcal L_{\mathrm{temp}}
==========================

w_t
\operatorname{SmoothL1}
(\hat r_t,\hat r_{t-1}).
]

Only use temporal pairs that are:

* from the same subject;
* from the same continuous sequence;
* from a compatible condition;
* temporally adjacent or overlapping.

Keep the temporal loss weak to avoid constant-output collapse.

### 16.3 Spectral consistency

When the auxiliary spectrum head is active:

[
\mathcal L_{\mathrm{spec}}
==========================

c(x)
\operatorname{SmoothL1}
\left(
\hat r,
\hat r_{\mathrm{spec}}
\right),
]

where (c(x)\in[0,1]) is detached confidence.

Low-confidence spectral predictions must have little influence.

### 16.4 Anchor loss

Anchor adapted parameters to the profile-initialised state:

[
\mathcal L_{\mathrm{anchor}}
============================

|\phi-\phi_0|_2^2.
]

Also maintain FiLM and affine identity penalties.

---

## 17. T4: affine-only TTT

Adapt only:

```text
delta_gain
delta_bias
```

Use one TTT step by default.

The base TCN and FiLM parameters remain frozen.

This tests whether held-out error is primarily low-dimensional RR calibration.

---

## 18. T5: FiLM and affine TTT

Adapt:

```text
delta_gamma
delta_beta
delta_gain
delta_bias
```

Do not adapt the TCN backbone.

Support:

```bash
--ttt-steps 0
--ttt-steps 1
--ttt-steps 2
--ttt-lr
--ttt-parameter-group affine_only
--ttt-parameter-group film_only
--ttt-parameter-group film_affine
```

Primary configuration:

```text
ttt_steps = 1
ttt_mode = episodic
```

For episodic target adaptation:

1. initialise from the source/profile-conditioned state;
2. use the unlabelled target prefix;
3. perform one TTT update;
4. freeze the adapted correction;
5. evaluate the remaining target windows;
6. reset before the next held-out subject.

Cumulative online adaptation may be implemented only as a secondary ablation.

---

## 19. Profile-conditioned adaptation gate

Implement:

```python
class ProfileAdaptationGate(nn.Module):
    ...
```

Predict:

```text
adaptation gate
TTT learning-rate multiplier
maximum correction multiplier
optional mean-alignment strength
```

Use:

[
g_s=\sigma(G(p_s)).
]

Set:

[
\eta_s=\eta_{\max}g_s.
]

Apply:

[
\phi_{\mathrm{used}}
====================

\phi_0+g_s\Delta\phi_{\mathrm{TTT}}.
]

The gate must control the strength of one predefined adaptation procedure.

It must not select among unrelated algorithms.

Permitted gate features include:

```text
source-profile distance
latent mean shift
latent variance shift
augmentation disagreement
spectral confidence
prediction variance
motion energy
number of profile windows
```

Log:

```text
gate_value
effective_ttt_lr
profile_distance
adaptation_delta_norm
```

---

## 20. T6: pseudo-held-out meta-training

Use source subjects to teach the profile and TTT mechanism how to adapt safely.

For each meta-training episode:

1. select one source subject as a pseudo-target;
2. use its first (K) windows as an unlabelled support set;
3. use separate windows as a labelled query set;
4. construct its subject profile from support windows;
5. initialise bounded FiLM and affine corrections;
6. perform one unsupervised TTT update on the support set;
7. evaluate RR loss on the query set;
8. update the profile encoder, correction initialisers and adaptation gate.

Formally:

[
\phi_s^0=G_\psi(p_s),
]

[
\phi_s^1
========

## \phi_s^0

\eta(p_s)
\nabla_\phi
\mathcal L_{\mathrm{TTT}}
(D_s^{\mathrm{support}}),
]

[
\mathcal L_{\mathrm{meta}}
==========================

\mathcal L_{\mathrm{RR}}
(D_s^{\mathrm{query}};\theta,\phi_s^1).
]

Support:

```text
first_order
full_second_order
```

Default:

```text
first_order
```

Initially freeze the base TCN during meta-training.

Update only:

```text
profile encoder
FiLM initialiser
affine initialiser
adaptation gate
optional spectrum head
```

The true LOSO target subject must never enter meta-training.

---

## 21. T7: profile-gated feature-mean alignment

Estimate:

[
\mu_{\mathrm{source}}
]

from source training representations and:

[
\mu_{\mathrm{target}}
]

from the unlabelled target profile prefix.

Apply:

[
z'
==

## z

\alpha(p)
(\mu_{\mathrm{target}}-\mu_{\mathrm{source}}).
]

Use:

[
0\leq\alpha(p)\leq\alpha_{\max}.
]

Default:

```text
alpha_max = 0.5
```

Initialise (\alpha) near zero.

Do not universally apply fixed values such as 0.75 or 1.0.

Treat mean alignment as a bounded part of the profile correction.

---

## 22. Required controls

### Static controls

Run:

```text
C0 plain TCN
C1 zero-profile conditioning
C2 real-profile affine
C3 shuffled-profile affine
C4 real Profile-FiLM
C5 shuffled Profile-FiLM
C6 real Profile-FiLM + affine
C7 shuffled Profile-FiLM + affine
```

### TTT controls

Run:

```text
C8 TTT without profile
C9 static profile without TTT
C10 profile-conditioned TTT
C11 profile-conditioned TTT with gate fixed to 1
C12 profile-conditioned TTT with gate fixed to 0
C13 shuffled-profile TTT
```

### Profile controls

Include:

```text
zero profile
mean source profile
random subject profile
dimension-shuffled profile
matched real profile
```

### Oracle diagnostic

Implement labelled target affine calibration as:

```text
oracle_target_affine_invalid_for_deployment
```

It must never appear in the deployable ranking.

Its purpose is to determine the maximum plausible benefit from gain and bias correction.

---

## 23. Training losses

For static profile training:

[
\mathcal L
==========

\mathcal L_{\mathrm{RR}}
+
\lambda_{\mathrm{FiLM}}\mathcal L_{\mathrm{FiLM}}
+
\lambda_{\mathrm{affine}}\mathcal L_{\mathrm{affine}}
+
\lambda_{\mathrm{spectrum}}\mathcal L_{\mathrm{peak}}.
]

Use Smooth L1 for RR prediction.

Support a correction-bound warmup where:

```text
FiLM scale
gain bound
bias bound
```

increase gradually from zero to their configured maximum.

Do not allow large profile corrections at the beginning of training.

---

## 24. Default configuration

Use conservative defaults:

```text
profile_windows              = 32
profile_hidden_dim           = 32
profile_dropout              = 0.2

film_scale                   = 0.03
affine_gain_bound            = 0.03
affine_bias_bound_bpm        = 1.0

ttt_steps                    = 1
ttt_lr                       = 1e-4
ttt_mode                     = episodic
ttt_parameter_group          = film_affine

lambda_aug                   = 1.0
lambda_temp                  = 0.05
lambda_spec                  = 0.25
lambda_anchor                = 1.0
lambda_film_identity         = 0.1
lambda_affine_identity       = 0.1

meta_mode                    = first_order
mean_alignment_alpha_max     = 0.5
```

All values must be configurable.

---

## 25. Leakage protection

Add runtime assertions confirming:

```text
held-out subject absent from source training
held-out subject absent from source validation
held-out subject absent from profile-normaliser fitting
target RR absent from target profile construction
target RR absent from TTT loss
target RR absent from gate inputs
target RR absent from profile-window selection
target RR absent from hyperparameter selection
target RR absent from checkpoint selection
```

For pseudo-held-out episodes, also assert:

```text
support and query windows are disjoint
query labels are not used in the support adaptation loss
real held-out subject is absent from meta-training
```

Terminate the run when any assertion fails.

Save:

```text
leakage_audit.json
```

---

## 26. Required outputs

Every run must save:

```text
config.json
manifest.json
implementation_audit.json
leakage_audit.json
training_history.csv
```

Aggregate outputs:

```text
summary.csv
per_seed_summary.csv
subject_rows.csv
window_predictions.csv
profile_vectors.csv
profile_diagnostics.csv
ttt_diagnostics.csv
gate_diagnostics.csv
control_comparisons.csv
oracle_calibration.csv
```

Required `subject_rows.csv` columns:

```text
seed
subject
method
n_windows
mae
rmse
rr_corr
bias
median_ae
p95_ae

profile_windows
profile_norm
profile_distance

film_gamma_delta_norm
film_beta_norm
hidden_delta_rms

affine_gain
affine_bias

gate_value
effective_ttt_lr

ttt_loss_pre
ttt_loss_post
ttt_delta_norm

prediction_delta_mean
prediction_delta_std
```

Required `window_predictions.csv` columns:

```text
seed
subject
method
window_index
condition
sequence_index
rr_true
rr_pred
rr_pred_plain
prediction_delta
is_profile_window
```

---

## 27. Statistical analysis

Implement analysis in:

```text
tcn_profile_ttt_analysis.py
```

The held-out subject is the unit of inference.

For every method:

1. calculate metrics within subject and seed;
2. calculate paired method differences within subject and seed;
3. aggregate only after subject-level metrics are available.

Report:

```text
mean
standard deviation
median
interquartile range
seed-wise mean
seed-wise standard deviation
subject win count
subject loss count
largest improvement
largest degradation
paired subject-bootstrap 95% confidence interval
```

Use at least 10,000 deterministic bootstrap resamples.

Primary comparisons:

```text
T1 versus T0
T2 versus T0
T3 versus T0
T4 versus T0
T5 versus T3
T6 versus T5
real profile versus shuffled profile
gated TTT versus fixed-strength TTT
T7 versus T6
```

Do not treat individual windows as independent samples for statistical inference.

---

## 28. Unit tests

### 28.1 Baseline reproduction

Verify that wrapped TCN outputs match the original TCN.

### 28.2 Identity initialisation

At initialisation:

```text
FiLM gamma = 1
FiLM beta = 0
affine gain = 1
affine bias = 0
gate produces no unintended correction
conditioned prediction equals plain prediction
```

### 28.3 Bounded output tests

Verify:

```text
FiLM scale remains within its configured bound
affine gain remains within its bound
affine bias remains within its bound
TTT correction remains bounded
mean-alignment alpha remains within [0, alpha_max]
```

### 28.4 Gradient tests

Verify:

```text
profile training updates the profile encoder
FiLM loss updates FiLM parameters
affine loss updates affine parameters
TTT gradients update only allowed adaptation parameters
base TCN remains frozen during TTT
meta-query loss updates the adaptation gate
```

### 28.5 Data tests

Verify:

```text
subject metadata is correct
profiles contain one subject only
temporal pairs are valid
support and query sets are disjoint
profile windows are deterministic
```

### 28.6 Leakage tests

Verify that target labels cannot enter:

```text
profile builder
profile normaliser
TTT objective
adaptation gate
hyperparameter selector
checkpoint selector
```

### 28.7 Numerical tests

Verify:

```text
no NaNs
no infinite values
prediction count matches label count
same seed gives identical profile selection
same seed reproduces evaluation outputs within tolerance
```

---

## 29. Smoke-test procedure

Run:

```bash
python tcn_profile_ttt_experiment.py \
  --mode all \
  --subjects S12 \
  --seed 0 \
  --profile-windows 8 \
  --epochs 2 \
  --meta-epochs 2 \
  --batch-limit 4 \
  --device cuda:0 \
  --out-dir /tmp/tcn_profile_ttt_smoke
```

The smoke test must:

* complete without target leakage;
* run all implemented phases;
* produce every expected output file;
* verify identity initialisation;
* verify bounded adaptation;
* verify that only permitted parameters change.

Do not launch full LOSO runs automatically.

---

## 30. Full experiment runner

Create:

```text
run_tcn_profile_ttt_experiment.sh
```

Support:

```bash
SEEDS="${SEEDS:-0 1 2}"

SUBJECTS="${SUBJECTS:-S12 S13 S14 S15 S16 S18 S19 S20 S22 S23 S24 S25 S27 S28 S29}"

CUDA_DEVICES="${CUDA_DEVICES:-0 1}"

MODES="${MODES:-t0_plain t1_profile_affine t2_profile_film t3_profile_film_affine t4_ttt_affine t5_ttt_film_affine t6_meta_gated_ttt}"

OUT_ROOT="${OUT_ROOT:-/projects/BLVMob/imu-rr-seated/results/tcn_profile_ttt}"
```

The runner must:

* create a unique timestamped run directory;
* distribute jobs across available GPUs;
* write separate logs per subject, seed and mode;
* preserve nonzero exit codes;
* support resumption;
* avoid overwriting completed outputs;
* run aggregate analysis after all requested jobs complete;
* save environment and commit information.

---

## 31. Acceptance criteria by phase

### Phase 1 is complete when

* T0 reproduces the plain TCN;
* T1–T3 run for one smoke-test subject;
* identity initialisation passes;
* real, zero and shuffled-profile controls run;
* all leakage tests pass;
* expected result files are produced.

### Phase 2 is complete when

* T4 and T5 perform one-step adaptation;
* only permitted parameters change;
* TTT loss is recorded before and after adaptation;
* adaptation deltas remain bounded;
* predictions do not collapse to a constant;
* target labels are not used.

### Phase 3 is complete when

* pseudo-held-out episodes work;
* support and query sets are disjoint;
* the adaptation gate receives gradients from query loss;
* the real held-out subject remains excluded;
* first-order meta-training completes a smoke test.

### Phase 4 is complete when

* profile-gated mean alignment is implemented;
* alpha is bounded;
* fixed-alignment controls are available;
* alignment remains disabled by default.

---

## 32. Decision rules

Do not claim that profiling is beneficial unless:

* real profiles outperform zero profiles;
* real profiles outperform shuffled profiles;
* mean MAE improves over the plain TCN;
* a majority of held-out subjects improve;
* seed variance does not materially increase.

Do not claim that Profile-FiLM is beneficial when profile concatenation or affine correction gives the same result with less complexity.

Do not claim that TTT is beneficial unless:

* TTT improves over the corresponding static profile model;
* adaptation loss decreases;
* prediction variance does not collapse;
* correction norms remain small;
* catastrophic subject degradation does not increase;
* results are consistent across seeds.

Do not claim that the learned gate is useful unless:

* gated TTT outperforms fixed gate (g=1);
* gated TTT outperforms TTT without profile information;
* gate values correspond to measurable adaptation reliability.

Do not claim that mean alignment is useful unless profile-gated alignment improves over both:

```text
no alignment
fixed alignment
```

The oracle target-affine result must remain clearly labelled invalid for deployment.

---

## 33. Final report

At completion, print:

1. best method by mean MAE;
2. best method by seed stability;
3. subject win count versus plain TCN;
4. real-profile versus shuffled-profile performance;
5. static profiling versus TTT performance;
6. gated versus fixed-strength TTT performance;
7. largest subject improvement;
8. largest subject degradation;
9. oracle affine-calibration ceiling;
10. average FiLM correction norm;
11. average affine gain and bias;
12. average TTT correction norm;
13. whether the dominant shift appears to be:

    * feature mean;
    * feature scale;
    * RR gain;
    * RR bias;
    * or not low-dimensional;
14. recommended next experiment.

The final recommended method must improve mean performance without materially degrading the stability of the plain TCN.

