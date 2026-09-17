# Experimental contract

## Data and SNR

For each prompt, sample one task and keep it fixed:

```
w ~ N(0, I_d / d)
y_t = x_t^T w + epsilon_t
```

The implementation uses an equivalent scaled weight tensor. Averaged over
tasks and standard-Gaussian feature marginals, signal power is one. SNR is an
amplitude ratio, so `noise_std = 1 / SNR` and power SNR is `SNR^2`.

Feature dependence is

```
x_t = rho_t x_{t-1} + sqrt(1-rho_t^2) z_t,
```

which preserves `x_t ~ N(0, I_d)` for every rho and across a change point.
Label noise uses the same construction independently of features. The clean
test query is sampled independently from `N(0, I_d)`, rather than as the next
state of the context chain.

## Outcomes

The primary behavioral event for a fixed noisy context is:

```
duplicate-context MSE <= fit_threshold
and clean independent-query MSE <= gen_threshold.
```

This is stored as `bo_candidate`. It permits a Transformer to retrieve labels
at observed points while using another rule off-context. It is empirical and
threshold dependent, not an asymptotic theorem.

`linear_bo_candidate` is a mechanism diagnostic. Predictions on independent
Gaussian probes define an implied through-origin linear estimator. It must fit
the noisy context, generalize on clean queries, and achieve the requested R2
on a disjoint holdout probe set. Low-rank probes or poor held-out linearity are
reported as `linear_indeterminate`, rather than forced into a BO class.

For OLS, ridge, and oracle GLS, exact isotropic clean risk is
`||w_hat - w||^2`. GLS knows the label-noise correlation matrix but never the
true task weight. Ridge alpha is fixed in the YAML and is not tuned on test
labels.

## Hypothesis tested, not assumed

For context correlation matrix `R`, the diagnostic effective context size is

```
k_eff = trace(R)^2 / trace(R^2).
```

For stationary AR(1), its large-k approximation is

```
k_eff ~= k (1-rho^2) / (1+rho^2).
```

The proposed phase law `k_eff * SNR^2 ~= C` is a hypothesis. Results include
raw `k`, `k/d`, exact finite-sample `k_eff`, `k_eff/d`, and
`k_eff*SNR^2`. Plots show both the raw grid and the collapse view; the code does
not fit or enforce the law.

## Staged experiment order

1. Run `smoke` to validate OLS/ridge/oracle-GLS and plotting cheaply.
2. Train and evaluate `stage0`: 9 standard models from three feature
   correlations and three training seeds. Keep `train_rho_x == test_rho_x`.
3. Inspect fitting/generalization and BO-frequency output before starting the
   larger groups.
4. Train `canonical`: 18 standard models from six regimes and three seeds.
   Evaluate both dependence-matched and dependence-shift protocols. The two
   change-point schedules have 79 exactly reversed transition coefficients and
   equal correlation-matrix effective rank.
5. Train `matched_rho`: 35 standard models from seven feature correlations and
   five seeds. This is the main matched-dependence phase experiment.
6. Train `dimension`: 36 standard models from four dimensions, three feature
   correlations, and three seeds. Contexts are derived from `k/d`; the largest
   checkpoint supports `d=80, k=320`.
7. Use the coarse matched results to write `boundary_suggestions.json`. Every
   observed crossing is retained; multiple crossings are never collapsed to a
   single threshold. With no crossing, the closest sampled BO probability
   drives a grid expansion or local refinement.
8. Train `architecture`: 72 unique checkpoints after semantic shape
   de-duplication. Evaluate these at the automatically suggested SNR points.
9. Instantiate the parameter-matched search, then train its 24 manifest rows.
   The standard depth-12 rows reuse existing checkpoints when their semantic
   IDs match.
10. Run aggregate mechanism hooks only on selected checkpoints and conditions.
    The output contains moments and small head Gram matrices, never full stored
    attention tensors.

Train dependence-matched models with multiple training seeds. Do not pool
those checkpoints as if they were evaluation seeds. `matched` means that the
feature and label-noise dependence schedules match; test SNR is deliberately
swept around the fixed training SNR and does not relabel the row as `shift`.
The `shift` protocol changes dependence and is an OOD robustness result.

Critical SNR output consists of sampled equalities and adjacent observed
brackets where BO frequency crosses the chosen probability. Missing cells are
never bridged, and the plotter does not assume monotonicity or extrapolate a
unique threshold.

## Batch execution

All commands below run from the repository root and use the active Python
environment. The original preset commands remain available:

```bash
python src/run_bo_suite.py train --preset all --dry-run
python src/run_bo_suite.py evaluate --preset all --device cuda --dry-run
```

The matrix groups and exact training counts are:

| Group | Experiments | Purpose |
|---|---:|---|
| `canonical` | 18 | six canonical regimes, three seeds |
| `stage0` | 9 | matched-dependence validation |
| `matched_rho` | 35 | main seven-rho phase experiment |
| `dimension` | 36 | four dimensions by three rho values by three seeds |
| `architecture` | 72 | three architecture families, two rho values, three seeds, de-duplicated by shape |

Planning writes standalone configs plus JSON/CSV manifests under
`results/bo_matrix`:

```bash
python src/run_bo_suite.py plan --group canonical
python src/run_bo_suite.py train-matrix --group canonical --device cuda:0 --dry-run
python -u src/run_bo_suite.py train-matrix --group canonical --device cuda:0 --resume
```

The same launch command is also the resume command. Completed semantic IDs are
skipped, interrupted checkpoints are resumed, and concurrent launchers contend
on per-experiment locks. Checkpoints are under `models/bo_matrix/<experiment_id>`;
plans, state, logs, failures, and manifests are under `results/bo_matrix`.

Dense matched evaluation uses eight amplitude-SNR values, eight `k/d` values,
and ten evaluation seeds by default:

```bash
python src/run_bo_suite.py evaluate-matrix \
  --group matched_rho --protocol matched --device cuda:0 --dry-run
python -u src/run_bo_suite.py evaluate-matrix \
  --group matched_rho --protocol matched --device cuda:0 --max-concurrent 2
```

Canonical OOD evaluation must remain a separate protocol:

```bash
python -u src/run_bo_suite.py evaluate-matrix \
  --group canonical --protocol shift --device cuda:0 --max-concurrent 2
```

Architecture boundary and parameter-matched workflows are:

```bash
python -u src/run_bo_suite.py evaluate-matrix \
  --group architecture --protocol matched --device cuda:0 \
  --boundary-suggestions results/bo_matrix/evaluation/plots/<bundle>/boundary_suggestions.json

python src/run_bo_suite.py parameter-match
PARAM_REPORT=results/bo_matrix/parameter_match/parameter_match_pmatch_e641b37fc64d8ffd26f8.json
python src/run_bo_suite.py train-matrix \
  --matrix-manifest "$PARAM_REPORT" \
  --device cuda:0 --dry-run
python -u src/run_bo_suite.py train-matrix \
  --matrix-manifest "$PARAM_REPORT" \
  --device cuda:0 --resume
```

Before choosing experiment concurrency, run the isolated short benchmark. It
does not change the scientific batch size:

```bash
python -u src/run_bo_suite.py benchmark --group stage0 --device cuda:0 \
  --max-concurrent 1 2 4 --steps 2000
```
