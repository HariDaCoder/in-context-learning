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

1. Run `bo_sweep.yaml` to validate OLS/ridge/GLS and plotting cheaply.
2. Run `bo_sweep_phase.yaml` with one fixed Transformer checkpoint. This is a
   coarse stationary scan and skips linear probes.
3. Put only observed transition neighborhoods into `bo_sweep_boundary.yaml`.
   Enable probes and use at least five evaluation seeds there.
4. Isolate dependent label noise with `bo_sweep_noise_markov.yaml`.
5. Compare `bo_sweep_change_forward.yaml` with its reverse file. Their 40
   transition coefficients are exact reversals and their correlation matrices
   have equal effective rank; an outcome difference therefore cannot be
   explained by static `k_eff` alone.
6. Sweep architecture only near the boundary. Report fixed-width heads,
   fixed-head-dimension heads, and fixed-width depth as separate protocols.

Train matched-distribution models with multiple training seeds. Do not pool
those checkpoints as if they were evaluation seeds. Also keep a separate OOD
protocol using one clean-IID Garg checkpoint across all test conditions; this
answers a different robustness question.

Critical SNR output consists of sampled equalities and adjacent observed
brackets where BO frequency crosses the chosen probability. Missing cells are
never bridged, and the plotter does not assume monotonicity or extrapolate a
unique threshold.

## Batch execution

All commands below run from the repository root and use the active Python
environment. Preview every command without starting work:

```bash
python src/run_bo_suite.py train --preset all --dry-run
python src/run_bo_suite.py evaluate --preset all --device cuda --dry-run
```

Run the six provided matched-distribution training configurations in sequence:

```bash
python -u src/run_bo_suite.py train --preset all
```

The launcher writes new or updated UUID checkpoint directories to
`results/bo_suite/trained_checkpoints.json`. Run those checkpoints through the
four coarse evaluation protocols and generate their plots with:

```bash
python -u src/run_bo_suite.py evaluate \
  --preset all \
  --device cuda \
  --checkpoint-manifest results/bo_suite/trained_checkpoints.json
```

You can also repeat `--run-dir` for selected checkpoints or repeat
`--checkpoint-manifest` to combine batches. `--include-boundary` is
deliberately opt-in: edit `bo_sweep_boundary.yaml` around the crossings found
by the coarse pass before enabling that more expensive probe-based evaluation.
