# Architecture training dynamics study

This is an opt-in v3 study. It does not change v1/v2 checkpoints, configs, or
plots. All architecture training is blocked until a fully matched audit has a
verified benign and harmful point for both `rho_x=0` and `rho_x=0.9`, at the
configured dimension and context length.

## Architecture definitions

| Family | Label / shapes | Scientific role |
|---|---|---|
| `joint_scale` | Tiny `(64,3,2)`, Small `(128,6,4)`, Standard `(256,12,8)` | Pilot only: width, layers, heads, and parameters all change together. Each has head dimension 32. |
| `head_sweep_fixed_width` | `(256,12,H)`, `H=1,2,4,8,16` | Main isolated head-count comparison; width/depth and parameter count are fixed. |
| `depth_sweep_fixed_width` | `(256,L,8)`, `L=1,3,6,12,24` | Depth comparison; parameter count changes and needs the matched-depth control. |
| `parameter_matched_depth` | closest instantiated widths at `L=3,6,12,24` | Controls parameter count against Standard. |
| `width_only` | `(W,6,4)`, `W=64,128,256` | Optional; never launched by default. |

`plan` writes exact instantiated parameter counts to
`results/arch_dynamics/architecture_counts.json`. It also records
`parameter_count_ratio_to_standard` and `relative_parameter_error`.

## Gate and safe planning

First produce an audit from completed Transformer evaluation bundles. The
audit must contain fully matched rows with `d`, `k`, `rho_x`, `rho_e`, and
SNR; rerun it after this revision if an old audit lacks these fields.

```bash
python src/run_bo_suite.py scientific-audit \
  --input results/bo_matrix/evaluation/result_manifest.json

python src/run_bo_suite.py plan --group arch_stageA_joint_scale \
  --architecture-audit results/bo_matrix/scientific_audit/scientific_audit.json
```

If the evidence is absent, it writes a blocked plan and prints
`ARCHITECTURE_BO_BLOCKED`; no architecture training begins.

For pipeline debugging only, a pair of SNRs can be supplied. They are marked
`exploratory_0` and `exploratory_1`, never benign/harmful.

```bash
python src/run_bo_suite.py plan --group arch_stageA_joint_scale \
  --debug-exploratory-snrs 0.8 3.2
```

## Ordered launch procedure

Use one process. All shapes receive the same stateless synthetic batch for a
given `(train_seed, step)`, batch size, optimizer, LR, total steps, matched
SNR, matched dependence, and diagnostic schedule.

```bash
# Stage A only: 3 joint-scale shapes × 2 rho × harmful/benign = 12 runs.
python src/run_bo_suite.py train-matrix --group arch_stageA_joint_scale \
  --device cuda:0 --precision bfloat16 --resume

# Plot stored diagnostics and events without retraining.
python src/run_bo_suite.py analyze-architecture --group arch_stageA_joint_scale
```

Inspect Stage A manually. To unlock Stage B/C, create a small review file:

```json
{"inspected": true, "stage_a_signature": "<value in Stage-A plan>"}
```

Then launch one controlled family at a time:

```bash
python src/run_bo_suite.py train-matrix --group arch_stageB_head_sweep \
  --device cuda:0 --precision bfloat16 --resume \
  --stage-a-inspection results/arch_dynamics/stage_a_review.json

python src/run_bo_suite.py train-matrix --group arch_stageC_depth_sweep \
  --device cuda:0 --precision bfloat16 --resume \
  --stage-a-inspection results/arch_dynamics/stage_a_review.json
```

The joint/head/depth pilot deduplicates Standard by training identity: there
are 11 unique shapes and at most `11 × 2 rho × 2 audit conditions × 1 seed =
44` runs. Stage A is never bypassed automatically.

## Diagnostics and interpretation

Diagnostics occur at `0, 50, 100, 200, 500, 1k, 2k, 5k, 10k, 20k, 50k,
100k, 200k, 350k, final`; edit `src/conf/arch_dynamics.json` to change the
schedule. At each step, it records train loss, direct/linear clean and fitting
metrics, BO flags, activation-response summaries, attention novelty/recency,
head diversity, and independent layer probes. It stores aggregated moments,
small `H×H` matrices, and bootstrap uncertainty, never full attentions.

The empirical event detector requires a configurable stable window of three
observed diagnostics:

- `t_generalize`: clean generalization ratio is at most `tau_gen`.
- `t_direct_fit`: duplicate-fit ratio is at most `tau_fit`.
- `t_linear_fit`: linear-fit ratio is at most `tau_fit` and held-out probe R²
  is at least `tau_probe`.
- `t_direct_BO` / `t_linear_BO`: direct or linear BO frequency reaches the
  configured event frequency.
- `BO_persistence_fraction`: fraction of later diagnostic observations that
  remain BO after onset.

The phase trajectory uses `(duplicate_fit_ratio, clean_gen_ratio)` and its
linear counterpart. The green rectangle is the configured BO region. This is
the primary figure for how a model enters, misses, or leaves that region.

Activation response uses three paired prompts with identical `x`, `w`, and
`epsilon`: `x-only`, `clean`, and `noisy`. At each head/layer it reports
`||o_clean-o_xonly||`, `||o_noisy-o_clean||`, and their ratio. These are
evaluation diagnostics rather than literal raw `W_V` projections or causal
importance measures.

Outputs are under `results/arch_dynamics/`: plans, exact counts, per-run
resumable state/diagnostics, and analysis figures/events/replication
suggestions.
