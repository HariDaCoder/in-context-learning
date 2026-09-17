# Running the experiment suite on a server

The launchers resolve paths from the repository root and contain no host or
accelerator assignment policy. Use the existing environment when it already
contains the project dependencies. For a new current-CUDA environment:

```bash
conda env create -f environment-gpu.yml
conda activate in-context-learning-gpu
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

Authenticate W&B interactively so the API key is stored by W&B rather than in
the repository or shell history:

```bash
wandb login
wandb status
```

The public entity is configured in `src/conf/wandb.yaml` and the schema
default. No API key belongs in YAML, source code, manifests, or job scripts.

Validate a checkout before starting long work:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
python src/run_bo_suite.py smoke --output-root results/smoke
python src/run_bo_suite.py train --preset all --dry-run
python src/run_bo_suite.py plan --group stage0 --device cuda:0
python src/run_bo_suite.py train-matrix --group stage0 \
  --device cuda:0 --max-concurrent 1 --resume --dry-run
```

The lock exclusivity test is intentionally skipped. CUDA generator statistics
are skipped when the test process has no CUDA device. All other tests must
pass.

## Matrix lifecycle

Every group uses the same four-step lifecycle:

```bash
python src/run_bo_suite.py plan --group GROUP --device cuda:0
python src/run_bo_suite.py train-matrix --group GROUP \
  --device cuda:0 --max-concurrent N --resume --dry-run
python -u src/run_bo_suite.py train-matrix --group GROUP \
  --device cuda:0 --max-concurrent N --resume
python -u src/run_bo_suite.py evaluate-matrix --group GROUP \
  --protocol matched --device cuda:0 --max-concurrent N
```

`--resume` is the default and is shown explicitly in unattended commands.
Completed experiment IDs are skipped. Interrupted runs retain `state.pt` and
resume from it. Each successful run also writes `final.pt` and
`completed.json`.

Before choosing `N`, benchmark isolated short runs without changing the
scientific batch size:

```bash
python -u src/run_bo_suite.py benchmark --group stage0 --device cuda:0 \
  --max-concurrent 1 2 4 --steps 2000
```

The benchmark summary reports wall time, aggregate steps per second, and
per-process peak CUDA allocation. Use the fastest stable concurrency for later
groups. A successful benchmark also writes a group recommendation; subsequent
`plan` and `train-matrix` calls use it when `--max-concurrent` is omitted.

## Unattended processes

Create a log directory and redirect one launcher process. The launcher creates
and supervises at most `N` independent training processes:

```bash
mkdir -p results/launcher_logs
nohup python -u src/run_bo_suite.py train-matrix --group stage0 \
  --device cuda:0 --max-concurrent 1 --resume \
  > results/launcher_logs/stage0.log 2>&1 &
echo $! > results/launcher_logs/stage0.pid
tail -f results/launcher_logs/stage0.log
```

Rerunning the same command is the resume procedure. Do not start two different
launcher commands that intentionally target the same experiment IDs.

## Evaluation and boundary refinement

Matched and shift are separate dependence protocols. A test-SNR sweep does not
turn a dependence-matched row into a shift row.

```bash
python -u src/run_bo_suite.py evaluate-matrix --group canonical \
  --protocol matched --device cuda:0 --max-concurrent 2
python -u src/run_bo_suite.py evaluate-matrix --group canonical \
  --protocol shift --device cuda:0 --max-concurrent 2
```

Each plot bundle writes `boundary_suggestions.json`. Repeat
`--boundary-suggestions` to take the union across selected coarse bundles:

```bash
python -u src/run_bo_suite.py evaluate-matrix --group architecture \
  --protocol matched --device cuda:0 --max-concurrent 2 \
  --boundary-suggestions results/bo_matrix/evaluation/plots/BUNDLE_A/boundary_suggestions.json \
  --boundary-suggestions results/bo_matrix/evaluation/plots/BUNDLE_B/boundary_suggestions.json
```

## Output layout

- `models/bo_matrix/<experiment_id>/state.pt`: latest resumable checkpoint.
- `models/bo_matrix/<experiment_id>/final.pt`: final model-only checkpoint.
- `models/bo_matrix/<experiment_id>/completed.json`: throughput/runtime record.
- `results/bo_matrix/manifests`: group JSON/CSV manifests.
- `results/bo_matrix/state`, `locks`, `logs`, `failures`: launcher state.
- `results/bo_matrix/summaries`: plan and training summaries.
- `results/bo_matrix/evaluation`: configs, result rows, plots, and manifests.
- `results/bo_matrix/benchmarks`: isolated concurrency benchmarks.
- `results/bo_matrix/parameter_match`: exact-count architecture reports.

The `models` and `results` trees are intentionally ignored by Git. Preserve
them on durable storage or copy them before deleting a checkout.
