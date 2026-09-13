# Running the experiment suite on a server

The code does not depend on a notebook or a fixed checkout directory. Create
the repository's pinned environment, activate it, and run commands from the
repository root:

```bash
conda env create -f environment.yml
conda activate in-context-learning
python -m unittest discover -s tests -p "test_*.py" -v
python src/run_bo_suite.py smoke
```

The training launcher invokes the active interpreter and resolves paths from
the repository location, regardless of the shell's current directory. Inspect
the batch first, then start it:

```bash
python src/run_bo_suite.py train --preset all --dry-run
python -u src/run_bo_suite.py train --preset all
```

The `all` training preset contains IID, stationary feature dependence,
stationary noise dependence, both types of dependence, and the two change-point
orders. Smaller batches are available as `--preset stationary` and
`--preset change-point`. Add derived architecture/seed YAML files with repeated
`--config path/to/config.yaml` arguments. Use `--preset none` when a batch
should contain only those explicitly supplied files.

Each training config creates a UUID directory under its configured `out_dir`.
At successful completion, the launcher records every new or updated checkpoint
directory in `results/bo_suite/trained_checkpoints.json`.

Use the chosen checkpoint directories for the coarse evaluation batch:

```bash
python -u src/run_bo_suite.py evaluate \
  --preset all \
  --device cuda \
  --checkpoint-manifest results/bo_suite/trained_checkpoints.json
```

Use repeated `--run-dir` arguments when you want a selected set of checkpoints,
or repeated `--checkpoint-manifest` arguments to combine training batches. With
neither option, evaluation runs only OLS, ridge, and/or oracle GLS as listed by
each sweep YAML. Output JSON and figures are written below `results/bo_suite`.
Use `--output-root /mounted/path/results` when results must survive cleanup of
the checkout.

For an unattended Linux session, redirect the same launcher through the
server's usual job mechanism. A plain shell example is:

```bash
mkdir -p results/logs
nohup python -u src/run_bo_suite.py train --preset all \
  > results/logs/train-all.log 2>&1 &
```

For Slurm or another scheduler, place the launcher command in the scheduler
script after activating the environment. GPU allocation and wall-time remain
scheduler settings; the experiment code itself requires no scheduler-specific
paths.
