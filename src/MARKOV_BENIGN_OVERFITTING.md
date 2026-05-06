# Markov Benign Overfitting Experiments

This experiment file is a YAML-driven runner with resume support. It now supports two modes:

- `mode: dynamics` for train/test loss curves over optimization steps.
- `mode: sweep` for the original parameter sweeps.

## Files

- Runner: `src/markov_benign_overfitting_experiments.py`
- Config: `src/conf/markov_benign_overfitting_small.yaml`
- Output directory: `models/markov_benign_overfitting/`

## What it runs

The runner executes four blocks:

1. Stationary Markov sweep over `scale`.
2. Nonstationary cases: `stationary`, `drift`, `switch`.
3. Input-noise sweep over `sigma_x`.
4. Label-noise sweep over `sigma_y`.

Each block saves intermediate progress so the script can resume after interruption.

For benign overfitting, use `mode: dynamics`. That mode trains a linear predictor on a fixed Markov dataset and logs both train and test MSE over steps, which is the right view for diagnosing overfitting.

## Resume behavior

The runner saves two files inside the output directory:

- `markov_benign_overfitting_state.json`
- `markov_benign_overfitting_results.json`

If the run stops halfway, rerun the same command with `--resume`. Completed sweep items are skipped.

## Run

From the repo root:

```bash
python src/markov_benign_overfitting_experiments.py --config src/conf/markov_benign_overfitting_small.yaml --resume
```

If you want to force a device:

```bash
python src/markov_benign_overfitting_experiments.py --config src/conf/markov_benign_overfitting_small.yaml --resume --device cuda
```

To run the old sweep mode instead, use `src/conf/markov_benign_overfitting.yaml`.

## Outputs

- `markov_benign_overfitting_results.json`: summary or sweep metrics.
- `markov_benign_overfitting_state.json`: resume checkpoint.
- `markov_benign_overfitting_results.png`: train/test loss plot in dynamics mode.

## Notes

- `device: auto` picks CUDA if available, otherwise CPU.
- The script uses atomic JSON writes so a keyboard interrupt should not corrupt the resume state.