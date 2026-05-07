"""Launcher for the Markov benign overfitting experiments using train.py.

This script generates per-run configs from the small-frame training template and
runs each experiment through the existing training pipeline, so you keep the
standard progress bar, checkpointing, and W&B logging.
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "train.py"
SMALL_TEMPLATE = PROJECT_ROOT / "src" / "conf" / "benign_harmful_dynamics_small.yaml"
SPEC_PATH = PROJECT_ROOT / "src" / "conf" / "markov_benign_overfitting.yaml"
GENERATED_CONFIG_DIR = PROJECT_ROOT / "src" / "conf" / "generated_markov"


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _normalize_inherit_paths(config: dict, template_path: Path) -> dict:
    """Make inherit entries absolute relative to the template file location."""
    inherit = config.get("inherit")
    if not isinstance(inherit, list):
        return config

    base_dir = template_path.parent
    normalized = []
    for item in inherit:
        if isinstance(item, str):
            candidate = Path(item)
            if not candidate.is_absolute():
                candidate = (base_dir / candidate).resolve()
            normalized.append(str(candidate))
        else:
            normalized.append(item)

    config["inherit"] = normalized
    return config


def _dump_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _merge_base_config(base_config: dict, modifications: dict) -> dict:
    config = copy.deepcopy(base_config)
    training = config.setdefault("training", {})

    if "data" in modifications:
        training["data"] = modifications["data"]
    if "data_kwargs" in modifications:
        training["data_kwargs"] = modifications["data_kwargs"]
    if "task" in modifications:
        training["task"] = modifications["task"]
    if "task_kwargs" in modifications:
        training["task_kwargs"] = modifications["task_kwargs"]
    if "out_dir" in modifications:
        config["out_dir"] = modifications["out_dir"]
    if "wandb" in modifications:
        config.setdefault("wandb", {})
        config["wandb"].update(modifications["wandb"])

    if "curriculum" in modifications:
        training["curriculum"] = modifications["curriculum"]

    return config


def _run_train(config_path: Path) -> None:
    command = [sys.executable, str(TRAIN_SCRIPT), "--config", str(config_path)]
    subprocess.run(command, check=True)


def _build_experiments(spec: dict, base_config: dict) -> list[tuple[str, dict]]:
    experiments: list[tuple[str, dict]] = []
    base_out_dir = PROJECT_ROOT / "models" / "markov_benign_overfitting"

    stationary_scales = spec.get("stationary_scales", [0.0, 0.3, 0.6, 0.9])
    sigma_x_values = spec.get("sigma_x_values", [0.1, 0.5, 1.0, 2.0])
    sigma_y_values = spec.get("sigma_y_values", [0.0, 0.01, 0.1, 0.5, 1.0])

    common_training = {
        "data": "markov",
        "task": "markov_noisy_linear_regression",
        "curriculum": {
            "dims": {"start": 20, "end": 20, "inc": 0, "interval": 1000000},
            "points": {"start": 40, "end": 40, "inc": 0, "interval": 1000000},
        },
    }

    for scale in stationary_scales:
        name = f"exp1_stationary_scale_{str(scale).replace('.', 'p')}"
        experiments.append(
            (
                name,
                {
                    **common_training,
                    "data_kwargs": {
                        "noise_std": 1.0,
                        "initial_std": 0.0,
                        "markov_mode": "stationary",
                        "markov_scale": float(scale),
                        "max_seq_length": 40,
                        "seed": 0,
                    },
                    "task_kwargs": {"noise_std": 0.1, "seed": 0},
                    "out_dir": str(base_out_dir / name),
                    "wandb": {
                        "name": f"Markov stationary scale={scale}",
                        "notes": f"Markov exp 1 stationary sweep, scale={scale}",
                    },
                },
            )
        )

    experiments.append(
        (
            "exp2_drift",
            {
                **common_training,
                "data_kwargs": {
                    "noise_std": 1.0,
                    "initial_std": 0.0,
                    "markov_mode": "drift",
                    "markov_scale": 0.6,
                    "markov_scale_start": 0.2,
                    "markov_scale_end": 0.9,
                    "max_seq_length": 40,
                    "seed": 0,
                },
                "task_kwargs": {"noise_std": 0.1, "seed": 0},
                "out_dir": str(base_out_dir / "exp2_drift"),
                "wandb": {
                    "name": "Markov drift",
                    "notes": "Markov exp 2 time-varying A_t drift",
                },
            },
        )
    )

    for sigma_x in sigma_x_values:
        name = f"exp3_sigma_x_{str(sigma_x).replace('.', 'p')}"
        experiments.append(
            (
                name,
                {
                    **common_training,
                    "data_kwargs": {
                        "noise_std": float(sigma_x),
                        "initial_std": 0.0,
                        "markov_mode": "stationary",
                        "markov_scale": 0.6,
                        "max_seq_length": 40,
                        "seed": 0,
                    },
                    "task_kwargs": {"noise_std": 0.1, "seed": 0},
                    "out_dir": str(base_out_dir / name),
                    "wandb": {
                        "name": f"Markov sigma_x={sigma_x}",
                        "notes": f"Markov exp 3 input-noise sweep, sigma_x={sigma_x}",
                    },
                },
            )
        )

    for sigma_y in sigma_y_values:
        name = f"exp4_sigma_y_{str(sigma_y).replace('.', 'p')}"
        experiments.append(
            (
                name,
                {
                    **common_training,
                    "data_kwargs": {
                        "noise_std": 1.0,
                        "initial_std": 0.0,
                        "markov_mode": "stationary",
                        "markov_scale": 0.6,
                        "max_seq_length": 40,
                        "seed": 0,
                    },
                    "task_kwargs": {"noise_std": float(sigma_y), "seed": 0},
                    "out_dir": str(base_out_dir / name),
                    "wandb": {
                        "name": f"Markov sigma_y={sigma_y}",
                        "notes": f"Markov exp 4 label-noise sweep, sigma_y={sigma_y}",
                    },
                },
            )
        )

    return experiments


def _describe_experiment(name: str, modifications: dict) -> str:
    data_kwargs = modifications.get("data_kwargs", {})
    task_kwargs = modifications.get("task_kwargs", {})

    if name.startswith("exp1_"):
        return (
            f"stationary A scale sweep: markov_scale={data_kwargs.get('markov_scale')}"
        )
    if name.startswith("exp2_"):
        return (
            f"drift sweep: markov_scale_start={data_kwargs.get('markov_scale_start')} -> "
            f"markov_scale_end={data_kwargs.get('markov_scale_end')}"
        )
    if name.startswith("exp3_"):
        return f"input-noise sweep: sigma_x={data_kwargs.get('noise_std')}"
    if name.startswith("exp4_"):
        return f"label-noise sweep: sigma_y={task_kwargs.get('noise_std')}"
    return "markov run"


def _print_sweep_summary(spec: dict) -> None:
    stationary_scales = spec.get("stationary_scales", [0.0, 0.3, 0.6, 0.9])
    sigma_x_values = spec.get("sigma_x_values", [0.1, 0.5, 1.0, 2.0])
    sigma_y_values = spec.get("sigma_y_values", [0.0, 0.01, 0.1, 0.5, 1.0])

    print("Markov sweep summary:")
    print(f"  exp1 stationary: data_kwargs.markov_scale in {stationary_scales[0]} -> {stationary_scales[-1]}")
    print("  exp2 drift: data_kwargs.markov_scale_start=0.2 -> data_kwargs.markov_scale_end=0.9")
    print(f"  exp3 input noise: data_kwargs.noise_std in {sigma_x_values[0]} -> {sigma_x_values[-1]}")
    print(f"  exp4 label noise: task_kwargs.noise_std in {sigma_y_values[0]} -> {sigma_y_values[-1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Markov experiments with train.py")
    parser.add_argument("--dry-run", action="store_true", help="Write configs but do not launch train.py")
    parser.add_argument("--exp", type=str, default="all", choices=["all", "exp1", "exp2", "exp3", "exp4"], help="Run only one experiment family")
    args = parser.parse_args()

    spec = _load_yaml(SPEC_PATH)
    base_config = _load_yaml(SMALL_TEMPLATE)
    base_config = _normalize_inherit_paths(base_config, SMALL_TEMPLATE)
    experiments = _build_experiments(spec, base_config)

    _print_sweep_summary(spec)

    for name, modifications in experiments:
        family = name.split("_")[0]
        if args.exp != "all" and args.exp != family:
            continue

        config = _merge_base_config(base_config, modifications)
        config_path = GENERATED_CONFIG_DIR / f"{name}.yaml"
        _dump_yaml(config_path, config)

        print(f"Running {name} via train.py")
        print(f"Sweep: {_describe_experiment(name, modifications)}")
        print(f"Config: {config_path}")
        if not args.dry_run:
            _run_train(config_path)


if __name__ == "__main__":
    main()
