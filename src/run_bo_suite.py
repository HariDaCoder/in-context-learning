"""Run the benign-overfitting experiment groups from any repository checkout.

The launcher uses the current Python interpreter and resolves every path from
the repository root, so it can be called from a login shell, tmux, or a batch
scheduler without depending on the caller's working directory.

Examples::

    python src/run_bo_suite.py smoke
    python src/run_bo_suite.py train --preset all --dry-run
    python src/run_bo_suite.py train --preset stationary
    python src/run_bo_suite.py evaluate --preset all --device cuda \
        --run-dir models/bo_iid/<uuid> --run-dir models/bo_markov/<uuid>
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"

TRAINING_PRESETS = {
    "stationary": (
        "bo_iid.yaml",
        "bo_markov.yaml",
        "bo_noise_markov.yaml",
        "bo_both_markov.yaml",
    ),
    "change-point": (
        "bo_change_point.yaml",
        "bo_change_point_reverse.yaml",
    ),
}

EVALUATION_PRESETS = {
    "stationary": (
        "bo_sweep_phase.yaml",
        "bo_sweep_noise_markov.yaml",
    ),
    "change-point": (
        "bo_sweep_change_forward.yaml",
        "bo_sweep_change_reverse.yaml",
    ),
}


def _preset_items(presets, name):
    """Expand one named preset while preserving order and removing duplicates."""
    names = tuple(presets) if name == "all" else (name,)
    return tuple(dict.fromkeys(item for preset in names for item in presets[preset]))


def _repository_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _display_path(path):
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def training_configs(preset, extra_configs=()):
    names = () if preset == "none" else _preset_items(TRAINING_PRESETS, preset)
    configs = [SOURCE_ROOT / "conf" / name for name in names]
    configs.extend(_repository_path(path) for path in extra_configs)
    return tuple(dict.fromkeys(configs))


def evaluation_configs(preset, include_boundary=False):
    configs = [SOURCE_ROOT / "conf" / name for name in _preset_items(EVALUATION_PRESETS, preset)]
    if include_boundary:
        configs.append(SOURCE_ROOT / "conf" / "bo_sweep_boundary.yaml")
    return tuple(configs)


def _command_text(command):
    if os.name == "nt":
        return subprocess.list2cmdline([str(value) for value in command])
    return shlex.join(str(value) for value in command)


def _run(command, cwd, dry_run=False):
    print(f"[{cwd}] $ {_command_text(command)}", flush=True)
    if not dry_run:
        subprocess.run([str(value) for value in command], cwd=str(cwd), check=True)


def _checkpoint_snapshot():
    return {
        path.parent.resolve(): path.stat().st_mtime_ns
        for path in (REPOSITORY_ROOT / "models").glob("**/state.pt")
    }


def _write_training_manifest(before, configs, output_root):
    after = _checkpoint_snapshot()
    changed = sorted(
        str(path.relative_to(REPOSITORY_ROOT))
        for path, modified in after.items()
        if path not in before or modified != before[path]
    )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "trained_checkpoints.json"
    document = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configs": [_display_path(path) for path in configs],
        "checkpoint_dirs": changed,
    }
    manifest.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(f"Wrote {len(changed)} new or updated checkpoint directories to {manifest}")


def run_training(args):
    configs = training_configs(args.preset, args.config)
    if not configs:
        raise ValueError("Select a non-empty --preset or provide at least one --config")
    for config in configs:
        if not config.is_file():
            raise FileNotFoundError(f"Training config does not exist: {config}")

    before = _checkpoint_snapshot() if not args.dry_run else {}
    for config in configs:
        # train.py expects inherited config paths to be relative to src/conf.
        try:
            config_argument = config.relative_to(SOURCE_ROOT)
        except ValueError:
            config_argument = config
        _run(
            (sys.executable, "train.py", "--config", config_argument),
            cwd=SOURCE_ROOT,
            dry_run=args.dry_run,
        )
    if not args.dry_run:
        _write_training_manifest(before, configs, _repository_path(args.output_root))


def checkpoint_dirs(run_dirs=(), manifests=()):
    paths = [_repository_path(path) for path in run_dirs]
    for manifest_value in manifests:
        manifest = _repository_path(manifest_value)
        with manifest.open(encoding="utf-8") as handle:
            document = json.load(handle)
        listed = document.get("checkpoint_dirs")
        if not isinstance(listed, list) or not all(isinstance(path, str) for path in listed):
            raise ValueError(f"{manifest}: checkpoint_dirs must be a list of paths")
        paths.extend(_repository_path(path) for path in listed)
    return tuple(dict.fromkeys(paths))


def _evaluate_one(config, args, checkpoints):
    output_root = _repository_path(args.output_root)
    result_path = output_root / f"{config.stem}.json"
    figure_path = output_root / f"{config.stem}_figures"
    command = [
        sys.executable,
        SOURCE_ROOT / "bo_experiment.py",
        "--config",
        config,
        "--device",
        args.device,
        "--output",
        result_path,
    ]
    for run_dir in checkpoints:
        command.extend(("--run-dir", run_dir))
    _run(command, cwd=REPOSITORY_ROOT, dry_run=args.dry_run)

    plot_command = [
        sys.executable,
        SOURCE_ROOT / "bo_plot.py",
        result_path,
        "--out-dir",
        figure_path,
    ]
    if args.log_y:
        plot_command.append("--log-y")
    _run(plot_command, cwd=REPOSITORY_ROOT, dry_run=args.dry_run)


def run_evaluation(args):
    checkpoints = checkpoint_dirs(args.run_dir, args.checkpoint_manifest)
    if not args.dry_run:
        for checkpoint in checkpoints:
            if not (checkpoint / "config.yaml").is_file() or not (checkpoint / "state.pt").is_file():
                raise FileNotFoundError(f"Checkpoint directory is incomplete: {checkpoint}")
    for config in evaluation_configs(args.preset, args.include_boundary):
        if not config.is_file():
            raise FileNotFoundError(f"Evaluation config does not exist: {config}")
        _evaluate_one(config, args, checkpoints)


def run_smoke(args):
    output_root = _repository_path(args.output_root)
    result_path = output_root / "bo_smoke.json"
    figure_path = output_root / "bo_smoke_figures"
    _run(
        (
            sys.executable,
            SOURCE_ROOT / "bo_experiment.py",
            "--config",
            SOURCE_ROOT / "conf" / "bo_sweep.yaml",
            "--output",
            result_path,
        ),
        cwd=REPOSITORY_ROOT,
        dry_run=args.dry_run,
    )
    plot_command = [
        sys.executable,
        SOURCE_ROOT / "bo_plot.py",
        result_path,
        "--out-dir",
        figure_path,
    ]
    if args.log_y:
        plot_command.append("--log-y")
    _run(plot_command, cwd=REPOSITORY_ROOT, dry_run=args.dry_run)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    smoke = commands.add_parser("smoke", help="Run the small baseline sweep and plots")
    smoke.add_argument("--output-root", type=Path, default=REPOSITORY_ROOT / "results" / "smoke")
    smoke.add_argument("--log-y", action="store_true")
    smoke.add_argument("--dry-run", action="store_true")
    smoke.set_defaults(function=run_smoke)

    train = commands.add_parser("train", help="Train a sequence of matched-distribution models")
    train.add_argument(
        "--preset",
        choices=(*TRAINING_PRESETS, "all", "none"),
        default="stationary",
    )
    train.add_argument(
        "--config",
        action="append",
        default=[],
        help="Append a custom training YAML; repeat for multiple configs",
    )
    train.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "bo_suite",
        help="Directory for the checkpoint manifest (model out_dir still comes from each YAML)",
    )
    train.add_argument("--dry-run", action="store_true")
    train.set_defaults(function=run_training)

    evaluate = commands.add_parser("evaluate", help="Run evaluation grids and generate plots")
    evaluate.add_argument(
        "--preset",
        choices=(*EVALUATION_PRESETS, "all"),
        default="stationary",
    )
    evaluate.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Checkpoint UUID directory; repeat to compare independent training runs",
    )
    evaluate.add_argument(
        "--checkpoint-manifest",
        action="append",
        default=[],
        help="Manifest written by the train command; repeat to combine batches",
    )
    evaluate.add_argument("--device", default="cpu", help="Device for Transformer checkpoints, such as cuda or cpu")
    evaluate.add_argument("--include-boundary", action="store_true", help="Also run the expensive hand-edited boundary grid")
    evaluate.add_argument("--output-root", type=Path, default=REPOSITORY_ROOT / "results" / "bo_suite")
    evaluate.add_argument("--log-y", action="store_true")
    evaluate.add_argument("--dry-run", action="store_true")
    evaluate.set_defaults(function=run_evaluation)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
