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

# Scientific group names are deliberately independent of the machine that runs
# them.  Resource assignment belongs in the launch instructions, not in stable
# experiment IDs or persisted manifests.
MATRIX_GROUPS = (
    "architecture", "canonical", "dimension", "matched_rho",
    "matched_snr_pilot", "stage0",
)


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


def _print_json(document):
    print(json.dumps(document, indent=2, sort_keys=True), flush=True)


def _matrix_plan_summary(plan):
    return {
        "group": plan.group,
        "counts": plan.counts,
        "device": plan.device,
        "precision": plan.precision,
        "max_concurrent": plan.max_concurrent,
        "resume": plan.resume,
        "manifest_json": _display_path(plan.manifest_json) if plan.manifest_json else None,
        "manifest_csv": _display_path(plan.manifest_csv) if plan.manifest_csv else None,
        "plan_json": _display_path(plan.summary_path),
    }


def _training_concurrency(args):
    if args.max_concurrent is not None:
        return args.max_concurrent
    if args.matrix_manifest is not None:
        return 1
    from bo_matrix_train import recommended_concurrency

    return recommended_concurrency(args.group) or 1


def run_matrix_plan(args):
    from bo_matrix_train import plan_group, plan_manifest

    planner = plan_manifest if args.matrix_manifest is not None else plan_group
    source = args.matrix_manifest if args.matrix_manifest is not None else args.group
    plan = planner(
        source,
        device=args.device,
        precision=args.precision,
        max_concurrent=_training_concurrency(args),
        resume=args.resume,
    )
    _print_json(_matrix_plan_summary(plan))


def run_matrix_training(args):
    from bo_matrix_train import train_group, train_manifest
    if args.matrix_manifest is None and args.group in {"matched_rho", "dimension", "architecture"} and not args.dry_run:
        from bo_scientific_audit import require_main_gate
        require_main_gate(REPOSITORY_ROOT)

    trainer = train_manifest if args.matrix_manifest is not None else train_group
    source = args.matrix_manifest if args.matrix_manifest is not None else args.group
    summary = trainer(
        source,
        device=args.device,
        precision=args.precision,
        max_concurrent=_training_concurrency(args),
        dry_run=args.dry_run,
        resume=args.resume,
    )
    _print_json(summary.as_dict(REPOSITORY_ROOT))
    if not summary.success:
        raise RuntimeError(
            "Matrix training contains failed, blocked, or locked experiments; "
            "inspect {}".format(_display_path(summary.summary_path))
        )


def run_scientific_audit(args):
    from bo_scientific_audit import audit

    report, json_path, csv_path, markdown_path = audit(args.input, args.output_dir)
    _print_json({
        "recommendation": report["recommendation"],
        "transformer_rows": report["transformer_rows"],
        "fully_matched_rows": report["fully_matched_rows"],
        "direct_bo_rows": report["direct_bo_rows"],
        "linear_bo_rows": report["linear_bo_rows"],
        "json": _display_path(json_path),
        "csv": _display_path(csv_path),
        "markdown": _display_path(markdown_path),
    })


def run_matrix_evaluation(args):
    from bo_matrix_eval import main as matrix_evaluation_main

    command = []
    for group in args.group:
        command.extend(("--group", group))
    for manifest in args.matrix_manifest:
        command.extend(("--matrix-manifest", str(manifest)))
    for experiment_id in args.experiment_id:
        command.extend(("--experiment-id", experiment_id))
    command.extend((
        "--protocol", args.protocol,
        "--n-eval", str(args.n_eval),
        "--batch-size", str(args.batch_size),
        "--device", args.device,
        "--max-workers", str(args.max_concurrent),
        "--output-root", str(args.output_root),
    ))
    for flag, values in (
        ("--snr", args.snr),
        ("--eval-seed", args.eval_seed),
        ("--k-over-d", args.k_over_d),
        ("--shift-rho", args.shift_rho),
    ):
        for value in values:
            command.extend((flag, str(value)))
    for path in args.boundary_suggestions:
        command.extend(("--boundary-suggestions", str(path)))
    if args.match_train_snr:
        command.append("--match-train-snr")
    if args.linear_probe:
        command.append("--linear-probe")
    if args.no_baselines:
        command.append("--no-baselines")
    if args.no_plot:
        command.append("--no-plot")
    if args.summary_only:
        command.append("--summary-only")
    if args.force:
        command.append("--force")
    if args.dry_run:
        command.append("--dry-run")
    return matrix_evaluation_main(command)


def run_matrix_benchmark(args):
    from bo_matrix_train import benchmark_group

    report = benchmark_group(
        args.group,
        device=args.device,
        precision=args.precision,
        concurrency_values=args.max_concurrent,
        steps=args.steps,
        dry_run=args.dry_run,
        resume=args.resume,
    )
    _print_json(report)


def run_parameter_match(args):
    from bo_architecture import ArchitectureSpec
    from bo_architecture_runner import parameter_match_report

    result = parameter_match_report(
        output_dir=args.output_dir,
        target=ArchitectureSpec(
            args.target_width,
            args.target_depth,
            args.target_heads,
            sweep_family="standard",
        ),
        depths=args.depths,
        width_min=args.width_min,
        width_max=args.width_max,
        width_step=args.width_step,
        n_dims=args.n_dims,
        max_context=args.max_context,
        precision=args.precision,
        dry_run=args.dry_run,
    )
    _print_json({
        "status": "planned" if result.dry_run else "complete",
        "report_id": result.plan["report_id"],
        "search_model_instantiations": result.plan["search_model_instantiations"],
        "match_count": len(result.matches),
        "training_experiment_count": (
            len(result.experiments)
            if not result.dry_run
            else result.plan["planned_training_experiment_count"]
        ),
        "json_path": _display_path(result.json_path),
        "csv_path": _display_path(result.csv_path),
    })


def run_mechanism(args):
    from bo_architecture_runner import mechanism_evaluation

    result = mechanism_evaluation(
        checkpoint_dir=args.checkpoint_dir,
        output_csv=args.output_csv,
        rhos=args.rho,
        snrs=args.snr,
        context_lengths=args.context_length,
        eval_seeds=args.eval_seed,
        n_eval=args.n_eval,
        batch_size=args.batch_size,
        device=args.device,
        query_positions=args.query_positions,
        dry_run=args.dry_run,
    )
    _print_json(result)


def run_scaling_analysis(args):
    from bo_scaling import write_scaling_report

    report = write_scaling_report(
        inputs=args.input,
        output=args.output,
        target=args.target_probability,
        figures_dir=args.figures_dir,
    )
    _print_json({
        "output": _display_path(_repository_path(args.output)),
        "observation_count": len(report["observations"]),
        "excluded_condition_count": len(report["excluded_conditions"]),
        "fit_status": report["fit"]["status"],
        "figures": report["figures"],
    })


def _add_resume_arguments(parser):
    parser.set_defaults(resume=True)
    parser.add_argument("--resume", dest="resume", action="store_true",
                        help="Resume interrupted experiments (default)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Block instead of resuming an interrupted experiment")


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

    plan = commands.add_parser(
        "plan", help="Expand a deterministic matrix group and write its manifests"
    )
    plan_source = plan.add_mutually_exclusive_group(required=True)
    plan_source.add_argument("--group", choices=MATRIX_GROUPS)
    plan_source.add_argument(
        "--matrix-manifest", type=Path,
        help="Matrix manifest or parameter-match report to plan",
    )
    plan.add_argument("--device", default="auto")
    plan.add_argument(
        "--precision", choices=("float32", "float16", "bfloat16"), default=None,
        help="Override precision; this changes semantic experiment IDs",
    )
    plan.add_argument(
        "--max-concurrent", type=int, default=None,
        help="Process count; defaults to the latest successful group benchmark, then 1",
    )
    _add_resume_arguments(plan)
    plan.set_defaults(function=run_matrix_plan)

    matrix_train = commands.add_parser(
        "train-matrix", help="Train, resume, or skip a deterministic matrix group"
    )
    matrix_train_source = matrix_train.add_mutually_exclusive_group(required=True)
    matrix_train_source.add_argument("--group", choices=MATRIX_GROUPS)
    matrix_train_source.add_argument(
        "--matrix-manifest", type=Path,
        help="Matrix manifest or parameter-match report to train",
    )
    matrix_train.add_argument("--device", default="auto")
    matrix_train.add_argument(
        "--precision", choices=("float32", "float16", "bfloat16"), default=None,
        help="Override precision; this changes semantic experiment IDs",
    )
    matrix_train.add_argument(
        "--max-concurrent", type=int, default=None,
        help="Process count; defaults to the latest successful group benchmark, then 1",
    )
    matrix_train.add_argument("--dry-run", action="store_true")
    _add_resume_arguments(matrix_train)
    matrix_train.set_defaults(function=run_matrix_training)

    matrix_evaluate = commands.add_parser(
        "evaluate-matrix",
        help="Evaluate matrix checkpoints with dependence-matched and/or shift protocols",
    )
    matrix_evaluate.add_argument(
        "--group", action="append", choices=MATRIX_GROUPS, default=[],
        help="Scientific matrix group; repeat to combine groups",
    )
    matrix_evaluate.add_argument(
        "--matrix-manifest", action="append", type=Path, default=[],
        help="Manifest containing training experiments; repeatable",
    )
    matrix_evaluate.add_argument("--experiment-id", action="append", default=[])
    matrix_evaluate.add_argument(
        "--protocol", choices=("matched", "shift", "all"), default="matched"
    )
    matrix_evaluate.add_argument("--snr", action="append", type=float, default=[])
    matrix_evaluate.add_argument("--match-train-snr", action="store_true")
    matrix_evaluate.add_argument("--linear-probe", action="store_true")
    matrix_evaluate.add_argument("--eval-seed", action="append", type=int, default=[])
    matrix_evaluate.add_argument("--k-over-d", action="append", type=float, default=[])
    matrix_evaluate.add_argument("--shift-rho", action="append", type=float, default=[])
    matrix_evaluate.add_argument(
        "--boundary-suggestions", action="append", type=Path, default=[],
        help="Use follow-up SNR samples from boundary_suggestions.json; repeatable",
    )
    matrix_evaluate.add_argument("--n-eval", type=int, default=32)
    matrix_evaluate.add_argument("--batch-size", type=int, default=16)
    matrix_evaluate.add_argument("--device", default="cpu")
    matrix_evaluate.add_argument("--max-concurrent", type=int, default=1)
    matrix_evaluate.add_argument(
        "--output-root", type=Path,
        default=Path("results") / "bo_matrix" / "evaluation",
    )
    matrix_evaluate.add_argument("--no-baselines", action="store_true")
    matrix_evaluate.add_argument("--no-plot", action="store_true")
    matrix_evaluate.add_argument("--summary-only", action="store_true")
    matrix_evaluate.add_argument("--force", action="store_true")
    matrix_evaluate.add_argument("--dry-run", action="store_true")
    matrix_evaluate.set_defaults(function=run_matrix_evaluation)

    scientific_audit = commands.add_parser(
        "scientific-audit", help="Write the audit that gates the main BO matrix"
    )
    scientific_audit.add_argument("--input", action="append", type=Path, required=True)
    scientific_audit.add_argument(
        "--output-dir", type=Path,
        default=REPOSITORY_ROOT / "results" / "bo_matrix" / "scientific_audit",
    )
    scientific_audit.set_defaults(function=run_scientific_audit)

    benchmark = commands.add_parser(
        "benchmark", help="Benchmark independent experiment concurrency with short runs"
    )
    benchmark.add_argument("--group", choices=MATRIX_GROUPS, default="stage0")
    benchmark.add_argument("--device", default="auto")
    benchmark.add_argument(
        "--precision", choices=("float32", "float16", "bfloat16"), default=None
    )
    benchmark.add_argument(
        "--max-concurrent", type=int, nargs="+", default=[1, 2, 4],
        help="Concurrency values to benchmark",
    )
    benchmark.add_argument("--steps", type=int, default=2000)
    benchmark.add_argument("--dry-run", action="store_true")
    _add_resume_arguments(benchmark)
    benchmark.set_defaults(function=run_matrix_benchmark)

    parameter_match = commands.add_parser(
        "parameter-match", help="Find exact-count parameter-matched depth variants"
    )
    parameter_match.add_argument(
        "--output-dir", type=Path,
        default=REPOSITORY_ROOT / "results" / "bo_matrix" / "parameter_match",
    )
    parameter_match.add_argument("--target-width", type=int, default=256)
    parameter_match.add_argument("--target-depth", type=int, default=12)
    parameter_match.add_argument("--target-heads", type=int, default=8)
    parameter_match.add_argument("--depths", type=int, nargs="+", default=[2, 4, 6, 12])
    parameter_match.add_argument("--width-min", type=int, default=32)
    parameter_match.add_argument("--width-max", type=int, default=768)
    parameter_match.add_argument("--width-step", type=int, default=8)
    parameter_match.add_argument("--n-dims", type=int, default=20)
    parameter_match.add_argument("--max-context", type=int, default=80)
    parameter_match.add_argument(
        "--precision", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parameter_match.add_argument("--dry-run", action="store_true")
    parameter_match.set_defaults(function=run_parameter_match)

    mechanism = commands.add_parser(
        "mechanism", help="Compute online aggregate attention diagnostics"
    )
    mechanism.add_argument("--checkpoint-dir", type=Path, required=True)
    mechanism.add_argument("--output-csv", type=Path, required=True)
    mechanism.add_argument("--rho", type=float, nargs="+", default=[0.0, 0.6, 0.9])
    mechanism.add_argument("--snr", type=float, nargs="+", default=[0.8, 3.2])
    mechanism.add_argument(
        "--context-length", type=int, nargs="+", default=[20, 40, 80]
    )
    mechanism.add_argument(
        "--eval-seed", type=int, nargs="+", default=[1001, 1002, 1003]
    )
    mechanism.add_argument("--n-eval", type=int, default=32)
    mechanism.add_argument("--batch-size", type=int, default=16)
    mechanism.add_argument("--device", default="auto")
    mechanism.add_argument(
        "--query-positions", choices=("last_x", "all_x", "all_y", "all"),
        default="last_x",
    )
    mechanism.add_argument("--dry-run", action="store_true")
    mechanism.set_defaults(function=run_mechanism)

    scaling = commands.add_parser(
        "analyze-scaling", help="Fit the explicitly exploratory critical-SNR scaling model"
    )
    scaling.add_argument("--input", action="append", type=Path, required=True)
    scaling.add_argument(
        "--output", type=Path,
        default=REPOSITORY_ROOT / "results" / "bo_matrix" / "scaling.json",
    )
    scaling.add_argument("--figures-dir", type=Path, default=None)
    scaling.add_argument("--target-probability", type=float, default=0.5)
    scaling.set_defaults(function=run_scaling_analysis)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
