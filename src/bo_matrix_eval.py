"""Build and run dense evaluation jobs for :mod:`bo_matrix` checkpoints.

The launcher is environment-neutral: job names encode scientific inputs, not
machines or accelerators.  It writes one Transformer-only result per training
checkpoint and one baseline result per unique evaluation grid.  Consequently,
combining a compatible result bundle never duplicates OLS/ridge/GLS rows.

Examples, run from any directory::

    python src/bo_matrix_eval.py --group stage0 --protocol matched --dry-run
    python -u src/bo_matrix_eval.py --group canonical --protocol all --device cuda

``matched`` and ``shift`` refer only to the dependence process. Dense test-SNR
sweeps are retained in both jobs and train/test SNRs are recorded separately.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import uuid

import yaml

from bo_matrix import (
    DEFAULT_EVAL_SEEDS,
    DENSE_TEST_RHOS,
    DENSE_TEST_SNRS,
    DIMENSION_K_OVER_D,
    GROUPS,
    ExperimentSpec,
    REPOSITORY_ROOT,
    expand_group,
    load_experiment_manifest,
)


SOURCE_ROOT = REPOSITORY_ROOT / "src"
DEFAULT_OUTPUT_ROOT = Path("results/bo_matrix/evaluation")
RESULT_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _stable_id(prefix: str, value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:20]
    return "{}_{}".format(prefix, digest)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{}.{}.tmp".format(path.name, uuid.uuid4().hex))
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _positive_floats(values: Sequence[float], name: str) -> Tuple[float, ...]:
    result = tuple(dict.fromkeys(float(value) for value in values))
    if not result or any(not math.isfinite(value) or value <= 0 for value in result):
        raise ValueError("{} must contain positive finite values".format(name))
    return result


def _nonnegative_ints(values: Sequence[int], name: str) -> Tuple[int, ...]:
    result = tuple(dict.fromkeys(values))
    if not result or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                         for value in result):
        raise ValueError("{} must contain nonnegative integers".format(name))
    return result


def load_boundary_suggestions(path: Path) -> Tuple[float, ...]:
    """Read the union of finite positive follow-up SNR samples from a plot artifact."""

    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("schema_version") != 1 or not isinstance(document.get("groups"), list):
        raise ValueError("invalid boundary suggestions artifact: {}".format(source))
    values = []
    for group in document["groups"]:
        if not isinstance(group, Mapping) or not isinstance(group.get("suggestions"), list):
            raise ValueError("invalid boundary suggestion group: {}".format(source))
        for condition in group["suggestions"]:
            if not isinstance(condition, Mapping):
                raise ValueError("invalid boundary suggestion condition: {}".format(source))
            suggested = condition.get("suggested_snrs")
            if not isinstance(suggested, list):
                raise ValueError("boundary condition lacks suggested_snrs: {}".format(source))
            for value in suggested:
                try:
                    numeric = float(value)
                except (TypeError, ValueError) as error:
                    raise ValueError("boundary SNR must be numeric: {}".format(value)) from error
                if not math.isfinite(numeric) or numeric <= 0:
                    raise ValueError("boundary SNR must be positive and finite")
                values.append(numeric)
    result = tuple(sorted(set(values)))
    if not result:
        raise ValueError("boundary suggestions contain no follow-up SNR samples")
    return result


def context_lengths_for_spec(
    spec: ExperimentSpec, k_over_d: Sequence[float] = DIMENSION_K_OVER_D
) -> Tuple[int, ...]:
    ratios = _positive_floats(k_over_d, "k_over_d")
    contexts = tuple(dict.fromkeys(int(round(ratio * spec.d)) for ratio in ratios))
    contexts = tuple(value for value in contexts if 0 < value <= spec.max_context)
    if not contexts:
        raise ValueError("no requested context length fits checkpoint {}".format(spec.experiment_id))
    return contexts


def _is_nonstationary(spec: ExperimentSpec) -> bool:
    return spec.rho_x_after is not None or spec.rho_e_after is not None


def evaluation_config(
    spec: ExperimentSpec,
    protocol: str,
    test_snrs: Sequence[float] = DENSE_TEST_SNRS,
    eval_seeds: Sequence[int] = DEFAULT_EVAL_SEEDS,
    k_over_d: Sequence[float] = DIMENSION_K_OVER_D,
    shift_test_rhos: Sequence[float] = DENSE_TEST_RHOS,
    n_eval: int = 32,
    batch_size: int = 16,
    baselines: Sequence[Any] = (),
    match_train_snr: bool = False,
    linear_probe: bool = False,
) -> Optional[Dict[str, Any]]:
    """Return one bo_experiment config for a dependence-matched/shifted job.

    Nonstationary matched jobs reproduce the checkpoint's complete schedule.
    Their shift control is stationary over the dense rho grid, which changes
    temporal ordering while keeping Gaussian one-time marginals unchanged.
    """
    if protocol not in {"matched", "shift"}:
        raise ValueError("protocol must be matched or shift")
    if isinstance(n_eval, bool) or not isinstance(n_eval, int) or n_eval < 1:
        raise ValueError("n_eval must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    snrs = (float(spec.train_snr),) if match_train_snr else _positive_floats(test_snrs, "test_snrs")
    seeds = _nonnegative_ints(eval_seeds, "eval_seeds")
    contexts = context_lengths_for_spec(spec, k_over_d)
    shift_rhos = tuple(dict.fromkeys(float(value) for value in shift_test_rhos))
    if any(not math.isfinite(value) or abs(value) >= 1 for value in shift_rhos):
        raise ValueError("shift_test_rhos must be finite with absolute value below one")

    nonstationary = _is_nonstationary(spec)
    if protocol == "matched":
        test_rhos = (spec.train_rho_x,)
        feature_rho_after = spec.rho_x_after
        feature_change_point = spec.feature_change_point
        noise_rho_after = spec.rho_e_after
        noise_change_point = spec.noise_change_point
    else:
        # Every stationary control differs from a nonstationary training
        # schedule.  For stationary training, exclude the matched rho.
        test_rhos = shift_rhos if nonstationary else tuple(
            rho for rho in shift_rhos if rho != spec.train_rho_x
        )
        if not test_rhos:
            return None
        feature_rho_after = None
        feature_change_point = None
        noise_rho_after = None
        noise_change_point = None

    return {
        "n_dims": [spec.d],
        "context_lengths": list(contexts),
        "rhos": list(test_rhos),
        "snrs": list(snrs),
        "eval_seeds": list(seeds),
        "n_eval": n_eval,
        "batch_size": min(batch_size, n_eval),
        "n_queries": 32,
        "n_probe_queries": None,
        "n_probe_test_queries": 64,
        "linear_probe": linear_probe,
        "query_batch_size": 256,
        "fit_threshold": 0.0001,
        "gen_threshold": 0.1,
        "linearity_threshold": 0.99,
        "baselines": list(baselines),
        "feature_rho_after": feature_rho_after,
        "feature_change_point": feature_change_point,
        "noise_rho": spec.train_rho_e,
        "noise_rho_after": noise_rho_after,
        "noise_change_point": noise_change_point,
    }


def _bundle_payload(protocol: str, config: Mapping[str, Any]) -> Dict[str, Any]:
    payload = dict(config)
    payload.pop("baselines", None)
    return {"dependence_protocol": protocol, "evaluation_config": payload}


@dataclass(frozen=True)
class EvaluationJob:
    job_id: str
    bundle_id: str
    group: str
    protocol: str
    kind: str
    config: Mapping[str, Any]
    config_path: str
    result_path: str
    log_path: str
    failure_path: str
    training_experiment_id: Optional[str] = None
    checkpoint_dir: Optional[str] = None

    def __post_init__(self) -> None:
        if self.protocol not in {"matched", "shift"}:
            raise ValueError("job protocol must be matched or shift")
        if self.kind not in {"baseline", "checkpoint"}:
            raise ValueError("job kind must be baseline or checkpoint")
        if self.kind == "checkpoint" and (not self.checkpoint_dir or not self.training_experiment_id):
            raise ValueError("checkpoint jobs need a training experiment and checkpoint directory")
        if self.kind == "baseline" and (self.checkpoint_dir or self.training_experiment_id):
            raise ValueError("baseline jobs cannot reference a checkpoint")

    def manifest_row(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "bundle_id": self.bundle_id,
            "group": self.group,
            "dependence_protocol": self.protocol,
            "kind": self.kind,
            "training_experiment_id": self.training_experiment_id,
            "checkpoint_dir": self.checkpoint_dir,
            "config_path": self.config_path,
            "result_path": self.result_path,
            "log_path": self.log_path,
            "failure_path": self.failure_path,
        }


def _job(
    group: str,
    protocol: str,
    kind: str,
    config: Mapping[str, Any],
    output_root: Path,
    bundle_id: str,
    spec: Optional[ExperimentSpec] = None,
) -> EvaluationJob:
    identity = {
        "bundle_id": bundle_id,
        "kind": kind,
        "training_experiment_id": spec.experiment_id if spec else None,
    }
    job_id = _stable_id("eval", identity)
    checkpoint_dir = spec.checkpoint_dir.as_posix() if spec else None
    return EvaluationJob(
        job_id=job_id,
        bundle_id=bundle_id,
        group=group,
        protocol=protocol,
        kind=kind,
        config=dict(config),
        config_path=(output_root / "configs" / "{}.yaml".format(job_id)).as_posix(),
        result_path=(output_root / "results" / "{}.json".format(job_id)).as_posix(),
        log_path=(output_root / "logs" / "{}.log".format(job_id)).as_posix(),
        failure_path=(output_root / "failures" / "{}.json".format(job_id)).as_posix(),
        training_experiment_id=spec.experiment_id if spec else None,
        checkpoint_dir=checkpoint_dir,
    )


def build_evaluation_jobs(
    specs: Iterable[ExperimentSpec],
    protocols: Sequence[str] = ("matched",),
    test_snrs: Sequence[float] = DENSE_TEST_SNRS,
    eval_seeds: Sequence[int] = DEFAULT_EVAL_SEEDS,
    k_over_d: Sequence[float] = DIMENSION_K_OVER_D,
    shift_test_rhos: Sequence[float] = DENSE_TEST_RHOS,
    n_eval: int = 32,
    batch_size: int = 16,
    include_baselines: bool = True,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    match_train_snr: bool = False,
    linear_probe: bool = False,
) -> Tuple[EvaluationJob, ...]:
    """Build semantically deduplicated checkpoint and baseline jobs."""
    protocol_values = tuple(dict.fromkeys(protocols))
    if not protocol_values or any(value not in {"matched", "shift"} for value in protocol_values):
        raise ValueError("protocols must contain matched and/or shift")
    output_root = Path(output_root)
    if output_root.is_absolute() or ".." in output_root.parts:
        raise ValueError("output_root must be repository-relative")
    unique_specs = {}
    for spec in specs:
        unique_specs.setdefault(spec.experiment_id, spec)
    if not unique_specs:
        raise ValueError("at least one training spec is required")

    jobs = {}
    baseline_configs = {}
    bundle_groups = {}
    for spec in unique_specs.values():
        for protocol in protocol_values:
            config = evaluation_config(
                spec, protocol, test_snrs, eval_seeds, k_over_d,
                shift_test_rhos, n_eval, batch_size, baselines=(), match_train_snr=match_train_snr,
                linear_probe=linear_probe,
            )
            if config is None:
                continue
            bundle_id = _stable_id("bundle", _bundle_payload(protocol, config))
            group = spec.group
            bundle_groups.setdefault(bundle_id, set()).add(group)
            checkpoint_job = _job(
                group, protocol, "checkpoint", config, output_root, bundle_id, spec
            )
            jobs.setdefault(checkpoint_job.job_id, checkpoint_job)
            baseline_config = dict(config)
            baseline_config["baselines"] = [
                "ols", {"kind": "ridge", "ridge_alpha": 1.0}, "gls"
            ]
            baseline_configs.setdefault(bundle_id, (protocol, baseline_config))

    if include_baselines:
        for bundle_id, (protocol, config) in baseline_configs.items():
            groups = sorted(bundle_groups[bundle_id])
            group = groups[0] if len(groups) == 1 else "combined"
            baseline_job = _job(
                group, protocol, "baseline", config, output_root, bundle_id
            )
            jobs.setdefault(baseline_job.job_id, baseline_job)
    return tuple(sorted(jobs.values(), key=lambda item: (item.bundle_id, item.kind != "baseline", item.job_id)))


def result_bundles(jobs: Iterable[EvaluationJob]) -> Dict[str, Tuple[EvaluationJob, ...]]:
    bundles: Dict[str, List[EvaluationJob]] = {}
    for job in jobs:
        bundles.setdefault(job.bundle_id, []).append(job)
    output = {}
    for bundle_id, members in bundles.items():
        baselines = [job for job in members if job.kind == "baseline"]
        if len(baselines) > 1:
            raise ValueError("bundle {} has duplicate baseline jobs".format(bundle_id))
        output[bundle_id] = tuple(sorted(members, key=lambda item: (item.kind != "baseline", item.job_id)))
    return output


def load_matrix_manifest(path: Path) -> Tuple[ExperimentSpec, ...]:
    """Backward-compatible alias for the shared verified manifest loader."""

    return load_experiment_manifest(path)


def _local(repository_root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("job path must be repository-relative: {}".format(path))
    return Path(repository_root) / path


def write_job_config(job: EvaluationJob, repository_root: Path = REPOSITORY_ROOT) -> Path:
    destination = _local(repository_root, job.config_path)
    _atomic_write(destination, yaml.safe_dump(dict(job.config), sort_keys=False))
    return destination


def _command_text(command: Sequence[Any]) -> str:
    values = [str(value) for value in command]
    return subprocess.list2cmdline(values) if os.name == "nt" else shlex.join(values)


def _valid_result(path: Path) -> bool:
    try:
        with path.open(encoding="utf-8") as handle:
            document = json.load(handle)
        return document.get("schema_version") == RESULT_SCHEMA_VERSION and isinstance(
            document.get("records"), list
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def run_job(
    job: EvaluationJob,
    repository_root: Path = REPOSITORY_ROOT,
    device: str = "cpu",
    dry_run: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    """Materialize and optionally execute one job, returning manifest status."""
    root = Path(repository_root)
    config_path = write_job_config(job, root)
    result_path = _local(root, job.result_path)
    log_path = _local(root, job.log_path)
    failure_path = _local(root, job.failure_path)
    command = [
        sys.executable, root / "src" / "bo_experiment.py", "--config", config_path,
        "--output", result_path, "--device", device,
    ]
    if job.kind == "checkpoint":
        command.extend(("--run-dir", _local(root, job.checkpoint_dir)))
    status = {**job.manifest_row(), "command": [str(value) for value in command]}
    if dry_run:
        return {**status, "status": "planned"}
    if not force and _valid_result(result_path):
        return {**status, "status": "skipped_existing"}
    if job.kind == "checkpoint":
        checkpoint_dir = _local(root, job.checkpoint_dir)
        missing = [name for name in ("state.pt", "config.yaml")
                   if not (checkpoint_dir / name).is_file()]
        if missing:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(log_path, "Skipped: checkpoint is missing {}\n".format(", ".join(missing)))
            return {**status, "status": "skipped_missing_checkpoint", "missing": missing}

    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = _utc_now()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write("started_at: {}\ncommand: {}\n".format(started, _command_text(command)))
            log.flush()
            completed = subprocess.run(
                [str(value) for value in command], cwd=str(root), stdout=log,
                stderr=subprocess.STDOUT, check=False,
            )
            log.write("\nfinished_at: {}\nreturn_code: {}\n".format(
                _utc_now(), completed.returncode
            ))
        if completed.returncode != 0:
            raise RuntimeError("bo_experiment exited with code {}".format(completed.returncode))
        if not _valid_result(result_path):
            raise RuntimeError("bo_experiment did not produce a valid result JSON")
        if failure_path.exists():
            failure_path.unlink()
        return {**status, "status": "completed", "started_at": started,
                "completed_at": _utc_now()}
    except BaseException as error:
        failure = {
            **status, "status": "failed", "failed_at": _utc_now(),
            "error_type": type(error).__name__, "error": str(error),
        }
        _atomic_write(failure_path, json.dumps(failure, indent=2, sort_keys=True) + "\n")
        return failure


def run_jobs(
    jobs: Sequence[EvaluationJob],
    repository_root: Path = REPOSITORY_ROOT,
    device: str = "cpu",
    dry_run: bool = False,
    force: bool = False,
    max_workers: int = 1,
) -> List[Dict[str, Any]]:
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    if max_workers == 1:
        return [run_job(job, repository_root, device, dry_run, force) for job in jobs]
    statuses = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_jobs = {
            pool.submit(run_job, job, repository_root, device, dry_run, force): job
            for job in jobs
        }
        for future in as_completed(future_jobs):
            statuses.append(future.result())
    order = {job.job_id: index for index, job in enumerate(jobs)}
    return sorted(statuses, key=lambda item: order[item["job_id"]])


def plot_bundles(
    jobs: Sequence[EvaluationJob],
    statuses: Sequence[Mapping[str, Any]],
    repository_root: Path = REPOSITORY_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    dry_run: bool = False,
    summary_only: bool = False,
) -> List[Dict[str, Any]]:
    root = Path(repository_root)
    status_by_id = {status["job_id"]: status for status in statuses}
    plots = []
    for bundle_id, members in sorted(result_bundles(jobs).items()):
        available = list(members) if dry_run else [
            job for job in members
            if status_by_id[job.job_id]["status"] in {"completed", "skipped_existing"}
        ]
        output_dir = root / output_root / "plots" / bundle_id
        command = [sys.executable, root / "src" / "bo_plot.py"]
        command.extend(_local(root, job.result_path) for job in available)
        command.extend(("--out-dir", output_dir))
        if summary_only:
            command.append("--summary-only")
        row = {"bundle_id": bundle_id, "result_count": len(available),
               "output_dir": str(output_dir.relative_to(root)),
               "command": [str(value) for value in command]}
        if dry_run:
            plots.append({**row, "status": "planned"})
            continue
        if not available:
            plots.append({**row, "status": "skipped_no_results"})
            continue
        log_path = output_dir / "plot.log"
        output_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                [str(value) for value in command], cwd=str(root), stdout=log,
                stderr=subprocess.STDOUT, check=False,
            )
        plots.append({**row, "status": "completed" if completed.returncode == 0 else "failed",
                      "return_code": completed.returncode,
                      "log_path": str(log_path.relative_to(root))})
    return plots


def write_result_manifest(
    jobs: Sequence[EvaluationJob],
    statuses: Sequence[Mapping[str, Any]],
    plots: Sequence[Mapping[str, Any]],
    repository_root: Path = REPOSITORY_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    destination = Path(repository_root) / output_root / "result_manifest.json"
    document = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "job_count": len(jobs),
        "status_counts": {
            status: sum(row["status"] == status for row in statuses)
            for status in sorted({row["status"] for row in statuses})
        },
        "dependence_protocol_note": (
            "matched/shift labels compare feature/noise dependence schedules only; "
            "train and test amplitude SNR are recorded separately"
        ),
        "jobs": list(statuses),
        "plots": list(plots),
    }
    _atomic_write(destination, json.dumps(document, indent=2, sort_keys=True) + "\n")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", choices=sorted(GROUPS),
                        help="Matrix group; repeat to combine (default: canonical)")
    parser.add_argument("--matrix-manifest", action="append", type=Path, default=[],
                        help="Training manifest from bo_matrix.write_manifest; repeatable")
    parser.add_argument("--experiment-id", action="append", default=[],
                        help="Keep only selected semantic experiment IDs")
    parser.add_argument("--protocol", choices=("matched", "shift", "all"), default="matched")
    parser.add_argument("--snr", action="append", type=float,
                        help="Override dense test SNRs; repeatable")
    parser.add_argument("--match-train-snr", action="store_true",
                        help="Evaluate each checkpoint only at its own training SNR")
    parser.add_argument("--linear-probe", action="store_true",
                        help="Run held-out linear-surrogate probes for linear BO diagnostics")
    parser.add_argument(
        "--boundary-suggestions", action="append", type=Path, default=[],
        help="Use suggested follow-up SNRs from bo_plot.py; repeatable",
    )
    parser.add_argument("--eval-seed", action="append", type=int,
                        help="Override evaluation seeds; repeatable")
    parser.add_argument("--k-over-d", action="append", type=float,
                        help="Override context ratios; repeatable")
    parser.add_argument("--shift-rho", action="append", type=float,
                        help="Override stationary shift rho grid; repeatable")
    parser.add_argument("--n-eval", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.snr and args.boundary_suggestions:
        raise ValueError("use either --snr or --boundary-suggestions, not both")
    if args.match_train_snr and (args.snr or args.boundary_suggestions):
        raise ValueError("--match-train-snr cannot be combined with --snr or --boundary-suggestions")
    group_names = args.group or (["canonical"] if not args.matrix_manifest else [])
    specs = []
    for group in group_names:
        specs.extend(expand_group(group))
    for manifest in args.matrix_manifest:
        specs.extend(load_matrix_manifest(manifest))
    unique_specs = {spec.experiment_id: spec for spec in specs}
    if args.experiment_id:
        selected = set(args.experiment_id)
        unknown = selected - set(unique_specs)
        if unknown:
            raise ValueError("unknown experiment IDs: {}".format(sorted(unknown)))
        unique_specs = {key: value for key, value in unique_specs.items() if key in selected}
    protocols = ("matched", "shift") if args.protocol == "all" else (args.protocol,)
    test_snrs = (
        args.snr
        or (
            tuple(sorted({
                snr
                for path in args.boundary_suggestions
                for snr in load_boundary_suggestions(path)
            }))
            if args.boundary_suggestions
            else DENSE_TEST_SNRS
        )
    )
    jobs = build_evaluation_jobs(
        unique_specs.values(), protocols=protocols,
        test_snrs=test_snrs,
        eval_seeds=args.eval_seed or DEFAULT_EVAL_SEEDS,
        k_over_d=args.k_over_d or DIMENSION_K_OVER_D,
        shift_test_rhos=args.shift_rho or DENSE_TEST_RHOS,
        n_eval=args.n_eval, batch_size=args.batch_size,
        include_baselines=not args.no_baselines, output_root=args.output_root,
        match_train_snr=args.match_train_snr,
        linear_probe=args.linear_probe,
    )
    statuses = run_jobs(
        jobs, REPOSITORY_ROOT, args.device, args.dry_run, args.force, args.max_workers
    )
    plots = [] if args.no_plot else plot_bundles(
        jobs, statuses, REPOSITORY_ROOT, args.output_root, args.dry_run, args.summary_only
    )
    manifest = write_result_manifest(
        jobs, statuses, plots, REPOSITORY_ROOT, args.output_root
    )
    print("Wrote {} jobs to {}".format(len(jobs), manifest), flush=True)
    return 1 if any(row["status"] == "failed" for row in statuses + plots) else 0


if __name__ == "__main__":
    raise SystemExit(main())
