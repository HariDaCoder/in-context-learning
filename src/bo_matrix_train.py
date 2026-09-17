"""Planning, concurrent training, and throughput benchmarks for BO matrices.

The functions here are intentionally environment-neutral.  They receive a
device string from the caller, invoke the current Python interpreter, and do
not contain scheduler or hardware-specific policy.  Importing this module has
no side effects and never starts training.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import yaml

from bo_architecture import ArchitectureSpec
from bo_matrix import (
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_MATRIX_ROOT,
    ExistingCheckpoint,
    ExperimentSpec,
    LockUnavailable,
    REPOSITORY_ROOT,
    RunRegistry,
    expand_group,
    load_experiment_manifest,
    write_manifest,
    write_training_config,
)


SOURCE_ROOT_NAME = "src"
MANIFEST_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, document: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{}.{}.tmp".format(path.name, uuid.uuid4().hex))
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _relative(path: Path, repository_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(repository_root).resolve()).as_posix()
    except ValueError as error:
        raise ValueError("launcher artifact lies outside the repository: {}".format(path)) from error


def _command_text(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def _validate_max_concurrent(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("max_concurrent must be a positive integer")
    return value


def recommended_concurrency(
    group_name: str,
    repository_root: Path = REPOSITORY_ROOT,
    matrix_root: Path = DEFAULT_MATRIX_ROOT,
) -> Optional[int]:
    """Return the latest successful benchmark recommendation, if present."""

    path = (
        Path(repository_root)
        / matrix_root
        / "benchmarks"
        / "recommendations"
        / "{}.json".format(group_name)
    )
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("group") != group_name:
        raise ValueError("benchmark recommendation group mismatch: {}".format(path))
    value = document.get("recommended_max_concurrent")
    return _validate_max_concurrent(value)


def _validate_run_label(value: str) -> str:
    if not isinstance(value, str) or not value or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in value
    ):
        raise ValueError("run label must be a nonempty path-safe identifier")
    return value


def _runtime_profile(device: str) -> str:
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a nonempty string")
    return hashlib.sha256(device.encode("utf-8")).hexdigest()[:12]


def _runtime_config_path(
    spec: ExperimentSpec,
    device: str,
    matrix_root: Path,
) -> Path:
    return (
        Path(matrix_root)
        / "runtime_configs"
        / spec.experiment_id
        / "{}.yaml".format(_runtime_profile(device))
    )


def _write_runtime_config(
    spec: ExperimentSpec,
    device: str,
    repository_root: Path,
    matrix_root: Path,
) -> Path:
    relative_path = _runtime_config_path(spec, device, matrix_root)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("matrix_root must be repository-relative")
    destination = Path(repository_root) / relative_path
    config = spec.to_training_config()
    config["training"]["device"] = device
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("{}.{}.tmp".format(destination.name, uuid.uuid4().hex))
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _reconcile_completion(
    registry: RunRegistry, spec: ExperimentSpec, repository_root: Path
) -> None:
    """Recover the tiny crash window after train.py wrote completed.json."""

    checkpoint = Path(repository_root) / spec.checkpoint_path
    completed_path = checkpoint.parent / "completed.json"
    if not checkpoint.is_file() or not completed_path.is_file():
        return
    with completed_path.open(encoding="utf-8") as handle:
        completed = json.load(handle)
    if completed.get("experiment_id") != spec.experiment_id:
        raise ValueError("completion metadata ID mismatch: {}".format(completed_path))
    if registry.read_state(spec).get("status") != "completed":
        registry.mark_completed(spec)


@dataclass(frozen=True)
class PlanItem:
    spec: ExperimentSpec
    action: str
    reason: str
    command: Tuple[str, ...]
    config_path: Path
    log_path: Path

    def as_dict(self, repository_root: Path) -> Dict[str, Any]:
        row = self.spec.manifest_row()
        row.update(
            action=self.action,
            reason=self.reason,
            command=list(self.command),
            config_path=_relative(self.config_path, repository_root),
            log_path=_relative(self.log_path, repository_root),
        )
        return row


@dataclass(frozen=True)
class GroupPlan:
    group: str
    device: str
    precision: str
    max_concurrent: int
    resume: bool
    items: Tuple[PlanItem, ...]
    manifest_json: Optional[Path]
    manifest_csv: Optional[Path]
    summary_path: Path

    @property
    def counts(self) -> Dict[str, int]:
        counts = {"total": len(self.items), "start": 0, "resume": 0, "skip": 0, "blocked": 0}
        for item in self.items:
            counts[item.action] = counts.get(item.action, 0) + 1
        return counts

    def as_dict(self, repository_root: Path) -> Dict[str, Any]:
        return {
            "kind": "plan",
            "created_at": _utc_now(),
            "group": self.group,
            "device": self.device,
            "precision": self.precision,
            "max_concurrent": self.max_concurrent,
            "resume": self.resume,
            "counts": self.counts,
            "manifest_json": (
                _relative(self.manifest_json, repository_root) if self.manifest_json else None
            ),
            "manifest_csv": (
                _relative(self.manifest_csv, repository_root) if self.manifest_csv else None
            ),
            "experiments": [item.as_dict(repository_root) for item in self.items],
        }


@dataclass(frozen=True)
class RunResult:
    experiment_id: str
    status: str
    action: str
    reason: str
    returncode: Optional[int]
    wall_time_seconds: float
    checkpoint_path: str
    log_path: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "action": self.action,
            "reason": self.reason,
            "returncode": self.returncode,
            "wall_time_seconds": self.wall_time_seconds,
            "checkpoint_path": self.checkpoint_path,
            "log_path": self.log_path,
        }


@dataclass(frozen=True)
class TrainingSummary:
    group: str
    dry_run: bool
    device: str
    precision: str
    max_concurrent: int
    wall_time_seconds: float
    results: Tuple[RunResult, ...]
    summary_path: Path

    @property
    def counts(self) -> Dict[str, int]:
        counts = {"total": len(self.results)}
        for result in self.results:
            counts[result.status] = counts.get(result.status, 0) + 1
        return counts

    @property
    def success(self) -> bool:
        return not any(result.status in {"failed", "blocked", "locked"} for result in self.results)

    def as_dict(self, repository_root: Path) -> Dict[str, Any]:
        return {
            "kind": "training",
            "created_at": _utc_now(),
            "group": self.group,
            "dry_run": self.dry_run,
            "device": self.device,
            "precision": self.precision,
            "max_concurrent": self.max_concurrent,
            "wall_time_seconds": self.wall_time_seconds,
            "counts": self.counts,
            "success": self.success,
            "summary_path": _relative(self.summary_path, repository_root),
            "results": [result.as_dict() for result in self.results],
        }


def _apply_precision(
    specs: Iterable[ExperimentSpec], precision: Optional[str]
) -> Tuple[ExperimentSpec, ...]:
    if precision is None:
        return tuple(specs)
    # Precision changes numerical training and therefore belongs in the stable
    # experiment ID. dataclasses.replace invokes ExperimentSpec validation.
    return tuple(replace(spec, precision=precision) for spec in specs)


def _plan_specs(
    group_name: str,
    specs: Sequence[ExperimentSpec],
    device: str,
    max_concurrent: int,
    resume: bool,
    repository_root: Path,
    matrix_root: Path,
    interpreter: str,
    write_group_manifest: bool,
) -> GroupPlan:
    root = Path(repository_root).resolve()
    source_root = root / SOURCE_ROOT_NAME
    if not source_root.is_dir():
        raise FileNotFoundError("source directory does not exist: {}".format(source_root))
    _validate_max_concurrent(max_concurrent)
    _runtime_profile(device)
    registry = RunRegistry(root, matrix_root=matrix_root)

    manifest_json = None
    manifest_csv = None
    if write_group_manifest:
        manifest_json, manifest_csv = write_manifest(
            group_name, specs, repository_root=root, matrix_root=matrix_root
        )

    items = []
    log_root = root / matrix_root / "logs" / group_name
    for spec in specs:
        # The canonical config is useful for audit/reproduction. The runtime
        # config has a device-specific path so independent planners cannot
        # overwrite the config of a process that is starting on another device.
        write_training_config(spec, repository_root=root)
        config_path = _write_runtime_config(spec, device, root, matrix_root)
        _reconcile_completion(registry, spec, root)
        decision = registry.decision(spec, resume=resume)
        config_argument = os.path.relpath(str(config_path), str(source_root))
        command = (str(interpreter), "train.py", "--config", config_argument)
        items.append(
            PlanItem(
                spec=spec,
                action=decision.action,
                reason=decision.reason,
                command=command,
                config_path=config_path,
                log_path=log_root / "{}.log".format(spec.experiment_id),
            )
        )

    precision_values = {item.spec.precision for item in items}
    plan_precision = next(iter(precision_values)) if len(precision_values) == 1 else "mixed"
    summary_path = root / matrix_root / "summaries" / "{}_plan.json".format(group_name)
    plan = GroupPlan(
        group=group_name,
        device=device,
        precision=plan_precision,
        max_concurrent=max_concurrent,
        resume=resume,
        items=tuple(items),
        manifest_json=manifest_json,
        manifest_csv=manifest_csv,
        summary_path=summary_path,
    )
    _atomic_json(summary_path, plan.as_dict(root))
    return plan


def plan_group(
    group_name: str,
    device: str = "auto",
    precision: Optional[str] = None,
    max_concurrent: int = 1,
    resume: bool = True,
    repository_root: Path = REPOSITORY_ROOT,
    matrix_root: Path = DEFAULT_MATRIX_ROOT,
    interpreter: Optional[str] = None,
) -> GroupPlan:
    """Expand a group, materialize configs/manifests, and report run actions."""

    specs = _apply_precision(expand_group(group_name), precision)
    return _plan_specs(
        group_name=group_name,
        specs=specs,
        device=device,
        max_concurrent=max_concurrent,
        resume=resume,
        repository_root=Path(repository_root),
        matrix_root=Path(matrix_root),
        interpreter=interpreter or sys.executable,
        write_group_manifest=True,
    )


def _manifest_plan_name(specs: Sequence[ExperimentSpec]) -> str:
    groups = sorted({spec.group for spec in specs})
    readable = groups[0] if len(groups) == 1 else "combined"
    readable = "".join(
        character if character.isalnum() or character in "_.-" else "_"
        for character in readable
    ).strip("._-") or "experiments"
    digest = hashlib.sha256(
        "\n".join(sorted(spec.experiment_id for spec in specs)).encode("utf-8")
    ).hexdigest()[:12]
    return "manifest_{}_{}".format(readable[:48], digest)


def plan_manifest(
    manifest_path: Path,
    device: str = "auto",
    precision: Optional[str] = None,
    max_concurrent: int = 1,
    resume: bool = True,
    repository_root: Path = REPOSITORY_ROOT,
    matrix_root: Path = DEFAULT_MATRIX_ROOT,
    interpreter: Optional[str] = None,
) -> GroupPlan:
    """Plan verified experiments from a matrix or parameter-match manifest."""

    specs = _apply_precision(load_experiment_manifest(manifest_path), precision)
    return _plan_specs(
        group_name=_manifest_plan_name(specs),
        specs=specs,
        device=device,
        max_concurrent=max_concurrent,
        resume=resume,
        repository_root=Path(repository_root),
        matrix_root=Path(matrix_root),
        interpreter=interpreter or sys.executable,
        write_group_manifest=False,
    )


def _static_result(item: PlanItem, status: str, repository_root: Path) -> RunResult:
    return RunResult(
        experiment_id=item.spec.experiment_id,
        status=status,
        action=item.action,
        reason=item.reason,
        returncode=None,
        wall_time_seconds=0.0,
        checkpoint_path=item.spec.checkpoint_path.as_posix(),
        log_path=_relative(item.log_path, repository_root),
    )


def _run_item(
    item: PlanItem,
    registry: RunRegistry,
    repository_root: Path,
    resume: bool,
    runner: Callable[..., Any],
) -> RunResult:
    start = time.perf_counter()
    item.log_path.parent.mkdir(parents=True, exist_ok=True)
    returncode = None
    try:
        with registry.claim(item.spec, resume=resume) as decision:
            if decision.action == "skip":
                return _static_result(item, "skipped", repository_root)
            with item.log_path.open("a", encoding="utf-8", newline="") as log:
                log.write("\n[{}] $ {}\n".format(_utc_now(), _command_text(item.command)))
                log.flush()
                completed = runner(
                    list(item.command),
                    cwd=str(Path(repository_root) / SOURCE_ROOT_NAME),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
                returncode = getattr(completed, "returncode", 0)
        status = "completed"
        reason = "training process completed successfully"
        action = decision.action
    except LockUnavailable as error:
        status, action, reason = "locked", item.action, str(error)
    except ExistingCheckpoint as error:
        status, action, reason = "blocked", item.action, str(error)
    except subprocess.CalledProcessError as error:
        status, action, reason = "failed", item.action, str(error)
        returncode = error.returncode
    except Exception as error:
        status, action, reason = "failed", item.action, "{}: {}".format(type(error).__name__, error)
    return RunResult(
        experiment_id=item.spec.experiment_id,
        status=status,
        action=action,
        reason=reason,
        returncode=returncode,
        wall_time_seconds=time.perf_counter() - start,
        checkpoint_path=item.spec.checkpoint_path.as_posix(),
        log_path=_relative(item.log_path, repository_root),
    )


def _execute_plan(
    plan: GroupPlan,
    dry_run: bool,
    repository_root: Path,
    matrix_root: Path,
    runner: Callable[..., Any],
    summary_path: Path,
) -> TrainingSummary:
    root = Path(repository_root).resolve()
    start = time.perf_counter()
    fixed_results = {}
    runnable = []
    for item in plan.items:
        if item.action == "skip":
            fixed_results[item.spec.experiment_id] = _static_result(item, "skipped", root)
        elif item.action == "blocked":
            fixed_results[item.spec.experiment_id] = _static_result(item, "blocked", root)
        elif dry_run:
            fixed_results[item.spec.experiment_id] = _static_result(item, "planned", root)
        else:
            runnable.append(item)

    if runnable:
        registry = RunRegistry(root, matrix_root=matrix_root)
        with ThreadPoolExecutor(max_workers=plan.max_concurrent) as executor:
            futures = {
                executor.submit(
                    _run_item,
                    item,
                    registry,
                    root,
                    plan.resume,
                    runner,
                ): item
                for item in runnable
            }
            for future in as_completed(futures):
                result = future.result()
                fixed_results[result.experiment_id] = result

    results = tuple(fixed_results[item.spec.experiment_id] for item in plan.items)
    summary = TrainingSummary(
        group=plan.group,
        dry_run=dry_run,
        device=plan.device,
        precision=plan.precision,
        max_concurrent=plan.max_concurrent,
        wall_time_seconds=time.perf_counter() - start,
        results=results,
        summary_path=summary_path,
    )
    _atomic_json(summary_path, summary.as_dict(root))
    return summary


def train_group(
    group_name: str,
    device: str = "auto",
    precision: Optional[str] = None,
    max_concurrent: int = 1,
    dry_run: bool = False,
    resume: bool = True,
    repository_root: Path = REPOSITORY_ROOT,
    matrix_root: Path = DEFAULT_MATRIX_ROOT,
    interpreter: Optional[str] = None,
    runner: Optional[Callable[..., Any]] = None,
) -> TrainingSummary:
    """Train a neutral group with per-experiment locks and bounded concurrency."""

    root = Path(repository_root).resolve()
    plan = plan_group(
        group_name=group_name,
        device=device,
        precision=precision,
        max_concurrent=max_concurrent,
        resume=resume,
        repository_root=root,
        matrix_root=matrix_root,
        interpreter=interpreter,
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    summary_path = root / matrix_root / "summaries" / group_name / "{}.json".format(run_id)
    return _execute_plan(
        plan=plan,
        dry_run=dry_run,
        repository_root=root,
        matrix_root=Path(matrix_root),
        runner=runner or subprocess.run,
        summary_path=summary_path,
    )


def train_manifest(
    manifest_path: Path,
    device: str = "auto",
    precision: Optional[str] = None,
    max_concurrent: int = 1,
    dry_run: bool = False,
    resume: bool = True,
    repository_root: Path = REPOSITORY_ROOT,
    matrix_root: Path = DEFAULT_MATRIX_ROOT,
    interpreter: Optional[str] = None,
    runner: Optional[Callable[..., Any]] = None,
) -> TrainingSummary:
    """Train verified experiments from a persisted manifest."""

    root = Path(repository_root).resolve()
    plan = plan_manifest(
        manifest_path=manifest_path,
        device=device,
        precision=precision,
        max_concurrent=max_concurrent,
        resume=resume,
        repository_root=root,
        matrix_root=matrix_root,
        interpreter=interpreter,
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    summary_path = root / matrix_root / "summaries" / plan.group / "{}.json".format(run_id)
    return _execute_plan(
        plan=plan,
        dry_run=dry_run,
        repository_root=root,
        matrix_root=Path(matrix_root),
        runner=runner or subprocess.run,
        summary_path=summary_path,
    )


@dataclass(frozen=True)
class BenchmarkExperimentSpec(ExperimentSpec):
    """A short-run spec whose ID and directory cannot collide with main runs."""

    benchmark_run_id: str = "benchmark"
    benchmark_concurrency: int = 1
    benchmark_slot: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.benchmark_run_id or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for character in self.benchmark_run_id
        ):
            raise ValueError("benchmark_run_id must be a path-safe identifier")
        _validate_max_concurrent(self.benchmark_concurrency)
        if self.benchmark_slot < 0:
            raise ValueError("benchmark_slot must be nonnegative")

    def identity_payload(self) -> Dict[str, Any]:
        payload = super().identity_payload()
        payload["benchmark"] = {
            "run_id": self.benchmark_run_id,
            "concurrency": self.benchmark_concurrency,
            "slot": self.benchmark_slot,
        }
        return payload

    @property
    def checkpoint_dir(self) -> Path:
        return (
            DEFAULT_CHECKPOINT_ROOT
            / "benchmarks"
            / self.benchmark_run_id
            / "concurrency_{}".format(self.benchmark_concurrency)
            / self.experiment_id
        )

    @property
    def config_path(self) -> Path:
        return (
            DEFAULT_MATRIX_ROOT
            / "benchmarks"
            / self.benchmark_run_id
            / "configs"
            / "{}.yaml".format(self.experiment_id)
        )

    def to_training_config(self) -> Dict[str, Any]:
        config = super().to_training_config()
        config["out_dir"] = "../{}".format(self.checkpoint_dir.parent.as_posix())
        return config

    def manifest_row(self) -> Dict[str, Any]:
        row = super().manifest_row()
        row.update(
            benchmark_run_id=self.benchmark_run_id,
            benchmark_concurrency=self.benchmark_concurrency,
            benchmark_slot=self.benchmark_slot,
        )
        return row


def _benchmark_spec(
    spec: ExperimentSpec,
    run_id: str,
    concurrency: int,
    slot: int,
    steps: int,
    precision: Optional[str],
) -> BenchmarkExperimentSpec:
    adjusted = replace(spec, training_steps=steps)
    if precision is not None:
        adjusted = replace(adjusted, precision=precision)
    values = {field.name: getattr(adjusted, field.name) for field in fields(ExperimentSpec)}
    values["group"] = "benchmark_{}".format(spec.group)
    return BenchmarkExperimentSpec(
        **values,
        benchmark_run_id=run_id,
        benchmark_concurrency=concurrency,
        benchmark_slot=slot,
    )


def _read_completion(spec: ExperimentSpec, repository_root: Path) -> Dict[str, Any]:
    path = Path(repository_root) / spec.checkpoint_dir / "completed.json"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("experiment_id") != spec.experiment_id:
        raise ValueError("completion metadata ID mismatch: {}".format(path))
    return document


def benchmark_group(
    group_name: str,
    device: str = "auto",
    precision: Optional[str] = None,
    concurrency_values: Sequence[int] = (1, 2, 4),
    steps: int = 2000,
    dry_run: bool = False,
    resume: bool = True,
    repository_root: Path = REPOSITORY_ROOT,
    matrix_root: Path = DEFAULT_MATRIX_ROOT,
    interpreter: Optional[str] = None,
    runner: Optional[Callable[..., Any]] = None,
    benchmark_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Benchmark independent-process concurrency using isolated short runs."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    values = tuple(concurrency_values)
    if not values:
        raise ValueError("concurrency_values must be nonempty")
    for value in values:
        _validate_max_concurrent(value)
    if len(set(values)) != len(values):
        raise ValueError("concurrency_values must not contain duplicates")

    root = Path(repository_root).resolve()
    run_id = benchmark_run_id or (
        "bench_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    base_specs = expand_group(group_name)
    if len(base_specs) < max(values):
        raise ValueError(
            "group {} has {} experiments, fewer than requested concurrency {}".format(
                group_name, len(base_specs), max(values)
            )
        )

    trials = []
    run = runner or subprocess.run
    for concurrency in values:
        specs = tuple(
            _benchmark_spec(
                base_specs[slot], run_id, concurrency, slot, steps, precision
            )
            for slot in range(concurrency)
        )
        trial_name = "benchmark_{}_concurrency_{}".format(group_name, concurrency)
        plan = _plan_specs(
            group_name=trial_name,
            specs=specs,
            device=device,
            max_concurrent=concurrency,
            resume=resume,
            repository_root=root,
            matrix_root=Path(matrix_root),
            interpreter=interpreter or sys.executable,
            write_group_manifest=False,
        )
        trial_summary_path = (
            root
            / matrix_root
            / "benchmarks"
            / run_id
            / "concurrency_{}".format(concurrency)
            / "training_summary.json"
        )
        wall_start = time.perf_counter()
        training = _execute_plan(
            plan=plan,
            dry_run=dry_run,
            repository_root=root,
            matrix_root=Path(matrix_root),
            runner=run,
            summary_path=trial_summary_path,
        )
        wall_time = time.perf_counter() - wall_start
        completed_ids = {
            result.experiment_id
            for result in training.results
            if result.status == "completed"
        }
        completions = (
            []
            if dry_run
            else [
                _read_completion(spec, root)
                for spec in specs
                if spec.experiment_id in completed_ids
            ]
        )
        completed_metadata = [document for document in completions if document]
        total_steps = sum(
            int(document.get("steps_completed_this_invocation", 0))
            for document in completed_metadata
        )
        memory_values = [
            document.get("max_cuda_memory_bytes") for document in completed_metadata
        ]
        memory_values = [value for value in memory_values if isinstance(value, (int, float))]
        trials.append(
            {
                "max_concurrent": concurrency,
                "experiment_ids": [spec.experiment_id for spec in specs],
                "checkpoint_dirs": [spec.checkpoint_dir.as_posix() for spec in specs],
                "dry_run": dry_run,
                "wall_time_seconds": wall_time,
                "total_steps_completed": total_steps,
                "aggregate_steps_per_second": (
                    total_steps / wall_time if total_steps and wall_time > 0 else None
                ),
                "sum_max_cuda_memory_bytes": sum(memory_values) if memory_values else None,
                "completed_processes": len(completed_metadata),
                "counts": training.counts,
                "success": training.success,
                "training_summary": _relative(training.summary_path, root),
            }
        )

    eligible = [
        trial
        for trial in trials
        if trial["success"] and trial["aggregate_steps_per_second"] is not None
    ]
    recommended = (
        max(eligible, key=lambda item: item["aggregate_steps_per_second"])["max_concurrent"]
        if eligible
        else None
    )
    output_path = root / matrix_root / "benchmarks" / run_id / "summary.json"
    document = {
        "kind": "concurrency_benchmark",
        "created_at": _utc_now(),
        "benchmark_run_id": run_id,
        "group": group_name,
        "device": device,
        "precision": precision or base_specs[0].precision,
        "steps_per_process": steps,
        "dry_run": dry_run,
        "recommended_max_concurrent": recommended,
        "trials": trials,
        "summary_path": _relative(output_path, root),
    }
    recommendation_path = (
        root
        / matrix_root
        / "benchmarks"
        / "recommendations"
        / "{}.json".format(group_name)
    )
    document["recommendation_path"] = _relative(recommendation_path, root)
    _atomic_json(output_path, document)
    if recommended is not None and not dry_run:
        _atomic_json(
            recommendation_path,
            {
                "kind": "concurrency_recommendation",
                "created_at": _utc_now(),
                "group": group_name,
                "benchmark_run_id": run_id,
                "recommended_max_concurrent": recommended,
                "benchmark_summary": _relative(output_path, root),
            },
        )
    return document
