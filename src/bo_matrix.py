"""Environment-neutral experiment matrices for benign-overfitting runs.

This module deliberately contains no scheduler, host, GPU, or collaborator
names.  It describes scientific experiments only.  Launchers can map the
neutral groups to compute resources outside the repository.

The stable training ID is computed from the semantic training configuration;
the group name and human-readable architecture/regime labels are excluded.
Consequently, an experiment shared by two groups resolves to the same
checkpoint directory and can be reused safely.
"""

from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import socket
import time
import uuid
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

from bo_architecture import ArchitectureSpec, architecture_specs


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MATRIX_SCHEMA_VERSION = 1
DEFAULT_MATRIX_ROOT = Path("results/bo_matrix")
DEFAULT_CHECKPOINT_ROOT = Path("models/bo_matrix")
SNR_DEFINITION = (
    "amplitude SNR: signal standard deviation / label-noise standard deviation; "
    "with w ~ N(0, I_d/d) and x ~ N(0, I_d), signal variance is one and "
    "label_noise_std = 1 / amplitude_snr"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:20]
    return "{}_{}".format(prefix, digest)


def _relative_posix(path: Path) -> str:
    if path.is_absolute():
        raise ValueError("matrix paths must be repository-relative: {}".format(path))
    if ".." in path.parts:
        raise ValueError("matrix paths may not escape the repository: {}".format(path))
    return path.as_posix()


def _atomic_write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.read_text(encoding="utf-8") == contents:
            return
    except FileNotFoundError:
        pass
    temporary = path.with_name("{}.{}.tmp".format(path.name, uuid.uuid4().hex))
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(contents)
        for attempt in range(5):
            try:
                os.replace(str(temporary), str(path))
                return
            except PermissionError:
                # Concurrent planners may materialize the same semantic config.
                # On Windows, replacing that identical destination can briefly
                # fail while the other process still has the file open.
                try:
                    if path.read_text(encoding="utf-8") == contents:
                        return
                except FileNotFoundError:
                    pass
                if attempt == 4:
                    raise
                time.sleep(0.01 * (attempt + 1))
    finally:
        if temporary.exists():
            temporary.unlink()


def expand_axes(
    base: Mapping[str, Any], axes: Mapping[str, Sequence[Any]]
) -> Tuple[Dict[str, Any], ...]:
    """Return the Cartesian product of named axes in deterministic key order."""

    names = tuple(sorted(axes))
    values = []
    for name in names:
        axis_values = tuple(axes[name])
        if not axis_values:
            return ()
        values.append(axis_values)
    rows = []
    for combination in itertools.product(*values):
        row = dict(base)
        row.update(zip(names, combination))
        rows.append(row)
    return tuple(rows)


STANDARD_ARCHITECTURE = ArchitectureSpec(256, 12, 8, sweep_family="standard")


def _architecture_variants() -> Tuple[ArchitectureSpec, ...]:
    # Use the shared architecture module as the single source of sweep
    # definitions. H=8/head_dim=32 and L=12 describe the same instantiated
    # standard model, so collapse shapes here to count actual checkpoints.
    unique = {}
    for architecture in architecture_specs():
        key = _canonical_json(architecture.model_dict())
        unique.setdefault(key, architecture)
    return tuple(unique.values())


ARCHITECTURE_VARIANTS = _architecture_variants()


def architecture_memberships(architecture: ArchitectureSpec) -> Tuple[Dict[str, Any], ...]:
    """Return every declared sweep family containing this instantiated shape."""

    shape = architecture.model_dict()
    matches = []
    for candidate in architecture_specs():
        if candidate.model_dict() == shape:
            matches.append(
                {
                    "sweep_family": candidate.sweep_family,
                    "n_embd": candidate.n_embd,
                    "n_layer": candidate.n_layer,
                    "n_head": candidate.n_head,
                    "head_dim": candidate.head_dim,
                }
            )
    if not matches:
        matches.append(
            {
                "sweep_family": architecture.sweep_family,
                "n_embd": architecture.n_embd,
                "n_layer": architecture.n_layer,
                "n_head": architecture.n_head,
                "head_dim": architecture.head_dim,
            }
        )
    return tuple(matches)


@dataclass(frozen=True)
class ExperimentSpec:
    """One deterministic training experiment."""

    group: str
    regime: str
    train_seed: int
    d: int
    train_rho_x: float
    train_rho_e: float
    architecture: ArchitectureSpec = STANDARD_ARCHITECTURE
    max_context: int = 80
    train_snr: float = 2.0
    rho_x_after: Optional[float] = None
    feature_change_point: Optional[int] = None
    rho_e_after: Optional[float] = None
    noise_change_point: Optional[int] = None
    batch_size: int = 64
    training_steps: int = 500001
    data_device: str = "model"
    precision: str = "float32"

    def __post_init__(self) -> None:
        if not self.group or not self.regime:
            raise ValueError("group and regime must be non-empty")
        if (
            isinstance(self.train_seed, bool)
            or not isinstance(self.train_seed, int)
            or self.train_seed < 0
        ):
            raise ValueError("train_seed must be a nonnegative integer")
        for name, value in (
            ("d", self.d),
            ("max_context", self.max_context),
            ("batch_size", self.batch_size),
            ("training_steps", self.training_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))
        if not math.isfinite(self.train_snr) or self.train_snr <= 0:
            raise ValueError("train_snr must be positive and finite")
        if self.data_device not in {"cpu", "model"}:
            raise ValueError("data_device must be 'cpu' or 'model'")
        if self.precision not in {"float32", "float16", "bfloat16"}:
            raise ValueError("unsupported precision: {}".format(self.precision))
        for name, rho in (
            ("train_rho_x", self.train_rho_x),
            ("train_rho_e", self.train_rho_e),
            ("rho_x_after", self.rho_x_after),
            ("rho_e_after", self.rho_e_after),
        ):
            if rho is not None and (not math.isfinite(rho) or abs(rho) >= 1):
                raise ValueError("{} must be finite with absolute value below one".format(name))
        if (self.rho_x_after is None) != (self.feature_change_point is None):
            raise ValueError("rho_x_after and feature_change_point must be set together")
        if (self.rho_e_after is None) != (self.noise_change_point is None):
            raise ValueError("rho_e_after and noise_change_point must be set together")
        for name, point in (
            ("feature_change_point", self.feature_change_point),
            ("noise_change_point", self.noise_change_point),
        ):
            if point is not None and not 0 <= point <= self.max_context:
                raise ValueError("{} must lie within the context".format(name))

    @property
    def label_noise_std(self) -> float:
        return 1.0 / self.train_snr

    def identity_payload(self) -> Dict[str, Any]:
        """Semantic fields only; labels, paths, and matrix membership are absent."""

        return {
            "schema_version": MATRIX_SCHEMA_VERSION,
            "architecture": self.architecture.model_dict(),
            "batch_size": self.batch_size,
            "d": self.d,
            "data_device": self.data_device,
            "feature_change_point": self.feature_change_point,
            "max_context": self.max_context,
            "noise_change_point": self.noise_change_point,
            "rho_e_after": self.rho_e_after,
            "rho_x_after": self.rho_x_after,
            "precision": self.precision,
            "train_rho_e": self.train_rho_e,
            "train_rho_x": self.train_rho_x,
            "train_seed": self.train_seed,
            "train_snr": self.train_snr,
            "training_steps": self.training_steps,
        }

    @property
    def experiment_id(self) -> str:
        return _stable_id("bo", self.identity_payload())

    @property
    def checkpoint_dir(self) -> Path:
        return DEFAULT_CHECKPOINT_ROOT / self.experiment_id

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "state.pt"

    @property
    def config_path(self) -> Path:
        return DEFAULT_MATRIX_ROOT / "configs" / "{}.yaml".format(self.experiment_id)

    @property
    def token_length(self) -> int:
        """Maximum interleaved GPT-2 sequence length, including the query."""

        return 2 * (self.max_context + 1)

    def to_training_config(self) -> Dict[str, Any]:
        """Build a standalone Quinine config compatible with ``train.py``."""

        data_kwargs = {"rho": self.train_rho_x}
        if self.rho_x_after is not None:
            data_kwargs.update(
                rho_after=self.rho_x_after,
                change_point=self.feature_change_point,
            )
        task_kwargs = {"snr": self.train_snr, "noise_rho": self.train_rho_e}
        if self.rho_e_after is not None:
            task_kwargs.update(
                noise_rho_after=self.rho_e_after,
                noise_change_point=self.noise_change_point,
            )
        points = self.max_context + 1  # context plus independent final query
        return {
            "model": {
                "family": self.architecture.family,
                "n_dims": self.d,
                "n_positions": points,
                "n_embd": self.architecture.n_embd,
                "n_layer": self.architecture.n_layer,
                "n_head": self.architecture.n_head,
            },
            "training": {
                "task": "dependent_linear_regression",
                "task_kwargs": task_kwargs,
                "data": "gaussian_ar1",
                "data_kwargs": data_kwargs,
                "query_mode": "independent",
                "seed": self.train_seed,
                "data_device": self.data_device,
                "precision": self.precision,
                "max_context": self.max_context,
                "experiment_id": self.experiment_id,
                "batch_size": self.batch_size,
                "learning_rate": 0.0001,
                "train_steps": self.training_steps,
                "save_every_steps": 1000,
                "keep_every_steps": 100000,
                "resume_id": self.experiment_id,
                "eval_after_train": False,
                "curriculum": {
                    "dims": {
                        "start": min(5, self.d),
                        "end": self.d,
                        "inc": 1,
                        "interval": 2000,
                    },
                    "points": {
                        "start": min(11, points),
                        "end": points,
                        "inc": 2,
                        "interval": 2000,
                    },
                },
            },
            # train.py runs with cwd=src, making this repository-relative.
            "out_dir": "../{}".format(DEFAULT_CHECKPOINT_ROOT.as_posix()),
            "wandb": {"name": self.experiment_id},
        }

    def manifest_row(self) -> Dict[str, Any]:
        architecture = self.architecture
        return {
            "experiment_id": self.experiment_id,
            "group": self.group,
            "regime": self.regime,
            "train_seed": self.train_seed,
            "d": self.d,
            "train_rho_x": self.train_rho_x,
            "train_rho_e": self.train_rho_e,
            "rho_x_after": self.rho_x_after,
            "feature_change_point": self.feature_change_point,
            "rho_e_after": self.rho_e_after,
            "noise_change_point": self.noise_change_point,
            "max_context": self.max_context,
            "train_snr": self.train_snr,
            "label_noise_std": self.label_noise_std,
            "architecture": architecture.sweep_family,
            "architecture_memberships": list(architecture_memberships(architecture)),
            "model_family": architecture.family,
            "n_embd": architecture.n_embd,
            "n_layer": architecture.n_layer,
            "n_head": architecture.n_head,
            "batch_size": self.batch_size,
            "training_steps": self.training_steps,
            "data_device": self.data_device,
            "precision": self.precision,
            "token_length": self.token_length,
            "config_path": _relative_posix(self.config_path),
            "checkpoint_dir": _relative_posix(self.checkpoint_dir),
            "checkpoint_path": _relative_posix(self.checkpoint_path),
        }


@dataclass(frozen=True)
class MatrixGroup:
    name: str
    description: str
    base: Mapping[str, Any]
    axes: Mapping[str, Sequence[Any]]
    variants: Tuple[Mapping[str, Any], ...] = ({},)
    max_context_policy: str = "fixed"


CANONICAL_REGIMES = (
    {"regime": "iid", "train_rho_x": 0.0, "train_rho_e": 0.0},
    {"regime": "feature_ar1", "train_rho_x": 0.8, "train_rho_e": 0.0},
    {"regime": "noise_ar1", "train_rho_x": 0.0, "train_rho_e": 0.8},
    {"regime": "both_ar1", "train_rho_x": 0.8, "train_rho_e": 0.8},
    {
        "regime": "change_forward",
        "train_rho_x": 0.0,
        "train_rho_e": 0.0,
        "rho_x_after": 0.9,
        "change_direction": "forward",
    },
    {
        "regime": "change_reverse",
        "train_rho_x": 0.9,
        "train_rho_e": 0.0,
        "rho_x_after": 0.0,
        "change_direction": "reverse",
    },
)


GROUPS = {
    "canonical": MatrixGroup(
        name="canonical",
        description="Six canonical data regimes with three training seeds",
        base={"d": 20, "max_context": 80, "train_snr": 2.0},
        axes={"train_seed": (0, 1, 2)},
        variants=CANONICAL_REGIMES,
    ),
    "stage0": MatrixGroup(
        name="stage0",
        description="Small matched-dependence validation matrix",
        base={
            "regime": "stationary_feature_ar1",
            "d": 20,
            "max_context": 80,
            "train_rho_e": 0.0,
            "train_snr": 2.0,
        },
        axes={"train_rho_x": (0.0, 0.6, 0.9), "train_seed": (0, 1, 2)},
    ),
    "matched_snr_pilot": MatrixGroup(
        name="matched_snr_pilot",
        description="Six-model pilot with train SNR matched to each test SNR",
        base={
            "regime": "stationary_feature_ar1",
            "d": 20,
            "train_rho_e": 0.0,
            "max_context": 80,
        },
        axes={
            "train_rho_x": (0.0, 0.9),
            "train_snr": (0.8, 1.6, 3.2),
            "train_seed": (0,),
        },
    ),
    "matched_rho": MatrixGroup(
        name="matched_rho",
        description="Main stationary training matrix for matched-rho evaluation",
        base={
            "regime": "stationary_feature_ar1",
            "d": 20,
            "max_context": 80,
            "train_rho_e": 0.0,
            "train_snr": 2.0,
        },
        axes={
            "train_rho_x": (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95),
            "train_seed": (0, 1, 2, 3, 4),
        },
    ),
    "dimension": MatrixGroup(
        name="dimension",
        description="Dimension scaling with max context equal to four times d",
        base={
            "regime": "stationary_feature_ar1",
            "train_rho_e": 0.0,
            "train_snr": 2.0,
        },
        axes={
            "d": (10, 20, 40, 80),
            "train_rho_x": (0.0, 0.6, 0.9),
            "train_seed": (0, 1, 2),
        },
        max_context_policy="four_times_dimension",
    ),
    "architecture": MatrixGroup(
        name="architecture",
        description="Unique head/width/depth variants in two representative regimes",
        base={
            "regime": "stationary_feature_ar1",
            "d": 20,
            "max_context": 80,
            "train_rho_e": 0.0,
            "train_snr": 2.0,
        },
        axes={
            "architecture": ARCHITECTURE_VARIANTS,
            "train_rho_x": (0.0, 0.9),
            "train_seed": (0, 1, 2),
        },
    ),
}


DIMENSION_K_OVER_D = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0)
DENSE_TEST_SNRS = (0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8)
DENSE_TEST_RHOS = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95)
DEFAULT_EVAL_SEEDS = tuple(range(1001, 1011))


def dimension_context_lengths(d: int) -> Tuple[int, ...]:
    if isinstance(d, bool) or d <= 0:
        raise ValueError("d must be a positive integer")
    return tuple(dict.fromkeys(int(round(ratio * d)) for ratio in DIMENSION_K_OVER_D))


def evaluation_axes_for_group(name: str) -> Dict[str, Any]:
    """Return the declared dense evaluation axes without expanding them."""

    if name not in GROUPS:
        raise KeyError("unknown matrix group {!r}; choose from {}".format(name, sorted(GROUPS)))
    axes = {
        "test_snrs": list(DENSE_TEST_SNRS),
        "eval_seeds": list(DEFAULT_EVAL_SEEDS),
        "protocols": ["matched"],
    }
    if name == "canonical":
        axes["protocols"].append("shift")
        axes["shift_test_rho_x"] = list(DENSE_TEST_RHOS)
    if name == "architecture":
        axes["context_selection"] = "boundary_suggestions"
    else:
        axes["k_over_d"] = list(DIMENSION_K_OVER_D)
        axes["context_rounding"] = "round(k_over_d * d)"
    return axes


def _make_experiment(group: MatrixGroup, values: Mapping[str, Any]) -> ExperimentSpec:
    row = dict(values)
    if group.max_context_policy == "four_times_dimension":
        row["max_context"] = max(dimension_context_lengths(row["d"]))
    elif group.max_context_policy != "fixed":
        raise ValueError("unknown max-context policy: {}".format(group.max_context_policy))
    direction = row.pop("change_direction", None)
    if direction is not None:
        forward_point = row["max_context"] // 2
        if direction == "forward":
            row["feature_change_point"] = forward_point
        elif direction == "reverse":
            # Reverse all max_context-1 transition coefficients exactly.
            row["feature_change_point"] = row["max_context"] - forward_point + 1
        else:
            raise ValueError("unknown change direction: {}".format(direction))
    return ExperimentSpec(group=group.name, **row)


def expand_group(name: str) -> Tuple[ExperimentSpec, ...]:
    """Expand and semantically deduplicate one named matrix group."""

    try:
        group = GROUPS[name]
    except KeyError as error:
        raise KeyError("unknown matrix group {!r}; choose from {}".format(name, sorted(GROUPS))) from error
    experiments = {}
    for variant in group.variants:
        base = dict(group.base)
        base.update(variant)
        for values in expand_axes(base, group.axes):
            experiment = _make_experiment(group, values)
            experiments.setdefault(experiment.experiment_id, experiment)
    return tuple(experiments.values())


def experiment_counts() -> Dict[str, int]:
    return {name: len(expand_group(name)) for name in GROUPS}


def write_training_config(spec: ExperimentSpec, repository_root: Path = REPOSITORY_ROOT) -> Path:
    """Materialize one generated YAML overlay and return its absolute path."""

    import yaml

    destination = Path(repository_root) / spec.config_path
    contents = yaml.safe_dump(spec.to_training_config(), sort_keys=False)
    _atomic_write_text(destination, contents)
    return destination


MANIFEST_FIELDS = (
    "experiment_id",
    "group",
    "regime",
    "train_seed",
    "d",
    "train_rho_x",
    "train_rho_e",
    "rho_x_after",
    "feature_change_point",
    "rho_e_after",
    "noise_change_point",
    "max_context",
    "train_snr",
    "label_noise_std",
    "architecture",
    "architecture_memberships",
    "model_family",
    "n_embd",
    "n_layer",
    "n_head",
    "batch_size",
    "training_steps",
    "data_device",
    "precision",
    "token_length",
    "config_path",
    "checkpoint_dir",
    "checkpoint_path",
)


def experiment_from_manifest_row(row: Mapping[str, Any]) -> ExperimentSpec:
    """Rebuild and verify one semantic experiment from a persisted row."""

    if not isinstance(row, Mapping):
        raise ValueError("experiment manifest rows must be mappings")
    architecture = ArchitectureSpec(
        n_embd=int(row["n_embd"]),
        n_layer=int(row["n_layer"]),
        n_head=int(row["n_head"]),
        family=row.get("model_family", "gpt2"),
        sweep_family=row.get("architecture", "manifest"),
    )

    def optional_float(name: str) -> Optional[float]:
        value = row.get(name)
        return None if value is None else float(value)

    def optional_int(name: str) -> Optional[int]:
        value = row.get(name)
        return None if value is None else int(value)

    spec = ExperimentSpec(
        group=str(row["group"]),
        regime=str(row["regime"]),
        train_seed=int(row["train_seed"]),
        d=int(row["d"]),
        train_rho_x=float(row["train_rho_x"]),
        train_rho_e=float(row["train_rho_e"]),
        architecture=architecture,
        max_context=int(row["max_context"]),
        train_snr=float(row["train_snr"]),
        rho_x_after=optional_float("rho_x_after"),
        feature_change_point=optional_int("feature_change_point"),
        rho_e_after=optional_float("rho_e_after"),
        noise_change_point=optional_int("noise_change_point"),
        batch_size=int(row.get("batch_size", 64)),
        training_steps=int(row.get("training_steps", 500001)),
        data_device=row.get("data_device", "model"),
        precision=row.get("precision", "float32"),
    )
    if row.get("experiment_id") != spec.experiment_id:
        raise ValueError("manifest experiment ID does not match its semantic fields")
    return spec


def load_experiment_manifest(path: Path) -> Tuple[ExperimentSpec, ...]:
    """Load a matrix manifest or a parameter-matching training report.

    Matrix manifests store rows in ``experiments``. Parameter-matching reports
    store the same verified rows in ``training_experiments``. Supporting both
    lets one report feed planning, training, and evaluation without hand-made
    YAML files.
    """

    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("schema_version") != MATRIX_SCHEMA_VERSION:
        raise ValueError("unsupported experiment manifest schema: {}".format(source))
    rows = document.get("experiments")
    if rows is None:
        rows = document.get("training_experiments")
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "experiment manifest must contain a nonempty experiments or "
            "training_experiments list: {}".format(source)
        )
    specs = []
    seen = set()
    for row in rows:
        spec = experiment_from_manifest_row(row)
        if spec.experiment_id in seen:
            raise ValueError(
                "duplicate experiment ID in manifest: {}".format(spec.experiment_id)
            )
        seen.add(spec.experiment_id)
        specs.append(spec)
    declared_count = document.get("experiment_count", document.get("training_experiment_count"))
    if declared_count is not None and int(declared_count) != len(specs):
        raise ValueError("manifest experiment count does not match its rows")
    return tuple(specs)


def write_manifest(
    group_name: str,
    specs: Iterable[ExperimentSpec],
    repository_root: Path = REPOSITORY_ROOT,
    matrix_root: Path = DEFAULT_MATRIX_ROOT,
) -> Tuple[Path, Path]:
    """Write matching JSON and CSV manifests using repository-relative paths."""

    root = Path(repository_root)
    matrix_root = Path(matrix_root)
    _relative_posix(matrix_root)
    rows = [spec.manifest_row() for spec in specs]
    json_path = root / matrix_root / "manifests" / "{}.json".format(group_name)
    csv_path = root / matrix_root / "manifests" / "{}.csv".format(group_name)
    document = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "group": group_name,
        "experiment_count": len(rows),
        "evaluation_axes": evaluation_axes_for_group(group_name),
        "snr_definition": SNR_DEFINITION,
        "experiments": rows,
    }
    _atomic_write_text(json_path, json.dumps(document, indent=2, sort_keys=True) + "\n")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_name("{}.{}.tmp".format(csv_path.name, uuid.uuid4().hex))
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            csv_rows = []
            for row in rows:
                csv_rows.append(
                    {
                        key: _canonical_json(value)
                        if isinstance(value, (dict, list, tuple))
                        else value
                        for key, value in row.items()
                    }
                )
            writer.writerows(csv_rows)
        os.replace(str(temporary), str(csv_path))
    finally:
        if temporary.exists():
            temporary.unlink()
    return json_path, csv_path


class LockUnavailable(RuntimeError):
    pass


class ExistingCheckpoint(RuntimeError):
    pass


class ExperimentLock:
    """Cross-platform lock based on atomic exclusive file creation."""

    def __init__(self, path: Path, timeout: float = 0.0, poll_interval: float = 0.1):
        self.path = Path(path)
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.token = uuid.uuid4().hex
        self._held = False

    def owner_metadata(self) -> Dict[str, Any]:
        """Return current lock metadata, or an empty mapping if unavailable."""

        try:
            owner = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}
        return owner if isinstance(owner, dict) else {}

    def recover_abandoned(self) -> bool:
        """Remove a lock only when its owner is provably dead on this host."""

        owner = self.owner_metadata()
        if not owner:
            return False
        if owner.get("hostname") != socket.gethostname():
            return False
        pid = owner.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return False
        except PermissionError:
            return False
        except OSError:
            try:
                self.path.unlink()
                return True
            except FileNotFoundError:
                return True

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout
        payload = _canonical_json(
            {
                "token": self.token,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "created_at": _utc_now(),
            }
        ).encode("utf-8")
        while True:
            try:
                descriptor = os.open(
                    str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644
                )
            except FileExistsError:
                if self.recover_abandoned():
                    continue
                if time.monotonic() >= deadline:
                    raise LockUnavailable("experiment is already locked: {}".format(self.path))
                time.sleep(self.poll_interval)
                continue
            try:
                os.write(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._held = True
            return

    def release(self) -> None:
        if not self._held:
            return
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
            if document.get("token") == self.token:
                self.path.unlink()
        except FileNotFoundError:
            pass
        finally:
            self._held = False

    def __enter__(self) -> "ExperimentLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


@dataclass(frozen=True)
class RunDecision:
    action: str
    reason: str
    experiment_id: str
    checkpoint_path: str


class RunRegistry:
    """Resume-friendly local state and locking for matrix launchers."""

    def __init__(
        self,
        repository_root: Path = REPOSITORY_ROOT,
        matrix_root: Path = DEFAULT_MATRIX_ROOT,
    ):
        self.repository_root = Path(repository_root)
        self.matrix_root = Path(matrix_root)
        _relative_posix(self.matrix_root)

    def _local(self, path: Path) -> Path:
        return self.repository_root / path

    def state_path(self, spec: ExperimentSpec) -> Path:
        return self._local(self.matrix_root / "state" / "{}.json".format(spec.experiment_id))

    def lock_path(self, spec: ExperimentSpec) -> Path:
        return self._local(self.matrix_root / "locks" / "{}.lock".format(spec.experiment_id))

    def checkpoint_path(self, spec: ExperimentSpec) -> Path:
        return self._local(spec.checkpoint_path)

    def read_state(self, spec: ExperimentSpec) -> Dict[str, Any]:
        path = self.state_path(spec)
        if not path.exists():
            return {}
        with path.open(encoding="utf-8") as handle:
            document = json.load(handle)
        if document.get("experiment_id") != spec.experiment_id:
            raise ValueError("state file experiment ID mismatch: {}".format(path))
        return document

    def _write_state(self, spec: ExperimentSpec, document: Mapping[str, Any]) -> None:
        payload = dict(document)
        payload["experiment_id"] = spec.experiment_id
        payload["checkpoint_path"] = _relative_posix(spec.checkpoint_path)
        payload["updated_at"] = _utc_now()
        _atomic_write_text(
            self.state_path(spec), json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )

    def decision(self, spec: ExperimentSpec, resume: bool = True) -> RunDecision:
        state = self.read_state(spec)
        checkpoint_exists = self.checkpoint_path(spec).is_file()
        if state.get("status") == "completed" and checkpoint_exists:
            return RunDecision(
                "skip",
                "completed checkpoint already exists",
                spec.experiment_id,
                _relative_posix(spec.checkpoint_path),
            )
        if checkpoint_exists:
            action = "resume" if resume else "blocked"
            reason = "resumable checkpoint exists" if resume else "checkpoint exists; enable resume"
            return RunDecision(
                action, reason, spec.experiment_id, _relative_posix(spec.checkpoint_path)
            )
        reason = "no checkpoint exists"
        if state.get("status") in {"running", "failed", "interrupted"}:
            reason = "previous {} attempt has no checkpoint; restart".format(state["status"])
        return RunDecision(
            "start", reason, spec.experiment_id, _relative_posix(spec.checkpoint_path)
        )

    def mark_running(self, spec: ExperimentSpec, action: str) -> None:
        previous = self.read_state(spec)
        self._write_state(
            spec,
            {
                "status": "running",
                "action": action,
                "attempt": int(previous.get("attempt", 0)) + 1,
                "started_at": _utc_now(),
            },
        )

    def mark_completed(self, spec: ExperimentSpec) -> None:
        if not self.checkpoint_path(spec).is_file():
            raise FileNotFoundError(
                "successful run did not produce {}".format(self.checkpoint_path(spec))
            )
        previous = self.read_state(spec)
        self._write_state(
            spec,
            {
                "status": "completed",
                "action": previous.get("action"),
                "attempt": int(previous.get("attempt", 1)),
                "started_at": previous.get("started_at"),
                "completed_at": _utc_now(),
            },
        )

    def mark_interrupted(self, spec: ExperimentSpec, message: str) -> None:
        previous = self.read_state(spec)
        self._write_state(
            spec,
            {
                "status": "interrupted",
                "action": previous.get("action"),
                "attempt": int(previous.get("attempt", 1)),
                "started_at": previous.get("started_at"),
                "message": message,
            },
        )

    def mark_failed(self, spec: ExperimentSpec, error: BaseException) -> Path:
        previous = self.read_state(spec)
        failure = {
            "experiment_id": spec.experiment_id,
            "failed_at": _utc_now(),
            "attempt": int(previous.get("attempt", 1)),
            "error_type": type(error).__name__,
            "error": str(error),
            "checkpoint_path": _relative_posix(spec.checkpoint_path),
        }
        self._write_state(
            spec,
            {
                "status": "failed",
                "action": previous.get("action"),
                "attempt": failure["attempt"],
                "started_at": previous.get("started_at"),
                "error_type": failure["error_type"],
                "error": failure["error"],
            },
        )
        failure_path = self._local(
            self.matrix_root
            / "failures"
            / spec.experiment_id
            / "{}.json".format(uuid.uuid4().hex)
        )
        _atomic_write_text(failure_path, json.dumps(failure, indent=2, sort_keys=True) + "\n")
        return failure_path

    @contextmanager
    def claim(
        self, spec: ExperimentSpec, resume: bool = True, lock_timeout: float = 0.0
    ) -> Iterator[RunDecision]:
        """Lock a run and update state around a caller-managed training command."""

        with ExperimentLock(self.lock_path(spec), timeout=lock_timeout):
            decision = self.decision(spec, resume=resume)
            if decision.action == "skip":
                yield decision
                return
            if decision.action == "blocked":
                raise ExistingCheckpoint(decision.reason)
            self.mark_running(spec, decision.action)
            try:
                yield decision
            except (KeyboardInterrupt, SystemExit) as error:
                self.mark_interrupted(spec, str(error) or type(error).__name__)
                raise
            except BaseException as error:
                self.mark_failed(spec, error)
                raise
            else:
                try:
                    self.mark_completed(spec)
                except BaseException as error:
                    self.mark_failed(spec, error)
                    raise


@dataclass(frozen=True)
class EvaluationSpec:
    """Evaluation metadata whose protocol label is checked on construction."""

    train_experiment_id: str
    train_rho_x: float
    test_rho_x: float
    train_rho_e: float
    test_rho_e: float
    train_seed: int
    eval_seed: int
    d: int
    context_length: int
    train_snr: float
    test_snr: float
    architecture: str
    checkpoint_path: str
    protocol: str
    precision: str = "float32"

    def __post_init__(self) -> None:
        matched = self.train_rho_x == self.test_rho_x and self.train_rho_e == self.test_rho_e
        expected = "matched" if matched else "shift"
        if self.protocol not in {"matched", "shift"}:
            raise ValueError("protocol must be 'matched' or 'shift'")
        if self.protocol != expected:
            raise ValueError(
                "protocol {!r} conflicts with train/test dependence; expected {!r}".format(
                    self.protocol, expected
                )
            )
        for name, value, allow_zero in (
            ("d", self.d, False),
            ("context_length", self.context_length, False),
            ("train_seed", self.train_seed, True),
            ("eval_seed", self.eval_seed, True),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (0 if allow_zero else 1)
            ):
                qualifier = "nonnegative" if allow_zero else "positive"
                raise ValueError("{} must be a {} integer".format(name, qualifier))
        for name, value in (("train_snr", self.train_snr), ("test_snr", self.test_snr)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("{} must be positive and finite".format(name))
        for name, value in (
            ("train_rho_x", self.train_rho_x),
            ("test_rho_x", self.test_rho_x),
            ("train_rho_e", self.train_rho_e),
            ("test_rho_e", self.test_rho_e),
        ):
            if not math.isfinite(value) or abs(value) >= 1:
                raise ValueError("{} must be finite with absolute value below one".format(name))
        _relative_posix(Path(self.checkpoint_path))

    @property
    def k_over_d(self) -> float:
        return self.context_length / self.d

    @property
    def evaluation_id(self) -> str:
        return _stable_id("eval", asdict(self))


def make_evaluation_spec(
    training: ExperimentSpec,
    test_rho_x: float,
    test_rho_e: float,
    test_snr: float,
    eval_seed: int,
    context_length: int,
    precision: str = "float32",
) -> EvaluationSpec:
    protocol = (
        "matched"
        if training.train_rho_x == test_rho_x and training.train_rho_e == test_rho_e
        else "shift"
    )
    return EvaluationSpec(
        train_experiment_id=training.experiment_id,
        train_rho_x=training.train_rho_x,
        test_rho_x=test_rho_x,
        train_rho_e=training.train_rho_e,
        test_rho_e=test_rho_e,
        train_seed=training.train_seed,
        eval_seed=eval_seed,
        d=training.d,
        context_length=context_length,
        train_snr=training.train_snr,
        test_snr=test_snr,
        architecture=training.architecture.sweep_family,
        checkpoint_path=_relative_posix(training.checkpoint_path),
        protocol=protocol,
        precision=precision,
    )


def iter_dense_evaluations(
    training: ExperimentSpec,
    include_shift: bool = False,
    test_snrs: Sequence[float] = DENSE_TEST_SNRS,
    eval_seeds: Sequence[int] = DEFAULT_EVAL_SEEDS,
    shift_test_rhos: Sequence[float] = DENSE_TEST_RHOS,
    context_lengths: Optional[Sequence[int]] = None,
    precision: str = "float32",
) -> Iterator[EvaluationSpec]:
    """Expand dense matched/shift evaluation rows for a stationary checkpoint.

    The matched rho is always included.  When ``include_shift`` is true, the
    dense rho grid is added and each row receives its protocol from the actual
    train/test values.  Nonstationary checkpoints require a schedule-aware
    evaluator and are rejected rather than being mislabeled by one scalar rho.
    """

    if training.rho_x_after is not None or training.rho_e_after is not None:
        raise ValueError("dense scalar-rho evaluation requires a stationary training spec")
    contexts = (
        tuple(context_lengths)
        if context_lengths is not None
        else dimension_context_lengths(training.d)
    )
    if not contexts:
        raise ValueError("context_lengths must be nonempty")
    if max(contexts) > training.max_context:
        raise ValueError("evaluation context exceeds the checkpoint max_context")
    test_rhos = [training.train_rho_x]
    if include_shift:
        test_rhos.extend(shift_test_rhos)
    test_rhos = tuple(dict.fromkeys(test_rhos))
    for context_length, test_rho, test_snr, eval_seed in itertools.product(
        contexts, test_rhos, test_snrs, eval_seeds
    ):
        yield make_evaluation_spec(
            training=training,
            test_rho_x=test_rho,
            test_rho_e=training.train_rho_e,
            test_snr=test_snr,
            eval_seed=eval_seed,
            context_length=context_length,
            precision=precision,
        )
