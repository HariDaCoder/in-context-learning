"""Architecture specifications and exact parameter-count matching.

The helpers in this module describe model families without assigning them to
particular machines.  Parameter matching always instantiates the candidate
models and counts their parameters; widths are never inferred from a scaling
formula.  A match is therefore exact for the supplied finite candidate grid.
"""

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace


@dataclass(frozen=True)
class ArchitectureSpec:
    """The three GPT-2 architecture axes varied by the experiments."""

    n_embd: int
    n_layer: int
    n_head: int
    family: str = "gpt2"
    sweep_family: str = "custom"

    def __post_init__(self):
        for name in ("n_embd", "n_layer", "n_head"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("{} must be a positive integer".format(name))
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if not self.family or not self.sweep_family:
            raise ValueError("family names must be nonempty")

    @property
    def head_dim(self):
        return self.n_embd // self.n_head

    def model_dict(self):
        return {
            "family": self.family,
            "n_embd": self.n_embd,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
        }


@dataclass(frozen=True)
class ParameterMatch:
    """Best instantiated candidate for one requested depth."""

    target: ArchitectureSpec
    candidate: ArchitectureSpec
    target_parameter_count: int
    parameter_count: int
    absolute_error: int
    relative_error: float
    candidate_grid_size: int

    def as_row(self):
        row = {
            "sweep_family": self.candidate.sweep_family,
            "family": self.candidate.family,
            "n_embd": self.candidate.n_embd,
            "n_layer": self.candidate.n_layer,
            "n_head": self.candidate.n_head,
            "head_dim": self.candidate.head_dim,
            "parameter_count": self.parameter_count,
            "target_parameter_count": self.target_parameter_count,
            "absolute_parameter_error": self.absolute_error,
            "relative_parameter_error": self.relative_error,
            "candidate_grid_size": self.candidate_grid_size,
            "target_n_embd": self.target.n_embd,
            "target_n_layer": self.target.n_layer,
            "target_n_head": self.target.n_head,
        }
        return row


def _unique_positive(values, name):
    result = []
    seen = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("{} entries must be positive integers".format(name))
        if value in seen:
            raise ValueError("{} must not contain duplicates".format(name))
        seen.add(value)
        result.append(value)
    if not result:
        raise ValueError("{} must be nonempty".format(name))
    return result


def fixed_width_head_specs(
    heads=(1, 2, 4, 8, 16), n_embd=256, n_layer=12
):
    """Vary head count while holding residual width and depth fixed."""

    heads = _unique_positive(heads, "heads")
    return [
        ArchitectureSpec(n_embd, n_layer, head, sweep_family="fixed_width_heads")
        for head in heads
    ]


def fixed_head_dim_specs(
    heads=(1, 2, 4, 8, 16), head_dim=32, n_layer=12
):
    """Vary head count while holding per-head dimension and depth fixed."""

    heads = _unique_positive(heads, "heads")
    if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim < 1:
        raise ValueError("head_dim must be a positive integer")
    return [
        ArchitectureSpec(
            head * head_dim,
            n_layer,
            head,
            sweep_family="fixed_head_dim",
        )
        for head in heads
    ]


def fixed_width_depth_specs(
    depths=(2, 4, 6, 12), n_embd=256, n_head=8
):
    """Vary depth while holding residual width and head count fixed."""

    depths = _unique_positive(depths, "depths")
    return [
        ArchitectureSpec(n_embd, depth, n_head, sweep_family="fixed_width_depth")
        for depth in depths
    ]


def architecture_specs(
    heads=(1, 2, 4, 8, 16),
    depths=(2, 4, 6, 12),
    fixed_width=256,
    fixed_head_dim=32,
    head_sweep_depth=12,
    depth_sweep_heads=8,
):
    """Return all three standard families, de-duplicated by family and shape."""

    result = []
    seen = set()
    groups = (
        fixed_width_head_specs(heads, fixed_width, head_sweep_depth),
        fixed_head_dim_specs(heads, fixed_head_dim, head_sweep_depth),
        fixed_width_depth_specs(depths, fixed_width, depth_sweep_heads),
    )
    for group in groups:
        for spec in group:
            key = (spec.sweep_family, spec.n_embd, spec.n_layer, spec.n_head)
            if key not in seen:
                seen.add(key)
                result.append(spec)
    return result


def count_parameters(model, trainable_only=True):
    """Count scalar parameters in an instantiated model."""

    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def _default_model_factory(config):
    # Lazy import keeps specification generation independent of transformers.
    from models import build_model

    return build_model(config)


def instantiate_and_count(
    spec,
    n_dims,
    n_positions,
    model_factory=None,
    trainable_only=True,
):
    """Instantiate ``spec`` and return its exact parameter count.

    Input dimension and positional capacity are explicit because both change
    the parameter count of this repository's Transformer wrapper.
    """

    for value, name in ((n_dims, "n_dims"), (n_positions, "n_positions")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("{} must be a positive integer".format(name))
    if not isinstance(spec, ArchitectureSpec):
        raise TypeError("spec must be an ArchitectureSpec")
    config = SimpleNamespace(
        family=spec.family,
        n_dims=n_dims,
        n_positions=n_positions,
        n_embd=spec.n_embd,
        n_layer=spec.n_layer,
        n_head=spec.n_head,
    )
    factory = model_factory or _default_model_factory
    model = factory(config)
    try:
        return count_parameters(model, trainable_only=trainable_only)
    finally:
        # Do not keep a succession of large candidates alive during a search.
        del model


def architecture_count_rows(
    specs,
    n_dims,
    n_positions,
    model_factory=None,
    trainable_only=True,
):
    """Instantiate each architecture and return rows with exact counts."""

    rows = []
    for spec in specs:
        count = instantiate_and_count(
            spec, n_dims, n_positions, model_factory, trainable_only
        )
        rows.append({
            "sweep_family": spec.sweep_family,
            "family": spec.family,
            "n_embd": spec.n_embd,
            "n_layer": spec.n_layer,
            "n_head": spec.n_head,
            "head_dim": spec.head_dim,
            "n_dims": n_dims,
            "n_positions": n_positions,
            "parameter_count": count,
            "trainable_only": trainable_only,
        })
    return rows


def parameter_matched_depth_specs(
    target,
    depths,
    candidate_widths,
    n_dims,
    n_positions,
    n_head=None,
    model_factory=None,
    trainable_only=True,
):
    """Find the closest exact parameter count at every requested depth.

    Every valid width in ``candidate_widths`` is instantiated.  The returned
    optimum is thus the closest model on that declared grid, with deterministic
    tie-breaking toward the smaller model and then the smaller width.
    """

    if not isinstance(target, ArchitectureSpec):
        raise TypeError("target must be an ArchitectureSpec")
    depths = _unique_positive(depths, "depths")
    widths = _unique_positive(candidate_widths, "candidate_widths")
    heads = target.n_head if n_head is None else n_head
    if isinstance(heads, bool) or not isinstance(heads, int) or heads < 1:
        raise ValueError("n_head must be a positive integer")
    valid_widths = [width for width in widths if width % heads == 0]
    if not valid_widths:
        raise ValueError("candidate_widths has no value divisible by n_head")

    target_count = instantiate_and_count(
        target, n_dims, n_positions, model_factory, trainable_only
    )
    matches = []
    for depth in depths:
        candidates = []
        for width in valid_widths:
            spec = ArchitectureSpec(
                width,
                depth,
                heads,
                family=target.family,
                sweep_family="parameter_matched_depth",
            )
            parameter_count = instantiate_and_count(
                spec, n_dims, n_positions, model_factory, trainable_only
            )
            candidates.append((abs(parameter_count - target_count), parameter_count, width, spec))
        error, parameter_count, _, best = min(candidates, key=lambda item: item[:3])
        matches.append(
            ParameterMatch(
                target=target,
                candidate=best,
                target_parameter_count=target_count,
                parameter_count=parameter_count,
                absolute_error=error,
                relative_error=error / target_count if target_count else 0.0,
                candidate_grid_size=len(valid_widths),
            )
        )
    return matches


def write_architecture_csv(rows, path):
    """Write architecture specs or parameter matches as a compact CSV."""

    normalized = []
    for item in rows:
        if isinstance(item, ParameterMatch):
            normalized.append(item.as_row())
        elif isinstance(item, ArchitectureSpec):
            row = asdict(item)
            row["head_dim"] = item.head_dim
            normalized.append(row)
        elif isinstance(item, dict):
            normalized.append(dict(item))
        else:
            raise TypeError("rows must contain architecture specs, matches, or dictionaries")
    if not normalized:
        raise ValueError("rows must be nonempty")
    fields = []
    for row in normalized:
        for key in row:
            if key not in fields:
                fields.append(key)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(normalized)
    return destination
