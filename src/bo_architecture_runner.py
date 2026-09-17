"""Reproducible parameter-matching and attention-mechanism workflows.

This runner is independent of the training-suite launcher.  It contains no
machine assignment policy: it describes architecture searches, converts their
results to stable :class:`bo_matrix.ExperimentSpec` objects, and evaluates
aggregate attention diagnostics for trusted local checkpoints.
"""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import uuid

from bo_architecture import (
    ArchitectureSpec,
    ParameterMatch,
    parameter_matched_depth_specs,
)
from bo_matrix import ExperimentSpec, SNR_DEFINITION


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER_SCHEMA_VERSION = 1


def _canonical_json(value):
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _stable_id(prefix, value):
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:20]
    return "{}_{}".format(prefix, digest)


def _atomic_write(path, contents):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{}.{}.tmp".format(path.name, uuid.uuid4().hex))
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(contents)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("{} must be a positive integer".format(name))
    return value


def _unique(values, name):
    result = tuple(values)
    if not result:
        raise ValueError("{} must be nonempty".format(name))
    if len(set(result)) != len(result):
        raise ValueError("{} must not contain duplicates".format(name))
    return result


def valid_width_grid(width_min, width_max, width_step, n_head):
    """Return the declared inclusive grid filtered to GPT-2-valid widths."""

    for value, name in (
        (width_min, "width_min"),
        (width_max, "width_max"),
        (width_step, "width_step"),
        (n_head, "n_head"),
    ):
        _positive_integer(value, name)
    if width_max < width_min:
        raise ValueError("width_max must be at least width_min")
    widths = tuple(
        width
        for width in range(width_min, width_max + 1, width_step)
        if width % n_head == 0
    )
    if not widths:
        raise ValueError("the requested range contains no width divisible by n_head")
    return widths


def parameter_match_plan(
    target,
    depths,
    width_min,
    width_max,
    width_step,
    n_dims,
    max_context,
):
    """Build a serializable search plan without instantiating any model."""

    if not isinstance(target, ArchitectureSpec):
        raise TypeError("target must be an ArchitectureSpec")
    depths = _unique(depths, "depths")
    for depth in depths:
        _positive_integer(depth, "depth")
    _positive_integer(n_dims, "n_dims")
    _positive_integer(max_context, "max_context")
    widths = valid_width_grid(width_min, width_max, width_step, target.n_head)
    semantic_plan = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "target": {
            **target.model_dict(),
            "sweep_family": target.sweep_family,
        },
        "depths": list(depths),
        "candidate_widths": list(widths),
        "candidate_n_head": target.n_head,
        "n_dims": n_dims,
        # One Transformer position contains one (x,y) pair in this project;
        # independent-query training needs context plus one query position.
        "n_positions": max_context + 1,
        "max_context": max_context,
        "search_model_instantiations": 1 + len(depths) * len(widths),
        "selection": (
            "minimum absolute exact parameter-count error on the declared grid; "
            "ties prefer fewer parameters, then smaller width"
        ),
    }
    semantic_plan["report_id"] = _stable_id("pmatch", semantic_plan)
    return semantic_plan


def matches_to_experiment_specs(
    matches,
    n_dims,
    max_context,
    train_rhos=(0.0, 0.9),
    train_seeds=(0, 1, 2),
    train_snr=2.0,
    batch_size=64,
    training_steps=500001,
    data_device="model",
    precision="float32",
):
    """Convert matches into stable, semantically de-duplicated training specs."""

    train_rhos = _unique(train_rhos, "train_rhos")
    train_seeds = _unique(train_seeds, "train_seeds")
    experiments = {}
    for match in matches:
        if not isinstance(match, ParameterMatch):
            raise TypeError("matches must contain ParameterMatch objects")
        for rho in train_rhos:
            if not math.isfinite(rho) or abs(rho) >= 1:
                raise ValueError("training rho must be finite with absolute value below one")
            for seed in train_seeds:
                spec = ExperimentSpec(
                    group="parameter_matched_depth",
                    regime="stationary_feature_ar1",
                    train_seed=seed,
                    d=n_dims,
                    train_rho_x=float(rho),
                    train_rho_e=0.0,
                    architecture=match.candidate,
                    max_context=max_context,
                    train_snr=train_snr,
                    batch_size=batch_size,
                    training_steps=training_steps,
                    data_device=data_device,
                    precision=precision,
                )
                # ExperimentSpec excludes group/regime labels from identity, so
                # this key also enables reuse with any existing matrix group.
                experiments.setdefault(spec.experiment_id, spec)
    return tuple(experiments.values())


@dataclass(frozen=True)
class ParameterMatchResult:
    plan: dict
    matches: tuple
    experiments: tuple
    json_path: Path
    csv_path: Path
    dry_run: bool


def parameter_match_report(
    output_dir,
    target=ArchitectureSpec(256, 12, 8, sweep_family="standard"),
    depths=(2, 4, 6, 12),
    width_min=32,
    width_max=768,
    width_step=8,
    n_dims=20,
    max_context=80,
    train_rhos=(0.0, 0.9),
    train_seeds=(0, 1, 2),
    train_snr=2.0,
    batch_size=64,
    training_steps=500001,
    data_device="model",
    precision="float32",
    model_factory=None,
    dry_run=False,
):
    """Search exact counts and write matched-architecture CSV/JSON artifacts.

    Dry-run returns the complete finite search plan and output paths without
    importing transformers, instantiating a model, or writing a file.
    """

    train_rhos = _unique(train_rhos, "train_rhos")
    train_seeds = _unique(train_seeds, "train_seeds")
    for rho in train_rhos:
        if not math.isfinite(rho) or abs(rho) >= 1:
            raise ValueError("training rho must be finite with absolute value below one")
    for seed in train_seeds:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("training seeds must be nonnegative integers")
    plan = parameter_match_plan(
        target, depths, width_min, width_max, width_step, n_dims, max_context
    )
    plan["train_rhos"] = list(train_rhos)
    plan["train_seeds"] = list(train_seeds)
    plan["planned_training_experiment_count"] = (
        len(plan["depths"]) * len(train_rhos) * len(train_seeds)
    )
    plan["training_protocol"] = {
        "train_snr": train_snr,
        "batch_size": batch_size,
        "training_steps": training_steps,
        "data_device": data_device,
        "precision": precision,
    }
    # The report filename identifies both the instantiated search and the
    # downstream training matrix, so variants cannot overwrite one another.
    plan.pop("report_id", None)
    plan["report_id"] = _stable_id("pmatch", plan)
    output_dir = Path(output_dir)
    stem = "parameter_match_{}".format(plan["report_id"])
    json_path = output_dir / "{}.json".format(stem)
    csv_path = output_dir / "{}.csv".format(stem)
    if dry_run:
        return ParameterMatchResult(plan, (), (), json_path, csv_path, True)

    matches = tuple(parameter_matched_depth_specs(
        target=target,
        depths=tuple(plan["depths"]),
        candidate_widths=tuple(plan["candidate_widths"]),
        n_dims=n_dims,
        n_positions=max_context + 1,
        n_head=target.n_head,
        model_factory=model_factory,
    ))
    experiments = matches_to_experiment_specs(
        matches,
        n_dims=n_dims,
        max_context=max_context,
        train_rhos=train_rhos,
        train_seeds=train_seeds,
        train_snr=train_snr,
        batch_size=batch_size,
        training_steps=training_steps,
        data_device=data_device,
        precision=precision,
    )
    experiment_ids_by_shape = {}
    for spec in experiments:
        shape = (spec.architecture.n_embd, spec.architecture.n_layer, spec.architecture.n_head)
        experiment_ids_by_shape.setdefault(shape, []).append(spec.experiment_id)

    match_rows = []
    for match in matches:
        row = match.as_row()
        shape = (match.candidate.n_embd, match.candidate.n_layer, match.candidate.n_head)
        row.update({
            "report_id": plan["report_id"],
            "n_dims": n_dims,
            "n_positions": max_context + 1,
            "training_experiment_ids": sorted(experiment_ids_by_shape.get(shape, ())),
        })
        match_rows.append(row)
    document = {
        **plan,
        "status": "complete",
        "snr_definition": SNR_DEFINITION,
        "train_rhos": list(train_rhos),
        "train_seeds": list(train_seeds),
        "train_snr": train_snr,
        "matches": match_rows,
        "training_experiment_count": len(experiments),
        "training_experiments": [spec.manifest_row() for spec in experiments],
        "reuse_rule": (
            "training experiment IDs hash semantic configuration only; matching IDs reuse "
            "the same checkpoint across matrix/report memberships"
        ),
    }
    _atomic_write(json_path, json.dumps(document, indent=2, sort_keys=True) + "\n")

    csv_fields_list = []
    for row in match_rows:
        for key in row:
            if key not in csv_fields_list:
                csv_fields_list.append(key)
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=csv_fields_list)
    writer.writeheader()
    for row in match_rows:
        serialized = dict(row)
        serialized["training_experiment_ids"] = json.dumps(
            serialized["training_experiment_ids"], separators=(",", ":")
        )
        writer.writerow(serialized)
    _atomic_write(csv_path, stream.getvalue())
    return ParameterMatchResult(plan, matches, experiments, json_path, csv_path, False)


def mechanism_evaluation_plan(
    checkpoint_dir,
    output_csv,
    rhos,
    snrs,
    context_lengths,
    eval_seeds,
    n_eval,
    batch_size,
    device="auto",
    query_positions="last_x",
):
    """Return a validated mechanism protocol without touching the checkpoint."""

    rhos = _unique(tuple(float(value) for value in rhos), "rhos")
    snrs = _unique(tuple(float(value) for value in snrs), "snrs")
    context_lengths = _unique(tuple(context_lengths), "context_lengths")
    eval_seeds = _unique(tuple(eval_seeds), "eval_seeds")
    for rho in rhos:
        if not math.isfinite(rho) or abs(rho) >= 1:
            raise ValueError("rhos must be finite with absolute value below one")
    for snr in snrs:
        if not math.isfinite(snr) or snr <= 0:
            raise ValueError("snrs must be positive and finite")
    for value in context_lengths:
        _positive_integer(value, "context length")
    for seed in eval_seeds:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("eval seeds must be nonnegative integers")
    _positive_integer(n_eval, "n_eval")
    _positive_integer(batch_size, "batch_size")
    if query_positions not in {"last_x", "all_x", "all_y", "all"}:
        raise ValueError("unsupported query_positions")
    protocol = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "checkpoint_dir": str(Path(checkpoint_dir)),
        "output_csv": str(Path(output_csv)),
        "rhos": list(rhos),
        "snrs": list(snrs),
        "context_lengths": list(context_lengths),
        "eval_seeds": list(eval_seeds),
        "n_eval": n_eval,
        "batch_size": batch_size,
        "device": device,
        "query_positions": query_positions,
        "condition_count": len(rhos) * len(snrs) * len(context_lengths) * len(eval_seeds),
        "context_distribution": "stationary Gaussian AR(1), unit N(0,I) marginals",
        "query_distribution": "independent N(0,I), appended after context",
        "query_label": "constant zero dummy label; no ground-truth query label is supplied",
        "noise_process": "independent Gaussian label noise",
        "snr_definition": SNR_DEFINITION,
        "retention": "aggregate scalar moments and head-by-head Gram matrices only",
    }
    semantic = dict(protocol)
    semantic.pop("output_csv")
    semantic.pop("device")
    protocol["protocol_id"] = _stable_id("mechanism", semantic)
    return protocol


def _seed_value(eval_seed, n_dims, context_length, batch_index, stream):
    payload = "bo-mechanism-v1/{}/{}/{}/{}/{}".format(
        eval_seed, n_dims, context_length, batch_index, stream
    )
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little")


def load_transformer_checkpoint(checkpoint_dir, device="auto"):
    """Load one trusted local training checkpoint and compact metadata."""

    import torch
    import yaml

    from models import build_model

    checkpoint_dir = Path(checkpoint_dir).resolve()
    config_path = checkpoint_dir / "config.yaml"
    state_path = checkpoint_dir / "state.pt"
    if not config_path.is_file() or not state_path.is_file():
        raise FileNotFoundError("checkpoint directory must contain config.yaml and state.pt")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("checkpoint config has no model mapping")
    architecture = dict(config["model"])
    model = build_model(SimpleNamespace(**architecture))
    # Checkpoints are locally produced/trusted.  weights_only is intentionally
    # omitted because it is unavailable in the project's PyTorch 1.11 stack.
    state = torch.load(str(state_path), map_location="cpu")
    if not isinstance(state, dict) or "model_state_dict" not in state:
        raise ValueError("state.pt has no model_state_dict")
    model.load_state_dict(state["model_state_dict"])
    selected_device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if device == "auto" else device
    selected_device = torch.device(selected_device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested, but CUDA is unavailable")
    model.to(selected_device).eval()
    training = config.get("training", {})
    data_kwargs = training.get("data_kwargs", {}) or {}
    task_kwargs = training.get("task_kwargs", {}) or {}
    return model, {
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_path": str(state_path.resolve()),
        "model_name": model.name,
        "architecture": architecture,
        "training_seed": training.get("seed"),
        "training_experiment_id": training.get("experiment_id"),
        "train_rho_x": data_kwargs.get("rho"),
        "train_rho_x_after": data_kwargs.get("rho_after"),
        "feature_change_point": data_kwargs.get("change_point"),
        "train_rho_e": task_kwargs.get("noise_rho", 0.0),
        "train_rho_e_after": task_kwargs.get("noise_rho_after"),
        "noise_change_point": task_kwargs.get("noise_change_point"),
        "train_snr": task_kwargs.get("snr"),
        "train_step": state.get("train_step"),
        "precision": training.get("precision", "float32"),
    }


def _mechanism_csv(rows, path):
    fields = [
        "protocol_id", "checkpoint_dir", "training_experiment_id", "training_seed",
        "checkpoint_path", "model_name", "architecture", "gpu_model", "precision",
        "train_rho_x", "test_rho_x", "train_rho_e", "test_rho_e", "protocol",
        "train_snr", "test_snr", "d", "k", "context_length", "k_over_d",
        "rho", "snr", "noise_std", "eval_seed",
        "n_eval", "record_type", "layer", "head", "peer_head", "metric",
        "value", "std", "n", "query_positions", "definition", "metadata_json",
    ]
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write(path, stream.getvalue())


def mechanism_evaluation(
    checkpoint_dir,
    output_csv,
    rhos=(0.0, 0.6, 0.9),
    snrs=(0.8, 3.2),
    context_lengths=(20, 40, 80),
    eval_seeds=(1001, 1002, 1003),
    n_eval=32,
    batch_size=16,
    device="auto",
    query_positions="last_x",
    dry_run=False,
    checkpoint_loader=None,
    analyzer_factory=None,
):
    """Evaluate aggregate attention mechanisms for dependent fixed contexts.

    For each condition, the model sees K dependent noisy context examples and
    one independent Gaussian query.  The appended query label is exactly zero;
    no query target is evaluated or exposed.  Each condition gets a separate
    online analyzer so CSV rows remain stratified by evaluation seed.
    """

    protocol = mechanism_evaluation_plan(
        checkpoint_dir, output_csv, rhos, snrs, context_lengths, eval_seeds,
        n_eval, batch_size, device, query_positions,
    )
    if dry_run:
        return {"status": "planned", "protocol": protocol, "rows_written": 0}

    import torch

    from bo_mechanism import AttentionMechanismAnalyzer
    from samplers import get_data_sampler
    from tasks import get_task_sampler

    loader = checkpoint_loader or load_transformer_checkpoint
    model, checkpoint_metadata = loader(checkpoint_dir, device)
    architecture = checkpoint_metadata["architecture"]
    n_dims = int(architecture["n_dims"])
    n_positions = int(architecture["n_positions"])
    if max(protocol["context_lengths"]) + 1 > n_positions:
        raise ValueError(
            "context plus independent query exceeds checkpoint n_positions={}".format(
                n_positions
            )
        )
    make_analyzer = analyzer_factory or AttentionMechanismAnalyzer
    metadata = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "protocol": protocol,
        "checkpoint": checkpoint_metadata,
        "seed_protocol": (
            "SHA256 bo-mechanism-v1 streams; rho and SNR excluded to use common random "
            "innovations across compared conditions"
        ),
    }
    metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    parameter = next(iter(model.parameters()), None)
    model_device = parameter.device if parameter is not None else torch.device("cpu")
    gpu_model = (
        torch.cuda.get_device_name(model_device)
        if model_device.type == "cuda" and torch.cuda.is_available()
        else None
    )
    rows = []
    for rho in protocol["rhos"]:
        for snr in protocol["snrs"]:
            for context_length in protocol["context_lengths"]:
                data_sampler = get_data_sampler("gaussian_ar1", n_dims, rho=rho)
                for eval_seed in protocol["eval_seeds"]:
                    analyzer = make_analyzer(model, query_positions=query_positions)
                    for batch_index, start in enumerate(range(0, n_eval, batch_size)):
                        count = min(batch_size, n_eval - start)
                        torch.manual_seed(
                            _seed_value(eval_seed, n_dims, context_length, batch_index, "weights")
                            % (2**63 - 1)
                        )
                        task = get_task_sampler(
                            "dependent_linear_regression",
                            n_dims,
                            count,
                            snr=snr,
                            noise_rho=0.0,
                        )()
                        torch.manual_seed(
                            _seed_value(eval_seed, n_dims, context_length, batch_index, "features")
                            % (2**63 - 1)
                        )
                        context = data_sampler.sample_xs(context_length, count)
                        torch.manual_seed(
                            _seed_value(eval_seed, n_dims, context_length, batch_index, "label-noise")
                            % (2**63 - 1)
                        )
                        context_labels = task.evaluate(context)
                        torch.manual_seed(
                            _seed_value(eval_seed, n_dims, context_length, batch_index, "query")
                            % (2**63 - 1)
                        )
                        query = torch.randn(count, 1, n_dims, dtype=context.dtype)
                        prompt_xs = torch.cat((context, query), dim=1)
                        prompt_ys = torch.cat(
                            (context_labels, torch.zeros(count, 1, dtype=context_labels.dtype)),
                            dim=1,
                        )
                        analyzer.update(prompt_xs, prompt_ys)
                    condition = {
                        "protocol_id": protocol["protocol_id"],
                        "checkpoint_dir": checkpoint_metadata["checkpoint_dir"],
                        "checkpoint_path": checkpoint_metadata.get(
                            "checkpoint_path", checkpoint_metadata["checkpoint_dir"]
                        ),
                        "training_experiment_id": checkpoint_metadata.get("training_experiment_id"),
                        "training_seed": checkpoint_metadata.get("training_seed"),
                        "model_name": checkpoint_metadata["model_name"],
                        "architecture": json.dumps(architecture, sort_keys=True),
                        "gpu_model": gpu_model,
                        "precision": checkpoint_metadata.get("precision", "float32"),
                        "train_rho_x": checkpoint_metadata.get("train_rho_x"),
                        "test_rho_x": rho,
                        "train_rho_e": checkpoint_metadata.get("train_rho_e"),
                        "test_rho_e": 0.0,
                        "protocol": (
                            "matched"
                            if checkpoint_metadata.get("train_rho_x") == rho
                            and checkpoint_metadata.get("train_rho_x_after") is None
                            and checkpoint_metadata.get("feature_change_point") is None
                            and checkpoint_metadata.get("train_rho_e", 0.0) == 0.0
                            and checkpoint_metadata.get("train_rho_e_after") is None
                            and checkpoint_metadata.get("noise_change_point") is None
                            else "shift"
                        ),
                        "train_snr": checkpoint_metadata.get("train_snr"),
                        "test_snr": snr,
                        "d": n_dims,
                        "k": context_length,
                        "context_length": context_length,
                        "k_over_d": context_length / n_dims,
                        "rho": rho,
                        "snr": snr,
                        "noise_std": 1.0 / snr,
                        "eval_seed": eval_seed,
                        "n_eval": n_eval,
                    }
                    for aggregate in analyzer.rows():
                        rows.append({
                            **condition,
                            **aggregate,
                            "metadata_json": metadata_json,
                        })
    _mechanism_csv(rows, output_csv)
    return {
        "status": "complete",
        "protocol": protocol,
        "rows_written": len(rows),
        "output_csv": str(Path(output_csv)),
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    matching = commands.add_parser("parameter-match", help="Find exact parameter-count depth matches")
    matching.add_argument("--output-dir", type=Path, default=Path("results/bo_matrix/parameter_match"))
    matching.add_argument("--target-width", type=int, default=256)
    matching.add_argument("--target-depth", type=int, default=12)
    matching.add_argument("--target-heads", type=int, default=8)
    matching.add_argument("--depths", type=int, nargs="+", default=[2, 4, 6, 12])
    matching.add_argument("--width-min", type=int, default=32)
    matching.add_argument("--width-max", type=int, default=768)
    matching.add_argument("--width-step", type=int, default=8)
    matching.add_argument("--n-dims", type=int, default=20)
    matching.add_argument("--max-context", type=int, default=80)
    matching.add_argument("--dry-run", action="store_true")

    mechanism = commands.add_parser("mechanism", help="Aggregate attention diagnostics for a checkpoint")
    mechanism.add_argument("--checkpoint-dir", type=Path, required=True)
    mechanism.add_argument("--output-csv", type=Path, required=True)
    mechanism.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.6, 0.9])
    mechanism.add_argument("--snrs", type=float, nargs="+", default=[0.8, 3.2])
    mechanism.add_argument("--context-lengths", type=int, nargs="+", default=[20, 40, 80])
    mechanism.add_argument("--eval-seeds", type=int, nargs="+", default=[1001, 1002, 1003])
    mechanism.add_argument("--n-eval", type=int, default=32)
    mechanism.add_argument("--batch-size", type=int, default=16)
    mechanism.add_argument("--device", default="auto")
    mechanism.add_argument(
        "--query-positions",
        choices=("last_x", "all_x", "all_y", "all"),
        default="last_x",
    )
    mechanism.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "parameter-match":
        target = ArchitectureSpec(
            args.target_width,
            args.target_depth,
            args.target_heads,
            sweep_family="standard",
        )
        result = parameter_match_report(
            output_dir=args.output_dir,
            target=target,
            depths=args.depths,
            width_min=args.width_min,
            width_max=args.width_max,
            width_step=args.width_step,
            n_dims=args.n_dims,
            max_context=args.max_context,
            dry_run=args.dry_run,
        )
        summary = {
            "status": "planned" if result.dry_run else "complete",
            "report_id": result.plan["report_id"],
            "search_model_instantiations": result.plan["search_model_instantiations"],
            "match_count": len(result.matches),
            "training_experiment_count": len(result.experiments),
            "json_path": str(result.json_path),
            "csv_path": str(result.csv_path),
        }
    else:
        summary = mechanism_evaluation(
            checkpoint_dir=args.checkpoint_dir,
            output_csv=args.output_csv,
            rhos=args.rhos,
            snrs=args.snrs,
            context_lengths=args.context_lengths,
            eval_seeds=args.eval_seeds,
            n_eval=args.n_eval,
            batch_size=args.batch_size,
            device=args.device,
            query_positions=args.query_positions,
            dry_run=args.dry_run,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
