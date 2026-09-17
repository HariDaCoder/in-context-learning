"""Plot BO sweeps with SNR legends, rho panels and observed SNR brackets.

    python src/bo_plot.py results/bo_smoke.json --out-dir results/bo_figures

Intervals are approximate normal 95% CIs across evaluation-seed means, within
one checkpoint. Model-training seeds and different protocols are never pooled.
One evaluation seed has no CI. Phase figures require a complete sampled grid.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


ROW_DISTRIBUTION_FIELDS = {
    "train_rho_x", "test_rho_x", "train_rho_e", "test_rho_e",
    "protocol", "evaluation_protocol", "train_seed", "eval_seed", "train_snr",
    "test_snr", "architecture", "gpu_model", "precision",
    "checkpoint_path", "train_distribution_id", "train_distribution",
    "test_distribution", "context_length",
}


def seed_summary(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    if not values:
        return {"mean": None, "ci95": None, "n_seeds": 0}
    return {"mean": statistics.mean(values), "n_seeds": len(values),
            "ci95": 1.96 * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None}


def _snr_value(value):
    return float(value)


def _snr_json(value):
    return "inf" if math.isinf(value) else value


def critical_snr_brackets(points, target=0.5):
    """Report every adjacent observed crossing, without monotonicity or extrapolation.

    An exact sampled equality is an observed point, not an interpolated critical
    SNR estimate. 'No crossing' can mean the entire sampled grid is on one side.
    Missing probabilities split the grid; they must not be bridged.
    """
    if not 0 < target < 1:
        raise ValueError("target must lie strictly between 0 and 1")
    points = sorted((float(snr), probability) for snr, probability in points)
    if len({snr for snr, _ in points}) != len(points):
        raise ValueError("SNR grid contains duplicate values")
    brackets, exact = [], []
    for snr, probability in points:
        if probability is not None and math.isfinite(probability) and probability == target:
            exact.append(_snr_json(snr))
    for (lower, p_lower), (upper, p_upper) in zip(points, points[1:]):
        if p_lower is None or p_upper is None:
            continue
        if not math.isfinite(p_lower) or not math.isfinite(p_upper):
            continue
        if (p_lower - target) * (p_upper - target) < 0:
            brackets.append({"lower_snr": _snr_json(lower), "upper_snr": _snr_json(upper),
                             "lower_probability": p_lower, "upper_probability": p_upper,
                             "direction": "upward" if p_upper > p_lower else "downward"})
    n_observed = sum(p is not None and math.isfinite(p) for _, p in points)
    status = "observed_crossings" if brackets else "sampled_equality" if exact else "no_observed_crossing"
    if n_observed < 2 and not exact:
        status = "insufficient_grid"
    return {"target_probability": target, "status": status, "brackets": brackets,
            "exact_sampled_points": exact, "n_observed": n_observed,
            "unique_critical_snr_estimated": False}


def _midpoint_snr(lower, upper):
    """Use a geometric midpoint on the positive SNR axis when it is finite."""
    if lower > 0 and upper > 0 and math.isfinite(lower) and math.isfinite(upper):
        return math.sqrt(lower * upper)
    return None


def propose_snr_boundary(points, target=0.5, expansion_factor=2.0):
    """Suggest the next SNR samples without assuming a monotone phase curve.

    Every observed crossing gets its own refinement point.  When no crossing
    was sampled, the closest observed probability anchors either an outward
    grid expansion (at an edge) or local densification (in the interior).
    This is a sampling proposal, not an interpolated critical-SNR estimate.
    """
    if not math.isfinite(expansion_factor) or expansion_factor <= 1:
        raise ValueError("expansion_factor must be finite and greater than one")
    analysis = critical_snr_brackets(points, target)
    observed = sorted(
        (float(snr), float(probability))
        for snr, probability in points
        if probability is not None and math.isfinite(probability)
    )
    suggestions = []
    closest = []
    if analysis["brackets"]:
        action = "refine_all_observed_crossings"
        for bracket in analysis["brackets"]:
            midpoint = _midpoint_snr(float(bracket["lower_snr"]), float(bracket["upper_snr"]))
            if midpoint is not None:
                suggestions.append(midpoint)
    elif analysis["exact_sampled_points"]:
        action = "refine_around_sampled_equalities"
        exact = {float(value) for value in analysis["exact_sampled_points"]}
        for index, (snr, _) in enumerate(observed):
            if snr not in exact:
                continue
            if index:
                midpoint = _midpoint_snr(observed[index - 1][0], snr)
                if midpoint is not None:
                    suggestions.append(midpoint)
            if index + 1 < len(observed):
                midpoint = _midpoint_snr(snr, observed[index + 1][0])
                if midpoint is not None:
                    suggestions.append(midpoint)
    elif observed:
        distance = min(abs(probability - target) for _, probability in observed)
        closest = [
            {"snr": _snr_json(snr), "probability": probability,
             "distance_to_target": abs(probability - target)}
            for snr, probability in observed
            if abs(probability - target) == distance
        ]
        action = "expand_from_closest_edge"
        for item in closest:
            snr = float(item["snr"])
            index = next(i for i, point in enumerate(observed) if point[0] == snr)
            if index == 0 and math.isfinite(snr):
                suggestions.append(snr / expansion_factor)
            if index == len(observed) - 1 and math.isfinite(snr):
                suggestions.append(snr * expansion_factor)
            if 0 < index < len(observed) - 1:
                action = "densify_around_closest_interior_point"
                for neighbor in (observed[index - 1][0], observed[index + 1][0]):
                    midpoint = _midpoint_snr(min(snr, neighbor), max(snr, neighbor))
                    if midpoint is not None:
                        suggestions.append(midpoint)
    else:
        action = "collect_initial_grid"
    suggestions = sorted({_snr_json(value) for value in suggestions}, key=float)
    return {
        **analysis,
        "action": action,
        "closest_sampled_points": closest,
        "suggested_snrs": suggestions,
        "proposal_is_not_boundary_estimate": True,
    }


def _validate_distribution_row(record, path):
    missing = ROW_DISTRIBUTION_FIELDS - set(record)
    if missing:
        raise ValueError(f"{path}: incomplete matched/shift metadata: {sorted(missing)}")
    if record["evaluation_protocol"] not in {"matched", "shift"}:
        raise ValueError(f"{path}: evaluation_protocol must be matched or shift")
    if record["protocol"] != record["evaluation_protocol"]:
        raise ValueError(f"{path}: protocol/evaluation_protocol disagree")
    aliases = (
        ("train_seed", "training_seed"), ("eval_seed", "seed"),
        ("test_rho_x", "rho"), ("test_snr", "snr"),
        ("checkpoint_path", "checkpoint_id"), ("context_length", "k"),
    )
    for new, old in aliases:
        if record[new] != record[old]:
            raise ValueError(f"{path}: inconsistent row metadata {new}/{old}")
    if record["evaluation_protocol"] == "matched":
        for train, test in (("train_rho_x", "test_rho_x"),
                            ("train_rho_e", "test_rho_e")):
            if record[train] != record[test]:
                raise ValueError(f"{path}: matched row has unequal {train}/{test}")


def _metadata_mode(document, path):
    flags = [ROW_DISTRIBUTION_FIELDS <= set(record) for record in document["records"]]
    partial = [bool(ROW_DISTRIBUTION_FIELDS & set(record)) and not complete
               for record, complete in zip(document["records"], flags)]
    if any(partial) or flags and any(flags) != all(flags):
        raise ValueError(f"{path}: matched/shift rows cannot be merged without complete metadata")
    declared = document.get("metadata", {}).get("distribution_metadata_version")
    if declared is not None and (declared != 1 or not all(flags)):
        raise ValueError(f"{path}: invalid distribution_metadata_version")
    return "matched_shift_v1" if flags and all(flags) else "legacy"


def load_groups(paths):
    groups = {}
    input_mode = None
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
        if document.get("schema_version") != 1:
            raise ValueError(f"{path}: unsupported result schema")
        mode = _metadata_mode(document, path)
        if input_mode is not None and mode != input_mode:
            raise ValueError(
                "Cannot combine legacy results with matched/shift-aware results; "
                "the legacy rows lack train/test distribution metadata"
            )
        input_mode = mode
        for record in document["records"]:
            if mode == "matched_shift_v1":
                _validate_distribution_row(record, path)
            # The full checkpoint path identifies a trained model. Equal model
            # architecture names do not imply equal training replicates.
            key = (record["protocol_id"], record["model"], record["checkpoint_id"],
                   record["training_seed"], record["d"])
            if mode == "matched_shift_v1" and record["checkpoint_id"] is not None:
                key += (record["train_distribution_id"],)
            identity_names = ("protocol_id", "model", "checkpoint_id", "training_seed", "d",
                              "train_distribution_id")[:len(key)]
            if key not in groups:
                groups[key] = {"identity": dict(zip(
                    identity_names, key)),
                    "protocol": document["metadata"]["protocol"],
                    "distribution_metadata_mode": mode, "cells": defaultdict(dict)}
            group = groups[key]
            if group["protocol"] != document["metadata"]["protocol"]:
                raise ValueError(f"{path}: same protocol_id has inconsistent protocol metadata")
            cell = (record["rho"], _snr_value(record["snr"]), record["k"])
            if record["seed"] in group["cells"][cell]:
                raise ValueError(f"Duplicate evaluation seed {record['seed']} for {key}, {cell}; do not pool repeated outputs")
            group["cells"][cell][record["seed"]] = record
    if not groups:
        raise ValueError("No result records found")
    return list(groups.values())


def _cell_summary(records, metric):
    return seed_summary(record["metrics"].get(metric, {}).get("mean") for record in records.values())


def _cell_evaluation_protocol(records):
    values = {record.get("evaluation_protocol") for record in records.values()}
    if values == {None} or not values:
        return None
    if None in values or len(values) != 1:
        raise ValueError("A result cell mixes matched/shift rows or lacks protocol metadata")
    return next(iter(values))


def summarize_group(group, target):
    cells = group["cells"]
    summary = []
    for (rho, snr, k), records in sorted(cells.items()):
        metrics = sorted(set().union(*(record["metrics"] for record in records.values())))
        cell = {"rho": rho, "snr": _snr_json(snr), "k": k,
                "metrics": {metric: _cell_summary(records, metric) for metric in metrics}}
        evaluation_protocol = _cell_evaluation_protocol(records)
        if evaluation_protocol is not None:
            cell["evaluation_protocol"] = evaluation_protocol
        summary.append(cell)
    snrs = sorted({snr for _, snr, _ in cells})
    brackets = []
    for rho, k in sorted({(rho, k) for rho, _, k in cells}):
        points = [(snr, _cell_summary(cells.get((rho, snr, k), {}), "bo_candidate")["mean"])
                  for snr in snrs]
        brackets.append({"rho": rho, "k": k, **critical_snr_brackets(points, target)})
    proposals = []
    for rho, k in sorted({(rho, k) for rho, _, k in cells}):
        points = [(snr, _cell_summary(cells.get((rho, snr, k), {}), "bo_candidate")["mean"])
                  for snr in snrs]
        proposals.append({"rho": rho, "k": k, **propose_snr_boundary(points, target)})
    return {**group["identity"], "protocol": group["protocol"],
            "seed_summaries": summary, "critical_snr_observations": brackets,
            "boundary_suggestions": proposals}


def summarize_training_replicates(groups, target):
    """Aggregate hierarchically: evaluation seeds, then training runs.

    Each checkpoint first contributes one mean per condition, regardless of its
    number of evaluation tasks/seeds. Confidence intervals are then computed
    across independent checkpoint runs of the same architecture and protocol.
    Baselines have no training replicate and are omitted here.
    """
    buckets = defaultdict(list)
    for group in groups:
        identity = group["identity"]
        if identity["checkpoint_id"] is None:
            continue
        key = (identity["protocol_id"], identity["model"], identity["d"],
               identity.get("train_distribution_id"))
        buckets[key].append(group)

    outputs = []
    for (protocol_id, model, d, train_distribution_id), replicas in sorted(
        buckets.items(), key=lambda item: tuple("" if value is None else str(value) for value in item[0])
    ):
        checkpoint_ids = [replica["identity"]["checkpoint_id"] for replica in replicas]
        if len(set(checkpoint_ids)) != len(checkpoint_ids):
            raise ValueError(f"Duplicate checkpoint in training-replicate group: {model}, d={d}")
        training_seeds = [replica["identity"]["training_seed"] for replica in replicas]
        if len(replicas) > 1 and any(seed is None for seed in training_seeds):
            outputs.append({
                "protocol_id": protocol_id,
                "model": model,
                "d": d,
                "status": "not_aggregated_missing_training_seed_metadata",
                "checkpoint_ids": checkpoint_ids,
                "training_seeds": training_seeds,
            })
            continue
        present_seeds = [seed for seed in training_seeds if seed is not None]
        if len(present_seeds) != len(set(present_seeds)):
            raise ValueError(f"Duplicate training seed in replicate group: {model}, d={d}")
        all_cells = sorted(set().union(*(replica["cells"] for replica in replicas)))
        cell_summaries = []
        for rho, snr, k in all_cells:
            metric_names = sorted(set().union(*(
                set().union(*(record["metrics"] for record in replica["cells"].get((rho, snr, k), {}).values()))
                for replica in replicas
            )))
            metrics = {}
            for metric in metric_names:
                per_checkpoint = [
                    _cell_summary(replica["cells"].get((rho, snr, k), {}), metric)["mean"]
                    for replica in replicas
                ]
                summary = seed_summary(per_checkpoint)
                summary["n_training_runs"] = summary.pop("n_seeds")
                metrics[metric] = summary
            cell = {"rho": rho, "snr": _snr_json(snr), "k": k, "metrics": metrics}
            cell_protocols = {
                _cell_evaluation_protocol(replica["cells"].get((rho, snr, k), {}))
                for replica in replicas if replica["cells"].get((rho, snr, k), {})
            }
            if len(cell_protocols) > 1:
                raise ValueError("Training replicas disagree on matched/shift protocol for one cell")
            if cell_protocols and None not in cell_protocols:
                cell["evaluation_protocol"] = next(iter(cell_protocols))
            cell_summaries.append(cell)

        snrs = sorted({snr for _, snr, _ in all_cells})
        brackets = []
        lookup = {(cell["rho"], _snr_value(cell["snr"]), cell["k"]): cell for cell in cell_summaries}
        for rho, k in sorted({(rho, k) for rho, _, k in all_cells}):
            points = [
                (snr, lookup.get((rho, snr, k), {}).get("metrics", {}).get(
                    "bo_candidate", {}
                ).get("mean"))
                for snr in snrs
            ]
            brackets.append({"rho": rho, "k": k, **critical_snr_brackets(points, target)})
        outputs.append({
            "protocol_id": protocol_id,
            "model": model,
            "d": d,
            "train_distribution_id": train_distribution_id,
            "protocol": replicas[0]["protocol"],
            "checkpoint_ids": checkpoint_ids,
            "training_seeds": training_seeds,
            "cells": cell_summaries,
            "critical_snr_observations": brackets,
            "boundary_suggestions": [
                {"rho": rho, "k": k, **propose_snr_boundary([
                    (snr, lookup.get((rho, snr, k), {}).get("metrics", {}).get(
                        "bo_candidate", {}
                    ).get("mean")) for snr in snrs
                ], target)}
                for rho, k in sorted({(rho, k) for rho, _, k in all_cells})
            ],
        })
    return outputs


def _filename(group):
    label = re.sub(r"[^a-zA-Z0-9_-]+", "_", group["identity"]["model"]).strip("_")
    suffix = hashlib.sha256(json.dumps(group["identity"], sort_keys=True).encode()).hexdigest()[:10]
    return f"{label}_d{group['identity']['d']}_{suffix}"


def _axis_value(record, axis_type):
    if axis_type == "k":
        return record["k"]
    if axis_type == "k_over_d":
        return record.get("k_over_d", record["k"] / record["d"])
    if axis_type == "effective_rank_over_d":
        return record.get(
            "effective_rank_over_d", record["effective_rank"] / record["d"]
        )
    if axis_type == "effective_information":
        return record.get(
            "effective_information", record["effective_rank"] * _snr_value(record["snr"]) ** 2
        )
    raise ValueError(f"Unknown axis type: {axis_type}")


def _axis_label(axis_type):
    return {
        "k": "Context examples k",
        "k_over_d": "Context ratio k / d",
        "effective_rank_over_d": "Effective context ratio k_eff / d",
        "effective_information": "Effective information k_eff * SNR^2",
    }[axis_type]


def _plot_observed_boundaries(group, out_dir, stem, title, target):
    """Plot every observed SNR crossing as an interval, never as a fitted law."""
    import matplotlib.pyplot as plt

    cells = group["cells"]
    snrs = sorted({snr for _, snr, _ in cells})
    points = []
    for rho, k in sorted({(rho, k) for rho, _, k in cells}):
        probabilities = [
            (snr, _cell_summary(cells.get((rho, snr, k), {}), "bo_candidate")["mean"])
            for snr in snrs
        ]
        observation = critical_snr_brackets(probabilities, target)
        records = next(
            (records for (cell_rho, _, cell_k), records in cells.items()
             if cell_rho == rho and cell_k == k and records),
            None,
        )
        if records is None:
            continue
        record = next(iter(records.values()))
        effective_ratio = _axis_value(record, "effective_rank_over_d")
        for bracket_index, bracket in enumerate(observation["brackets"]):
            lower, upper = float(bracket["lower_snr"]), float(bracket["upper_snr"])
            midpoint = _midpoint_snr(lower, upper)
            if midpoint is not None:
                points.append({
                    "rho": rho, "k": k, "effective_ratio": effective_ratio,
                    "lower": lower, "upper": upper, "midpoint": midpoint,
                    "kind": f"crossing {bracket_index + 1}",
                })
        for exact in observation["exact_sampled_points"]:
            exact = float(exact)
            points.append({
                "rho": rho, "k": k, "effective_ratio": effective_ratio,
                "lower": exact, "upper": exact, "midpoint": exact,
                "kind": "sampled equality",
            })
    if not points:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = plt.get_cmap("viridis")
    ks = sorted({point["k"] for point in points})
    for index, k in enumerate(ks):
        color = colors(index / max(len(ks) - 1, 1))
        selected = [point for point in points if point["k"] == k]
        for axis, x_key in zip(axes, ("rho", "effective_ratio")):
            xs = [point[x_key] for point in selected]
            ys = [point["midpoint"] for point in selected]
            lower_errors = [point["midpoint"] - point["lower"] for point in selected]
            upper_errors = [point["upper"] - point["midpoint"] for point in selected]
            axis.errorbar(
                xs, ys, yerr=(lower_errors, upper_errors), fmt="o", capsize=3,
                color=color, label=f"k={k}", alpha=0.85,
            )
    axes[0].set_xlabel("Feature dependence rho")
    axes[1].set_xlabel("Effective context ratio k_eff / d")
    for axis in axes:
        axis.set_ylabel("Observed critical-SNR interval")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle(
        title + f"\nAll observed BO p={target:g} crossings; markers are geometric interval centers",
        fontsize=10,
    )
    fig.tight_layout()
    path = out_dir / f"{stem}_observed_phase_boundaries.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return str(path)


def plot_group(group, out_dir, metric="clean_query_mse", target=0.5, log_y=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = group["cells"]
    rhos = sorted({rho for rho, _, _ in cells})
    snrs = sorted({snr for _, snr, _ in cells})
    ks = sorted({k for _, _, k in cells})
    identity = group["identity"]
    title = f"{identity['model']}, d={identity['d']}"
    protocol = group["protocol"]
    title += f", noise rho={protocol['noise_rho']:g}"
    if protocol["feature_rho_after"] is not None:
        title += (
            f", feature change -> {protocol['feature_rho_after']:g}"
            f" at {protocol['feature_change_point']}"
        )
    if protocol["noise_rho_after"] is not None:
        title += (
            f", noise change -> {protocol['noise_rho_after']:g}"
            f" at {protocol['noise_change_point']}"
        )
    if identity["checkpoint_id"] is not None:
        title += f", checkpoint={Path(identity['checkpoint_id']).name}, training seed={identity['training_seed']}"
    stem = _filename(group)
    paths = []
    colors = plt.get_cmap("viridis")
    for axis_type in ("k", "k_over_d", "effective_rank_over_d", "effective_information"):
        fig, axes_grid = plt.subplots(1, len(rhos), figsize=(4.4 * len(rhos), 3.9), squeeze=False)
        for ax, rho in zip(axes_grid[0], rhos):
            for index, snr in enumerate(snrs):
                xs, means, errors = [], [], []
                for k in ks:
                    records = cells.get((rho, snr, k), {})
                    summary = _cell_summary(records, metric)
                    if summary["mean"] is None:
                        continue
                    record = next(iter(records.values()))
                    x = _axis_value(record, axis_type)
                    xs.append(x)
                    means.append(summary["mean"])
                    errors.append(summary["ci95"])
                color = colors(index / max(len(snrs) - 1, 1))
                label = "infinity (no noise)" if math.isinf(snr) else f"{snr:g}"
                ax.plot(xs, means, marker="o", color=color, label=label)
                valid = [(x, mean, error) for x, mean, error in zip(xs, means, errors) if error is not None]
                if valid:
                    ci_x, ci_mean, ci_error = zip(*valid)
                    ax.errorbar(ci_x, ci_mean, yerr=ci_error, fmt="none", capsize=3, color=color)
            ax.set_title(f"feature rho = {rho:g}")
            ax.set_xlabel(_axis_label(axis_type))
            ax.set_ylabel(metric.replace("_", " "))
            if log_y:
                ax.set_yscale("symlog", linthresh=1e-4)
            ax.grid(alpha=0.25)
            ax.legend(title="Amplitude SNR", fontsize=8)
        fig.suptitle(title + "\n95% normal CIs across evaluation seed means; each checkpoint separate", fontsize=10)
        fig.tight_layout()
        path = out_dir / f"{stem}_{metric}_vs_{axis_type}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(str(path))

    # Collapse view: each SNR gets a panel and each dependence strength is a
    # separate curve against k_eff/d. Agreement or disagreement is visible
    # without fitting the hypothesized scaling law to the same observations.
    fig, axes_grid = plt.subplots(1, len(snrs), figsize=(4.4 * len(snrs), 3.9), squeeze=False)
    rho_colors = plt.get_cmap("plasma")
    for ax, snr in zip(axes_grid[0], snrs):
        for index, rho in enumerate(rhos):
            xs, means, errors = [], [], []
            for k in ks:
                records = cells.get((rho, snr, k), {})
                summary = _cell_summary(records, metric)
                if summary["mean"] is None:
                    continue
                record = next(iter(records.values()))
                xs.append(record.get(
                    "effective_rank_over_d", record["effective_rank"] / identity["d"]
                ))
                means.append(summary["mean"])
                errors.append(summary["ci95"])
            ordering = sorted(range(len(xs)), key=xs.__getitem__)
            xs = [xs[i] for i in ordering]
            means = [means[i] for i in ordering]
            errors = [errors[i] for i in ordering]
            color = rho_colors(index / max(len(rhos) - 1, 1))
            ax.plot(xs, means, marker="o", color=color, label=f"feature rho={rho:g}")
            valid = [(x, mean, error) for x, mean, error in zip(xs, means, errors) if error is not None]
            if valid:
                ci_x, ci_mean, ci_error = zip(*valid)
                ax.errorbar(ci_x, ci_mean, yerr=ci_error, fmt="none", capsize=3, color=color)
        snr_label = "infinity" if math.isinf(snr) else f"{snr:g}"
        ax.set_title(f"Amplitude SNR = {snr_label}")
        ax.set_xlabel("Effective context ratio k_eff / d")
        ax.set_ylabel(metric.replace("_", " "))
        if log_y:
            ax.set_yscale("symlog", linthresh=1e-4)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(title + "\nEffective-rank collapse diagnostic; 95% CIs across evaluation seeds", fontsize=10)
    fig.tight_layout()
    path = out_dir / f"{stem}_{metric}_collapse.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(str(path))

    # A metadata-aware diagnostic keeps matched and distribution-shift cells
    # visibly distinct.  Points are cell means (evaluation seeds are averaged
    # within a cell); no trend line is fitted across heterogeneous conditions.
    protocol_points = defaultdict(list)
    for records in cells.values():
        evaluation_protocol = _cell_evaluation_protocol(records)
        if evaluation_protocol is None:
            continue
        summary = _cell_summary(records, metric)
        if summary["mean"] is None:
            continue
        record = next(iter(records.values()))
        x = record.get("effective_rank_over_d", record["effective_rank"] / identity["d"])
        protocol_points[evaluation_protocol].append((x, summary["mean"]))
    if {"matched", "shift"} <= set(protocol_points):
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        markers = {"matched": "o", "shift": "x"}
        for evaluation_protocol in ("matched", "shift"):
            points = protocol_points[evaluation_protocol]
            ax.scatter([point[0] for point in points], [point[1] for point in points],
                       marker=markers[evaluation_protocol], label=evaluation_protocol, alpha=0.8)
        ax.set_xlabel("Effective context ratio k_eff / d")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(title + "\nMatched versus shifted evaluation cells", fontsize=10)
        if log_y:
            ax.set_yscale("symlog", linthresh=1e-4)
        ax.grid(alpha=0.25)
        ax.legend(title="Evaluation protocol")
        fig.tight_layout()
        path = out_dir / f"{stem}_{metric}_matched_vs_shift.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(str(path))

    boundary_path = _plot_observed_boundaries(group, out_dir, stem, title, target)
    if boundary_path is not None:
        paths.append(boundary_path)

    # Do not fill missing conditions or invent interpolated phase boundaries.
    complete = all((rho, snr, k) in cells for rho in rhos for snr in snrs for k in ks)
    if not complete:
        print(f"Skipped phase heatmaps for {stem}: incomplete rho/SNR/k grid")
        return paths
    fig, axes_grid = plt.subplots(2, len(rhos), figsize=(4.5 * len(rhos), 7), squeeze=False)
    for row, heat_metric in enumerate(("clean_query_mse", "bo_candidate")):
        values = [_cell_summary(records, heat_metric)["mean"] for records in cells.values()]
        finite = [value for value in values if value is not None and math.isfinite(value)]
        upper = 1 if heat_metric == "bo_candidate" else max(finite, default=1)
        for column, rho in enumerate(rhos):
            ax = axes_grid[row][column]
            grid = [[_cell_summary(cells[(rho, snr, k)], heat_metric)["mean"] for k in ks] for snr in snrs]
            if any(value is None for line in grid for value in line):
                ax.text(0.5, 0.5, "Unavailable metric cells", ha="center", va="center")
                ax.set_axis_off()
                continue
            plot = ax.imshow(grid, origin="lower", aspect="auto", vmin=0, vmax=max(upper, 1e-15), cmap="viridis")
            ax.set_xticks(range(len(ks)), [f"{k / identity['d']:g}" for k in ks])
            ax.set_yticks(range(len(snrs)), ["inf" if math.isinf(snr) else f"{snr:g}" for snr in snrs])
            ax.set_xlabel("k / d (sampled grid)")
            ax.set_ylabel("Amplitude SNR (sampled grid)")
            ax.set_title(f"feature rho = {rho:g}: {heat_metric.replace('_', ' ')}", fontsize=10)
            if heat_metric == "clean_query_mse":
                # The fit contour exposes where interpolation begins. Coarse
                # Transformer scans report duplicate fit; probe scans and
                # classical estimators report context fit.
                fit_grid = []
                for snr in snrs:
                    fit_row = []
                    for k in ks:
                        records = cells[(rho, snr, k)]
                        context_fit = _cell_summary(records, "context_fit_mse")["mean"]
                        if context_fit is None:
                            context_fit = _cell_summary(records, "duplicate_context_fit_mse")["mean"]
                        fit_row.append(context_fit)
                    fit_grid.append(fit_row)
                fit_values = [value for fit_row in fit_grid for value in fit_row
                              if value is not None and math.isfinite(value)]
                threshold = protocol.get("fit_threshold")
                if (threshold is not None and fit_values
                        and min(fit_values) < threshold < max(fit_values)
                        and all(value is not None and math.isfinite(value)
                                for fit_row in fit_grid for value in fit_row)):
                    ax.contour(range(len(ks)), range(len(snrs)), fit_grid,
                               levels=[threshold], colors="white", linewidths=1.4)
                    ax.plot([], [], color="white", label=f"fit MSE={threshold:g}")
                    ax.legend(fontsize=7, loc="upper right")
            elif heat_metric == "bo_candidate":
                bo_values = [value for row in grid for value in row
                             if value is not None and math.isfinite(value)]
                if bo_values and min(bo_values) < target < max(bo_values):
                    ax.contour(range(len(ks)), range(len(snrs)), grid,
                               levels=[target], colors="white", linewidths=1.4)
                    ax.plot([], [], color="white", label=f"BO p={target:g}")
                    ax.legend(fontsize=7, loc="upper right")
            fig.colorbar(plot, ax=ax, shrink=0.8)
    fig.suptitle(title + "\nBO candidate uses direct duplicate-fit and clean-query thresholds", fontsize=10)
    fig.tight_layout()
    path = out_dir / f"{stem}_phase.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(str(path))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="JSON outputs; duplicate seed/condition rows are rejected")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--metric", default="clean_query_mse")
    parser.add_argument("--target-probability", type=float, default=0.5)
    parser.add_argument("--summary-only", action="store_true", help="Write observed brackets/seed summaries without matplotlib")
    parser.add_argument("--log-y", action="store_true", help="Use a symlog y-axis for curve/collapse plots")
    args = parser.parse_args()
    if not 0 < args.target_probability < 1:
        parser.error("--target-probability must lie strictly between 0 and 1")
    groups = load_groups(args.inputs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries, figures = [], []
    for group in groups:
        summaries.append(summarize_group(group, args.target_probability))
        if not args.summary_only:
            figures.extend(plot_group(
                group, out_dir, args.metric, args.target_probability, args.log_y
            ))
    output = {
        "schema_version": 1, "groups": summaries,
        "training_replicates": summarize_training_replicates(groups, args.target_probability),
        "figures": figures,
        "uncertainty": "approximate normal 95% CI across evaluation-seed means per checkpoint; absent for fewer than two seeds",
        "training_replicate_uncertainty": "each checkpoint is averaged over its evaluation seeds first; approximate 95% CI is then across equal-weight checkpoint runs of one architecture/protocol",
        "critical_snr_method": "all observed adjacent crossings plus sampled equalities; missing cells split grid; no interpolation, extrapolation, or monotonicity assumption",
        "boundary_proposal_method": "refine every observed crossing; otherwise refine equalities or expand/densify from the closest sampled probability",
    }
    destination = out_dir / "seed_summary_and_snr_brackets.json"
    destination.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    boundary_output = {
        "schema_version": 1,
        "target_probability": args.target_probability,
        "method": output["boundary_proposal_method"],
        "warning": "Suggested SNRs are follow-up samples, not estimated phase boundaries.",
        "groups": [
            {
                "identity": {key: summary.get(key) for key in (
                    "protocol_id", "model", "checkpoint_id", "training_seed", "d",
                    "train_distribution_id"
                ) if key in summary},
                "suggestions": summary["boundary_suggestions"],
            }
            for summary in summaries
        ],
    }
    boundary_destination = out_dir / "boundary_suggestions.json"
    boundary_destination.write_text(
        json.dumps(boundary_output, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Wrote {len(figures)} figures, {destination}, and {boundary_destination}")


if __name__ == "__main__":
    main()
