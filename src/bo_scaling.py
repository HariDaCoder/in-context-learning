"""Exploratory critical-SNR scaling analysis with explicit exclusions.

The analysis never chooses among multiple observed crossings. A condition is
eligible for the log-linear fit only when the sampled BO-frequency curve has
one crossing interval or one exact equality. Interval centers are geometric
midpoints and remain labelled as such; they are not claimed as exact phase
transitions.
"""

import json
import math
from pathlib import Path
import statistics

import torch

from bo_plot import _cell_summary, critical_snr_brackets, load_groups


def _architecture(group):
    for records in group["cells"].values():
        if records:
            architecture = next(iter(records.values())).get("architecture", {})
            return architecture if isinstance(architecture, dict) else {}
    return {}


def boundary_observations(groups, target=0.5):
    observations, exclusions = [], []
    for group in groups:
        architecture = _architecture(group)
        cells = group["cells"]
        snrs = sorted({snr for _, snr, _ in cells})
        for rho, k in sorted({(rho, k) for rho, _, k in cells}):
            probabilities = [
                (snr, _cell_summary(cells.get((rho, snr, k), {}), "bo_candidate")["mean"])
                for snr in snrs
            ]
            boundary = critical_snr_brackets(probabilities, target)
            condition = {
                **group["identity"],
                "rho": rho,
                "k": k,
                "target_probability": target,
            }
            candidates = []
            for bracket in boundary["brackets"]:
                lower = float(bracket["lower_snr"])
                upper = float(bracket["upper_snr"])
                candidates.append({
                    "critical_snr": math.sqrt(lower * upper),
                    "lower_snr": lower,
                    "upper_snr": upper,
                    "source": "single_observed_bracket_geometric_midpoint",
                })
            for exact in boundary["exact_sampled_points"]:
                exact = float(exact)
                candidates.append({
                    "critical_snr": exact,
                    "lower_snr": exact,
                    "upper_snr": exact,
                    "source": "exact_sampled_equality",
                })
            if len(candidates) != 1:
                exclusions.append({
                    **condition,
                    "reason": "no_crossing" if not candidates else "multiple_crossings",
                    "n_candidates": len(candidates),
                    "boundary": boundary,
                })
                continue
            records = next(
                records for (cell_rho, _, cell_k), records in cells.items()
                if cell_rho == rho and cell_k == k and records
            )
            record = next(iter(records.values()))
            effective_rank = float(record["effective_rank"])
            n_head = architecture.get("n_head")
            n_layer = architecture.get("n_layer")
            if not all(isinstance(value, int) and value > 0 for value in (n_head, n_layer)):
                exclusions.append({**condition, "reason": "missing_transformer_architecture"})
                continue
            observations.append({
                **condition,
                **candidates[0],
                "effective_rank": effective_rank,
                "effective_rank_over_d": effective_rank / group["identity"]["d"],
                "n_head": n_head,
                "n_layer": n_layer,
                "n_embd": architecture.get("n_embd"),
            })
    return observations, exclusions


def fit_log_linear(observations):
    """Fit c-alpha*log(k_eff)-beta*log(H)-gamma*log(L)."""
    if len(observations) < 4:
        return {"status": "insufficient_observations", "n_observations": len(observations)}
    design = torch.tensor([
        [
            1.0,
            -math.log(row["effective_rank"]),
            -math.log(row["n_head"]),
            -math.log(row["n_layer"]),
        ]
        for row in observations
    ], dtype=torch.float64)
    response = torch.tensor(
        [math.log(row["critical_snr"]) for row in observations], dtype=torch.float64
    ).unsqueeze(-1)
    rank = int(torch.linalg.matrix_rank(design).item())
    if rank < design.shape[1]:
        return {
            "status": "rank_deficient_design",
            "n_observations": len(observations),
            "design_rank": rank,
            "required_rank": design.shape[1],
        }
    coefficients = torch.linalg.lstsq(design, response).solution[:, 0]
    predicted = design @ coefficients[:, None]
    residual = (response - predicted).square().sum()
    centered = (response - response.mean()).square().sum()
    r_squared = 1.0 - float(residual / centered) if centered > 0 else None
    return {
        "status": "fit",
        "n_observations": len(observations),
        "design_rank": rank,
        "intercept_c": float(coefficients[0]),
        "alpha_effective_rank": float(coefficients[1]),
        "beta_heads": float(coefficients[2]),
        "gamma_layers": float(coefficients[3]),
        "r_squared_log_space": r_squared,
        "formula": "log(SNR_c)=c-alpha*log(k_eff)-beta*log(H)-gamma*log(L)",
        "caveat": "Exploratory fit to unique observed brackets/equalities; bracket centers are not exact thresholds.",
    }


def _geometric_summary(values):
    logs = [math.log(value) for value in values]
    center = statistics.mean(logs)
    if len(logs) > 1:
        error = 1.96 * statistics.stdev(logs) / math.sqrt(len(logs))
    else:
        error = None
    return {
        "value": math.exp(center),
        "lower": math.exp(center - error) if error is not None else None,
        "upper": math.exp(center + error) if error is not None else None,
        "n_training_runs": len(values),
    }


def plot_scaling_observations(observations, out_dir):
    """Plot observed architecture/dependence scaling without fitting a trend."""
    if not observations:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    protocols = (
        (
            "fixed_width_heads", "n_head", "Number of heads H",
            lambda row: row.get("n_embd") == 256 and row["n_layer"] == 12,
        ),
        (
            "fixed_head_dim", "n_head", "Number of heads H (head dimension 32)",
            lambda row: row.get("n_embd") == 32 * row["n_head"] and row["n_layer"] == 12,
        ),
        (
            "fixed_width_depth", "n_layer", "Number of layers L",
            lambda row: row.get("n_embd") == 256 and row["n_head"] == 8,
        ),
    )
    paths = []
    for name, x_key, x_label, selector in protocols:
        selected = [row for row in observations if selector(row)]
        buckets = {}
        for row in selected:
            key = (row[x_key], row["rho"], row["k"])
            buckets.setdefault(key, []).append(row["critical_snr"])
        if len({key[0] for key in buckets}) < 2:
            continue
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        series = sorted({(rho, k) for _, rho, k in buckets})
        for rho, k in series:
            xs, ys, lower, upper = [], [], [], []
            for x in sorted({key[0] for key in buckets if key[1:] == (rho, k)}):
                summary = _geometric_summary(buckets[(x, rho, k)])
                xs.append(x)
                ys.append(summary["value"])
                lower.append(
                    summary["value"] - summary["lower"] if summary["lower"] is not None else 0
                )
                upper.append(
                    summary["upper"] - summary["value"] if summary["upper"] is not None else 0
                )
            ax.errorbar(xs, ys, yerr=(lower, upper), marker="o", capsize=3,
                        label=f"rho={rho:g}, k={k}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Observed critical-SNR interval center")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        ax.set_title("Unique observed crossings; 95% CI across training-run log centers")
        fig.tight_layout()
        path = destination / f"critical_snr_{name}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(str(path))

    standard = [
        row for row in observations
        if row.get("n_embd") == 256 and row["n_head"] == 8 and row["n_layer"] == 12
    ]
    buckets = {}
    for row in standard:
        buckets.setdefault((row["rho"], row["k"]), []).append(row["critical_snr"])
    if len({rho for rho, _ in buckets}) > 1:
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        for k in sorted({k for _, k in buckets}):
            xs, ys, lower, upper = [], [], [], []
            for rho in sorted(rho for rho, cell_k in buckets if cell_k == k):
                summary = _geometric_summary(buckets[(rho, k)])
                xs.append(rho)
                ys.append(summary["value"])
                lower.append(summary["value"] - summary["lower"] if summary["lower"] else 0)
                upper.append(summary["upper"] - summary["value"] if summary["upper"] else 0)
            ax.errorbar(xs, ys, yerr=(lower, upper), marker="o", capsize=3, label=f"k={k}")
        ax.set_xlabel("Feature dependence rho")
        ax.set_ylabel("Observed critical-SNR interval center")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        ax.set_title("Dependence phase boundary; unique observed crossings only")
        fig.tight_layout()
        path = destination / "critical_snr_vs_rho.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(str(path))
    return paths


def write_scaling_report(inputs, output, target=0.5, figures_dir=None):
    groups = load_groups(inputs)
    observations, exclusions = boundary_observations(groups, target)
    figures = plot_scaling_observations(observations, figures_dir) if figures_dir else []
    document = {
        "schema_version": 1,
        "target_probability": target,
        "fit": fit_log_linear(observations),
        "observations": observations,
        "excluded_conditions": exclusions,
        "figures": figures,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
    return document
