#!/usr/bin/env python3
"""
Publication-quality plotting for Benign/Harmful Overfitting ICL experiments.

Generates:
1. Dynamics plots (loss vs training step, colored by SNR)
2. SNR slice plots (loss vs log(SNR) at fixed training steps)
3. Heatmaps (step × log-SNR, color = OOD loss or gap)
4. Gap plots (Δ_OOD = L_OOD - L_train)
5. Markov comparison overlays
6. Phase boundary diagram
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.ticker import LogLocator, LogFormatter

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

SNR_CMAP = plt.cm.viridis
TRAIN_STYLE = {"linestyle": ":", "alpha": 0.5, "linewidth": 1.0}
ID_STYLE = {"linestyle": "--", "alpha": 0.7, "linewidth": 1.5}
OOD_STYLE = {"linestyle": "-", "alpha": 1.0, "linewidth": 2.0}

MARKOV_MARKERS = {0.0: "o", 0.3: "s", 0.6: "^", 0.8: "D", 0.9: "v"}
MARKOV_COLORS = {0.0: "#1f77b4", 0.3: "#ff7f0e", 0.6: "#2ca02c",
                 0.8: "#d62728", 0.9: "#9467bd"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_results(results_file):
    """Load consolidated results JSON."""
    with open(results_file, "r") as f:
        return json.load(f)


def group_by_markov_scale(results):
    """Group run results by markov_scale."""
    grouped = defaultdict(list)
    for run_name, data in results.items():
        ms = data["markov_scale"]
        grouped[ms].append(data)
    return dict(grouped)


def group_by_snr(results):
    """Group run results by SNR."""
    grouped = defaultdict(list)
    for run_name, data in results.items():
        snr = data["snr"]
        grouped[snr].append(data)
    return dict(grouped)


def average_over_seeds(runs):
    """Average metrics over seeds for runs with same (markov_scale, snr)."""
    if not runs:
        return None
    # Find common steps
    all_steps = [set(r["steps"]) for r in runs]
    common_steps = sorted(set.intersection(*all_steps)) if all_steps else []
    if not common_steps:
        # Fallback: use the first run's steps
        common_steps = runs[0]["steps"]

    avg = {"steps": common_steps, "train_mse": [], "id_test_mse": [], "ood_test_mse": []}
    for step in common_steps:
        train_vals, id_vals, ood_vals = [], [], []
        for r in runs:
            if step in r["steps"]:
                idx = r["steps"].index(step)
                train_vals.append(r["train_mse"][idx])
                id_vals.append(r["id_test_mse"][idx])
                ood_vals.append(r["ood_test_mse"][idx])
        avg["train_mse"].append(float(np.mean(train_vals)) if train_vals else None)
        avg["id_test_mse"].append(float(np.mean(id_vals)) if id_vals else None)
        avg["ood_test_mse"].append(float(np.mean(ood_vals)) if ood_vals else None)

    return avg


def find_characteristic_steps(steps, ood_losses):
    """Automatically identify characteristic training steps."""
    steps = np.array(steps)
    losses = np.array(ood_losses)

    result = {}

    # Early: first 10% of steps
    early_idx = max(0, len(steps) // 10)
    result["early"] = int(steps[early_idx])

    # Best OOD: minimum OOD loss
    best_idx = np.argmin(losses)
    result["best"] = int(steps[best_idx])

    # Late: last step
    result["late"] = int(steps[-1])

    # Knee: steepest descent region (first derivative minimum)
    if len(losses) > 5:
        diffs = np.diff(losses)
        knee_idx = np.argmin(diffs) + 1
        result["knee"] = int(steps[knee_idx])

    # Overfit: if loss increases after best, find the point where it's 10% above best
    if best_idx < len(losses) - 1:
        post_best = losses[best_idx:]
        threshold = losses[best_idx] * 1.1
        overfit_mask = post_best > threshold
        if overfit_mask.any():
            overfit_idx = best_idx + np.argmax(overfit_mask)
            result["overfit"] = int(steps[overfit_idx])

    return result


# ---------------------------------------------------------------------------
# Plot 1: Dynamics (loss vs training step, colored by SNR)
# ---------------------------------------------------------------------------
def plot_dynamics(results, out_dir, markov_scale=None):
    """Plot loss curves vs training step, one curve per SNR."""
    by_ms = group_by_markov_scale(results)

    for ms, runs in by_ms.items():
        if markov_scale is not None and ms != markov_scale:
            continue

        # Group by SNR and average over seeds
        snr_groups = defaultdict(list)
        for r in runs:
            snr_groups[r["snr"]].append(r)

        fig, ax = plt.subplots(figsize=(12, 7))
        snr_values = sorted(snr_groups.keys())
        norm = mcolors.LogNorm(vmin=min(snr_values), vmax=max(snr_values))

        for snr in snr_values:
            avg = average_over_seeds(snr_groups[snr])
            if avg is None:
                continue

            color = SNR_CMAP(norm(snr))
            steps = avg["steps"]

            ax.plot(steps, avg["ood_test_mse"], color=color,
                    label=f"SNR={snr:g} (OOD)", **OOD_STYLE)
            ax.plot(steps, avg["id_test_mse"], color=color,
                    label=f"SNR={snr:g} (ID)", **ID_STYLE)
            ax.plot(steps, avg["train_mse"], color=color,
                    label=f"SNR={snr:g} (Train)", **TRAIN_STYLE)

        ms_tag = f"{ms:g}".replace(".", "p")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Clean-Query MSE")
        ax.set_title(f"Training Dynamics — Markov Scale = {ms:g}"
                     + (" (IID)" if ms == 0 else ""))
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        # Custom legend: show SNR colors and line style legend separately
        handles, labels = ax.get_legend_handles_labels()
        # Deduplicate
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),
                  loc="upper right", fontsize=8, ncol=2)

        path = os.path.join(out_dir, f"dynamics_ms{ms_tag}.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  [PLOT] Saved {path}")


# ---------------------------------------------------------------------------
# Plot 2: SNR Slice (loss vs log-SNR at fixed steps)
# ---------------------------------------------------------------------------
def plot_snr_slice(results, out_dir, markov_scale=None):
    """For each markov_scale, plot OOD loss vs log(SNR) at characteristic steps."""
    by_ms = group_by_markov_scale(results)

    for ms, runs in by_ms.items():
        if markov_scale is not None and ms != markov_scale:
            continue

        snr_groups = defaultdict(list)
        for r in runs:
            snr_groups[r["snr"]].append(r)
        snr_values = sorted(snr_groups.keys())

        # We need to pick characteristic steps from one representative run
        # Use the lowest SNR run as reference (it typically shows the most structure)
        ref_avg = average_over_seeds(snr_groups[snr_values[0]])
        if ref_avg is None:
            continue
        char_steps = find_characteristic_steps(ref_avg["steps"], ref_avg["ood_test_mse"])

        fig, ax = plt.subplots(figsize=(10, 6))
        step_colors = plt.cm.Set1(np.linspace(0, 1, len(char_steps)))

        for (step_name, step_val), color in zip(char_steps.items(), step_colors):
            ood_at_step = []
            for snr in snr_values:
                avg = average_over_seeds(snr_groups[snr])
                if avg is None or step_val not in avg["steps"]:
                    ood_at_step.append(None)
                    continue
                idx = avg["steps"].index(step_val)
                ood_at_step.append(avg["ood_test_mse"][idx])

            valid = [(s, v) for s, v in zip(snr_values, ood_at_step) if v is not None]
            if valid:
                xs, ys = zip(*valid)
                ax.plot(xs, ys, "o-", color=color, linewidth=2,
                        label=f"step={step_val} ({step_name})", markersize=5)

        ms_tag = f"{ms:g}".replace(".", "p")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("SNR")
        ax.set_ylabel("OOD Clean-Query MSE")
        ax.set_title(f"SNR Slice — Markov Scale = {ms:g}")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=9)

        path = os.path.join(out_dir, f"snr_slice_ms{ms_tag}.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  [PLOT] Saved {path}")


# ---------------------------------------------------------------------------
# Plot 3: Heatmap (step × log-SNR, color = OOD loss or gap)
# ---------------------------------------------------------------------------
def plot_heatmap(results, out_dir, markov_scale=None, metric="ood"):
    """Heatmap of OOD loss or generalization gap over step × SNR grid."""
    by_ms = group_by_markov_scale(results)

    for ms, runs in by_ms.items():
        if markov_scale is not None and ms != markov_scale:
            continue

        snr_groups = defaultdict(list)
        for r in runs:
            snr_groups[r["snr"]].append(r)
        snr_values = sorted(snr_groups.keys())

        # Build the grid
        # Use common steps across all SNR values
        all_steps = None
        avgs = {}
        for snr in snr_values:
            avg = average_over_seeds(snr_groups[snr])
            if avg is None:
                continue
            avgs[snr] = avg
            step_set = set(avg["steps"])
            all_steps = step_set if all_steps is None else all_steps & step_set

        if not all_steps or not avgs:
            continue

        common_steps = sorted(all_steps)
        # Subsample to manageable grid
        if len(common_steps) > 200:
            indices = np.linspace(0, len(common_steps) - 1, 200, dtype=int)
            common_steps = [common_steps[i] for i in indices]

        # Build 2D array
        Z = np.full((len(snr_values), len(common_steps)), np.nan)
        for i, snr in enumerate(snr_values):
            avg = avgs.get(snr)
            if avg is None:
                continue
            for j, step in enumerate(common_steps):
                if step in avg["steps"]:
                    idx = avg["steps"].index(step)
                    if metric == "gap":
                        ood = avg["ood_test_mse"][idx]
                        train = avg["train_mse"][idx]
                        Z[i, j] = (ood - train) if (ood is not None and train is not None) else np.nan
                    else:
                        Z[i, j] = avg["ood_test_mse"][idx] if avg["ood_test_mse"][idx] is not None else np.nan

        fig, ax = plt.subplots(figsize=(14, 6))
        X, Y = np.meshgrid(common_steps, np.log10(snr_values))

        if metric == "gap":
            cmap = "RdBu_r"
            vmax = np.nanpercentile(np.abs(Z), 95) if not np.all(np.isnan(Z)) else 1
            im = ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto")
            ax.set_title(f"Generalisation Gap (OOD − Train) — Markov Scale = {ms:g}")
        else:
            cmap = "viridis"
            Z_log = np.log10(np.maximum(Z, 1e-8))
            im = ax.pcolormesh(X, Y, Z_log, cmap=cmap, shading="auto")
            ax.set_title(f"OOD Clean-Query MSE (log₁₀) — Markov Scale = {ms:g}")

            # Add contour lines
            try:
                cs = ax.contour(X, Y, Z_log, levels=5, colors="white",
                                linewidths=0.8, alpha=0.7)
                ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
            except Exception:
                pass

        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("log₁₀(SNR)")

        ms_tag = f"{ms:g}".replace(".", "p")
        metric_tag = "gap" if metric == "gap" else "ood"
        path = os.path.join(out_dir, f"heatmap_{metric_tag}_ms{ms_tag}.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  [PLOT] Saved {path}")


# ---------------------------------------------------------------------------
# Plot 4: Gap plot
# ---------------------------------------------------------------------------
def plot_gap(results, out_dir, markov_scale=None):
    """Plot Δ_OOD = L_OOD - L_train vs training step, colored by SNR."""
    plot_heatmap(results, out_dir, markov_scale=markov_scale, metric="gap")


# ---------------------------------------------------------------------------
# Plot 5: Markov comparison overlay
# ---------------------------------------------------------------------------
def plot_markov_comparison(results, out_dir):
    """For each SNR, overlay IID and all Markov scales."""
    by_snr = group_by_snr(results)

    for snr, runs in by_snr.items():
        ms_groups = defaultdict(list)
        for r in runs:
            ms_groups[r["markov_scale"]].append(r)

        fig, ax = plt.subplots(figsize=(12, 7))

        for ms in sorted(ms_groups.keys()):
            avg = average_over_seeds(ms_groups[ms])
            if avg is None:
                continue

            color = MARKOV_COLORS.get(ms, "gray")
            label = f"s={ms:g}" + (" (IID)" if ms == 0 else "")

            ax.plot(avg["steps"], avg["ood_test_mse"], color=color,
                    linewidth=2, label=f"{label} OOD", alpha=0.9)
            ax.plot(avg["steps"], avg["id_test_mse"], color=color,
                    linewidth=1.2, linestyle="--", label=f"{label} ID", alpha=0.6)

        snr_tag = f"{snr:g}".replace(".", "p")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Clean-Query MSE")
        ax.set_title(f"Markov Comparison — SNR = {snr:g}")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

        path = os.path.join(out_dir, f"markov_comparison_snr{snr_tag}.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  [PLOT] Saved {path}")


# ---------------------------------------------------------------------------
# Plot 6: Phase boundary diagram
# ---------------------------------------------------------------------------
def plot_phase_boundary(results, out_dir):
    """Plot critical SNR for harmful onset vs markov_scale.

    Harmful onset: first step where OOD loss increases by >10% above its minimum.
    """
    by_ms = group_by_markov_scale(results)

    ms_values = []
    critical_snrs = []

    for ms in sorted(by_ms.keys()):
        runs = by_ms[ms]
        snr_groups = defaultdict(list)
        for r in runs:
            snr_groups[r["snr"]].append(r)

        best_snr_harmful = None
        for snr in sorted(snr_groups.keys()):
            avg = average_over_seeds(snr_groups[snr])
            if avg is None or len(avg["ood_test_mse"]) < 5:
                continue

            losses = np.array(avg["ood_test_mse"])
            valid = losses[~np.isnan(losses)]
            if len(valid) < 5:
                continue

            best_idx = np.argmin(valid)
            if best_idx < len(valid) - 1:
                post_best = valid[best_idx:]
                if np.any(post_best > valid[best_idx] * 1.1):
                    # This SNR shows harmful overfitting
                    best_snr_harmful = snr
                    break

        if best_snr_harmful is not None:
            ms_values.append(ms)
            critical_snrs.append(best_snr_harmful)

    if not ms_values:
        print("  [PLOT] Not enough data for phase boundary diagram")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ms_values, critical_snrs, "o-", color="#d62728",
            linewidth=2, markersize=8, label="Empirical boundary")

    # Theoretical prediction: SNR_c ∝ d(1+s) / (T(1-s))
    if len(ms_values) > 1:
        d, T = 20, 40  # defaults
        s_grid = np.linspace(0, 0.95, 100)
        snr_theory = critical_snrs[0] * (1 + s_grid) / (1 - s_grid + 1e-8) \
                     * (1 - ms_values[0]) / (1 + ms_values[0] + 1e-8)
        ax.plot(s_grid, snr_theory, "--", color="gray", alpha=0.5,
                label=r"Theory: $\propto \frac{1+s}{1-s}$")

    ax.set_xlabel("Markov Scale $s$")
    ax.set_ylabel("Critical SNR (harmful onset)")
    ax.set_title("Phase Boundary: Benign → Harmful")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = os.path.join(out_dir, "phase_boundary.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [PLOT] Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot benign/harmful ICL results")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--results_file", type=str, required=True)
    args = parser.parse_args()

    results = load_results(args.results_file)
    out_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[PLOT] Loaded {len(results)} runs from {args.results_file}")
    print(f"[PLOT] Output directory: {out_dir}")

    # Generate all plots
    print("\n--- Dynamics Plots ---")
    plot_dynamics(results, out_dir)

    print("\n--- SNR Slice Plots ---")
    plot_snr_slice(results, out_dir)

    print("\n--- Heatmaps (OOD) ---")
    plot_heatmap(results, out_dir, metric="ood")

    print("\n--- Heatmaps (Gap) ---")
    plot_gap(results, out_dir)

    print("\n--- Markov Comparison ---")
    plot_markov_comparison(results, out_dir)

    print("\n--- Phase Boundary ---")
    plot_phase_boundary(results, out_dir)

    print(f"\n[PLOT] All plots saved to {out_dir}")


if __name__ == "__main__":
    main()
