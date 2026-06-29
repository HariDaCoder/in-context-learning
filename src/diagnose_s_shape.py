"""
Diagnostic script for the S-shaped test loss anomaly.

For each per-noise-level run directory, this script:
1. Loads the final checkpoint and config.
2. Evaluates Transformer + OLS baseline on fresh data.
3. Computes: test loss, bias², variance, condition number, trace((X^TX)^{-1}),
   smallest singular value, parameter norm.
4. Saves all diagnostics to a single JSON and generates plots.

Usage:
    python src/diagnose_s_shape.py \\
        --run_root models/benign_harmful_dynamics_tiny \\
        --run_id_prefix tiny_match_noise \\
        --noise_start 0.5 --noise_end 5.0 --noise_step 0.5 \\
        --num_eval 2048 --num_variance_trials 20
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from munch import Munch

# Ensure src is on path
sys.path.insert(0, os.path.dirname(__file__))

import models as model_module
from samplers import get_data_sampler
from tasks import get_task_sampler


def _build_noise_list(start, end, step):
    values = []
    cur = float(start)
    eps = step * 1e-6
    while cur <= float(end) + eps:
        values.append(round(cur, 6))
        cur += step
    return values


def _format_tag(v):
    return f"{float(v):g}".replace(".", "p")


def load_model_and_conf(run_path):
    config_path = os.path.join(run_path, "config.yaml")
    with open(config_path) as fp:
        conf = Munch.fromDict(yaml.safe_load(fp))
    model = model_module.build_model(conf.model)
    state_path = os.path.join(run_path, "state.pt")
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    train_step = int(state.get("train_step", -1))
    return model, conf, train_step


def compute_ols_diagnostics(xs, ys, test_idx):
    """
    For each sample in the batch, compute OLS diagnostics at a specific
    context position `test_idx` (using points [0..test_idx-1] as train,
    point test_idx as test).
    
    Returns dict of per-sample diagnostics.
    """
    if test_idx < 1:
        return None

    train_xs = xs[:, :test_idx]         # (B, test_idx, d)
    train_ys = ys[:, :test_idx]         # (B, test_idx)
    test_x = xs[:, test_idx:test_idx+1] # (B, 1, d)

    B, n, d = train_xs.shape
    
    # Compute X^T X for each sample in batch
    XtX = train_xs.transpose(-2, -1) @ train_xs  # (B, d, d)

    eigenvalues = torch.linalg.eigvalsh(XtX)   # (B, d), sorted ascending
    
    cond_numbers = eigenvalues[:, -1] / eigenvalues[:, 0].clamp(min=1e-12)
    smallest_sv = eigenvalues[:, 0].sqrt()
    trace_inv = (1.0 / eigenvalues.clamp(min=1e-12)).sum(dim=-1)

    # OLS prediction
    ws, _, _, _ = torch.linalg.lstsq(train_xs, train_ys.unsqueeze(2))
    pred = (test_x @ ws)[:, 0, 0]
    param_norm = ws.squeeze(-1).norm(dim=-1)

    return {
        "condition_number": cond_numbers.tolist(),
        "smallest_singular_value": smallest_sv.tolist(),
        "trace_inv_XtX": trace_inv.tolist(),
        "ols_param_norm": param_norm.tolist(),
        "eigenvalue_min": eigenvalues[:, 0].tolist(),
        "eigenvalue_max": eigenvalues[:, -1].tolist(),
    }


def compute_bias_variance(model, data_sampler, n_dims, n_points, batch_size,
                          noise_std, num_trials=20, device="cpu"):
    """
    Bias-variance decomposition:
    - Fix w and X (single task + single input batch).
    - Sample noise K times, compute predictions.
    - bias² = E[(pred_clean - y_clean)²]
    - variance = Var[pred_noisy] across noise realizations
    """
    # Sample a fixed task (w) and fixed inputs (X)
    from tasks import NoisyLinearRegression

    # Create task with no noise to get clean signal
    clean_task = NoisyLinearRegression(
        n_dims, batch_size, noise_std=0.0,
        noise_type="normal", w_distribution="gaussian"
    )
    xs = data_sampler.sample_xs(n_points, batch_size)
    ys_clean = clean_task.evaluate(xs)

    # Predict on clean data
    model.eval()
    with torch.no_grad():
        pred_clean = model(xs.to(device), ys_clean.to(device)).cpu()
    
    # Last position prediction
    bias_sq = ((pred_clean[:, -1] - ys_clean[:, -1]) ** 2).mean().item()

    # Now sample noise K times, fix w and X
    preds_noisy = []
    for _ in range(num_trials):
        noise = torch.randn_like(ys_clean) * noise_std
        ys_noisy = ys_clean + noise
        with torch.no_grad():
            pred_noisy = model(xs.to(device), ys_noisy.to(device)).cpu()
        preds_noisy.append(pred_noisy[:, -1])

    preds_stack = torch.stack(preds_noisy, dim=0)  # (K, B)
    variance = preds_stack.var(dim=0).mean().item()

    return {
        "bias_squared": bias_sq,
        "variance": variance,
        "irreducible": noise_std ** 2,
        "decomposition_sum": bias_sq + variance + noise_std ** 2,
    }


def run_diagnostics(run_path, noise_std, num_eval, num_variance_trials, device):
    """Run full diagnostics for a single noise-level run."""
    print(f"\n{'='*60}")
    print(f"  Diagnosing: {run_path}")
    print(f"  noise_std = {noise_std}")
    print(f"{'='*60}")

    if not os.path.isdir(run_path):
        print(f"  [SKIP] Directory not found: {run_path}")
        return None

    state_path = os.path.join(run_path, "state.pt")
    if not os.path.exists(state_path):
        print(f"  [SKIP] No state.pt found in {run_path}")
        return None

    model, conf, train_step_num = load_model_and_conf(run_path)
    model = model.to(device).eval()

    n_dims = conf.model.n_dims
    n_points = conf.training.curriculum.points.end
    batch_size = min(64, num_eval)

    data_sampler = get_data_sampler("gaussian", n_dims)

    # Build noisy task sampler
    task_kwargs = {"noise_std": noise_std, "noise_type": "normal",
                   "w_distribution": "gaussian", "w_kwargs": {"scale": 1.0}}
    task_sampler = get_task_sampler(
        "noisy_linear_regression", n_dims, batch_size, **task_kwargs
    )

    # --- Evaluate test loss ---
    all_test_losses = []
    all_ols_diag = []
    num_batches = num_eval // batch_size

    for b in range(num_batches):
        xs = data_sampler.sample_xs(n_points, batch_size)
        task = task_sampler()
        ys = task.evaluate(xs)

        # Transformer prediction
        with torch.no_grad():
            pred_tf = model(xs.to(device), ys.to(device)).cpu()
        test_loss_per_sample = ((pred_tf[:, -1] - ys[:, -1]) ** 2)
        all_test_losses.append(test_loss_per_sample)

        # OLS diagnostics at last position
        ols_diag = compute_ols_diagnostics(xs.cpu(), ys.cpu(), n_points - 1)
        if ols_diag is not None:
            all_ols_diag.append(ols_diag)

    test_losses = torch.cat(all_test_losses)
    test_loss_mean = test_losses.mean().item()
    test_loss_std = test_losses.std().item()

    # Aggregate OLS diagnostics
    agg_ols = {}
    if all_ols_diag:
        for key in all_ols_diag[0]:
            all_vals = []
            for d in all_ols_diag:
                all_vals.extend(d[key])
            agg_ols[key] = {
                "mean": float(np.mean(all_vals)),
                "std": float(np.std(all_vals)),
                "min": float(np.min(all_vals)),
                "max": float(np.max(all_vals)),
                "median": float(np.median(all_vals)),
            }

    # --- Bias-variance decomposition ---
    bv = compute_bias_variance(
        model, data_sampler, n_dims, n_points, batch_size,
        noise_std, num_trials=num_variance_trials, device=device
    )

    # --- Train loss from file ---
    train_loss_file = os.path.join(run_path, "train_losses.json")
    final_train_loss = None
    if os.path.exists(train_loss_file):
        with open(train_loss_file) as f:
            tl = json.load(f)
        if tl:
            last_step = max(tl.keys(), key=int)
            final_train_loss = float(tl[last_step])

    # --- Model parameter norm ---
    total_param_norm = sum(p.norm().item() ** 2 for p in model.parameters()) ** 0.5

    result = {
        "noise_std": noise_std,
        "noise_variance": noise_std ** 2,
        "train_step": train_step_num,
        "final_train_loss": final_train_loss,
        "test_loss_mean": test_loss_mean,
        "test_loss_std": test_loss_std,
        "test_loss_se": test_loss_std / (num_eval ** 0.5),
        "model_param_norm": total_param_norm,
        "bias_variance": bv,
        "ols_diagnostics": agg_ols,
    }

    print(f"  train_loss = {final_train_loss}")
    print(f"  test_loss  = {test_loss_mean:.4f} ± {test_loss_std:.4f}")
    print(f"  bias²      = {bv['bias_squared']:.4f}")
    print(f"  variance   = {bv['variance']:.4f}")
    print(f"  irreduc.   = {bv['irreducible']:.4f}")
    print(f"  sum        = {bv['decomposition_sum']:.4f}")
    if agg_ols:
        print(f"  cond_num   = {agg_ols['condition_number']['median']:.1f}")
        print(f"  min_sv     = {agg_ols['smallest_singular_value']['median']:.4f}")

    return result


def generate_plots(results, out_dir):
    """Generate diagnostic plots from collected results."""
    os.makedirs(out_dir, exist_ok=True)

    sigmas = [r["noise_std"] for r in results]
    sigma2 = [r["noise_variance"] for r in results]

    # --- Plot 1: Test loss + Train loss vs sigma ---
    fig, ax = plt.subplots(figsize=(10, 6))
    test_means = [r["test_loss_mean"] for r in results]
    test_ses = [r["test_loss_se"] for r in results]
    train_losses = [r["final_train_loss"] or 0 for r in results]

    ax.errorbar(sigmas, test_means, yerr=test_ses, marker="o", linewidth=2,
                capsize=4, label="Test Loss (TF)", color="royalblue")
    ax.plot(sigmas, train_losses, marker="s", linewidth=2, linestyle="--",
            label="Train Loss", color="coral")
    ax.set_xlabel("Noise σ")
    ax.set_ylabel("Loss")
    ax.set_title("Train/Test Loss vs Noise Level")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_train_test_loss.png"), dpi=200)
    plt.close(fig)

    # --- Plot 2: Bias-Variance Decomposition ---
    fig, ax = plt.subplots(figsize=(10, 6))
    bias2 = [r["bias_variance"]["bias_squared"] for r in results]
    var = [r["bias_variance"]["variance"] for r in results]
    irr = [r["bias_variance"]["irreducible"] for r in results]
    total = [b + v + i for b, v, i in zip(bias2, var, irr)]

    ax.stackplot(sigmas, irr, var, bias2,
                 labels=["Irreducible (σ²)", "Variance", "Bias²"],
                 colors=["#BBDEFB", "#FFF176", "#EF9A9A"], alpha=0.8)
    ax.plot(sigmas, test_means, "ko-", linewidth=2, label="Actual Test Loss", zorder=5)
    ax.set_xlabel("Noise σ")
    ax.set_ylabel("Loss")
    ax.set_title("Bias-Variance Decomposition")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_bias_variance.png"), dpi=200)
    plt.close(fig)

    # --- Plot 3: OLS Condition Number / Trace ---
    if results[0]["ols_diagnostics"]:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        cond = [r["ols_diagnostics"]["condition_number"]["median"] for r in results]
        trace = [r["ols_diagnostics"]["trace_inv_XtX"]["median"] for r in results]
        min_sv = [r["ols_diagnostics"]["smallest_singular_value"]["median"] for r in results]

        axes[0].plot(sigmas, cond, "o-", color="teal")
        axes[0].set_title("Condition Number (median)")
        axes[0].set_xlabel("Noise σ")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(sigmas, trace, "s-", color="purple")
        axes[1].set_title("tr((X^T X)^{-1}) (median)")
        axes[1].set_xlabel("Noise σ")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(sigmas, min_sv, "^-", color="crimson")
        axes[2].set_title("Smallest Singular Value (median)")
        axes[2].set_xlabel("Noise σ")
        axes[2].grid(True, alpha=0.3)

        fig.suptitle("OLS Numerical Diagnostics")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "03_ols_diagnostics.png"), dpi=200)
        plt.close(fig)

    # --- Plot 4: Model Parameter Norm ---
    fig, ax = plt.subplots(figsize=(10, 6))
    pnorms = [r["model_param_norm"] for r in results]
    ax.plot(sigmas, pnorms, "D-", color="darkorange", linewidth=2)
    ax.set_xlabel("Noise σ")
    ax.set_ylabel("||θ||₂")
    ax.set_title("Transformer Parameter Norm vs Noise")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_param_norm.png"), dpi=200)
    plt.close(fig)

    # --- Plot 5: Train-Test Gap ---
    fig, ax = plt.subplots(figsize=(10, 6))
    gap = [t - (tr or 0) for t, tr in zip(test_means, train_losses)]
    ax.plot(sigmas, gap, "o-", color="darkgreen", linewidth=2)
    ax.set_xlabel("Noise σ")
    ax.set_ylabel("Test Loss − Train Loss")
    ax.set_title("Generalization Gap vs Noise")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_generalization_gap.png"), dpi=200)
    plt.close(fig)

    print(f"\n[DONE] All plots saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="S-shape anomaly diagnostic tool."
    )
    parser.add_argument("--run_root", type=str, required=True,
                        help="Root directory containing per-noise run subdirectories.")
    parser.add_argument("--run_id_prefix", type=str, default="tiny_match_noise",
                        help="Prefix for per-noise run directory names.")
    parser.add_argument("--noise_start", type=float, default=0.5)
    parser.add_argument("--noise_end", type=float, default=5.0)
    parser.add_argument("--noise_step", type=float, default=0.5)
    parser.add_argument("--num_eval", type=int, default=2048,
                        help="Number of eval examples per noise level.")
    parser.add_argument("--num_variance_trials", type=int, default=20,
                        help="Number of noise realizations for variance estimation.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for plots and JSON.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    noise_values = _build_noise_list(args.noise_start, args.noise_end, args.noise_step)
    print(f"[INFO] Noise levels: {noise_values}")

    all_results = []
    for noise_std in noise_values:
        tag = _format_tag(noise_std)
        run_id = f"{args.run_id_prefix}_std{tag}"
        run_path = os.path.join(args.run_root, run_id)

        result = run_diagnostics(
            run_path, noise_std, args.num_eval,
            args.num_variance_trials, device
        )
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("[ERROR] No valid runs found. Exiting.")
        sys.exit(1)

    out_dir = args.out_dir or os.path.join(args.run_root, "s_shape_diagnostics")
    os.makedirs(out_dir, exist_ok=True)

    # Save JSON
    json_path = os.path.join(out_dir, "diagnostics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"[DONE] Saved diagnostics JSON: {json_path}")

    # Generate plots
    generate_plots(all_results, out_dir)


if __name__ == "__main__":
    main()
