"""
SNR sweep experiment for benign-to-harmful overfitting in ICL.

Outputs in results_dir:
- results.csv: one row per seed/variance/checkpoint
- results_aggregated.csv: mean/std across seeds
- transition_summary.csv: automatic transition estimates per checkpoint
- 01_train_loss_vs_snr.png, 02_test_loss_vs_snr.png, 03_gap_vs_snr.png
- interpretation_notes.md and experiment_metadata.json

Example:
    python src/run_benign_overfitting_snr_experiment.py --mode all --skip_existing
"""

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from eval import eval_model, get_model_from_run, sanitize_sampler_kwargs


DEFAULT_VARIANCES = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
DEFAULT_CHECKPOINTS = [50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def abs_from_repo(path):
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(repo_root(), path))


def tag_float(value):
    return f"{float(value):g}".replace(".", "p")


def run_id(prefix, variance, seed):
    return f"{prefix}_var{tag_float(variance)}_seed{int(seed)}"


def signal_variance(conf):
    """For x~N(0,I), w~N(0,scale^2 I): Var(x^T w)=n_dims*scale^2."""
    n_dims = float(conf["model"]["n_dims"])
    task_kwargs = conf.get("training", {}).get("task_kwargs", {}) or {}
    w_kwargs = task_kwargs.get("w_kwargs", {}) or {}
    scale = float(w_kwargs.get("scale", task_kwargs.get("scale", 1.0)))
    return n_dims * scale * scale


def make_generated_config(base_config, out_root, prefix, variance, seed):
    rid = run_id(prefix, variance, seed)
    generated_dir = os.path.join(os.path.dirname(base_config), "generated_benign_snr")
    payload = {
        # Keep POSIX-style relative path inside YAML so it remains portable.
        "inherit": [f"../{os.path.basename(base_config)}"],
        "out_dir": out_root,
        "training": {
            "resume_id": rid,
            "seed": int(seed),
            "task_kwargs": {
                "noise_type": "normal",
                "noise_std": math.sqrt(float(variance)),
                "w_distribution": "gaussian",
                "w_kwargs": {"scale": 1.0},
                "loss_type": "l2",
            },
        },
        "wandb": {"name": rid},
    }
    path = os.path.join(generated_dir, f"{rid}.yaml")
    dump_yaml(payload, path)
    return rid, path


def train_grid(args, base_conf, out_root, variances, seeds):
    for variance in variances:
        for seed in seeds:
            rid, config_path = make_generated_config(args.config, out_root, args.run_id_prefix, variance, seed)
            final_state = os.path.join(out_root, rid, "state.pt")
            if args.skip_existing and os.path.exists(final_state):
                print(f"[SKIP] {rid}")
                continue
            cmd = [sys.executable, os.path.join("src", "train.py"), "--config", config_path]
            print("[TRAIN]", " ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, cwd=repo_root(), check=True)


def load_train_loss(run_path, step):
    path = os.path.join(run_path, "train_losses.json")
    if not os.path.exists(path):
        return float("nan")
    with open(path, "r", encoding="utf-8") as f:
        losses = json.load(f)
    if str(step) in losses:
        return float(losses[str(step)])
    previous = sorted(int(k) for k in losses if int(k) <= int(step))
    return float(losses[str(previous[-1])]) if previous else float("nan")


def set_eval_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def eval_test_loss(run_path, step, variance, num_eval_examples, eval_batch_size, eval_seed):
    set_eval_seed(eval_seed)
    model, conf = get_model_from_run(run_path, step=step)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    data_kwargs, task_kwargs = sanitize_sampler_kwargs(
        conf.training.task,
        conf.training.data,
        getattr(conf.training, "data_kwargs", {}) or {},
        getattr(conf.training, "task_kwargs", {}) or {},
    )
    task_kwargs = dict(task_kwargs)
    task_kwargs.update({"noise_type": "normal", "noise_std": math.sqrt(float(variance)), "loss_type": "l2"})
    batch_size = int(eval_batch_size or conf.training.batch_size)
    num_eval_examples = max(batch_size, (num_eval_examples // batch_size) * batch_size)
    metrics = eval_model(
        model,
        task_name=conf.training.task,
        data_name=conf.training.data,
        n_dims=conf.model.n_dims,
        n_points=conf.training.curriculum.points.end,
        prompting_strategy="standard",
        num_eval_examples=num_eval_examples,
        batch_size=batch_size,
        data_sampler_kwargs=data_kwargs,
        task_sampler_kwargs=task_kwargs,
    )
    return float(metrics["mean"][-1])


def collect_rows(args, base_conf, out_root, variances, seeds, checkpoints):
    sig_var = signal_variance(base_conf)
    rows = []
    for variance in variances:
        snr = sig_var / float(variance)
        for seed in seeds:
            rid = run_id(args.run_id_prefix, variance, seed)
            path = os.path.join(out_root, rid)
            if not os.path.isdir(path):
                print(f"[WARN] missing run dir: {path}")
                continue
            for step in checkpoints:
                if not os.path.exists(os.path.join(path, f"model_{step}.pt")):
                    print(f"[WARN] missing checkpoint: {rid} step={step}")
                    continue
                train_loss = load_train_loss(path, step)
                test_loss = float("nan") if args.dry_run else eval_test_loss(
                    path, step, variance, args.num_eval_examples, args.eval_batch_size,
                    args.eval_seed + int(seed) * 100000 + int(step),
                )
                gap = test_loss - train_loss if np.isfinite(test_loss) and np.isfinite(train_loss) else float("nan")
                rows.append({
                    "step": int(step), "seed": int(seed), "variance": float(variance),
                    "snr": float(snr), "train_loss": train_loss, "test_loss": test_loss,
                    "gap": gap, "run_id": rid,
                })
    return rows


def save_results(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "seed", "variance", "snr", "train_loss", "test_loss", "gap", "run_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[DONE] {path}")
    return path


def aggregate(results_path, out_dir):
    df = pd.read_csv(results_path)
    grouped = df.groupby(["step", "variance", "snr"], as_index=False).agg(
        train_loss_mean=("train_loss", "mean"), train_loss_std=("train_loss", "std"),
        test_loss_mean=("test_loss", "mean"), test_loss_std=("test_loss", "std"),
        gap_mean=("gap", "mean"), gap_std=("gap", "std"), num_seeds=("seed", "nunique"),
    )
    path = os.path.join(out_dir, "results_aggregated.csv")
    grouped.to_csv(path, index=False)
    print(f"[DONE] {path}")
    return grouped


def plot_metric(grouped, metric, ylabel, out_dir, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    for step, sub in grouped.groupby("step"):
        sub = sub.sort_values("snr")
        x = sub["snr"].to_numpy(float)
        y = sub[f"{metric}_mean"].to_numpy(float)
        std = sub[f"{metric}_std"].fillna(0).to_numpy(float)
        ax.plot(x, y, marker="o", label=f"step={int(step)}")
        ax.fill_between(x, y - std, y + std, alpha=0.12)
    ax.set_xlabel("SNR = Var(signal) / Var(noise)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs SNR")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"[DONE] {path}")


def detect_transitions(grouped, out_dir):
    rows = []
    for step, sub in grouped.groupby("step"):
        sub = sub.sort_values("snr")
        snr = sub["snr"].to_numpy(float)
        gap = sub["gap_mean"].to_numpy(float)
        mask = np.isfinite(snr) & np.isfinite(gap)
        snr, gap = snr[mask], gap[mask]
        if len(snr) < 2:
            continue
        grad = np.gradient(gap, snr)
        idx = int(np.nanargmax(grad))
        med = np.nanmedian(gap)
        mad = np.nanmedian(np.abs(gap - med))
        threshold = float(med + 1.4826 * mad)
        above = np.where(gap > threshold)[0]
        first = int(above[0]) if len(above) else -1
        rows.append({
            "transition_step": int(step),
            "method1_transition_snr_argmax_dgap_dsnr": float(snr[idx]),
            "method1_max_gradient": float(grad[idx]),
            "method2_threshold": threshold,
            "method2_transition_snr_first_gap_gt_threshold": float(snr[first]) if first >= 0 else float("nan"),
        })
    path = os.path.join(out_dir, "transition_summary.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[DONE] {path}")


def write_notes(out_dir):
    text = """# Interpretation Notes

Assumptions:
- Data model: y = x^T w + epsilon, epsilon ~ N(0, sigma^2).
- x ~ N(0,I), w ~ N(0, scale^2 I), so SNR = n_dims * scale^2 / sigma^2.
- Only noise variance and seed are varied by the runner; architecture, optimizer, batch size, prompt format, context length, and training distribution remain fixed by the base config.
- Train loss is read from train_losses.json; test loss is final-position MSE on fresh standard prompts.

Reading plots:
- Small gap with train/test loss both decreasing suggests benign overfitting.
- Train loss decreasing while test loss/gap rises sharply suggests harmful overfitting.

If no transition is visible:
1. Increase training steps or model capacity.
2. Add lower SNR / larger variance values and use a denser grid near the suspected threshold.
3. Increase num_eval_examples and seeds.
4. Plot against log(SNR) and raw variance.
"""
    path = os.path.join(out_dir, "interpretation_notes.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[DONE] {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/conf/benign_overfitting_snr.yaml")
    parser.add_argument("--mode", choices=["train", "eval", "plot", "all"], default="all")
    parser.add_argument("--variances", default=",".join(map(str, DEFAULT_VARIANCES)))
    parser.add_argument("--checkpoints", default=",".join(map(str, DEFAULT_CHECKPOINTS)))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--num_eval_examples", type=int, default=1280)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--eval_seed", type=int, default=12345)
    parser.add_argument("--run_id_prefix", default="snr_sweep")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    args.config = abs_from_repo(args.config)
    base_conf = load_yaml(args.config)
    out_root = abs_from_repo(args.out_dir or base_conf["out_dir"])
    results_dir = args.results_dir or os.path.join(out_root, "snr_experiment_results")
    os.makedirs(results_dir, exist_ok=True)
    variances = parse_float_list(args.variances)
    seeds = parse_int_list(args.seeds)
    checkpoints = parse_int_list(args.checkpoints)

    with open(os.path.join(results_dir, "experiment_metadata.json"), "w", encoding="utf-8") as f:
        json.dump({
            "config": args.config, "out_root": out_root, "results_dir": results_dir,
            "variances": variances, "seeds": seeds, "checkpoints": checkpoints,
            "signal_variance": signal_variance(base_conf),
        }, f, indent=2)

    if args.mode in ["train", "all"]:
        train_grid(args, base_conf, out_root, variances, seeds)
    results_path = os.path.join(results_dir, "results.csv")
    if args.mode in ["eval", "all"]:
        results_path = save_results(collect_rows(args, base_conf, out_root, variances, seeds, checkpoints), results_dir)
    if args.mode in ["plot", "eval", "all"]:
        grouped = aggregate(results_path, results_dir)
        plot_metric(grouped, "train_loss", "Train loss", results_dir, "01_train_loss_vs_snr.png")
        plot_metric(grouped, "test_loss", "Test loss", results_dir, "02_test_loss_vs_snr.png")
        plot_metric(grouped, "gap", "Generalization gap = test_loss - train_loss", results_dir, "03_gap_vs_snr.png")
        detect_transitions(grouped, results_dir)
        write_notes(results_dir)


if __name__ == "__main__":
    main()