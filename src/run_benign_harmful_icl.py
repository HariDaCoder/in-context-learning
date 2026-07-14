#!/usr/bin/env python3
"""
Unified Benign/Harmful Overfitting ICL Experiment Runner.

Supports both IID (markov_scale=0) and Markov (markov_scale>0) data,
fixed training corpus, finite task pool, and three-way evaluation
(train / ID-test / OOD-test) with clean query labels.

Usage:
    # Full pipeline
    python src/run_benign_harmful_icl.py --mode all

    # Training only
    python src/run_benign_harmful_icl.py --mode train

    # Evaluation only (uses existing checkpoints)
    python src/run_benign_harmful_icl.py --mode eval

    # Plotting only
    python src/run_benign_harmful_icl.py --mode plot

    # Pilot run (tiny model, small grid)
    python src/run_benign_harmful_icl.py --mode all --pilot
"""

import argparse
import copy
import glob
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import torch
import yaml

# -- Resolve paths ----------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from eval import get_model_from_run, eval_model
from samplers import get_data_sampler
from tasks import get_task_sampler

BASE_CONFIG = SCRIPT_DIR / "conf" / "benign_harmful_icl_base.yaml"

# ---------------------------------------------------------------------------
# Default experiment grids
# ---------------------------------------------------------------------------
DEFAULT_SNR_GRID = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
DEFAULT_MARKOV_SCALES = [0.0, 0.3, 0.6, 0.8, 0.9]
DEFAULT_SEEDS = [0, 1, 2]

PILOT_SNR_GRID = [1.0, 10.0, 100.0]
PILOT_MARKOV_SCALES = [0.0, 0.6]
PILOT_SEEDS = [0]

MODEL_PRESETS = {
    "tiny":   {"n_embd": 32,  "n_layer": 1, "n_head": 1},
    "small":  {"n_embd": 64,  "n_layer": 2, "n_head": 2},
    "medium": {"n_embd": 128, "n_layer": 4, "n_head": 4},
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def snr_to_noise_std(snr):
    """Convert SNR to label noise std: sigma_n = 1/sqrt(SNR)."""
    return 1.0 / math.sqrt(snr)


def format_tag(val):
    """Format a float as a filename-safe string: 0.3 -> '0p3'."""
    return f"{float(val):g}".replace(".", "p").replace("-", "m")


def find_uuid_subdir(run_dir):
    """Find the UUID subdirectory created by train.py inside run_dir."""
    if not os.path.isdir(run_dir):
        return None
    for name in os.listdir(run_dir):
        subdir = os.path.join(run_dir, name)
        if os.path.isdir(subdir) and os.path.exists(os.path.join(subdir, "state.pt")):
            return subdir
    return None


def get_train_step_from_state(state_dir):
    """Read the current training step from state.pt."""
    state_path = os.path.join(state_dir, "state.pt")
    if os.path.exists(state_path):
        try:
            state = torch.load(state_path, map_location="cpu")
            return int(state.get("train_step", 0))
        except Exception:
            pass
    return 0


def get_checkpoint_steps(state_dir):
    """Find all saved model steps in the UUID directory."""
    steps = set()
    for p in glob.glob(os.path.join(state_dir, "model_*.pt")):
        match = re.search(r"model_(\d+)\.pt", os.path.basename(p))
        if match:
            steps.add(int(match.group(1)))

    # Also include the final step from state.pt
    state_path = os.path.join(state_dir, "state.pt")
    if os.path.exists(state_path):
        try:
            state = torch.load(state_path, map_location="cpu")
            final_step = int(state.get("train_step", -1))
            if final_step >= 0:
                steps.add(final_step)
        except Exception:
            pass

    return sorted(steps)


# ---------------------------------------------------------------------------
# Phase 1: Config Generation
# ---------------------------------------------------------------------------
def generate_configs(args):
    """Generate one YAML config per (markov_scale, snr, seed) combination."""
    base_cfg = load_yaml(BASE_CONFIG)
    out_root = Path(args.out_dir)
    configs = []

    for ms in args.markov_scales:
        for snr in args.snr_grid:
            for seed in args.seeds:
                cfg = copy.deepcopy(base_cfg)

                # Model preset
                preset = MODEL_PRESETS.get(args.model_size, MODEL_PRESETS["small"])
                cfg["model"]["n_embd"] = preset["n_embd"]
                cfg["model"]["n_layer"] = preset["n_layer"]
                cfg["model"]["n_head"] = preset["n_head"]

                # Data: Markov scale
                cfg["training"]["data_kwargs"]["markov_scale"] = ms
                cfg["training"]["data_kwargs"]["normalize_variance"] = True
                cfg["training"]["data_kwargs"]["seed"] = seed

                # Task: noise from SNR
                noise_std = snr_to_noise_std(snr)
                cfg["training"]["task_kwargs"]["noise_std"] = noise_std

                # Training params
                cfg["training"]["train_steps"] = args.train_steps
                cfg["training"]["keep_every_steps"] = args.keep_every_steps
                cfg["training"]["num_tasks"] = args.num_tasks
                cfg["training"]["num_training_examples"] = args.num_training_examples
                cfg["training"]["seed"] = seed

                # Naming
                run_name = f"ms{format_tag(ms)}_snr{format_tag(snr)}_seed{seed}"
                run_dir = str(out_root / run_name)
                cfg["out_dir"] = run_dir

                cfg["wandb"]["name"] = f"BH-ICL ms={ms} SNR={snr} s={seed}"
                cfg["wandb"]["notes"] = (
                    f"Benign/harmful ICL: markov_scale={ms}, SNR={snr}, "
                    f"noise_std={noise_std:.4f}, seed={seed}, "
                    f"model={args.model_size}"
                )

                config_path = os.path.join(run_dir, "config.yaml")
                save_yaml(config_path, cfg)
                configs.append({
                    "config_path": config_path,
                    "run_dir": run_dir,
                    "markov_scale": ms,
                    "snr": snr,
                    "seed": seed,
                    "run_name": run_name,
                })

    print(f"[CONFIG] Generated {len(configs)} experiment configs in {out_root}")
    return configs


# ---------------------------------------------------------------------------
# Phase 2: Training
# ---------------------------------------------------------------------------
def run_training(configs, args):
    """Run train.py for each config, optionally parallelising across GPUs."""
    num_gpus = max(1, torch.cuda.device_count())
    print(f"[TRAIN] Starting training for {len(configs)} runs on {num_gpus} GPU(s)")

    # Filter out already-completed runs
    to_run = []
    for c in configs:
        uuid_dir = find_uuid_subdir(c["run_dir"])
        if uuid_dir is not None:
            current_step = get_train_step_from_state(uuid_dir)
            cfg = load_yaml(c["config_path"])
            target_steps = int(cfg["training"]["train_steps"])
            if current_step >= target_steps:
                print(f"  [SKIP] {c['run_name']} already at step {current_step}/{target_steps}")
                continue
            else:
                print(f"  [RESUME] {c['run_name']} from step {current_step}/{target_steps}")
                # Inject resume_id
                cfg["training"]["resume_id"] = os.path.basename(uuid_dir)
                save_yaml(c["config_path"], cfg)
        to_run.append(c)

    if not to_run:
        print("[TRAIN] All runs already complete!")
        return

    # Launch in batches of num_gpus
    for batch_start in range(0, len(to_run), num_gpus):
        batch = to_run[batch_start: batch_start + num_gpus]
        procs = []
        for gpu_idx, c in enumerate(batch):
            gpu_id = gpu_idx % num_gpus
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            cmd = [sys.executable, str(SCRIPT_DIR / "train.py"),
                   "--config", c["config_path"]]
            print(f"  [GPU {gpu_id}] {c['run_name']}")
            p = subprocess.Popen(cmd, env=env, cwd=str(REPO_ROOT))
            procs.append((p, c))

        # Wait for this batch
        for p, c in procs:
            rc = p.wait()
            if rc != 0:
                print(f"  [ERROR] {c['run_name']} exited with code {rc}")
            else:
                print(f"  [DONE] {c['run_name']}")

    print("[TRAIN] Training phase complete")


# ---------------------------------------------------------------------------
# Phase 3: Three-way Evaluation
# ---------------------------------------------------------------------------
def evaluate_checkpoints(configs, args):
    """Evaluate each run's checkpoints for train/ID/OOD MSE.

    - Train MSE:    same data/task distribution as training (with fixed corpus seeds)
    - ID-test MSE:  fresh random xs, but w drawn from a finite task pool (num_tasks)
    - OOD-test MSE: fresh random xs, fresh random w (no pool constraint)

    The clean-query property is guaranteed by the NoisyContextCleanQueryRegression
    task class: noise is only on context positions, never on the query.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_eval = args.num_eval_examples
    all_results = {}

    for c in configs:
        uuid_dir = find_uuid_subdir(c["run_dir"])
        if uuid_dir is None:
            print(f"  [SKIP] No completed run in {c['run_dir']}")
            continue

        steps = get_checkpoint_steps(uuid_dir)
        if not steps:
            print(f"  [SKIP] No checkpoints in {uuid_dir}")
            continue

        # Load config to get eval parameters
        _, conf = get_model_from_run(uuid_dir, only_conf=True)
        n_dims = conf.model.n_dims
        n_points = conf.training.curriculum.points.end
        batch_size = min(int(conf.training.batch_size), 128)
        # Ensure num_eval is divisible by batch_size
        actual_num_eval = (num_eval // batch_size) * batch_size
        if actual_num_eval == 0:
            actual_num_eval = batch_size

        task_name = conf.training.task
        data_name = conf.training.data
        data_kwargs = dict(getattr(conf.training, "data_kwargs", {}) or {})
        task_kwargs = dict(getattr(conf.training, "task_kwargs", {}) or {})
        num_tasks = getattr(conf.training, "num_tasks", None)

        # Sanitize kwargs for eval
        from eval import sanitize_sampler_kwargs
        clean_data_kwargs, clean_task_kwargs = sanitize_sampler_kwargs(
            task_name, data_name, data_kwargs, task_kwargs
        )

        print(f"  [EVAL] {c['run_name']}: {len(steps)} checkpoints")

        run_result = {
            "markov_scale": c["markov_scale"],
            "snr": c["snr"],
            "seed": c["seed"],
            "steps": [],
            "train_mse": [],
            "id_test_mse": [],
            "ood_test_mse": [],
        }

        for step in steps:
            try:
                # Load model at this checkpoint
                model, _ = get_model_from_run(
                    uuid_dir, step=step if step < steps[-1] else -1
                )
                model = model.to(device).eval()

                with torch.no_grad():
                    # --- Train MSE: standard eval (same distribution as training) ---
                    train_metrics = eval_model(
                        model=model,
                        task_name=task_name,
                        data_name=data_name,
                        n_dims=n_dims,
                        n_points=n_points,
                        prompting_strategy="standard",
                        num_eval_examples=actual_num_eval,
                        batch_size=batch_size,
                        data_sampler_kwargs=clean_data_kwargs,
                        task_sampler_kwargs=clean_task_kwargs,
                    )
                    train_loss = float(train_metrics["mean"][-1])

                    # --- ID-test MSE: fresh xs, w from finite task pool ---
                    # Create task_sampler_kwargs with num_tasks for pool-based sampling
                    id_task_kwargs = dict(clean_task_kwargs)
                    # We build a pool_dict and pass it through the task_sampler
                    # by using get_task_sampler with num_tasks
                    id_data_sampler = get_data_sampler(data_name, n_dims, **clean_data_kwargs)
                    id_task_sampler = get_task_sampler(
                        task_name, n_dims, batch_size,
                        num_tasks=num_tasks if num_tasks else 32,
                        **id_task_kwargs,
                    )
                    id_losses = []
                    for _ in range(actual_num_eval // batch_size):
                        xs = id_data_sampler.sample_xs(n_points, batch_size, device=device)
                        task = id_task_sampler()
                        ys = task.evaluate(xs)
                        pred = model(xs, ys).detach()
                        metric_fn = task.get_metric()
                        pw_loss = metric_fn(pred, ys)  # (B, n_points)
                        id_losses.append(pw_loss[:, -1].mean().item())
                    id_loss = sum(id_losses) / len(id_losses)

                    # --- OOD-test MSE: fresh xs, fresh w (no pool) ---
                    ood_task_sampler = get_task_sampler(
                        task_name, n_dims, batch_size,
                        # No num_tasks → fresh w each time
                        **clean_task_kwargs,
                    )
                    ood_losses = []
                    for _ in range(actual_num_eval // batch_size):
                        xs = id_data_sampler.sample_xs(n_points, batch_size, device=device)
                        task = ood_task_sampler()
                        ys = task.evaluate(xs)
                        pred = model(xs, ys).detach()
                        metric_fn = task.get_metric()
                        pw_loss = metric_fn(pred, ys)  # (B, n_points)
                        ood_losses.append(pw_loss[:, -1].mean().item())
                    ood_loss = sum(ood_losses) / len(ood_losses)

                run_result["steps"].append(step)
                run_result["train_mse"].append(train_loss)
                run_result["id_test_mse"].append(id_loss)
                run_result["ood_test_mse"].append(ood_loss)

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"    [ERROR] Step {step}: {e}")
                continue

        all_results[c["run_name"]] = run_result

        # Save per-run results immediately
        results_path = os.path.join(c["run_dir"], "eval_results.json")
        with open(results_path, "w") as f:
            json.dump(run_result, f, indent=2)

    # Save consolidated results
    consolidated_path = os.path.join(args.out_dir, "all_eval_results.json")
    with open(consolidated_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[EVAL] Saved consolidated results to {consolidated_path}")

    return all_results


# ---------------------------------------------------------------------------
# Phase 4: Plotting (delegates to plot_benign_harmful_icl.py)
# ---------------------------------------------------------------------------
def run_plotting(args):
    """Invoke the dedicated plotting module."""
    plot_script = str(SCRIPT_DIR / "plot_benign_harmful_icl.py")
    cmd = [
        sys.executable, plot_script,
        "--results_dir", args.out_dir,
        "--results_file", os.path.join(args.out_dir, "all_eval_results.json"),
    ]
    print(f"[PLOT] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("[PLOT] Plotting complete")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Benign/Harmful Overfitting ICL Experiments"
    )
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "train", "eval", "plot", "config"],
                        help="Which phase(s) to run")
    parser.add_argument("--pilot", action="store_true",
                        help="Use pilot grid (tiny model, small sweep)")
    parser.add_argument("--model_size", type=str, default="small",
                        choices=["tiny", "small", "medium"])
    parser.add_argument("--out_dir", type=str,
                        default=str(REPO_ROOT / "models" / "benign_harmful_icl"))

    # Grid parameters (overridden by --pilot)
    parser.add_argument("--snr_grid", type=float, nargs="+", default=None)
    parser.add_argument("--markov_scales", type=float, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)

    # Training parameters
    parser.add_argument("--train_steps", type=int, default=200000)
    parser.add_argument("--keep_every_steps", type=int, default=500)
    parser.add_argument("--num_tasks", type=int, default=32)
    parser.add_argument("--num_training_examples", type=int, default=1024)
    parser.add_argument("--num_eval_examples", type=int, default=512)

    args = parser.parse_args()

    # Apply pilot defaults
    if args.pilot:
        args.model_size = "tiny"
        args.train_steps = min(args.train_steps, 50000)
        args.snr_grid = args.snr_grid or PILOT_SNR_GRID
        args.markov_scales = args.markov_scales or PILOT_MARKOV_SCALES
        args.seeds = args.seeds or PILOT_SEEDS
    else:
        args.snr_grid = args.snr_grid or DEFAULT_SNR_GRID
        args.markov_scales = args.markov_scales or DEFAULT_MARKOV_SCALES
        args.seeds = args.seeds or DEFAULT_SEEDS

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Save experiment metadata
    meta = {
        "mode": args.mode,
        "model_size": args.model_size,
        "snr_grid": args.snr_grid,
        "markov_scales": args.markov_scales,
        "seeds": args.seeds,
        "train_steps": args.train_steps,
        "num_tasks": args.num_tasks,
        "num_training_examples": args.num_training_examples,
        "pilot": args.pilot,
    }
    with open(os.path.join(args.out_dir, "experiment_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    configs = generate_configs(args)

    if args.mode in ("all", "train"):
        run_training(configs, args)

    if args.mode in ("all", "eval"):
        evaluate_checkpoints(configs, args)

    if args.mode in ("all", "plot"):
        run_plotting(args)

    print("\n" + "=" * 60)
    print("  EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"  Results directory: {args.out_dir}")
    print(f"  Mode: {args.mode}")
    print(f"  Runs: {len(configs)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
