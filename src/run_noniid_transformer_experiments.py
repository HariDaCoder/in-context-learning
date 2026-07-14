import os
import sys
import argparse
import subprocess
import json
import yaml
import glob
import re
from pathlib import Path
import torch

# Resolve paths
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

# Imports from codebase
from src.eval import eval_model, get_model_from_run

# Matplotlib configuration (non-interactive backend for Kaggle/headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_CONFIG = REPO_ROOT / "src" / "conf" / "markov_transformer_dynamics.yaml"

# Define sweeps
AR1_SWEEP = [0.0, 0.3, 0.6, 0.9]
MARKOV_SWEEP = [0.0, 0.3, 0.6, 0.9]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def find_uuid_and_step(run_dir):
    """
    Finds the run UUID subdirectory and the last saved train_step.
    Returns (uuid_str, step_int) or (None, 0).
    """
    if not os.path.exists(run_dir):
        return None, 0
    subdirs = [d for d in glob.glob(os.path.join(run_dir, "*")) if os.path.isdir(d)]
    for subdir in subdirs:
        state_path = os.path.join(subdir, "state.pt")
        if os.path.exists(state_path):
            try:
                state = torch.load(state_path, map_location="cpu")
                step = int(state.get("train_step", 0))
                uuid_str = os.path.basename(subdir)
                return uuid_str, step
            except Exception:
                pass
    return None, 0


def get_checkpoint_steps(run_dir):
    """Find all saved model steps (model_*.pt) and the final state.pt in the directory (handles UUID subdirs)."""
    existing_uuid, _ = find_uuid_and_step(run_dir)
    actual_dir = os.path.join(run_dir, existing_uuid) if existing_uuid else run_dir

    steps = []
    # Search for model_*.pt
    for p in glob.glob(os.path.join(actual_dir, "model_*.pt")):
        match = re.search(r"model_(\d+)\.pt", os.path.basename(p))
        if match:
            steps.append(int(match.group(1)))
    steps = sorted(list(set(steps)))

    # Check for final state
    state_path = os.path.join(actual_dir, "state.pt")
    if os.path.exists(state_path):
        try:
            state = torch.load(state_path, map_location="cpu")
            final_step = int(state.get("train_step", -1))
            if final_step >= 0 and final_step not in steps:
                steps.append(final_step)
        except Exception as e:
            print(f"[WARN] Error reading state.pt in {actual_dir}: {e}")
    
    return sorted(steps)


def run_training_for_config(config_path, run_dir, resume):
    """Run train.py with the generated config."""
    existing_uuid, current_step = find_uuid_and_step(run_dir)
    
    if resume and existing_uuid is not None:
        try:
            # Load config to check target steps
            cfg = load_yaml(config_path)
            target_steps = int(cfg["training"]["train_steps"])
            if current_step >= target_steps:
                print(f"[INFO] Already finished training {current_step}/{target_steps} steps. Skipping.")
                return
            else:
                print(f"[INFO] Resuming training from step {current_step}/{target_steps} with UUID {existing_uuid}...")
                # Inject the resume_id into the config so train.py uses the same folder
                cfg["training"]["resume_id"] = existing_uuid
                save_yaml(config_path, cfg)
        except Exception as e:
            print(f"[WARN] Could not check resume state for {run_dir}, restarting: {e}")
    
    # Run the train.py script
    cmd = [sys.executable, "src/train.py", "--config", str(config_path)]
    print(f"[RUNNING] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def execute_ar1_sweep(out_root, train_steps, resume):
    """Run the AR(1) rho sweep."""
    print("\n" + "="*50)
    print("STARTING AR(1) SWEEP EXPERIMENTS")
    print("="*50)
    
    os.makedirs(out_root, exist_ok=True)
    base_cfg = load_yaml(BASE_CONFIG)
    
    for rho in AR1_SWEEP:
        run_name = f"ar1_rho_{str(rho).replace('.', 'p')}"
        run_dir = os.path.join(out_root, run_name)
        os.makedirs(run_dir, exist_ok=True)
        
        # Customize config
        cfg = copy_config(base_cfg)
        cfg["inherit"] = [str(BASE_CONFIG.resolve())]
        cfg["out_dir"] = run_dir
        cfg["training"]["train_steps"] = train_steps
        cfg["training"]["data"] = "ar1"
        cfg["training"]["data_kwargs"] = {
            "rho": float(rho),
            "noise_std": 1.0
        }
        cfg["wandb"]["name"] = f"Noniid-Trans-AR1-rho{rho}"
        cfg["wandb"]["notes"] = f"2-layer GPT2 on AR1 data, rho={rho}"
        
        config_path = os.path.join(run_dir, "config.yaml")
        save_yaml(config_path, cfg)
        
        run_training_for_config(config_path, run_dir, resume)


def execute_markov_sweep(out_root, train_steps, resume):
    """Run the Markov scale sweep."""
    print("\n" + "="*50)
    print("STARTING MARKOV SCALE SWEEP EXPERIMENTS")
    print("="*50)
    
    os.makedirs(out_root, exist_ok=True)
    base_cfg = load_yaml(BASE_CONFIG)
    
    for scale in MARKOV_SWEEP:
        run_name = f"markov_scale_{str(scale).replace('.', 'p')}"
        run_dir = os.path.join(out_root, run_name)
        os.makedirs(run_dir, exist_ok=True)
        
        # Customize config
        cfg = copy_config(base_cfg)
        cfg["inherit"] = [str(BASE_CONFIG.resolve())]
        cfg["out_dir"] = run_dir
        cfg["training"]["train_steps"] = train_steps
        cfg["training"]["data"] = "markov"
        cfg["training"]["data_kwargs"] = {
            "markov_scale": float(scale),
            "markov_mode": "stationary",
            "noise_std": 1.0,
            "initial_std": 0.0
        }
        cfg["wandb"]["name"] = f"Noniid-Trans-Markov-scale{scale}"
        cfg["wandb"]["notes"] = f"2-layer GPT2 on Markov data, scale={scale}"
        
        config_path = os.path.join(run_dir, "config.yaml")
        save_yaml(config_path, cfg)
        
        run_training_for_config(config_path, run_dir, resume)


def copy_config(d):
    """Helper to deep copy dict recursively."""
    if isinstance(d, dict):
        return {k: copy_config(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [copy_config(v) for v in d]
    return d


def evaluate_run_checkpoints(run_dir, num_eval_examples=512):
    """Evaluate all checkpoints in a run and return steps and test losses."""
    existing_uuid, _ = find_uuid_and_step(run_dir)
    actual_dir = os.path.join(run_dir, existing_uuid) if existing_uuid else run_dir

    steps = get_checkpoint_steps(run_dir)
    if not steps:
        print(f"[WARN] No checkpoints found in {actual_dir}")
        return [], []
    
    print(f"[EVAL] Evaluating {len(steps)} checkpoints in {actual_dir}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    _, conf = get_model_from_run(actual_dir, only_conf=True)
    batch_size = int(conf.training.batch_size)
    
    # We evaluate on ID matched test loss
    data_name = conf.training.data
    data_kwargs = conf.training.data_kwargs
    task_name = conf.training.task
    task_kwargs = conf.training.task_kwargs
    
    test_losses = []
    
    for step in steps:
        try:
            # Load model at specific step
            model, _ = get_model_from_run(actual_dir, step=step if step < steps[-1] else -1)
            model = model.to(device).eval()
            
            with torch.no_grad():
                metrics = eval_model(
                    model=model,
                    task_name=task_name,
                    data_name=data_name,
                    n_dims=conf.model.n_dims,
                    n_points=conf.training.curriculum.points.end,
                    prompting_strategy="standard",
                    num_eval_examples=num_eval_examples,
                    batch_size=batch_size,
                    data_sampler_kwargs=data_kwargs,
                    task_sampler_kwargs=task_kwargs,
                    verbose=False
                )
                # Extracted pointwise loss at the query token (last position)
                scalar_loss = metrics["mean"][-1]
                test_losses.append(scalar_loss)
            
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[ERROR] Failed evaluating step {step} in {actual_dir}: {e}")
            test_losses.append(None)
            
    # Filter out steps that failed to evaluate
    valid_steps = [s for s, l in zip(steps, test_losses) if l is not None]
    valid_losses = [l for l in test_losses if l is not None]
    
    return valid_steps, valid_losses


def plot_comparison(run_dirs_labels, title, x_label, out_plot_path):
    """Plot test error vs training steps for multiple runs on a single chart."""
    plt.figure(figsize=(10, 6.5))
    
    for run_dir, label in run_dirs_labels:
        if not os.path.exists(run_dir):
            print(f"[WARN] Skipping non-existent directory: {run_dir}")
            continue
        
        steps, losses = evaluate_run_checkpoints(run_dir)
        if steps:
            plt.plot(steps, losses, marker="o", linewidth=2.5, label=label)
            
            # Print final results
            print(f"[{label}] Final Step: {steps[-1]}, Test MSE: {losses[-1]:.4f}")
            
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Test MSE (Query Position)", fontsize=12)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    plt.savefig(out_plot_path, dpi=200)
    plt.close()
    print(f"[SUCCESS] Saved comparison plot to {out_plot_path}")


def main():
    parser = argparse.ArgumentParser(description="2-Layer Transformer sweeps on Non-IID data")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "plot", "all"], default="all",
                        help="Mode: train, eval (runs plots), or all.")
    parser.add_argument("--data_type", type=str, choices=["ar1", "markov", "both"], default="both",
                        help="Sweep type: ar1, markov, or both.")
    parser.add_argument("--train_steps", type=int, default=200000,
                        help="Number of steps to train each config.")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume training if checkpoints exist.")
    parser.add_argument("--out_dir", type=str, default=str(REPO_ROOT / "models" / "noniid_transformer"),
                        help="Root directory for outputs.")
    args = parser.parse_args()

    out_root = os.path.abspath(args.out_dir)
    os.makedirs(out_root, exist_ok=True)
    
    # 1. Training Phase
    if args.mode in ["train", "all"]:
        if args.data_type in ["ar1", "both"]:
            execute_ar1_sweep(out_root, args.train_steps, args.resume)
        if args.data_type in ["markov", "both"]:
            execute_markov_sweep(out_root, args.train_steps, args.resume)
            
    # 2. Plotting Phase
    if args.mode in ["eval", "plot", "all"]:
        print("\n" + "="*50)
        print("GENERATING VISUALIZATIONS AND COMPARISONS")
        print("="*50)
        
        # Plot AR(1) Comparison
        if args.data_type in ["ar1", "both"]:
            ar1_runs = []
            for rho in AR1_SWEEP:
                run_name = f"ar1_rho_{str(rho).replace('.', 'p')}"
                ar1_runs.append((
                    os.path.join(out_root, run_name),
                    f"AR(1) rho={rho}" + (" (IID)" if rho == 0.0 else "")
                ))
            plot_path = os.path.join(out_root, "noniid_transformer_ar1_comparison.png")
            plot_comparison(ar1_runs, "AR(1) Correlation Level Effect on 2-Layer Transformer Dynamics", "rho", plot_path)
            
        # Plot Markov Comparison
        if args.data_type in ["markov", "both"]:
            markov_runs = []
            for scale in MARKOV_SWEEP:
                run_name = f"markov_scale_{str(scale).replace('.', 'p')}"
                markov_runs.append((
                    os.path.join(out_root, run_name),
                    f"Markov scale={scale}" + (" (IID)" if scale == 0.0 else "")
                ))
            plot_path = os.path.join(out_root, "noniid_transformer_markov_comparison.png")
            plot_comparison(markov_runs, "Markov Scale Effect on 2-Layer Transformer Dynamics", "scale", plot_path)


if __name__ == "__main__":
    main()
