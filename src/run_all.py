"""
Comprehensive experiment runner for in-context learning research.
Organized by figures - systematically designed from scratch.
"""

import os
import copy
import yaml
import subprocess
import argparse
from pathlib import Path

TEMPLATE_FILE = "conf/template.yaml"
CONFIGS_DIR = "conf/experiments"
TRAIN_SCRIPT = "train.py"

os.makedirs(CONFIGS_DIR, exist_ok=True)


def create_config(base_config, modifications, config_name):
    """Create config file with modifications"""
    new_config = copy.deepcopy(base_config)
    
    if 'data' in modifications:
        new_config['training']['data'] = modifications['data']
    if 'data_kwargs' in modifications:
        new_config['training']['data_kwargs'] = modifications['data_kwargs']
    if 'task' in modifications:
        new_config['training']['task'] = modifications['task']
    if 'task_kwargs' in modifications:
        new_config['training']['task_kwargs'] = modifications['task_kwargs']
    if 'out_dir' in modifications:
        new_config['out_dir'] = modifications['out_dir']
    if 'wandb' in modifications:
        if 'name' in modifications['wandb']:
            new_config['wandb']['name'] = modifications['wandb']['name']
        if 'notes' in modifications['wandb']:
            new_config['wandb']['notes'] = modifications['wandb']['notes']
    
    config_path = os.path.join(CONFIGS_DIR, f"{config_name}.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)
    
    return config_path


def run_experiment(config_path, experiment_name):
    """Execute a single experiment"""
    print(f"\n{'='*70}")
    print(f"Running: {experiment_name}")
    print(f"Config: {config_path}")
    print(f"{'='*70}\n")
    
    cmd = f"python {TRAIN_SCRIPT} --config {config_path}"
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"\n✓ SUCCESS: {experiment_name}\n")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ FAILED: {experiment_name} (Error: {e})\n")


def main():
    parser = argparse.ArgumentParser(description='Run experiments for specific figures')
    parser.add_argument('--figure', type=str, default='all', 
                       choices=['1', '2', '3', 'all'],
                       help='Which figure experiments to run: 1, 2, 3, or all')
    args = parser.parse_args()
    
    with open(TEMPLATE_FILE, 'r') as f:
        base_config = yaml.safe_load(f)

    experiments = []

    # ============================================================================
    # FIGURE 1: Exponential Weights + Different Noise Types
    # ============================================================================
    if args.figure in ['1', 'all']:
        print("📈 Building FIGURE 1 experiments...")
        
        for noise_type in ['normal', 'laplace', 'exponential']:
            noise_kw = {'rate': 1.0} if noise_type == 'exponential' else {}
            name = f"fig1_exp_w_noise_{noise_type}"
            experiments.append({
                'name': name,
                'modifications': {
                    'data': 'gaussian',
                    'task': 'noisy_linear_regression',
                    'task_kwargs': {
                        'w_distribution': 'exponential',
                        'w_kwargs': {'rate': 1.0},
                        'noise_type': noise_type,
                        'noise_kwargs': noise_kw,
                        'noise_std': 1.0,
                    },
                    'out_dir': f"../models/{name}",
                    'wandb': {
                        'name': f"Fig1: Exp w + {noise_type} noise",
                        'notes': f"Figure 1: Exponential weights with {noise_type} noise",
                    },
                },
            })

    # ============================================================================
    # FIGURE 2: Gamma Data Sampler + VAR(1) Correlation
    # ============================================================================
    if args.figure in ['2', 'all']:
        print("📊 Building FIGURE 2 experiments...")
        
        # Gamma data: concentration=2.0, rate=1.0
        for noise_type in ['normal', 'laplace']:
            name = f"fig2_gamma_data_noise_{noise_type}"
            experiments.append({
                'name': name,
                'modifications': {
                    'data': 'gamma',
                    'data_kwargs': {'concentration': 2.0, 'rate': 1.0},
                    'task': 'noisy_linear_regression',
                    'task_kwargs': {
                        'w_distribution': 'gaussian',
                        'w_kwargs': {'scale': 1.0},
                        'noise_type': noise_type,
                        'noise_std': 1.0,
                    },
                    'out_dir': f"../models/{name}",
                    'wandb': {
                        'name': f"Fig2: Gamma data + {noise_type} noise",
                        'notes': f"Figure 2: Gamma data (k=2, λ=1), {noise_type} noise",
                    },
                },
            })
        
        # VAR(1) data with rho=0.4
        for noise_type in ['normal', 'laplace']:
            name = f"fig2_var1_rho04_noise_{noise_type}"
            experiments.append({
                'name': name,
                'modifications': {
                    'data': 'vr1',
                    'data_kwargs': {'ar1_mat': None},
                    'task': 'noisy_linear_regression',
                    'task_kwargs': {
                        'w_distribution': 'gaussian',
                        'w_kwargs': {'scale': 1.0},
                        'noise_type': noise_type,
                        'noise_std': 1.0,
                    },
                    'out_dir': f"../models/{name}",
                    'wandb': {
                        'name': f"Fig2: VAR(1) ρ=0.4 + {noise_type}",
                        'notes': f"Figure 2: VAR(1) with ρ=0.4, {noise_type} noise",
                    },
                },
            })

    # ============================================================================
    # FIGURE 3: Comprehensive Noise Type Study
    # ============================================================================
    if args.figure in ['3', 'all']:
        print("📉 Building FIGURE 3 experiments...")
        
        noise_configs = [
            ('bernoulli', [0.1, 0.2, 0.3, 0.4], 'p'),
            ('exponential', [0.5, 1.5, 2.0], 'rate'),
            ('gamma', [(2.0, 2.0), (3.0, 1.0), (4.0, 1.0)], 'k'),
            ('poisson', [0.5, 2.0, 3.0], 'lambda'),
            ('t-student', [3.0], 'df'),
        ]
        
        for noise_type, param_values, param_name in noise_configs:
            for param_val in param_values:
                if noise_type == 'bernoulli':
                    noise_kw = {'p': param_val}
                    val_str = f"p{param_val}"
                elif noise_type == 'exponential':
                    noise_kw = {'rate': param_val}
                    val_str = f"rate{param_val}"
                elif noise_type == 'gamma':
                    noise_kw = {'concentration': param_val[0], 'rate': param_val[1]}
                    val_str = f"k{param_val[0]}_lambda{param_val[1]}"
                elif noise_type == 'poisson':
                    noise_kw = {'lambda': param_val}
                    val_str = f"lambda{param_val}"
                elif noise_type == 't-student':
                    noise_kw = {'df': param_val}
                    val_str = f"df{param_val}"
                
                name = f"fig3_noise_{noise_type}_{val_str}"
                experiments.append({
                    'name': name,
                    'modifications': {
                        'data': 'gaussian',
                        'task': 'noisy_linear_regression',
                        'task_kwargs': {
                            'w_distribution': 'gaussian',
                            'w_kwargs': {'scale': 1.0},
                            'noise_type': noise_type,
                            'noise_kwargs': noise_kw,
                            'noise_std': 1.0,
                        },
                        'out_dir': f"../models/{name}",
                        'wandb': {
                            'name': f"Fig3: {noise_type} {val_str}",
                            'notes': f"Figure 3: {noise_type} noise with {param_name}={param_val}",
                        },
                    },
                })

    # ============================================================================
    # RUN ALL EXPERIMENTS
    # ============================================================================
    print(f"\n{'#'*70}")
    print(f"🚀 Running experiments for: FIGURE {args.figure.upper()}")
    print(f"📊 TOTAL EXPERIMENTS: {len(experiments)}")
    print(f"{'#'*70}\n")
    
    for i, exp in enumerate(experiments, 1):
        print(f"[{i}/{len(experiments)}] ", end="")
        
        config_path = create_config(
            base_config,
            exp['modifications'],
            exp['name']
        )
        
        run_experiment(config_path, exp['name'])
    
    print(f"\n{'#'*70}")
    print(f"✅ ALL EXPERIMENTS COMPLETED!")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
