"""
Script để chạy nhiều experiments với các cấu hình khác nhau.
Mỗi experiment sẽ sửa đổi các tham số: data, task, task_kwargs, và wandb name/notes.
"""

import os
import copy
import yaml
import subprocess
from pathlib import Path

# Link path to file template and folder containing configs
TEMPLATE_FILE = "conf/template.yaml"
CONFIGS_DIR = "conf/experiments"
TRAIN_SCRIPT = "train.py"

# Create directory for config files
os.makedirs(CONFIGS_DIR, exist_ok=True)


def create_config(base_config, modifications, config_name):
    """
    Create a new config file based on the template and modifications.
    
    Args:
        base_config: Dict containing the original config from the template
        modifications: Dict containing the changes to apply
        config_name: Name of the new config file
    
    Returns:
        Path to the created config file
    """
    # Copy original config deeply to avoid mutating shared nested dicts
    new_config = copy.deepcopy(base_config)
    
    # Apply modifications
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
    
    # Save new config
    config_path = os.path.join(CONFIGS_DIR, f"{config_name}.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)
    
    return config_path


def run_experiment(config_path, experiment_name):
    """
    Chạy experiment với config đã cho.
    Run experiment using the specified config file.
    
    Args:
        config_path: Đường dẫn đến file config
        experiment_name: Tên experiment (để logging)
    """
    print(f"\n{'='*60}")
    print(f"Starting experiment: {experiment_name}")
    print(f"Config: {config_path}")
    print(f"{'='*60}\n")
    
    # Run train.py with config
    cmd = f"python {TRAIN_SCRIPT} --config {config_path}"
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"\n✓ Experiment '{experiment_name}' completed successfully!\n")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Experiment '{experiment_name}' failed with error: {e}\n")


def main():
    # Read template config
    with open(TEMPLATE_FILE, 'r') as f:
        base_config = yaml.safe_load(f)

    experiments = []

    # NoisyLinearRegression: vary weight distribution parameters
    exp_weights = [
        {
            'dist': 'exponential',
            'param': 'rate',
            'values': [0.5, 1.0, 2.0, 5.0],
        },
        {
            'dist': 'laplace',
            'param': 'scale',
            'values': [0.5, 1.0, 2.0],
        },
    ]

    for group in exp_weights:
        for val in group['values']:
            name = f"noisy_linear_w_{group['dist']}_{group['param']}{val}"
            experiments.append({
                'name': name,
                'modifications': {
                    'data': 'gaussian',
                    'task': 'noisy_linear_regression',
                    'task_kwargs': {
                        'w_distribution': group['dist'],
                        'w_kwargs': {group['param']: val},
                    },
                    'out_dir': f"../models/{name}",
                    'wandb': {
                        'name': f"NoisyLinear w {group['dist']} {group['param']}={val}",
                        'notes': f"Noisy linear regression with {group['dist']} weights, {group['param']}={val}",
                    },
                },
            })

    # UniformHypersphereRegression: scale sweep 1..6
    for scale in range(1, 7):
        name = f"uniform_hypersphere_scale{scale}"
        experiments.append({
            'name': name,
            'modifications': {
                'data': 'gaussian',
                'task': 'uniform_hypersphere_regression',
                'task_kwargs': {'scale': float(scale)},
                'out_dir': f"../models/{name}",
                'wandb': {
                    'name': f"Uniform Hypersphere scale={scale}",
                    'notes': f"Uniform hypersphere regression with scale={scale}",
                },
            },
        })
    
    # ============================================================
    # RUN ALL EXPERIMENTS
    # ============================================================
    print(f"\n{'#'*60}")
    print(f"Total experiments to run: {len(experiments)}")
    print(f"{'#'*60}\n")
    
    for i, exp in enumerate(experiments, 1):
        print(f"\nExperiment {i}/{len(experiments)}")
        
        # Create config file
        config_path = create_config(
            base_config,
            exp['modifications'],
            exp['name']
        )
        
        # Run experiment
        run_experiment(config_path, exp['name'])
    
    print(f"\n{'#'*60}")
    print(f"All experiments completed!")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()