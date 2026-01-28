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
from typing import Optional

# Resolve project root regardless of where the script is invoked
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE = PROJECT_ROOT / "conf" / "template.yaml"
CONFIGS_DIR = PROJECT_ROOT / "conf" / "experiments"
MODELS_DIR = PROJECT_ROOT / "models"
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "train.py"

CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


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
    
    config_path = CONFIGS_DIR / f"{config_name}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)
    
    return config_path


def run_experiment(config_path, experiment_name):
    """Execute a single experiment"""
    print(f"\n{'='*70}")
    print(f"Running: {experiment_name}")
    print(f"Config: {config_path}")
    print(f"{'='*70}\n")
    
    cmd = ["python", str(TRAIN_SCRIPT), "--config", str(config_path)]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ SUCCESS: {experiment_name}\n")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ FAILED: {experiment_name} (Error: {e})\n")


def resolve_template_path(template_arg: Optional[str]) -> Path:
    """Resolve template path with fallbacks and helpful error."""
    candidates = []
    if template_arg:
        candidates.append(Path(template_arg).expanduser())
    candidates.extend([
        DEFAULT_TEMPLATE,
        PROJECT_ROOT / "src" / "conf" / "template.yaml",
        PROJECT_ROOT / "conf" / "base.yaml",
    ])

    for p in candidates:
        if p.exists():
            return p.resolve()

    msg = "Cannot find template.yaml. Checked: " + ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(msg)


def normalize_inherit_paths(config: dict, template_path: Path) -> dict:
    """Make 'inherit' entries absolute relative to the template location."""
    if 'inherit' in config and isinstance(config['inherit'], list):
        base_dir = template_path.parent
        new_inherit = []
        for item in config['inherit']:
            if isinstance(item, str):
                p = Path(item)
                if not p.is_absolute():
                    p = (base_dir / p).resolve()
                new_inherit.append(str(p))
            else:
                new_inherit.append(item)
        config['inherit'] = new_inherit
    return config


def main():
    parser = argparse.ArgumentParser(description='Run experiments for specific figures')
    parser.add_argument('--figure', type=str, default='all', 
                       choices=['1', '2', '3', 'all'],
                       help='Which figure experiments to run: 1, 2, 3, or all')
    parser.add_argument('--template', type=str, default=None,
                        help='Path to template.yaml (optional). If omitted, will try conf/template.yaml and fallbacks.')
    args = parser.parse_args()
    
    template_path = resolve_template_path(args.template)
    with open(template_path, 'r') as f:
        base_config = yaml.safe_load(f)
    base_config = normalize_inherit_paths(base_config, template_path)

    experiments = []

    # ============================================================================
    # FIGURE 1: Weight Distribution Parameters + Uniform Hypersphere Scale
    # ============================================================================
    if args.figure in ['1', 'all']:
        print("📈 Building FIGURE 1 experiments...")
        
        # Exponential weight distribution with varying rates
        for rate in [0.5, 1.5, 2.0, 3.0]:
            name = f"fig1_exp_w_rate{rate}"
            experiments.append({
                'name': name,
                'modifications': {
                    'data': 'gaussian',
                    'task': 'noisy_linear_regression',
                    'task_kwargs': {
                        'w_distribution': 'exponential',
                        'w_kwargs': {'rate': rate},
                        'noise_type': 'normal',
                        'noise_kwargs': {},
                        'noise_std': 0.0,
                        'loss_type': 'l2',
                    },
                    'out_dir': str(MODELS_DIR / name),
                    'wandb': {
                        'name': f"Fig1: Exponential w rate={rate}",
                        'notes': f"Figure 1: Exponential weight distribution, rate={rate}",
                    },
                },
            })
        
        # Laplace weight distribution with varying scales
        for scale in [0.5, 1.5, 2.0, 3.0]:
            name = f"fig1_laplace_w_scale{scale}"
            experiments.append({
                'name': name,
                'modifications': {
                    'data': 'gaussian',
                    'task': 'noisy_linear_regression',
                    'task_kwargs': {
                        'w_distribution': 'laplace',
                        'w_kwargs': {'scale': scale},
                        'noise_type': 'normal',
                        'noise_kwargs': {},
                        'noise_std': 0.0,
                        'loss_type': 'l2',
                    },
                    'out_dir': str(MODELS_DIR / name),
                    'wandb': {
                        'name': f"Fig1: Laplace w scale={scale}",
                        'notes': f"Figure 1: Laplace weight distribution, scale={scale}",
                    },
                },
            })
        
        # Uniform hypersphere regression with varying scales
        for scale in range(1, 7):
            name = f"fig1_uniform_hypersphere_scale{scale}"
            experiments.append({
                'name': name,
                'modifications': {
                    'data': 'gaussian',
                    'task': 'uniform_hypersphere_regression',
                    'task_kwargs': {'scale': float(scale)},
                    'out_dir': str(MODELS_DIR / name),
                    'wandb': {
                        'name': f"Fig1: Uniform Hypersphere scale={scale}",
                        'notes': f"Figure 1: Uniform hypersphere regression, scale={scale}",
                    },
                },
            })

    # ============================================================================
    # FIGURE 2: Gamma Data Sampler + VAR(1) Correlation
    # ============================================================================
    if args.figure in ['2', 'all']:
        print("📊 Building FIGURE 2 experiments...")

        # Gamma data experiments (no noise)
        gamma_settings = [
            ("k2_r2", {'concentration': 2.0, 'rate': 2.0}),
            ("k3_r05", {'concentration': 3.0, 'rate': 0.5}),
            ("k4_r2", {'concentration': 4.0, 'rate': 2.0}),
            ("k5_r1", {'concentration': 3.0, 'rate': 1.0}),
        ]

        for tag, g_kwargs in gamma_settings:
            name = f"fig2_gamma_{tag}"
            experiments.append({
                'name': name,
                'modifications': {
                    'data': 'gamma',
                    'data_kwargs': g_kwargs,
                    'task': 'noisy_linear_regression',
                    'task_kwargs': {
                        'w_distribution': 'gaussian',
                        'w_kwargs': {'scale': 1.0},
                        'noise_type': 'normal',
                        'noise_kwargs': {},
                        'noise_std': 0.0,
                        'loss_type': 'l2',
                    },
                    'out_dir': str(MODELS_DIR / name),
                    'wandb': {
                        'name': f"Fig2: Gamma {tag}",
                        'notes': f"Figure 2: Gamma data {g_kwargs}, no noise",
                    },
                },
            })

        # VAR(1) correlation experiments (no noise)
        var1_rhos = [0.2, 0.5, 0.8]

        for rho in var1_rhos:
            name = f"fig2_var1_rho{int(rho * 100):02d}"
            experiments.append({
                'name': name,
                'modifications': {
                    'data': 'vr1',
                    'data_kwargs': {'ar1_mat': [[rho]]},
                    'task': 'noisy_linear_regression',
                    'task_kwargs': {
                        'w_distribution': 'gaussian',
                        'w_kwargs': {'scale': 1.0},
                        'noise_type': 'normal',
                        'noise_kwargs': {},
                        'noise_std': 0.0,
                        'loss_type': 'l2',
                    },
                    'out_dir': str(MODELS_DIR / name),
                    'wandb': {
                        'name': f"Fig2: VAR(1) ρ={rho}",
                        'notes': f"Figure 2: VAR(1) with ρ={rho}, no noise",
                    },
                },
            })

    # ============================================================================
    # FIGURE 3: Comprehensive Noise Type Study
    # ============================================================================
    if args.figure in ['3', 'all']:
        print("📉 Building FIGURE 3 experiments...")
        
        noise_configs = [
            ('bernoulli', [0.3], 'p'),
            ('gamma', [(4.0, 1.0)], 'k'),
            ('poisson', [2.0, 3.0], 'lambda'),
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
                        'out_dir': str(MODELS_DIR / name),
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
