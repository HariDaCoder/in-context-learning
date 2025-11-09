import os
import uuid
import yaml
import argparse
import sys
import tempfile
from quinine import QuinineArgumentParser

from schema import schema as quinine_schema
from train import main as train_main


def prepare_out_dir(args):
    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        # Persist the resolved config for this run (mirrors train.py behaviour)
        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)


def run_one_experiment(base_config_path: str, task: str, task_kwargs: dict, data_kwargs: dict, run_name: str, resume_id: str = None, data_type: str = None):
    """
    Run a single experiment with specified task, task_kwargs, and data_kwargs.
    
    Args:
        base_config_path: Path to base config yaml file
        task: Task name (e.g., 'sparse_linear_regression', 'noisy_linear_regression')
        task_kwargs: Dictionary of task-specific kwargs (e.g., {'noise_type': 'normal', 'sparsity': 3})
        data_kwargs: Dictionary of data sampler kwargs (e.g., {'sparsity': 5})
        run_name: Name for wandb run
        resume_id: Optional resume_id for the run
        data_type: Optional data type override (e.g., 'sparse_gaussian' for sparse data experiments)
    """
    config_dir = os.path.dirname(base_config_path)

    # Read base config
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # Modify config for this experiment
    base_config['training']['task'] = task
    base_config['training']['task_kwargs'] = task_kwargs
    base_config['training']['data_kwargs'] = data_kwargs
    if data_type is not None:
        base_config['training']['data'] = data_type
    base_config['wandb']['name'] = run_name
    if resume_id is not None:
        base_config['training']['resume_id'] = resume_id

    # Create temporary config file
    temp_config_file = tempfile.NamedTemporaryFile(
        mode='w+t', 
        delete=False, 
        suffix='.yaml',
        dir=config_dir
    )
    
    try:
        # Write modified config to temp file
        yaml.dump(base_config, temp_config_file, default_flow_style=False)
        temp_config_file.close()

        # Parse config using Quinine
        cli_args_list = ["--config", temp_config_file.name]
        qparser = QuinineArgumentParser(schema=quinine_schema)
        original_argv = sys.argv
        try:
            sys.argv = ["run_one_script_placeholder"] + cli_args_list
            args = qparser.parse_quinfig()
        finally:
            sys.argv = original_argv

        # Prepare output directory and run training
        prepare_out_dir(args)
        print(f"\n{'='*60}")
        print(f"Running: {run_name}")
        print(f"Task: {task}")
        print(f"Task kwargs: {task_kwargs}")
        print(f"Data kwargs: {data_kwargs}")
        if data_type is not None:
            print(f"Data type: {data_type}")
        print(f"{'='*60}\n")
        train_main(args)

    finally:
        # Clean up temp file
        if os.path.exists(temp_config_file.name):
            os.remove(temp_config_file.name)


def get_default_experiments():
    """
    Define default experiments for sparse_linear_regression and noisy_linear_regression.
    Returns a list of experiment configs: (task, task_kwargs, data_kwargs, run_name, data_type)
    """
    experiments = []
    
    # ===== Sparse Linear Regression Experiments =====
    # Sparse w (weight sparsity)
    for sparsity in [3, 5, 7]:
        experiments.append((
            "sparse_linear_regression",
            {"sparsity": sparsity},  # task_kwargs
            {},  # data_kwargs
            f"sparse_w_sparsity_{sparsity}",
            None  # data_type: use default from config
        ))
    
    # Sparse data (data sparsity) - using sparse_gaussian data
    for data_sparsity in [5, 10, 15]:
        experiments.append((
            "sparse_linear_regression",
            {"sparsity": 3},  # task_kwargs (w sparsity)
            {"sparsity": data_sparsity},  # data_kwargs (data sparsity)
            f"sparse_data_sparsity_{data_sparsity}",
            "sparse_gaussian"  # data_type override
        ))
    
    # ===== Noisy Linear Regression Experiments =====
    # Different noise types
    noise_types = [
        "normal",
        "uniform",
        "laplace",
        "t-student",
        "cauchy",
        "exponential",
        "rayleigh",
        "beta",
        "poisson",
    ]
    
    for noise_type in noise_types:
        experiments.append((
            "noisy_linear_regression",
            {"noise_type": noise_type, "noise_std": 2.0},  # task_kwargs
            {},  # data_kwargs
            f"noisy_{noise_type}",
            None  # data_type: use default from config
        ))
    
    # Different noise_std values for normal noise
    for noise_std in [0.5, 1.0, 2.0, 3.0]:
        experiments.append((
            "noisy_linear_regression",
            {"noise_type": "normal", "noise_std": noise_std},  # task_kwargs
            {},  # data_kwargs
            f"noisy_normal_std_{noise_std}",
            None  # data_type: use default from config
        ))
    
    return experiments


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run experiments for sparse_linear_regression and noisy_linear_regression"
    )
    parser.add_argument(
        "--config",
        default="src/conf/toy.yaml",
        help="Base config yaml (e.g., src/conf/toy.yaml)",
    )
    parser.add_argument(
        "--task",
        choices=["sparse", "noisy", "both", "custom"],
        default="both",
        help="Which task(s) to run: 'sparse', 'noisy', 'both', or 'custom'",
    )
    parser.add_argument(
        "--sparse_w_sparsities",
        nargs="*",
        type=int,
        default=[3, 5, 7],
        help="Weight sparsity values for sparse_linear_regression (w sparsity)",
    )
    parser.add_argument(
        "--sparse_data_sparsities",
        nargs="*",
        type=int,
        default=[5, 10, 15],
        help="Data sparsity values for sparse_linear_regression (data sparsity)",
    )
    parser.add_argument(
        "--noise_types",
        nargs="*",
        default=[
            "normal",
            "uniform",
            "laplace",
            "t-student",
            "cauchy",
            "exponential",
            "rayleigh",
            "beta",
            "poisson",
        ],
        help="Noise types for noisy_linear_regression",
    )
    parser.add_argument(
        "--noise_stds",
        nargs="*",
        type=float,
        default=[0.5, 1.0, 2.0, 3.0],
        help="Noise standard deviations for noisy_linear_regression",
    )
    parser.add_argument(
        "--base_run_name",
        default="sweep",
        help="Base prefix for wandb.name",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip runs that already have config.yaml in output directory",
    )
    return parser


def main():
    parser = build_parser()
    cli_args = parser.parse_args()

    experiments = []
    
    # Build experiment list based on task selection
    if cli_args.task in ["sparse", "both"]:
        # Sparse w experiments (weight sparsity, regular gaussian data)
        for sparsity in cli_args.sparse_w_sparsities:
            experiments.append((
                "sparse_linear_regression",
                {"sparsity": sparsity},
                {},
                f"{cli_args.base_run_name}_sparse_w_{sparsity}",
                None  # data_type: use default from config
            ))
        
        # Sparse data experiments (sparse_gaussian data)
        for data_sparsity in cli_args.sparse_data_sparsities:
            experiments.append((
                "sparse_linear_regression",
                {"sparsity": 3},  # w sparsity
                {"sparsity": data_sparsity},  # data sparsity
                f"{cli_args.base_run_name}_sparse_data_{data_sparsity}",
                "sparse_gaussian"  # data_type override
            ))
    
    if cli_args.task in ["noisy", "both"]:
        # Different noise types
        for noise_type in cli_args.noise_types:
            experiments.append((
                "noisy_linear_regression",
                {"noise_type": noise_type, "noise_std": 2.0},
                {},
                f"{cli_args.base_run_name}_noisy_{noise_type}",
                None  # data_type: use default from config
            ))
        
        # Different noise_std for normal noise
        for noise_std in cli_args.noise_stds:
            experiments.append((
                "noisy_linear_regression",
                {"noise_type": "normal", "noise_std": noise_std},
                {},
                f"{cli_args.base_run_name}_noisy_normal_std_{noise_std}",
                None  # data_type: use default from config
            ))
    
    if cli_args.task == "custom":
        # Use default experiments
        default_experiments = get_default_experiments()
        # Add base_run_name prefix
        experiments = [
            (task, tk, dk, f"{cli_args.base_run_name}_{name}", dt)
            for task, tk, dk, name, dt in default_experiments
        ]
    
    # Run experiments
    print(f"\n{'='*60}")
    print(f"Total experiments to run: {len(experiments)}")
    print(f"{'='*60}\n")
    
    for idx, exp in enumerate(experiments, 1):
        # Handle both 4-tuple and 5-tuple formats
        if len(exp) == 4:
            task, task_kwargs, data_kwargs, run_name = exp
            data_type = None
        else:
            task, task_kwargs, data_kwargs, run_name, data_type = exp
        
        print(f"\n[{idx}/{len(experiments)}] Preparing: {run_name}")
        
        # Check if should skip existing
        if cli_args.skip_existing:
            # Try to find existing run by checking base out_dir
            base_config = yaml.safe_load(open(cli_args.config))
            base_out_dir = base_config.get('out_dir', '../models')
            # Check if any subdirectory has this run_name in config
            if os.path.exists(base_out_dir):
                task_dir = os.path.join(base_out_dir, task)
                if os.path.exists(task_dir):
                    for run_id in os.listdir(task_dir):
                        run_path = os.path.join(task_dir, run_id)
                        config_path = os.path.join(run_path, 'config.yaml')
                        if os.path.exists(config_path):
                            with open(config_path) as f:
                                existing_config = yaml.safe_load(f)
                                if existing_config.get('wandb', {}).get('name') == run_name:
                                    print(f"  -> Skipping (already exists): {run_name}")
                                    continue
        
        # Generate resume_id from run_name (sanitize for filesystem)
        resume_id = run_name.replace(" ", "_").replace("/", "_")
        
        try:
            run_one_experiment(
                cli_args.config,
                task,
                task_kwargs,
                data_kwargs,
                run_name,
                resume_id=resume_id,
                data_type=data_type
            )
        except Exception as e:
            print(f"\n{'!'*60}")
            print(f"ERROR in experiment: {run_name}")
            print(f"Error: {str(e)}")
            print(f"{'!'*60}\n")
            # Continue with next experiment
            continue
    
    print(f"\n{'='*60}")
    print(f"All experiments completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
