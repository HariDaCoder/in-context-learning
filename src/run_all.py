import os
import uuid
import yaml
import argparse

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


def build_parser():
    parser = argparse.ArgumentParser(description="Run all noisy_linear_regression variants")
    parser.add_argument(
        "--config",
        default="src/conf/toy.yaml",
        help="Base config yaml (e.g., src/conf/toy.yaml)",
    )
    parser.add_argument(
        "--noise_types",
        nargs="*",
        default=[
            "uniform",
            "normal",
            "exponential",
            "beta",
            "poisson",
            "cauchy",
            "laplace",
        ],
        help="Which noise_type values to iterate",
    )
    parser.add_argument(
        "--base_run_name",
        default="noisy_sweep",
        help="Prefix for wandb.name; final name will be '<noise>_<base_run_name>'",
    )
    return parser


def run_one(base_config_path: str, noise_type: str, base_run_name: str):
    # Build a fresh Quinine parser each time and override via CLI-like args
    qparser = QuinineArgumentParser(schema=quinine_schema)
    args = qparser.parse_quinfig(
        args_list=[
            "--config",
            base_config_path,
            "--training.task",
            "noisy_linear_regression",
            "--training.task_kwargs.noise_type",
            noise_type,
            "--wandb.name",
            f"{noise_type}_{base_run_name}",
        ]
    )

    # Make output directory unique and persist resolved config
    prepare_out_dir(args)

    # Kick off training for this configuration
    train_main(args)


def main():
    parser = build_parser()
    cli_args = parser.parse_args()

    for noise in cli_args.noise_types:
        run_one(cli_args.config, noise, cli_args.base_run_name)


if __name__ == "__main__":
    main()

