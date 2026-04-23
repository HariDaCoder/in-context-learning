import argparse
import os
import subprocess
import sys


def _build_scales(start, end, step):
    if step <= 0:
        raise ValueError("step must be > 0")
    values = []
    current = start
    eps = step * 1e-6
    while current <= end + eps:
        values.append(round(current, 6))
        current += step
    return values


def main():
    parser = argparse.ArgumentParser(
        description="Wrapper to plot train loss + test loss over OOD noise scales."
    )
    parser.add_argument("--run_path", type=str, required=True)
    parser.add_argument("--wandb_run", type=str, default=None)
    parser.add_argument("--wandb_metric", type=str, default="overall_loss")
    parser.add_argument("--noise_start", type=float, default=0.0)
    parser.add_argument("--noise_end", type=float, default=5.0)
    parser.add_argument("--noise_step", type=float, default=0.5)
    parser.add_argument("--step_stride", type=int, default=5000)
    parser.add_argument("--num_eval_examples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--include_nonstation", action="store_true")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--prefix", type=str, default="train_test_noise_0_5")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    plot_script = os.path.join(os.path.dirname(__file__), "plot_benign_harmful_dynamics.py")

    scales = _build_scales(args.noise_start, args.noise_end, args.noise_step)

    cmd = [
        sys.executable,
        plot_script,
        "--run_path",
        args.run_path,
        "--ood_noise_scales",
        *[str(v) for v in scales],
        "--use_absolute_noise_std",
        "--step_stride",
        str(args.step_stride),
        "--num_eval_examples",
        str(args.num_eval_examples),
        "--prefix",
        args.prefix,
    ]

    if args.batch_size is not None:
        cmd.extend(["--batch_size", str(args.batch_size)])

    if args.include_nonstation:
        cmd.append("--include_nonstation")

    if args.wandb_run:
        cmd.extend(["--wandb_run", args.wandb_run, "--wandb_metric", args.wandb_metric])

    if args.out_dir:
        cmd.extend(["--out_dir", args.out_dir])

    print("[INFO] Noise scales:", scales)
    print("[INFO] Running:", " ".join(cmd))

    subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
