import argparse
import os
import subprocess
import sys
import time

import yaml


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_out_dir_from_config(config_path):
    conf = _load_yaml(config_path)
    out_dir = conf.get("out_dir")
    if not out_dir:
        raise ValueError(f"Missing out_dir in config: {config_path}")
    return out_dir


def _resolve_out_root(config_path, config_out_dir):
    base_dir = os.path.dirname(config_path)
    return os.path.abspath(os.path.join(base_dir, config_out_dir))


def _find_latest_run_id(out_root):
    if not os.path.isdir(out_root):
        return None

    candidates = []
    for name in os.listdir(out_root):
        full = os.path.join(out_root, name)
        if not os.path.isdir(full):
            continue
        cfg = os.path.join(full, "config.yaml")
        if os.path.exists(cfg):
            candidates.append((os.path.getmtime(full), name))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _format_noise_tag(noise_value):
    return f"{float(noise_value):g}".replace(".", "p")


def _build_noise_list(start, end, step):
    if step <= 0:
        raise ValueError("noise_step must be > 0")
    values = []
    cur = float(start)
    eps = step * 1e-6
    while cur <= float(end) + eps:
        values.append(round(cur, 6))
        cur += step
    return values


def main():
    parser = argparse.ArgumentParser(
        description="Auto-resume trainer for benign/harmful dynamics using small model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/conf/benign_harmful_dynamics_small.yaml",
        help="Training config path.",
    )
    parser.add_argument("--noise_start", type=float, default=0.5)
    parser.add_argument("--noise_end", type=float, default=5.0)
    parser.add_argument("--noise_step", type=float, default=0.5)
    parser.add_argument(
        "--run_id_prefix",
        type=str,
        default="small_match_noise",
        help="Prefix used to build per-noise resume_id.",
    )
    parser.add_argument(
        "--retry_wait_seconds",
        type=int,
        default=15,
        help="Wait seconds before restarting training after failure.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=1000,
        help="Maximum retries before giving up.",
    )
    parser.add_argument("--plot_after_success", action="store_true")
    parser.add_argument("--plot_prefix", type=str, default="small_matched_noise")
    parser.add_argument("--plot_step_stride", type=int, default=5000)
    parser.add_argument("--plot_num_eval_examples", type=int, default=256)
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.abspath(os.path.join(repo_root, args.config))

    config_out_dir = _get_out_dir_from_config(config_path)
    out_root = _resolve_out_root(config_path, config_out_dir)
    os.makedirs(out_root, exist_ok=True)

    noise_values = _build_noise_list(args.noise_start, args.noise_end, args.noise_step)
    print(f"[INFO] Matched train/test noise list: {noise_values}")

    for noise_std in noise_values:
        noise_tag = _format_noise_tag(noise_std)
        run_id = f"{args.run_id_prefix}_std{noise_tag}"
        print(f"[INFO] ===== Noise std={noise_std} | run_id={run_id} =====")

        for attempt in range(1, args.max_retries + 1):
            train_cmd = [
                sys.executable,
                os.path.join("src", "train.py"),
                "--config",
                args.config,
                "--training.resume_id",
                run_id,
                "--training.task_kwargs.noise_std",
                str(noise_std),
            ]

            print(f"[INFO] Attempt {attempt}/{args.max_retries}")
            print("[INFO] Train command:", " ".join(train_cmd))

            result = subprocess.run(train_cmd, cwd=repo_root)
            if result.returncode == 0:
                print("[DONE] Training completed successfully for this noise.")
                break

            print(
                f"[WARN] Train exited with code {result.returncode}. "
                f"Restarting in {args.retry_wait_seconds}s..."
            )
            time.sleep(args.retry_wait_seconds)
        else:
            raise RuntimeError(
                f"Reached max retries without successful training completion for noise_std={noise_std}."
            )

        if args.plot_after_success:
            run_path = os.path.join(out_root, run_id)
            plot_cmd = [
                sys.executable,
                os.path.join("src", "plot_benign_harmful_dynamics.py"),
                "--run_path",
                run_path,
                "--ood_noise_scales",
                str(noise_std),
                "--use_absolute_noise_std",
                "--step_stride",
                str(args.plot_step_stride),
                "--num_eval_examples",
                str(args.plot_num_eval_examples),
                "--prefix",
                f"{args.plot_prefix}_std{noise_tag}",
            ]
            print("[INFO] Plot command:", " ".join(plot_cmd))
            subprocess.run(plot_cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
