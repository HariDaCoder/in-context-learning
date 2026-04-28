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
    parser.add_argument("--aggregate", action="store_true", help="Aggregate multiple runs (one per noise) into a single combined plot.")
    parser.add_argument("--run_id_prefix", type=str, default="small_match_noise", help="Prefix used to find per-noise run directories (run_id_prefix_std{noise}).")
    parser.add_argument("--force_rerun", action="store_true", help="Force re-run of per-run plotting to regenerate dynamics JSON before aggregation.")
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
        "--skip_id_profile",
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

    if not args.aggregate:
        subprocess.run(cmd, cwd=repo_root, check=True)
        return

    # Aggregate mode: for each noise, ensure dynamics JSON exists (run plot script per-run if needed),
    # then load train_losses.json and the dynamics JSON and plot all train/test curves on one figure.
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg_prefix = args.prefix + "_aggregate"
    out_dir = args.out_dir or os.path.join(repo_root, "aggregate_plots")
    os.makedirs(out_dir, exist_ok=True)

    combined_png = os.path.join(out_dir, f"{agg_prefix}.png")
    combined_json = os.path.join(out_dir, f"{agg_prefix}.json")

    all_series = {}
    all_meta = {"noise_scales": scales, "runs": {}}

    for noise in scales:
        noise_tag = ("%g" % float(noise)).replace('.', 'p')
        run_id = f"{args.run_id_prefix}_std{noise_tag}"
        run_dir = os.path.join(args.run_path, run_id)

        if not os.path.isdir(run_dir):
            print(f"[WARN] Run directory missing: {run_dir}. Skipping noise={noise}")
            continue

        # dynamics JSON path we'll request
        dyn_dir = os.path.join(run_dir, "dynamics")
        os.makedirs(dyn_dir, exist_ok=True)
        dyn_json = os.path.join(dyn_dir, f"{agg_prefix}_std{noise_tag}.json")

        # If missing or forced, call plot_benign_harmful_dynamics.py for this run to generate dynamics JSON
        if args.force_rerun or not os.path.exists(dyn_json):
            per_cmd = [
                sys.executable,
                plot_script,
                "--run_path",
                run_dir,
                "--ood_noise_scales",
                str(noise),
                "--use_absolute_noise_std",
                "--skip_id_profile",
                "--step_stride",
                str(args.step_stride),
                "--num_eval_examples",
                str(args.num_eval_examples),
                "--prefix",
                f"{agg_prefix}_std{noise_tag}",
            ]
            print("[INFO] Generating dynamics for:", run_dir)
            subprocess.run(per_cmd, cwd=repo_root, check=True)

        if not os.path.exists(dyn_json):
            print(f"[WARN] dynamics JSON still missing: {dyn_json}. Skipping.")
            continue

        with open(dyn_json, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        steps = payload.get('steps', [])
        series = payload.get('series', {})

        # test series key
        test_series = series.get('id_matched') or series.get('id')
        if test_series is None:
            print(f"[WARN] No id/test series in {dyn_json}. Skipping.")
            continue

        # load train loss file
        train_loss_file = os.path.join(run_dir, 'train_losses.json')
        train_steps = []
        train_losses = []
        if os.path.exists(train_loss_file):
            with open(train_loss_file, 'r') as f:
                d = json.load(f)
            train_steps = sorted([int(k) for k in d.keys()])
            train_losses = [float(d[str(s)]) for s in train_steps]
        else:
            print(f"[WARN] train_losses.json missing for {run_dir}")

        all_meta['runs'][str(noise)] = {'run_dir': run_dir, 'dyn_json': dyn_json}
        all_series[str(noise)] = {'steps': steps, 'test': test_series, 'train_steps': train_steps, 'train': train_losses}

    # Plot combined
    if not all_series:
        raise RuntimeError('No series collected for aggregation')

    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.get_cmap('tab10')
    for idx, (noise, data) in enumerate(sorted(all_series.items(), key=lambda x: float(x[0]))):
        color = cmap(idx % 10)
        ax.plot(data['steps'], data['test'], label=f"test_{noise}", color=color, linewidth=2)
        if data['train']:
            ax.plot(data['train_steps'], data['train'], linestyle='--', label=f"train_{noise}", color=color, linewidth=1.2, alpha=0.8)

    ax.set_xlabel('Training step')
    ax.set_ylabel('Loss')
    ax.set_title('Combined train/test loss across noise scales')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(combined_png, dpi=220, bbox_inches='tight')
    plt.close(fig)

    with open(combined_json, 'w', encoding='utf-8') as f:
        json.dump({'meta': all_meta, 'series': all_series}, f, indent=2)

    print(f"[DONE] Saved combined plot: {combined_png}")
    print(f"[DONE] Saved combined metadata: {combined_json}")


if __name__ == "__main__":
    main()
