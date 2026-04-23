import argparse
import json
import os
import re
from urllib.parse import urlparse

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import torch
except ImportError:
    torch = None

try:
    import wandb
except ImportError:
    wandb = None


CHECKPOINT_PATTERN = re.compile(r"^model_(\d+)\.pt$")


def _decode_scale_tag(tag):
    return float(tag.replace("p", "."))


def _normalize_wandb_run_ref(run_ref):
    """Convert a W&B URL or path to the api.run() ref format: entity/project/run_id."""
    if not run_ref:
        return None

    ref = run_ref.strip()
    if "://" in ref:
        ref = urlparse(ref).path

    parts = [part for part in ref.strip("/").split("/") if part]
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx >= 2 and runs_idx + 1 < len(parts):
            return "/".join([parts[runs_idx - 2], parts[runs_idx - 1], parts[runs_idx + 1]])
    if len(parts) >= 3:
        return "/".join(parts[-3:])
    return ref


def _load_wandb_series(run_ref, metric_name="overall_loss"):
    """Load a metric series from a W&B run history."""
    if wandb is None:
        raise ImportError("Missing dependency: wandb. Please install wandb to load W&B history.")

    api = wandb.Api()
    run = api.run(_normalize_wandb_run_ref(run_ref))

    steps = []
    values = []
    for row in run.scan_history(keys=["_step", metric_name]):
        step = row.get("_step")
        value = row.get(metric_name)
        if step is None or value is None:
            continue
        steps.append(int(step))
        values.append(float(value))

    return steps, values


def _extract_loss_scalar(metric_dict, reduction):
    means = metric_dict["mean"]
    if reduction == "last":
        return float(means[-1])
    if reduction == "mean":
        return float(sum(means) / max(len(means), 1))
    raise ValueError(f"Unsupported reduction: {reduction}")


def _discover_checkpoints(run_path):
    if torch is None:
        raise ImportError("Missing dependency: torch. Please install torch in your environment.")

    saved_steps = []
    for name in os.listdir(run_path):
        match = CHECKPOINT_PATTERN.match(name)
        if match:
            saved_steps.append(int(match.group(1)))
    saved_steps = sorted(set(saved_steps))

    state_path = os.path.join(run_path, "state.pt")
    final_step = -1
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location="cpu")
        final_step = int(state.get("train_step", -1))

    all_steps = sorted(set(saved_steps + ([final_step] if final_step >= 0 else [])))
    return saved_steps, final_step, all_steps


def _filter_steps(steps, min_step=None, max_step=None, step_stride=None):
    filtered = []
    for step in steps:
        if min_step is not None and step < min_step:
            continue
        if max_step is not None and step > max_step:
            continue
        if step_stride is not None and step_stride > 0 and step % step_stride != 0:
            continue
        filtered.append(step)
    return filtered


def _detect_harmful_onset(steps, id_losses, ood_losses, rel_threshold, abs_threshold):
    if not steps:
        return {"best_step": None, "harmful_onset_step": None}

    best_idx = min(range(len(ood_losses)), key=lambda idx: ood_losses[idx])
    best_step = int(steps[best_idx])
    best_ood = float(ood_losses[best_idx])
    id_at_best = float(id_losses[best_idx])

    harmful_step = None
    for i in range(best_idx + 1, len(steps)):
        ood_now = float(ood_losses[i])
        id_now = float(id_losses[i])

        rel_increase = ood_now >= best_ood * (1.0 + rel_threshold)
        abs_increase = (ood_now - best_ood) >= abs_threshold
        id_not_worse = id_now <= id_at_best + abs_threshold

        if (rel_increase or abs_increase) and id_not_worse:
            harmful_step = int(steps[i])
            break

    return {
        "best_step": best_step,
        "harmful_onset_step": harmful_step,
        "best_ood_loss": best_ood,
    }


def _plot_series(steps, series, transitions, output_path, reduction, train_series=None):
    if plt is None:
        raise ImportError(
            "Missing dependency: matplotlib. Please install matplotlib in your environment."
        )

    fig, ax = plt.subplots(figsize=(11, 6.5))

    if "id" not in series:
        raise ValueError("ID series is required to plot dynamics.")

    color_map = {}
    palette = plt.get_cmap("tab10")

    ax.plot(steps, series["id"], marker="o", linewidth=2.8, color="black", label="ID")

    if train_series is not None:
        train_steps, train_losses = train_series
        ax.plot(
            train_steps,
            train_losses,
            linestyle="--",
            linewidth=2.2,
            color="dimgray",
            label="train loss (wandb)",
        )

    ood_names = [name for name in series if name != "id"]
    for idx, name in enumerate(ood_names):
        color = palette(idx % 10)
        color_map[name] = color
        ax.plot(steps, series[name], marker="o", linewidth=2, color=color, label=name)

        best_step = transitions.get(name, {}).get("best_step")
        harmful_step = transitions.get(name, {}).get("harmful_onset_step")

        if best_step is not None and best_step in steps:
            best_idx = steps.index(best_step)
            ax.scatter(
                [best_step],
                [series[name][best_idx]],
                color=color,
                edgecolors="white",
                linewidths=0.9,
                s=70,
                zorder=5,
            )

        if harmful_step is not None and harmful_step in steps:
            harm_idx = steps.index(harmful_step)
            ax.scatter(
                [harmful_step],
                [series[name][harm_idx]],
                color=color,
                marker="x",
                s=80,
                linewidths=2,
                zorder=6,
            )

    harmful_steps = [
        info.get("harmful_onset_step")
        for info in transitions.values()
        if info.get("harmful_onset_step") is not None
    ]
    if harmful_steps:
        earliest_harm = min(harmful_steps)
        ax.axvline(earliest_harm, color="red", linestyle="--", alpha=0.55)
        ax.text(
            earliest_harm,
            ax.get_ylim()[1],
            " harmful onset",
            color="red",
            fontsize=9,
            va="top",
            ha="left",
        )

    ax.set_title("ICL Benign/Harmful Training Dynamics")
    ax.set_xlabel("Training step")
    ax.set_ylabel(f"Loss ({reduction})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] Saved plot PNG: {output_path}")


def _collect_noise_series(series, include_nonstation=False):
    """Return list of tuples (eval_name, scale_value) for OOD Gaussian-noise evals."""
    selected = []
    for eval_name in series.keys():
        if eval_name.startswith("ood_gaussian_noise_x"):
            selected.append((eval_name, _decode_scale_tag(eval_name.split("_x", 1)[1])))
            continue
        if eval_name.startswith("ood_gaussian_noise_std"):
            selected.append((eval_name, _decode_scale_tag(eval_name.split("_std", 1)[1])))
            continue
        if include_nonstation and eval_name.startswith("ood_nonstation_gaussian_noise_x"):
            selected.append((eval_name, _decode_scale_tag(eval_name.split("_x", 1)[1])))
            continue
        if include_nonstation and eval_name.startswith("ood_nonstation_gaussian_noise_std"):
            selected.append((eval_name, _decode_scale_tag(eval_name.split("_std", 1)[1])))
            continue
    selected.sort(key=lambda item: item[1])
    return selected


def _plot_fixed_step_noise(
    steps,
    series,
    fixed_steps,
    output_path,
    reduction,
    x_label,
    include_nonstation=False,
):
    if plt is None:
        raise ImportError(
            "Missing dependency: matplotlib. Please install matplotlib in your environment."
        )

    selected_noise = _collect_noise_series(series, include_nonstation=include_nonstation)
    if not selected_noise:
        raise ValueError("No OOD Gaussian-noise series found to build fixed-step slices.")

    step_to_idx = {int(step): idx for idx, step in enumerate(steps)}
    valid_fixed_steps = [int(s) for s in fixed_steps if int(s) in step_to_idx]
    if not valid_fixed_steps:
        raise ValueError(
            "None of fixed_steps_for_noise_plot are in evaluated steps. "
            f"Requested={fixed_steps}, available={steps}"
        )

    xs = [scale for _, scale in selected_noise]
    eval_names = [name for name, _ in selected_noise]

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    palette = plt.get_cmap("viridis")

    for idx, step in enumerate(valid_fixed_steps):
        step_idx = step_to_idx[step]
        ys = [float(series[name][step_idx]) for name in eval_names]
        color = palette(idx / max(len(valid_fixed_steps) - 1, 1))
        ax.plot(xs, ys, marker="o", linewidth=2, color=color, label=f"step={step}")

        if "id" in series:
            id_loss = float(series["id"][step_idx])
            ax.axhline(
                id_loss,
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.5,
            )

    ax.set_title("Fixed-Step Noise Slices (ID/OOD)")
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"Loss ({reduction})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot benign/harmful ICL dynamics across checkpoints.")
    parser.add_argument("--run_path", type=str, required=True, help="Path to a specific run directory.")
    parser.add_argument(
        "--ood_noise_scales",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0],
        help="Noise scales for OOD evaluation.",
    )
    parser.add_argument(
        "--use_noise_multipliers",
        action="store_true",
        help="Interpret ood_noise_scales as multipliers of training noise_std (default).",
    )
    parser.add_argument(
        "--use_absolute_noise_std",
        action="store_true",
        help="Interpret ood_noise_scales as absolute noise_std values.",
    )
    parser.add_argument("--include_nonstation", action="store_true", help="Add nonstation OOD input shift.")
    parser.add_argument("--nonstation_coef_base", type=float, default=0.5)
    parser.add_argument("--nonstation_coef_amplitude", type=float, default=0.4)
    parser.add_argument("--nonstation_noise_std", type=float, default=0.1)
    parser.add_argument("--num_eval_examples", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--min_step", type=int, default=None)
    parser.add_argument("--max_step", type=int, default=None)
    parser.add_argument(
        "--step_stride",
        type=int,
        default=None,
        help="Keep checkpoints where step %% step_stride == 0.",
    )
    parser.add_argument(
        "--reduction",
        type=str,
        default="last",
        choices=["last", "mean"],
        help="How to reduce per-prompt losses into one scalar per step.",
    )
    parser.add_argument("--harmful_rel_threshold", type=float, default=0.15)
    parser.add_argument("--harmful_abs_threshold", type=float, default=1e-4)
    parser.add_argument(
        "--wandb_run",
        type=str,
        default=None,
        help="Optional W&B run URL or path (entity/project/run_id) to overlay train loss from history.",
    )
    parser.add_argument(
        "--wandb_metric",
        type=str,
        default="overall_loss",
        help="W&B history metric to plot as training loss.",
    )
    parser.add_argument(
        "--fixed_steps_for_noise_plot",
        type=int,
        nargs="+",
        default=None,
        help="If set, also create noise-slice plot (x=noise scale/std, y=loss) at these training steps.",
    )
    parser.add_argument(
        "--noise_plot_include_nonstation",
        action="store_true",
        help="Include nonstation OOD-noise series in fixed-step noise plot if available.",
    )
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--prefix", type=str, default="benign_harmful_dynamics")

    args = parser.parse_args()

    if torch is None:
        raise ImportError("Missing dependency: torch. Please install torch in your environment.")

    from eval import build_benign_harmful_dynamics_evals, eval_model, get_model_from_run

    use_noise_multipliers = True
    if args.use_absolute_noise_std:
        use_noise_multipliers = False
    elif args.use_noise_multipliers:
        use_noise_multipliers = True

    _, conf = get_model_from_run(args.run_path, only_conf=True)

    nonstation_kwargs = {
        "coef_base": args.nonstation_coef_base,
        "coef_amplitude": args.nonstation_coef_amplitude,
        "noise_std": args.nonstation_noise_std,
    }
    eval_profile = build_benign_harmful_dynamics_evals(
        conf,
        ood_noise_scales=args.ood_noise_scales,
        include_nonstation=args.include_nonstation,
        nonstation_data_kwargs=nonstation_kwargs,
        use_noise_multipliers=use_noise_multipliers,
    )

    saved_steps, final_step, candidate_steps = _discover_checkpoints(args.run_path)
    if not candidate_steps:
        raise FileNotFoundError(
            f"No checkpoints found in {args.run_path}. Need state.pt and/or model_*.pt files."
        )

    eval_steps = _filter_steps(
        candidate_steps,
        min_step=args.min_step,
        max_step=args.max_step,
        step_stride=args.step_stride,
    )
    if not eval_steps:
        raise ValueError("No checkpoints left after filtering; adjust min/max/stride.")

    saved_step_set = set(saved_steps)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Evaluating {len(eval_steps)} checkpoints: {eval_steps}")
    print(f"[INFO] Evaluation profile: {list(eval_profile.keys())}")

    batch_size = args.batch_size or int(conf.training.batch_size)
    if args.num_eval_examples % batch_size != 0:
        raise ValueError("num_eval_examples must be divisible by batch_size.")

    wandb_train_series = None
    if args.wandb_run:
        wandb_train_series = _load_wandb_series(args.wandb_run, metric_name=args.wandb_metric)
        print(
            f"[INFO] Loaded W&B train series: {len(wandb_train_series[0])} points from {args.wandb_run}"
        )

    series = {name: [] for name in eval_profile.keys()}

    for step in eval_steps:
        load_step = step if step in saved_step_set else -1
        if load_step == -1 and step != final_step:
            raise FileNotFoundError(f"Missing model_{step}.pt and not equal to final state step.")

        model, _ = get_model_from_run(args.run_path, step=load_step)
        model = model.to(device).eval()

        print(f"[INFO] Evaluating step={step} (load_step={load_step})")
        with torch.no_grad():
            for eval_name, kwargs in eval_profile.items():
                eval_kwargs = dict(kwargs)
                eval_kwargs["num_eval_examples"] = args.num_eval_examples
                eval_kwargs["batch_size"] = batch_size
                eval_kwargs["verbose"] = False

                metrics = eval_model(model, **eval_kwargs)
                scalar_loss = _extract_loss_scalar(metrics, args.reduction)
                series[eval_name].append(scalar_loss)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    id_losses = series.get("id", [])
    transitions = {}
    for eval_name, losses in series.items():
        if eval_name == "id":
            continue
        transitions[eval_name] = _detect_harmful_onset(
            eval_steps,
            id_losses,
            losses,
            rel_threshold=args.harmful_rel_threshold,
            abs_threshold=args.harmful_abs_threshold,
        )

    out_dir = args.out_dir or os.path.join(args.run_path, "dynamics")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, f"{args.prefix}.json")
    png_path = os.path.join(out_dir, f"{args.prefix}.png")

    payload = {
        "run_path": args.run_path,
        "steps": eval_steps,
        "reduction": args.reduction,
        "series": series,
        "transitions": transitions,
        "eval_profile": eval_profile,
        "num_eval_examples": args.num_eval_examples,
        "batch_size": batch_size,
        "use_noise_multipliers": use_noise_multipliers,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    _plot_series(eval_steps, series, transitions, png_path, args.reduction, train_series=wandb_train_series)

    if args.fixed_steps_for_noise_plot:
        x_label = "Noise scale (multiplier)" if use_noise_multipliers else "Noise std"
        noise_png_path = os.path.join(out_dir, f"{args.prefix}_fixed_step_noise.png")
        _plot_fixed_step_noise(
            eval_steps,
            series,
            args.fixed_steps_for_noise_plot,
            noise_png_path,
            args.reduction,
            x_label=x_label,
            include_nonstation=args.noise_plot_include_nonstation,
        )

        fixed_payload = {
            "requested_fixed_steps": args.fixed_steps_for_noise_plot,
            "evaluated_steps": eval_steps,
            "noise_series": [name for name, _ in _collect_noise_series(series, args.noise_plot_include_nonstation)],
            "use_noise_multipliers": use_noise_multipliers,
        }
        fixed_json_path = os.path.join(out_dir, f"{args.prefix}_fixed_step_noise.json")
        with open(fixed_json_path, "w", encoding="utf-8") as f:
            json.dump(fixed_payload, f, indent=2)

        print(f"[DONE] Saved fixed-step noise plot: {noise_png_path}")
        print(f"[DONE] Saved fixed-step noise metadata: {fixed_json_path}")

    print(f"[DONE] Saved metrics JSON: {json_path}")
    print(f"[DONE] Saved plot PNG: {png_path}")
    print("[DONE] Transition summary:")
    for name, info in transitions.items():
        print(
            f"  - {name}: best_step={info.get('best_step')}, "
            f"harmful_onset_step={info.get('harmful_onset_step')}"
        )


if __name__ == "__main__":
    main()
