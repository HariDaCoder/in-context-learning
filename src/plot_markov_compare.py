"""Overlay train/test dynamics for multiple Markov runs on one figure.

This script does not use the benign/harmful OOD plotting pipeline.
Instead it:
- reads each Markov run's config and saved checkpoints
- loads train loss history from train_losses.json
- computes fresh in-distribution test loss directly from checkpoints
- overlays all runs on one combined plot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from eval import get_model_from_run
from samplers import get_data_sampler
from tasks import get_task_sampler


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _discover_checkpoint_steps(run_path: Path) -> list[int]:
    saved_steps = []
    for name in run_path.iterdir():
        if name.is_file() and name.name.startswith("model_") and name.suffix == ".pt":
            try:
                saved_steps.append(int(name.stem.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue

    state_path = run_path / "state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu")
        final_step = int(state.get("train_step", -1))
        if final_step >= 0:
            saved_steps.append(final_step)

    return sorted(set(saved_steps))


def _load_train_loss_series(run_path: Path) -> tuple[list[int], list[float]]:
    loss_file = run_path / "train_losses.json"
    if not loss_file.exists():
        return [], []

    loss_dict = _load_json(loss_file)
    steps = sorted(int(key) for key in loss_dict.keys())
    values = [float(loss_dict[str(step)]) for step in steps]
    return steps, values


def _evaluate_markov_checkpoint(model, conf, num_eval_examples: int, batch_size: int) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    n_dims = conf.model.n_dims
    n_points = int(conf.training.curriculum.points.end)
    data_sampler = get_data_sampler(conf.training.data, n_dims=n_dims, **dict(conf.training.data_kwargs))
    task_sampler = get_task_sampler(
        conf.training.task,
        n_dims,
        batch_size,
        num_tasks=getattr(conf.training, "num_tasks", None),
        **dict(conf.training.task_kwargs),
    )

    num_batches = max(num_eval_examples // batch_size, 1)
    losses = []

    with torch.no_grad():
        for _ in range(num_batches):
            xs = data_sampler.sample_xs(n_points, batch_size, n_dims)
            task = task_sampler()
            ys = task.evaluate(xs)
            pred = model(xs.to(device), ys.to(device))
            loss = task.get_training_metric()(pred, ys.to(device))
            losses.append(float(loss.item()))

    return sum(losses) / max(len(losses), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay Markov train/test loss for multiple runs.")
    parser.add_argument("--run_paths", type=str, nargs="+", required=True, help="Run directories to overlay.")
    parser.add_argument("--labels", type=str, nargs="*", default=None, help="Optional labels for the runs; must match run_paths length.")
    parser.add_argument("--out_dir", type=str, default=None, help="Where to save the combined plot and JSON.")
    parser.add_argument("--prefix", type=str, default="markov_compare", help="Output filename prefix.")
    parser.add_argument("--num_eval_examples", type=int, default=256, help="Number of eval examples per checkpoint.")
    parser.add_argument("--batch_size", type=int, default=None, help="Eval batch size.")
    parser.add_argument("--step_stride", type=int, default=1000, help="Only keep checkpoints where step % step_stride == 0.")
    parser.add_argument("--ymin", type=float, default=0.0, help="Lower y-axis bound for the combined plot.")
    parser.add_argument("--ymax", type=float, default=1.0, help="Upper y-axis bound for the combined plot.")
    args = parser.parse_args()

    run_paths = [Path(p).resolve() for p in args.run_paths]

    if args.labels is not None and len(args.labels) not in (0, len(run_paths)):
        raise ValueError("--labels must be omitted or have the same length as --run_paths")

    labels = args.labels if args.labels else [run_path.name for run_path in run_paths]

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (Path.cwd() / "markov_compare_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_series = []
    combined_meta = {"runs": []}

    for run_path, label in zip(run_paths, labels):
        config_path = run_path / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.yaml in run directory: {run_path}")

        _, conf = get_model_from_run(str(run_path), only_conf=True)
        batch_size = args.batch_size or int(conf.training.batch_size)
        if args.num_eval_examples % batch_size != 0:
            raise ValueError("num_eval_examples must be divisible by batch_size.")

        train_steps, train_losses = _load_train_loss_series(run_path)

        checkpoints = [step for step in _discover_checkpoint_steps(run_path) if step % args.step_stride == 0]
        if not checkpoints:
            raise ValueError(f"No checkpoints left after applying step_stride={args.step_stride} in {run_path}")

        test_steps = []
        test_losses = []
        for step in checkpoints:
            load_step = step if (run_path / f"model_{step}.pt").exists() else -1
            model, _ = get_model_from_run(str(run_path), step=load_step)
            test_loss = _evaluate_markov_checkpoint(model, conf, args.num_eval_examples, batch_size)
            test_steps.append(step)
            test_losses.append(test_loss)

        all_series.append(
            {
                "label": label,
                "run_path": str(run_path),
                "test_steps": test_steps,
                "test": test_losses,
                "train_steps": train_steps,
                "train": train_losses,
            }
        )
        combined_meta["runs"].append({"label": label, "run_path": str(run_path)})

    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.get_cmap("tab10")

    for idx, data in enumerate(all_series):
        color = cmap(idx % 10)
        ax.plot(data["test_steps"], data["test"], color=color, linewidth=2.4, label=f"test: {data['label']}")
        if data["train"]:
            ax.plot(
                data["train_steps"],
                data["train"],
                color=color,
                linestyle="--",
                linewidth=1.4,
                alpha=0.8,
                label=f"train: {data['label']}",
            )

    ax.set_title("Markov train/test loss across runs")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_ylim(args.ymin, args.ymax)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    fig.tight_layout()

    png_path = out_dir / f"{args.prefix}.png"
    json_out_path = out_dir / f"{args.prefix}.json"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    with open(json_out_path, "w", encoding="utf-8") as handle:
        json.dump({"meta": combined_meta, "series": all_series}, handle, indent=2)

    print(f"[DONE] Saved combined plot: {png_path}")
    print(f"[DONE] Saved combined metadata: {json_out_path}")


if __name__ == "__main__":
    main()
