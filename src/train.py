import os
import random
import uuid

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model
from training_data import sample_training_batch

torch.backends.cudnn.benchmark = True


def train_step(model, xs, ys, optimizer, loss_func):
    optimizer.zero_grad()
    output = model(xs, ys)
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    return random.sample(range(total_seeds), count)


def validate_training_config(args):
    training, model = args.training, args.model
    if model.n_embd <= 0 or model.n_head <= 0 or model.n_embd % model.n_head:
        raise ValueError("n_embd must be positive and divisible by n_head")
    if training.batch_size < 1 or training.train_steps < 1:
        raise ValueError("batch_size and train_steps must be positive")
    if training.learning_rate <= 0 or training.save_every_steps < 1:
        raise ValueError("learning_rate and save_every_steps must be positive")
    if args.wandb.log_every_steps < 1:
        raise ValueError("wandb.log_every_steps must be positive")
    for name in ("dims", "points"):
        schedule = getattr(training.curriculum, name)
        if (
            schedule.start < 1
            or schedule.end < schedule.start
            or schedule.inc < 0
            or schedule.interval < 1
        ):
            raise ValueError(f"invalid {name} curriculum")
    if training.curriculum.dims.end > model.n_dims:
        raise ValueError("dimension curriculum exceeds model.n_dims")
    if training.curriculum.points.end > model.n_positions:
        raise ValueError("point curriculum exceeds model.n_positions")
    if training.query_mode == "independent":
        if training.task != "dependent_linear_regression":
            raise ValueError(
                "independent query mode requires dependent_linear_regression"
            )
        if training.curriculum.points.start < 2:
            raise ValueError("independent query mode needs context plus a query")
        if any(key in training.data_kwargs for key in ("scale", "bias")):
            raise ValueError(
                "independent query mode currently assumes N(0,I) feature marginals"
            )
    try:
        device = torch.device(training.device if training.device != "auto" else "cpu")
    except (TypeError, RuntimeError, ValueError) as error:
        raise ValueError(f"invalid training device: {training.device}") from error
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("training device must be auto, cpu, cuda, or cuda:<index>")


def train(model, args):
    device = next(model.parameters()).device
    log_enabled = not args.test_run and args.wandb.mode != "disabled"
    if log_enabled:
        import wandb

    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"] + 1
        for i in range(state["train_step"] + 1):
            curriculum.update()
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"].cpu())
            random.setstate(state["python_rng_state"])
            if device.type == "cuda" and state.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(
                    [rng.cpu() for rng in state["cuda_rng_state"]]
                )

    n_dims = model.n_dims
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(
        args.training.data,
        n_dims=n_dims,
        **args.training.data_kwargs,
    )
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples

    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        if (
            "sparse" in args.training.task
            or args.training.task == "dependent_linear_regression"
        ):
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        xs, ys, task = sample_training_batch(
            data_sampler,
            task_sampler,
            curriculum,
            bsize,
            args.training.query_mode,
            data_sampler_args,
            task_sampler_args,
        )

        loss_func = task.get_training_metric()

        xs_device, ys_device = xs.to(device), ys.to(device)
        loss, output = train_step(model, xs_device, ys_device, optimizer, loss_func)

        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys_device).mean(dim=0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        if i % args.wandb.log_every_steps == 0 and log_enabled:
            log_values = {
                "overall_loss": loss,
                "pointwise/loss": dict(
                    zip(point_wise_tags, point_wise_loss.cpu().numpy())
                ),
                "n_points": curriculum.n_points,
                "n_dims": curriculum.n_dims_truncated,
            }
            if args.training.task == "dependent_linear_regression":
                # The signal has unit expected power. This normalizer is the
                # zero-predictor risk, and is not an OLS excess-risk claim.
                log_values["loss_over_zero_predictor"] = loss / (
                    1 + task.noise_std**2
                )
            else:
                log_values["excess_loss"] = loss / baseline_loss
            wandb.log(
                log_values,
                step=i,
            )

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if (
            i % args.training.save_every_steps == 0
            or i == args.training.train_steps - 1
        ) and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
                "torch_rng_state": torch.get_rng_state(),
                "python_rng_state": random.getstate(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state_all() if device.type == "cuda" else None
                ),
            }
            temporary_state_path = state_path + ".tmp"
            torch.save(training_state, temporary_state_path)
            os.replace(temporary_state_path, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):
    validate_training_config(args)
    random.seed(args.training.seed)
    torch.manual_seed(args.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.training.seed)
    device = args.training.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested, but CUDA is unavailable")

    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    elif args.wandb.mode != "disabled":
        import wandb

        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
            mode=args.wandb.mode,
        )

    model = build_model(args.model)
    model.to(device)
    model.train()

    train(model, args)

    if not args.test_run and args.training.eval_after_train:
        from eval import get_run_metrics

        _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
