import json
import os
import sys

from munch import Munch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import yaml

import models
from samplers import get_data_sampler, sample_transformation
from tasks import get_task_sampler


def get_model_from_run(run_path, step=-1, only_conf=False):
    config_path = os.path.join(run_path, "config.yaml")
    with open(config_path) as fp:  # we don't Quinfig it to avoid inherits
        conf = Munch.fromDict(yaml.safe_load(fp))
    if only_conf:
        return None, conf

    model = models.build_model(conf.model)

    if step == -1:
        state_path = os.path.join(run_path, "state.pt")
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
    else:
        model_path = os.path.join(run_path, f"model_{step}.pt")
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    return model, conf


# Functions for evaluation


def eval_batch(model, task_sampler, xs, xs_p=None):
    task = task_sampler()
    if torch.cuda.is_available() and model.name.split("_")[0] in ["gpt2", "lstm"]:
        device = "cuda"
    else:
        device = "cpu"

    if xs_p is None:
        ys = task.evaluate(xs)
        pred = model(xs.to(device), ys.to(device)).detach()
        metrics = task.get_metric()(pred.cpu(), ys)
    else:
        b_size, n_points, _ = xs.shape
        metrics = torch.zeros(b_size, n_points)
        for i in range(n_points):
            xs_comb = torch.cat((xs[:, :i, :], xs_p[:, i:, :]), dim=1)
            ys = task.evaluate(xs_comb)

            pred = model(xs_comb.to(device), ys.to(device), inds=[i]).detach()
            metrics[:, i] = task.get_metric()(pred.cpu(), ys)[:, i]

    return metrics


# Functions for generating different kinds of train/test data


def gen_standard(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)

    return xs, None


def gen_opposite_quadrants(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    pattern = torch.randn([b_size, 1, xs.shape[2]]).sign()

    xs_train_pre = xs.abs() * pattern
    xs_test_post = -xs_train_pre

    return xs_train_pre, xs_test_post


def gen_random_quadrants(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    pattern = torch.randn([b_size, 1, xs.shape[2]]).sign()

    xs_train_pre = xs.abs() * pattern
    xs_test_post = xs

    return xs_train_pre, xs_test_post


def gen_orthogonal_train_test(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    n_dim = xs.shape[2]
    n_points = min(n_points, n_dim)
    # raise ValueError("number of points should be at most the dimension.")
    xs_train_pre = xs
    xs_test_post = torch.zeros(xs.shape)
    for i in range(n_points):
        xs_test_post_i = xs[:, i : i + 1, :]
        xs_train_pre_i = xs[:, :i, :]
        _, _, Vt = torch.linalg.svd(xs_train_pre_i, full_matrices=False)
        xs_train_pre_i_projection = Vt.transpose(1, 2) @ Vt
        xs_test_post_i_orthogonalized = (
            xs_test_post_i - xs_test_post_i @ xs_train_pre_i_projection
        )
        xs_test_post_i_normalized = (
            xs_test_post_i_orthogonalized
            * xs_test_post_i.norm(dim=2).unsqueeze(2)
            / xs_test_post_i_orthogonalized.norm(dim=2).unsqueeze(2)
        )

        xs_test_post[:, i : i + 1, :] = xs_test_post_i_normalized

    return xs_train_pre, xs_test_post


def gen_overlapping_train_test(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    xs_train_pre = xs
    xs_test_post = xs.clone()
    b_size = xs.shape[0]
    for i in range(1, n_points):
        xs_train_pre_i = xs[:, :i, :]
        perm = torch.stack([torch.randperm(i) for _ in range(b_size)]).unsqueeze(dim=1)
        ind_mat = (perm == 0) + 0.0
        xs_test_post[:, i : i + 1, :] = ind_mat @ xs_train_pre_i

    return xs_train_pre, xs_test_post


def aggregate_metrics(metrics, bootstrap_trials=1000):
    """
    Takes as input a tensor of shape (num_eval, n_points) and returns a dict with
    per-point mean, stddev, and bootstrap limits
    """
    results = {}
    results["mean"] = metrics.mean(dim=0)
    results["std"] = metrics.std(dim=0, unbiased=True)
    n = len(metrics)
    bootstrap_indices = torch.randint(n, size=(bootstrap_trials, n))
    bootstrap_means = metrics[bootstrap_indices].mean(dim=1).sort(dim=0)[0]
    results["bootstrap_low"] = bootstrap_means[int(0.05 * bootstrap_trials), :]
    results["bootstrap_high"] = bootstrap_means[int(0.95 * bootstrap_trials), :]

    return {k: v.tolist() for k, v in results.items()}


def eval_model(
    model,
    task_name,
    data_name,
    n_dims,
    n_points,
    prompting_strategy,
    num_eval_examples=1280,
    batch_size=64,
    data_sampler_kwargs={},
    task_sampler_kwargs={},
):
    """
    Evaluate a model on a task with a variety of strategies.
       Args:
       - task: which base task we are evaluating on. E.g., "linear_regression"
       - prompting_strategy: how to construct the prompt, e.g., "random_quadrants"
       - num_eval_examples: total number of examples to evaluate on
       - **sampler_kwargs: remaining arguments to pass directly to the sampler
    """
    print(f"[DEBUG] eval_model: task={task_name}, data={data_name}, strategy={prompting_strategy}")

    assert num_eval_examples % batch_size == 0
    print(f"[DEBUG] Creating data sampler with kwargs: {data_sampler_kwargs}")
    data_sampler = get_data_sampler(data_name, n_dims, **data_sampler_kwargs)
    print(f"[DEBUG] Creating task sampler with kwargs: {task_sampler_kwargs}")
    task_sampler = get_task_sampler(
        task_name, n_dims, batch_size, **task_sampler_kwargs
    )
    print(f"[DEBUG] Samplers created, starting eval batches...")

    all_metrics = []

    generating_func = globals()[f"gen_{prompting_strategy}"]
    for i in range(num_eval_examples // batch_size):
        print(f"[DEBUG]   Batch {i+1}/{num_eval_examples // batch_size}")
        xs, xs_p = generating_func(data_sampler, n_points, batch_size)

        metrics = eval_batch(model, task_sampler, xs, xs_p)
        all_metrics.append(metrics)

    metrics = torch.cat(all_metrics, dim=0)

    return aggregate_metrics(metrics)


def build_evals(conf):
    n_dims = conf.model.n_dims
    n_points = conf.training.curriculum.points.end
    batch_size = conf.training.batch_size

    task_name = conf.training.task
    data_name = conf.training.data

    # Sanitize kwargs to avoid passing unsupported keys during evaluation
    data_whitelist = {
        "gaussian": {"bias", "scale"},
        "sparse_gaussian": {"k", "bias", "scale"},
        "ar1": {"rho", "noise_std", "bias", "scale", "compute_gradient"},
        "vr1": {"ar1_mat", "noise_std", "bias", "scale"},
        "ar2": {"ar1_coef", "ar2_coef", "noise_std", "bias", "scale"},
        "vr2": {"ar1_mat", "ar2_mat", "noise_std", "bias", "scale"},
        "nonstation": {"coef_base", "coef_amplitude", "noise_std", "bias", "scale"},
        "exponential": {"bias", "scale", "rate"},
        "laplace": {"bias", "scale", "loc", "laplace_scale"},
        "gamma": {"bias", "scale", "concentration", "rate"},
        "beta": {"bias", "scale", "alpha", "beta"},
        "uniform": {"bias", "scale", "low", "high"},
    }
    task_whitelist = {
        "linear_regression": {"scale", "uniform"},
        "sparse_linear_regression": {"scale", "sparsity", "valid_coords"},
        "linear_classification": {"scale", "uniform"},
        "relu_2nn_regression": {"scale", "hidden_layer_size"},
        "decision_tree": {"depth"},
        "noisy_linear_regression": {"scale", "noise_std", "renormalize_ys", "noise_type", "uniform", "w_distribution", "w_kwargs", "noise_kwargs", "loss_type"},
        "ar1_linear_regression": {"scale", "ar_coef", "noise_std", "compute_gradient"},
        "uniform_hypersphere_regression": {"scale"},
        "linear_regression": {"scale", "uniform"},
        "sparse_linear_regression": {"scale", "sparsity", "valid_coords"},
        "sparse_regression_killer": {"scale", "k_sparse"},
        "heavy_tail_noise_killer": {"scale", "noise_type", "df", "noise_scale"},
        "bounded_support_killer": {"scale", "rate"},
        "mixture_tasks_killer": {"scale"},
        "transfer_tradeoff_task": {"scale", "prior_type", "mixture_std"},
    
    }
    original_data_kwargs = conf.training.data_kwargs if hasattr(conf.training, "data_kwargs") else {}
    original_task_kwargs = conf.training.task_kwargs if hasattr(conf.training, "task_kwargs") else {}
    cleaned_data_kwargs = {k: v for k, v in (original_data_kwargs or {}).items() if k in data_whitelist.get(data_name, set())}
    cleaned_task_kwargs = {k: v for k, v in (original_task_kwargs or {}).items() if k in task_whitelist.get(task_name, set())}

    base_kwargs = {
        "task_name": task_name,
        "n_dims": n_dims,
        "n_points": n_points,
        "batch_size": batch_size,
        "data_name": data_name,
        "prompting_strategy": "standard",
        # "data_sampler_kwargs": conf.training.data_kwargs if hasattr(conf.training, "data_kwargs") else {},
        # "task_sampler_kwargs": conf.training.task_kwargs
        "data_sampler_kwargs": cleaned_data_kwargs,
        "task_sampler_kwargs": cleaned_task_kwargs
    }

    evaluation_kwargs = {}

    evaluation_kwargs["standard"] = {"prompting_strategy": "standard"}
    # evaluation_kwargs["gradient"] = {
    #     "prompting_strategy": "standard",
    #     # "task_sampler_kwargs": {"compute_gradient": True}
    # }
    
    # task_name =["linear_regression" if task_name == "ar1_linear_regression" else task_name][0]
    if task_name not in ["linear_regression", "ar1_linear_regression"]:
        if task_name != "linear_regression":
            if task_name in ["relu_2nn_regression"]:
                evaluation_kwargs["linear_regression"] = {"task_name": "linear_regression"}
            for name, kwargs in evaluation_kwargs.items():
                # allow kwargs to override base_kwargs values
                evaluation_kwargs[name] = base_kwargs.copy()
                evaluation_kwargs[name].update(kwargs)
            return evaluation_kwargs

    for strategy in [
        "random_quadrants",
        "orthogonal_train_test",
        "overlapping_train_test",
    ]:
        evaluation_kwargs[strategy] = {"prompting_strategy": strategy}

    for method in ["half_subspace", "skewed"]:
        if "subspace" in method:
            eigenvals = torch.zeros(n_dims)
            eigenvals[: n_dims // 2] = 1
        else:
            eigenvals = 1 / (torch.arange(n_dims) + 1)

        scale = sample_transformation(eigenvals, normalize=True)
        evaluation_kwargs[f"{method}"] = {
            "data_sampler_kwargs": {"scale": scale},
        }

    for dim in ["x", "y"]:
        for scale in [0.333, 0.5, 2, 3]:
            if dim == "x":
                eigenvals = scale * torch.ones(n_dims)
                t = sample_transformation(eigenvals)
                scaling_args = {"data_sampler_kwargs": {"scale": t}}
            else:
                eigenvals = scale * torch.ones(n_dims)
                scaling_args = {"task_sampler_kwargs": {"scale": scale}}

            evaluation_kwargs[f"scale-{dim}={scale}"] = scaling_args

    evaluation_kwargs[f"noisyLR"] = {
        "task_sampler_kwargs": {"renormalize_ys": True, "noise_std": 1},
        "task_name": "noisy_linear_regression",
    }

    # Case 1: Scale Mismatch OOD test
    if conf.training.task == "scale_mismatch_killer":
        evaluation_kwargs = {}
        # Standard eval (in-distribution)
        evaluation_kwargs["standard"] = base_kwargs.copy()
        # OOD eval: w ~ N(100, 1)
        ood_kwargs = base_kwargs.copy()
        ood_kwargs["task_sampler_kwargs"] = dict(base_kwargs.get("task_sampler_kwargs", {}))
        ood_kwargs["task_sampler_kwargs"]["train_mode"] = False
        evaluation_kwargs["ood_scale_mismatch"] = ood_kwargs
        return evaluation_kwargs

    # Case 2: Over-Skeptic OOD test
    if conf.training.task == "noisy_linear_regression" and conf.training.task_kwargs.get("noise_std", 0) >= 20:
        evaluation_kwargs = {}
        # Standard eval (noisy)
        evaluation_kwargs["standard"] = base_kwargs.copy()
        # OOD eval: linear regression, no noise
        ood_kwargs = base_kwargs.copy()
        ood_kwargs["task_name"] = "linear_regression"
        ood_kwargs["task_sampler_kwargs"] = {}
        evaluation_kwargs["ood_clean"] = ood_kwargs
        return evaluation_kwargs
    # Case 3: Anti-Sparsity Trap (Train sparse, eval densee)
    if conf.training.task == "sparse_linear_regression" and conf.training.task_kwargs.get("sparsity", 0) <= 2:
        evaluation_kwargs = {}
        evaluation_kwargs = {}
        # Standard eval (mixed)
        evaluation_kwargs["standard"] = base_kwargs.copy()
        # OOD eval: linear regression only
        ood_kwargs = base_kwargs.copy()
        ood_kwargs["task_name"] = "linear_regression"
        ood_kwargs["task_sampler_kwargs"] = {}
        evaluation_kwargs["ood_linear"] = ood_kwargs
        return evaluation_kwargs
    # Case 4: Task Confusion (Train mixed, eval linear)
    if conf.training.task == "mixture_tasks_killer":
        evaluation_kwargs = {}
        # Standard eval (mixed)
        evaluation_kwargs["standard"] = base_kwargs.copy()
        # OOD eval: linear regression only
        ood_kwargs = base_kwargs.copy()
        ood_kwargs["task_name"] = "linear_regression"
        ood_kwargs["task_sampler_kwargs"] = {}
        evaluation_kwargs["ood_linear"] = ood_kwargs
        return evaluation_kwargs
    for name, kwargs in evaluation_kwargs.items():
        # allow kwargs to override base_kwargs values
        evaluation_kwargs[name] = base_kwargs.copy()
        evaluation_kwargs[name].update(kwargs)

    return evaluation_kwargs



def compute_evals(all_models, evaluation_kwargs, save_path=None, recompute=False):
    try:
        with open(save_path) as fp:
            all_metrics = json.load(fp)
    except Exception:
        all_metrics = {}

    for eval_name, kwargs in tqdm(evaluation_kwargs.items()):
        print(f"[DEBUG] Evaluating: {eval_name}")
        metrics = {}
        if eval_name in all_metrics and not recompute:
            print(f"[DEBUG]   Cached, skipping")
            metrics = all_metrics[eval_name]
        for model in all_models:
            if model.name in metrics and not recompute:
                print(f"[DEBUG]   Model {model.name} cached, skipping")
                continue

            print(f"[DEBUG]   Evaluating model: {model.name}")
            metrics[model.name] = eval_model(model, **kwargs)
            print(f"[DEBUG]   Model {model.name} done")
        all_metrics[eval_name] = metrics

    if save_path is not None:
        print(f"[DEBUG] Saving metrics to {save_path}")
        with open(save_path, "w") as fp:
            json.dump(all_metrics, fp, indent=2)

    return all_metrics


def get_run_metrics(
    run_path, step=-1, cache=True, skip_model_load=False, skip_baselines=False
):
    print(f"[DEBUG] get_run_metrics: run_path={run_path}, step={step}, skip_baselines={skip_baselines}")
    
    if skip_model_load:
        print(f"[DEBUG] Loading config only...")
        _, conf = get_model_from_run(run_path, only_conf=True)
        all_models = []
    else:
        print(f"[DEBUG] Loading model and config...")
        model, conf = get_model_from_run(run_path, step)
        print(f"[DEBUG] Model loaded, moving to CUDA...")
        model = model.cuda().eval()
        print(f"[DEBUG] Model on CUDA, preparing baseline models...")
        all_models = [model]
        if not skip_baselines:
            print(f"[DEBUG] Getting relevant baselines for task: {conf.training.task}")
            all_models += models.get_relevant_baselines(conf.training.task)
            print(f"[DEBUG] Total models: {len(all_models)}")
    
    print(f"[DEBUG] Building evaluation kwargs...")
    evaluation_kwargs = build_evals(conf)
    print(f"[DEBUG] Evaluation kwargs built, total evals: {len(evaluation_kwargs)}")

    if not cache:
        save_path = None
    elif step == -1:
        save_path = os.path.join(run_path, "metrics.json")
    else:
        save_path = os.path.join(run_path, f"metrics_{step}.json")

    recompute = False
    if save_path is not None and os.path.exists(save_path):
        checkpoint_created = os.path.getmtime(run_path)
        cache_created = os.path.getmtime(save_path)
        if checkpoint_created > cache_created:
            recompute = True

    print(f"[DEBUG] Starting compute_evals...")
    all_metrics = compute_evals(all_models, evaluation_kwargs, save_path, recompute)
    print(f"[DEBUG] compute_evals completed")
    return all_metrics



def conf_to_model_name(conf):
    if conf.model.family == "gpt2":
        return {
            (3, 2): "Transformer-xs",
            (6, 4): "Transformer-small",
            (12, 8): "Transformer",
        }[(conf.model.n_layer, conf.model.n_head)]
    else:
        return conf.wandb.name


def baseline_names(name):
    if "OLS" in name:
        return "Least Squares"
    if name == "averaging":
        return "Averaging"
    if "NN" in name:
        k = name.split("_")[1].split("=")[1]
        return f"{k}-Nearest Neighbors"
    if "lasso" in name:
        alpha = name.split("_")[1].split("=")[1]
        return f"Lasso (alpha={alpha})"
    if "gd" in name:
        return "2-layer NN, GD"
    if "decision_tree" in name:
        return "Greedy Tree Learning"
    if "xgboost" in name:
        return "XGBoost"
    return name


def read_run_dir(run_dir):
    all_runs = {}
    for task in os.listdir(run_dir):
        task_dir = os.path.join(run_dir, task)
        for run_id in os.listdir(task_dir):
            run_path = os.path.join(task_dir, run_id)
            _, conf = get_model_from_run(run_path, only_conf=True)
            params = {}
            params["run_id"] = run_id
            params["task"] = task
            params["model"] = conf_to_model_name(conf)
            params["kwargs"] = "_".join(
                f"{k}={v}" for k, v in conf.training.task_kwargs.items()
            )
            num_tasks = (
                conf.training.num_tasks if "num_tasks" in conf.training else None
            )
            params["num_tasks"] = num_tasks if num_tasks is not None else -1
            num_examples = (
                conf.training.num_training_examples
                if "num_training_examples" in conf.training
                else None
            )
            params["num_examples"] = num_examples if num_examples is not None else -1
            params["n_dims"] = conf.model.n_dims
            params["n_layer"] = conf.model.n_layer
            params["n_head"] = conf.model.n_head
            params["run_name"] = conf.wandb.name

            for k, v in params.items():
                if k not in all_runs:
                    all_runs[k] = []
                all_runs[k].append(v)

    df = pd.DataFrame(all_runs).sort_values("run_name")
    assert len(df) == len(df.run_name.unique())
    return df

if __name__ == "__main__":
    run_dir = sys.argv[1]
    for task in os.listdir(run_dir):
        task_dir = os.path.join(run_dir, task)
        print(f"Evaluating task {task}")
        for run_id in tqdm(os.listdir(task_dir)):
            run_path = os.path.join(run_dir, task, run_id)
            metrics = get_run_metrics(run_path)
