import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
import yaml

from samplers import MarkovSampler
from tasks import MarkovNoisyLinearRegression


RESULT_FILE = "markov_benign_overfitting_results.json"
STATE_FILE = "markov_benign_overfitting_state.json"
PLOT_FILE = "markov_benign_overfitting_results.png"


def mse(pred, target):
    return ((pred - target) ** 2).mean().item()


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def _resolve_path(repo_root, maybe_relative_path):
    return os.path.normpath(os.path.abspath(os.path.join(repo_root, maybe_relative_path)))


def _load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    except (OSError, json.JSONDecodeError, ValueError):
        return default


def _save_json_atomic(path, payload):
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)
    os.replace(temp_path, path)


def make_A(d, scale, seed=None):
    generator = None if seed is None else torch.Generator().manual_seed(int(seed))
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=generator))
    return scale * q


def make_stationary_A_seq(A, T):
    return [A for _ in range(T)]


def make_linear_drift_A_seq(d, T, start_scale=0.2, end_scale=0.9, seed=0):
    sequence = []
    for t in range(T):
        scale = start_scale + (end_scale - start_scale) * (t / max(T - 1, 1))
        sequence.append(make_A(d, scale, seed=seed + t))
    return sequence


def make_regime_switch_A_seq(d, T, first_scale=0.2, second_scale=0.9, seed=0):
    first_A = make_A(d, first_scale, seed=seed)
    second_A = make_A(d, second_scale, seed=seed + 1)
    return [first_A if t < T // 2 else second_A for t in range(T)]


def generate_markov_data(T, d, A_seq, sigma_x, sigma_y, w, batch_size=64, device="cpu", seed=0):
    torch.manual_seed(int(seed))
    sampler = MarkovSampler(d, noise_std=sigma_x, initial_std=0.0)
    task = MarkovNoisyLinearRegression(d, batch_size, noise_std=sigma_y, w=w)

    seeds = [seed + i for i in range(batch_size)]
    x0 = torch.zeros(batch_size, d)
    xs = sampler.sample_xs(
        T,
        batch_size,
        A_seq=A_seq,
        sigma_x=sigma_x,
        x0=x0,
        seeds=seeds,
        device=device,
    )
    ys = task.evaluate(xs)
    return xs, ys


def fit_ols(X_train, Y_train):
    return torch.linalg.pinv(X_train) @ Y_train.unsqueeze(-1)


def evaluate_split(xs, ys, train_T):
    X_train = xs[:, :train_T, :]
    Y_train = ys[:, :train_T]
    X_test = xs[:, train_T:, :]
    Y_test = ys[:, train_T:]

    w_hat = fit_ols(X_train, Y_train)
    train_pred = (X_train @ w_hat).squeeze(-1)
    test_pred = (X_test @ w_hat).squeeze(-1)

    return {
        "train_err": mse(train_pred, Y_train),
        "test_err": mse(test_pred, Y_test),
    }


def _item_done(existing_items, key, value):
    return any(item.get(key) == value for item in existing_items)


def _compute_exp1(cfg, fixed_w, existing_items):
    items = list(existing_items)
    for index, scale in enumerate(cfg["stationary_scales"]):
        if _item_done(items, "scale", float(scale)):
            continue
        A = make_A(cfg["d"], scale, seed=cfg["seed"])
        A_seq = make_stationary_A_seq(A, cfg["T"])
        xs, ys = generate_markov_data(
            cfg["T"],
            cfg["d"],
            A_seq,
            sigma_x=cfg["sigma_x"],
            sigma_y=cfg["sigma_y"],
            w=fixed_w,
            batch_size=cfg["num_trials"],
            device=cfg["device"],
            seed=cfg["seed"] + index,
        )
        metrics = evaluate_split(xs, ys, cfg["train_T"])
        items.append({"scale": float(scale), **metrics})
    return items


def _compute_exp2(cfg, fixed_w, existing_items):
    all_cases = [
        ("stationary", make_stationary_A_seq(make_A(cfg["d"], 0.6, seed=cfg["seed"]), cfg["T"])),
        ("drift", make_linear_drift_A_seq(cfg["d"], cfg["T"], 0.2, 0.9, seed=cfg["seed"])),
        ("switch", make_regime_switch_A_seq(cfg["d"], cfg["T"], 0.2, 0.9, seed=cfg["seed"])),
    ]
    items = list(existing_items)
    for index, (label, A_seq) in enumerate(all_cases):
        if _item_done(items, "label", label):
            continue
        xs, ys = generate_markov_data(
            cfg["T"],
            cfg["d"],
            A_seq,
            sigma_x=cfg["sigma_x"],
            sigma_y=cfg["sigma_y"],
            w=fixed_w,
            batch_size=cfg["num_trials"],
            device=cfg["device"],
            seed=cfg["seed"] + 100 + index,
        )
        metrics = evaluate_split(xs, ys, cfg["train_T"])
        items.append({"label": label, **metrics})
    return items


def _compute_exp3(cfg, fixed_w, existing_items):
    items = list(existing_items)
    A_seq = make_stationary_A_seq(make_A(cfg["d"], 0.6, seed=cfg["seed"]), cfg["T"])
    for index, sigma_x in enumerate(cfg["sigma_x_values"]):
        if _item_done(items, "sigma_x", float(sigma_x)):
            continue
        xs, ys = generate_markov_data(
            cfg["T"],
            cfg["d"],
            A_seq,
            sigma_x=sigma_x,
            sigma_y=cfg["sigma_y"],
            w=fixed_w,
            batch_size=cfg["num_trials"],
            device=cfg["device"],
            seed=cfg["seed"] + 200 + index,
        )
        metrics = evaluate_split(xs, ys, cfg["train_T"])
        items.append({"sigma_x": float(sigma_x), **metrics})
    return items


def _compute_exp4(cfg, fixed_w, existing_items):
    items = list(existing_items)
    A_seq = make_stationary_A_seq(make_A(cfg["d"], 0.6, seed=cfg["seed"]), cfg["T"])
    for index, sigma_y in enumerate(cfg["sigma_y_values"]):
        if _item_done(items, "sigma_y", float(sigma_y)):
            continue
        xs, ys = generate_markov_data(
            cfg["T"],
            cfg["d"],
            A_seq,
            sigma_x=cfg["sigma_x"],
            sigma_y=sigma_y,
            w=fixed_w,
            batch_size=cfg["num_trials"],
            device=cfg["device"],
            seed=cfg["seed"] + 300 + index,
        )
        metrics = evaluate_split(xs, ys, cfg["train_T"])
        items.append({"sigma_y": float(sigma_y), **metrics})
    return items


def plot_results(all_results, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    def plot_numeric(ax, items, x_key, title):
        x = [row[x_key] for row in items]
        train_err = [row["train_err"] for row in items]
        test_err = [row["test_err"] for row in items]
        ax.plot(x, train_err, marker="o", label="train")
        ax.plot(x, test_err, marker="o", label="test")
        ax.set_title(title)
        ax.set_xlabel(x_key)
        ax.set_ylabel("mse")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plot_numeric(axes[0], all_results["exp1"], "scale", "EXP 1: stationary Markov")
    plot_numeric(axes[2], all_results["exp3"], "sigma_x", "EXP 3: input noise sweep")
    plot_numeric(axes[3], all_results["exp4"], "sigma_y", "EXP 4: label noise sweep")

    exp2 = all_results["exp2"]
    labels = [row["label"] for row in exp2]
    train_err = [row["train_err"] for row in exp2]
    test_err = [row["test_err"] for row in exp2]
    x = list(range(len(labels)))
    axes[1].plot(x, train_err, marker="o", label="train")
    axes[1].plot(x, test_err, marker="o", label="test")
    axes[1].set_title("EXP 2: nonstationary A_t")
    axes[1].set_xlabel("setting")
    axes[1].set_ylabel("mse")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _flatten_markov_split(xs, ys, train_T):
    train_x = xs[:, :train_T, :].reshape(-1, xs.shape[-1])
    train_y = ys[:, :train_T].reshape(-1)
    test_x = xs[:, train_T:, :].reshape(-1, xs.shape[-1])
    test_y = ys[:, train_T:].reshape(-1)
    return train_x, train_y, test_x, test_y


def _load_dynamics_state(state_path):
    state = _load_json(state_path, default={})
    return {
        "step": int(state.get("step", 0)),
        "train_losses": list(state.get("train_losses", [])),
        "test_losses": list(state.get("test_losses", [])),
        "config_path": state.get("config_path"),
        "model_w": state.get("model_w"),
    }


def _save_dynamics_state(state_path, state):
    payload = {
        "step": int(state["step"]),
        "train_losses": list(state["train_losses"]),
        "test_losses": list(state["test_losses"]),
        "config_path": state.get("config_path"),
        "model_w": state["model_w"],
    }
    _save_json_atomic(state_path, payload)


def _plot_dynamics(train_steps, train_losses, test_losses, out_path, title):
    if not train_steps:
        return
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(train_steps, train_losses, linewidth=2.2, label="train_loss", color="dimgray")
    ax.plot(train_steps, test_losses, linewidth=2.6, label="test_loss", color="black")
    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run_dynamics(cfg, output_dir, state_path, results_path, plot_path, resume=False):
    device = cfg["device"]
    train_steps = int(cfg.get("train_steps", 2000))
    log_every = int(cfg.get("log_every_steps", 50))
    lr = float(cfg.get("learning_rate", 1e-3))
    seed = int(cfg["seed"])

    A = make_A(cfg["d"], cfg.get("markov_scale", 0.6), seed=seed)
    A_seq = make_stationary_A_seq(A, cfg["T"])
    w_true = torch.randn(int(cfg["d"]), device=device)

    xs, ys = generate_markov_data(
        cfg["T"],
        cfg["d"],
        A_seq,
        sigma_x=cfg["sigma_x"],
        sigma_y=cfg["sigma_y"],
        w=w_true,
        batch_size=cfg["num_trials"],
        device=device,
        seed=seed,
    )
    train_x, train_y, test_x, test_y = _flatten_markov_split(xs, ys, cfg["train_T"])
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    state = _load_dynamics_state(state_path) if resume else {"step": 0, "train_losses": [], "test_losses": [], "model_w": None}

    if state.get("model_w") is not None:
        w = torch.tensor(state["model_w"], dtype=torch.float32, device=device, requires_grad=True)
    else:
        w = torch.zeros(int(cfg["d"]), device=device, requires_grad=True)

    optimizer = torch.optim.SGD([w], lr=lr)

    start_step = int(state.get("step", 0))
    train_losses = list(state.get("train_losses", []))
    test_losses = list(state.get("test_losses", []))
    recorded_steps = [log_every * idx for idx in range(len(train_losses))]

    for step in range(start_step, train_steps):
        optimizer.zero_grad()
        pred = train_x @ w
        loss = ((pred - train_y) ** 2).mean()
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == train_steps - 1:
            with torch.no_grad():
                train_loss = ((train_x @ w - train_y) ** 2).mean().item()
                test_loss = ((test_x @ w - test_y) ** 2).mean().item()
            train_losses.append(float(train_loss))
            test_losses.append(float(test_loss))
            recorded_steps.append(step)
            state = {
                "step": step + 1,
                "train_losses": train_losses,
                "test_losses": test_losses,
                "model_w": w.detach().cpu().tolist(),
                "config_path": cfg.get("config_path"),
            }
            _save_dynamics_state(state_path, state)
            _plot_dynamics(recorded_steps, train_losses, test_losses, plot_path, "Markov benign overfitting dynamics")

    summary = {
        "mode": "dynamics",
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_test_loss": test_losses[-1] if test_losses else None,
        "recorded_steps": recorded_steps,
        "train_losses": train_losses,
        "test_losses": test_losses,
    }
    _save_json_atomic(results_path, summary)
    return summary


def _default_config_path(repo_root):
    return os.path.join(repo_root, "src", "conf", "markov_benign_overfitting_small.yaml")


def main():
    parser = argparse.ArgumentParser(description="Markov benign overfitting experiments")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot_each_step", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = args.config or _default_config_path(repo_root)
    cfg = _load_yaml(config_path)

    output_dir = _resolve_path(repo_root, cfg["out_dir"])
    os.makedirs(output_dir, exist_ok=True)

    cfg["output_dir"] = output_dir
    if args.device is not None:
        cfg["device"] = args.device
    if cfg.get("device", "auto") == "auto":
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["config_path"] = config_path

    state_path = os.path.join(output_dir, STATE_FILE)
    results_path = os.path.join(output_dir, RESULT_FILE)
    plot_path = os.path.join(output_dir, PLOT_FILE)

    mode = cfg.get("mode", "sweep")
    if mode == "dynamics":
        summary = run_dynamics(cfg, output_dir, state_path, results_path, plot_path, resume=args.resume)
        print(f"Saved dynamics results to {results_path}")
        print(f"Saved dynamics state to {state_path}")
        print(f"Saved dynamics plot to {plot_path}")
        print(summary)
        return

    state = _load_json(state_path, default={}) if args.resume else {}
    results = state.get("results", _load_json(results_path, default={}))
    results = {
        "exp1": results.get("exp1", []),
        "exp2": results.get("exp2", []),
        "exp3": results.get("exp3", []),
        "exp4": results.get("exp4", []),
    }

    torch.manual_seed(int(cfg["seed"]))
    fixed_w = torch.randn(int(cfg["d"]))

    results["exp1"] = _compute_exp1(cfg, fixed_w, results["exp1"])
    _save_json_atomic(state_path, {"results": results, "config_path": config_path})
    if args.plot_each_step:
        plot_results(results, plot_path)

    results["exp2"] = _compute_exp2(cfg, fixed_w, results["exp2"])
    _save_json_atomic(state_path, {"results": results, "config_path": config_path})
    if args.plot_each_step:
        plot_results(results, plot_path)

    results["exp3"] = _compute_exp3(cfg, fixed_w, results["exp3"])
    _save_json_atomic(state_path, {"results": results, "config_path": config_path})
    if args.plot_each_step:
        plot_results(results, plot_path)

    results["exp4"] = _compute_exp4(cfg, fixed_w, results["exp4"])
    _save_json_atomic(state_path, {"results": results, "config_path": config_path})

    _save_json_atomic(results_path, results)
    plot_results(results, plot_path)

    print(f"Saved results to {results_path}")
    print(f"Saved state to {state_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()