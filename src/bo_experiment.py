"""Evaluate fixed-context benign-overfitting diagnostics; does not train models.

Run from the repository root::

    python src/bo_experiment.py --config src/conf/bo_sweep.yaml --output results/bo_smoke.json
    python src/bo_experiment.py --config my_sweep.yaml --run-dir models/my_run --output results/tf.json

SNR is amplitude SNR: sigma = 1 / SNR. Each evaluation seed represents a
fresh group of tasks, never a model-training seed. Common random numbers are
used across feature rho and SNR for a given (seed, dimension, context length).
"""

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from bo_eval import LinearEstimator, evaluate_context
from dependence import temporal_correlation, temporal_effective_rank
from samplers import get_data_sampler
from tasks import get_task_sampler


DEFAULTS = {
    "n_dims": [8], "context_lengths": [4, 8, 16],
    "rhos": [0.0, 0.6, 0.9], "snrs": [0.5, 2.0, 8.0],
    "eval_seeds": [1001, 1002, 1003], "n_eval": 32, "batch_size": 16,
    "n_queries": 64, "n_probe_queries": None, "n_probe_test_queries": 64,
    "query_batch_size": 256, "fit_threshold": 1e-4, "gen_threshold": 0.1,
    "linearity_threshold": 0.99, "feature_rho_after": None,
    "feature_change_point": None, "noise_rho": 0.0, "noise_rho_after": None,
    "noise_change_point": None, "baselines": ["ols", "ridge", "gls"],
    "linear_probe": True,
}


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle)
    if not isinstance(supplied, dict):
        raise ValueError("Sweep config must be a YAML mapping")
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown sweep settings: {sorted(unknown)}")
    config = {**DEFAULTS, **supplied}
    for name in ("n_dims", "context_lengths", "rhos", "snrs", "eval_seeds"):
        if not isinstance(config[name], list) or not config[name]:
            raise ValueError(f"{name} must be a nonempty list")
        if len(set(config[name])) != len(config[name]):
            raise ValueError(f"{name} must not contain duplicates")
    for name in ("n_dims", "context_lengths"):
        for value in config[name]:
            _positive_int(value, name)
    for name in ("n_eval", "batch_size", "n_queries", "n_probe_test_queries", "query_batch_size"):
        _positive_int(config[name], name)
    if config["n_probe_queries"] is not None:
        _positive_int(config["n_probe_queries"], "n_probe_queries")
        if config["n_probe_queries"] < max(config["n_dims"]):
            raise ValueError("n_probe_queries must be at least the largest dimension")
    if not isinstance(config["linear_probe"], bool):
        raise ValueError("linear_probe must be boolean")
    for seed in config["eval_seeds"]:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("eval_seeds must be nonnegative integers")
    config["snrs"] = [float(value) for value in config["snrs"]]
    if any(math.isnan(value) or value <= 0 for value in config["snrs"]):
        raise ValueError("snrs must be positive (use 'inf' for noiseless)")
    if len(set(config["snrs"])) != len(config["snrs"]):
        raise ValueError("snrs contains duplicate numeric values")
    for name in ("fit_threshold", "gen_threshold"):
        if not math.isfinite(config[name]) or config[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 0 <= config["linearity_threshold"] <= 1:
        raise ValueError("linearity_threshold must lie in [0, 1]")
    # Validate exact transition conventions for every requested context length.
    for k, rho in itertools.product(config["context_lengths"], config["rhos"]):
        temporal_correlation(k, rho=rho, rho_after=config["feature_rho_after"],
                             change_point=config["feature_change_point"])
        temporal_correlation(k, rho=config["noise_rho"],
                             rho_after=config["noise_rho_after"],
                             change_point=config["noise_change_point"])
    return config


def _seed(seed, d, k, batch_index, stream):
    payload = f"bo-evaluation-v1/{seed}/{d}/{k}/{batch_index}/{stream}"
    value = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")
    torch.manual_seed(value % (2**63 - 1))


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value == math.inf else None
    return value


def _checkpoint(path, config, device):
    # Lazy import keeps classical experiments independent of transformers/sklearn.
    from models import build_model

    path = Path(path).resolve()
    with (path / "config.yaml").open(encoding="utf-8") as handle:
        training_config = yaml.safe_load(handle)
    architecture = training_config["model"]
    if any(d != architecture["n_dims"] for d in config["n_dims"]):
        raise ValueError(f"{path}: checkpoint n_dims={architecture['n_dims']}; use a matching sweep")
    if max(config["context_lengths"]) + 1 > architecture["n_positions"]:
        raise ValueError(f"{path}: k+1 must be <= n_positions={architecture['n_positions']} (query needs one position)")
    model = build_model(SimpleNamespace(**architecture))
    # ``weights_only`` was added after the PyTorch 1.11 stack used by the
    # original pinned PyTorch 1.11 stack, so keep this call for trusted checkpoints.
    state = torch.load(path / "state.pt", map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    metadata = {
        "model": model.name, "checkpoint_id": str(path),
        "training_seed": training_config.get("training", {}).get("seed"),
        "architecture": architecture, "training_config": training_config.get("training", {}),
        "train_step": state.get("train_step"),
    }
    return model, metadata


def _estimators(config):
    if not isinstance(config["baselines"], list):
        raise ValueError("baselines must be a list")
    models = []
    names = set()
    for specification in config["baselines"]:
        arguments = {"kind": specification} if isinstance(specification, str) else dict(specification)
        unknown = set(arguments) - {"kind", "ridge_alpha", "name"}
        if unknown:
            raise ValueError(f"Unknown baseline settings: {sorted(unknown)}")
        model = LinearEstimator(**arguments)
        if model.name in names:
            raise ValueError(f"Duplicate baseline name: {model.name}")
        names.add(model.name)
        models.append((model, {"model": model.name, "checkpoint_id": None,
                               "training_seed": None, "estimator": arguments}))
    return models


def summarize_metrics(batches):
    """Keep task variability distinct from variation across evaluation seeds."""
    result = {}
    for key in batches[0]:
        values = torch.cat([batch[key].detach().cpu().reshape(-1).double() for batch in batches])
        finite = values[torch.isfinite(values)]
        result[key] = {
            "mean": finite.mean().item() if finite.numel() else None,
            "std": finite.std(unbiased=True).item() if finite.numel() > 1 else None,
            "n_finite": finite.numel(), "n_total": values.numel(),
        }
    return result


def run_sweep(config, run_dirs=(), device="cpu"):
    models = _estimators(config)
    models.extend(_checkpoint(path, config, device) for path in run_dirs)
    if not models:
        raise ValueError("At least one baseline or --run-dir is required")
    identifiers = [(meta["model"], meta["checkpoint_id"]) for _, meta in models]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate model/checkpoint entries")
    protocol_keys = ("feature_rho_after", "feature_change_point", "noise_rho",
                     "noise_rho_after", "noise_change_point", "fit_threshold",
                     "gen_threshold", "linearity_threshold", "n_queries",
                     "n_probe_queries", "n_probe_test_queries", "linear_probe")
    protocol = {key: config[key] for key in protocol_keys}
    protocol["query_distribution"] = "independent N(0,I_d), independent of context"
    protocol["snr_convention"] = "amplitude; sigma=1/snr; E[signal variance]=1"
    protocol_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
    records = []
    conditions = list(itertools.product(config["n_dims"], config["context_lengths"],
                                         config["rhos"], config["snrs"], config["eval_seeds"]))
    for index, (d, k, rho, snr, seed) in enumerate(conditions, 1):
        data_sampler = get_data_sampler("gaussian_ar1", d, rho=rho,
                                       rho_after=config["feature_rho_after"],
                                       change_point=config["feature_change_point"])
        covariance = temporal_correlation(k, rho=config["noise_rho"],
                                          rho_after=config["noise_rho_after"],
                                          change_point=config["noise_change_point"])
        measurements = [[] for _ in models]
        for batch_index, start in enumerate(range(0, config["n_eval"], config["batch_size"])):
            count = min(config["batch_size"], config["n_eval"] - start)
            _seed(seed, d, k, batch_index, "weights")
            task = get_task_sampler("dependent_linear_regression", d, count, snr=snr,
                                    noise_rho=config["noise_rho"],
                                    noise_rho_after=config["noise_rho_after"],
                                    noise_change_point=config["noise_change_point"])()
            _seed(seed, d, k, batch_index, "features")
            xs = data_sampler.sample_xs(k, count)
            _seed(seed, d, k, batch_index, "label-noise")
            ys = task.evaluate(xs)  # Draw once: identical observed labels for every estimator/query.
            _seed(seed, d, k, batch_index, "queries")
            queries = torch.randn(count, config["n_queries"], d)
            probes = probe_test = None
            if config["linear_probe"]:
                _seed(seed, d, k, batch_index, "probe-fit")
                probes = torch.randn(count, config["n_probe_queries"] or 2 * d, d)
                _seed(seed, d, k, batch_index, "probe-test")
                probe_test = torch.randn(count, config["n_probe_test_queries"], d)
            for model_index, (model, _) in enumerate(models):
                with torch.no_grad():
                    measurements[model_index].append(evaluate_context(
                        model, xs, ys, task.w_b, queries,
                        probe_queries=probes, probe_test_queries=probe_test,
                        fit_threshold=config["fit_threshold"],
                        gen_threshold=config["gen_threshold"],
                        linearity_threshold=config["linearity_threshold"],
                        query_batch_size=config["query_batch_size"],
                        noise_covariance=covariance,
                        run_linear_probe=config["linear_probe"],
                    ))
        for (_, metadata), batches in zip(models, measurements):
            records.append({
                "protocol_id": protocol_id, "model": metadata["model"],
                "checkpoint_id": metadata["checkpoint_id"], "training_seed": metadata["training_seed"],
                "seed": seed, "d": d, "k": k, "k_over_d": k / d,
                "rho": rho, "snr": snr, "noise_std": 1 / snr, "n_eval": config["n_eval"],
                "effective_rank": float(temporal_effective_rank(
                    k, rho=rho, rho_after=config["feature_rho_after"],
                    change_point=config["feature_change_point"])),
                "metrics": summarize_metrics(batches),
            })
            records[-1]["effective_rank_over_d"] = records[-1]["effective_rank"] / d
            records[-1]["effective_information"] = records[-1]["effective_rank"] * snr**2
        print(f"[{index}/{len(conditions)}] d={d} k={k} rho={rho} snr={snr} eval_seed={seed}", flush=True)
    return _json_safe({
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "config": config, "protocol": protocol, "protocol_id": protocol_id,
            "models": [metadata for _, metadata in models],
            "torch_version": str(torch.__version__),
            "seed_protocol": "SHA256 bo-evaluation-v1 streams; rho/SNR excluded for matched draws; k/d included",
            "uncertainty": "metric std is across tasks; plots compute approximate 95% CIs across eval seed means per checkpoint",
            "nonfinite_encoding": "noiseless SNR='inf'; unavailable metric summaries=null; n_finite records exclusions",
            "bo_interpretation": (
                "bo_candidate is thresholded direct duplicate-fit plus clean-query behavior; "
                "linear_bo_candidate additionally requires an identifiable held-out linear probe; "
                "neither is a proof of asymptotic benign overfitting"
            ),
        }, "records": records,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-dir", action="append", default=[], help="Repeat to compare checkpoints; each remains a separate series")
    parser.add_argument("--device", default="cpu", help="Transformer device, e.g. cpu or cuda; baselines use CPU")
    args = parser.parse_args()
    result = run_sweep(load_config(args.config), args.run_dir, args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote {len(result['records'])} records to {output}")


if __name__ == "__main__":
    main()
