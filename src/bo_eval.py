"""Fixed-context diagnostics for benign-overfitting experiments.

All tensors are batched by task: context ``xs`` is (B, K, d), labels are
(B, K), and query sets are (B, Q, d). ``evaluate_context`` returns per-task
scalars. Thresholds use raw squared-output units, not normalized errors.

For a Transformer, re-querying a context point measures direct interpolation
but may be implemented by retrieval rather than one linear predictor. We
therefore report both the direct behavioral BO flag and a stricter implied
linear-predictor flag, fitted on independent Gaussian probes and assessed on a
second held-out probe set. These are empirical candidates, not a theorem or an
asymptotic claim. Exact isotropic population risk is reported only for actual
linear estimators; probe risk describes the linear surrogate.
"""

import math

import torch


class LinearEstimator:
    """Batched minimum-norm OLS, ridge, or oracle GLS, without an intercept.

    ``fit(xs, ys, noise_covariance=None)`` returns self and stores weights in
    ``weights_`` with shape (B, d). Ridge minimizes
    ``||Xw-y||^2 + ridge_alpha * ||w||^2`` (no division by K).
    GLS whitens using the supplied *context noise* covariance; it does not
    whiten the feature-process covariance. Rectangular and rank-deficient
    designs use a pseudoinverse, including the underdetermined case.
    """

    def __init__(self, kind="ols", ridge_alpha=1.0, name=None):
        if kind not in {"ols", "ridge", "gls"}:
            raise ValueError("kind must be 'ols', 'ridge', or 'gls'")
        if not math.isfinite(ridge_alpha) or ridge_alpha < 0:
            raise ValueError("ridge_alpha must be finite and nonnegative")
        self.kind = kind
        self.ridge_alpha = float(ridge_alpha)
        self.name = name or (f"ridge_alpha={ridge_alpha:g}" if kind == "ridge" else kind)
        self.weights_ = None

    def fit(self, xs, ys, noise_covariance=None):
        _validate_context(xs, ys)
        design, targets = xs, ys.unsqueeze(-1)
        if self.kind == "gls":
            if noise_covariance is None:
                raise ValueError("oracle GLS requires the context noise covariance")
            covariance = torch.as_tensor(noise_covariance, dtype=xs.dtype, device=xs.device)
            if covariance.shape not in {(xs.shape[1], xs.shape[1]),
                                        (xs.shape[0], xs.shape[1], xs.shape[1])}:
                raise ValueError("noise_covariance must have shape (K,K) or (B,K,K)")
            if not torch.isfinite(covariance).all() or not torch.allclose(
                covariance, covariance.transpose(-1, -2)
            ):
                raise ValueError("noise_covariance must be finite and symmetric")
            factor, info = torch.linalg.cholesky_ex(covariance)
            if (info != 0).any():
                raise ValueError("noise_covariance must be positive definite")
            design = torch.linalg.solve_triangular(factor, design, upper=False)
            targets = torch.linalg.solve_triangular(factor, targets, upper=False)
        if self.kind == "ridge" and self.ridge_alpha > 0:
            # SVD avoids a poorly conditioned normal-equation solve and handles
            # K < d without constructing a d-by-d matrix.
            left, singular, right = torch.linalg.svd(design, full_matrices=False)
            shrinkage = singular / (singular.square() + self.ridge_alpha)
            projected = left.transpose(-1, -2) @ targets
            weights = right.transpose(-1, -2) @ (shrinkage.unsqueeze(-1) * projected)
        else:
            weights = torch.linalg.pinv(design) @ targets
        self.weights_ = weights.squeeze(-1)
        return self

    def predict(self, queries):
        if self.weights_ is None:
            raise RuntimeError("call fit before predict")
        _validate_queries(queries, self.weights_.shape[0], self.weights_.shape[1], "queries")
        return (queries @ self.weights_.unsqueeze(-1)).squeeze(-1)


def _validate_context(xs, ys):
    if xs.ndim != 3 or min(xs.shape) < 1:
        raise ValueError("xs must have nonempty shape (B,K,d)")
    if ys.shape != xs.shape[:2]:
        raise ValueError("ys_noisy must have shape (B,K)")
    if not xs.is_floating_point() or not ys.is_floating_point():
        raise ValueError("features and labels must be floating point tensors")
    if xs.device != ys.device or xs.dtype != ys.dtype:
        raise ValueError("features and labels must have the same device and dtype")
    if not torch.isfinite(xs).all() or not torch.isfinite(ys).all():
        raise ValueError("features and labels must be finite")


def _validate_queries(queries, batch_size, n_dims, name):
    if queries.ndim != 3 or queries.shape[0] != batch_size or queries.shape[2] != n_dims or queries.shape[1] < 1:
        raise ValueError(f"{name} must have nonempty shape (B,Q,d)")
    if not queries.is_floating_point() or not torch.isfinite(queries).all():
        raise ValueError(f"{name} must be finite floating point values")


def _model_device_dtype(model, xs):
    if hasattr(model, "parameters"):
        parameter = next(iter(model.parameters()), None)
        if parameter is not None:
            return parameter.device, parameter.dtype
    return xs.device, xs.dtype


def _fixed_context_predictions(model, xs, ys, queries, query_batch_size):
    """Append exactly one query to each context replica; use a zero label.

    ``query_batch_size`` bounds the number of replicas in a forward call,
    rather than the number of queries per task. Queries never become context
    for any other query, even across chunk boundaries.
    """
    b_size, k, n_dims = xs.shape
    n_queries = queries.shape[1]
    device, dtype = _model_device_dtype(model, xs)
    context = xs.to(device=device, dtype=dtype)
    labels = ys.to(device=device, dtype=dtype)
    flat_queries = queries.reshape(-1, n_dims).to(device=device, dtype=dtype)
    predictions = []
    for start in range(0, b_size * n_queries, query_batch_size):
        end = min(start + query_batch_size, b_size * n_queries)
        task_ids = torch.arange(start, end, device=device) // n_queries
        context_and_query = torch.cat((context[task_ids], flat_queries[start:end, None]), dim=1)
        context_and_dummy = torch.cat((labels[task_ids], labels.new_zeros((end - start, 1))), dim=1)
        pred = model(context_and_query, context_and_dummy, inds=[k])
        if pred.shape not in {(end - start,), (end - start, 1)}:
            raise ValueError("model must return one prediction per context replica for inds=[K]")
        predictions.append(pred.reshape(-1).to(device=xs.device, dtype=xs.dtype))
    return torch.cat(predictions).reshape(b_size, n_queries)


def _mean_squared(predictions, targets):
    return (predictions - targets).square().mean(dim=1)


def _r2(predictions, targets):
    residual = (predictions - targets).square().sum(dim=1)
    total = (targets - targets.mean(dim=1, keepdim=True)).square().sum(dim=1)
    # A constant-zero predictor is linear. Handle constant targets without
    # declaring a mismatched constant predictor a perfect linear surrogate.
    # Do not use an absolute epsilon here: that would make sufficiently small
    # nonlinear functions appear perfectly linear just by changing units.
    constant_score = torch.where(residual == 0, torch.ones_like(total), torch.zeros_like(total))
    return torch.where(total > 0, 1 - residual / total.clamp_min(torch.finfo(total.dtype).tiny), constant_score)


def _flags(fits, generalizes, identifiable):
    return {
        "bo_candidate": identifiable & fits & generalizes,
        "harmful_overfitting": identifiable & fits & ~generalizes,
        "underfitting": identifiable & ~fits,
        "indeterminate": ~identifiable,
    }


@torch.no_grad()
def evaluate_context(
    model,
    xs,
    ys_noisy,
    w_true,
    queries,
    probe_queries=None,
    probe_test_queries=None,
    fit_threshold=1e-4,
    gen_threshold=0.1,
    linearity_threshold=0.99,
    query_batch_size=256,
    noise_covariance=None,
    run_linear_probe=True,
):
    """Evaluate one fixed noisy context per task without revealing test labels.

    ``w_true`` is the effective label-generating weight, shaped (B,d) or
    (B,d,1). Main ``queries`` must be fresh clean queries; isotropic Gaussian
    queries are the intended population. Explicit probe sets must be fresh,
    mutually independent, and independent of both the context and main test
    queries. A fitting set needs at least d points and a holdout needs at
    least two points. Defaults independently sample max(2*d,32) iid probes
    and holdouts, using PyTorch's current RNG. Rank-deficient probes result
    in an indeterminate classification.

    Common metrics are ``context_fit_mse`` and ``clean_query_mse``. For a
    Transformer, the former is the *surrogate* fit. Its additional metrics
    are ``duplicate_context_fit_mse``, ``probe_fit_mse``, ``probe_clean_mse``,
    ``probe_r2`` (held out), ``implied_parameter_error`` (squared L2),
    ``probe_isotropic_clean_risk``, ``probe_identifiable``, and
    ``linearity_pass``. ``bo_candidate`` is behavioral: direct duplicate fit
    and direct clean-query generalization. ``linear_bo_candidate`` additionally
    requires the implied linear estimator to fit/generalize and pass held-out
    linearity. This distinguishes retrieval-plus-generalization from a single
    linear interpolator without discarding either mechanism.

    Classical estimators instead report ``parameter_error`` and
    ``exact_isotropic_clean_risk`` (both squared L2 weight error). Their flags
    use actual training fit and exact isotropic clean risk. A noisy-label
    risk would additionally contain irreducible test noise and is deliberately
    not used to diagnose benign overfitting. Model train/eval mode is restored
    even when prediction fails. Set ``run_linear_probe=False`` for a cheaper
    coarse Transformer scan; direct behavioral metrics remain available.
    Output tensors are on ``xs.device``.
    """
    _validate_context(xs, ys_noisy)
    b_size, _, n_dims = xs.shape
    _validate_queries(queries, b_size, n_dims, "queries")
    if w_true.shape == (b_size, n_dims, 1):
        w_true = w_true.squeeze(-1)
    if w_true.shape != (b_size, n_dims) or not torch.isfinite(w_true).all():
        raise ValueError("w_true must be finite with shape (B,d) or (B,d,1)")
    if not isinstance(query_batch_size, int) or query_batch_size < 1:
        raise ValueError("query_batch_size must be a positive integer")
    if not isinstance(run_linear_probe, bool):
        raise ValueError("run_linear_probe must be boolean")
    if not all(math.isfinite(v) and v >= 0 for v in (fit_threshold, gen_threshold)):
        raise ValueError("fit and generalization thresholds must be finite and nonnegative")
    if not math.isfinite(linearity_threshold) or not 0 <= linearity_threshold <= 1:
        raise ValueError("linearity_threshold must lie in [0,1]")
    w_true = w_true.to(device=xs.device, dtype=xs.dtype)
    queries = queries.to(device=xs.device, dtype=xs.dtype)
    clean_targets = (queries @ w_true.unsqueeze(-1)).squeeze(-1)

    if isinstance(model, LinearEstimator):
        model.fit(xs, ys_noisy, noise_covariance=noise_covariance)
        fit_mse = _mean_squared(model.predict(xs), ys_noisy)
        clean_mse = _mean_squared(model.predict(queries), clean_targets)
        parameter_error = (model.weights_ - w_true).square().sum(dim=1)
        finite = torch.isfinite(fit_mse) & torch.isfinite(parameter_error) & torch.isfinite(clean_mse)
        return {
            "context_fit_mse": fit_mse,
            "clean_query_mse": clean_mse,
            "parameter_error": parameter_error,
            "exact_isotropic_clean_risk": parameter_error,
            "linear_bo_candidate": finite & (fit_mse <= fit_threshold) & (parameter_error <= gen_threshold),
            **_flags(fit_mse <= fit_threshold, parameter_error <= gen_threshold, finite),
        }

    if run_linear_probe:
        if probe_queries is None:
            probe_queries = torch.randn((b_size, max(2 * n_dims, 32), n_dims), dtype=xs.dtype, device=xs.device)
        if probe_test_queries is None:
            probe_test_queries = torch.randn((b_size, max(2 * n_dims, 32), n_dims), dtype=xs.dtype, device=xs.device)
        for name, values in (("probe_queries", probe_queries), ("probe_test_queries", probe_test_queries)):
            _validate_queries(values, b_size, n_dims, name)
        if probe_queries.shape[1] < n_dims or probe_test_queries.shape[1] < 2:
            raise ValueError("need at least d fitting probes and two held-out probes")
        probe_queries = probe_queries.to(device=xs.device, dtype=xs.dtype)
        probe_test_queries = probe_test_queries.to(device=xs.device, dtype=xs.dtype)
        if probe_queries.shape == probe_test_queries.shape and torch.equal(probe_queries, probe_test_queries):
            raise ValueError("fitting and held-out probes must be separate independent samples")

    was_training = getattr(model, "training", None)
    try:
        if hasattr(model, "eval"):
            model.eval()
        duplicated = _fixed_context_predictions(model, xs, ys_noisy, xs, query_batch_size)
        direct = _fixed_context_predictions(model, xs, ys_noisy, queries, query_batch_size)
        if run_linear_probe:
            probe_predictions = _fixed_context_predictions(model, xs, ys_noisy, probe_queries, query_batch_size)
            holdout_predictions = _fixed_context_predictions(model, xs, ys_noisy, probe_test_queries, query_batch_size)
    finally:
        if was_training is not None and hasattr(model, "train"):
            model.train(was_training)

    duplicate_fit = _mean_squared(duplicated, ys_noisy)
    direct_clean = _mean_squared(direct, clean_targets)
    direct_finite = torch.isfinite(torch.stack((duplicate_fit, direct_clean))).all(dim=0)
    direct_flags = _flags(
        duplicate_fit <= fit_threshold,
        direct_clean <= gen_threshold,
        direct_finite,
    )
    direct_result = {
        "clean_query_mse": direct_clean,
        "duplicate_context_fit_mse": duplicate_fit,
        **direct_flags,
    }
    if not run_linear_probe:
        return direct_result

    implied_weights = (torch.linalg.pinv(probe_queries) @ probe_predictions.unsqueeze(-1)).squeeze(-1)
    probe_fit = _mean_squared((xs @ implied_weights.unsqueeze(-1)).squeeze(-1), ys_noisy)
    probe_clean = _mean_squared((queries @ implied_weights.unsqueeze(-1)).squeeze(-1), clean_targets)
    probe_r2 = _r2((probe_test_queries @ implied_weights.unsqueeze(-1)).squeeze(-1), holdout_predictions)
    parameter_error = (implied_weights - w_true).square().sum(dim=1)
    identifiable = torch.linalg.matrix_rank(probe_queries) == n_dims
    probe_finite = torch.isfinite(
        torch.stack((probe_fit, probe_clean, probe_r2, parameter_error))
    ).all(dim=0)
    linearity_pass = probe_finite & identifiable & (probe_r2 >= linearity_threshold)
    linear_flags = _flags(
        (duplicate_fit <= fit_threshold) & (probe_fit <= fit_threshold),
        (direct_clean <= gen_threshold) & (probe_clean <= gen_threshold),
        linearity_pass,
    )
    return {
        "context_fit_mse": probe_fit,
        **direct_result,
        "probe_fit_mse": probe_fit,
        "probe_clean_mse": probe_clean,
        "probe_r2": probe_r2,
        "implied_parameter_error": parameter_error,
        "probe_isotropic_clean_risk": parameter_error,
        "probe_identifiable": identifiable,
        "linearity_pass": linearity_pass,
        "linear_bo_candidate": linear_flags["bo_candidate"],
        "linear_harmful_overfitting": linear_flags["harmful_overfitting"],
        "linear_underfitting": linear_flags["underfitting"],
        "linear_indeterminate": linear_flags["indeterminate"],
    }
