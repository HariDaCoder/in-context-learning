"""Unit-marginal Gaussian dependence shared by data and label-noise samplers."""

import math
import numbers

import torch


def validate_ar1(rho=0.0, rho_after=None, change_point=None):
    """Validate a stationary process or a single change in its transition law.

    ``change_point`` counts observations in the first segment. The transition
    into zero-based index t uses ``rho_after`` when t >= change_point. A change
    beyond a requested prefix is allowed, and zero uses rho_after throughout.
    """
    for name, value in (("rho", rho), ("rho_after", rho_after)):
        if value is None and name == "rho_after":
            continue
        if not isinstance(value, numbers.Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number with absolute value < 1")
        if abs(value) >= 1:
            raise ValueError(f"{name} must have absolute value < 1")
    if (rho_after is None) != (change_point is None):
        raise ValueError("rho_after and change_point must be provided together")
    if change_point is not None and (
        isinstance(change_point, bool)
        or not isinstance(change_point, numbers.Integral)
        or change_point < 0
    ):
        raise ValueError("change_point must be a nonnegative integer")


def ar1_from_innovations(innovations, rho=0.0, rho_after=None, change_point=None):
    """Transform independent normals of shape (batch, time, ...) into AR(1).

    The first observation is standard normal and innovations are scaled by
    sqrt(1-rho_t**2), keeping every marginal variance equal to one, including
    across a change point. The input is not modified.
    """
    validate_ar1(rho, rho_after, change_point)
    if innovations.ndim < 2:
        raise ValueError("innovations must have a batch and time dimension")
    samples = innovations.clone()
    for t in range(1, samples.shape[1]):
        coefficient = rho_after if change_point is not None and t >= change_point else rho
        samples[:, t] = (
            coefficient * samples[:, t - 1]
            + math.sqrt(1 - coefficient**2) * innovations[:, t]
        )
    return samples


def temporal_correlation(n_points, rho=0.0, rho_after=None, change_point=None):
    """Return the exact time-correlation matrix, including across a change.

    For i < j, R[i,j] is the product of transition coefficients from i+1
    through j. Computation uses float64 so effective-rank diagnostics do not
    depend on the data sampler's precision.
    """
    validate_ar1(rho, rho_after, change_point)
    if isinstance(n_points, bool) or not isinstance(n_points, numbers.Integral) or n_points < 0:
        raise ValueError("n_points must be a nonnegative integer")
    correlation = torch.eye(n_points, dtype=torch.float64)
    for j in range(1, n_points):
        coefficient = rho_after if change_point is not None and j >= change_point else rho
        correlation[:j, j] = coefficient * correlation[:j, j - 1]
        correlation[j, :j] = correlation[:j, j]
    return correlation


def temporal_effective_rank(n_points, rho=0.0, rho_after=None, change_point=None):
    """Return exact trace(R)**2 / trace(R**2), with rank zero for no points.

    This is a correlation-matrix diagnostic, not an asserted effective sample
    size or a proven threshold law for a trained Transformer.
    """
    correlation = temporal_correlation(n_points, rho, rho_after, change_point)
    if n_points == 0:
        return 0.0
    return (correlation.trace().square() / correlation.square().sum()).item()
