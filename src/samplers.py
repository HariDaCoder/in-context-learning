import math

import torch

from dependence import (
    ar1_from_innovations,
    temporal_correlation,
    temporal_effective_rank,
    validate_ar1,
)


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "gaussian_ar1": GaussianAR1Sampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None, device=None):
        device = torch.device("cpu" if device is None else device)
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims, device=device)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims, device=device)
            generator = torch.Generator(device=device)
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(
                    n_points, self.n_dims, generator=generator, device=device
                )
        if self.scale is not None:
            xs_b = xs_b @ self.scale.to(device)
        if self.bias is not None:
            xs_b += self.bias.to(device)
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b


class GaussianAR1Sampler(GaussianSampler):
    """Gaussian Markov features with unit marginals before scale/bias.

    ``rho=0`` is the IID control. Set ``rho_after`` and ``change_point`` for
    dependence drift while preserving the feature marginal distribution.
    """

    def __init__(
        self, n_dims, bias=None, scale=None, rho=0.0, rho_after=None, change_point=None
    ):
        super().__init__(n_dims, bias=bias, scale=scale)
        validate_ar1(rho, rho_after, change_point)
        self.rho = rho
        self.rho_after = rho_after
        self.change_point = change_point

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None, device=None):
        # Reuse the legacy sampler's per-item random stream, making rho=0 an
        # exact seeded IID control. Apply scale/bias after temporal dependence.
        innovations = GaussianSampler(self.n_dims).sample_xs(
            n_points, b_size, seeds=seeds, device=device
        )
        xs_b = ar1_from_innovations(
            innovations, self.rho, self.rho_after, self.change_point
        )
        if self.scale is not None:
            xs_b = xs_b @ self.scale.to(xs_b.device)
        if self.bias is not None:
            xs_b += self.bias.to(xs_b.device)
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b

    def correlation(self, n_points):
        return temporal_correlation(n_points, self.rho, self.rho_after, self.change_point)
