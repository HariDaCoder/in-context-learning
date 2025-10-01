import math

import torch


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "ar1":AR1Sampler,
        "var1":VAR1Sampler,
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

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b
    
    
class AR1Sampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=0.9, sigma=0.5, init_state=None):
        super().__init__(n_dims)
        # phi là số thực, có thể lấy trace nếu scale là ma trận
        if torch.is_tensor(scale) and scale.ndim == 2:
            self.phi = torch.trace(scale).item()
        elif isinstance(scale, (int, float)):
            self.phi = float(scale)
        else:
            raise ValueError("scale phải là số hoặc ma trận 2D torch.Tensor")

        self.sigma = sigma
        self.bias = bias
        self.init_state = init_state if init_state is not None else torch.ones(n_dims)

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        xs_b = torch.zeros(b_size, n_points, self.n_dims)

        for b in range(b_size):
            if seeds is not None:
                torch.manual_seed(seeds[b])

            state = self.init_state.clone()
            for t in range(n_points):
                # noise = torch.randn(self.n_dims) * self.sigma
                state = self.phi * state
                if self.bias is not None:
                    state += self.bias
                xs_b[b, t] = state

        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b

class VAR1Sampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None, sigma=0.5, init_state=None):
        super().__init__(n_dims)
        if scale is None:
            self.phi = 0.9 * torch.eye(n_dims)
        else:
            assert scale.shape == (n_dims, n_dims), "scale phải có shape (n_dims, n_dims)"
            self.phi = scale

        self.bias = bias
        self.sigma = sigma
        self.init_state = init_state if init_state is not None else torch.ones(n_dims)

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        xs_b = torch.zeros(b_size, n_points, self.n_dims)

        for b in range(b_size):
            if seeds is not None:
                torch.manual_seed(seeds[b])

            state = self.init_state.clone()
            for t in range(n_points):
                noise = torch.randn(self.n_dims) * self.sigma
                state = self.phi @ state + noise
                if self.bias is not None:
                    state += self.bias
                xs_b[b, t] = state

        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b
def test_ar1_sampler():
    n_dims = 3
    n_points = 5
    b_size = 2
    phi = 0.5
    sigma = 0.0
    init_state = torch.tensor([1.0, 2.0, 3.0])

    seeds = [42, 123]

    sampler = AR1Sampler(n_dims=n_dims, scale=phi, sigma=sigma, init_state=init_state)
    xs = sampler.sample_xs(n_points=n_points, b_size=b_size, seeds=seeds)

    print("Output shape:", xs.shape)  # should be (b_size, n_points, n_dims)
    print("First batch:\n", xs[0])
    print("Second batch:\n", xs[1])

    # Test reproducibility
    xs2 = sampler.sample_xs(n_points=n_points, b_size=b_size, seeds=seeds)
    assert torch.allclose(xs, xs2), "Output not reproducible with same seeds!"
    print("Reproducibility test passed.")

    # Test AR1 dynamics roughly
    print("\nCheck AR1 dynamics:")
    for t in range(1, n_points):
        expected = phi * xs[0, t-1]
        print(f"t={t}, previous*phi: {expected}, current: {xs[0, t]}")
        
        
def test_var1_sampler():
    n_dims = 2
    n_points = 4
    b_size = 2
    phi = torch.tensor([[0.5, 0.1], [0.0, 0.7]])
    sigma = 0.1
    init_state = torch.tensor([1.0, 2.0])
    seeds = [42, 123]

    sampler = VAR1Sampler(n_dims=n_dims, scale=phi, sigma=sigma, init_state=init_state)
    xs = sampler.sample_xs(n_points=n_points, b_size=b_size, seeds=seeds)
    print(xs)


if __name__ == "__main__":
    test_var1_sampler()
