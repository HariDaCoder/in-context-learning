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
        # "var1":VAR1Sampler,
        "ar2":AR2Sampler,
        "vr2":VR2Sampler,
        "nonstation":NonStationarySampler,
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

# code này là thêm:
class AR1Sampler(DataSampler):
    def __init__(self, n_dims, rho=0.9, noise_std=1.0, bias=None, scale=None,compute_gradient=False):
        super().__init__(n_dims)
        assert 0 <= abs(rho) < 1, "|rho| must be < 1 for a stable AR(1)"
        self.rho = float(rho)
        self.noise_std = float(noise_std)
        self.bias = bias
        self.scale = scale
        self.compute_gradient = compute_gradient

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        # Shape: (batch, time, dims)
        xs_b = torch.zeros(b_size, n_points, self.n_dims)

        generators = None
        if seeds is not None:
            assert len(seeds) == b_size
            generators = []
            for seed in seeds:
                g = torch.Generator()
                g.manual_seed(int(seed))
                generators.append(g)

        # Initialize x_0 ~ N(0, I)
        if generators is None:
            xs_b[:, 0, :] = torch.randn(b_size, self.n_dims)
        else:
            for i in range(b_size):
                xs_b[i, 0, :] = torch.randn(self.n_dims, generator=generators[i])

        # AR(1): x_t = rho * x_{t-1} + eps_t, eps_t ~ N(0, noise_std^2 I)
        for t in range(1, n_points):
            if generators is None:
                eps_t = self.noise_std * torch.randn(b_size, self.n_dims)
            else:
                eps_t = torch.zeros(b_size, self.n_dims)
                for i in range(b_size):
                    eps_t[i] = self.noise_std * torch.randn(self.n_dims, generator=generators[i])
            xs_b[:, t, :] = self.rho * xs_b[:, t - 1, :] + eps_t

        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias

        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0

        return xs_b
class AR2Sampler(DataSampler):
    def __init__(self, n_dims, ar1_coef=0.5, ar2_coef=0.3, noise_std=1.0, bias=None, scale=None):
        super().__init__(n_dims)
        assert abs(ar2_coef) < 1, "|ar2_coef| must be < 1 for a stable AR(2)"
        
        self.ar1_coef = float(ar1_coef)
        self.ar2_coef = float(ar2_coef)
        self.noise_std = float(noise_std)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        # Shape: (batch, time, dims)
        xs_b = torch.zeros(b_size, n_points, self.n_dims)

        generators = None
        if seeds is not None:
            assert len(seeds) == b_size
            generators = []
            for seed in seeds:
                g = torch.Generator()
                g.manual_seed(int(seed))
                generators.append(g)

        # Initialize first two time steps
        for t in range(2):
            if generators is None:
                xs_b[:, t, :] = torch.randn(b_size, self.n_dims)
            else:
                for i in range(b_size):
                    xs_b[i, t, :] = torch.randn(self.n_dims, generator=generators[i])

        # AR(2): x_t = ar1_coef * x_{t-1} + ar2_coef * x_{t-2} + eps_t
        for t in range(2, n_points):
            if generators is None:
                eps_t = self.noise_std * torch.randn(b_size, self.n_dims)
            else:
                eps_t = torch.zeros(b_size, self.n_dims)
                for i in range(b_size):
                    eps_t[i] = self.noise_std * torch.randn(self.n_dims, generator=generators[i])
            xs_b[:, t, :] = (
                self.ar1_coef * xs_b[:, t - 1, :] +
                self.ar2_coef * xs_b[:, t - 2, :] +
                eps_t
            )
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias

        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0

        return xs_b
class VR2Sampler(DataSampler):
    def __init__(self, n_dims, ar1_mat=None, ar2_mat=None, noise_std=1.0, bias=None, scale=None):
        super().__init__(n_dims)
        
        if ar1_mat is None:
            ar1_mat = 0.5 * torch.eye(n_dims)
        if ar2_mat is None:
            ar2_mat = 0.3 * torch.eye(n_dims)
            
        # Check 
        assert ar1_mat.shape == (n_dims, n_dims), "ar1_mat must be n_dims x n_dims"
        assert ar2_mat.shape == (n_dims, n_dims), "ar2_mat must be n_dims x n_dims"
        
        self.ar1_mat = torch.tensor(ar1_mat, dtype=torch.float32)
        self.ar2_mat = torch.tensor(ar2_mat, dtype=torch.float32)
        self.noise_std = float(noise_std)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        xs_b = torch.zeros(b_size, n_points, self.n_dims)

        generators = None
        if seeds is not None:
            generators = [torch.Generator().manual_seed(int(seed)) for seed in seeds]

        # Initialize first two time points
        for t in range(2):
            if generators is None:
                xs_b[:, t, :] = torch.randn(b_size, self.n_dims)
            else:
                for i in range(b_size):
                    xs_b[i, t, :] = torch.randn(self.n_dims, generator=generators[i])
        
        # VR(2): x_t = A1 * x_{t-1} + A2 * x_{t-2} + eps_t
        for t in range(2, n_points):
            if generators is None:
                eps_t = self.noise_std * torch.randn(b_size, self.n_dims)
            else:
                eps_t = torch.zeros(b_size, self.n_dims)
                for i in range(b_size):
                    eps_t[i] = self.noise_std * torch.randn(self.n_dims, generator=generators[i])
                    
            # Matrix multiplication for each sample in batch
            xs_b[:, t, :] = (torch.matmul(xs_b[:, t-1, :], self.ar1_mat.T) + 
                            torch.matmul(xs_b[:, t-2, :], self.ar2_mat.T) + 
                            eps_t)
            
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias

        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
            
        return xs_b
 
class NonStationarySampler(DataSampler):
    def __init__(self, n_dims, coef_base=0.5, coef_amplitude=0.4, noise_std = 0.1,  bias=None, scale=None):
        super().__init__(n_dims)
        self.coef_base = float(coef_base)
        self.coef_amplitude = float(coef_amplitude)
        self.noise_std = float(noise_std)
        self.scale = scale
        self.bias = bias
    def get_transition_matrix(self, t, n_points):
        t_norm = t / (n_points - 1) if n_points > 1 else 0.0
        time_varying_factor = self.coef_base + self.coef_amplitude * math.sin(2 * math.pi * t_norm)
        A_t = time_varying_factor * torch.eye(self.n_dims)
        return A_t

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        xs_b = torch.zeros(b_size, n_points, self.n_dims)
        generators = None
        if seeds is not None:
            assert len(seeds) == b_size
            generators = [torch.Generator().manual_seed(int(seed)) for seed in seeds]
        if generators is None:
            xs_b[:,0,:] = torch.randn(b_size, self.n_dims) * self.noise_std
        else:
            for i in range(b_size):
                xs_b[i, 0, :] = torch.randn(self.n_dims, generator=generators[i]) * self.noise_std
        for t in range(1, n_points):
            A_t = self.get_transition_matrix(t, n_points)

            if generators is None:
                eps_t = self.noise_std * torch.randn(b_size, self.n_dims)
            else:
                eps_t = torch.zeros(b_size, self.n_dims)
                for i in range(b_size):
                    eps_t[i] = self.noise_std * torch.randn(self.n_dims, generator=generators[i])
            xs_b[:, t, :] = (torch.matmul(xs_b[:, t-1, :], A_t) + eps_t)

        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        
        return xs_b
