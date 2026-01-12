import math

import torch


def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()

def absolute_error(ys_pred, ys):
    return (ys - ys_pred).abs()

def mean_absolute_error(ys_pred, ys):
    return (ys - ys_pred).abs().mean()

def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)


class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(
    task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, **kwargs
):
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "sparse_linear_regression": SparseLinearRegression,
        "linear_classification": LinearClassification,
        "noisy_linear_regression": NoisyLinearRegression,
        "quadratic_regression": QuadraticRegression,
        "relu_2nn_regression": Relu2nnRegression,
        "decision_tree": DecisionTree,
        "uniform_hypersphere_regression": UniformHypersphereRegression,
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError

class UniformHypersphereRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super(UniformHypersphereRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            w_b = torch.randn(self.b_size, self.n_dims, 1)  
            self.w_b = w_b / w_b.norm(dim=1, keepdim=True)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                w = torch.randn(self.n_dims, 1, generator=generator)
                self.w_b[i] = w / torch.norm(w)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        # Scale by sqrt(n_dims) because weights are normalized to unit norm
        # whereas LinearRegression uses un-normalized random weights with expected norm ~sqrt(n_dims)
        ys_linear = self.scale * math.sqrt(self.n_dims) * (xs_b @ w_b)[:, :, 0] 
        # ys_b = ys_linear + torch.randn_like(ys_linear)
        return ys_linear

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        w = torch.randn(num_tasks, n_dims, 1)
        w_normalized = w / torch.norm(w, dim=1, keepdim=True)
        return {"w": w_normalized}

    @staticmethod
    def get_metric():
        return absolute_error

    @staticmethod
    def get_training_metric():
        return mean_absolute_error
    
class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, uniform=False):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            if uniform:
                self.w_b = torch.rand(self.b_size, self.n_dims, 1) * 2 - 1
            else:
                self.w_b = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class SparseLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        sparsity=3,
        valid_coords=None,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(SparseLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.sparsity = sparsity
        if valid_coords is None:
            valid_coords = n_dims
        assert valid_coords <= n_dims

        for i, w in enumerate(self.w_b):
            mask = torch.ones(n_dims).bool()
            if seeds is None:
                perm = torch.randperm(valid_coords)
            else:
                generator = torch.Generator()
                generator.manual_seed(seeds[i])
                perm = torch.randperm(valid_coords, generator=generator)
            mask[perm[:sparsity]] = False
            w[mask] = 0

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class LinearClassification(LinearRegression):
    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy

class NoisyLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        noise_std=1.0,
        renormalize_ys=False,
        noise_type="laplace",  # "normal", "uniform", "laplace", "t-student", "cauchy", "exponential", "rayleigh", "beta", "poisson"        
        w_distribution="beta",
        w_kwargs=None,
        uniform=False,
    ):
        super(NoisyLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale, uniform
        )
        self.noise_std = float(noise_std)
        self.renormalize_ys = renormalize_ys
        self.noise_type = noise_type.lower()
        self.w_distribution = w_distribution.lower()
        self.w_kwargs = w_kwargs or {}
        self.w_b = self._compose_weights(pool_dict, seeds)

    def _compose_weights(self, pool_dict, seeds):
        target_shape = (self.b_size, self.n_dims, 1)
        if pool_dict is not None:
            indices = torch.randperm(len(pool_dict["w"]))[: self.b_size]
            return pool_dict["w"][indices]
        
        if seeds is None:
            return self._sample_distribution(target_shape, generator=None)
        w_b = torch.zeros(target_shape)
        for i, seed in enumerate(seeds):
            gen = torch.Generator().manual_seed(int(seed))
            w_b[i] = self._sample_distribution((1, self.n_dims, 1), generator=gen).squeeze(0)
        return w_b
        
    def _sample_distribution(self, shape, generator=None, device='cpu'):
        def to_val(val):
            return torch.tensor(val, device=device) if not torch.is_tensor(val) else val.to(device)
        if self.w_distribution == "gaussian":
            scale = self.w_kwargs.get("scale", 1.0)
            return (scale * torch.randn(shape, generator=generator)).to(device)
        elif self.w_distribution == "uniform":
            low = self.w_kwargs.get("low", -1.0)
            high = self.w_kwargs.get("high", 1.0)
            return torch.empty(shape, generator=generator).uniform_(low, high).to(device)
        elif self.w_distribution == "laplace":
            scale = self.w_kwargs.get("scale", 1.0)
            laplace_dist = torch.distributions.Laplace(loc=0.0, scale=scale)
            return laplace_dist.sample(shape).to(device)
        elif self.w_distribution == "exponential":
            rate = self.w_kwargs.get("rate", 1.0)
            exp_dist = torch.distributions.Exponential(rate=rate)
            return exp_dist.sample(shape).to(device)
        elif self.w_distribution == "beta":
            alpha = self.w_kwargs.get("alpha", 2.0)
            beta = self.w_kwargs.get("beta", 5.0)
            beta_dist = torch.distributions.Beta(concentration1=alpha, concentration0=beta)
            return beta_dist.sample(shape).to(device)
        elif self.w_distribution == "poisson":
            rate = self.w_kwargs.get("rate", 3.0)
            dist = torch.distributions.Poisson(rate=rate)
            return dist.sample(shape).to(device)
        elif self.w_distribution == "cauchy":
            scale = self.w_kwargs.get("scale", 1.0)
            cauchy_dist = torch.distributions.StudentT(df=1, loc=0.0, scale=scale)
            return cauchy_dist.sample(shape).to(device)
        elif self.w_distribution == "t-student":
            df = self.w_kwargs.get("df", 3.0)
            scale = self.w_kwargs.get("scale", 1.0)
            t_dist = torch.distributions.StudentT(df=df, loc=0.0, scale=scale)
            return t_dist.sample(shape).to(device)
        elif self.w_distribution == "rayleigh":
            lambda_param = self.w_kwargs.get("lambda_param", 1.0)
            sigma = lambda_param
            X = torch.randn(shape, generator=generator) * sigma
            Y = torch.randn(shape, generator=generator) * sigma
            R = torch.sqrt(X**2 + Y**2)
            return R.to(device)
        elif self.w_distribution == "bernoulli":
            p = self.w_kwargs.get("p", 0.5)
            if not (0 <= p <= 1):
                raise ValueError(f"For Bernoulli distribution, p must be between 0 and 1, got {p}")
            bernoulli_dist = torch.distributions.Bernoulli(probs=p)
            X = bernoulli_dist.sample(shape).to(device)
            # Center around 0: (0 or 1) -> (-p or 1-p), standardized by sqrt(p*(1-p))
            return (X - p) / math.sqrt(p * (1 - p)) if p != 0 and p != 1 else X.to(device)
        else: 
            raise ValueError(f"Unsupported weight distribution: {self.w_distribution}")
    def sample_noise(self, shape, device='cpu'):
        # 1.
        if self.noise_type == "normal":
            noise = torch.randn(shape, device=device) * self.noise_std
        # 2.
        elif self.noise_type == "uniform":
            a = math.sqrt(3) * self.noise_std
            noise = torch.empty(shape, device=device).uniform_(-a, a)
        # 3.
        elif self.noise_type == "laplace":
            scale_param = self.noise_std / math.sqrt(2.0)
            laplace_dist = torch.distributions.Laplace(loc=0, scale=scale_param)
            noise = laplace_dist.sample(shape, device=device)
        # 4.
        elif self.noise_type == "t-student":
            df = 3.0
            scale_param = self.noise_std / math.sqrt(df / (df-2.0))
            t_dist = torch.distributions.StudentT(df=df, loc=0, scale=scale_param)
            noise = t_dist.sample(shape, device=device)
        # 5.
        elif self.noise_type == "cauchy":
            scale_param = self.noise_std 
            cauchy_dist = torch.distributions.StudentT(df=1, loc=0, scale=scale_param)
            noise = cauchy_dist.sample(shape, device=device)   
        # 6.
        elif self.noise_type == "exponential":
            exp_noise = torch.distributions.Exponential(rate=1.0 / self.noise_std)
            noise = exp_noise.sample(shape, device=device) - self.noise_std
        # 7.
        elif self.noise_type == "rayleigh":
            lambda_param = self.noise_std / math.sqrt(2.0 - math.pi / 2.0)
            # R = sqrt(X^2 + Y^2) với X, Y ~ N(0, sigma^2), 
            # where sigma = lambda_param.
            sigma = lambda_param

            X = torch.randn(shape, device=device) * sigma
            Y = torch.randn(shape, device=device) * sigma
            R = torch.sqrt(X**2 + Y**2)
            mean = lambda_param * math.sqrt(math.pi / 2.0)
            noise = R - mean
        # 8.
        elif self.noise_type == "beta":
            alpha, beta = 2.0, 5.0
            mean = alpha / (alpha + beta)
            var = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1))
            std = math.sqrt(var)
            beta_dist = torch.distributions.Beta(concentration1=alpha, concentration0=beta)
            X = beta_dist.sample(shape).to(device)
            noise = (X - mean) / std * self.noise_std
        # 9.
        elif self.noise_type == "poisson":
            lam = 3.0
            poisson_noise = torch.distributions.Poisson(lam)
            X = poisson_noise.sample(shape).to(device)
            scale_factor = self.noise_std / math.sqrt(lam)
            noise = (X - lam) * scale_factor
        #10 
        elif self.noise_type == "bernoulli":
            p = self.noise_std # probability parameter (0 to 1)
            if not (0 <= p <= 1):
                raise ValueError(f"For Bernoulli noise, noise_std must be between 0 and 1, got {p}")
            bernoulli_dist = torch.distributions.Bernoulli(probs=p)
            X = bernoulli_dist.sample(shape).to(device)
            # Center around 0: X is 0 or 1, so (X - 0.5) * 2 gives -1 or 1
            noise = (X - p) / math.sqrt(p * (1 - p))
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}")
        return noise

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        noise = self.sample_noise(ys_b.shape, device=ys_b.device)
        ys_b_noisy = ys_b + noise

        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()
        return ys_b_noisy
    
    @staticmethod
    def get_metric():
        return absolute_error

    @staticmethod
    def get_training_metric():
        return mean_absolute_error

class QuadraticRegression(LinearRegression):
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b_quad = ((xs_b**2) @ w_b)[:, :, 0]
        #         ys_b_quad = ys_b_quad * math.sqrt(self.n_dims) / ys_b_quad.std()
        # Renormalize to Linear Regression Scale
        ys_b_quad = ys_b_quad / math.sqrt(3)
        ys_b_quad = self.scale * ys_b_quad
        return ys_b_quad


class Relu2nnRegression(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        hidden_layer_size=100,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(Relu2nnRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.hidden_layer_size = hidden_layer_size

        if pool_dict is None and seeds is None:
            self.W1 = torch.randn(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.randn(self.b_size, hidden_layer_size, 1)
        elif seeds is not None:
            self.W1 = torch.zeros(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.zeros(self.b_size, hidden_layer_size, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.W1[i] = torch.randn(
                    self.n_dims, hidden_layer_size, generator=generator
                )
                self.W2[i] = torch.randn(hidden_layer_size, 1, generator=generator)
        else:
            assert "W1" in pool_dict and "W2" in pool_dict
            assert len(pool_dict["W1"]) == len(pool_dict["W2"])
            indices = torch.randperm(len(pool_dict["W1"]))[:batch_size]
            self.W1 = pool_dict["W1"][indices]
            self.W2 = pool_dict["W2"][indices]

    def evaluate(self, xs_b):
        W1 = self.W1.to(xs_b.device)
        W2 = self.W2.to(xs_b.device)
        # Renormalize to Linear Regression Scale
        ys_b_nn = (torch.nn.functional.relu(xs_b @ W1) @ W2)[:, :, 0]
        ys_b_nn = ys_b_nn * math.sqrt(2 / self.hidden_layer_size)
        ys_b_nn = self.scale * ys_b_nn
        #         ys_b_nn = ys_b_nn * math.sqrt(self.n_dims) / ys_b_nn.std()
        return ys_b_nn

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        return {
            "W1": torch.randn(num_tasks, n_dims, hidden_layer_size),
            "W2": torch.randn(num_tasks, hidden_layer_size, 1),
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class DecisionTree(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, depth=4):

        super(DecisionTree, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.depth = depth

        if pool_dict is None:

            # We represent the tree using an array (tensor). Root node is at index 0, its 2 children at index 1 and 2...
            # dt_tensor stores the coordinate used at each node of the decision tree.
            # Only indices corresponding to non-leaf nodes are relevant
            self.dt_tensor = torch.randint(
                low=0, high=n_dims, size=(batch_size, 2 ** (depth + 1) - 1)
            )

            # Target value at the leaf nodes.
            # Only indices corresponding to leaf nodes are relevant.
            self.target_tensor = torch.randn(self.dt_tensor.shape)
        elif seeds is not None:
            self.dt_tensor = torch.zeros(batch_size, 2 ** (depth + 1) - 1)
            self.target_tensor = torch.zeros_like(dt_tensor)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.dt_tensor[i] = torch.randint(
                    low=0,
                    high=n_dims - 1,
                    size=2 ** (depth + 1) - 1,
                    generator=generator,
                )
                self.target_tensor[i] = torch.randn(
                    self.dt_tensor[i].shape, generator=generator
                )
        else:
            raise NotImplementedError

    def evaluate(self, xs_b):
        dt_tensor = self.dt_tensor.to(xs_b.device)
        target_tensor = self.target_tensor.to(xs_b.device)
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i in range(xs_b.shape[0]):
            xs_bool = xs_b[i] > 0
            # If a single decision tree present, use it for all the xs in the batch.
            if self.b_size == 1:
                dt = dt_tensor[0]
                target = target_tensor[0]
            else:
                dt = dt_tensor[i]
                target = target_tensor[i]

            cur_nodes = torch.zeros(xs_b.shape[1], device=xs_b.device).long()
            for j in range(self.depth):
                cur_coords = dt[cur_nodes]
                cur_decisions = xs_bool[torch.arange(xs_bool.shape[0]), cur_coords]
                cur_nodes = 2 * cur_nodes + 1 + cur_decisions

            ys_b[i] = target[cur_nodes]

        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
