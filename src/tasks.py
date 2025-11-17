import math

import torch


def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


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
        "uniform_hypersphere_regression": UniformHypersphereRegression,
        "noisy_linear_regression": NoisyLinearRegression,
        "quadratic_regression": QuadraticRegression,
        "relu_2nn_regression": Relu2nnRegression,
        "decision_tree": DecisionTree,
        "ar1_linear_regression": AR1LinearRegression,
        "exponential_weighted_regression": ExponentialWeightedRegression,
        "laplace_weighted_regression": LaplaceWeightedRegression,
    }

    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        
        # Simple return for all tasks - no special case needed
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
        ys_linear = self.scale * (xs_b @ w_b)[:, :, 0] 
        ys_b = ys_linear + torch.randn_like(ys_linear)
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        w = torch.randn(num_tasks, n_dims, 1)
        w_normalized = w / torch.norm(w, dim=1, keepdim=True)
        return {"w": w_normalized}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
class LaplaceWeightedRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, weight_scale=1.0):
        super(LaplaceWeightedRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.weight_scale = weight_scale # self.weight_scale as weight_scale

        if pool_dict is None and seeds is None:
            laplace_dist = torch.distributions.Laplace(loc=0, scale=self.weight_scale)
            self.w_b = laplace_dist.sample((self.b_size, self.n_dims, 1))
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                laplace_dist = torch.distributions.Laplace(loc=0, scale=self.weight_scale)
                self.w_b[i] = laplace_dist.sample((self.n_dims, 1))
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]
            
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_linear = self.scale * (xs_b @ w_b)[:, :, 0] 
        ys_b = ys_linear + torch.randn_like(ys_linear)
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, weight_scale=1.0):
        laplace_dist = torch.distributions.Laplace(loc=0, scale=weight_scale)
        return {"w": laplace_dist.sample((num_tasks, n_dims, 1))}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
class ExponentialWeightedRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, rate=1.0):
        super(ExponentialWeightedRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.rate = rate

        if pool_dict is None and seeds is None:
            exp_dist = torch.distributions.Exponential(rate=self.rate)
            self.w_b = exp_dist.sample((self.b_size, self.n_dims, 1))
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                exp_dist = torch.distributions.Exponential(rate=self.rate)
                self.w_b[i] = exp_dist.sample((self.n_dims, 1))
        else: 
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_linear = self.scale * (xs_b @ w_b)[:, :, 0] 
        ys_b = ys_linear + torch.randn_like(ys_linear)
        return ys_b
    
    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, rate=1.0):
        exp_dist = torch.distributions.Exponential(rate=rate)
        return {"w": exp_dist.sample((num_tasks, n_dims, 1))}
    
    @staticmethod
    def get_metric():
        return squared_error
    @staticmethod
    def get_training_metric():
        return mean_squared_error
class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1,uniform=False):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            if uniform:
                self.w_b = torch.rand(self.b_size, self.n_dims, 1)*2 -1
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
        noise_std=2.0,
        renormalize_ys=False,
        noise_type="normal",  # "normal", "uniform", "laplace", "t-student", "cauchy", "exponential", "rayleigh", "beta", "poisson"        
        uniform=False,
    ):
        super(NoisyLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale, uniform
        )
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys
        self.noise_type = noise_type.lower()

    def sample_noise(self, shape):
        # 1.
        if self.noise_type == "normal":
            noise = torch.randn(shape) * self.noise_std
        # 2.
        elif self.noise_type == "uniform":
            a = math.sqrt(3) * self.noise_std
            noise = torch.empty(shape).uniform_(-a, a)
        # 3.
        elif self.noise_type == "laplace":
            scale_param = self.noise_std / math.sqrt(2.0)
            laplace_dist = torch.distributions.Laplace(loc=0, scale=scale_param)
            noise = laplace_dist.sample(shape)
        # 4.
        elif self.noise_type == "t-student":
            df = 3.0
            scale_param = self.noise_std / math.sqrt(df / (df-2.0))
            t_dist = torch.distributions.StudentT(df=df, loc=0, scale=scale_param)
            noise = t_dist.sample(shape)
        # 5.
        elif self.noise_type == "cauchy":
            scale_param = self.noise_std * 0.5 
            cauchy_dist = torch.distributions.StudentT(df=1, loc=0, scale=scale_param)
            noise = cauchy_dist.sample(shape)   
        # 6.
        elif self.noise_type == "exponential":
            exp_noise = torch.distributions.Exponential(rate=1.0 / self.noise_std)
            noise = exp_noise.sample(shape) - self.noise_std
        # 7.
        elif self.noise_type == "rayleigh":
            lambda_param = self.noise_std / math.sqrt(2.0 - math.pi / 2.0)
            # R = sqrt(X^2 + Y^2) với X, Y ~ N(0, sigma^2), 
            # where sigma = lambda_param.
            sigma = lambda_param

            X = torch.randn(shape) * sigma
            Y = torch.randn(shape) * sigma
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
            X = beta_dist.sample(shape)
            noise = (X - mean) / std * self.noise_std
        # 9.
        elif self.noise_type == "poisson":
            lam = 3.0
            poisson_noise = torch.distributions.Poisson(lam)
            X = poisson_noise.sample(shape)
            scale_factor = self.noise_std / math.sqrt(lam)
            noise = (X - lam) * scale_factor     
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}")
        return noise

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        noise = self.sample_noise(ys_b.shape)
        ys_b_noisy = ys_b + noise

        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()
        return ys_b_noisy


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
            self.target_tensor = torch.zeros_like(self.dt_tensor)
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
class AR1LinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, ar_coef=0.5, noise_std=1.0,compute_gradient=False):
        """
        AR(1) Linear Regression: y_t = x_t^T w + epsilon_t
        where epsilon_t = ar_coef * epsilon_{t-1} + u_t, u_t ~ N(0, noise_std^2)
        
        scale: a constant by which to scale the randomly sampled weights
        ar_coef: AR(1) coefficient for error terms
        noise_std: standard deviation of innovation noise
        """
        super(AR1LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.ar_coef = ar_coef
        self.noise_std = noise_std
        self.compute_gradient = compute_gradient
        if pool_dict is None and seeds is None:
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
        """
        Generate AR(1) linear regression data with correlated errors
        """
        w_b = self.w_b.to(xs_b.device)
        batch_size, n_points, n_dims = xs_b.shape
        
        # Generate linear predictions
        ys_linear = self.scale * (xs_b @ w_b)[:, :, 0]
        
        # Generate AR(1) error terms
        ys_ar1 = torch.zeros_like(ys_linear)
        for b in range(batch_size):
            # Generate AR(1) process for errors
            errors = torch.zeros(n_points, device=xs_b.device)
            for t in range(n_points):
                if t == 0:
                    # Initial error
                    errors[t] = torch.randn(1, device=xs_b.device) * self.noise_std
                else:
                    # AR(1) error: epsilon_t = ar_coef * epsilon_{t-1} + u_t
                    errors[t] = self.ar_coef * errors[t-1] + torch.randn(1, device=xs_b.device) * self.noise_std
            
            # Add AR(1) errors to linear predictions
            ys_ar1[b] = ys_linear[b] + errors
        
        return ys_ar1

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

# class AR2RegressionTask:
    # def __init__(self, ar1_coef=0.5, ar2_coef=0.3, noise_std=1.0):
    #     """
    #     AR(2) Regression Task: y_t = ar1_coef * y_{t-1} + ar2_coef * y_{t-2} + epsilon_t
    #     where epsilon_t ~ N(0, noise_std^2)
        
    #     ar1_coef: AR(1) coefficient
    #     ar2_coef: AR(2) coefficient
    #     noise_std: standard deviation of innovation noise
    #     """
    #     self.ar1_coef = ar1_coef
    #     self.ar2_coef = ar2_coef
    #     self.noise_std = noise_std
    # def evaluate(self, xs):
    #     batch_size, seq_len, dim = xs.shape
    #     ys = torch.zeros(xs)

    #     ys[:, 0:2, :] = xs[:, 0:2, :]  # Initialize first two values

    #     for t in range(2, seq_len):
    #         ys[:, t, :] = (self.ar1_coef * ys[:, t-1, :] +
    #                        self.ar2_coef * ys[:, t-2, :] + 
    #                        self.noise_std * torch.randn_like(xs[:, t, :]))
    #     return ys
    # def get_metric(self):
    #     return lambda pred, target: ((pred - target) ** 2).mean(dim=-1)
