from statistics import variance
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso
import warnings
from sklearn import tree
import xgboost as xgb

from base_models import NeuralNetwork, ParallelNetworks


def build_model(conf):
    if conf.family == "gpt2":
        model = TransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )
    else:
        raise NotImplementedError

    return model


def get_relevant_baselines(task_name):
    task_to_baselines = {
        "laplace_weighted_regression": [
            (LeastSquaresModel, {}),
            (RidgeModel, {"alpha": 0.5}),
        ],
        "exponential_weighted_regression": [
            (LeastSquaresModel, {}),
            (RidgeModel, {"alpha": 0.5}),
        ],
        "uniform_hypersphere_regression": [
            (LeastSquaresModel, {}),
            (RidgeModel, {"alpha": 0.1}),
            (RidgeModel, {"alpha": 0.5}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "linear_regression": [
            (LeastSquaresModel, {}),
            (RidgeModel, {"alpha": 0.1}),
            (RidgeModel, {"alpha": 0.5}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "linear_classification": [
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "sparse_linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
            (RidgeModel, {"alpha": 0.5}),
        ]
        + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]],
        "relu_2nn_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
            (
                GDModel,
                {
                    "model_class": NeuralNetwork,
                    "model_class_args": {
                        "in_size": 20,
                        "hidden_size": 100,
                        "out_size": 1,
                    },
                    "opt_alg": "adam",
                    "batch_size": 100,
                    "lr": 5e-3,
                    "num_steps": 100,
                },
            ),
        ],
        "decision_tree": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (DecisionTreeModel, {"max_depth": 4}),
            (DecisionTreeModel, {"max_depth": None}),
            (XGBoostModel, {}),
            (AveragingModel, {}),
        ],
        "noisy_linear_regression": [
            (LeastSquaresModel, {}),
            # (RidgeModel, {"alpha": 0.1}),
            (RidgeModel, {"alpha": 0.5}),
            (RidgeModelWithVarianceAdjustment, {"alpha": 0.5, "ar_coef": 0.5}),
            (FeasibleGLSModel, {"ar_coef": None}),
            (GLSModel, {"ar_coef": 0.5}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        # "ar1_linear_regression": [
        #     (LeastSquaresModel, {}),
        #     (RidgeModel, {"alpha": 0.1}),
        #     (RidgeModel, {"alpha": 1.0}),
        #     (RidgeModelWithVarianceAdjustment, {"alpha": 1.0, "ar_coef": 0.5}),
        #     (FeasibleGLSModel, {"ar_coef": None}),
        #     (GLSModel, {"ar_coef": 0.5}),
        #     (NNModel, {"n_neighbors": 3}),
        #     (AveragingModel, {}),
        # ],
    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b): # Create sequence context by interleaving x's and y's
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction[:, ::2, 0][:, inds]  # predict only on xs


class NNModel:
    def __init__(self, n_neighbors, weights="uniform"):
        # should we be picking k optimally
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.name = f"NN_n={n_neighbors}_{weights}"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]
            dist = (train_xs - test_x).square().sum(dim=2).sqrt()

            if self.weights == "uniform":
                weights = torch.ones_like(dist)
            else:
                weights = 1.0 / dist
                inf_mask = torch.isinf(weights).float()  # deal with exact match
                inf_row = torch.any(inf_mask, axis=1)
                weights[inf_row] = inf_mask[inf_row]

            pred = []
            k = min(i, self.n_neighbors)
            ranks = dist.argsort()[:, :k]
            for y, w, n in zip(train_ys, weights, ranks):
                y, w = y[n], w[n]
                pred.append((w * y).sum() / w.sum())
            preds.append(torch.stack(pred))

        return torch.stack(preds, dim=1)


# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
class LeastSquaresModel:
    def __init__(self, driver=None):
        self.driver = driver
        self.name = f"OLS_driver={driver}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws, _, _, _ = torch.linalg.lstsq(
                train_xs, train_ys.unsqueeze(2), driver=self.driver
            )

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


class AveragingModel:
    def __init__(self):
        self.name = "averaging"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            train_zs = train_xs * train_ys.unsqueeze(dim=-1)
            w_p = train_zs.mean(dim=1).unsqueeze(dim=-1)
            pred = test_x @ w_p
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


# Lasso regression (for sparse linear regression).
# Seems to take more time as we decrease alpha.
class LassoModel:
    def __init__(self, alpha, max_iter=100000):
        # the l1 regularizer gets multiplied by alpha.
        self.alpha = alpha
        self.max_iter = max_iter
        self.name = f"lasso_alpha={alpha}_max_iter={max_iter}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # If all points till now have the same label, predict that label.

                    clf = Lasso(
                        alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter
                    )

                    # Check for convergence.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        try:
                            clf.fit(train_xs, train_ys)
                        except Warning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise

                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

                    test_x = xs[j, i : i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


# Gradient Descent and variants.
# Example usage: gd_model = GDModel(NeuralNetwork, {'in_size': 50, 'hidden_size':400, 'out_size' :1}, opt_alg = 'adam', batch_size = 100, lr = 5e-3, num_steps = 200)
class GDModel:
    def __init__(
        self,
        model_class,
        model_class_args,
        opt_alg="sgd",
        batch_size=1,
        num_steps=1000,
        lr=1e-3,
        loss_name="squared",
    ):
        # model_class: torch.nn model class
        # model_class_args: a dict containing arguments for model_class
        # opt_alg can be 'sgd' or 'adam'
        # verbose: whether to print the progress or not
        # batch_size: batch size for sgd
        self.model_class = model_class
        self.model_class_args = model_class_args
        self.opt_alg = opt_alg
        self.lr = lr
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.loss_name = loss_name

        self.name = f"gd_model_class={model_class}_model_class_args={model_class_args}_opt_alg={opt_alg}_lr={lr}_batch_size={batch_size}_num_steps={num_steps}_loss_name={loss_name}"

    def __call__(self, xs, ys, inds=None, verbose=False, print_step=100):
        # inds is a list containing indices where we want the prediction.
        # prediction made at all indices by default.
        # xs: bsize X npoints X ndim.
        # ys: bsize X npoints.
        xs, ys = xs.cuda(), ys.cuda()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            model = ParallelNetworks(
                ys.shape[0], self.model_class, **self.model_class_args
            )
            model.cuda()
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])

                train_xs, train_ys = xs[:, :i], ys[:, :i]
                test_xs, test_ys = xs[:, i : i + 1], ys[:, i : i + 1]

                if self.opt_alg == "sgd":
                    optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
                elif self.opt_alg == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                else:
                    raise NotImplementedError(f"{self.opt_alg} not implemented.")

                if self.loss_name == "squared":
                    loss_criterion = nn.MSELoss()
                else:
                    raise NotImplementedError(f"{self.loss_name} not implemented.")

                # Training loop
                for j in range(self.num_steps):

                    # Prepare batch
                    mask = torch.zeros(i).bool()
                    perm = torch.randperm(i)
                    mask[perm[: self.batch_size]] = True
                    train_xs_cur, train_ys_cur = train_xs[:, mask, :], train_ys[:, mask]

                    if verbose and j % print_step == 0:
                        model.eval()
                        with torch.no_grad():
                            outputs = model(train_xs_cur)
                            loss = loss_criterion(
                                outputs[:, :, 0], train_ys_cur
                            ).detach()
                            outputs_test = model(test_xs)
                            test_loss = loss_criterion(
                                outputs_test[:, :, 0], test_ys
                            ).detach()
                            print(
                                f"ind:{i},step:{j}, train_loss:{loss.item()}, test_loss:{test_loss.item()}"
                            )

                    optimizer.zero_grad()

                    model.train()
                    outputs = model(train_xs_cur)
                    loss = loss_criterion(outputs[:, :, 0], train_ys_cur)
                    loss.backward()
                    optimizer.step()

                model.eval()
                pred = model(test_xs).detach()

                assert pred.shape[1] == 1 and pred.shape[2] == 1
                pred = pred[:, 0, 0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class DecisionTreeModel:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.name = f"decision_tree_max_depth={max_depth}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = tree.DecisionTreeRegressor(max_depth=self.max_depth)
                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class XGBoostModel:
    def __init__(self):
        self.name = "xgboost"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = xgb.XGBRegressor()

                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0].item()

            preds.append(pred)

        return torch.stack(preds, dim=1)

class RidgeModel:
    def __init__(self, alpha=1.0):
        """
        Ridge regression model with L2 regularization.
        alpha: regularization strength (larger values = more regularization)
        """
        self.alpha = alpha
        self.name = f"ridge_alpha={alpha}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            # Ridge regression: (X'X + alpha*I)^(-1) X'y
            # Add regularization term to diagonal
            XtX = train_xs.transpose(-2, -1) @ train_xs
            Xty = train_xs.transpose(-2, -1) @ train_ys.unsqueeze(-1)
            
            # Add alpha * I to diagonal
            reg_matrix = XtX + self.alpha * torch.eye(XtX.shape[-1], device=XtX.device)
            
            try:
                ws = torch.linalg.solve(reg_matrix, Xty)
                pred = test_x @ ws
                preds.append(pred[:, 0, 0])
            except torch.linalg.LinAlgError:
                # Fallback to least squares if singular
                ws, _, _, _ = torch.linalg.lstsq(train_xs, train_ys.unsqueeze(2))
                pred = test_x @ ws
                preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


class RidgeModelWithVarianceAdjustment:
    def __init__(self, alpha=1.0, ar_coef=0.5):
        """
        Ridge regression with variance adjustment for AR(1) data.
        alpha: regularization strength
        ar_coef: AR(1) coefficient for variance adjustment
        """
        self.alpha = alpha
        self.ar_coef = ar_coef
        self.name = f"ridge_var_adj_alpha={alpha}_ar={ar_coef}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            # Create AR(1) covariance matrix for variance adjustment
            n = train_xs.shape[1]
            ar_cov = self._create_ar1_covariance(n, self.ar_coef)
            
            # Weighted Ridge regression: (X'V^(-1)X + alpha*I)^(-1) X'V^(-1)y
            try:
                ar_cov_inv = torch.linalg.inv(ar_cov)
                XtV_inv = train_xs.transpose(-2, -1) @ ar_cov_inv
                XtV_invX = XtV_inv @ train_xs
                XtV_invy = XtV_inv @ train_ys.unsqueeze(-1)
                
                # Add regularization
                reg_matrix = XtV_invX + self.alpha * torch.eye(XtV_invX.shape[-1], device=XtV_invX.device)
                ws = torch.linalg.solve(reg_matrix, XtV_invy)
                pred = test_x @ ws
                preds.append(pred[:, 0, 0])
            except torch.linalg.LinAlgError:
                # Fallback to regular ridge
                XtX = train_xs.transpose(-2, -1) @ train_xs
                Xty = train_xs.transpose(-2, -1) @ train_ys.unsqueeze(-1)
                reg_matrix = XtX + self.alpha * torch.eye(XtX.shape[-1], device=XtX.device)
                ws = torch.linalg.solve(reg_matrix, Xty)
                pred = test_x @ ws
                preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)

    def _create_ar1_covariance(self, n, ar_coef):
        """Create AR(1) covariance matrix: V[i,j] = ar_coef^|i-j|"""
        indices = torch.arange(n, dtype=torch.float32)
        diff = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
        return torch.pow(ar_coef, diff)


class FeasibleGLSModel:
    def __init__(self, ar_coef=None):
        """
        Feasible GLS for AR(1) data with unknown AR coefficient.
        ar_coef: if None, estimate from residuals; otherwise use fixed value
        """
        self.ar_coef = ar_coef
        self.name = f"feasible_gls_ar={'est' if ar_coef is None else ar_coef}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            pred = torch.zeros_like(ys[:, 0])
            for j in range(ys.shape[0]):
                x_j, y_j = train_xs[j], train_ys[j]
                
                # Step 1: OLS to get initial residuals
                try:
                    w_ols, _, _, _ = torch.linalg.lstsq(x_j, y_j.unsqueeze(-1))
                    residuals = y_j - (x_j @ w_ols).squeeze()
                except torch.linalg.LinAlgError:
                    pred[j] = 0.0
                    continue
                
                # Step 2: Estimate AR coefficient from residuals
                if self.ar_coef is None and len(residuals) > 1:
                    # Estimate AR(1) coefficient using Yule-Walker equations
                    ar_coef_est = self._estimate_ar_coef(residuals)
                else:
                    ar_coef_est = self.ar_coef if self.ar_coef is not None else 0.0
                
                # Step 3: Create covariance matrix and perform GLS
                if len(residuals) > 1:
                    n = len(residuals)
                    ar_cov = self._create_ar1_covariance(n, ar_coef_est)
                    
                    try:
                        ar_cov_inv = torch.linalg.inv(ar_cov)
                        XtV_inv = x_j.transpose(-1, -2) @ ar_cov_inv
                        XtV_invX = XtV_inv @ x_j
                        XtV_invy = XtV_inv @ y_j.unsqueeze(-1)
                        
                        w_gls = torch.linalg.solve(XtV_invX, XtV_invy)
                        y_pred = (test_x[j] @ w_gls).squeeze()
                        pred[j] = y_pred
                    except torch.linalg.LinAlgError:
                        # Fallback to OLS
                        y_pred = (test_x[j] @ w_ols).squeeze()
                        pred[j] = y_pred
                else:
                    # Not enough data for GLS, use OLS
                    y_pred = (test_x[j] @ w_ols).squeeze()
                    pred[j] = y_pred

            preds.append(pred)

        return torch.stack(preds, dim=1)

    def _estimate_ar_coef(self, residuals):
        """Estimate AR(1) coefficient using Yule-Walker equations (returns a torch.Tensor scalar)."""
        # Ensure residuals is a torch tensor
        if not isinstance(residuals, torch.Tensor):
            residuals = torch.tensor(residuals, dtype=torch.float32)

        if residuals.numel() <= 1:
            # return tensor scalar on same device
            return torch.tensor(0.0, dtype=torch.float32, device=residuals.device)

        # Use unbiased-ish estimators:
        n = residuals.shape[0]
        # gamma_0: variance (use unbiased? here regular torch.var with unbiased=False to match mean-of-squares)
        gamma_0 = torch.var(residuals, unbiased=False)
        gamma_1 = torch.mean(residuals[:-1] * residuals[1:])

        # avoid division by (near) zero
        if gamma_0.item() <= 1e-10:
            ar_coef = torch.tensor(0.0, dtype=torch.float32, device=residuals.device)
        else:
            ar_coef = gamma_1 / gamma_0
            # ensure tensor type & correct device
            if not isinstance(ar_coef, torch.Tensor):
                ar_coef = torch.tensor(ar_coef, dtype=torch.float32, device=residuals.device)
            else:
                ar_coef = ar_coef.to(dtype=torch.float32, device=residuals.device)

            # clamp safely as tensor
            ar_coef = torch.clamp(ar_coef, -0.99, 0.99)

        return ar_coef  # tensor scalar

    def _create_ar1_covariance(self, n, ar_coef, device=None, dtype=torch.float32):
        """Create AR(1) covariance matrix V[i,j] = ar_coef**|i-j|.
        ar_coef may be float or torch scalar; this returns a torch.Tensor (n x n).
        """
        if device is None:
            # default CPU
            device = torch.device("cpu")

        # make ar_coef a tensor scalar on correct device
        if not isinstance(ar_coef, torch.Tensor):
            ar_coef_t = torch.tensor(ar_coef, dtype=dtype, device=device)
        else:
            ar_coef_t = ar_coef.to(device=device, dtype=dtype)

        indices = torch.arange(n, dtype=dtype, device=device)
        diff = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1)).to(dtype=dtype)

        # use torch.pow with tensor base and tensor exponent
        # (ensure ar_coef_t is broadcastable)
        return torch.pow(ar_coef_t, diff)


class GLSModel:
    def __init__(self, ar_coef=0.5):
        """
        GLS with known AR(1) covariance structure.
        ar_coef: known AR(1) coefficient
        """
        self.ar_coef = ar_coef
        self.name = f"gls_ar={ar_coef}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            # Create AR(1) covariance matrix
            n = train_xs.shape[1]
            ar_cov = self._create_ar1_covariance(n, self.ar_coef)
            
            try:
                ar_cov_inv = torch.linalg.inv(ar_cov)
                XtV_inv = train_xs.transpose(-2, -1) @ ar_cov_inv
                XtV_invX = XtV_inv @ train_xs
                XtV_invy = XtV_inv @ train_ys.unsqueeze(-1)
                
                w_gls = torch.linalg.solve(XtV_invX, XtV_invy)
                pred = test_x @ w_gls
                preds.append(pred[:, 0, 0])
            except torch.linalg.LinAlgError:
                # Fallback to OLS
                ws, _, _, _ = torch.linalg.lstsq(train_xs, train_ys.unsqueeze(2))
                pred = test_x @ ws
                preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)

    def _create_ar1_covariance(self, n, ar_coef):
        """Create AR(1) covariance matrix"""
        indices = torch.arange(n, dtype=torch.float32)
        diff = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
        return torch.pow(ar_coef, diff)
class WeightedLeastSquaresModel:
    def __init__(self, variance_model='ols_residual'):
        """WLS: Heteroscedasticity (V is diagnol matrix)"""
        self.variance_model = variance_model
        self.name = f"wls_var_model={variance_model}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))
                continue

            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            weights = self._estimate_weights(train_xs, train_ys)
            sqrt_w = torch.sqrt(torch.clamp(weights, min=1e-8))

            weighted_xs = train_xs * sqrt_w.unsqueeze(-1)
            weighted_ys = train_ys * sqrt_w

            try:
                ws, _, _, _ = torch.linalg.lstsq(weighted_xs, weighted_ys.unsqueeze(-1))
            except torch.linalg.LinAlgError:
                # fall back to standard OLS if the weighted system is ill-conditioned
                ws, _, _, _ = torch.linalg.lstsq(train_xs, train_ys.unsqueeze(-1))

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)

    def _estimate_weights(self, train_xs, train_ys):
        """Return diagonal weights (inverse variances) for WLS."""
        if self.variance_model == "uniform":
            return torch.ones_like(train_ys)

        if self.variance_model == "ols_residual":
            try:
                ws, _, _, _ = torch.linalg.lstsq(train_xs, train_ys.unsqueeze(-1))
                preds = (train_xs @ ws).squeeze(-1)
                residuals = train_ys - preds
                variances = residuals.pow(2)
                variances = torch.clamp(variances, min=1e-6)
                weights = 1.0 / variances
                return weights
            except torch.linalg.LinAlgError:
                return torch.ones_like(train_ys)

        raise ValueError(f"Unknown variance_model '{self.variance_model}' for WLS")
