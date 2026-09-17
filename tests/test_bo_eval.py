"""Scientific invariants for the standalone fixed-context BO evaluator."""

import pathlib
import sys
import unittest

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from bo_eval import LinearEstimator, evaluate_context


class RecordingLeastSquares(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, xs, ys, inds=None):
        self.calls.append((xs.clone(), ys.clone(), inds, self.training))
        weights = torch.linalg.pinv(xs[:, :-1]) @ ys[:, :-1, None]
        return (xs[:, -1:] @ weights).squeeze(-1)


class RetrieveOtherwiseZero(torch.nn.Module):
    def forward(self, xs, ys, inds=None):
        matches = (xs[:, :-1] == xs[:, -1:]).all(dim=-1)
        return (matches * ys[:, :-1]).sum(dim=1, keepdim=True)


class SquareQuery(torch.nn.Module):
    def forward(self, xs, ys, inds=None):
        return xs[:, -1:, 0].square()


class BOEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.dtype = torch.float64

    def test_rank_deficient_ols_is_minimum_norm(self):
        xs = torch.tensor([[[1., 2., 0.], [2., 4., 0.]]], dtype=self.dtype)
        ys = torch.tensor([[3., 6.]], dtype=self.dtype)
        estimator = LinearEstimator("ols").fit(xs, ys)
        expected = torch.tensor([[0.6, 1.2, 0.]], dtype=self.dtype)
        torch.testing.assert_close(estimator.weights_, expected)
        torch.testing.assert_close(estimator.predict(xs), ys)

    def test_ridge_matches_penalized_normal_equations(self):
        xs = torch.randn(3, 4, 7, dtype=self.dtype)
        ys = torch.randn(3, 4, dtype=self.dtype)
        alpha = 0.7
        expected = torch.linalg.solve(
            xs.transpose(-1, -2) @ xs + alpha * torch.eye(7, dtype=self.dtype),
            xs.transpose(-1, -2) @ ys.unsqueeze(-1),
        ).squeeze(-1)
        actual = LinearEstimator("ridge", ridge_alpha=alpha).fit(xs, ys).weights_
        torch.testing.assert_close(actual, expected)
        zero = LinearEstimator("ridge", ridge_alpha=0).fit(xs, ys)
        torch.testing.assert_close(zero.weights_, LinearEstimator("ols").fit(xs, ys).weights_)

    def test_gls_with_iid_noise_equals_ols(self):
        for k, d in ((9, 3), (3, 9)):
            xs = torch.randn(2, k, d, dtype=self.dtype)
            ys = torch.randn(2, k, dtype=self.dtype)
            expected = LinearEstimator("ols").fit(xs, ys).weights_
            covariance = 2.7 * torch.eye(k, dtype=self.dtype)
            actual = LinearEstimator("gls").fit(xs, ys, covariance).weights_
            torch.testing.assert_close(actual, expected)

    def test_gls_matches_weighted_normal_equations_when_full_rank(self):
        xs = torch.randn(2, 7, 3, dtype=self.dtype)
        ys = torch.randn(2, 7, dtype=self.dtype)
        positions = torch.arange(7)
        covariance = 0.6 ** (positions[:, None] - positions[None, :]).abs().to(self.dtype)
        weighted_x = torch.linalg.solve(covariance, xs)
        weighted_y = torch.linalg.solve(covariance, ys.unsqueeze(-1))
        expected = torch.linalg.solve(
            xs.transpose(-1, -2) @ weighted_x,
            xs.transpose(-1, -2) @ weighted_y,
        ).squeeze(-1)
        actual = LinearEstimator("gls").fit(xs, ys, covariance).weights_
        torch.testing.assert_close(actual, expected)

    def test_gls_supports_rank_deficient_underdetermined_design(self):
        xs = torch.tensor([[[1., 2., 0.], [2., 4., 0.]]], dtype=self.dtype)
        ys = torch.tensor([[3., 7.]], dtype=self.dtype)
        covariance = torch.tensor([[1., 0.4], [0.4, 1.]], dtype=self.dtype)
        factor = torch.linalg.cholesky(covariance)
        whitened_x = torch.linalg.solve_triangular(factor, xs, upper=False)
        whitened_y = torch.linalg.solve_triangular(factor, ys.unsqueeze(-1), upper=False)
        expected = (torch.linalg.pinv(whitened_x) @ whitened_y).squeeze(-1)
        actual = LinearEstimator("gls").fit(xs, ys, covariance).weights_
        torch.testing.assert_close(actual, expected)
        with self.assertRaisesRegex(ValueError, "positive definite"):
            LinearEstimator("gls").fit(xs, ys, torch.ones(2, 2, dtype=self.dtype))

    def test_oracle_gls_improves_average_parameter_risk_for_correlated_noise(self):
        torch.manual_seed(8127)
        batch, k, d, rho = 600, 24, 3, 0.9
        xs = torch.randn(batch, k, d, dtype=self.dtype)
        weights = torch.randn(batch, d, 1, dtype=self.dtype)
        positions = torch.arange(k)
        covariance = rho ** (positions[:, None] - positions[None, :]).abs().to(self.dtype)
        factor = torch.linalg.cholesky(covariance)
        noise = (factor @ torch.randn(batch, k, 1, dtype=self.dtype)).squeeze(-1)
        ys = (xs @ weights).squeeze(-1) + noise
        ols = LinearEstimator("ols").fit(xs, ys).weights_
        gls = LinearEstimator("gls").fit(xs, ys, covariance).weights_
        ols_risk = (ols - weights.squeeze(-1)).square().sum(dim=1).mean()
        gls_risk = (gls - weights.squeeze(-1)).square().sum(dim=1).mean()
        self.assertLess(gls_risk.item(), 0.7 * ols_risk.item())

    def test_classical_reports_true_parameter_risk(self):
        xs = torch.tensor([[[1., 0.]]], dtype=self.dtype)
        ys = torch.tensor([[2.]], dtype=self.dtype)
        true_weights = torch.tensor([[[1.], [3.]]], dtype=self.dtype)
        queries = torch.eye(2, dtype=self.dtype).unsqueeze(0)
        metrics = evaluate_context(LinearEstimator(), xs, ys, true_weights, queries)
        self.assertAlmostEqual(metrics["exact_isotropic_clean_risk"].item(), 10.)
        self.assertAlmostEqual(metrics["clean_query_mse"].item(), 5.)
        self.assertTrue(metrics["harmful_overfitting"].item())
        self.assertNotIn("probe_r2", metrics)

    def test_context_is_fixed_dummy_label_zero_and_mode_restored(self):
        xs = torch.randn(2, 5, 2, dtype=self.dtype)
        weights = torch.randn(2, 2, 1, dtype=self.dtype)
        ys = (xs @ weights).squeeze(-1)
        queries = torch.randn(2, 7, 2, dtype=self.dtype)
        model = RecordingLeastSquares()
        model.train()
        metrics = evaluate_context(model, xs, ys, weights, queries, query_batch_size=3)
        self.assertTrue(model.training)
        self.assertTrue(metrics["bo_candidate"].all())
        self.assertTrue((metrics["probe_r2"] > 1 - 1e-10).all())
        self.assertNotIn("exact_isotropic_clean_risk", metrics)
        for supplied_x, supplied_y, inds, training in model.calls:
            self.assertEqual(supplied_x.shape[1], xs.shape[1] + 1)
            self.assertLessEqual(supplied_x.shape[0], 3)
            self.assertEqual(inds, [xs.shape[1]])
            self.assertFalse(training)
            torch.testing.assert_close(supplied_y[:, -1], torch.zeros_like(supplied_y[:, -1]))
            for task_x, task_y in zip(supplied_x, supplied_y):
                matching_tasks = (xs == task_x[:-1]).all(dim=2).all(dim=1).nonzero().flatten()
                self.assertEqual(len(matching_tasks), 1)
                torch.testing.assert_close(task_y[:-1], ys[matching_tasks[0]])

    def test_model_mode_is_restored_after_forward_error(self):
        class Fails(torch.nn.Module):
            def forward(self, *args, **kwargs):
                raise RuntimeError("intentional test failure")

        xs = torch.randn(1, 3, 2, dtype=self.dtype)
        model = Fails()
        for initial_mode in (True, False):
            model.train(initial_mode)
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                evaluate_context(model, xs, torch.zeros(1, 3, dtype=self.dtype),
                                 torch.zeros(1, 2, dtype=self.dtype), xs)
            self.assertEqual(model.training, initial_mode)

    def test_query_labels_are_never_obtained_from_ground_truth(self):
        class CopiesQueryLabel(torch.nn.Module):
            def forward(self, xs, ys, inds=None):
                return ys[:, -1:]

        xs = torch.tensor([[[1.], [2.]]], dtype=self.dtype)
        ys = torch.tensor([[100., 200.]], dtype=self.dtype)
        true_weights = torch.tensor([[3.]], dtype=self.dtype)
        query = torch.tensor([[[4.], [5.]]], dtype=self.dtype)
        result = evaluate_context(CopiesQueryLabel(), xs, ys, true_weights, query)
        self.assertAlmostEqual(result["clean_query_mse"].item(), (12 ** 2 + 15 ** 2) / 2)

    def test_coarse_mode_skips_linear_probes_but_keeps_direct_flags(self):
        xs = torch.tensor([[[1.], [2.]]], dtype=self.dtype)
        ys = torch.tensor([[1., 1.]], dtype=self.dtype)
        queries = torch.tensor([[[3.], [4.]]], dtype=self.dtype)
        result = evaluate_context(
            RetrieveOtherwiseZero(),
            xs,
            ys,
            torch.zeros(1, 1, dtype=self.dtype),
            queries,
            run_linear_probe=False,
        )
        self.assertTrue(result["bo_candidate"].item())
        self.assertNotIn("probe_r2", result)
        self.assertNotIn("linear_bo_candidate", result)

    def test_retrieval_and_generalization_are_separate_from_linear_interpolation(self):
        xs = torch.tensor([[[1.], [2.]]], dtype=self.dtype)
        ys = torch.tensor([[1., 1.]], dtype=self.dtype)
        queries = torch.tensor([[[3.], [4.]]], dtype=self.dtype)
        probes = torch.tensor([[[-2.], [-1.]]], dtype=self.dtype)
        holdouts = torch.tensor([[[-3.], [-4.]]], dtype=self.dtype)
        result = evaluate_context(RetrieveOtherwiseZero(), xs, ys, torch.zeros(1, 1, dtype=self.dtype),
                                  queries, probes, holdouts)
        self.assertEqual(result["duplicate_context_fit_mse"].item(), 0.)
        self.assertEqual(result["clean_query_mse"].item(), 0.)
        self.assertEqual(result["probe_fit_mse"].item(), 1.)
        self.assertTrue(result["bo_candidate"].item())
        self.assertFalse(result["linear_bo_candidate"].item())
        self.assertTrue(result["linear_underfitting"].item())

    def test_nonlinear_predictor_is_indeterminate_on_heldout_probes(self):
        xs = torch.tensor([[[1.], [2.]]], dtype=self.dtype)
        ys = xs[:, :, 0].square()
        queries = torch.tensor([[[0.], [0.1]]], dtype=self.dtype)
        probes = torch.tensor([[[-2.], [-1.], [1.], [2.]]], dtype=self.dtype)
        holdouts = torch.tensor([[[-3.], [-0.5], [0.5], [3.]]], dtype=self.dtype)
        result = evaluate_context(SquareQuery(), xs, ys, torch.zeros(1, 1, dtype=self.dtype),
                                  queries, probes, holdouts)
        self.assertEqual(result["duplicate_context_fit_mse"].item(), 0.)
        self.assertLess(result["probe_r2"].item(), 0.99)
        self.assertTrue(result["linear_indeterminate"].item())
        self.assertTrue(result["bo_candidate"].item())

    def test_rank_deficient_probes_cannot_identify_weight_vector(self):
        xs = torch.eye(2, dtype=self.dtype).unsqueeze(0)
        ys = torch.ones(1, 2, dtype=self.dtype)
        weights = torch.ones(1, 2, dtype=self.dtype)
        probes = torch.tensor([[[1., 0.], [2., 0.]]], dtype=self.dtype)
        holdouts = torch.tensor([[[3., 0.], [4., 0.]]], dtype=self.dtype)
        result = evaluate_context(RecordingLeastSquares(), xs, ys, weights, xs, probes, holdouts)
        self.assertFalse(result["probe_identifiable"].item())
        self.assertTrue(result["linear_indeterminate"].item())

    def test_small_nonlinear_outputs_are_not_treated_as_constant(self):
        class SmallSquare(torch.nn.Module):
            def forward(self, xs, ys, inds=None):
                return 1e-8 * xs[:, -1:, 0].square()

        xs = torch.tensor([[[1.], [2.]]], dtype=torch.float32)
        probes = torch.tensor([[[-2.], [-1.], [1.], [2.]]], dtype=torch.float32)
        holdouts = torch.tensor([[[-3.], [-0.5], [0.5], [3.]]], dtype=torch.float32)
        result = evaluate_context(SmallSquare(), xs, torch.zeros(1, 2), torch.zeros(1, 1),
                                  xs, probes, holdouts)
        self.assertTrue(result["linear_indeterminate"].item())

    def test_holdout_cannot_be_fitting_set(self):
        xs = torch.eye(2, dtype=self.dtype).unsqueeze(0)
        with self.assertRaisesRegex(ValueError, "independent"):
            evaluate_context(RecordingLeastSquares(), xs, torch.ones(1, 2, dtype=self.dtype),
                             torch.ones(1, 2, dtype=self.dtype), xs, xs, xs.clone())


if __name__ == "__main__":
    unittest.main()
