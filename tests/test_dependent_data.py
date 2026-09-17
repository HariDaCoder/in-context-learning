"""Distribution invariants for the dependent regression experiment."""

import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dependence import ar1_from_innovations, temporal_correlation, temporal_effective_rank
from samplers import GaussianSampler, get_data_sampler
from tasks import DependentLinearRegression, NoisyLinearRegression, get_task_sampler


class DependentDataTests(unittest.TestCase):
    def test_zero_dependence_is_exact_seeded_iid_control(self):
        kwargs = dict(n_points=11, b_size=3, seeds=[9, 100, 123])
        iid = GaussianSampler(5).sample_xs(**kwargs)
        markov = get_data_sampler("gaussian_ar1", 5, rho=0).sample_xs(**kwargs)
        torch.testing.assert_close(markov, iid, rtol=0, atol=0)

    def test_stationarity_and_change_point_covariance(self):
        torch.manual_seed(401)
        for rho, rho_after, change_point in [(0.8, None, None), (0.7, -0.4, 3)]:
            with self.subTest(rho=rho, rho_after=rho_after):
                sampler = get_data_sampler(
                    "gaussian_ar1", 1, rho=rho, rho_after=rho_after, change_point=change_point
                )
                draws = sampler.sample_xs(7, 20000).squeeze(-1).double()
                empirical = draws.T @ draws / len(draws)
                expected = temporal_correlation(7, rho, rho_after, change_point)
                torch.testing.assert_close(empirical, expected, rtol=0, atol=0.035)

    def test_change_point_transition_and_prefix_semantics(self):
        innovations = torch.ones(1, 4, 1, dtype=torch.float64)
        samples = ar1_from_innovations(innovations, rho=0, rho_after=0.8, change_point=2)
        torch.testing.assert_close(samples.flatten(), torch.tensor([1, 1, 1.4, 1.72], dtype=torch.float64))
        expected = torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0.8, 0.64], [0, 0.8, 1, 0.8], [0, 0.64, 0.8, 1]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(temporal_correlation(4, 0, 0.8, 2), expected)
        torch.testing.assert_close(temporal_correlation(2, 0, 0.8, 10), torch.eye(2, dtype=torch.float64))

    def test_effective_rank_is_exact_and_handles_negative_correlation(self):
        self.assertEqual(temporal_effective_rank(0), 0)
        self.assertEqual(temporal_effective_rank(9), 9)
        n, rho = 12, 0.8
        denominator = n + 2 * sum((n - lag) * rho ** (2 * lag) for lag in range(1, n))
        self.assertAlmostEqual(temporal_effective_rank(n, rho), n**2 / denominator, places=12)
        self.assertAlmostEqual(temporal_effective_rank(n, -rho), temporal_effective_rank(n, rho), places=12)

    def test_forward_and_reverse_change_protocols_have_equal_information(self):
        forward = temporal_correlation(41, rho=0.0, rho_after=0.9, change_point=21)
        reverse = temporal_correlation(41, rho=0.9, rho_after=0.0, change_point=21)
        torch.testing.assert_close(forward, reverse.flip((0, 1)))
        self.assertAlmostEqual(
            temporal_effective_rank(41, 0.0, 0.9, 21),
            temporal_effective_rank(41, 0.9, 0.0, 21),
            places=12,
        )
        forward_80 = temporal_correlation(80, rho=0.0, rho_after=0.9, change_point=40)
        reverse_80 = temporal_correlation(80, rho=0.9, rho_after=0.0, change_point=41)
        torch.testing.assert_close(forward_80, reverse_80.flip((0, 1)))

    def test_seeded_batch_permutation_scale_bias_and_truncation(self):
        scale = torch.diag(torch.tensor([1.0, 2.0, 3.0]))
        bias = torch.tensor([0.5, 1.5, -1.0])
        raw_sampler = get_data_sampler("gaussian_ar1", 3, rho=0.6)
        sampler = get_data_sampler("gaussian_ar1", 3, rho=0.6, scale=scale, bias=bias)
        raw = raw_sampler.sample_xs(8, 2, seeds=[7, 91])
        expected = raw @ scale + bias
        expected[:, :, 2:] = 0
        actual = sampler.sample_xs(8, 2, n_dims_truncated=2, seeds=[7, 91])
        torch.testing.assert_close(actual, expected)
        reversed_batch = sampler.sample_xs(8, 2, n_dims_truncated=2, seeds=[91, 7])
        torch.testing.assert_close(reversed_batch, actual.flip(0))

    def test_signal_power_is_invariant_across_curriculum_dimensions(self):
        torch.manual_seed(812)
        for active_dims in [1, 5, 20]:
            with self.subTest(active_dims=active_dims):
                task = DependentLinearRegression(20, 12000, snr=2, valid_coords=active_dims)
                xs = torch.randn(12000, 3, 20)
                signal = task.evaluate_clean(xs)
                self.assertAlmostEqual(signal.square().mean().item(), 1.0, delta=0.055)
                self.assertEqual(torch.count_nonzero(task.w_b[:, active_dims:]).item(), 0)
                torch.testing.assert_close(signal, (xs @ task.w_b).squeeze(-1))

    def test_noise_power_and_covariance_match_amplitude_snr(self):
        torch.manual_seed(221)
        task = DependentLinearRegression(
            3, 20000, snr=2, noise_rho=0.75, noise_rho_after=-0.3, noise_change_point=3
        )
        xs = torch.randn(20000, 6, 3)
        noise = (task.evaluate(xs) - task.evaluate_clean(xs)).double()
        empirical = noise.T @ noise / len(noise)
        torch.testing.assert_close(empirical, task.noise_correlation(6) / 4, rtol=0, atol=0.012)

    def test_seeded_noise_reproducible_and_separate_from_weights(self):
        seeds = [12, 84, 109]
        task = DependentLinearRegression(5, 3, seeds=seeds, noise_rho=0.5)
        clone = DependentLinearRegression(5, 3, seeds=seeds, noise_rho=0.5)
        xs = torch.ones(3, 5, 5, dtype=torch.float64)
        first = task.evaluate(xs)
        torch.testing.assert_close(first, task.evaluate(xs))
        torch.testing.assert_close(first, clone.evaluate(xs))
        iid_noise_task = DependentLinearRegression(5, 3, seeds=seeds)
        noise = iid_noise_task.evaluate(torch.zeros(3, 5, 5))
        self.assertFalse(torch.allclose(noise, iid_noise_task.w_b.squeeze(-1) * math.sqrt(5)))

    def test_noiseless_and_factory_weight_pool(self):
        sampler = get_task_sampler("dependent_linear_regression", 4, 3, num_tasks=8, snr=float("inf"))
        task = sampler(valid_coords=2)
        xs = torch.randn(3, 7, 4)
        torch.testing.assert_close(task.evaluate(xs), task.evaluate_clean(xs))
        self.assertEqual(task.w_b.shape, (3, 4, 1))
        pool = {"w": torch.ones(6, 4, 1)}
        pooled = DependentLinearRegression(4, 3, pool_dict=pool, valid_coords=2)
        torch.testing.assert_close(pool["w"], torch.ones(6, 4, 1))
        torch.testing.assert_close(pooled.w_b[:, :2], torch.full((3, 2, 1), 1 / math.sqrt(2)))

    def test_legacy_noisy_task_keeps_original_scaling(self):
        task = NoisyLinearRegression(4, 2, seeds=[13, 14], scale=3, noise_std=0)
        xs = torch.randn(2, 6, 4)
        torch.testing.assert_close(task.evaluate(xs), 3 * (xs @ task.w_b).squeeze(-1))

    def test_invalid_experiment_settings_fail_early(self):
        for rho in [1, -1, float("nan"), float("inf")]:
            with self.subTest(rho=rho), self.assertRaises(ValueError):
                get_data_sampler("gaussian_ar1", 3, rho=rho)
        for kwargs in [dict(rho_after=0.2), dict(change_point=2), dict(rho_after=0.2, change_point=-1), dict(rho_after=0.2, change_point=1.5)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                get_data_sampler("gaussian_ar1", 3, **kwargs)
        for snr in [0, -1, True, float("nan"), -float("inf")]:
            with self.subTest(snr=snr), self.assertRaises(ValueError):
                DependentLinearRegression(3, 2, snr=snr)
        for valid_coords in [0, 4, 1.5, True]:
            with self.subTest(valid_coords=valid_coords), self.assertRaises(ValueError):
                DependentLinearRegression(3, 2, valid_coords=valid_coords)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for generator comparison")
    def test_cpu_and_gpu_ar1_generators_have_same_distribution(self):
        n = 30000
        rho = 0.8
        sampler = get_data_sampler("gaussian_ar1", 1, rho=rho)
        torch.manual_seed(144)
        cpu = sampler.sample_xs(4, n, device="cpu").squeeze(-1).double()
        torch.cuda.manual_seed_all(144)
        gpu = sampler.sample_xs(4, n, device="cuda").squeeze(-1).double().cpu()
        for draws in (cpu, gpu):
            self.assertAlmostEqual(draws.square().mean().item(), 1.0, delta=0.03)
            self.assertAlmostEqual((draws[:, :-1] * draws[:, 1:]).mean().item(), rho, delta=0.03)


if __name__ == "__main__":
    unittest.main()
