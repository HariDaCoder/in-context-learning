"""Tests for the dependent-context/independent-query training protocol."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samplers import get_data_sampler
from tasks import get_task_sampler
from training_data import sample_training_batch


class TrainingDataTests(unittest.TestCase):
    def test_independent_query_breaks_feature_and_noise_dependence(self):
        torch.manual_seed(921)
        batch_size = 20000
        curriculum = SimpleNamespace(n_points=4, n_dims_truncated=1)
        data_sampler = get_data_sampler("gaussian_ar1", 1, rho=0.8)
        task_sampler = get_task_sampler(
            "dependent_linear_regression",
            1,
            batch_size,
            snr=2.0,
            noise_rho=0.7,
        )
        xs, ys, task = sample_training_batch(
            data_sampler,
            task_sampler,
            curriculum,
            batch_size,
            "independent",
            {},
            {},
        )
        features = xs.squeeze(-1).double()
        self.assertAlmostEqual((features[:, 0] * features[:, 1]).mean().item(), 0.8, delta=0.025)
        self.assertAlmostEqual((features[:, 2] * features[:, 3]).mean().item(), 0.0, delta=0.025)

        noise = (ys - task.evaluate_clean(xs)).double()
        noise /= task.noise_std
        self.assertAlmostEqual((noise[:, 0] * noise[:, 1]).mean().item(), 0.7, delta=0.025)
        self.assertAlmostEqual((noise[:, 2] * noise[:, 3]).mean().item(), 0.0, delta=0.025)

    def test_causal_mode_preserves_whole_markov_prompt(self):
        torch.manual_seed(77)
        batch_size = 12000
        curriculum = SimpleNamespace(n_points=3, n_dims_truncated=1)
        data_sampler = get_data_sampler("gaussian_ar1", 1, rho=0.6)
        task_sampler = get_task_sampler(
            "dependent_linear_regression", 1, batch_size, snr=float("inf")
        )
        xs, ys, task = sample_training_batch(
            data_sampler, task_sampler, curriculum, batch_size, "causal", {}, {}
        )
        empirical = (xs[:, 1, 0] * xs[:, 2, 0]).mean().item()
        self.assertAlmostEqual(empirical, 0.6, delta=0.03)
        torch.testing.assert_close(ys, task.evaluate_clean(xs))

    def test_unknown_query_mode_fails(self):
        curriculum = SimpleNamespace(n_points=2, n_dims_truncated=1)
        with self.assertRaisesRegex(ValueError, "query_mode"):
            sample_training_batch(
                get_data_sampler("gaussian_ar1", 1),
                get_task_sampler("dependent_linear_regression", 1, 2),
                curriculum,
                2,
                "unknown",
                {},
                {},
            )


if __name__ == "__main__":
    unittest.main()
